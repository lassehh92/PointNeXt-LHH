import argparse
import os
import yaml
import glob
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter

from openpoints.models import build_model_from_cfg
from openpoints.utils import set_random_seed, ConfusionMatrix, load_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, EasyConfig, dist_utils, resume_exp_directory, get_mious
from openpoints.dataset import get_features_by_keys
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg

from torch import distributed as dist

def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def load_data(data_path, cfg):
    label, feat = None, None
    data = np.load(data_path)  # xyzrgb
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    feat = np.clip(feat / 255., 0, 1).astype(np.float32)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = args.voxel_size # set voxel size via argparser

    if voxel_size is not None:
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]  # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)  # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


@torch.no_grad()
def inference(model, data_list, cfg):
    import time
    model.eval()  # set model to eval mode
    ignored_labels = torch.Tensor(cfg.ignore_index).cuda() if cfg.ignore_index is not None else None
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=ignored_labels)
    set_random_seed(0)
    
    cfg.pointview = cfg.get('pointview', False)
    if cfg.pointview:
        from openpoints.dataset.vis3d import write_ply
        cfg.pw_dir = os.path.join(cfg.run_dir, 'pointview')
        os.makedirs(cfg.pw_dir, exist_ok=True)

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    os.makedirs(cfg.save_path, exist_ok=True)
    
    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'
    points_per_sec_total = []
    for cloud_idx, data_path in enumerate(data_list):
        start_time = time.time()
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)))

        for idx_subcloud in pbar:

            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]]")
            if not (nearest_neighbor and idx_subcloud > 0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part = feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    data['heights'] = torch.from_numpy(
                        coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                logits = model(data)

            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        if not nearest_neighbor:
            # average merge overlapped multi voxels logits to original point set
            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
        else:
            # interpolate logits by nearest neighbor
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
        pred = all_logits.argmax(dim=1)
        end_time = time.time()
        points_per_sec = len(all_logits) / (end_time - start_time)
        points_per_sec_total.append(points_per_sec)
        logging.info(f'Inference time: {end_time - start_time:.2f}s ({points_per_sec:.2f} points/s)')
        if label is not None:
            cm.update(pred, label)

        file_name = f'{os.path.basename(data_path.split(".")[0])}'

        pred = pred.cpu().numpy().squeeze()
        feat = feat*255

        # output ply file
        write_ply(coord, feat, pred, os.path.join(cfg.pw_dir, f'SemSeg-{file_name}.ply'))

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'[{cloud_idx}]/[{len_data}] ({file_name}) cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                    f'\niou per cls is: {ious}')

            all_cm.value += cm.value

            tp, union, count = all_cm.tp, all_cm.union, all_cm.count
            if cfg.distributed:
                dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'TOTAL metrics: test_oa: {oa:.2f}, test_macc: {macc:.2f}, test_miou: {miou:.2f}'
                    f'\niou per cls is: {ious}')
                logging.info('TOTAL counts: {}'.format(count.detach().cpu().numpy()))
                logging.info('TOTAL tps: {}'.format(tp.detach().cpu().numpy()))

    logging.info(f'Average inference speed: {np.mean(points_per_sec_total):.2f} points/s')

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, cm
    else:
        return None, None, None, None, None, None


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Semantic segmentation inference script')
    parser.add_argument('--cfg', type=str, required=False, help='config file',
                        default="cfgs/novafos3d/pointnext-xl.yaml")
    parser.add_argument("--source", type=str, help="Sample to run inference on or a dir of samples",
                        default="/home/simon/data/novafos3D/Area_5_cloud-49.npy")
    parser.add_argument('--radius', type=float, default=0.1, help='Radius of initial set abstraction ball query')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size to use')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size used for voxel downsampling')
    parser.add_argument('--mode', type=str, help='Wandb project name', default="test")
    parser.add_argument('--pointview', type=bool, help='whether to output the results as a point cloud in ply-format or not', default=True)
    parser.add_argument('--pretrained_path', type=str,
                        default="/home/lasse/Git/PointNeXt/log/novafos3d/novafos3d-train-pointnext-xl-ngpus1-seed2696-20230210-150344-2PXLfpA5HQ8UYCXUJSr5gR/checkpoint/novafos3d-train-pointnext-xl-ngpus1-seed2696-20230210-150344-2PXLfpA5HQ8UYCXUJSr5gR_ckpt_best.pth",
                        help='path to a pretrained model'
                        )

    args, opts = parser.parse_known_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml
    cfg.mode = "test"
    cfg.pointview = args.pointview

    if args.voxel_size is not None:
        cfg.dataset.common.voxel_size = args.voxel_size

    if args.radius is not None:
        cfg.model.encoder_args.radius = args.radius

    assert args.pretrained_path is not None, "Make sure to specify path to pretrained model"
    cfg.pretrained_path = args.pretrained_path

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)

    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)

    # set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)

    logging.info(f'Testing model {os.path.basename(cfg.pretrained_path)}...')

    if os.path.isdir(args.source):
        data_list = list_full_paths(args.source)
    else:
        data_list = [args.source]

    assert [os.path.exists(data) for data in data_list], f"Data path in {data_list} does not exist!"

    # Run Inference
    inference(model, data_list, cfg)

    # wandb config
    cfg.wandb.name = cfg.run_name
