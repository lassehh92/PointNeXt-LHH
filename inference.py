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
from openpoints.dataset.s3dis.s3dis import S3DIS
from openpoints.dataset.novafos3d.novafos3d import Novafos3D
from openpoints.dataset.semantic_kitti.semantickitti import SemanticKITTI
from openpoints.transforms import build_transforms_from_cfg

from torch import distributed as dist


def generate_data_list(cfg):
    if 's3dis' in cfg.dataset.common.NAME.lower():
        raw_root = os.path.join(cfg.dataset.common.data_root, 'raw')
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    # elif 'inropa' in cfg.dataset.common.NAME.lower():
    #     import pandas as pd
    #     raw_root = os.path.join(cfg.dataset.common.data_root)
    #     label_file = os.path.join(raw_root, 'labels', 'test.csv')
    #     df = pd.read_csv(open(label_file, 'r'))
    #     data_list = df['file_name'].values.tolist()
    elif 'novafos3d' in cfg.dataset.common.NAME.lower():
        raw_root = os.path.join(cfg.dataset.common.data_root)
        data_list = sorted(os.listdir(raw_root))
        data_list = [os.path.join(raw_root, item) for item in data_list if
                     'Area_{}'.format(cfg.dataset.common.test_area) in item]
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data_list = glob.glob(os.path.join(cfg.dataset.common.data_root, cfg.dataset.test.split, "*.pth"))
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    return data_list


def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'inropa' in cfg.dataset.common.NAME.lower():
        # If data_root is a file path, load the data from the file
        point_path = os.path.join(cfg.dataset.common.data_root, 'point_clouds', data_path)
        label_path = os.path.join(cfg.dataset.common.data_root, 'labels', data_path)
        coord = np.load(point_path).astype(np.float32)
        label = np.load(label_path).astype(np.long)
    elif 'novafos3d' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
            label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path).astype(np.float32)
        if cfg.dataset.test.split == 'test':
            coord, label = data[:, :3], None
        else:
            coord, label = data[:, :3], data[:, 3]

            # Remove unlabeled points
            mask = np.isin(label, [0], invert=True)
            coord = coord[mask]
            label = label[mask]

            # subtract 1 from label to make it start from 0
            label -= 1

    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
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
def inferece(model, data_list, cfg):
    import time
    model.eval()  # set model to eval mode
    ignored_labels = torch.Tensor(cfg.ignore_index).cuda() if cfg.ignore_index is not None else None
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=ignored_labels)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    if not 'inropa' in cfg.dataset.common.NAME.lower():
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
                    if 'semantickitti' in cfg.dataset.common.NAME.lower():
                        data['heights'] = torch.from_numpy((coord_part[:, gravity_dim:gravity_dim + 1] - coord_part[:,
                                                                                                         gravity_dim:gravity_dim + 1].min()).astype(
                            np.float32)).unsqueeze(0)
                    else:
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
                """visualization in debug mode. !!! visulization is not correct, should remove ignored idx."""
                # from openpoints.dataset.vis3d import vis_points, vis_multi_points
                # vis_multi_points([coord, coord_part],
                #                  labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])

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
        """visualization in debug mode"""
        # from openpoints.dataset.vis3d import vis_points, vis_multi_points
        # vis_multi_points([coord, coord], labels=[label.cpu().numpy(), all_logits.argmax(dim=1).squeeze().cpu().numpy()])

        if 's3dis' in dataset_name:
            file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{os.path.basename(data_path.split(".")[0])}'
        elif dataset_name == 'semantickitti':
            file_name = f'{dataset_name}-{"-".join(data_path.split(".")[0].split("/")[-2:])}'
        else:
            file_name = f'{dataset_name}-{os.path.basename(data_path.split(".")[0])}'

        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels

            if feat is not None:
                write_obj(coord, feat, os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt, os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            # output pred labels
            write_obj(coord, pred, os.path.join(cfg.vis_dir, f'pred-{file_name}.obj'))

        if cfg.get('save_pred', False):

            if 'scannet' in cfg.dataset.common.NAME.lower():
                pred = pred.cpu().numpy().squeeze()
                label_int_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12,
                                     12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
                pred = np.vectorize(label_int_mapping.get)(pred)
                save_file_name = data_path.split('/')[-1].split('_')
                save_file_name = save_file_name[0] + '_' + save_file_name[1] + '.txt'
                save_file_name = os.path.join(cfg.save_path, save_file_name)
                np.savetxt(save_file_name, pred, fmt="%d")

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

    if 'scannet' in cfg.dataset.common.NAME.lower():
        logging.info(
            f" Please select and zip all the files (DON'T INCLUDE THE FOLDER) in {cfg.save_path} and submit it to"
            f" Scannet Benchmark https://kaldir.vc.in.tum.de/scannet_benchmark/. ")

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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use')
    parser.add_argument('--voxel_size', type=float, default=0.032, help='Voxel size used for voxel downsampling')
    parser.add_argument('--mode', type=str, help='Wandb project name', default="test")
    parser.add_argument('--visualize', type=bool, help='whether to visualize the results of not', default=True)
    parser.add_argument('--pretrained_path', type=str,
                        default="/home/simon/aau/PHD-RESEARCH-3D-point-segmentation/repos/PointNeXtSimon/log/novafos3d/novafos3d-train-pointnext-xl-ngpus1-seed42---epochs-200-20221215-005419-YAvLWDKpvvgYhPvqQD2je8/checkpoint/novafos3d-train-pointnext-xl-ngpus1-seed42---epochs-200-20221215-005419-YAvLWDKpvvgYhPvqQD2je8_ckpt_best.pth",
                        help='path to a pretrained model'
                        )

    args, opts = parser.parse_known_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml
    cfg.mode = "test"
    cfg.visualize = args.visualize

    if 's3dis' in cfg.dataset.common.NAME.lower():
        cfg.cmap = np.array(S3DIS.cmap)
        classes = S3DIS.classes
    # elif 'inropa' in cfg.dataset.common.NAME.lower():
    #     cfg.cmap = np.array(Inropa.cmap)
    #     classes = Inropa.classes
    elif 'novafos3d' in cfg.dataset.common.NAME.lower():
        cfg.cmap = np.array(Novafos3D.cmap)
        classes = Novafos3D.classes
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.cmap = np.array(SemanticKITTI.cmap)
        classes = SemanticKITTI.classes
    else:
        raise NotImplementedError

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

    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
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
        data_list = generate_data_list(cfg)
    else:
        data_list = [args.source]

    assert [os.path.exists(data) for data in data_list], f"Data path in {data_list} does not exist!"

    miou, macc, oa, ious, accs, cm = inferece(model, data_list, cfg)
    logging.info(f"Test mIoU: {miou:.4f}, mAcc: {macc:.4f}, OA: {oa:.4f}")

    for cls in classes:
        # print class wise iou and acc
        logging.info(f"{cls}: IoU: {ious[classes.index(cls)]:.4f}, Acc: {accs[classes.index(cls)]:.4f}")
