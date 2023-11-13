import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import wandb   

from openpoints.utils import EasyConfig, dist_utils, generate_exp_directory, resume_exp_directory
from examples.segmentation.main import main as segmentation_main


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Semantic segmentation training script')
    parser.add_argument('--cfg', type=str, help='config file', default="cfgs/novafos3d/pointnext-xl.yaml")
    parser.add_argument('--deterministic', type=int, help='Whether to run training deterministic', default=1)
    parser.add_argument('--radius', type=float, default=0.1, help='Radius of initial set abstraction ball query')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size to use')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size used for voxel downsampling')
    parser.add_argument('--voxel_max', type=float, default=30000, help='subsample the max number of point per point cloud. Set None to use all points.')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--epochs', type=int, default=50, help="Epochs to use")
    parser.add_argument('--pretrained_path', type=str,
                        help='path to a pretrained model')
    

    # default="/home/lasse/Git/PointNeXt/log/novafos3d/novafos3d-train-pointnext-xl-ngpus1-seed2696-20230210-150344-2PXLfpA5HQ8UYCXUJSr5gR/checkpoint/novafos3d-train-pointnext-xl-ngpus1-seed2696-20230210-150344-2PXLfpA5HQ8UYCXUJSr5gR_ckpt_best.pth",
    
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml
    cfg.deterministic = args.deterministic
    cfg.mode = args.mode

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if args.voxel_size is not None:
        cfg.dataset.common.voxel_size = args.voxel_size

    if args.radius is not None:
        cfg.model.encoder_args.radius = args.radius

    if args.voxel_max is not None:
        cfg.dataset.train.voxel_max = args.voxel_max

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    if cfg.epochs is not None:
        cfg.epochs = args.epochs

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
        f'batch_size={args.batch_size}',
        f'voxel_size={args.voxel_size}',
        f'voxel_max={args.voxel_max}',
        f'radius={args.radius}'
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = tags
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # # logger
    # setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)

    # # set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    # torch.backends.cudnn.enabled = True
    # # logging.info(cfg)

    # if cfg.model.get('in_channels', None) is None:
    #     cfg.model.in_channels = cfg.model.encoder_args.in_channels
    # model = build_model_from_cfg(cfg.model).to(cfg.rank)
    # model_size = cal_model_parm_nums(model)
    # # logging.info(model)
    # logging.info('Number of params: %.4f M' % (model_size / 1e6))

    # best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)

    # logging.info(f'Testing model {os.path.basename(cfg.pretrained_path)}...')

    # wandb config
    cfg.wandb.name = cfg.run_name

    gpu = 0 # 1
    segmentation_main(gpu, cfg)
