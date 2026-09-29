"""
3D object detection model training pipeline using pseudo labels based on OpenPCDet framework.

This script trains 3D object detection models (e.g., CenterPoint, TransFusion) using
pseudo labels generated from various pseudo-labeling approaches. The pipeline includes:
1. Configuration parsing and pseudo label file loading
2. Class mapping and filtering based on available pseudo labels
3. Dataset configuration adaptation for pseudo classes
4. Model architecture configuration for target classes
5. Training loop with checkpointing and logging
6. Evaluation on test set with trained models
"""

# Overwriting functions from pcdet
import src.model_train.core_overwrite

from pathlib import Path
import argparse
import datetime
import glob
import os
import json
import yaml
import shutil

from src.model_train.eval_utils import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
import pcdet.datasets
from src.model_train.optimization import build_optimizer, build_scheduler
from src.model_train.train_utils import train_model
from src.model_train.mapping import map_data_config, map_model_config
from src.model_train.dataset_info_creation import create_dataset_info
from src.labeling.mapping import get_class_mapping, get_mapping_classes
from src.data import EXP_TRAIN_OUT_PATH


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for training')
    parser.add_argument('--label_file', type=str, required=True, help='specify the (pseudo) label file - train')
    parser.add_argument('--pseudo_cfg_file', type=str, required=True, help='specify the config from the pseudo label generation')
    parser.add_argument('--subsample_ratio', type=float, default=1.0, help='subsample ratio for the dataset')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--no_cam_cache', action='store_true', default=False, help='if true, do not create the camera cache')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    assert args.label_file.endswith('.json') and Path(args.label_file).exists(), f"Label file {args.label_file} does not exist or is not a JSON file"
    assert args.pseudo_cfg_file.endswith('.yaml') and Path(args.pseudo_cfg_file).exists(), f"Pseudo config file {args.pseudo_cfg_file} does not exist or is not a YAML file"

    with open(args.pseudo_cfg_file, 'r') as f:
        pseudo_config = yaml.safe_load(f)

    with open(args.label_file, 'r') as f:
        label_file = json.load(f)
        mapping_name = label_file['mapping_name']
        assert label_file['split'] == 'all', "Label file should contain all annotations"
    
    # Filtering down class labels
    label_files_classes = set([obj['detection_name'] for sample in label_file['results'].values() for obj in sample])
    class_mapping = get_class_mapping(pseudo_config['data']['dataset'])[mapping_name]
    class_mapping = {k:v for k, v in class_mapping.items() if v in label_files_classes}
    pseudo_classes = get_mapping_classes()[mapping_name]
    pseudo_classes = [c for c in pseudo_classes if c in label_files_classes]
    pseudo_classes = sorted(pseudo_classes, key=lambda x: min([cfg.CLASS_NAMES.index(c) for c in class_mapping.keys() \
                                                               if class_mapping[c] == x and c in cfg.CLASS_NAMES]))

    cfg.DATA_CONFIG = map_data_config(cfg.DATA_CONFIG, pseudo_config, pseudo_classes, class_mapping)
    cfg.DATA_CONFIG.mapping_name = mapping_name

    cfg.DATA_CONFIG.SUBSAMPLE_RATIO = args.subsample_ratio

    config = map_model_config(cfg, pseudo_classes, class_mapping)

    args.extra_tag = f"{args.extra_tag}_s{args.subsample_ratio}" if args.subsample_ratio != 1.0 else args.extra_tag

    return args, config

def fix_configs_after_subsampling(cfg, args):
    nuscenes_dataset = pcdet.datasets.__all__['NuScenesDataset'](
                dataset_cfg=cfg.DATA_CONFIG, class_names=None,
                root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
                logger=common_utils.create_logger(), training=True
            )
    all_names = [list(info['gt_names']) for info in nuscenes_dataset.infos]
    classes_in_subsample = set([str(name) for names in all_names for name in names])
    class_mapping = {c: c for c in classes_in_subsample}
    pseudo_classes = [c for c in cfg.DATA_CONFIG.pseudo_classes if c in classes_in_subsample]
    
    with open(args.pseudo_cfg_file, 'r') as f:
        pseudo_config = yaml.safe_load(f)

    cfg.DATA_CONFIG = map_data_config(cfg.DATA_CONFIG, pseudo_config, pseudo_classes, class_mapping)

    cfg = map_model_config(cfg, pseudo_classes, class_mapping)

    return cfg

def main():
    args, cfg = parse_config()

    output_dir = EXP_TRAIN_OUT_PATH / Path(args.label_file).stem / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.label_file).resolve() != (output_dir / Path(args.label_file).name).resolve():
        shutil.copyfile(args.label_file, output_dir / Path(args.label_file).name)


    cfg.DATA_CONFIG = create_dataset_info(cfg.DATA_CONFIG, Path(args.label_file), output_dir, no_cam=args.no_cam_cache)

    if cfg.DATA_CONFIG.SUBSAMPLE_RATIO != 1.0:
        fix_configs_after_subsampling(cfg, args)

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH)
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))

        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (Path(args.label_file).stem, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (Path(args.label_file).stem, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (Path(args.label_file).stem, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH)
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (Path(args.label_file).stem, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
