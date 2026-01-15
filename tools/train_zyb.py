# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Tianyu Li
# ---------------------------------------------
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
import sys
tools_dir = os.path.dirname(os.path.abspath(__file__))
LaneSegNet_dir = os.path.dirname(tools_dir)
sys.path.append(tools_dir)
sys.path.append(LaneSegNet_dir)
import projects


import pickle
from collections import defaultdict
# enable tensor core
torch.backends.mudnn.allow_tf32 = True

def process_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return [process_tensor(item) for item in x]
    else:
        return x
    
def get_activation_and_weights_hook(layer_name, save_dir="activations"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    def hook(module, input, output):
        processed_output = process_tensor(output)
        save_path = os.path.join(save_dir, f"{layer_name}.out.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(processed_output, f)
        print(f"已保存 {layer_name} 的激活值")
        # print(f"zyb debug: {layer_name}.training: {module.training}")
        return 
        
        # 保存 module 
        hook_id = None
        for id, h in module._forward_hooks.items():
            if h == hook:  # 找到当前hook
                hook_id = id
                break
        if hook_id is not None:
            del module._forward_hooks[hook_id]  # 移除引用，避免pickle序列化
        # print(f"zyb debug: {layer_name}.training: {module.training}")
        
        # 保存路径：{layer_name}.module.pth
        save_path = os.path.join(save_dir, f"{layer_name}.module.pth")
        
        # 直接保存整个模块
        torch.save(module.cpu(), save_path)
        print(f"已保存整个 {layer_name} 模块权重")
        
        printBatchNorm2d= True
        if printBatchNorm2d:
            restored_layer_train_gpu = module
            import torch.nn as nn
            if isinstance(restored_layer_train_gpu, nn.BatchNorm2d):
                print("\n===== BN2d 完整参数与状态 =====")
                # 可学习参数
                if hasattr(restored_layer_train_gpu, 'weight'):
                    print(f"weight（缩放系数）: {restored_layer_train_gpu.weight.shape}, 设备: {restored_layer_train_gpu.weight.device}")
                if hasattr(restored_layer_train_gpu, 'bias') and restored_layer_train_gpu.bias is not None:
                    print(f"bias（偏移系数）: {restored_layer_train_gpu.bias.shape}, 设备: {restored_layer_train_gpu.bias.device}")
                
                # 缓冲区（训练统计信息）
                if hasattr(restored_layer_train_gpu, 'running_mean'):
                    print(f"running_mean（移动平均均值）: {restored_layer_train_gpu.running_mean.shape}, 设备: {restored_layer_train_gpu.running_mean.device}")
                if hasattr(restored_layer_train_gpu, 'running_var'):
                    print(f"running_var（移动平均方差）: {restored_layer_train_gpu.running_var.shape}, 设备: {restored_layer_train_gpu.running_var.device}")
                if hasattr(restored_layer_train_gpu, 'num_batches_tracked'):
                    print(f"num_batches_tracked（跟踪批次数）: {restored_layer_train_gpu.num_batches_tracked.item()}")  #  scalar
                
                # 训练状态
                print(f"training（是否训练模式）: {restored_layer_train_gpu.training}")
    
    return hook


try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation version >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        if 'auto_scale_lr' in cfg and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file.')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name='mmdet')

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # if torch.cuda.is_available():
    #     checkpoint_path = "/data/yibiao.zhou/LaneSegNet/LaneSegNet/tools/lanesegnet_r50_8x1_24e_olv2_subset_A.pth"
    #     print(f"zyb debug: 使用 CUDA 设备，loading pretrained weights from {checkpoint_path}")
    # elif hasattr(torch, 'musa') and torch.musa.is_available():
    #     checkpoint_path = "/data/yibiao.zhou/LaneSegNet-Adaption/LaneSegNet/tools/lanesegnet_r50_8x1_24e_olv2_subset_A.pth"
    #     print(f"zyb debug: 使用 MUSA 设备，loading pretrained weights from {checkpoint_path}")
    # else:
    #     print("zyb debug: 需要 CUDA 或 MUSA 设备")
    #     exit(-1) 
    
    
    # checkpoint = load_checkpoint(model, checkpoint_path , map_location='cpu')
    # print(f"zyb debug: loaded pretrained weights from {checkpoint_path}")
    # model.train()
    # print(f"after .train(), model.training: {model.training}")
    # logger.info(f'Model:\n{model}')
    
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    auto_scale_lr(cfg, distributed=distributed, logger=logger)
    
    # print("zyb debug: print model named_modules")
    # for name, module in model.named_modules():
    #     print(name)  # 输出类似：backbone.blocks.0.conv、neck.fpn_convs.0.conv等
    #     print(type(module))
    # print("zyb debug: print model named_modules")
    # return
     
    target_layers_activations = [  # 要监控的中间层（根据模型结构修改）
        "img_backbone.conv1",
        "img_backbone.bn1",
        "img_backbone.relu",
        "img_backbone.maxpool",        
        # img_backbone.layer1.0
        "img_backbone.layer1.0.conv1",
        "img_backbone.layer1.0.bn1",
        "img_backbone.layer1.0.conv3",
        "img_backbone.layer1.0.bn3",
        "img_backbone.layer1.0.relu",
        # img_backbone.layer1.1
        "img_backbone.layer1.1.conv1",
        "img_backbone.layer1.1.bn1",
        "img_backbone.layer1.1.conv3",
        "img_backbone.layer1.1.bn3",
        "img_backbone.layer1.1.relu",
        # output of img_backbone
        "img_backbone.layer4.2.conv3",
        "img_backbone.layer4.2.bn3",
        "img_backbone.layer4.2.relu",
        # img_neck
        "img_neck.lateral_convs.0.conv",
        "img_neck.lateral_convs.1.conv",
        "img_neck.fpn_convs.0.conv",
        # last 2 layer of bev_constructor
        "bev_constructor.can_bus_mlp.3",
        "bev_constructor.can_bus_mlp.norm",
        # pts_bbox_head
        "pts_bbox_head.loss_cls",
        "pts_bbox_head.loss_bbox",
        "pts_bbox_head.loss_lane_type",
        "pts_bbox_head.loss_mask",
        "pts_bbox_head.loss_dice",
        "pts_bbox_head.activate",
    ]
    
    target_layers_weight = [  # 要监控的中间层（根据模型结构修改）
        # "img_backbone.conv1",
        # "img_backbone.bn1",
        # "img_backbone.relu",
        # "img_backbone.maxpool",        
        # # img_backbone.layer1.0
        # "img_backbone.layer1.0.conv1",
        # "img_backbone.layer1.0.bn1",
        # "img_backbone.layer1.0.conv3",
        # "img_backbone.layer1.0.bn3",
        # "img_backbone.layer1.0.relu",
        # # img_backbone.layer1.1
        # "img_backbone.layer1.1.conv1",
        # "img_backbone.layer1.1.bn1",
        # "img_backbone.layer1.1.conv3",
        # "img_backbone.layer1.1.bn3",
        # "img_backbone.layer1.1.relu",
        # output of img_backbone
        "img_backbone.layer4.2.conv3",
        "img_backbone.layer4.2.bn3",
        "img_backbone.layer4.2.relu",
        # # img_neck
        # "img_neck.lateral_convs.0.conv",
        # "img_neck.lateral_convs.1.conv",
        # "img_neck.fpn_convs.0.conv",
        # # last 2 layer of bev_constructor
        # "bev_constructor.can_bus_mlp.3",
        # "bev_constructor.can_bus_mlp.norm",
        # # pts_bbox_head
        # "pts_bbox_head.loss_cls",
        # "pts_bbox_head.loss_bbox",
        # "pts_bbox_head.loss_lane_type",
        # "pts_bbox_head.loss_mask",
        # "pts_bbox_head.loss_dice",
        # "pts_bbox_head.activate",
    ]
    
    target_layers = list(set(target_layers_weight))
    target_layers = list(set(target_layers_activations))
    
    if torch.cuda.is_available():
        activation_saving_dir = "/data/yibiao.zhou/LaneSegNet/debuging_files/nv_activations"
        print(f"zyb debug: 使用 CUDA 设备，中间变量保存到{activation_saving_dir}")
    elif hasattr(torch, 'musa') and torch.musa.is_available():
        activation_saving_dir = "/data/yibiao.zhou/LaneSegNet-Adaption/LaneSegNet/debug_files/forward_loss/mt_activations"
        print(f"zyb debug: 使用 MUSA 设备，中间变量保存到{activation_saving_dir}")
    else:
        print("zyb debug: 需要 CUDA 或 MUSA 设备来保存中间变量，当前没有可用设备，程序退出")
        exit(-1) 
                    
    # hooks = []
    # for layer_name in target_layers:
    #     # 获取目标模块
    #     module = dict(model.named_modules())[layer_name]
    #     # 注册钩子（指定层名和保存目录）
    #     hook = module.register_forward_hook(get_activation_and_weights_hook(layer_name, save_dir=activation_saving_dir))
    #     hooks.append(hook)
    
    print(f"before train_model, model.training: {model.training}")
    channel_last = __import__('os').getenv('CHANNEL_LAST', '0') == '1'
    if channel_last:
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m = m.to(memory_format=torch.channels_last)
        print(f'zyb debug: set to channel last')
    
    from torch_musa.utils.compare_tool import NanInfTracker, CompareWithCPU
    # with NanInfTracker(enabled=True, 
    #                    should_log_to_file=True, 
    #                    output_dir="/data/yibiao.zhou/LaneSegNet-Adaption/Debug_Files/debug_files/nan_inf_logs_2026/",
    #                    target_list=["torch.ops.aten.native_batch_norm", "torch.ops.aten.native_batch_norm_backward"],
    #                    ):
    # with CompareWithCPU(enabled=False, 
    #                    should_log_to_file=True, 
    #                    dump_error_data = True,
    #                    output_dir="/data/yibiao.zhou/LaneSegNet-Adaption/Debug_Files/debug_files/nan_inf_logs_2026/",
    #                    target_list=["torch.ops.aten.native_batch_norm", "torch.ops.aten.native_batch_norm_backward"],
    #                    ):
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
