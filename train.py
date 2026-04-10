"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

"""
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/dfine_hgnetv2_n_IRSTD.yml --use-amp
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # ===== priority 0: 基础/最常用参数 =====
    # 配置文件路径（yaml），内部会先加载该配置
    parser.add_argument(
        '-c', '--config', type=str, default='',
        help='path to config file (yaml)'
    )
    # 从某个 checkpoint 继续训练（包括模型和优化器状态）
    parser.add_argument(
        '-r', '--resume', type=str,
        help='resume training from given checkpoint'
    )
    # 从某个 checkpoint 做微调（通常只加载模型权重，不加载优化器）
    parser.add_argument(
        '-t', '--tuning', type=str,
        help='finetune from given checkpoint'
    )
    # 指定运行设备，例如 "cuda:0" 或 "cpu"
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0',
        help='device to use, e.g. "cuda:0" or "cpu"'
    )
    # 随机种子，保证实验可复现
    parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed for experiment reproducibility'
    )
    # 是否启用自动混合精度训练（节省显存、加速）
    parser.add_argument(
        '--use-amp', action='store_true',
        help='enable automatic mixed precision training'
    )
    # 输出目录，用于保存日志、模型等
    parser.add_argument(
        '--output-dir', type=str,
        help='directory to save checkpoints and logs'
    )
    # TensorBoard 日志目录
    parser.add_argument(
        '--summary-dir', type=str,
        help='directory to save tensorboard summaries'
    )
    # 只做测试，不进行训练
    parser.add_argument(
        '--test-only', action='store_true', default=False,
        help='only run evaluation / testing, no training'
    )

    # ===== priority 1: 覆盖 / 更新 yaml 配置的参数 =====
    # 以 key value 形式更新 yaml 配置，例如：
    # -u train_dataloader.dataset.img_folder E:/Dataset/IRSTD-1k/IRSTD1k_Img/
    parser.add_argument(
        '-u', '--update', nargs='+',
        help='override config from command line, e.g. key value pairs'
    )

    # ===== 环境相关参数（多进程/分布式等） =====
    # 打印方法："builtin" 或 其他自定义方法名
    parser.add_argument(
        '--print-method', type=str, default='rich',
        help='print method to use, e.g. "builtin"'
    )
    # 只在指定 rank 上打印日志（多卡训练时用）
    parser.add_argument(
        '--print-rank', type=int, default=0,
        help='rank id that is allowed to print logs'
    )
    # 分布式训练时的本地 rank，由启动脚本/torchrun 注入
    parser.add_argument(
        '--local-rank', type=int,
        help='local rank id for distributed training'
    )

    args = parser.parse_args()
    main(args)
