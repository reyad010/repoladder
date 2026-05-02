# Genetic NAS search driver for layer-reduction LST.
# Default mode: gpu-cpu (split, two-process IPC with tee_server.py).
# Run via: bash run.sh   (which launches ../artifact2_e2e/tee_server.py + this script)

import argparse
import gc
import os
import sys

# Make ../shared/ importable (contains nas/, models/, modules/, utils/, dataloader.py)
# and ../artifact2_e2e/ (contains shmio.py + shm_bridge.so for the SHM channel).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
sys.path.insert(0, os.path.join(_HERE, '..', 'artifact2_e2e'))

import numpy as np
import torch
import torch.cuda

from nas.config import GeneticArchSearchConfig
from nas.manager_genetic import ArchSearchRunManager
from models.lst_vgg.ladder_side_tune_model import LadderSideTuneNet as VGG_LST
from models.lst_resnet.ladder_side_tune_model import LadderSideTuneNet as ResNet_LST
from models.lst_vit.ladder_side_tune_model import LadderSideTuneNet as ViT_LST
from dataloader import SingleLoader, num_class_map


MODEL_MAP_LST = {
    'vgg':      VGG_LST,
    'resnet':   ResNet_LST,
    'vit-base': ViT_LST,
}


def parse_args():
    p = argparse.ArgumentParser(description='Genetic NAS for layer-reduction LST')
    p.add_argument('--path', type=str, default=None,
                   help='Output dir for logs (default: ./results/<model>_<dataset>)')
    p.add_argument('--gpu', default='0')
    p.add_argument('--manual_seed', type=int, default=42)

    p.add_argument('--model_name', type=str, default='vgg',
                   choices=['vgg', 'resnet', 'vit-base'])
    p.add_argument('--dataset', type=str, default='gtsrb')
    p.add_argument('--train_batch_size', type=int, default=128)
    p.add_argument('--test_batch_size', type=int, default=128)
    p.add_argument('--rf', type=int, default=4)

    # GA hyperparameters
    p.add_argument('--npop', type=int, default=16, help='Population size')
    p.add_argument('--n_generation', type=int, default=20, help='# generations')
    p.add_argument('--mutation_sigma', type=float, default=8.0)
    p.add_argument('--reward_beta', type=float, default=30.0,
                   help='Reward weight on NAS score (vs. latency)')

    # Latency measurement mode
    p.add_argument('--test_device', default='split',
                   choices=['split', 'single'],
                   help='split: two-process IPC with tee_server.py (gpu-cpu / gpu-tee). '
                        'single: single-process, side runs on CPU (no tee_server needed).')

    # Optimizer args (consumed by RunManager.build_optimizer)
    p.add_argument('--init_lr', type=float, default=0.0001)
    p.add_argument('--lr_schedule_type', type=str, default='no')
    p.add_argument('--opt_type', type=str, default='adam', choices=['adam', 'sgd'])
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--no_nesterov', action='store_true')
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--no_decay_keys', type=str, default=None,
                   choices=[None, 'bn', 'bn#bias'])
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--print_frequency', type=int, default=100)
    p.add_argument('--validation_frequency', type=int, default=1)
    p.add_argument('--n_worker', type=int, default=4)

    return p.parse_args()


def main():
    args = parse_args()
    if args.path is None:
        args.path = f'./results/{args.model_name}_{args.dataset}'
    os.makedirs(args.path, exist_ok=True)

    for k, v in args.__dict__.items():
        print(f'{k:<20} {v}')

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    num_class = num_class_map[args.dataset]
    image_size = 224 if args.model_name == 'vit-base' else 64
    loader = SingleLoader(
        task=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        image_size=image_size,
        device='cuda:0',
    )

    search_space = {
        'connection': ['empty', 'regular'],
        'side':       ['empty', 'same'],
        'reduction_factor': [4, 8, 16, 32, 64],
    }

    super_net = MODEL_MAP_LST[args.model_name](
        search_space=search_space, num_classes=num_class,
    )
    arch_search_config = GeneticArchSearchConfig(search_space, **args.__dict__)
    arch_search_config.model_name = args.model_name

    runner = ArchSearchRunManager(
        args, super_net, run_config=None,
        arch_search_config=arch_search_config, loader=loader,
    )
    final_config = runner.train()
    print(f'\n[NAS] best config: {final_config}')


if __name__ == '__main__':
    main()
