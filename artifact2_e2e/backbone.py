# GPU-side backbone process for end-to-end latency measurement.
# Communicates with tee_server.py via shared memory (shm_bridge / shmio).
# This artifact ships the LST path for VGG only; LoRA / NPLO baselines
# from the paper are not included in the artifact bundle.

import argparse
import copy
import gc
import os
import sys
import time

# Make ../shared/ importable (contains nas/, models/, modules/, utils/, dataloader.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
# shmio.py lives next to this file (artifact2_e2e/) and is imported transitively.
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.cuda

from nas.config import GeneticArchSearchConfig
from models.lst_vgg.ladder_side_tune_model import LadderSideTuneNet as VGG_LST
from models.lst_resnet.ladder_side_tune_model import LadderSideTuneNet as ResNet_LST
from models.lst_vit.ladder_side_tune_model import LadderSideTuneNet as ViT_LST
from dataloader import num_class_map
from utils.model_deploy import measure_inference_latency
from utils.tensor_transfer_channel import send_value


MODEL_MAP_LST = {
    'vgg':      VGG_LST,
    'resnet':   ResNet_LST,
    'vit-base': ViT_LST,
}

# Conv backbones use NHWC (channels_last) on GPU; ViT does not.
CHANNELS_LAST = {'vgg': True, 'resnet': True, 'vit-base': False}


def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--manual_seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='vgg',
                        choices=['vgg', 'resnet', 'vit-base'])
    parser.add_argument('--dataset', type=str, default='gtsrb')
    parser.add_argument('--rf', type=int, default=4)
    parser.add_argument('--reward_beta', type=float, default=30.0)

    parser.add_argument('--cfg', type=str, default=None,
                        help='Single config to run (e.g. lst.8.vgg.gtsrb). '
                             'Overrides the default 3-config workload. Mutually '
                             'exclusive with --configs_file.')
    parser.add_argument('--configs_file', type=str, default=None,
                        help='JSON file with NAS-identified configs to replay '
                             '(format: {"all_cfgs": ["lst.8.vgg.gtsrb", [name, gene]]}).')
    parser.add_argument('--verbose_trace', action='store_true', default=False)
    parser.add_argument('--batch_sizes', type=str, default=None,
                        help='Comma-separated batch sizes (e.g. "1,2,4"). Each cfg is '
                             'replayed at every batch size; latency is reported per '
                             '(cfg, batch_size). Default: single run at the model default.')
    return parser.parse_args()


def deal_lst_cfgs(cfg):
    parts = cfg.split('_')
    return 'lst', parts[0], parts[1], None


def deal_standard_cfgs(cfg):
    parts = cfg.split('.')
    if len(parts) == 4:
        peft, rf, model_name, dataset = parts
        return peft, model_name, dataset, int(rf)
    peft, model_name, dataset = parts
    return peft, model_name, dataset, None


def _patch_dummy_batch(m, B):
    """Override m.get_dummy_inputs / get_dummy_features to return batch=B tensors,
    so transfer_shapes (and the SHM regions allocated from them) match the side."""
    if B is None or not hasattr(m, 'get_dummy_inputs'):
        return
    orig_inputs = m.get_dummy_inputs
    orig_feats = getattr(m, 'get_dummy_features', None)

    def _rebatch(t, channels_last):
        if t is None or t.shape[0] == B:
            return t
        single = t[:1]
        out = single.expand(B, *single.shape[1:]).contiguous()
        if channels_last and out.ndim == 4:
            out = out.contiguous(memory_format=torch.channels_last)
        return out

    def new_inputs(device='cpu', channels_last=True, **kw):
        t = orig_inputs(device=device, channels_last=channels_last, **kw)
        if not isinstance(t, torch.Tensor):
            return t
        return _rebatch(t, channels_last)
    m.get_dummy_inputs = new_inputs

    if orig_feats is not None:
        def new_feats(device='cpu', channels_last=True, **kw):
            feats = orig_feats(device=device, channels_last=channels_last, **kw)
            return [_rebatch(f, channels_last) for f in feats]
        m.get_dummy_features = new_feats


if __name__ == '__main__':
    args = parser_set()
    torch.cuda.empty_cache()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0'

    for k, v in args.__dict__.items():
        print(f'{k:<20} {v}')

    # Default workload: the three LST configs reported in the paper (Table 1).
    # Override with --cfg (single config) or --configs_file (JSON list).
    all_cfgs = [
        'lst.8.vgg.gtsrb',
        'lst.8.resnet.cifar10',
        'lst.8.vit-base.cifar100',
    ]
    if args.cfg and args.configs_file:
        raise SystemExit('--cfg and --configs_file are mutually exclusive')
    if args.cfg:
        all_cfgs = [args.cfg]
    if args.configs_file:
        import json
        with open(args.configs_file) as f:
            file_cfgs = json.load(f)
        if isinstance(file_cfgs, list):
            all_cfgs = [
                (f"{t['model_name']}_{t['dataset']}_beta{t['beta']}_hof{t['hof_idx']}",
                 t['config_all'])
                for t in file_cfgs
            ]
        else:
            if 'all_cfgs' in file_cfgs:
                all_cfgs = [tuple(x) if isinstance(x, list) else x
                            for x in file_cfgs['all_cfgs']]

    if args.batch_sizes:
        batch_size_list = [int(b) for b in args.batch_sizes.split(',') if b.strip()]
    else:
        batch_size_list = [None]
    all_cfgs = [(item, B) for item in all_cfgs for B in batch_size_list]

    test_device = 'split'
    init_latency_test = True
    latencies = []
    time_breakdown = []

    model = None
    super_net = None
    for test_round, (item, batch_size) in enumerate(all_cfgs):
        # Drop previous model + super_net before constructing the next one.
        del model, super_net
        gc.collect()
        torch.cuda.empty_cache()
        model = None
        super_net = None

        cfg, gene = item if isinstance(item, tuple) else (item, None)
        peft, model_name, dataset, cfg_rf = (
            deal_lst_cfgs if gene is not None else deal_standard_cfgs
        )(cfg)
        if cfg_rf is not None:
            args.rf = cfg_rf
        args.model_name = model_name
        args.dataset = dataset

        if peft != 'lst':
            raise NotImplementedError(
                f'This artifact bundle ships LST only; got peft={peft!r} from cfg={cfg!r}'
            )

        num_class = num_class_map[args.dataset]
        channels_last = CHANNELS_LAST[args.model_name]
        print('=' * 50 + f' Config {test_round + 1}: {cfg} ' + '=' * 50, flush=True)

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
        if gene is not None:
            cur_config = {'all': gene}
        else:
            cur_config = arch_search_config.get_good_ladder_config(search_space, rf=args.rf)
        model = super_net.convert_to_normal_net(cur_config['all']).to(device)

        _patch_dummy_batch(model, batch_size)
        results = measure_inference_latency(
            net=model, device=test_device, test_num=10, channels_last=channels_last,
            gpu=args.gpu, lst_config=cur_config, transfer_type='shm',
            init_latency_test=init_latency_test, batch_size=batch_size,
        )
        latency = results[test_device]
        time_breakdown.append(results.get('time_breakdown', None))
        init_latency_test = False
        latencies.append(latency)
        torch.cuda.empty_cache()
        print(cfg, batch_size, latency, flush=True)
        if args.verbose_trace:
            print(f'[backbone] time breakdown for {cur_config["all"]}: '
                  f'{results.get("time_breakdown", None)}', flush=True)
        time.sleep(10.0)

    # Tell tee_server.py to exit.
    send_value('iter', ['__done__'], 'list')

    for (item, batch_size), latency in zip(all_cfgs, latencies):
        cfg = item[0] if isinstance(item, tuple) else item
        if batch_size is None:
            print(f'{cfg}: {latency}')
        else:
            print(f'{cfg}  bs={batch_size}: {latency}')
    print(f'backbone final latency: {latencies}')
