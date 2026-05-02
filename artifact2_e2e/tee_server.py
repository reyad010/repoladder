#!/usr/bin/env python3
# Enclave-side inference server.
# Runs the LST side network as a separate process; communicates with backbone.py
# via POSIX shared memory (shm_bridge.cpp / shmio.py).
# Run inside SGX:   gramine-sgx ./pytorch tee_server.py
# Run on CPU:       python tee_server.py

import argparse
import copy
import gc
import math
import os
import random
import sys
import time
from collections import defaultdict

# Make ../shared/ importable (contains models/, modules/, utils/, dataloader.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'shared'))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from utils.tensor_transfer_channel import recv_value, send_value, _ensure_dir
from shmio import close_shm, ack, close_all, alloc_shm_tensor
from models.lst_vgg.ladder_side_tune_model import LadderSideTuneNet as VGG_LST
from models.lst_resnet.ladder_side_tune_model import LadderSideTuneNet as ResNet_LST
from models.lst_vit.ladder_side_tune_model import LadderSideTuneNet as ViT_LST


model_map_lst = {
    'vgg':      VGG_LST,
    'resnet':   ResNet_LST,
    'vit-base': ViT_LST,
}


def summarize_mask_unmask(side_time_breakdown, warmup, cfg_all, model_name):
    """Print a table of mask/unmask latency (mean ± std) per (op, tag) pair,
    averaged over post-warmup iterations only."""
    post_warmup = side_time_breakdown[warmup:]
    if not post_warmup:
        return

    stats = defaultdict(list)
    shapes = {}
    op_total = defaultdict(list)
    shape_stats = defaultdict(list)

    for tl in post_warmup:
        iter_op_sum = defaultdict(int)
        for entry in tl.get('_mask_unmask_log', []):
            op, tag, shape, t0, t1 = entry
            lat = t1 - t0
            key = (op, tag)
            stats[key].append(lat)
            shapes[key] = shape
            iter_op_sum[op] += lat
            shape_stats[shape].append(lat)
        for op, total in iter_op_sum.items():
            op_total[op].append(total)

    if not stats:
        return

    def sort_key(k):
        op, tag = k
        return (0 if op == 'unmask' else 1, tag)

    keys = sorted(stats.keys(), key=sort_key)
    cfg_str = str(cfg_all)
    n_iters = len(post_warmup)
    header = (f"Mask/Unmask Summary  |  cfg={cfg_str}  model={model_name}  "
              f"({n_iters} iters, {warmup} warmup skipped)")
    width = max(len(header), 110)
    print('=' * width, flush=True)
    print(header, flush=True)
    print('=' * width, flush=True)
    print(f"{'Op':<8}  {'Tag':<40}  {'Shape':<32}  {'Size(MB)':>9}  "
          f"{'Mean(ms)':>10}  {'Std(ms)':>9}  {'BW(GB/s)':>10}", flush=True)
    print('-' * width, flush=True)

    for key in keys:
        op, tag = key
        shape = shapes[key]
        lats = stats[key]
        n_elems = math.prod(shape)
        size_mb = n_elems * 4 / 1024 / 1024
        mean_ms = np.mean(lats) / 1e6
        std_ms = np.std(lats) / 1e6
        bw_gbs = 3 * size_mb / mean_ms
        print(f"{op:<8}  {tag:<40}  {str(shape):<32}  {size_mb:>9.1f}  "
              f"{mean_ms:>10.2f}  {std_ms:>9.2f}  {bw_gbs:>10.1f}", flush=True)

    print('-' * width, flush=True)
    for op in ['unmask', 'mask']:
        if op not in op_total:
            continue
        lats = op_total[op]
        mean_ms = np.mean(lats) / 1e6
        std_ms = np.std(lats) / 1e6
        print(f"{op:<8}  {'[TOTAL]':<40}  {'':32}  {'':>9}  "
              f"{mean_ms:>10.2f}  {std_ms:>9.2f}  {'':>10}", flush=True)

    print('-' * width, flush=True)
    shape_order = sorted(shape_stats.keys(), key=lambda s: (len(s), s))
    for shape in shape_order:
        lats = shape_stats[shape]
        n_elems = math.prod(shape)
        size_mb = n_elems * 4 / 1024 / 1024
        mean_ms = np.mean(lats) / 1e6
        std_ms = np.std(lats) / 1e6
        bw_gbs = 3 * size_mb / mean_ms
        print(f"{'':8}  {'':40}  {str(shape):<32}  {size_mb:>9.1f}  "
              f"{mean_ms:>10.2f}  {std_ms:>9.2f}  {bw_gbs:>10.1f}", flush=True)

    print('=' * width, flush=True)
    print(flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _patch_dummy_batch(m, B):
    """Mirror of backbone.py's _patch_dummy_batch — keeps both sides' transfer_shapes
    in sync when --batch_sizes is used. No-op for B in (None, 0)."""
    if not B or not hasattr(m, 'get_dummy_inputs'):
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


def role_side_server(args, verbose_trace=False):
    n_threads = int(getattr(args, 'threads', 4) or 4)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)
    print(f'set threads to {n_threads}')

    for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[k] = str(n_threads)
    time.sleep(3)

    print("[side] server started; waiting for cfg...", flush=True)
    search_space = {
        'connection': ['empty', 'regular'],
        'side':       ['empty', 'same'],
        'reduction_factor': [4, 8, 16, 32, 64],
    }
    count = 0
    super_net = None
    model = None
    while True:
        print('=' * 50 + f"[side] cfg {count + 1}" + '=' * 50, flush=True)
        recv_args = recv_value('iter', 'list')
        if recv_args == ['__done__']:
            ack('iter')
            close_all()
            break
        warmup, iters, model_name, num_classes = recv_args[:4]
        batch_size = recv_args[4] if len(recv_args) > 4 else None
        cfg_all = recv_value('cfg', 'list')
        ack('iter')
        ack('cfg')

        print(f"[side] cfg received: model={model_name}, num_class={num_classes}, "
              f"warmup={warmup}, iters={iters}", flush=True)
        print(f"[side] cfg: {cfg_all}", flush=True)

        # Drop the previous config's model before building a new one.
        del model
        model = None
        gc.collect()

        # Artifact bundle: LST only. The original repo also handled LoRA / NPLO
        # baselines; those branches are not shipped here.
        if super_net is None or super_net.model_name != model_name:
            if super_net is not None:
                del super_net
                super_net = None
                gc.collect()
            super_net = model_map_lst[model_name](search_space, num_classes=num_classes)
        model = super_net.convert_to_normal_net(cfg_all)

        _patch_dummy_batch(model, batch_size)

        model.convert_to_side_model('cpu')
        model.eval()

        shm_tensors = {}
        tee_owned = getattr(model, '_owned_shm_keys', None)
        for shm_name, shape in model.transfer_shapes.items():
            if tee_owned is not None:
                if shm_name not in tee_owned:
                    continue
            elif '_out' not in shm_name:
                continue
            t, _ = alloc_shm_tensor(
                shm_name, shape,
                dtype=getattr(model, 'compute_dtype', torch.float32), pin=False,
            )
            shm_tensors[shm_name] = t
        print(f'allocate shm tensor {shm_tensors.keys()}')
        print("[side] side model ready", flush=True)

        side_time_breakdown = []
        with torch.no_grad():
            for i in range(int(iters + warmup)):
                output, timeline = model(i, False)
                side_time_breakdown.append(copy.deepcopy(timeline))
                send_value('sync', [time.time_ns()], 'list')

        close_shm('sync', unlink=False)

        summarize_mask_unmask(side_time_breakdown, warmup, cfg_all, model_name)
        if verbose_trace:
            print(f'[side] time breakdown for cfg: {cfg_all}: {side_time_breakdown}',
                  flush=True)
        count += 1
        # Drop tensors that view the shm regions BEFORE close_all() tears the
        # mappings down; otherwise tensors point into munmap'd memory.
        shm_tensors.clear()
        del shm_tensors
        close_all()
        gc.collect()
        time.sleep(10.0)

    print(f'[side] server terminated', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--share_root', default='./file_share')
    ap.add_argument('--verbose_trace', action='store_true', default=False)
    ap.add_argument('--threads', type=int, default=4,
                    help='Number of CPU threads for the side process '
                         '(torch.set_num_threads + OMP/MKL/etc).')
    args = ap.parse_args()

    _ensure_dir(args.share_root)
    role_side_server(args, verbose_trace=args.verbose_trace)


if __name__ == '__main__':
    set_seed(42)
    main()
