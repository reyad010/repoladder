# Description: This file contains some utility functions.
import gc
import statistics
import psutil, torch, threading
try:from shmio import close_shm, ack, alloc_shm_tensor, tensor_hash, close_all
except Exception as e:close_shm, ack, alloc_shm_tensor, tensor_hash = None, None, None, None
import time
import hashlib
import os, json, pathlib


def _cpus_for_numa(node: int, n: int, same: bool):
    """Return a list of CPU cores from a specific NUMA node."""
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        raw = f.read().strip()

    cpus = []
    for part in raw.split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            cpus.extend(range(a, b + 1))
        else:
            cpus.append(int(part))

    # skip the first CPU (often used by kernel or interrupts)
    if n is None:
        return cpus

    if same:
        # use first n cores (skipping CPU 0)
        return cpus[4:n + 4]
    else:
        # use offset cores to avoid overlap with the 'side' process
        return [24] # cpus[10:10 + n]

def set_cores(device, gpu='0', flex=False, same=False, threads=1):
    """Configure CPU core affinity and threading policy."""
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    total_cores = 8
    gpu_to_node = {'0': 0, '1': 1}
    target_node = gpu_to_node.get(str(gpu), 0)

    if device != 'split':
        # --- Normal GPU-only mode ---
        n_threads = total_cores
        candidate_cpus = _cpus_for_numa(target_node, n=total_cores, same=True)
        mode_desc = "Normal (GPU-only): use 8 cores"

    elif device == 'split' and not flex:
        # --- split (two-process), fixed single core ---
        n_threads = threads
        candidate_cpus = _cpus_for_numa(target_node, n=threads, same=same)
        mode_desc = f"split fixed mode: {n_threads} thread(s) (same={same})"

    else:  # device == 'split' and flex == True
        # --- split (two-process), flexible mode: same 8 cores as side, but single-threaded ---
        n_threads = threads
        candidate_cpus = _cpus_for_numa(target_node, n=total_cores, same=True)
        mode_desc = f"GPU-TEE flexible mode: same 8 cores as side ({n_threads} thread(s))"

    # Apply CPU affinity
    try:
        psutil.Process().cpu_affinity(candidate_cpus) # taskset -c candidates_cpus
        print(f"[backbone] {mode_desc}")
        print(f"[backbone] allowed CPU pool: {candidate_cpus}")
    except Exception as e:
        print(f"Warning: set cpu affinity failed: {e}")

    # Cap threading libraries and torch
    for k in [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS'
    ]:
        os.environ[k] = str(n_threads)
    torch.set_num_threads(n_threads)

from typing import List, Dict, Any
from statistics import mean

MS_SCALE = 1e6  # your timestamps look like nanoseconds → ms

def _merge_and_normalize_once(
    bb: Dict[str, int],
    sd: Dict[str, int],
    *,
    scale: float = MS_SCALE,
) -> Dict[str, float]:
    """Merge two timestamp dicts and convert all values to (t - backbone_start)/scale."""
    if "backbone_start" not in bb:
        raise KeyError("backbone_start missing in backbone dict")

    t0 = bb["backbone_start"]
    merged = {}
    merged.update(bb)
    merged.update(sd)

    # normalize to ms since backbone_start
    return {k: (float(v) - float(t0)) / scale for k, v in merged.items()}

def merge_normalize_all(
    backbone_time_breakdown: List[Dict[str, int]],
    side_time_breakdown: List[Dict[str, int]],
    *,
    scale: float = MS_SCALE,
) -> List[Dict[str, float]]:
    """Element-wise merge + normalize for all iterations."""
    if len(backbone_time_breakdown) != len(side_time_breakdown):
        raise ValueError(f"length mismatch: backbone={len(backbone_time_breakdown)} side={len(side_time_breakdown)}")
    out: List[Dict[str, float]] = []
    for bb, sd in zip(backbone_time_breakdown, side_time_breakdown):
        out.append(_merge_and_normalize_once(bb, sd, scale=scale))
    return out

def average_time_breakdown(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average values across iterations, key-by-key.
    If a key is missing in some iterations, it’s averaged over the ones where it’s present.
    """
    # collect values per key
    buckets: Dict[str, List[float]] = {}
    for d in dicts:
        for k, v in d.items():
            buckets.setdefault(k, []).append(v)
    # mean per key
    return {k: mean(vals) for k, vals in buckets.items() if vals}

def make_time_hash() -> str:
    """
    Hash only the current time (ns) → hex string A.
    """
    t = str(time.time_ns()).encode("utf-8")
    return hashlib.sha1(t).hexdigest()  # or .hexdigest()[:16] if you prefer shorter

def load_time_breakdown_list_by_hash(
    A: str,
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """
    Read LIST[DICT] from results/A.
    """
    path = pathlib.Path(results_dir) / A
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list, found {type(data).__name__}")
    return data

from utils.tensor_transfer_channel import send_value, recv_value

def measure_inference_latency(net, device, test_num=5, channels_last=False, gpu='0', **kwargs):
    if kwargs.get('init_latency_test', False): set_cores(device, gpu=gpu, flex=False, same=False, threads=1)
    warmup = 5
    lst_config = kwargs.get('lst_config')
    trans = kwargs.get('transfer_type', 'shm')
    random_inputs = net.get_dummy_inputs(device='cuda:0', channels_last=channels_last)

    if device == 'single':
        if hasattr(net, 'forward_gpu_cpu'):
            net.forward = net.forward_gpu_cpu
        device_str = device
        device_cpu = torch.device('cpu')
        device_gpu = torch.device('cuda:0')
        net = net.to(device_gpu)
        if hasattr(net, 'side_device_init'):
            net.side_device_init(device_cpu)

        time_breakdown_lst = []
        latencies = []
        net.eval()
        for i in range(warmup + test_num):
            with torch.no_grad():
                # Measure time for one forward pass
                verbose = False
                torch.cuda.synchronize()
                # random_inputs = random_inputs_lst[i]
                output, time_breakdown = net(random_inputs, verbose=verbose, return_timeline=True) # if device not in split_device else net.forward_split(random_inputs)
                torch.cuda.synchronize()
                end_time = time.time_ns()
                if i == 10 and False:print(output.reshape(-1, )[:5])

                latencies.append((end_time - time_breakdown['backbone_start'])/1e6)
                time_breakdown_lst.append(time_breakdown)
                # print(f'iter {i} latency: {latencies[-1]}')

        avg = statistics.mean(latencies[warmup:])
        std = statistics.stdev(latencies[warmup:])
        # print(f'[sip] time breakdown for cfg {lst_config["all"]}: {time_breakdown_lst}', flush=True)
        print(f'Average latency per batch over {test_num} runs: {avg:.2f} ms ± {std:.2f} ms')

        return {device_str: avg}
    elif device == 'split' and trans == 'shm':
        # 1) convert net and pre-allocate shm BEFORE notifying tee_server,
        #    so backbone is ready to write tensors the moment tee_server starts polling.
        dup = net.module if hasattr(net, 'module') else net
        dup.convert_to_backbone_model('cuda:0')
        dup.transfer_type = trans

        shm_tensors = {}
        # NPLO models with plan-aware Round-2 ownership expose `_owned_shm_keys` —
        # GPU may own '_out' keys for trivial-but-consumed (fwd_only) blocks where
        # the side op runs on GPU and ships the result to TEE. Fall back to the
        # legacy "GPU owns non-'_out'" rule for LST/LoRA and unpruned NPLO paths
        # that don't set the attribute.
        gpu_owned = getattr(dup, '_owned_shm_keys', None)
        for shm_name, shape in dup.transfer_shapes.items():
            if gpu_owned is not None:
                if shm_name not in gpu_owned:
                    continue
            elif '_out' in shm_name:
                continue
            t, _ = alloc_shm_tensor(shm_name, shape, dtype=getattr(dup, 'compute_dtype', torch.float32), pin=True)
            shm_tensors[shm_name] = t
        print(f'allocate shm tensor {shm_tensors.keys()}')
        dup.eval()

        # 2) send control — tee_server will start polling for 'x' immediately after
        # batch_size is appended as a 5th element when present so the side process can
        # apply the same dummy-input patch and pre-allocate matching SHM regions.
        # Backwards-compatible: tee_server treats a missing 5th element as "leave default (100)".
        iter_msg = [warmup, test_num, net.model_name, net.num_classes]
        bs = kwargs.get('batch_size')
        if bs is not None:
            iter_msg.append(int(bs))
        send_value('iter', iter_msg, 'list', method=trans)
        send_value('cfg', lst_config['all'], 'list', method=trans)

        latencies = []
        backbone_time_breakdown = []
        for i in range(warmup + test_num):
            with torch.no_grad():
                # lat_writer.write_start(_)
                # random_inputs = net.get_dummy_inputs(device='cuda:0', channels_last=channels_last)
                torch.cuda.synchronize()
                verbose = False
                # random_inputs = random_inputs_lst[i]
                backbone_timeline = dup(shm_tensors, i, verbose, inputs=random_inputs) # , inputs=random_inputs
                torch.cuda.synchronize()
                backbone_time_breakdown.append(backbone_timeline)
                recv_end_time = recv_value(f'sync', 'list')[0]
                # ack(f'sync')
                latencies.append((recv_end_time - backbone_timeline['backbone_start'])/1e6)
                # print(f'iter {i} latency: {latencies[-1]}')
            torch.cuda.synchronize()

        formal = latencies[warmup:]  # 50 values
        sorted_formal = sorted(formal)
        k = int(len(formal) * 0.10)  # 10% = 5
        k = k if k != 0 else 1
        trimmed = sorted_formal[k:-k]
        avg_ms = statistics.mean(trimmed)
        std_ms = statistics.stdev(trimmed)

        # Drop torch tensors that view the shm regions BEFORE close_shm()
        # unmaps them — otherwise tensors point into munmap'd memory. Also
        # release the cuda host-pinned pages here (close_shm calls
        # _host_unregister) before the caller constructs the next config's
        # super_net, so the pinning peak doesn't overlap with the new build.
        shm_tensors.clear()
        del shm_tensors

        close_shm('iter', unlink=False)
        close_shm('cfg', unlink=False)

        for shm_name, shape in dup.transfer_shapes.items():
            if gpu_owned is not None:
                if shm_name not in gpu_owned:
                    continue
            elif '_out' in shm_name:
                continue
            close_shm(shm_name, unlink=True)
        # close_shm('time_breakdown', unlink=True)
        # close_shm('sync', unlink=True)

        gc.collect()
        torch.cuda.empty_cache()

        print(f'[backbone]: final latency: {avg_ms:.2f} ms ± {std_ms:.2f} ms')
        return {'split': float(avg_ms), 'time_breakdown': backbone_time_breakdown}
    elif device in ['gpu', 'cpu']:
        print(f'test latency in device {device}')
        net = net.to("cuda:0") if device == 'gpu' else net.to("cpu")
        random_inputs = random_inputs.to("cuda:0") if device == 'gpu' else random_inputs.to("cpu")
        time_breakdown_lst = []
        latencies = []
        net.eval()
        for i in range(warmup + test_num):
            with torch.no_grad():
                # Measure time for one forward pass
                verbose = False #if (warmup <= i <= warmup + 2) else False
                torch.cuda.synchronize()
                # random_inputs = random_inputs_lst[i]
                output, time_breakdown = net(random_inputs, verbose=verbose, return_timeline=True, sync=False) # if device not in split_device else net.forward_split(random_inputs)
                torch.cuda.synchronize()
                end_time = time.time_ns()
                if i == 10:
                    print(output.reshape(-1, )[:5])

                latencies.append((end_time - time_breakdown['backbone_start'])/1e6)
                time_breakdown_lst.append(time_breakdown)
                # print(f'iter {i} latency: {latencies[-1]}')

        avg = statistics.mean(latencies[warmup:])
        std = statistics.stdev(latencies[warmup:])
        # print(f'[sip] time breakdown for cfg {lst_config["all"]}: {time_breakdown_lst}', flush=True)
        print(f'Average latency per batch over {test_num} runs: {avg:.2f} ms ± {std:.2f} ms')
        return {device: avg}
    elif device == 'special':
        print(f'test latency in device {device}')
        # import timm
        # net = timm.create_model("vit_base_patch16_224", pretrained=False, embed_dim=768)
        # import torchvision
        # net = torchvision.models.vit_b_16()
        net = net.to("cpu")
        net.lora_layer = []
        random_inputs = random_inputs.to("cpu")
        time_breakdown_lst = []
        latencies = []
        net.eval()
        time_breakdown = {}
        for i in range(warmup + test_num):
            with torch.no_grad():
                # Measure time for one forward pass
                verbose = False #if (warmup <= i <= warmup + 2) else False
                torch.cuda.synchronize()
                # random_inputs = random_inputs_lst[i]
                output, time_breakdown = net(random_inputs, verbose=verbose, return_timeline=True, sync=False) # if device not in split_device else net.forward_split(random_inputs)
                # time_breakdown['backbone_start'] = time.time_ns()
                # output = net(random_inputs)
                torch.cuda.synchronize()
                end_time = time.time_ns()
                if i == 10:
                    print(output.reshape(-1, )[:5])

                latencies.append((end_time - time_breakdown['backbone_start'])/1e6)
                time_breakdown_lst.append(time_breakdown)
                # print(f'iter {i} latency: {latencies[-1]}')

        avg = statistics.mean(latencies[warmup:])
        std = statistics.stdev(latencies[warmup:])
        # print(f'[sip] time breakdown for cfg {lst_config["all"]}: {time_breakdown_lst}', flush=True)
        print(f'Average latency per batch over {test_num} runs: {avg:.2f} ms ± {std:.2f} ms')
        return {device: avg}
    else: raise NotImplementedError

