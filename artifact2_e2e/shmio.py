# shmio.py
import numpy as np
import torch
from typing import Any, Optional, Tuple
import os, time, json, struct, ctypes, hashlib
try:
    import cupy as cp
except:
    print(f'cupy is not available')
    cp = None

_shm_registry = {}
_here = os.path.dirname(__file__)

try:
    _lib = ctypes.CDLL(os.path.join(_here, "shm_bridge.so"))
    _lib.shm_owner_create_named.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    _lib.shm_owner_create_named.restype = ctypes.c_int
    _lib.shm_client_open_named.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    _lib.shm_client_open_named.restype = ctypes.c_int
    _lib.shm_get_write_ptr_named.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)]
    _lib.shm_get_write_ptr_named.restype = ctypes.POINTER(ctypes.c_float)
    _lib.shm_publish_done_named.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    _lib.shm_publish_done_named.restype = ctypes.c_int
    _lib.shm_wait_ready_get_ptr_named.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _lib.shm_wait_ready_get_ptr_named.restype = ctypes.POINTER(ctypes.c_float)
    _lib.shm_size_named.argtypes = [ctypes.c_char_p]
    _lib.shm_size_named.restype = ctypes.c_uint
    _lib.shm_capacity_named.argtypes = [ctypes.c_char_p]
    _lib.shm_capacity_named.restype = ctypes.c_uint
    _lib.shm_ack_named.argtypes = [ctypes.c_char_p]
    _lib.shm_ack_named.restype = None
    _lib.shm_close_all.argtypes = []
    _lib.shm_close_all.restype = None
    _lib.shm_close_named.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _lib.shm_close_named.restype = ctypes.c_int
    _lib.shm_unlink_named.argtypes = [ctypes.c_char_p]
    _lib.shm_unlink_named.restype = ctypes.c_int
except:
    _lib = None
    print(f"shm_bridge.so is not available")

def _encode(value: Any, typ: str) -> bytes:
    if typ == "string":
        return value.encode("utf-8")
    if typ == "list":
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if typ == "int":
        return struct.pack("<q", int(value))
    if typ == "float":
        return struct.pack("<d", float(value))
    raise ValueError(f"unsupported typ: {typ}")

def _decode(buf: bytes, typ: str) -> Any:
    if typ == "string":
        return buf.decode("utf-8")
    if typ == "list":
        return json.loads(buf.decode("utf-8")) if buf else []
    if typ == "int":
        return struct.unpack("<q", buf[:8])[0] if len(buf) >= 8 else 0
    if typ == "float":
        return struct.unpack("<d", buf[:8])[0] if len(buf) >= 8 else 0.0
    raise ValueError(f"unsupported typ: {typ}")

def _to_shm_name(path_or_name: str) -> bytes:
    name = path_or_name
    if name.startswith("/dev/shm/"):
        name = name[len("/dev/shm/"):]
    if not name.startswith("/"):
        name = "/" + name
    return name.encode("utf-8")

def tensor_hash(t: torch.Tensor, algo: str = "sha256", include_meta: bool = True) -> str:
    import hashlib
    if t.requires_grad:
        t = t.detach()

    # Canonicalize: make standard contiguous layout on CPU
    t = t.to(memory_format=torch.contiguous_format).contiguous().cpu()

    if t.dtype == torch.bfloat16:
        arr = t.view(torch.uint16).numpy()
        meta_dtype = "bfloat16"
    else:
        arr = t.numpy()
        meta_dtype = str(arr.dtype)

    h = hashlib.new(algo)
    if include_meta:
        h.update(str(tuple(arr.shape)).encode())
        h.update(meta_dtype.encode())
    h.update(arr.tobytes(order="C"))  # now invariant to layout/strides
    return h.hexdigest()

def tensor_hash2(t: torch.Tensor):
    return t.reshape(-1)

_owner_opened = {}
_client_opened = {}

def _ensure_owner(name_b: bytes, capacity_floats: int):
    key = name_b.decode('utf-8')
    if key in _owner_opened:
        return
    rc = _lib.shm_owner_create_named(name_b, ctypes.c_uint(capacity_floats))
    if rc != 0:
        raise RuntimeError(f"shm_owner_create_named failed: {rc}")
    _owner_opened[key] = capacity_floats

def _ensure_client(name_b: bytes, timeout_ms: int):
    key = name_b.decode('utf-8')
    if key in _client_opened:
        return
    deadline = time.time() + (timeout_ms / 1000 if timeout_ms >= 0 else 365*24*3600)
    last_err = None
    while True:
        rc = _lib.shm_client_open_named(name_b, ctypes.c_uint(0))
        if rc == 0:
            _client_opened[key] = True
            return
        last_err = rc
        if timeout_ms >= 0 and time.time() > deadline:
            raise RuntimeError(f"shm_client_open_named timed out waiting for owner (last rc={last_err})")
        time.sleep(0.001)

def alloc_shm_tensor(path: str, shape, *, dtype=torch.float32, pin=False):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"shm bridge supports float32 and bfloat16 only, got {dtype}")
    if '/' not in path:
        path = '/dev/shm/' + path

    name_b = _to_shm_name(path)
    numel = int(np.prod(shape, dtype=np.int64))
    # bf16 is 2 bytes/elem; 1 float unit = 4 bytes → ceil(numel/2) float units needed
    capacity_floats = (numel + 1) // 2 if dtype == torch.bfloat16 else numel
    t0 = time.time_ns()
    _ensure_owner(name_b, capacity_floats)
    t1 = time.time_ns()

    cap = ctypes.c_uint(0)
    ptr = _lib.shm_get_write_ptr_named(name_b, ctypes.byref(cap))
    t2 = time.time_ns()

    if not ptr:
        raise RuntimeError("shm_get_write_ptr_named returned null")
    if dtype == torch.bfloat16:
        if cap.value * 2 < numel:
            raise RuntimeError(f"capacity too small: {cap.value * 2} bf16 slots < {numel}")
    else:
        if cap.value < numel:
            raise RuntimeError(f"capacity too small: {cap.value} < {numel}")
    t3 = time.time_ns()
    addr = ctypes.addressof(ptr.contents)

    if dtype == torch.bfloat16:
        buf = (ctypes.c_uint16 * numel).from_address(addr)
        np_view = np.ctypeslib.as_array(buf)
        feat_cpu = torch.from_numpy(np_view).view(torch.bfloat16).view(tuple(shape))
        nbytes = capacity_floats * 4
    else:
        buf = (ctypes.c_float * numel).from_address(addr)
        np_view = np.ctypeslib.as_array(buf)
        feat_cpu = torch.from_numpy(np_view).view(tuple(shape))
        nbytes = numel * np.dtype(np.float32).itemsize
    t4 = time.time_ns()
    t5 = time.time_ns()

    s_prev = _shm_registry.get(path)
    was_pinned = bool(s_prev and s_prev.get("pinned"))

    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"
    _shm_registry[path] = dict(numel=numel, addr=addr, nbytes=nbytes,
                               pinned=was_pinned, dtype=dtype_str,
                               capacity_floats=capacity_floats, name_b=name_b)
    t6 = time.time_ns()

    if pin and not was_pinned:
        _host_register(addr, nbytes)
        _shm_registry[path]["pinned"] = True
        # print(f'pinned to {path}')

    t7 = time.time_ns()

    # print(f'alloc shm: {path}, latency: {(t7 - t0) / 1e6:.2f} ms, latency: {(t1 - t0) / 1e6:.2f} ms, '
    #       f'latency: {(t2 - t1) / 1e6:.2f} ms, latency: {(t3 - t2) / 1e6:.2f} ms, '
    #       f'latency: {(t4 - t3) / 1e6:.2f} ms, latency: {(t5 - t4) / 1e6:.2f} ms, '
    #       f'latency: {(t6 - t5) / 1e6:.2f} ms, latency: {(t7 - t6) / 1e6:.2f} ms, shape: {shape}')

    if pin:
        return feat_cpu, (addr, nbytes)
    else:
        return feat_cpu, (0, 0)

def _maybe_pin_region(path: str, addr: int, nbytes: int):
    if '/' not in path:
        path = '/dev/shm/' + path
    # reuse your existing registry
    s = _shm_registry.get(path)
    if s and s.get("pinned"):
        return
    _host_register(addr, nbytes)
    _shm_registry[path] = dict(addr=addr, nbytes=nbytes, pinned=True)

def pin_shm(path: str):
    s = _shm_registry.get(path)
    if not s:
        raise RuntimeError("pin_shm called before alloc_shm_tensor")
    if not s["pinned"]:
        _host_register(s["addr"], s["nbytes"])
        s["pinned"] = True
    return (s["addr"], s["nbytes"])

def unpin_shm(path: str):
    s = _shm_registry.get(path)
    if s and s["pinned"]:
        _host_unregister(s["addr"])
        s["pinned"] = False

def _host_register(addr: int, nbytes: int):
    if cp is None:
        raise RuntimeError("CuPy is required for pin=True or pin_shm(). Install cupy in this env.")
    try:
        cp.cuda.runtime.hostRegister(addr, nbytes, 0)
    except Exception as e:
        if "cudaErrorHostMemoryAlreadyRegistered" not in str(e):
            raise

def _host_unregister(addr: int):
    if cp is None:
        return
    try:
        cp.cuda.runtime.hostUnregister(addr)
    except Exception as e:
        if "cudaErrorHostMemoryNotRegistered" not in str(e):
            pass

def send_tensor_zc(path: str):
    s = _shm_registry.get(path)
    if not s:
        raise RuntimeError("send_tensor_zc requires alloc_shm_tensor to have been called first")
    rc = _lib.shm_publish_done_named(s["name_b"], ctypes.c_uint(s["capacity_floats"]))
    if rc != 0:
        raise RuntimeError(f"shm_publish_done_named failed: {rc}")
    # os.sched_yield()

def send_tensor(tensor: torch.Tensor, path: str, shape):
    name_b = _to_shm_name(path)
    ten = tensor.contiguous().to(torch.float32)
    numel = ten.numel()

    _ensure_owner(name_b, numel)

    cap = ctypes.c_uint(0)
    ptr = _lib.shm_get_write_ptr_named(name_b, ctypes.byref(cap))
    if not ptr:
        raise RuntimeError("shm_get_write_ptr_named returned null")
    if cap.value < numel:
        raise RuntimeError(f"capacity too small: {cap.value} < {numel}")

    addr = ctypes.addressof(ptr.contents)
    buf = (ctypes.c_float * numel).from_address(addr)
    np_view = np.ctypeslib.as_array(buf)
    np_view[:] = ten.view(-1).cpu().numpy()

    rc = _lib.shm_publish_done_named(name_b, ctypes.c_uint(numel))
    if rc != 0:
        raise RuntimeError(f"shm_publish_done_named failed: {rc}")

def send_eos(path: str):
    name_b = _to_shm_name(path)
    key = name_b.decode('utf-8')
    if key not in _owner_opened:
        _ensure_owner(name_b, 0)
    _lib.shm_get_write_ptr_named(name_b, None)
    rc = _lib.shm_publish_done_named(name_b, 0)
    if rc != 0:
        raise RuntimeError(f"shm_publish_done_named(0) failed: {rc}")

def recv_tensor(path: str, shape, timeout_ms: int = 3000000):
    name_b = _to_shm_name(path)
    _ensure_client(name_b, timeout_ms)

    ptr = _lib.shm_wait_ready_get_ptr_named(name_b, ctypes.c_int(timeout_ms))
    if not ptr:
        raise RuntimeError("timeout waiting for data")

    start_ts = time.time_ns()
    n = _lib.shm_size_named(name_b)
    if n == 0:
        _lib.shm_ack_named(name_b)
        return None  # EOS

    addr = ctypes.addressof(ptr.contents)
    buf = (ctypes.c_float * n).from_address(addr)
    np_view = np.ctypeslib.as_array(buf)
    ten = torch.from_numpy(np_view).view(shape)
    end_ts = time.time_ns()
    print(f"[shmio   ] recv_tensor time: {(end_ts - start_ts) / 1e6:.2f} ms, output shape: {ten.shape}")
    return ten

def ack(path: str = None):
    if '/' not in path:
        path = '/dev/shm/' + path
    if path:
        name_b = _to_shm_name(path)
        _lib.shm_ack_named(name_b)
    else:
        for key in _client_opened.keys():
            _lib.shm_ack_named(key.encode('utf-8'))

def close_shm(path: str, *, unlink: bool = False):
    """Unmap+close this mapping from the current process. If unlink=True, also shm_unlink(name)."""
    if '/' not in path:
        path = '/dev/shm/' + path

    name_b = _to_shm_name(path)
    key = name_b.decode('utf-8')

    s = _shm_registry.get(key) or _shm_registry.get(path)
    if s and s.get("pinned"):
        try:
            _host_unregister(s["addr"])
        except Exception:
            pass
        s["pinned"] = False

    # Close (and optionally unlink)
    _lib.shm_close_named(name_b, 1 if unlink else 0)

    _shm_registry.pop(key, None)
    _owner_opened.pop(key, None)
    _client_opened.pop(key, None)

def close_all():
    global _owner_opened, _client_opened
    for k, s in list(_shm_registry.items()):
        if s.get("pinned"):
            try:
                cp.cuda.runtime.hostUnregister(s["addr"])
            except Exception:
                pass
            s["pinned"] = False
    _owner_opened.clear()
    _client_opened.clear()
    _lib.shm_close_all()

def send_value_shm(path: str, value: Any, typ: str, **kwargs):
    """Owner writes DATA-ONLY payload starting at offset 0; publish size in floats."""
    if typ == "tensor": #GPU DMA used to transfer tensor
        send_tensor_shm(path=path, src=value, **kwargs)
    else:
        send_non_tensor_shm(path=path, value=value, typ=typ, **kwargs)

def recv_value_shm(path: str, typ: str, timeout_ms: int=3000000, **kwargs):
    """Client reads size_f floats -> size_f*4 bytes; strip padding zeros; decode."""
    if typ == "tensor":
        return recv_tensor_shm(path=path, timeout_ms=timeout_ms, **kwargs)
    else:
        return recv_non_tensor_shm(path=path, typ=typ, timeout_ms=timeout_ms, **kwargs)

# the most important part for zero copy transfer
def send_tensor_shm(path: str, src: Any, **kwargs):
    direction = kwargs.get("direction", "d2h")
    direction = (direction or "").lower()
    # t0 = time.time_ns()
    if not isinstance(src, torch.Tensor):
        if direction == "d2h":
            raise TypeError("send_value_shm(type='tensor') expects a torch.Tensor")
        else:
            src = kwargs.get('shared_memory', None)

    if kwargs.get('check_value', '0') == '2':
        # expects a helper tensor_hash(...) in your file
        print(f"send tensor {path}: value: {tensor_hash(src)[:5]}, {src.reshape(-1, )[:1].item():.2f}, {src.reshape(-1, )[-1:].item():.2f},"
              # f" {src[0, 2, 1].item():.2f}, {src[0, 1, 1].item():.2f}, channels_last: {src.is_contiguous(memory_format=torch.channels_last)}"
              , flush=True)
    elif kwargs.get('check_value', '0') == '1':
        print(f"send tensor {path}: shape: {src.shape}, \tvalue: {src.reshape(-1, )[:1].item():.2f}, {src.reshape(-1, )[-1:].item():.2f},"
              # f" {src[0, 2, 1, 1].item():.2f}, {src[0, 1, 1, 1].item():.2f}, "
              , flush=True)

    # t, (addr, nbytes) = alloc_shm_tensor(path, tuple(src.shape), pin=True)

    if direction not in ("d2h", "h2d"):
        raise ValueError("direction must be either 'd2h' or 'h2d'")

    if direction == "h2d":
        t = kwargs.get('shared_memory', None)
        if t is None:
            raise RuntimeError("send_tensor_shm(direction='h2d') requires shared_memory=<cpu_shm_tensor>")
        if src.ndim == 4 and src.is_contiguous(memory_format=torch.channels_last):
            N, C, H, W = src.shape
            cl_strides = (H * W * C, 1, W * C, C)  # in elements
            t = torch.as_strided(t, size=(N, C, H, W), stride=cl_strides)

        t.copy_(src, non_blocking=True) # both src and t is in CPU or TEE
        # torch.cuda.synchronize()
        send_tensor_zc(path)
        return

    if direction == "d2h":
        t = kwargs.get('shared_memory', None)
        if t is None:
            raise RuntimeError("send_tensor_shm(direction='d2h') requires shared_memory=<cpu_shm_tensor>")

        if src.ndim == 4 and src.is_contiguous(memory_format=torch.channels_last):
            N, C, H, W = src.shape
            cl_strides = (H * W * C, 1, W * C, C)  # in elements
            t = torch.as_strided(t, size=(N, C, H, W), stride=cl_strides)

        t.copy_(src, non_blocking=True) #DMA
        torch.cuda.synchronize()
        send_tensor_zc(path)
        return

def recv_tensor_shm(path: str, timeout_ms: int=3000000, **kwargs):

    # start_ts = time.time_ns()
    direction = kwargs.get("direction", "d2h")
    direction = (direction or "").lower()
    if direction not in ("d2h", "h2d"):
        raise ValueError("direction must be either 'd2h' or 'h2d'")

    name_b = _to_shm_name(path)
    _ensure_client(name_b, timeout_ms)
    shape = kwargs.get('shape', None)

    t0_ptr = time.time_ns()
    ptr = _lib.shm_wait_ready_get_ptr_named(name_b, ctypes.c_int(timeout_ms))
    if not ptr:
        raise RuntimeError("timeout waiting for data")

    start_ts = time.time_ns()
    n = _lib.shm_size_named(name_b)
    if n == 0:
        _lib.shm_ack_named(name_b)
        return None  # EOS

    addr = ctypes.addressof(ptr.contents)
    channels_last = bool(kwargs.get('channels_last', False))
    dtype = kwargs.get('dtype', torch.float32)

    if direction == "h2d":
        _host_register(addr, n * 4)

    if dtype == torch.bfloat16:
        numel = int(np.prod(shape))
        buf = (ctypes.c_uint16 * numel).from_address(addr)
        np_view = np.ctypeslib.as_array(buf)
        ten = torch.from_numpy(np_view).view(torch.bfloat16).view(tuple(shape))
    elif channels_last and len(shape) == 4:
        buf = (ctypes.c_float * n).from_address(addr)
        # Zero-copy channels_last reinterpretation.
        # PyTorch channels_last (NCHW dims) uses strides (in elements):
        #   (H*W*C, 1, W*C, C)
        N, C, H, W = shape
        total = N * C * H * W
        if total != n:
            raise ValueError(f"Element count mismatch: SHM has {n}, but shape {shape} needs {total}")

        base = np.ctypeslib.as_array(buf)  # 1D float32 over the SHM
        itemsize = base.dtype.itemsize  # 4 bytes for float32
        # Convert element strides → byte strides for NumPy
        strides_elems = (H * W * C, 1, W * C, C)
        strides_bytes = tuple(s * itemsize for s in strides_elems)

        # Create a strided NumPy view with NCHW dims, channels_last layout (zero-copy)
        np_view = np.lib.stride_tricks.as_strided(
            base, shape=(N, C, H, W), strides=strides_bytes
        )
        ten = torch.from_numpy(np_view)
    else:
        buf = (ctypes.c_float * n).from_address(addr)
        # Original path: plain C-order interpretation
        np_view = np.ctypeslib.as_array(buf)
        ten = torch.from_numpy(np_view).view(shape)

    end_ts = time.time_ns()

    if direction == "d2h":
        out = ten
    elif direction == "h2d":
        # Allocate dst on GPU on-the-fly
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", torch.float32)  # SHM is float32; you may override to cast
        if channels_last and len(shape) == 4:
            dst = torch.empty(tuple(shape), device=device, dtype=dtype, memory_format=torch.channels_last)
        else:
            dst = torch.empty(tuple(shape), device=device, dtype=dtype)

        if not dst.is_cuda:
            raise ValueError("dst must be a CUDA tensor for direction='h2d'")

        dst.copy_(ten, non_blocking=True)
        torch.cuda.synchronize()
        out = dst
    else: raise ValueError("direction must be either 'd2h' or 'h2d'")

    if kwargs.get('check_value', '0') == '2':
        # expects a helper tensor_hash(...) in your file
        print(f"recv tensor {path}: value: {tensor_hash(ten)[:5]}, {ten.reshape(-1, )[:1].item():.2f}, {ten.reshape(-1, )[-1:].item():.2f},"
              # f" {ten[0, 2, 1].item():.2f}, {ten[0, 1, 1].item():.2f}, "
              f"contiguous tensor: {ten.is_contiguous(memory_format=torch.channels_last if channels_last else torch.contiguous_format)}"
              , flush=True)
        # print(f"[shmio   ] recv_tensor time: {(end_ts - start_ts) / 1e6:.2f} ms, ptr time:  {(start_ts - t0_ptr) / 1e6:.2f} ms, output shape: {ten.shape}")
    elif kwargs.get('check_value', '0') == '1':
        print(f"recv tensor {path}: shape: {out.shape}, \tvalue: {out.reshape(-1, )[:1].item():.2f}, {out.reshape(-1, )[-1:].item():.2f},"
              # f" {ten[0, 2, 1].item():.2f}, {ten[0, 1, 1].item():.2f}, "
              , flush=True)

        # print(f"[shmio   ] recv_tensor {path} time: {(end_ts - start_ts) / 1e6:.2f} ms,  {(start_ts - t0_ptr) / 1e6:.2f} ms, output shape: {ten.shape}")

    return out

def send_value_shm_fake(path: str, value: Any, typ: str, **kwargs):
    """No-transfer-latency variant of send_value_shm.

    Maintains the shm publish handshake but skips the actual D->H/H->H data copy,
    so receiver wakes up but sees no real data. Used to measure the contribution
    of data transfer to end-to-end latency (compare A vs A_hat). Non-tensor types
    are tiny and pass through unchanged.
    """
    if typ == "tensor":
        send_tensor_shm_fake(path=path, src=value, **kwargs)
    else:
        send_non_tensor_shm(path=path, value=value, typ=typ, **kwargs)

def recv_value_shm_fake(path: str, typ: str, timeout_ms: int=3000000, **kwargs):
    """No-transfer-latency variant of recv_value_shm.

    Waits for the publish handshake but skips data extraction (and the H->D copy
    on the GPU side); returns a zero tensor of the expected shape.
    """
    if typ == "tensor":
        return recv_tensor_shm_fake(path=path, timeout_ms=timeout_ms, **kwargs)
    else:
        return recv_non_tensor_shm(path=path, typ=typ, timeout_ms=timeout_ms, **kwargs)

def send_tensor_shm_fake(path: str, src: Any, **kwargs):
    """Fake tensor send: only the publish step happens; no data is copied."""
    name_b = _to_shm_name(path)
    s = _shm_registry.get(path)
    if s is not None:
        capacity_floats = s.get("capacity_floats", 1)
    else:
        capacity_floats = 1
    rc = _lib.shm_publish_done_named(name_b, ctypes.c_uint(int(capacity_floats)))
    if rc != 0:
        raise RuntimeError(f"shm_publish_done_named (fake) failed: {rc}")
    return

_fake_h2d_cache = {}

def recv_tensor_shm_fake(path: str, timeout_ms: int=3000000, **kwargs):
    """Fake tensor recv: wait for publish but skip the actual data movement.

    d2h (TEE/CPU receives): build a zero-copy torch view over the shm region —
    same construction as the real recv path, just without any DMA having
    happened. Content is stale/zeros but for latency measurement that's fine.
    h2d (GPU receives): return a cached GPU buffer of the requested
    (shape, dtype, channels_last). Avoids per-call torch.zeros + layout-change
    copies that otherwise dominate the fake-recv cost.
    """
    direction = (kwargs.get("direction", "d2h") or "").lower()
    if direction not in ("d2h", "h2d"):
        raise ValueError("direction must be either 'd2h' or 'h2d'")

    name_b = _to_shm_name(path)
    _ensure_client(name_b, timeout_ms)
    shape = kwargs.get('shape', None)
    chlast = bool(kwargs.get('channels_last', False))
    dtype = kwargs.get('dtype', torch.float32)

    ptr = _lib.shm_wait_ready_get_ptr_named(name_b, ctypes.c_int(timeout_ms))
    if not ptr:
        raise RuntimeError("timeout waiting for data")
    n = _lib.shm_size_named(name_b)
    if n == 0:
        _lib.shm_ack_named(name_b)
        return None  # EOS

    if direction == "d2h":
        addr = ctypes.addressof(ptr.contents)
        if dtype == torch.bfloat16:
            numel = int(np.prod(shape))
            buf = (ctypes.c_uint16 * numel).from_address(addr)
            np_view = np.ctypeslib.as_array(buf)
            return torch.from_numpy(np_view).view(torch.bfloat16).view(tuple(shape))
        if chlast and len(shape) == 4:
            buf = (ctypes.c_float * n).from_address(addr)
            N, C, H, W = shape
            base = np.ctypeslib.as_array(buf)
            itemsize = base.dtype.itemsize
            strides_elems = (H * W * C, 1, W * C, C)
            strides_bytes = tuple(s * itemsize for s in strides_elems)
            np_view = np.lib.stride_tricks.as_strided(
                base, shape=(N, C, H, W), strides=strides_bytes
            )
            return torch.from_numpy(np_view)
        buf = (ctypes.c_float * n).from_address(addr)
        np_view = np.ctypeslib.as_array(buf)
        return torch.from_numpy(np_view).view(shape)

    device = kwargs.get("device", "cuda")
    key = (path, tuple(shape), dtype, chlast, str(device))
    dst = _fake_h2d_cache.get(key)
    if dst is None:
        if chlast and len(shape) == 4:
            dst = torch.empty(tuple(shape), device=device, dtype=dtype,
                              memory_format=torch.channels_last)
        else:
            dst = torch.empty(tuple(shape), device=device, dtype=dtype)
        _fake_h2d_cache[key] = dst
    return dst

def send_non_tensor_shm(path: str, value: Any, typ: str, **kwargs):
    payload = _encode(value, typ)
    nbytes = len(payload)
    n_floats = (nbytes + 3) // 4 # (nbytes) // 4 + 1#
    total_bytes = n_floats * 4

    name_b = _to_shm_name(path)
    _ensure_owner(name_b, max(1, n_floats))

    cap = ctypes.c_uint(0)
    wptr = _lib.shm_get_write_ptr_named(name_b, ctypes.byref(cap))
    if not wptr:
        raise RuntimeError("shm_get_write_ptr_named returned NULL")
    if cap.value < n_floats:
        raise RuntimeError(f"capacity too small: need {n_floats}, have {cap.value}")

    addr = ctypes.addressof(wptr.contents)
    ctypes.memset(addr, 0, total_bytes)
    if nbytes:
        ctypes.memmove(addr, payload, nbytes)

    rc = _lib.shm_publish_done_named(name_b, ctypes.c_uint(n_floats))
    if rc != 0:
        raise RuntimeError(f"shm_publish_done_named failed: rc={rc}")
    if kwargs.get('wait_ack', False):
        _lib.shm_ack_named(name_b)

def recv_non_tensor_shm(path: str, typ: str, *, timeout_ms: int=3000000, **kwargs):

    name_b = _to_shm_name(path)
    _ensure_client(name_b, timeout_ms)
    ptr = _lib.shm_wait_ready_get_ptr_named(name_b, ctypes.c_int(timeout_ms))
    if not ptr:
        raise RuntimeError("timeout waiting for data")
    n_floats = int(_lib.shm_size_named(name_b))
    addr = ctypes.addressof(ptr.contents)
    raw = ctypes.string_at(addr, n_floats * 4)
    raw = raw.rstrip(b"\x00")
    val = _decode(raw, typ) if n_floats > 0 else ("" if typ == "string" else ([] if typ == "list" else 0))
    _lib.shm_ack_named(name_b)
    return val


