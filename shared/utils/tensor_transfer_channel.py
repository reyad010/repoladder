
# utils/tensor_transfer_channel.py
import os, struct, time, json
from typing import Any, Optional, Tuple, List
import numpy as np
import torch, hashlib
try:
    import shmio as _shm
except Exception as e:
    _shm = None

# ----- hardcoded shared folder & wire layout -----
share_root = './file_share'
FLAG_OFF, SIZE_OFF, TIMELINE_OFF, DATA_OFF = 0, 1, 9, 17
CAP_BYTES = 256 * 1024 * 1024  # 256MB per channel file

# (kept here because some callers use it)
num_class_map = {
    'cifar10': 10, 'cifar100': 100, 'subcifar': 2, 'subgtsrb': 2, 'svhn': 10, 'pubfig': 83,
    'gtsrb': 43, 'eurosat': 10, 'flower': 102, 'imagenet': 1000, 'imagenet64': 1000,
    'mnist': 10, 'mnistm': 10, 'food': 101, 'pet': 37, 'resisc': 45, 'voc': 21, 'lingspam': 2,
}

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _path(name: str) -> str:
    _ensure_dir(share_root)
    return os.path.join(share_root, f"{name}.bin")

def _open_osp(name: str, size: int = CAP_BYTES) -> int:
    path = _path(name)
    existed = os.path.exists(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    st = os.fstat(fd)
    if (not existed) or (st.st_size != size):
        os.ftruncate(fd, size)
        os.pwrite(fd, b"\x00", FLAG_OFF)                      # flag=0 (idle)
        os.pwrite(fd, struct.pack("<Q", 0), SIZE_OFF)         # size=0
        os.pwrite(fd, struct.pack("<Q", 0), TIMELINE_OFF)     # timeline=0
        os.fsync(fd)
    return fd

# ---------- (de)serialization ----------
def _encode(value: Any, typ: str, *, channels_last: bool = False) -> bytes:
    typ = typ.lower()
    if typ == 'tensor':
        t: torch.Tensor = value
        src = t.detach().cpu()
        if channels_last:
            src = src.contiguous(memory_format=torch.channels_last)
        else:
            src = src.contiguous()
        return src.numpy().tobytes(order='C')
    if typ == 'list':
        return json.dumps(value).encode('utf-8')
    if typ == 'int':
        return struct.pack('<q', int(value))      # int64
    if typ == 'float':
        return struct.pack('<d', float(value))    # float64
    if typ == 'string':
        return (value if isinstance(value, str) else str(value)).encode('utf-8')
    raise ValueError(f"unsupported type for encode: {typ}")

def _decode(raw: bytes, typ: str, *, shape: Optional[Tuple[int, ...]] = None,
            channels_last: bool = False):
    typ = typ.lower()
    if typ == 'tensor':
        if shape is None:
            raise ValueError("recv_value(type='tensor') requires shape=...")
        arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
        t = torch.from_numpy(arr)
        if channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        return t
    if typ == 'list':
        return json.loads(raw.decode('utf-8'))
    if typ == 'int':
        return struct.unpack('<q', raw[:8])[0]
    if typ == 'float':
        return struct.unpack('<d', raw[:8])[0]
    if typ == 'string':
        return raw.decode('utf-8')
    raise ValueError(f"unsupported type for decode: {typ}")

# ---------- OSP primitives ----------
def send_value_osp(name: str, value: Any, typ: str,
                   *, timeline_ns: Optional[int] = None,
                   channels_last: bool = False,
                   wait_ack: bool = False) -> Tuple[int, int]:
    """Write one payload: [flag=1][size][timeline][data]. Returns (t_start_write_ns, t_published_ns)."""
    fd = _open_osp(name)
    try:
        # wait until idle (flag==0)
        while os.pread(fd, 1, FLAG_OFF) != b'\x00':
            time.sleep(0.0005)
        payload = _encode(value, typ, channels_last=channels_last)
        t0 = time.time_ns()
        os.pwrite(fd, struct.pack('<Q', len(payload)), SIZE_OFF)
        os.pwrite(fd, struct.pack('<Q', int(timeline_ns or t0)), TIMELINE_OFF)
        os.pwrite(fd, payload, DATA_OFF)
        os.fsync(fd)
        os.pwrite(fd, b'\x01', FLAG_OFF)  # publish
        os.fsync(fd)
        if wait_ack:
            while os.pread(fd, 1, FLAG_OFF) != b'\x00':
                time.sleep(0.0005)
        t1 = time.time_ns()
        return t0, t1
    finally:
        os.close(fd)

def recv_value_osp(name: str, typ: str, *,
                   shape: Optional[Tuple[int, ...]] = None,
                   channels_last: bool = False,
                   return_timeline: bool = False):
    """Read one payload. If return_timeline=True, returns (value, producer_start_ns, consumer_read_ns)."""
    fd = _open_osp(name)
    try:
        while os.pread(fd, 1, FLAG_OFF) != b'\x01':
            time.sleep(0.0005)
        sz = struct.unpack('<Q', os.pread(fd, 8, SIZE_OFF))[0]
        start_ns = struct.unpack('<Q', os.pread(fd, 8, TIMELINE_OFF))[0]
        raw = os.pread(fd, sz, DATA_OFF)
        os.pwrite(fd, b'\x00', FLAG_OFF)  # ack
        os.fsync(fd)
        val = _decode(raw, typ, shape=shape, channels_last=channels_last)
        if return_timeline:
            return val, int(start_ns), int(time.time_ns())
        return val
    finally:
        os.close(fd)

def send_value(name: str, value: Any, typ: str, *, method: str = 'shm', **kwargs):
    if method == 'osp':
        return send_value_osp(name, value, typ, **kwargs)
    elif method == 'shm':
        if '/' not in name:
            name = '/dev/shm/' + name
        return _shm.send_value_shm(name, value, typ, **kwargs)
    else:
        raise NotImplementedError(f"transfer method '{method}' is not supported")

def recv_value(name: str, typ: str, *, method: str = 'shm', **kwargs):
    if method == 'osp':
        return recv_value_osp(name, typ, **kwargs)
    elif method == 'shm':
        if '/' not in name:
            name = '/dev/shm/' + name
        return _shm.recv_value_shm(name, typ, **kwargs)
    else:
        raise NotImplementedError(f"transfer method '{method}' is not supported")


def send_value_no_transfer(name: str, value: Any, typ: str, *, method: str = 'shm', **kwargs):
    """Fake-transfer send: maintains shm handshake, skips the actual data copy.

    Used by *_no_transfer_latency forward paths to quantify the impact of data
    transfer on end-to-end latency (compare A vs A_hat).
    """
    if method != 'shm':
        raise NotImplementedError(f"no-transfer mode requires method='shm', got '{method}'")
    if '/' not in name:
        name = '/dev/shm/' + name
    return _shm.send_value_shm_fake(name, value, typ, **kwargs)


def recv_value_no_transfer(name: str, typ: str, *, method: str = 'shm', **kwargs):
    """Fake-transfer recv: waits for handshake, returns zero tensor without H->D copy."""
    if method != 'shm':
        raise NotImplementedError(f"no-transfer mode requires method='shm', got '{method}'")
    if '/' not in name:
        name = '/dev/shm/' + name
    return _shm.recv_value_shm_fake(name, typ, **kwargs)


# small helpers
def reset_channel(name: str):
    """Remove the channel file to reset its state (optional)."""
    try: os.remove(_path(name))
    except FileNotFoundError: pass

def tensor_hash(t: torch.Tensor, algo: str = "sha256", include_meta: bool = True) -> str:
    # compare by *value* only; ignore autograd graph & device
    if t.requires_grad:
        t = t.detach()
    # make a dense, standard layout on CPU
    t = t.contiguous().cpu()

    # Numpy view (no copy if already contiguous/CPU)
    if t.dtype == torch.bfloat16:
        # numpy’s bfloat16 support can be finicky; use uint16 view
        arr = t.view(torch.uint16).numpy()
        meta_dtype = "bfloat16"
    else:
        arr = t.numpy()
        meta_dtype = str(arr.dtype)

    h = hashlib.new(algo)
    if include_meta:
        # include shape + dtype so different shapes/dtypes don’t collide
        h.update(str(tuple(arr.shape)).encode())
        h.update(meta_dtype.encode())
    # value bytes (C-order)
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()