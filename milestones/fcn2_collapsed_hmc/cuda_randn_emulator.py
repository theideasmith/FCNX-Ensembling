"""
CPU emulation of torch.randn(n, device='cuda') right after torch.manual_seed(seed), for a given GPU.

PyTorch's CUDA normal_ kernel (distribution_nullary_kernel, float32, curand4 engine):
  block = 256 threads, unroll = 4,
  grid  = min(ceil(n / block), SMs * (maxThreadsPerSM // block)),
  each thread idx = blockIdx*256 + threadIdx runs curand_init(seed, subsequence=idx, offset=0) (Philox4x32-10),
  then for linear = idx, idx + stride, ... (stride = block*grid*unroll, while linear < rounded n):
      (n0, n1, n2, n3) = curand_normal4(state)      # two Box-Muller pairs from one Philox block
      out[linear + k * block*grid] = n_k  for k = 0..3 (if < n)
Only the grid depends on the GPU, which is why the 4090 and the 4080 disagree for large tensors.
The Box-Muller is evaluated in float32 like curand's, so values can differ from the GPU's logf/sincosf in
the last bit; data_fingerprint() of the result is compared with the run's stored one.
"""

import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
GPUS = {"rtx4090": (128, 1536), "rtx4080": (76, 1536)}


def _mulhilo(a, b):
    p = a.astype(np.uint64) * b
    return (p >> np.uint64(32)).astype(np.uint32), (p & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def philox4x32_10(ctr, key):
    """ctr: (4, n) uint32, key: (2,) uint32 -> (4, n) uint32."""
    c0, c1, c2, c3 = (ctr[i].copy() for i in range(4))
    k0 = np.full(c0.shape, key[0], np.uint32)
    k1 = np.full(c0.shape, key[1], np.uint32)
    with np.errstate(over="ignore"):
        for r in range(10):
            hi0, lo0 = _mulhilo(c0, M0)
            hi1, lo1 = _mulhilo(c2, M1)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            if r < 9:
                k0 = k0 + W0
                k1 = k1 + W1
    return np.stack([c0, c1, c2, c3])


def _box_muller(x, y):
    """curand's _curand_box_muller in float32 arithmetic (x, y converted to float32 first, as on the GPU)."""
    f = np.float32
    INV = f(2.3283064365386963e-10)            # CURAND_2POW32_INV
    INV2PI = f(2.3283064365386963e-10 * 2 * np.pi)
    u = x.astype(f) * INV + f(INV / f(2))
    v = y.astype(f) * INV2PI + f(INV2PI / f(2))
    s = np.sqrt(f(-2) * np.log(u))
    return s * np.sin(v), s * np.cos(v)


def cuda_randn(n, seed, gpu="rtx4090", block=256, unroll=4):
    sms, max_threads = GPUS[gpu]
    grid = min((n + block - 1) // block, sms * (max_threads // block))
    stride = block * grid * unroll
    n_iter = (n - 1) // stride + 1
    threads = block * grid
    out = np.zeros(n, np.float32)
    key = np.array([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], np.uint32)
    idx = np.arange(threads, dtype=np.uint64)
    for it in range(n_iter):
        ctr = np.stack([np.full(threads, it, np.uint32), np.zeros(threads, np.uint32),
                        (idx & np.uint64(0xFFFFFFFF)).astype(np.uint32), (idx >> np.uint64(32)).astype(np.uint32)])
        r = philox4x32_10(ctr, key)
        a0, a1 = _box_muller(r[0], r[1])
        b0, b1 = _box_muller(r[2], r[3])
        vals = (a0, a1, b0, b1)
        base = it * stride + np.arange(threads)
        for k in range(4):
            li = base + k * threads
            ok = li < n
            out[li[ok]] = vals[k][ok].astype(np.float32)
    return out


def fingerprint(X):
    X = X.astype(np.float64)
    return f"{X.sum():.10e}/{(X ** 2).sum():.10e}"
