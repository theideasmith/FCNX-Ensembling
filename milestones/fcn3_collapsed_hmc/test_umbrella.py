"""Exactness test of umbrella_count.py on the tiny network of test_count_move.py (d=4, P=40, N1=6, N2=4, thr r^2=0.1).

Plain HMC mixes there, so the outer-count distribution P(n) is known directly. The umbrella estimate tags neuron 0,
runs one HMC chain per spring window, removes the springs by WHAM and predicts

    P(n=m+1)/P(n=m) = R(m) (N1 - m)/(m + 1)

for every m. The predicted P(n) must match a direct HMC histogram.

    python test_umbrella.py [--iters 1500]   (CPU, windows in parallel, ~5 min)
"""
import argparse
import math
import multiprocessing as mp

import numpy as np
import torch

from fcn3_sampler import FCN3
from test_count_move import make
from umbrella_count import Spring, odds, run_window, wham

THR, K = 0.1, 300.0
CS = np.round(np.linspace(0.0, 0.45, 16), 4)


def make_spring(seed=0):
    m0 = make(seed)
    m = type("SpringFCN3", (Spring, FCN3), {})(m0.X, m0.Y, m0.N1, m0.N2, chi=1.0, kappa=0.3, sigA=1.0, sigB=1.0,
                                               sigw=1.0, ks=(1, 3), seed=seed + 1)
    return m


def window(args):
    i, iters, seed = args
    torch.set_num_threads(1)
    m = make_spring()
    m.gen.manual_seed(seed + i)
    m.np_rng = np.random.default_rng(seed + i)
    g = torch.Generator().manual_seed(seed + i)
    ZW, ZB = torch.randn(m.N1, m.d, generator=g), torch.randn(m.N2, m.N1, generator=g)
    m.set_spring(0, CS[i], K)
    rec = []
    run_window(m, ZW, ZB, 0.15, iters, iters // 10, 8, 0, rec=rec, thr=THR)
    return np.array(rec)


def direct(iters, seed):
    torch.set_num_threads(1)
    m = make_spring()
    m.set_spring(0, 0.0, 0.0)                      # no spring
    m.gen.manual_seed(seed)
    m.np_rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    ZW, ZB = torch.randn(m.N1, m.d, generator=g), torch.randn(m.N2, m.N1, generator=g)
    rec = []
    run_window(m, ZW, ZB, 0.15, iters, iters // 10, 8, 0, rec=rec, thr=THR)
    x = np.array(rec)
    return (x[:, 0] > THR) + x[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--direct-iters", type=int, default=6000)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(len(CS) + 4) as pool:
        dres = [pool.apply_async(direct, (a.direct_iters, 100 + s)) for s in range(4)]
        recs = pool.map(window, [(i, a.iters, 7) for i in range(len(CS))])
        ns = np.concatenate([r.get() for r in dres]).astype(int)
    x, lw, f = wham([r[:, 0] for r in recs], CS, K)
    mm = np.concatenate([r[:, 1] for r in recs]).astype(int)
    res = odds(x, mm, lw, THR, 6)
    logP = np.zeros(7)
    for m_ in range(6):
        logP[m_ + 1] = logP[m_] + res[m_]["log_ratio"] if m_ in res else np.nan
    P = np.exp(logP - np.logaddexp.reduce(logP[np.isfinite(logP)]))
    h = np.bincount(ns, minlength=7) / len(ns)
    bm = lambda v: np.array([b.mean() for b in np.array_split(v, 40)]).std(ddof=1) / math.sqrt(40)
    print("n                ", " ".join(f"{n:7d}" for n in range(7)))
    print("umbrella P(n)    ", " ".join(f"{p:7.4f}" for p in P))
    print("direct HMC P(n)  ", " ".join(f"{p:7.4f}" for p in h))
    print("earlier HMC P(n)  ", "0.0007  0.0507  0.2486  0.3739  0.2481  0.0699  0.0081  (test_count_move, 12000 it)")
    nu = float((np.arange(7) * P).sum())
    print(f"<n>: umbrella {nu:.4f}, direct {ns.mean():.4f} ± {bm(ns.astype(float)):.4f}")
    for m_, r in res.items():
        print(f"  m={m_}: log P(m+1)/P(m) = {r['log_ratio']:+.3f}  ({r['n_samples']} samples)")




# ---------------------------------------------------------------------------------------------- fast path (GPU)
def fast_test(iters=1500, direct_iters=12000, dev="cuda:0"):
    """Same comparison with FastFCN3 on the GPU and both jump sweeps every iteration: checks the spring inside the
    read-in sweep (FastSweep.spring)."""
    from fcn3_fast import FastFCN3
    m0 = make(0)
    m = type("SpringFast", (Spring, FastFCN3), {})(m0.X.to(dev), m0.Y.to(dev), m0.N1, m0.N2, chi=1.0, kappa=0.3,
                                                    sigA=1.0, sigB=1.0, sigw=1.0, ks=(1, 3), seed=1)
    g = torch.Generator(device=dev).manual_seed(5)
    ZW, ZB = torch.randn(m.N1, m.d, device=dev, generator=g), torch.randn(m.N2, m.N1, device=dev, generator=g)
    m.set_spring(0, 0.0, 0.0)
    rec = []
    run_window(m, ZW, ZB, 0.15, direct_iters, direct_iters // 10, 8, 1, rec=rec, thr=THR)
    x = np.array(rec)
    ns = ((x[:, 0] > THR) + x[:, 1]).astype(int)
    recs = []
    for i, c in enumerate(CS):
        m.set_spring(0, c, K)
        r = []
        run_window(m, ZW, ZB, 0.15, iters, iters // 10, 8, 1, rec=r, thr=THR)
        recs.append(np.array(r))
    report(recs, ns, "fast path, jump sweeps with spring")


def report(recs, ns, label):
    x, lw, f = wham([r[:, 0] for r in recs], CS, K)
    mm = np.concatenate([r[:, 1] for r in recs]).astype(int)
    res = odds(x, mm, lw, THR, 6)
    logP = np.zeros(7)
    for m_ in range(6):
        logP[m_ + 1] = logP[m_] + res[m_]["log_ratio"] if m_ in res else np.nan
    P = np.exp(logP - np.logaddexp.reduce(logP[np.isfinite(logP)]))
    h = np.bincount(ns, minlength=7) / len(ns)
    bm = lambda v: np.array([b.mean() for b in np.array_split(v, 40)]).std(ddof=1) / math.sqrt(40)
    print(label)
    print("n                ", " ".join(f"{n:7d}" for n in range(7)))
    print("umbrella P(n)    ", " ".join(f"{p:7.4f}" for p in P))
    print("direct P(n)      ", " ".join(f"{p:7.4f}" for p in h))
    print(f"<n>: umbrella {float((np.arange(7) * P).sum()):.4f}, direct {ns.mean():.4f} ± {bm(ns.astype(float)):.4f}",
          flush=True)


if __name__ == "__main__":
    import sys
    if "--fast" in sys.argv:
        fast_test()
    else:
        main()
