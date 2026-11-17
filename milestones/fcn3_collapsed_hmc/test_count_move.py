"""Exactness test of FCN3.count_move (NCMC insert/delete of a read-in neuron across r^2 = thr).

On a tiny network, plain HMC mixes, so its stationary distribution is the reference. A chain of HMC + count moves must
give the same distribution of the outer count n = #{j : r_j^2 > thr} (r^2 = w0^2/(1+2|w|^2), the classification the
move uses) and of other observables.

    python test_count_move.py [--iters 12000]   (CPU, ~1.5 h)
"""
import argparse
import math
import time

import numpy as np
import torch

from fcn3_sampler import FCN3


def make(seed=0, d=4, P=40, N1=6, N2=4, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(P, d, generator=g)
    Y = X[:, 0] - 0.3 * (X[:, 0] ** 3 - 3 * X[:, 0]) / math.sqrt(6)
    return FCN3(X.to(dev), Y.to(dev), N1, N2, chi=1.0, kappa=0.3, sigA=1.0, sigB=1.0, sigw=1.0, ks=(1, 3), seed=seed + 1)


def chain(m, iters, step, n_leap, count_every, thr, bw, K, L, seed):
    torch.manual_seed(seed)
    m.gen.manual_seed(seed)
    m.np_rng = np.random.default_rng(seed)
    ZW = torch.randn(m.N1, m.d)
    ZB = torch.randn(m.N2, m.N1)
    u, gW, gB = m.U(ZW, ZB)
    ns, w0abs, acc_c, att_c = [], [], 0, 0
    for it in range(iters):
        ZW, ZB = m.flip_signs(ZW, ZB)
        u, gW, gB = m.U(ZW, ZB)
        ZW, ZB, u, gW, gB, acc, Lh = m.hmc_step(ZW, ZB, u, gW, gB, step, n_leap)
        if count_every and it % count_every == 0:
            ZW, ZB, st = m.count_move(ZW, ZB, K=K, n_leap=L, step=step, thr=thr, bw=bw)
            if math.isfinite(st["count_logA"]):
                att_c += 1
                acc_c += st["count_acc"]
        if it >= iters // 10:
            w0 = (ZW[:, 0] * m.sw).abs()
            ns.append(int((m._r2(ZW) > thr).sum()))
            w0abs.append(float(w0.mean()))
    return np.array(ns), np.array(w0abs), acc_c / max(att_c, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12000)
    a = ap.parse_args()
    m = make()
    thr, bw = 0.1, (0.3, 0.3)
    res = {}
    for name, ce in (("HMC only", 0), ("HMC + count moves", 1)):
        t0 = time.time()
        ns, w0, ar = chain(m, a.iters, step=0.15, n_leap=8, count_every=ce, thr=thr, bw=bw, K=6, L=3, seed=7)
        h = np.bincount(ns, minlength=m.N1 + 1) / len(ns)
        # batch-means standard errors
        bm = lambda x: np.array([b.mean() for b in np.array_split(x, 40)]).std(ddof=1) / math.sqrt(40)
        res[name] = (h, ns.mean(), bm(ns.astype(float)), w0.mean(), bm(w0))
        print(f"{name:20s} <n>={ns.mean():.4f}±{bm(ns.astype(float)):.4f}  <|w0|>={w0.mean():.4f}±{bm(w0):.4f}  "
              f"P(n)={np.round(h, 4)}  count acc {ar:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    (h0, n0, e0, w0_, f0), (h1, n1, e1, w1_, f1) = res["HMC only"], res["HMC + count moves"]
    zn = (n1 - n0) / math.hypot(e0, e1)
    zw = (w1_ - w0_) / math.hypot(f0, f1)
    print(f"difference in <n>: {zn:+.2f} sigma, in <|w0|>: {zw:+.2f} sigma  ->  {'PASS' if abs(zn) < 3 and abs(zw) < 3 else 'FAIL'}")


if __name__ == "__main__":
    main()
