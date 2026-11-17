"""
Checks of fcn3_fast.py against the reference fcn3_sampler.py (CUDA).

 1. kernels: fused read-in passes vs the direct torch expressions;
 2. float32 HMC gradient vs the reference float64 gradient (realistic size);
 3. graphed sweep == eager sweep (same random numbers), and the sweep's caches (H, f) vs a recomputation;
 4. invariance on a tiny problem: fast HMC alone vs fast jump moves alone vs reference HMC (batch means).

    python test_fcn3_fast.py [--device cuda:1] [--iters 20000]
"""
import argparse
import math
import sys

import numpy as np
import torch

from fcn3_sampler import FCN3, make_data
from fcn3_fast import FastFCN3, _pass1, _pass2
from test_fcn3_sampler import tiny_problem, stats, batch_means, NAMES


def realistic(dev, N=1500, d=150, P=3000, cls=FastFCN3, **kw):
    X, Y, _, _ = make_data(d, P, -0.7348469228349535, 1.0, 0, dev)
    X, Y = X.float(), Y.float()
    m = cls(X, Y, N, N, N / 1500, 0.015, sigA=(8 / 1500) ** 0.5, sigB=1.2554, sigw=0.7071, ks=(1, 3), seed=5, **kw)
    g = torch.Generator(device=dev).manual_seed(1)
    ZW = torch.randn(N, d, device=dev, generator=g)
    ZB = torch.randn(N, N, device=dev, generator=g)
    return m, ZW, ZB


def test_kernels(dev):
    g = torch.Generator(device=dev).manual_seed(0)
    for P, N2, C in ((3000, 1500, 16), (777, 301, 5), (12, 3, 8)):
        H = torch.randn(P, N2, device=dev, generator=g)
        f1 = torch.rand(P, device=dev, generator=g)
        bj = 0.1 * torch.randn(N2, device=dev, generator=g)
        a = torch.randn(N2, device=dev, generator=g) / N2
        phi = torch.rand(P, C, device=dev, generator=g) * 2 - 1
        Bc = 0.1 * torch.randn(N2, C, device=dev, generator=g)
        Dm, d0, e0a, S2 = torch.empty(P, N2, device=dev), torch.empty(8, P, device=dev), torch.empty(8, P, device=dev), \
            torch.empty(4, C, P, device=dev)
        _pass1(H, f1, bj, a, Dm, d0, e0a)
        _pass2(H, f1, bj, a, phi, Bc.T.contiguous(), S2)
        d0, e0a, S2 = d0.sum(0), e0a.sum(0), S2.sum(0).T
        Hd, h0 = H.double(), (H - f1[:, None] * bj[None]).double()
        ad = a.double()
        rDm = (2 / math.sqrt(math.pi)) * torch.exp(-h0 ** 2) * ad
        rd0 = (torch.erf(Hd) - torch.erf(h0)) @ ad
        re0 = torch.erf(h0) @ ad
        rS2 = torch.stack([torch.erf(h0 + phi[:, c:c + 1].double() * Bc[:, c].double()[None]) @ ad for c in range(C)], 1)
        err = lambda x, y: float((x.double() - y).abs().max() / y.abs().max())
        print(f"[1] P={P} N2={N2} C={C}: rel err Dm {err(Dm, rDm):.1e} d0 {err(d0, rd0):.1e} e0a {err(e0a, re0):.1e}"
              f" pass2 {err(S2, rS2):.1e}")


def test_grad(dev):
    m, ZW, ZB = realistic(dev)
    u, gW, gB = m.U(ZW, ZB)
    fW, fB, ok = m.grad32(ZW, ZB)
    uf, _ = m.U_exact(ZW, ZB)
    e = lambda x, y: float((x - y).norm() / y.norm())
    print(f"[2] U ref {u:.10e} U_exact {uf:.10e}; float32 grad rel err W {e(fW, gW):.1e} B {e(fB, gB):.1e} ok={bool(ok)}")


def test_graph(dev, compile=False):
    states = []
    for use_graph in (False, True):
        m, ZW, ZB = realistic(dev, N=600, P=1200, use_graph=use_graph, compile=compile)
        W, B = m.sw * ZW, m.sb * ZB
        a = m.draw_a(W, B)
        W, B, a, st = m.sweep(W, B, a)
        W, B, a, st2 = m.sweep(W, B, a)
        fs = m.fsweep
        F1 = torch.erf(m.X @ W.T)
        H = F1 @ B.T
        f = torch.erf(H) @ a
        print(f"[3] graph={use_graph}: {st2} cache drift |H| {float((fs.H - H).abs().max()):.1e}"
              f" |f| {float((fs.f - f).abs().max()):.1e}")
        states.append((W, B, a))
    print(f"    graph vs eager: max |dW| {float((states[0][0] - states[1][0]).abs().max()):.1e}"
          f" |dB| {float((states[0][1] - states[1][1]).abs().max()):.1e}"
          f" |da| {float((states[0][2] - states[1][2]).abs().max()):.1e}")


def test_invariance(dev, n_iter, seed=3, N1=4, N2=3):
    X, Y = tiny_problem()
    X, Y = X.float().to(dev), Y.float().to(dev)
    xs = torch.randn(6, X.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(7)).float().to(dev)
    res = {}
    for mode in ("ref-hmc", "fast-hmc", "fast-moves"):
        cls = FCN3 if mode == "ref-hmc" else FastFCN3
        m = cls(X, Y, N1=N1, N2=N2, chi=1.0, kappa=1.0, n_cand=8, redraw_frac=0.5, seed=seed)
        g = torch.Generator(device=dev).manual_seed(seed)
        ZW = torch.randn(N1, X.shape[1], device=dev, generator=g)
        ZB = torch.randn(N2, N1, device=dev, generator=g)
        u, gW, gB = m.U(ZW, ZB)
        out, acc = [], []
        for it in range(n_iter):
            if mode.endswith("hmc"):
                ZW, ZB = m.flip_signs(ZW, ZB)
                if mode == "ref-hmc":
                    u, gW, gB = m.U(ZW, ZB)
                ZW, ZB, u, gW, gB, ac, _ = m.hmc_step(ZW, ZB, u, gW, gB, 0.1, 12)
                acc.append(ac)
                W, B = m.sw * ZW, m.sb * ZB
                a = m.draw_a(W, B)
            else:
                W, B = m.sw * ZW, m.sb * ZB
                a = m.draw_a(W, B)
                W, B, a, st = m.sweep(W, B, a)
                acc.append(st["mid_changed"] + st["in_changed"])
                ZW, ZB = W / m.sw, B / m.sb
            if it >= n_iter // 10:
                out.append(stats(m, W.double(), B.double(), a.double(), xs.double()))
        res[mode] = batch_means(np.array(out))
        print(f"[4] {mode:10s}: mean acc/changed {np.mean(acc):.3f}", flush=True)
    print("    stat        ref-HMC          fast-HMC   z      fast-moves  z")
    zs = []
    for i, nm in enumerate(NAMES):
        m0, s0 = res["ref-hmc"][0][i], res["ref-hmc"][1][i]
        row = f"    {nm:8s} {m0:9.4f}±{s0:.4f}"
        for k in ("fast-hmc", "fast-moves"):
            m1, s1 = res[k][0][i], res[k][1][i]
            z = (m1 - m0) / math.sqrt(s0 ** 2 + s1 ** 2)
            zs.append(z)
            row += f"  {m1:9.4f}±{s1:.4f} {z:+.2f}"
        print(row)
    print(f"    max |z| = {np.max(np.abs(zs)):.2f} over {len(zs)} comparisons")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--skip-invariance", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    test_kernels(dev)
    test_grad(dev)
    test_graph(dev, compile=False)
    test_graph(dev, compile=True)
    if not a.skip_invariance:
        test_invariance(dev, a.iters)
