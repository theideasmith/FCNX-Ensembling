"""
Exactness tests for fcn3_sampler.py on a tiny network (CPU, float64).

 1. The collapsed potential and its hand-written chunked gradient against autograd of the direct P x P marginal
    likelihood of y given (W, B).
 2. Two independent samplers of the same posterior must agree:
      A: collapsed HMC alone (+ exact draws of a),
      B: jump moves alone (middle and read-in sweeps, both direction kernels) + exact draws of a, no HMC.
    Statistics are compared with batch-means error bars.

    python test_fcn3_sampler.py              # both tests (~ a few minutes)
"""
import argparse
import math

import numpy as np
import torch

from fcn3_sampler import FCN3, he_norm


def tiny_problem(d=5, P=12, seed=0, eps=0.5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(P, d, dtype=torch.float64, generator=g)
    x0 = X[:, 0]
    Y = x0 + eps * (x0 ** 3 - 3 * x0) / math.sqrt(6)
    return X, Y


def test_gradient():
    X, Y = tiny_problem()
    m = FCN3(X, Y, N1=4, N2=3, chi=2.0, kappa=0.3, chunk=5)     # chunk < P exercises the chunk loop
    g = torch.Generator().manual_seed(1)
    ZW = torch.randn(4, X.shape[1], dtype=torch.float64, generator=g)
    ZB = torch.randn(3, 4, dtype=torch.float64, generator=g)
    u, gW, gB = m.U(ZW, ZB)

    def direct(ZW, ZB):
        W, B = m.sw * ZW, m.sb * ZB
        F2 = torch.erf(torch.erf(X @ W.T) @ B.T)
        K = m.sa2 * F2 @ F2.T + m.v * torch.eye(m.P, dtype=torch.float64)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(Y[:, None], L)[:, 0]
        return 0.5 * (ZW ** 2).sum() + 0.5 * (ZB ** 2).sum() + 0.5 * Y @ alpha + torch.log(torch.diagonal(L)).sum()

    a, b = ZW.clone().requires_grad_(), ZB.clone().requires_grad_()
    ud = direct(a, b)
    ud.backward()
    print(f"[1] U: chunked {u:.12f} direct {float(ud):.12f}  |dU| = {abs(u - float(ud)):.1e}")
    print(f"    grad W rel err {float((gW - a.grad).norm() / a.grad.norm()):.1e},"
          f" grad B rel err {float((gB - b.grad).norm() / b.grad.norm()):.1e}")


def test_series():
    """Taylor-series read-in energies against direct erf, at a realistic weight scale."""
    X, Y = tiny_problem(d=20, P=60)
    m = FCN3(X, Y, N1=400, N2=50, chi=1.0, kappa=0.05)
    g = torch.Generator().manual_seed(2)
    W = m.sw * torch.randn(400, 20, dtype=torch.float64, generator=g)
    B = m.sb * torch.randn(50, 400, dtype=torch.float64, generator=g)
    a = m.draw_a(W, B)
    m._build(W, B, a)
    j = 7
    phi = torch.erf(X @ torch.randn(20, 5, dtype=torch.float64, generator=g) * m.sw)
    Bc = m.sb * 3 * torch.randn(50, 5, dtype=torch.float64, generator=g)
    h0 = m.H - m.F1[:, j:j + 1] * m.B[:, j][None, :]
    direct = torch.stack([(torch.erf(h0 + phi[:, i:i + 1] * Bc[:, i][None]) - torch.erf(h0)) @ m.a for i in range(5)], 1)
    K = m._series_order(float(Bc.abs().max()))
    ser = m._erf_series(h0, (2 / math.sqrt(math.pi)) * torch.exp(-h0 * h0), phi, Bc, K)
    print(f"[0] series order K={K} for max|b|={float(Bc.abs().max()):.3f}: max |series - direct| = "
          f"{float((ser - direct).abs().max()):.1e} (tolerance {m.tol_f:.1e})")


def stats(m, W, B, a, xs):
    """A few scalar statistics of one state."""
    f = torch.erf(torch.erf(xs @ W.T) @ B.T) @ a
    return np.array([float((W[:, 0] ** 2).mean() / m.sw ** 2), float((W[:, 1:] ** 2).sum(1).mean() / m.sw ** 2),
                     float((B ** 2).mean() / m.sb ** 2), float((a ** 2).mean() / m.sa2),
                     float(f[0]), float(f[1]), float((f ** 2).mean()), float((W[:, 0] ** 4).mean() / m.sw ** 4)])


NAMES = ["w0^2", "|wp|^2", "B^2", "a^2", "f(x1)", "f(x2)", "<f^2>", "w0^4"]


def batch_means(S, nb=20):
    n = len(S) // nb * nb
    b = S[:n].reshape(nb, -1, S.shape[1]).mean(1)
    return b.mean(0), b.std(0, ddof=1) / math.sqrt(nb)


def test_invariance(n_iter, seed=3, N1=4, N2=3):
    X, Y = tiny_problem()
    xs = torch.randn(6, X.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    res = {}
    for mode in ("hmc", "moves"):
        m = FCN3(X, Y, N1=N1, N2=N2, chi=1.0, kappa=1.0, n_cand=8, redraw_frac=0.5, seed=seed)
        g = torch.Generator().manual_seed(seed)
        ZW = torch.randn(N1, X.shape[1], dtype=torch.float64, generator=g)
        ZB = torch.randn(N2, N1, dtype=torch.float64, generator=g)
        u, gW, gB = m.U(ZW, ZB)
        out, acc, direct = [], [], []
        for it in range(n_iter):
            if mode == "hmc":
                ZW, ZB = m.flip_signs(ZW, ZB)
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
                direct.append(st["in_direct"])
                ZW, ZB = W / m.sw, B / m.sb
            if it >= n_iter // 10:
                out.append(stats(m, W, B, a, xs))
        mu, se = batch_means(np.array(out))
        res[mode] = (mu, se)
        print(f"[2] N1={N1} N2={N2} {mode:5s}: mean acc/changed {np.mean(acc):.3f}"
              + (f", series used in {1 - np.mean(direct):.0%} of read-in moves" if mode == "moves" else ""))
    print("    stat        HMC             moves           z")
    zs = []
    for i, nm in enumerate(NAMES):
        (m1, s1), (m2, s2) = (res["hmc"][0][i], res["hmc"][1][i]), (res["moves"][0][i], res["moves"][1][i])
        z = (m1 - m2) / math.sqrt(s1 ** 2 + s2 ** 2)
        zs.append(z)
        print(f"    {nm:8s} {m1:9.4f}±{s1:.4f}  {m2:9.4f}±{s2:.4f}  {z:+.2f}")
    print(f"    max |z| = {np.max(np.abs(zs)):.2f} over {len(zs)} statistics")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--N1", type=int, default=4)
    ap.add_argument("--N2", type=int, default=3)
    ap.add_argument("--skip-invariance", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    test_series()
    test_gradient()
    if not a.skip_invariance:
        test_invariance(a.iters, N1=a.N1, N2=a.N2)
