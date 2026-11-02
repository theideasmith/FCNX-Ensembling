"""
Independent checks of fcn3_matrix_solver.py at one solved point (default: thesis parameters, S = {1, 3}).

 1. B_n(h1 e1, q_h) by Gauss-Hermite equals the closed form c_n(h1, q_h) (eq. 3 with w0 -> h1).
 2. Readout identity (sigma_A^2 K + kappa/P) t = y at the solution.
 3. Middle law by brute force in the ORIGINAL coordinates (h1, h3) on a dense tensor grid and a dense q_h
    grid (no whitening, no s-lattice, no control variate), restricted to the same central region:
    K, H, <q_h> against the solver.
 4. E_{x,x'} J(x,x')^2 by Monte Carlo over explicit read-in vectors w in R^d (random perp directions),
    against the Gauss-Jacobi pair quadrature.
"""
import argparse
import math

import numpy as np
import torch

from fcn3_matrix_solver import DT, FCN3Solver, Numerics, Params, cn_const, jxx, readin_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--n_mc", type=int, default=4000)
    a = ap.parse_args()
    p = Params(kappa=a.kappa)
    S = FCN3Solver(p, verbose=False)
    res = S.solve()
    dev = S.dev
    lt = S.last
    mid = S.mid
    print(f"solved in {res['walltime_s']:.2f}s, converged={res['converged']}")

    # 1. features
    h1 = torch.linspace(-3, 3, 13, dtype=DT, device=dev)
    for qh in (0.1, 0.46, 2.0):
        arg = (h1[:, None] * mid.Phi[0][None, :]) / math.sqrt(1 + 2 * qh)
        Bgh = torch.erf(arg) @ mid.WPhi
        Bcf = readin_features(h1, torch.full_like(h1, qh), p.ks).T
        print(f"[1] q_h={qh}: max |B_GH - c_closed| = {float((Bgh - Bcf).abs().max()):.2e}")

    # 2. readout identity
    t = lt["t"]
    K = torch.tensor(res["K"], dtype=DT, device=dev)
    y = torch.tensor(p.y(), dtype=DT, device=dev)
    r = (p.sigA ** 2 * K + p.lam * torch.eye(p.nS, dtype=DT, device=dev)) @ t - y
    print(f"[2] |(sA^2 K + k/P) t - y| = {float(r.abs().max()):.2e}")

    # 3. brute-force middle law in (h1, h3, q_h)
    L, u = lt["L"], mid.u
    G = L @ L.T
    qbar, M = lt["state"]["qbar"], lt["state"]["M"]
    scut = res["middle_scut"]
    sd = torch.sqrt(torch.diag(G))
    n1, n3, nqh = 801, 401, 64
    # the region |s| <= scut is a slab; the grid must cover it in h1, h3 (scut ~ 9 sigma)
    g1 = torch.linspace(-1.05, 1.05, n1, dtype=DT, device=dev) * (scut + 4) * sd[0]
    g3 = torch.linspace(-1.0, 1.0, n3, dtype=DT, device=dev) * 12 * sd[1]
    k = M / 2
    qg = qbar * torch.exp(torch.linspace(-8, 8, nqh, dtype=DT, device=dev) / math.sqrt(k))
    lq = (k - 1) * torch.log(qg) - k * qg / qbar + torch.log(torch.gradient(qg)[0])
    wq = torch.exp(lq - torch.logsumexp(lq, 0))
    H1, H3 = torch.meshgrid(g1, g3, indexing="ij")
    h = torch.stack([H1, H3], -1)                                        # (n1, n3, 2)
    Linv = torch.linalg.inv(L)
    z = h @ Linv.T
    s = z @ u
    lprior = -0.5 * (z * z).sum(-1)
    inside = s.abs() <= scut + 0.5 * S.nu.ds                              # same lattice region as the solver
    hx = h @ mid.Phi                                                      # (n1, n3, nx)
    acc = dict(Z=0, K=0, H=0, q=0, Z0=0, H0=0)
    for j in range(nqh):
        B = torch.erf(hx / math.sqrt(1 + 2 * float(qg[j]))) @ mid.WPhi
        tB = B @ t
        lw = lprior + 0.5 * p.gt * tB * tB
        lw = torch.where(inside, lw, torch.full_like(lw, -torch.inf))
        acc.setdefault("lw", []).append(lw + torch.log(wq[j]))
    LW = torch.stack(acc["lw"])                                          # (nqh, n1, n3)
    W = torch.exp(LW - LW.max())
    Zt = W.sum()
    Kb = torch.zeros(2, 2, dtype=DT, device=dev)
    for j in range(nqh):
        B = torch.erf(hx / math.sqrt(1 + 2 * float(qg[j]))) @ mid.WPhi
        Kb += torch.einsum("ab,abi,abj->ij", W[j], B, B)
    Kb /= Zt
    Hb = torch.einsum("qab,abi,abj->ij", W, h, h) / Zt
    Eqb = float((W.sum((1, 2)) * qg).sum() / Zt)
    W0 = torch.where(inside, torch.exp(lprior), torch.zeros_like(lprior))
    H0b = torch.einsum("ab,abi,abj->ij", W0, h, h) / W0.sum()
    Eq0b = float((wq * qg).sum())
    Hs = torch.tensor(res["H"], dtype=DT, device=dev)
    print("[3] brute force vs solver (relative differences)")
    print("    K :", ((Kb - K) / K).cpu().numpy().round(5).tolist())
    dHb, dHs = Hb - H0b, Hs - G
    print("    H-G:", ((dHb - dHs) / dHs).cpu().numpy().round(5).tolist())
    print(f"    <q_h>/qbar - 1: brute {Eqb / Eq0b - 1:.6e}  solver {res['qh_over_qbar'] - 1:.6e}")

    # 4. E J(x,x')^2 by explicit vectors
    ri = S.readin
    g = ri.coarse
    Lg = ri.logp(g, lt["Ahat"], lt["Linv"], lt["mperp"])
    wcut = res["readin_wcut"]
    if wcut is not None:
        Lg = torch.where(g["W"] <= wcut, Lg, torch.full_like(Lg, -torch.inf))
    pg = torch.exp(Lg - torch.logsumexp(Lg.reshape(-1), 0)).reshape(-1)
    gen = torch.Generator(device=dev).manual_seed(1)
    idx = torch.multinomial(pg, a.n_mc, replacement=True, generator=gen)
    w0 = g["W"].reshape(-1)[idx] * (2 * torch.randint(0, 2, (a.n_mc,), generator=gen, device=dev) - 1)
    q = g["Q"].reshape(-1)[idx]
    v = torch.randn(a.n_mc, p.d - 1, dtype=DT, device=dev, generator=gen)
    v = v / v.norm(dim=1, keepdim=True) * torch.sqrt(q)[:, None]
    w = torch.cat([w0[:, None], v], 1)
    nrm = 1 + 2 * (w * w).sum(1)
    kk = (2 / math.pi) * torch.asin(2 * (w @ w.T) / torch.sqrt(torch.outer(nrm, nrm)))
    off = ~torch.eye(a.n_mc, dtype=torch.bool, device=dev)
    mc = (kk[off] ** 2).mean()
    se = (kk[off] ** 2).std() / math.sqrt(a.n_mc)          # rough (pairs are correlated)
    EJ2, _ = ri.pair_trace(lt["Ahat"], lt["Linv"], lt["mperp"], wcut)
    print(f"[4] E J^2: MC {float(mc):.6e} (+- ~{float(se):.1e})  quadrature {float(EJ2):.6e}")


if __name__ == "__main__":
    main()
