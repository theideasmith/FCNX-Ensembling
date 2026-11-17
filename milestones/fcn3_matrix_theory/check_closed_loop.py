"""
Closed-loop check of fcn3_matrix_solver.py against equations (E1)-(E8) of notes/fcn3_matrix_eos.tex.

Takes ONLY the solver's converged state (J, qbar_h, M, t) and re-evaluates every equation over the FULL
distributions, with numerics written independently of the solver:
  - e_n(h, q_h) and c_n(w0, q) by Gauss-Hermite projection in x0 (not the closed form);
  - the hidden-field distribution (E2) on a plain tensor grid in z = L^-1 h (no s lattice, no rotated rule)
    times a uniform grid in log q_h;
  - the read-in distribution (E5) on a plain (w0, log q) grid with w0 of both signs;
  - E_{w,w'} k(w,w')^2 (E8) by Monte Carlo over pairs drawn from the read-in grid distribution, with the cosine
    between perpendicular parts drawn from its exact distribution.
It reports the fixed-point residuals of t (E4), J (E6), qbar_h (E7), M (E8), and the mass at the edge of every
grid (to show that nothing is cut off).

    python check_closed_loop.py            # thesis point, n_max = 3 (the hidden-field grid is 2-D)
"""
import argparse
import math

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

from fcn3_matrix_solver import FCN3Solver, Params

DT = torch.float64


def hermite_basis(ks, x):
    H = [torch.ones_like(x), x.clone()]
    for n in range(1, max(ks)):
        H.append(x * H[n] - n * H[n - 1])
    return torch.stack([H[n] / math.sqrt(math.factorial(n)) for n in ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--chi", type=float, default=10.0)
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--N1", type=float, default=1600)
    ap.add_argument("--nz", type=int, default=1201, help="grid points per z direction")
    ap.add_argument("--zmax", type=float, default=48.0)
    ap.add_argument("--nqh", type=int, default=240)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--nw", type=int, default=2001)
    ap.add_argument("--nq", type=int, default=1601)
    ap.add_argument("--npairs", type=int, default=4_000_000)
    ap.add_argument("--ray", type=int, default=0, help="use the kappa0 = 0.015 ray point at this d (ray_k0p015.py)")
    ap.add_argument("--reference", action="store_true", help="check fcn3_matrix_solver.FCN3Solver instead of the fast solver")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.ray:
        import ray_k0p015 as R
        from fcn3_fast import FastSolver as _FS
        J11 = float(_FS(R.params(a.ray, float("inf"), 1.0), verbose=False).state0["J"][0, 0])
        p = R.params(a.ray, a.N1, math.sqrt((R.S0 / a.ray) / J11))
        p.nmax = 3
    else:
        p = Params(nmax=3, kappa=a.kappa, chi=a.chi, d=a.d, N1=a.N1)
    if a.reference:
        S = FCN3Solver(p, verbose=False, device=dev)
    else:
        from fcn3_fast import FastSolver
        S = FastSolver(p, verbose=False, device=dev)
    res = S.solve()
    st = S.last["state"]
    J = st["J"].to(dev)
    qbar, M = float(st["qbar"]), float(st["M"])
    t = S.last["t"].to(dev)
    y = torch.tensor(p.y(), dtype=DT, device=dev)
    print(f"solver: converged={res['converged']} ({res['walltime_s']:.1f}s)  L1={res['L1']:.6f}  L3={res['L3']:.6f}  "
          f"qbar={qbar:.6f}  M={M:.4f}  far masses: read-in {res['readin_far_mass']}  hidden {res['hidden_far_mass']}")

    xg, wg = hermegauss(a.nx)
    x0 = torch.tensor(xg, dtype=DT, device=dev)
    wx = torch.tensor(wg / math.sqrt(2 * math.pi), dtype=DT, device=dev)
    Phi = hermite_basis(p.ks, x0)
    PW = (Phi * wx).T

    # ------------------------------------------------------------ (E2)-(E3) hidden field, full distribution
    G = p.sigB ** 2 * J
    L = torch.linalg.cholesky(G)
    k = M / 2
    zg = torch.linspace(-a.zmax, a.zmax, a.nz, dtype=DT, device=dev)
    Z1, Z2 = torch.meshgrid(zg, zg, indexing="ij")
    z = torch.stack([Z1.reshape(-1), Z2.reshape(-1)], 1)
    h = z @ L.T
    lpz = -0.5 * (z * z).sum(1)
    lqh = math.log(qbar) + torch.linspace(-14, 3, a.nqh, dtype=DT, device=dev)
    qh = torch.exp(lqh)
    lwq = k * (lqh - math.log(qbar)) - k * qh / qbar
    hx = h @ Phi
    lse, Kacc, Hacc, Eqacc, edge = [], [], [], [], {}
    for j in range(a.nqh):
        e = torch.erf(hx / math.sqrt(1 + 2 * float(qh[j]))) @ PW
        te = e @ t
        lw = lpz + lwq[j] + 0.5 * p.gt * te * te
        m = lw.max()
        w = torch.exp(lw - m)
        lse.append(m + torch.log(w.sum()))
        Kacc.append((m, torch.einsum("p,pi,pj->ij", w, e, e)))
        Hacc.append((m, torch.einsum("p,pi,pj->ij", w, h, h)))
        Eqacc.append((m, w.sum() * qh[j]))
        edge[j] = (m, w.reshape(a.nz, a.nz))
    lse = torch.stack(lse)
    logZ = torch.logsumexp(lse, 0)
    comb = lambda lst: sum(torch.exp(m - logZ) * v for m, v in lst)
    Kb, Hb, Eqb = comb(Kacc), comb(Hacc), float(comb(Eqacc))
    pq = torch.exp(lse - logZ)
    zedge = sum(float(torch.exp(m - logZ) * (W[0].sum() + W[-1].sum() + W[:, 0].sum() + W[:, -1].sum()))
                for m, W in edge.values())
    print(f"hidden field: mass at q_h grid edges {float(pq[0] + pq[-1]):.1e}, at z grid edges {zedge:.1e}")

    tn = torch.linalg.solve(p.sigA ** 2 * Kb + p.lam * torch.eye(2, dtype=DT, device=dev), y)       # (E4)

    # ------------------------------------------------------------ (E5)-(E8) read-in, full distribution
    q0 = p.sigw ** 2 * (p.d - 1) / p.d
    w0 = torch.linspace(-6, 6, a.nw, dtype=DT, device=dev)
    lq = torch.linspace(math.log(q0) - 18, math.log(q0) + 2, a.nq, dtype=DT, device=dev)
    W0, LQ = torch.meshgrid(w0, lq, indexing="ij")
    Q = torch.exp(LQ)
    c = torch.erf(W0[..., None] * x0 / torch.sqrt(1 + 2 * Q)[..., None]) @ PW                   # (nw, nq, 2)
    knorm = lambda n2: (2 / math.pi) * torch.asin(2 * n2 / (1 + 2 * n2))
    Pp = knorm(W0 * W0 + Q) - (c * c).sum(-1)
    Ginv = torch.linalg.inv(G)
    Vt = Ginv @ (G - Hb) @ Ginv                                                                  # as written
    mcoef = M * (qbar - Eqb) / qbar ** 2
    Sact = p.beta * (W0 * W0 + Q) - 0.5 * (p.d - 1) * LQ \
        + 0.5 * p.rB * torch.einsum("abi,ij,abj->ab", c, Vt, c) + 0.5 * p.rB * mcoef * Pp
    lp = -Sact
    pw = torch.exp(lp - torch.logsumexp(lp.reshape(-1), 0))
    Jn = torch.einsum("ab,abi,abj->ij", pw, c, c)
    EJxx = float((pw * knorm(W0 * W0 + Q)).sum())
    qbar_n = p.sigB ** 2 * (EJxx - float(torch.trace(Jn)))
    print(f"read-in: mass at w0 grid edges {float(pw[0].sum() + pw[-1].sum()):.1e}, at q grid edges "
          f"{float(pw[:, 0].sum() + pw[:, -1].sum()):.1e}, mass with |w0| > 1: {float(pw[W0.abs() > 1].sum()):.3e}")
    # E k^2 by Monte Carlo over pairs
    gen = torch.Generator(device=dev).manual_seed(0)
    flat = pw.reshape(-1)
    acc, acc2, n = 0.0, 0.0, 0
    for _ in range(a.npairs // 500_000):
        i1 = torch.multinomial(flat, 500_000, replacement=True, generator=gen)
        i2 = torch.multinomial(flat, 500_000, replacement=True, generator=gen)
        a1, b1 = W0.reshape(-1)[i1], Q.reshape(-1)[i1]
        a2, b2 = W0.reshape(-1)[i2], Q.reshape(-1)[i2]
        # cosine between independent uniform directions in R^(d-1): u = 2 B - 1, B ~ Beta((d-2)/2, (d-2)/2)
        bdist = torch.distributions.Beta(torch.tensor((p.d - 2) / 2, dtype=DT, device=dev),
                                         torch.tensor((p.d - 2) / 2, dtype=DT, device=dev))
        u = 2 * bdist.sample((500_000,)) - 1
        num = 2 * (a1 * a2 + torch.sqrt(b1 * b2) * u)
        den = torch.sqrt((1 + 2 * (a1 * a1 + b1)) * (1 + 2 * (a2 * a2 + b2)))
        k2 = ((2 / math.pi) * torch.asin(num / den)) ** 2
        acc += float(k2.sum()); acc2 += float((k2 * k2).sum()); n += len(k2)
    EJ2 = acc / n
    se = math.sqrt(max(acc2 / n - EJ2 ** 2, 0) / n)
    M_n = qbar_n ** 2 / (p.sigB ** 4 * (EJ2 - float((Jn * Jn).sum())))
    M_se = M_n * se / (EJ2 - float((Jn * Jn).sum()))

    rel = lambda A, B: ((A - B).abs() / B.abs()).cpu().numpy().ravel()
    print("\nfixed-point residuals (independent evaluation vs solver state):")
    print(f"  (E2-E4) t       rel diff {rel(tn, t)}")
    print(f"  (E3)    H       independent {Hb.cpu().numpy().ravel()}  solver {np.ravel(res['H'])}")
    print(f"  (E3)    <q_h>/qbar  independent {Eqb / qbar:.8f}  solver {res['qh_over_qbar']:.8f}")
    print(f"  (E6)    J       rel diff {rel(Jn, J)}")
    print(f"  (E7)    qbar    {qbar_n:.8f} vs {qbar:.8f}  rel {abs(qbar_n / qbar - 1):.2e}")
    print(f"  (E8)    M       {M_n:.4f} +- {M_se:.4f} (Monte Carlo) vs {M:.4f}  rel {abs(M_n / M - 1):.2e}")


if __name__ == "__main__":
    main()
