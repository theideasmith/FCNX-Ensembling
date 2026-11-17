#!/usr/bin/env python3
"""NNGP (infinite-width) learnabilities L1, L3 of the three-layer erf network of run_fcn3.py.

Network: f(x) = sum_k a_k erf(sum_j B_kj erf(w_j . x)), w ~ N(0, sigma_w^2/d I), B ~ N(0, sigma_B^2/N1),
a ~ N(0, sigma_A^2/(N2 chi)); likelihood exp(-chi/(2 kappa) sum (f - y)^2); y = phi_1(x0) + eps phi_3(x0).
At N1, N2 -> infinity f is a Gaussian process with the arcsin kernel composed twice,
    K1 = (2/pi) asin(2 sw2 x.x'/d / sqrt((1 + 2 sw2 |x|^2/d)(1 + 2 sw2 |x'|^2/d)))
    K2 = (2/pi) asin(2 sB2 K1(x,x') / sqrt((1 + 2 sB2 K1(x,x))(1 + 2 sB2 K1(x',x'))))
    K  = (sigma_A^2/chi) K2,  noise variance kappa/chi,
so the posterior mean is K(x,X)(K(X,X) + kappa/chi)^-1 Y (chi cancels).

Two ways to get L_n = E_x[phi_n(x0) fbar(x)] / (target coefficient of phi_n):
  eigen(): eigenlearning. Eigenvalues of K on the sphere |x|^2 = d (Gegenbauer projections of K(t), t = x.x'/d,
           by Gauss-Jacobi quadrature), kappa_e from  P = sum_i lam_i/(lam_i + kappa_e) + kappa/kappa_e  (as
           julia_lib/self_consistent_kappa_solver.jl, kappa_eff = P kappa_e), L_k = lam_k/(lam_k + kappa_e).
  gpr():   exact GP regression on a run's training set, L_n from the same x0-quadrature test set as the sampler.

    python nngp_fcn3.py            # table along the fixed-theory line, eigen vs exact GPR on dataset 0
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import eval_gegenbauer, gammaln, roots_jacobi

HERE = Path(__file__).resolve().parent

# run_fcn3.py defaults used on the onset / fixed-theory lines
SIGW2, SIGB, SIGA2, EPS = 0.5, 1.25540121081682, 8 / 1500, -0.3 * math.sqrt(6)


def kernel_t(t, d=None, sw2=SIGW2, sB=SIGB, sA2=SIGA2, chi=1.0):
    """Output kernel as a function of t = x.x'/d on the sphere |x|^2 = d."""
    a = 2 * sw2 / (1 + 2 * sw2)
    K1 = (2 / math.pi) * np.arcsin(a * t)
    K11 = (2 / math.pi) * math.asin(a)
    b = 2 * sB ** 2 / (1 + 2 * sB ** 2 * K11)
    return (sA2 / chi) * (2 / math.pi) * np.arcsin(b * K1)


def sphere_eigs(d, kmax=40, nq=600, **kw):
    """(lam_k, multiplicity N_k) for k = 0..kmax of the dot-product kernel K(t) on the sphere of radius sqrt(d)."""
    al = (d - 3) / 2
    t, w = roots_jacobi(nq, al, al) if d <= 400 else (None, None)
    if t is None or not np.all(np.isfinite(t)):         # Gauss-Jacobi nodes fail at large d: Gauss-Legendre instead
        t, w = np.polynomial.legendre.leggauss(8000)
        lw = np.log(w) + al * np.log1p(-t * t)
        w = np.exp(lw - lw.max())
    w = w / w.sum()                                     # E over the law of t = cos(angle), (1 - t^2)^((d-3)/2)
    K = kernel_t(t, d, **kw)
    nu = (d - 2) / 2
    lam, mult = np.zeros(kmax + 1), np.zeros(kmax + 1)
    for k in range(kmax + 1):
        Pk = eval_gegenbauer(k, nu, t) / eval_gegenbauer(k, nu, 1.0)
        lam[k] = (w * K * Pk).sum()
        mult[k] = 1.0 if k == 0 else math.exp(math.log(2 * k + d - 2) - math.log(k) + gammaln(k + d - 2)
                                                - gammaln(k) - gammaln(d - 1))
    return lam, mult


def eigen(d, P, kappa, kmax=40, **kw):
    """Eigenlearning L1, L3 (and kappa_e). The tail beyond kmax enters through the trace K(1) = sum_k N_k lam_k."""
    lam, mult = sphere_eigs(d, kmax, **kw)
    lam = np.maximum(lam, 0.0)
    trace = float(kernel_t(1.0, d, **kw))

    def g(ke):        # sum_i lam_i/(lam_i + ke) + kappa/ke - P, with sum_i lam_i/(lam_i+ke) = (trace - sum N lam^2/(lam+ke))/ke
        s = (trace - (mult * lam ** 2 / (lam + ke)).sum()) / ke
        return s + kappa / ke - P

    ke = brentq(g, 1e-30, 1e6, xtol=1e-300, rtol=1e-14, maxiter=500)
    L = lam / (lam + ke)
    return dict(L1=float(L[1]), L3=float(L[3]), kappa_e=ke, lam1=lam[1], lam3=lam[3])


def gpr(run, device="cuda:0", n_perp=4000, cg=None, cg_tol=1e-6, cg_max=3000):
    """Exact GP regression on the run's training set (make_data with the run's seed); L1, L3 from the sampler's
    x0-quadrature test set (X0Quadrature, n_perp perpendicular samples). Solve (K + kappa/chi) alpha = Y by Cholesky
    of the float64 kernel matrix on the GPU, or (cg=True, default for P > 24000, where the Cholesky factor does not fit) by
    conjugate gradients with the kernel built block by block in float32 and accumulated in float64."""
    import json
    import torch
    sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
    from fcn3_sampler import X0Quadrature, make_data
    a = json.loads((Path(run) / "config.json").read_text())["args"]
    d, P = a["d"], a["P"]
    dev = torch.device(device)
    X, Y, _, _ = make_data(d, P, a["eps"], 1.0, a["seed"], dev)
    X, Y = X.double(), Y.double()
    sw2, sB2, sA2 = a["sigw"] ** 2, a["sigB"] ** 2, a["sigA"] ** 2 / a["chi"]

    def k1(A, Bm, na, nb):
        return (2 / math.pi) * torch.asin(2 * sw2 * (A @ Bm.T) / d / torch.sqrt(torch.outer(1 + 2 * sw2 * na / d,
                                                                                              1 + 2 * sw2 * nb / d)))

    def k1diag(n):
        return (2 / math.pi) * torch.asin(2 * sw2 * n / d / (1 + 2 * sw2 * n / d))

    def K(A, Bm):
        na, nb = (A * A).sum(1), (Bm * Bm).sum(1)
        da, db = k1diag(na), k1diag(nb)
        return sA2 * (2 / math.pi) * torch.asin(2 * sB2 * k1(A, Bm, na, nb)
                                                / torch.sqrt(torch.outer(1 + 2 * sB2 * da, 1 + 2 * sB2 * db)))

    v = a["kappa"] / a["chi"]
    if cg is None:
        cg = P > 24000
    if not cg:
        Kxx = torch.empty(P, P, dtype=torch.float64, device=dev)
        for i in range(0, P, 4096):
            Kxx[i:i + 4096] = K(X[i:i + 4096], X)
        Kxx.diagonal().add_(v)
        L = torch.linalg.cholesky(Kxx)
        del Kxx
        alpha = torch.cholesky_solve(Y[:, None], L)[:, 0]
        del L
    else:
        X32 = X.float()

        def matvec(u):
            out = torch.empty_like(u)
            u32 = u.float()
            for i in range(0, P, 8192):
                Kb = K(X32[i:i + 8192], X32)
                out[i:i + 8192] = (Kb @ u32).double()
            return out + v * u
        alpha = torch.zeros_like(Y)
        r = Y.clone()
        p_ = r.clone()
        rr = float(r @ r)
        y2 = float(Y @ Y)
        for it in range(cg_max):
            Ap = matvec(p_)
            st = rr / float(p_ @ Ap)
            alpha += st * p_
            r -= st * Ap
            rr_new = float(r @ r)
            if rr_new < cg_tol ** 2 * y2:
                break
            p_ = r + (rr_new / rr) * p_
            rr = rr_new
        print(f"  CG: {it + 1} iterations, relative residual {math.sqrt(rr_new / y2):.1e}", flush=True)
    xq = X0Quadrature(d, n_perp, device=device, dtype=torch.float64)
    acc = torch.zeros(len(xq.x0), dtype=torch.float64, device=dev)
    for g in range(len(xq.x0)):
        Xt = torch.cat([xq.x0[g].expand(xq.xp.shape[0], 1), xq.xp], 1)
        for i in range(0, len(Xt), 2048):
            acc[g] += (K(Xt[i:i + 2048], X) @ alpha).sum()
    fx0 = acc / xq.xp.shape[0]
    fb = {n: float((xq.w0 * xq.phis[n] * fx0).sum()) for n in (1, 3)}
    del alpha
    torch.cuda.empty_cache()
    return dict(L1=fb[1], L3=fb[3] / a["eps"])


if __name__ == "__main__":
    import json
    ratio = 0.015 / 24000
    runs = {}
    for f in sorted((HERE / "runs").glob("*/config.json")):
        a = json.loads(f.read_text())["args"]
        if a["d"] == 150 and a["seed"] == 0 and a["init"] == "prior" and not a.get("init_from") \
                and abs(a["kappa"] / a["P"] / ratio - 1) < 1e-9 and (f.parent / "summary.json").exists():
            runs[int(a["P"])] = f.parent
    print("P       eigen L1  L3      | exact GPR on dataset 0: L1  L3")
    for P, r in sorted(runs.items()):
        e = eigen(150, P, ratio * P)
        s = f"{P:6d}  {e['L1']:.4f}  {e['L3']:.4f}"
        if P <= int(sys.argv[1] if len(sys.argv) > 1 else 24000):
            g = gpr(r)
            s += f"  | {g['L1']:.4f}  {g['L3']:.4f}"
        print(s, flush=True)
