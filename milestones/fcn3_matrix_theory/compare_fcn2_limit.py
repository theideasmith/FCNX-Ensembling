"""
FCN3 theory at N1 -> infinity vs the FCN2 read-in theory.

N1 enters the FCN3 equations only through rB = sigma_B^2 N2 / N1, which scales both back-couplings of the middle
layer onto the read-in layer (Ahat and m_perp). At N1 = inf the read-in law is its prior, and the middle layer is a
single layer of N2 erf neurons acting on a fixed Gaussian field: h_S ~ N(0, G_S) on the x0 tower, perp part
q_h ~ Gamma(shape M/2, mean qbar). That is the FCN2 read-in problem with
    w0 ~ N(0, sigma_w^2/d)            ->  h1 ~ N(0, G11)
    q ~ Gamma((d-1)/2, mean s0(d-1)/d) ->  q_h ~ Gamma(M/2, mean qbar)
    N, chi, kappa, P, sigma_A^2, y     ->  N2, chi, kappa, P, sigma_A^2, y   (identical readout/tilt structure:
                                            D = (sqrt(Chat) kappa/P) t, beta Chat kappa^2/P^2 = sigma_A^2 chi/(2 N2))
except that G_S also has G13, G33: the He3(x0) content of the infinite first layer's features (~1/d).

Three solutions per point:
  fcn3        FCN3 solver at N1 = inf (rB = 0)
  fcn3_he1    the same middle layer with G_S projected onto He1 (G13 = 0, G33 -> 1e-12 G11)
  fcn2        FCN2 q-theory (readin_action_theory_q) with the mapped prior above
fcn3_he1 vs fcn2 checks the two codes against each other (should agree to quadrature accuracy; both integrate
over the full distributions);
fcn3 vs fcn2 is the physical difference between a 3-layer net with an infinite first layer and a 2-layer net.

    python compare_fcn2_limit.py                      # thesis point d = 150, plus an N1 sweep
    python compare_fcn2_limit.py --dscan 75,150,300,600,1200
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from fcn3_fast import FastSolver
from fcn3_matrix_solver import Numerics, Params, solve_t, solve_t_convex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fcn2_collapsed_hmc"))
from readin_action_theory_q import KernelQ, solve_loop_q  # noqa: E402


class KernelQMapped(KernelQ):
    """KernelQ with an independent prior: w0 ~ N(0, v0), q ~ Gamma(shape k, mean qbar). Tilt beta = 1/(2 v0)."""

    def __init__(self, v0, k, qbar, wmax=None, nw=2401, nq=800, a=None, ks=(1, 3)):
        sd = math.sqrt(v0)
        wmax = wmax or 120 * sd
        a = a or 10 * sd
        # build on a dummy (d, s0) and then overwrite the prior-dependent pieces
        super().__init__(d=4, s0=1.0, wmax=wmax, nw=nw, nq=nq, a=a, ks=ks)
        self.beta = 1 / (2 * v0)
        self.q = np.geomspace(qbar * 1e-2, qbar * 4, nq)
        dq = np.diff(self.q); tq = np.zeros(nq); tq[:-1] += dq / 2; tq[1:] += dq / 2
        self.tq = tq
        from readin_action_theory_q import he_coeff
        W, Q = np.meshgrid(self.w, self.q, indexing="ij")
        self.C = np.stack([he_coeff(W, Q, kk) for kk in ks])
        self.c1, self.c3 = self.C[0], self.C[1]
        self.base = self.beta * W * W + (k / qbar) * Q - (k - 1) * np.log(Q)
        self.logm = np.log(np.outer(self.tw, tq))
        self.W, self.Q = W, Q


def fcn2_mapped(p: Params, G11, qbar, M):
    lam = p.kappa / (p.P * p.sigA ** 2)
    Chat = G11 * p.sigA ** 2 * p.chi * p.P ** 2 / (p.N2 * p.kappa ** 2)
    K = KernelQMapped(G11, M / 2, qbar, ks=p.ks)
    D, res = solve_loop_q(Chat, lam, p.eps, K)
    pr, _, J, _ = K.stats(D)
    y = p.y()
    fbar = J @ np.linalg.solve(J + lam * np.eye(len(y)), y)
    pw = pr.sum(1)
    dens = pw / K.tw
    nlp = -np.log(np.maximum(dens, 1e-300)); nlp -= nlp[0]
    i = np.argmax(np.diff(nlp) < 0) if np.any(np.diff(nlp) < 0) else None   # first maximum of -log p
    out = dict(L1=fbar[0], L3=fbar[1] / p.eps, fbar=fbar.tolist(), res=res,
               h1sq_over_G11=float(pw @ K.w ** 2 / G11), qh_over_qbar=float((pr * K.Q).sum() / qbar),
               barrier=float(nlp[i]) if i else None, hcut=float(K.w[i]) if i else None,
               mass_beyond_cut=float(pw[i:].sum()) if i else 0.0)
    return out, (K.w / math.sqrt(G11), nlp)


def fcn3_middle(s, G):
    """Solve the middle layer alone on a given prior field covariance G (rB = 0, so nothing feeds back)."""
    p = s.p
    st0 = s.last["state"]
    L = torch.linalg.cholesky(G)
    y = torch.tensor(p.y(), dtype=s.last["t"].dtype, device=s.last["t"].device)
    t = s.last["t"].clone()
    for _ in range(3):       # convex minimisation; the frame is rebuilt from the converged t each time
        t, st, _ = solve_t_convex(s.mid, p, L, st0["qbar"], st0["M"], t)
    fbar = (y - p.lam * t).cpu().numpy()
    Ehh = L @ st["Ezz"] @ L.T
    return dict(L1=fbar[0], L3=fbar[1] / p.eps, fbar=fbar.tolist(),
                h1sq_over_G11=float(Ehh[0, 0] / G[0, 0]), qh_over_qbar=float(st["Eq"]) / st0["qbar"]), None


def compare(p: Params, nu: Numerics, device):
    s = FastSolver(p, nu, device=device, verbose=False)
    r = s.solve()
    G = s.last["L"] @ s.last["L"].T
    st = s.last["state"]
    G11, qbar, M = float(G[0, 0]), st["qbar"], st["M"]
    Gp = torch.zeros_like(G); Gp[0, 0] = G[0, 0]
    for i in range(1, p.nS):
        Gp[i, i] = 1e-12 * G[0, 0]
    he1, prof_he1 = fcn3_middle(s, Gp)
    f2, prof_f2 = fcn2_mapped(p, G11, qbar, M)
    full = dict(L1=r["L1"], L3=r["L3"], fbar=r["fbar"], qh_over_qbar=r["qh_over_qbar"])
    Gn = G.cpu().numpy()
    field = dict(G11=G11, qbar=qbar, M=M, d_minus_1=p.d - 1, qbar_over_G11=qbar / G11,
                 sqrt_G33_over_G11=math.sqrt(Gn[1, 1] / Gn[0, 0]) if p.nS > 1 else 0.0,
                 rho13=Gn[0, 1] / math.sqrt(Gn[0, 0] * Gn[1, 1]) if p.nS > 1 else 0.0)
    return dict(field=field, fcn3=full, fcn3_he1=he1, fcn2=f2), dict(he1=prof_he1, fcn2=prof_f2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dscan", default="")
    ap.add_argument("--n1scan", default="1600,1e4,1e5,1e6,inf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="compare_fcn2_limit.json")
    a = ap.parse_args()
    res = dict(n1scan=[], dscan=[])
    for n1 in [float(x) for x in a.n1scan.split(",") if x]:
        s = FastSolver(Params(N1=n1), Numerics(), device=a.device, verbose=False)
        r = s.solve()
        row = dict(N1=n1, L1=r["L1"], L3=r["L3"], d_w0sq=r["d_w0sq"], readin_far_mass=r["readin_far_mass"])
        print("N1 sweep", row, flush=True)
        res["n1scan"].append(row)
    ds = [int(x) for x in a.dscan.split(",") if x] or [150]
    for d in ds:
        r, _ = compare(Params(d=d, N1=float("inf")), Numerics(), a.device)
        r["d"] = d
        res["dscan"].append(r)
        f = r["field"]
        print(f"\nd={d}: G11={f['G11']:.4e}  qbar/G11={f['qbar_over_G11']:.1f}  M={f['M']:.1f}  (d-1={d-1})  "
              f"sqrt(G33/G11)={f['sqrt_G33_over_G11']:.2e}  rho13={f['rho13']:.3f}")
        for k in ("fcn3", "fcn3_he1", "fcn2"):
            v = r[k]
            print(f"  {k:9s} L1={v['L1']:.7f}  L3={v['L3']:.7f}  <q_h>/qbar={v['qh_over_qbar']:.6f}  "
                  f"barrier={v.get('barrier')}" + (f"  d<h1^2>={v['h1sq_over_G11']:.5f}" if 'h1sq_over_G11' in v else ""))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
