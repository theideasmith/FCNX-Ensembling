"""
FCN3 theory (E1)-(E8) along the kappa0 = 0.015 ray of milestones/fcn2_collapsed_hmc, compared with the FCN2
read-in theory (readin_action_theory_q, He1..He11) and the converged collapsed-HMC results of FINDINGS.md.

Ray (FINDINGS.md): N = 1500, P = 3000 (d/150)^2, kappa = 0.015 (d/150), sigma_A^2 = sa0 = (8/1500)(150/d),
chi = 1, sigma_w^2 = 0.5, eps = -0.3 sqrt(6), target He1 + eps He3. The FCN3 network gets N2 = N hidden units.

FCN3 has one more prior parameter, sigma_B. It is set so that the hidden field's prior He1 variance equals the
FCN2 read-in prior: G11 = sigma_B^2 J11(prior) = sigma_w^2 / d. Then FCN3's h1 plays the role of FCN2's w0,
and H11 is compared with FCN2's <w0^2>.

Per d:
  fcn3_inf     FCN3, N1 = inf (read-in fixed at its prior)
  fcn3_N1      FCN3, N1 = N2 = 1500
  fcn3_he1     FCN3 at N1 = inf with the field projected on He1 (G13 = G33 = ... = 0)
  fcn2_mapped  FCN2 theory with FCN3's N1 = inf prior (G11, qbar_h, M)  [should equal fcn3_he1]
  fcn2         FCN2 theory on the ray (prior w0 ~ N(0, sigma_w^2/d), q ~ chi^2_{d-1})
  hmc          FCN2 exact posterior (FINDINGS.md, headline table)

    python ray_k0p015.py [--ds 75,150,300,600,800,1200] [--out ray_k0p015.json]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from compare_fcn2_limit import fcn2_mapped, fcn3_middle
from fcn3_fast import FastSolver
from fcn3_matrix_solver import Numerics, Params, solve_t_convex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fcn2_collapsed_hmc"))
from readin_action_theory_q import analyse_q  # noqa: E402

S0, N, EPS = 0.5, 1500, -0.3 * math.sqrt(6)
KS = (1, 3, 5, 7, 9, 11)
# FINDINGS.md headline table (converged chains): d -> (<w0^2>, L1, L3)
HMC = {75: (0.0381, 1.051, 0.318), 150: (0.0382, 1.025, 0.370), 300: (0.0360, 1.023, 0.376),
       600: (0.0354, 1.013, 0.381), 800: (0.0353, 1.012, 0.382), 1200: (0.0345, 1.010, 0.382)}


def ray(d):
    return dict(P=3000.0 * (d / 150) ** 2, kappa=0.015 * (d / 150), sa0=(8 / 1500) * (150 / d))


def params(d, N1, sigB):
    r = ray(d)
    return Params(d=d, P=r["P"], N1=N1, N2=N, chi=1.0, kappa=r["kappa"], eps=EPS, sigA=math.sqrt(r["sa0"]),
                  sigB=sigB, sigw=math.sqrt(S0), nmax=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="75,150,300,600,800,1200")
    ap.add_argument("--out", default="ray_k0p015.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = []
    for d in [int(x) for x in a.ds.split(",")]:
        t0 = time.time()
        # sigma_B from the prior J11
        J11 = float(FastSolver(params(d, float("inf"), 1.0), device=a.device, verbose=False).state0["J"][0, 0])
        sigB = math.sqrt((S0 / d) / J11)
        row = dict(d=d, sigB=sigB, **ray(d))
        res = {}
        # N1 = inf: the read-in layer is at its prior, G is fixed, and t is the minimum of a convex potential.
        # Frame from the converged t, then the transverse rule doubled until mu_1, mu_3 change < 1e-6.
        p_inf = params(d, float("inf"), sigB)
        t1 = time.time()
        tinf, lv, prev = None, Numerics().zeta_levels, None
        for k in range(5):
            S = FastSolver(p_inf, Numerics(zeta_levels=lv), device=a.device, verbose=False)
            st = S.state0
            G = p_inf.sigB ** 2 * st["J"]
            L = torch.linalg.cholesky(G)
            y = torch.tensor(p_inf.y(), dtype=G.dtype, device=G.device)
            t = tinf if tinf is not None else y / p_inf.lam
            for _ in range(3):
                t, stinf, its = solve_t_convex(S.mid, p_inf, L, st["qbar"], st["M"], t)
            tinf = t
            fb = (y - p_inf.lam * tinf).cpu().numpy()
            cur = (float(fb[0]), float(fb[1] / p_inf.eps))
            if prev is not None and max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1])) < 1e-6:
                break
            prev = cur
            lv = tuple((thr, 2 * n) for thr, n in lv)
        Hinf = L @ stinf["Ezz"] @ L.T
        res["fcn3_inf"] = dict(L1=cur[0], L3=cur[1], H11=float(Hinf[0, 0]), zeta_counts=S.mid.counts,
                               time=time.time() - t1)
        S.last = dict(t=tinf, L=L, state=st)
        Gp = torch.zeros_like(G)
        Gp[0, 0] = G[0, 0]
        for i in range(1, S.S):
            Gp[i, i] = 1e-12 * G[0, 0]
        he1, _ = fcn3_middle(S, Gp)
        res["fcn3_he1"] = dict(L1=float(he1["L1"]), L3=float(he1["L3"]), H11=he1["h1sq_over_G11"] * float(G[0, 0]))
        f2m, _ = fcn2_mapped(p_inf, float(G[0, 0]), st["qbar"], st["M"])
        res["fcn2_mapped"] = dict(L1=float(f2m["L1"]), L3=float(f2m["L3"]), H11=f2m["h1sq_over_G11"] * float(G[0, 0]))
        Gn = G.cpu().numpy()
        row["field"] = dict(G11=float(Gn[0, 0]), G11_fcn2=S0 / d, qbar=st["qbar"], qbar_fcn2=S0 * (d - 1) / d,
                            M=st["M"], d_minus_1=d - 1, sqrt_G33_over_G11=math.sqrt(Gn[1, 1] / Gn[0, 0]),
                            rho13=Gn[0, 1] / math.sqrt(Gn[0, 0] * Gn[1, 1]))
        S2 = FastSolver(params(d, float(N), sigB), device=a.device, verbose=False)
        r2 = S2.solve()
        res["fcn3_N1"] = dict(L1=r2["L1"], L3=r2["L3"], H11=r2["H"][0][0], readin_far=r2["readin_far_mass"],
                              hidden_far=r2["hidden_far_mass"], d_w0sq=r2["d_w0sq"], time=r2["walltime_s"])
        rr = ray(d)
        q = analyse_q(N, d, S0, rr["sa0"], rr["P"], rr["kappa"], EPS, ks=KS, nq=int(500 * max(1, np.sqrt(d / 600))))
        res["fcn2"] = dict(L1=float(q["L1"]), L3=float(q["L3"]), H11=float(q["w0_sq"]))
        if d in HMC:
            w2, l1, l3 = HMC[d]
            res["hmc"] = dict(L1=l1, L3=l3, H11=w2)
        row["res"] = res
        out.append(row)
        f = row["field"]
        print(f"\nd={d}  sigma_B={sigB:.4f}  ({time.time() - t0:.1f}s)   field: G11 d/s0={f['G11'] * d / S0:.4f}  "
              f"qbar/qbar2={f['qbar'] / f['qbar_fcn2']:.4f}  M={f['M']:.1f} (d-1={d - 1})  "
              f"sqrt(G33/G11)={f['sqrt_G33_over_G11']:.2e}  rho13={f['rho13']:.3f}")
        for k in ("fcn3_inf", "fcn3_N1", "fcn3_he1", "fcn2_mapped", "fcn2", "hmc"):
            if k in res:
                v = res[k]
                extra = ""
                if "readin_far" in v:
                    extra = f"  read-in far mass {v['readin_far']}"
                print(f"  {k:12s} L1={v['L1']:.5f}  L3={v['L3']:.5f}  <h1^2>={v['H11']:.5f}{extra}", flush=True)
    with open(Path(__file__).resolve().parent / a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
