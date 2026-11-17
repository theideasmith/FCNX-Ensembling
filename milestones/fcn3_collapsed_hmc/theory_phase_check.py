#!/usr/bin/env python3
"""Is the FCN3 theory at small P (d = 150, kappa = 0.015, the onset line of queue_onset.sh) stuck on a local maximum of
its free energy W?  Two solutions compete near P ~ 750-1000: one where a group of hidden neurons sits far out in He1
(no outer read-in well) and one with an outer read-in well.  Each P is solved from the default start (solve_fcn3: n_max = 3
from the prior, then n_max = 11) and warm-started (n_max = 11) from the read-in distribution of the solution at a
reference P on each side.  The equilibrium is the solution with the LARGEST W (fcn3_fixed: "the equilibrium is the
global maximum of W").

    python theory_phase_check.py [--Ps 500 750 1000 1500] [--refs 500 1500]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
import fcn3_fixed as F  # noqa: E402
from fcn3_matrix_solver import DT, Params  # noqa: E402

D, N, KAPPA, SIGB, SIGW2 = 150, 1500, 0.015, 1.25540121081682, 0.5


def params(P, nmax=11):
    return Params(d=D, P=float(P), N1=N, N2=N, chi=1.0, kappa=KAPPA, eps=-0.3 * math.sqrt(6), sigA=math.sqrt(8 / 1500),
                  sigB=SIGB, sigw=math.sqrt(SIGW2), nmax=nmax)


def describe(Fobj, o):
    """W, L3, outer read-in neurons (|w0| > 0.6, the barrier at these P), far hidden neurons
    (|h1| > 0.8), both per 1500."""
    pw = Fobj.last["pw"].reshape(Fobj.ri.W.shape).detach().cpu().numpy()
    w0 = Fobj.ri.W.cpu().numpy()
    with torch.no_grad():
        p = Fobj.p
        L = F.scaled_cholesky(p.sigB ** 2 * Fobj.last["J"])
        e, h, _ = Fobj.mc.features(L, p.sigB ** 2 * Fobj.last["T"])
        lt = Fobj.mc.logw(o["M"]) + 0.5 * p.gt * (e @ Fobj.t) ** 2
        w = torch.softmax(lt, 0)
        far = float(w[h[:, 0].abs() > 0.8].sum())
    return dict(W=o["W"], L3=o["L3"], outer=N * float(pw[np.abs(w0) > 0.6].sum() / pw.sum()), far=N * far,
                theta=torch.tensor(o["theta"], dtype=DT), M=o["M"])


def solve_default(P, dev):
    p3 = params(P, 3)
    F3 = F.BlazingFreeEnergy(p3, device=dev)
    o3 = F3.solve()
    Fn = F.BlazingFreeEnergy(params(P), device=dev)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    return describe(Fn, o)


def solve_from(P, ref, dev):
    Fn = F.BlazingFreeEnergy(params(P), device=dev)
    o = Fn.solve(theta0=ref["theta"].to(Fn.dev), M0=ref["M"])
    return describe(Fn, o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[500, 750, 1000, 1500])
    ap.add_argument("--refs", type=int, nargs="+", default=[500, 1500])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    fmt = lambda r: (f"W = {r['W']:.3f}  L3 = {r['L3']:.4f}  outer read-in = {r['outer']:5.1f}  "
                     f"far hidden = {r['far']:5.1f}")
    base = {}
    for P in sorted(set(a.Ps) | set(a.refs)):
        base[P] = solve_default(P, a.device)
        print(f"P = {P:5d}  default start          : {fmt(base[P])}", flush=True)
    for P in a.Ps:
        for R in a.refs:
            if R == P:
                continue
            try:
                r = solve_from(P, base[R], a.device)
                dW = r["W"] - base[P]["W"]
                print(f"P = {P:5d}  start from P = {R:5d} : {fmt(r)}   W - W(default) = {dW:+.3f}", flush=True)
            except Exception as e:
                print(f"P = {P:5d}  start from P = {R:5d} : failed ({e!r})"[:200], flush=True)


if __name__ == "__main__":
    main()
