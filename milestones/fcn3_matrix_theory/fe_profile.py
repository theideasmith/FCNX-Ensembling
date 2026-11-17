"""
Free-energy profile W*(log J33): at each fixed log J33, the other coordinates (log J11, atanh rho13, log T) are set
by Newton on their gradient components (stationary = maximum in those directions). Local maxima of the profile
in log J33 are the branches of (E1)-(E8); the largest W is the equilibrium. M is held at the value given.

    python fe_profile.py --d 150 --N1 1500 --M 164.1
"""
import argparse
import json
import math

import numpy as np
import torch

import ray_k0p015 as R
from fcn3_fast import FastSolver
from fcn3_free_energy import FreeEnergy


def constrained(F, x, M, fix, tol=1e-6, maxit=30):
    free = [i for i in range(len(x)) if i != fix]
    W, g = F._value_grad(x, M)
    if W is None:
        raise RuntimeError("infeasible start")
    for it in range(maxit):
        gf = g[free]
        if float(gf.abs().max()) < tol * float(F.p.N1):
            break
        H = torch.stack([F._fd_col(x, g, j, M)[free] for j in free], 1)
        H = 0.5 * (H + H.T)
        step = -torch.linalg.solve(H, gf)
        mx = float(step.abs().max())
        if mx > 0.5:
            step *= 0.5 / mx
        a = 1.0
        while a > 1e-4:
            xn = x.clone()
            xn[free] += a * step
            W1, g1 = F._value_grad(xn, M)
            if W1 is not None and float(g1[free].abs().max()) < float(gf.abs().max()):
                break
            a *= 0.5
        else:
            break
        x, W, g = xn, W1, g1
    return x, W, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--N1", type=float, default=1500)
    ap.add_argument("--M", type=float, default=164.1)
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=-4.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--start", default=None, help="json with x to start from (with --lo)")
    a = ap.parse_args()
    J11 = float(FastSolver(R.params(a.d, float("inf"), 1.0), verbose=False).state0["J"][0, 0])
    p = R.params(a.d, a.N1, math.sqrt((R.S0 / a.d) / J11))
    p.nmax = 3
    F = FreeEnergy(p, verbose=False)
    x = F.encode(F.J0, F.T0)
    lo = a.lo if a.lo is not None else float(x[1])
    if a.lo is not None and a.start:
        x = torch.tensor(json.load(open(a.start))["x"], dtype=x.dtype, device=x.device)
    rows = []
    v = lo
    xprev = x.clone()
    step = a.step
    while v <= a.hi + 1e-9:
        x = xprev.clone()
        x[1] = v
        try:
            x, W, g = constrained(F, x, a.M, 1)
        except RuntimeError as e:
            if step > a.step / 16:
                v -= step / 2
                step /= 2
                print(f"logJ33={v + step:.3f}: {e}; shorter step", flush=True)
                continue
            print(f"logJ33={v:.3f}: {e}; stopping", flush=True)
            break
        xprev = x.clone()
        h = F.last["h"]
        t = h["t"]
        fb = (F.y - p.lam * t).cpu().numpy()
        pw = F.last["pw"].reshape(F.ri.W.shape)
        far = float(pw[F.ri.W > 1.0].sum())
        rows.append(dict(logJ33=v, W=W, dW_dlogJ33=float(g[1]), grad_free=float(g[[0, 2, 3]].abs().max()),
                         L1=float(fb[0]), L3=float(fb[1] / p.eps), readin_far=far, x=x.cpu().numpy().tolist()))
        print(f"logJ33={v:7.2f}  W={W:12.4f}  dW/dlogJ33={float(g[1]):+10.3f}  |grad other|={rows[-1]['grad_free']:.1e}"
              f"  mu1={fb[0]:.5f} mu3={fb[1] / p.eps:.5f}  read-in mass |w0|>1: {far:.2e}", flush=True)
        v += step
        step = min(a.step, step * 2)
    json.dump(rows, open(f"fe_profile_d{a.d}_N1_{int(a.N1)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
