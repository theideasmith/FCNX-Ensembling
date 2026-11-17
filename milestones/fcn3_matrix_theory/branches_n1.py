"""
All solutions of (E1)-(E8) at one parameter point with finite N1, via the read-in far-basin logit.

At fixed ell_r (log mass ratio of the read-in far basin, imposed), every other equation is solved by Newton
(smooth: basins and quadrature frozen). Then g(ell_r) = log P(far)/P(near) [full distribution] - ell_r.
Every zero of g is a solution of the full equations; their number is the number of solutions.

    python branches_n1.py --d 150 --N1 1500 [--nmax 11]
"""
import argparse
import json
import math

import numpy as np
import torch

import ray_k0p015 as R
from fcn3_fast import FastSolver
from fcn3_matrix_solver import NoConvergence


def newton_sub(S, u, idx_fix, tol=1e-9, maxit=25):
    """Newton on all unknowns except idx_fix (held at u[idx_fix]); drops the residual row idx_fix."""
    n = len(u)
    keep = torch.tensor([i for i in range(n) if i != idx_fix], device=u.device)

    def Rs(v):
        w = u.clone()
        w[keep] = v
        r = S.residual(w)
        return None if r is None else r[keep]

    v = u[keep].clone()
    r = Rs(v)
    for it in range(maxit):
        if float(r.abs().max()) < tol:
            break
        J = torch.zeros(len(r), len(v), dtype=v.dtype, device=v.device)
        for j in range(len(v)):
            h = 1e-6 * (1 + abs(float(v[j])))
            e = v.clone()
            e[j] += h
            J[:, j] = (Rs(e) - r) / h
        step = -torch.linalg.lstsq(J, r[:, None], rcond=1e-14).solution[:, 0]
        a = 1.0
        while a > 0.02:
            r1 = Rs(v + a * step)
            if r1 is not None and float(r1.abs().max()) < float(r.abs().max()):
                break
            a *= 0.5
        else:
            raise NoConvergence(f"sub-Newton stalled at |R|={float(r.abs().max()):.1e}")
        v, r = v + a * step, r1
    else:
        raise NoConvergence(f"sub-Newton |R|={float(r.abs().max()):.1e}")
    w = u.clone()
    w[keep] = v
    full = S.residual(w)
    return w, float(full[idx_fix])              # g = delta - ell at the imposed ell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--N1", type=float, default=1500)
    ap.add_argument("--nmax", type=int, default=11)
    ap.add_argument("--lo", type=float, default=-25.0)
    ap.add_argument("--hi", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=0.5)
    a = ap.parse_args()
    J11 = float(FastSolver(R.params(a.d, float("inf"), 1.0), verbose=False).state0["J"][0, 0])
    p = R.params(a.d, a.N1, math.sqrt((R.S0 / a.d) / J11))
    p.nmax = a.nmax
    S = FastSolver(p, verbose=False)
    r0 = S.solve(refine=False)
    u0 = S.u.clone()
    mr = S.frame["mr"]
    print(f"reference solution: L1={r0['L1']:.5f} L3={r0['L3']:.5f} far logits {r0['far_logits']} (read-in groups {mr})",
          flush=True)
    if mr < 1:
        print("no read-in far basin at the reference solution")
        return
    i_r = S.nx + S.S                                   # first read-in far logit
    rows = []
    for direction in (-1, +1):
        u = u0.clone()
        ell = float(u0[i_r])
        while a.lo <= ell <= a.hi:
            u[i_r] = ell
            try:
                u, g = newton_sub(S, u, i_r)
            except (NoConvergence, torch.linalg.LinAlgError) as e:
                # rebuild the frozen basins/cells at the current point (keep the logits) and retry once
                try:
                    n0 = S.nx + S.S
                    u_prev = u.clone()
                    u = S.build_frame(u_prev[:n0], u_prev[n0:].cpu().numpy())
                    if S.frame["mr"] < 1:
                        print(f"  ell_r={ell:7.2f}: read-in far basin gone after rebuild", flush=True)
                        break
                    u[i_r] = ell
                    u, g = newton_sub(S, u, i_r)
                    print(f"  (rebuilt quadrature at ell_r={ell:.2f}: groups r/h {S.frame['mr']}/{S.frame['mh']})")
                except (NoConvergence, torch.linalg.LinAlgError) as e2:
                    print(f"  ell_r={ell:7.2f}: {e}; after rebuild: {e2}", flush=True)
                    break
            y = S.y
            fb = (y - p.lam * S.last["t"]).cpu().numpy()
            rows.append(dict(ell=ell, g=g, L1=float(fb[0]), L3=float(fb[1] / p.eps),
                             others=u[i_r + 1:].cpu().numpy().tolist()))
            print(f"  ell_r={ell:7.2f}  g={g:+.4e}  L1={fb[0]:.5f} L3={fb[1] / p.eps:.5f}", flush=True)
            ell += direction * a.step
    rows.sort(key=lambda r: r["ell"])
    roots = [(r1["ell"], r2["ell"]) for r1, r2 in zip(rows, rows[1:]) if (r1["g"] > 0) != (r2["g"] > 0)]
    print("sign changes of g between ell_r =", roots)
    json.dump(dict(rows=rows, roots=roots), open(f"branches_d{a.d}_N1_{int(a.N1)}_n{a.nmax}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
