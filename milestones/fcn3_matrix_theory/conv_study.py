"""Numerical convergence of fcn3_matrix_solver: refine (or widen) each grid at one point and report the largest
relative change of the observables.

    python conv_study.py [nmax] [kappa]
"""
import sys

import numpy as np

from fcn3_matrix_solver import FCN3Solver, Numerics, Params

nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
p = Params(nmax=nmax, kappa=kappa)
keys = ["d_w0sq", "L1", "L3", "qh_over_qbar", "M"]


def run(init=None, **kw):
    S = FCN3Solver(Params(nmax=nmax, kappa=kappa), Numerics(**kw), verbose=False)
    r = S.solve(init)
    H = np.array(r["H"])
    far = r["readin_far_mass"][0] if r["readin_far_mass"] else 0.0
    v = [r[k] for k in keys] + [H[0, 0], H[1, 1], H[0, 1], far] + list(r["fbar"][2:])
    return np.array(v), r, S.state()


base, r0, st0 = run()
names = keys + ["H11", "H33", "H13", "readin_far_mass"] + [f"fbar{k}" for k in p.ks[2:]]
print(f"S={p.ks} kappa={kappa}: {r0['walltime_s']:.1f}s  far logits {r0['far_logits']}")
print("base:", dict(zip(names, [float(f'{x:.7g}') for x in base])))
for kw in [dict(ds=0.15), dict(nqh=320), dict(nx=128), dict(smax=60.0), dict(lqh_below=18.0, lqh_above=3.0),
           dict(zeta_levels=((1e-3, 48), (1e-5, 12), (1e-8, 6), (0.0, 4))),
           dict(nw=2401), dict(nq=2801), dict(lq_below=22.0, lq_above=2.5), dict(wmax=8.0),
           dict(pair_block=(3, 4)), dict(nu=48)]:
    v, r, _ = run(init=st0, **kw)
    rel = np.abs(v - base) / np.maximum(np.abs(base), 1e-30)
    print(f"{str(kw):60s} max rel change {rel.max():.1e} ({names[int(rel.argmax())]})  {r['walltime_s']:.1f}s", flush=True)
