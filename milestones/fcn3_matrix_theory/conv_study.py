"""Numerical convergence of fcn3_matrix_solver: double each quadrature knob at one point."""
import sys, json
import numpy as np
from fcn3_matrix_solver import FCN3Solver, Numerics, Params

nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
p = Params(nmax=nmax, kappa=kappa)
keys = ["d_w0sq", "L1", "L3", "qh_over_qbar", "M", "readin_barrier", "middle_barrier"]
def run(**kw):
    S = FCN3Solver(Params(nmax=nmax, kappa=kappa), Numerics(**kw), verbose=False)
    r = S.solve()
    H = np.array(r["H"])
    v = [r[k] for k in keys] + [H[0, 0], H[1, 1], H[0, 1]] + list(r["fbar"][2:])
    return np.array(v), r
base, r0 = run()
names = keys + ["H11", "H33", "H13"] + [f"fbar{k}" for k in p.ks[2:]]
print(f"S={p.ks} kappa={kappa}: {r0['walltime_s']:.2f}s, {r0['middle_passes']} passes, ESS={r0['middle_ess']:.4f}, "
      f"MS eig={np.round(r0['MS_eig'], 2).tolist()}, outer read-in well w0={r0['readin_w_outer']:.2f} nlp={r0['readin_nlp_outer']:.1f}")
print("base:", dict(zip(names, [float(f'{x:.6g}') for x in base])))
d = Numerics()
for kw in [dict(nqh=8), dict(nx=48), dict(ds=0.4), dict(nqh=32), dict(nx=128), dict(nqh=20), dict(zeta_levels=((1e-3, 48), (1e-5, 12), (1e-8, 6), (0.0, 4))), dict(ds=0.1), dict(nw=2401, nq=512),
           dict(nwc=320, nqc=80, nu=48), dict(smax=60)]:
    v, r = run(**kw)
    rel = np.abs(v - base) / np.maximum(np.abs(base), 1e-30)
    print(f"{str(kw):45s} max rel change {rel.max():.1e} ({names[int(rel.argmax())]})  {r['walltime_s']:.2f}s")
