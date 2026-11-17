"""Exactness test of lam_count.py on the tiny network of test_count_move.py (d=4, P=40, N1=6, N2=4), reference sampler.

On the tiny network neuron j switches identity (r^2 above/below 0.1) at every lam, so the lam = 1 window gives
P_1(out)/P_1(in) directly. The estimator of lam_count.py (anchor at lam0 + identity-resolved MBAR along lam) must
reproduce it.     python test_lam.py   (CPU, windows in parallel, ~5 min)
"""
import math
import multiprocessing as mp

import numpy as np
import torch

from lam_count import GRID, Lam, identity_free_energies, run_window
from fcn3_sampler import FCN3
from test_count_move import make

THR = 0.1
J = 0


def model():
    m0 = make(0)
    return type("LamTiny", (Lam, FCN3), {})(m0.X, m0.Y, m0.N1, m0.N2, chi=1.0, kappa=0.3, sigA=1.0, sigB=1.0,
                                            sigw=1.0, ks=(1, 3), seed=1)


def window(args):
    lam, iters, seed = args
    torch.set_num_threads(1)
    m = model()
    m.gen.manual_seed(seed); m.np_rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    ZW, ZB = torch.randn(m.N1, m.d, generator=g), torch.randn(m.N2, m.N1, generator=g)
    m.set_lam(J, lam)
    rec = []
    run_window(m, ZW, ZB, 0.15, iters, iters // 10, leap=8, sweep_every=0, rec_every=2, rec=rec)
    x = np.array(rec)
    return dict(lam=lam, r2=x[:, 0], zeta=x[:, 1], m=x[:, 2], u=x[:, 3:])


def main(iters=6000):
    with mp.get_context("spawn").Pool(len(GRID)) as pool:
        wins = pool.map(window, [(l, iters, 11 + i) for i, l in enumerate(GRID)])
    print("P(out) by lam:", " ".join(f"{w['lam']:.2f}:{np.mean(w['r2'] > THR):.3f}" for w in wins))
    direct = [w for w in wins if w["lam"] == 1.0][0]
    p1 = np.mean(direct["r2"] > THR)
    print(f"direct, lam = 1 window: P(out) = {p1:.4f}  ->  log odds {math.log(p1 / (1 - p1)):+.3f}")
    for drop_top in (False, True):
        # drop_top: estimate without using the lam >= 0.6 windows' identity switching (as at d = 150)
        use = wins
        Fo = identity_free_energies(use, GRID, [w["r2"] > THR for w in use])[0]
        Fi = identity_free_energies(use, GRID, [w["r2"] <= THR for w in use])[0]
        for l0 in (0.0, 0.2, 0.4):
            w0 = [w for w in wins if w["lam"] == l0][0]
            po = np.mean(w0["r2"] > THR)
            k0 = GRID.index(l0)
            logR = math.log(po / (1 - po)) - (0 - Fo[k0]) + (0 - Fi[k0])
            print(f"  estimator, anchor lam0 = {l0}: log odds {logR:+.3f}")
        break




# ---------------------------------------------------------------------------------------------- confined windows
U0 = None
ZTHR = 0.5


def window_conf(args):
    lam, side, iters, seed = args
    torch.set_num_threads(1)
    m = model()
    m.gen.manual_seed(seed); m.np_rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    ZW, ZB = torch.randn(m.N1, m.d, generator=g), torch.randn(m.N2, m.N1, generator=g)
    u0 = torch.tensor([1.0, 0.5, -0.3, 0.2], dtype=torch.float64)
    m.set_lam(J, lam)
    if side != "none":
        m.set_conf(+1 if side == "out" else -1, ZTHR, u0)
        # start inside the region
        uh = (u0 / u0.norm()).float()
        for _ in range(1000):
            if m.in_region(ZW, ZB):
                break
            ZB[:, J] = torch.randn(m.N2, generator=g) + (2.0 if side == "out" else -2.0) * uh * torch.sign(ZW[J, 0])
    else:
        m.set_conf(0, ZTHR, u0)
    rec = []
    run_window(m, ZW, ZB, 0.15, iters, iters // 10, leap=8, sweep_every=0, rec_every=2, rec=rec)
    x = np.array(rec)
    zs = x[:, 1]
    return dict(lam=lam, side=side, zeta=zs, r2=x[:, 0], u=x[:, 3:])


def main_conf(iters=6000):
    from scipy.special import betainc
    jobs = [(l, s, iters, 100 + i) for i, (l, s) in enumerate([(l, s) for s in ("out", "in") for l in GRID])]
    jobs.append((1.0, "none", 4 * iters, 999))
    with mp.get_context("spawn").Pool(min(len(jobs), 24)) as pool:
        res = pool.map(window_conf, jobs)
    free = [r for r in res if r["side"] == "none"][0]
    pf = np.mean(free["zeta"] > ZTHR)
    print(f"unconfined lam = 1: P(zeta > {ZTHR}) = {pf:.4f}, log odds {math.log(pf / (1 - pf)):+.3f}")
    F = {}
    for X in ("out", "in"):
        ws = [r for r in res if r["side"] == X]
        F[X] = identity_free_energies(ws, GRID, [np.ones(len(r["zeta"]), bool) for r in ws])[0]
    N2 = 4
    p0 = 0.5 * betainc((N2 - 1) / 2, 0.5, 1 - ZTHR ** 2)
    lo0 = math.log(p0 / (1 - p0))
    est = lo0 + F["out"][0] - F["in"][0]
    print(f"lam = 0 analytic log odds {lo0:+.3f};  estimator (confined windows) log odds at lam = 1: {est:+.3f}")
    viol = [float(np.mean((r['zeta'] > ZTHR) != (r['side'] == 'out'))) for r in res if r['side'] != 'none']
    print(f"fraction of recorded samples outside their region (must be 0): {max(viol):.3f}")


if __name__ == "__main__":
    import sys
    main_conf() if "--conf" in sys.argv else main()
