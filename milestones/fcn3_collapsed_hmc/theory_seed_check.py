#!/usr/bin/env python3
"""How well does the solver determine the read-in action? The same theory solved with different Monte Carlo seeds
(hidden-field samples), each from the default start (n_max = 3, then 11). For each solve, -log p(w0) on the full
read-in grid (no pruning, as plot_ray_readin_fcn2_vs_fcn3.full_readin_theory) and:
    L3, outer read-in neurons per network (beyond the barrier), inner-well and outer-well bottoms (|w0| and depth),
    barrier top (|w0| and height), far-well neurons per network (|w0| > 1.2), and the free energy W.
Also the cached solutions used in the plots (onset scan; ray d = 150 at P = 3000).

    PYTHONPATH=.:../fcn3_matrix_theory python theory_seed_check.py [--P 3000] [--seeds 0 1 2 3 4]
"""
import argparse
import gc
import math
from dataclasses import replace

import numpy as np
import torch

import plot_onset as po
from fcn3_matrix_solver import DT, trapz_weights

HERE = po.HERE


def solve(P, seed, device):
    import fcn3_fixed as F
    p = po._params(P)
    F3 = F.BlazingFreeEnergy(replace(p, nmax=3), device=device, verbose=False, seed=seed)
    o3 = F3.solve()
    Fn = F.BlazingFreeEnergy(p, device=device, verbose=False, seed=seed)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    with torch.no_grad():
        lw = (Fn.lp0.reshape(-1) - 0.5 * (Fn.theta_ref @ Fn.Tst)).reshape(Fn.ri.W.shape)
        lw = lw - torch.logsumexp(lw.reshape(-1), 0)
        lm = torch.logsumexp(lw, 1)
        nlp = -(lm - torch.log(trapz_weights(Fn.ri.w)) + math.log(0.5))
    w, nlp = Fn.ri.w.cpu().numpy(), nlp.cpu().numpy()
    out = dict(w=w, nlp=nlp, L3=float(o["L3"]), W=float(o["W"]))
    del F3, Fn
    gc.collect()
    torch.cuda.empty_cache()
    return out


def features(w, nlp):
    ko = (w > 0.4) & (w < 1.2)
    io = np.argmin(nlp[ko])
    wo, yo = w[ko][io], nlp[ko][io]
    kb = (w > 0.1) & (w < wo)
    ib = np.argmax(nlp[kb])
    wb, yb = w[kb][ib], nlp[kb][ib]
    ki = w < wb
    ii = np.argmin(nlp[ki])
    wi, yi = w[ki][ii], nlp[ki][ii]
    p = np.exp(-(nlp - nlp.min())) * np.gradient(w)
    p /= p.sum()
    return dict(wi=wi, wo=wo, depth=yo - yi, wb=wb, barrier=yb - yi, outer=1500 * p[w >= wb].sum(),
                far=1500 * p[w > 1.2].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--one", type=int, default=None, help="solve this seed only and cache it (one process per seed)")
    a = ap.parse_args()
    P = a.P
    cache = lambda s: HERE / "plots" / "cache" / f"theory_seed_P{P}_s{s}.npz"
    if a.one is not None:
        r = solve(P, a.one, a.device)
        np.savez(cache(a.one), **r)
        print(f"   solved seed {a.one}: L3 {r['L3']:.4f}", flush=True)
        return
    rows = []
    for s in a.seeds:
        if not cache(s).exists():
            continue
        r = dict(np.load(cache(s)))
        rows.append((f"seed {s}", float(r["L3"]), float(r["W"]), features(r["w"], r["nlp"])))
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
    k = dn > 1e-300
    rows.append(("cached onset scan", float(z[f"sc_{P}"][1]), float(z[f"W_{P}"]), features(w[k], -np.log(0.5 * dn[k]))))
    if P == 3000:
        import plot_ray_readin_fcn2_vs_fcn3 as pr
        wr, nr, L1r, L3r = pr.fcn3_theory()[150]
        rows.append(("cached ray d=150", L3r, float("nan"), features(wr, nr)))
    print(f"\nP = {P} (d = 150, kappa = 0.015); |w0| positions, depths/heights in nats relative to the inner-well bottom")
    print(f"   {'solve':18s} {'L3':>7s} {'W':>12s} {'inner':>6s} {'outer':>6s} {'outer depth':>11s} {'barrier at':>10s} "
          f"{'barrier height':>14s} {'outer n':>8s} {'far n':>6s}")
    for lab, L3, W, f in rows:
        print(f"   {lab:18s} {L3:7.4f} {W:12.3f} {f['wi']:6.3f} {f['wo']:6.3f} {f['depth']:11.2f} {f['wb']:10.3f} "
              f"{f['barrier']:14.2f} {f['outer']:8.1f} {f['far']:6.2f}")


if __name__ == "__main__":
    main()
