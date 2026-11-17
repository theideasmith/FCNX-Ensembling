#!/usr/bin/env python3
"""Do the outer read-in neurons share a perpendicular direction at large P, or do they point in unrelated directions?
d = 150, N1 = N2 = 1500, kappa = 0.015, onset runs (datasets 0-2), every saved network after burn-in.

If their directions are unrelated, the perpendicular part they put into the output is a sum of n_out random terms and
shrinks like 1/sqrt(n_out) (an effect of having few outer neurons). If they share a direction, it is a common field and
does not shrink with more neurons.

For read-in neuron j: u_j = w_j,perp / |w_j,perp| (149 components), s_j = sign(w_j0); outer = |w0| beyond the theory's
barrier top (plot_onset.barrier_cut), inner = |w0| < 0.3. Per network:
  pair cos     mean over pairs of s_j s_k u_j . u_k            (unrelated directions: 0)
  rms cos      root mean square over pairs of u_j . u_k         (unrelated: 1/sqrt(149) = 0.082)
  to group     mean over j of s_j u_j . (normalised sum of s_k u_k over the other outer neurons)
  top eig      largest eigenvalue of (1/n) sum_j u_j u_j^T, times 149 (no sign convention needed;
               unrelated directions: about (1 + sqrt(149/n))^2 for n neurons)
  to y, to r   mean over j of |u_j . v|, v = training set's sum_mu y_mu x_perp,mu or sum_mu (y - f_bar)_mu x_perp,mu
               (normalised; f_bar from the last saved network); unrelated: sqrt(2 / (pi 149)) = 0.065
  group . y/r  |overlap| of the outer neurons' common direction (normalised sum of s_j u_j) with those two vectors
Each statistic is also computed for the same number of random unit vectors (column "random").

    PYTHONPATH=.:../fcn3_matrix_theory python readin_align_P.py [--Ps 750 3000 24000 48000]
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_sampler import make_data

HERE = Path(__file__).resolve().parent
dev = torch.device("cuda:0")
F = torch.nn.functional


def stats(u, s, vy, vr):
    n = u.shape[0]
    U = u * s[:, None]
    G = U @ U.T
    off = ~torch.eye(n, dtype=torch.bool, device=u.device)
    tot = U.sum(0)
    loo = F.normalize(tot[None, :] - U, dim=1)
    g = F.normalize(tot, dim=0)
    ev = torch.linalg.eigvalsh(u.T @ u / n)[-1] * u.shape[1]
    return dict(n=n, pair=float(G[off].mean()), rms=float(G[off].pow(2).mean().sqrt()),
                to_group=float((U * loo).sum(1).mean()), top=float(ev),
                to_y=float((u @ vy).abs().mean()), to_r=float((u @ vr).abs().mean()),
                gy=float((g @ vy).abs()), gr=float((g @ vr).abs()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[750, 3000, 24000, 48000])
    a_ = ap.parse_args()
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    gen = torch.Generator(device=dev).manual_seed(0)
    keys = ["n", "pair", "rms", "to_group", "top", "to_y", "to_r", "gy", "gr"]
    print("columns: " + "  ".join(keys))
    for P in a_.Ps:
        cut = po.barrier_cut(z, P)
        print(f"\nP = {P}: outer |w0| > {cut:.3f}")
        for run in sorted(glob.glob(str(HERE / "runs" / f"*_onset_d150_P{P}_N1500x1500_prior_r1*"))):
            run = Path(run)
            nets = sorted((run / "nets").glob("net_*.pt"))
            if not nets:
                continue
            a = json.loads((run / "config.json").read_text())["args"]
            X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
            X, Y = X.double(), Y.double()
            v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
            nt = torch.load(nets[-1], map_location="cpu", weights_only=False)
            W = (nt["sw"] * nt["ZW"]).to(dev).double()
            B = (nt["sb"] * nt["ZB"]).to(dev).double()
            Phi = torch.erf(torch.erf(X @ W.T) @ B.T)
            M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
            fb = Phi @ torch.linalg.solve(M, Phi.T @ Y / v)
            vy = F.normalize(X[:, 1:].T @ Y, dim=0)
            vr = F.normalize(X[:, 1:].T @ (Y - fb), dim=0)
            del Phi, M, X
            res = {"outer": [], "random": [], "inner": []}
            for f in nets:
                t = torch.load(f, map_location="cpu", weights_only=False)
                w = (t["sw"] * t["ZW"]).to(dev).double()
                w0 = w[:, 0]
                u = F.normalize(w[:, 1:], dim=1)
                s = torch.sign(w0)
                mo = w0.abs() > cut
                res["outer"].append(stats(u[mo], s[mo], vy, vr))
                ur = F.normalize(torch.randn(int(mo.sum()), u.shape[1], generator=gen, device=dev,
                                             dtype=torch.float64), dim=1)
                res["random"].append(stats(ur, torch.ones(ur.shape[0], device=dev, dtype=torch.float64), vy, vr))
                res["inner"].append(stats(u[w0.abs() < 0.3], s[w0.abs() < 0.3], vy, vr))
            print(f"  dataset {a['seed']} ({len(nets)} networks)")
            for gname, rs in res.items():
                m = {k: np.mean([r[k] for r in rs]) for k in keys}
                print(f"    {gname:6s} n {m['n']:6.1f}  pair {m['pair']:+.4f}  rms {m['rms']:.4f}  to group "
                      f"{m['to_group']:+.3f}  top eig {m['top']:6.2f}  to y {m['to_y']:.3f}  to r {m['to_r']:.3f}  "
                      f"group.y {m['gy']:.2f}  group.r {m['gr']:.2f}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
