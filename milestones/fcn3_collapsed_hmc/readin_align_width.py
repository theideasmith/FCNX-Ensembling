#!/usr/bin/env python3
"""Do the outer read-in neurons point their perpendicular weights the same way, and does that change with width?
P = 750, kappa = 0.015, d = 150, N = 1500 (onset r10-r12), 3000, 6000 (queue_wide.sh), every saved network after
burn-in. For read-in neuron j: u_j = w_j,perp / |w_j,perp| (149 components), s_j = sign(w_j0); outer = |w0| beyond the
theory's barrier top (plot_onset.barrier_cut), inner = |w0| < 0.3.

  count       outer read-in neurons per network
  pair cos    mean over pairs of s_j s_k u_j . u_k (0 for unrelated directions)
  to group    mean over j of s_j u_j . (sum of s_k u_k over the other outer neurons, normalised)
  to y, to r  mean over j of |u_j . v| with v the training set's sum_mu y_mu x_perp,mu or sum_mu (y - f_bar)_mu x_perp,mu
              (normalised; f_bar from the last saved network); unrelated directions give sqrt(2 / (pi 149)) = 0.065
  group dir   |overlap| of the outer neurons' common direction (normalised sum of s_j u_j) with those two vectors

    PYTHONPATH=.:../fcn3_matrix_theory python readin_align_width.py
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_sampler import make_data

HERE = Path(__file__).resolve().parent
P = 750
dev = torch.device("cuda:0")


def runs():
    out = [(1500, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_onset_d150_P{P}_N1500x1500_prior_r1*")))]
    for N in (3000, 6000):
        out += [(N, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_wide_d150_P{P}_N{N}x{N}_prior_r1*")))]
    return [(N, Path(r)) for N, r in out if (Path(r) / "summary.json").exists()]


def group_stats(u, s, m, vy, vr):
    U = u[m] * s[m, None]
    n = U.shape[0]
    G = U @ U.T
    pair = float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
    tot = U.sum(0)
    loo = torch.nn.functional.normalize(tot[None, :] - U, dim=1)
    to_group = float((U * loo).sum(1).mean())
    g = torch.nn.functional.normalize(tot, dim=0)
    return dict(n=n, pair=pair, to_group=to_group, to_y=float((u[m] @ vy).abs().mean()),
                to_r=float((u[m] @ vr).abs().mean()), gy=float((g @ vy).abs()), gr=float((g @ vr).abs()))


@torch.no_grad()
def main():
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    cut = po.barrier_cut(z, P)
    print(f"outer: |w0| > {cut:.2f};  unrelated directions: pair cos 0, |u . v| {math.sqrt(2 / (math.pi * 149)):.3f}")
    for N, run in runs():
        a = json.loads((run / "config.json").read_text())["args"]
        X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
        X, Y = X.double(), Y.double()
        v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
        nets = sorted((run / "nets").glob("net_*.pt"))
        nt = torch.load(nets[-1], map_location="cpu", weights_only=False)
        W = (nt["sw"] * nt["ZW"]).to(dev).double()
        B = (nt["sb"] * nt["ZB"]).to(dev).double()
        Phi = torch.erf(torch.erf(X @ W.T) @ B.T)
        M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
        fb = Phi @ torch.linalg.solve(M, Phi.T @ Y / v)
        vy = torch.nn.functional.normalize(X[:, 1:].T @ Y, dim=0)
        vr = torch.nn.functional.normalize(X[:, 1:].T @ (Y - fb), dim=0)
        del Phi, M
        res = {"outer": [], "inner": []}
        for f in nets:
            w = (lambda t: t["sw"] * t["ZW"])(torch.load(f, map_location="cpu", weights_only=False)).to(dev).double()
            w0 = w[:, 0]
            u = torch.nn.functional.normalize(w[:, 1:], dim=1)
            s = torch.sign(w0)
            res["outer"].append(group_stats(u, s, w0.abs() > cut, vy, vr))
            res["inner"].append(group_stats(u, s, w0.abs() < 0.3, vy, vr))
        line = f"N={N} {run.name[-3:]}"
        for gname, rs in res.items():
            mean = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
            line += (f" | {gname}: count {mean['n']:6.1f}, pair cos {mean['pair']:+.4f}, to group {mean['to_group']:+.3f}, "
                     f"to y {mean['to_y']:.3f}, to r {mean['to_r']:.3f}, group dir . y {mean['gy']:.2f}, . r {mean['gr']:.2f}")
        print(line, flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
