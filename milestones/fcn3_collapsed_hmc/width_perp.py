#!/usr/bin/env python3
"""Does the input reach the network through directions other than x0, and does that shrink with width?
P = 750, kappa = 0.015, d = 150: N = 1500 (onset runs r10-r12) against N = 3000, 6000 (queue_wide.sh), last saved
network of each run, measured on its training set and on a fresh set of the same size.

  output        mean square of the other-direction part of the network's mean output f_bar (f_bar minus its x0
                tower to He21), as a fraction of the mean square of f_bar; the theory sets this part to zero
  pull          each hidden neuron's exact readout term 0.5 sa2 p^2 / (1 - l) with p = sum_mu phi_i(x_mu) alpha_mu,
                alpha = (y - f_bar) / v, split as in outer_he3_fresh.py into the part through x0 and the part through
                the neuron's other-direction part (its dependence on the other input directions); median over hidden
                neurons and over the 2 % with the largest |h . He3|. The theory keeps only the x0 part.
  direction     the perpendicular direction the hidden layer projects onto most (top right singular vector of the
                N2 x 149 matrix of hidden He1 projections onto each perpendicular coordinate, as plot_perp_projections)
                and its overlap with the training set's sum_mu y_mu x_perp,mu and sum_mu (y - f_bar)_mu x_perp,mu
                (a random direction overlaps ~ 0.08); its singular value relative to the same in an untrained network.

    PYTHONPATH=.:../fcn3_matrix_theory python width_perp.py
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from hidden_equation_test import e_of
from fcn3_sampler import make_data
from fcn3_matrix_solver import readin_features, hermite_norm

HERE = Path(__file__).resolve().parent
P, K = 750, 21
ks = tuple(range(1, K + 1, 2))
dev = torch.device("cuda:0")


def feat(X, W, B):
    return torch.cat([torch.erf(torch.erf(X[i:i + 4096] @ W.float().T) @ B.float().T)
                      for i in range(0, X.shape[0], 4096)]).double()


def perp_he1(W, B):
    """(N2, d) hidden He1 projections onto each input coordinate."""
    n2 = (W * W).sum(1, keepdim=True)
    C1 = readin_features(W, n2 - W * W, (1,))[0]                    # (N1, d)
    return B @ C1


def runs():
    out = [(1500, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_onset_d150_P{P}_N1500x1500_prior_r1*")))]
    for N in (3000, 6000):
        out += [(N, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_wide_d150_P{P}_N{N}x{N}_prior_r1*")))]
    return [(N, Path(r)) for N, r in out if (Path(r) / "summary.json").exists()]


@torch.no_grad()
def main():
    g = torch.Generator(device=dev).manual_seed(0)
    for N, run in runs():
        a = json.loads((run / "config.json").read_text())["args"]
        X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
        Xf, Yf, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, 12345, dev)
        X, Y, Xf, Yf = X.float(), Y.float(), Xf.float(), Yf.float()
        v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
        nt = torch.load(sorted((run / "nets").glob("net_*.pt"))[-1], map_location="cpu", weights_only=False)
        W = (nt["sw"] * nt["ZW"]).to(dev).double()
        B = (nt["sb"] * nt["ZB"]).to(dev).double()
        Phi = feat(X, W, B)
        M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
        Minv = torch.linalg.inv(M)
        ab = Minv @ (Phi.T @ Y.double() / v)
        lev = 1 - torch.diagonal(Minv) / sa2
        del M, Minv
        nrm = 1 + 2 * (W * W).sum(1)
        Jfull = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
        C = readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), ks)
        h = B @ C.T
        qh = ((B @ (Jfull - C.T @ C)) * B).sum(1)
        del Jfull
        e = e_of(h, qh, ks, dev)
        fbn = ab @ e
        top = (h[:, 1].abs() >= h[:, 1].abs().quantile(0.98)).cpu().numpy()
        line = [f"N={N} {run.name[-3:]}"]
        for lab, XX, YY, Ph in (("train", X, Y, Phi), ("fresh", Xf, Yf, None)):
            Ph = feat(XX, W, B) if Ph is None else Ph
            Hn = hermite_norm(ks, XX[:, 0].double())
            fb = Ph @ ab
            fx0 = fbn @ Hn
            fperp = fb - fx0
            alpha = (YY.double() - fb) / v
            dphi = Ph - (e @ Hn).T
            p_x0 = e @ (Hn @ alpha)
            p_o = dphi.T @ alpha
            gain = lambda p: (0.5 * sa2 * p ** 2 / (1 - lev)).cpu().numpy()
            gx, go = gain(p_x0), gain(p_o)
            line.append(f"{lab}: output other-dir share {float(fperp.pow(2).mean() / fb.pow(2).mean()):.4f}; "
                        f"pull x0 / other median {np.median(gx):.2f} / {np.median(go):.2f}, "
                        f"top 2% |h3| {np.median(gx[top]):.1f} / {np.median(go[top]):.1f}")
            if lab == "train":
                xp = X[:, 1:].double()
                vy = torch.nn.functional.normalize(xp.T @ Y.double(), dim=0)
                vr = torch.nn.functional.normalize(xp.T @ (Y.double() - fb), dim=0)
        Pc = perp_he1(W, B)[:, 1:]
        U, S, Vh = torch.linalg.svd(Pc, full_matrices=False)
        u = Vh[0]
        Zw = torch.randn(W.shape, generator=g, device=dev, dtype=torch.float64) * nt["sw"]
        Zb = torch.randn(B.shape, generator=g, device=dev, dtype=torch.float64) * nt["sb"]
        S0 = torch.linalg.svdvals(perp_he1(Zw, Zb)[:, 1:])
        line.append(f"strongest perp direction: singular value / untrained {float(S[0] / S0[0]):.2f} "
                    f"(2nd {float(S[1] / S0[1]):.2f}); overlap with sum y x_perp {float((u @ vy).abs()):.2f}, "
                    f"with sum residual x_perp {float((u @ vr).abs()):.2f}")
        print("\n   ".join(line), flush=True)
        del Phi, Pc, B, W
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
