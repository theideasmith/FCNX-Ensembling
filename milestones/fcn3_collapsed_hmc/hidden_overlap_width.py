#!/usr/bin/env python3
"""Overlap between hidden units' perpendicular fields, split by degree in x_perp, against width.
Tests notes/perpendicular_term_fcn2_fcn3.tex: two hidden units' parts of degree m >= 2 in x_perp overlap at order
N1^(-1/2) for independent weight rows; the degree-1 part (the d - 1 directions of x_perp) is shared at any N.

Read-in kernel of the perpendicular parts (eq. Q2 of the note), with D_j = 1 + 2|w_j|^2 and u_jk = w_j,perp . w_k,perp:
  A_jk = (2/pi) [asin(2 (w_j0 w_k0 + u_jk) / sqrt(D_j D_k)) - asin(2 w_j0 w_k0 / sqrt(D_j D_k))]
Its degree-1 part in x_perp is the u-linear term A1_jk = (2/pi) 2 u_jk / sqrt(D_j D_k - 4 w_j0^2 w_k0^2); the rest,
A2 = A - A1, is degree >= 2. Hidden unit overlaps O^(m) = B A^(m) B^T, as cosines O_kl / sqrt(O_kk O_ll); reported as
the rms over pairs k != l. Same for untrained networks of the same size (ZW, ZB standard normal).
P = 750, kappa = 0.015, d = 150; last saved network of each run.

    PYTHONPATH=.:../fcn3_matrix_theory python hidden_overlap_width.py
"""
import glob
import math
from pathlib import Path

import numpy as np
import torch

from fcn3_matrix_solver import readin_features

HERE = Path(__file__).resolve().parent
P = 750
dev = torch.device("cuda:0")


def runs():
    out = [(1500, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_onset_d150_P{P}_N1500x1500_prior_r1*")))]
    for N in (3000, 6000):
        out += [(N, r) for r in sorted(glob.glob(str(HERE / "runs" / f"*_wide_d150_P{P}_N{N}x{N}_prior_r1*")))]
    return [(N, Path(r)) for N, r in out if (Path(r) / "summary.json").exists()]


def kernels(W):
    w0 = W[:, 0]
    D = 1 + 2 * (W * W).sum(1)
    sD = torch.sqrt(torch.outer(D, D))
    ww = torch.outer(w0, w0)
    u = W[:, 1:] @ W[:, 1:].T
    A = (2 / math.pi) * (torch.asin((2 * (ww + u) / sD).clamp(-1, 1)) - torch.asin(2 * ww / sD))
    A1 = (2 / math.pi) * 2 * u / torch.sqrt(sD ** 2 - 4 * ww ** 2)
    return A1, A - A1


def rms_cos(B, K, mask=None):
    O = B @ K @ B.T
    d = torch.sqrt(O.diagonal().clamp_min(1e-300))
    C = O / torch.outer(d, d)
    if mask is not None:
        C = C[mask][:, mask]
    n = C.shape[0]
    off = C[~torch.eye(n, dtype=torch.bool, device=C.device)]
    return float(off.pow(2).mean().sqrt())


@torch.no_grad()
def main():
    g = torch.Generator(device=dev).manual_seed(0)
    rows = {}
    for N, run in runs():
        nt = torch.load(sorted((run / "nets").glob("net_*.pt"))[-1], map_location="cpu", weights_only=False)
        W = (nt["sw"] * nt["ZW"]).to(dev).double()
        B = (nt["sb"] * nt["ZB"]).to(dev).double()
        A1, A2 = kernels(W)
        C = readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), (3,))[0]
        h3 = (B @ C).abs()
        top = h3 >= h3.quantile(0.98)
        r = [rms_cos(B, A1), rms_cos(B, A2), rms_cos(B, A1, top), rms_cos(B, A2, top)]
        W0 = torch.randn(W.shape, generator=g, device=dev, dtype=torch.float64) * nt["sw"]
        B0 = torch.randn(B.shape, generator=g, device=dev, dtype=torch.float64) * nt["sb"]
        P1, P2 = kernels(W0)
        r += [rms_cos(B0, P1), rms_cos(B0, P2)]
        rows.setdefault(N, []).append(r)
        print(f"N={N} {run.name[-3:]}: rms cos between hidden units, degree 1 / degree >= 2: all {r[0]:.4f} / {r[1]:.4f}, "
              f"top 2% |h3| {r[2]:.4f} / {r[3]:.4f};  untrained {r[4]:.4f} / {r[5]:.4f}", flush=True)
        del A1, A2, P1, P2
        torch.cuda.empty_cache()
    print("\nmean over datasets          degree 1            degree >= 2          top 2% |h3|: deg 1, deg >= 2")
    for N, rs in rows.items():
        m = np.mean(rs, 0)
        print(f"N={N}:  trained {m[0]:.4f}  untrained {m[4]:.4f}   trained {m[1]:.4f}  untrained {m[5]:.4f}"
              f"   {m[2]:.4f}, {m[3]:.4f}")
    Ns = sorted(rows)
    m = {N: np.mean(rows[N], 0) for N in Ns}
    print(f"ratio N={Ns[0]} -> {Ns[-1]} (N^-1/2 predicts {math.sqrt(Ns[0] / Ns[-1]):.2f}): degree 1 "
          f"{m[Ns[-1]][0] / m[Ns[0]][0]:.2f}, degree >= 2 {m[Ns[-1]][1] / m[Ns[0]][1]:.2f}, untrained degree >= 2 "
          f"{m[Ns[-1]][5] / m[Ns[0]][5]:.2f}")


if __name__ == "__main__":
    main()
