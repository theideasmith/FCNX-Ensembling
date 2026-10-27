#!/usr/bin/env python3
"""
Effective drive coupling from collapsed-HMC samples, compared with the read-in theory's Ĉ.

For neuron i at a test value w0 (own w_perp kept), the exact single-neuron drive is
    drive_i(w0) = ½ σ_A² r_i² / (1 + G_i),   r_i = (2/T) φ_i(w0)·e_{-i}   (training set, cavity residual),
and the theory's drive is β (D·c(w0, T_i))² with D = √Ĉ Δ, Δ = (1 - L1, ε(1 - L3)) the population residual
of the HMC posterior-mean predictor (50k fresh test points) and T_i = 1 + 2|w_perp,i|² the neuron's own
perp norm. Their ratio, averaged over neurons, is Ĉ_eff/Ĉ (1 if the theory's coupling is right);
the same without the 1/(1+G_i) factor separates the self-reaction.

    python diag_effective_coupling.py --groups grok_k0p07 grok_k0p1 ...
"""

import argparse

import numpy as np
import torch

from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import cavity_pieces, group_args, sample_files
from neuron_gibbs import NeuronGibbs
from plot_ray_trend import hmc_stats
from readin_action_theory import features, reduced_params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--w0", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.3])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"{'group':>26s} {'d':>4s} {'N':>5s} {'P':>6s} {'κ':>6s} | {'Δ1':>7s} {'Δ3':>7s} | "
          + " ".join(f"{'w0=' + str(w):>12s}" for w in args.w0) + " | mean G")
    for g in args.groups:
        a = group_args(g)
        h = hmc_stats(g, a, 20, 50000, args.device)
        Delta = np.array([1 - h["L1"], a["eps"] * (1 - h["L3"])])
        _, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
        D = np.sqrt(Chat) * Delta
        X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
        ng = NeuronGibbs(X, Y, a["N"], a["T"], a["s0"], a["sa0"], a["chi"])
        w0g = torch.tensor(args.w0, device=dev, dtype=torch.float64)
        ratio, ratio_noG, Gs = [], [], []
        for f in sample_files(g, args.snapshots):
            W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
            G, R = cavity_pieces(ng, W, w0g, {"exact": None})
            meas = R["exact"].cpu().numpy()                          # (N, K) = ½σ_A² r²/(1+G)
            Gn = G.cpu().numpy()
            q = (W[:, 1:] ** 2).sum(1).cpu().numpy()
            T_i = 1 + 2 * q
            c1, c3 = features(np.abs(np.array(args.w0))[None, :], T_i[:, None])
            pred = beta * (D[0] * c1 + D[1] * c3 * np.sign(np.array(args.w0))[None, :] ** 3) ** 2
            ratio.append(meas.mean(0) / pred.mean(0))
            ratio_noG.append((meas * (1 + Gn)).mean(0) / pred.mean(0))
            Gs.append(Gn.mean())
        r, r0 = np.mean(ratio, 0), np.mean(ratio_noG, 0)
        print(f"{g:>26s} {a['d']:>4d} {a['N']:>5d} {a['P']:>6d} {a['T'] / 2:>6.3f} | {Delta[0]:>7.4f} {Delta[1]:>7.4f} | "
              + " ".join(f"{x:>5.3f}({y:>5.3f})" for x, y in zip(r, r0)) + f" | {np.mean(Gs):.3f}", flush=True)
    print("entries: Ĉ_eff/Ĉ with the exact 1/(1+G) factor (in parentheses: without it)")


if __name__ == "__main__":
    main()
