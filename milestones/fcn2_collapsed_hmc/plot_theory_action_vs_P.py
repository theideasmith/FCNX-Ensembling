#!/usr/bin/env python3
"""
Theory-only read-in action -log p(w0) (normalized, readin_action_theory.py) as P grows toward
L3 → 1 at fixed d, N, σ_w², σ_A², κ, ε. Left: full range; right: zoom on the centre, where the
single well splits into an inner double well at high learnability.

    python plot_theory_action_vs_P.py --d 150 --N 1500 --kappa 0.015
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from readin_action_theory import Kernel, analyse, features, reduced_params, solve_loop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--s0", type=float, default=0.5)
    ap.add_argument("--sa0", type=float, default=None, help="default 8/N")
    ap.add_argument("--kappa", type=float, default=0.015)
    ap.add_argument("--eps", type=float, default=-0.3 * 6 ** 0.5)
    ap.add_argument("--Ps", type=float, nargs="+", default=[3e3, 6e3, 1e4, 3e4, 1e5, 1e6])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sa0 = args.sa0 if args.sa0 is not None else 8 / args.N

    w = np.linspace(-2.4, 2.4, 9601)
    norm = plt.Normalize(np.log10(min(args.Ps)), np.log10(max(args.Ps)))
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for P in args.Ps:
        T, beta, lam, Chat = reduced_params(args.N, args.d, args.s0, sa0, P, args.kappa)
        K = Kernel(T, beta)
        D, res = solve_loop(Chat, lam, args.eps, K)
        _, _, logZ, _, _ = K.stats(D)
        o = analyse(D, lam, args.eps, K)
        c1, c3 = features(np.abs(w), T)
        nlp = beta * (w * w - (D[0] * c1 + D[1] * c3) ** 2) + logZ + np.log(2.0)   # normalized on the full line
        col = plt.cm.viridis(0.9 * norm(np.log10(P)))
        lab = rf"$P={P:.0e}$: $L_3={o['L3']:.3f}$"
        ax.plot(w, nlp, color=col, lw=2, label=lab)
        axz.plot(w, nlp, color=col, lw=2)
        print(f"P={P:.0e}: L1={o['L1']:.4f} L3={o['L3']:.4f} D={np.round(D, 3)} residual={res:.1e}  "
              + ", ".join(f"{b['basin']}:{b['weight']:.3f}" for b in o["basins"]))
    ax.set_ylim(-3, 40)
    ax.set_xlim(w[0], w[-1])
    axz.set_xlim(-0.45, 0.45)
    axz.set_ylim(-3, 6)
    for a in (ax, axz):
        a.set_xlabel(r"$w_0$", fontsize=13)
        a.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_ylabel(r"$-\log p(w_0)$  shifted by log normalization", fontsize=13)
    axz.set_title("zoom on the centre", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    fig.suptitle(rf"theory only: $d={args.d}$, $N={args.N}$, $\sigma_w^2={args.s0:g}$, $\sigma_A^2={sa0:.3g}$, "
                 rf"$\kappa={args.kappa:g}$, $\epsilon={args.eps:.3g}$", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / f"theory_action_vs_P_d{args.d}_k{args.kappa:g}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
