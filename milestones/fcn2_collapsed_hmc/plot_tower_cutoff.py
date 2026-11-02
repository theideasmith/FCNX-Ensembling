#!/usr/bin/env python3
"""
How many He_k(x0) modes the theory needs, at high vs low learnability: the κ0 = 0.015 ray (L1 ≈ 1.0, L3 ≈ 0.38)
against the κ0 = 0.1 ray of the grokking paper (L1 ≈ 0.88, L3 ≈ 0.22). One column per ray, colour = d (same log-d
scale as plot_large_d_actions.py).
  top:    theory learnabilities vs the cutoff He_{n_max}, as % deviation from the converged tower (He13);
          L3 solid, L1 dashed.
  bottom: the discrepancy spectrum |Δ_n| = |y_n - fbar_n| over the input distribution, n = 1..11: HMC (exact
          population coefficients of the posterior-mean predictor, filled) and the converged theory (open, line).
At high learnability Δ1 is nearly zero and the spurious He5, He7, ... content of fbar is the bulk of the
non-He3 discrepancy, so truncating the tower moves L3 much more.
Theory: per-neuron perp norm. Uses cuda:0 for HMC when available, otherwise the CPU.

    python plot_tower_cutoff.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chains import chain_fbar, mean_se
from diag_cavity_action import group_args, sample_files
from readin_action_theory import reduced_params
from readin_action_theory_q import KernelQ, solve_loop_q

RAYS = [(r"$\kappa_0 = 0.015$ ray (high learnability)", ["larged_d75", "pscan_d150_P3000", "mix_d300", "w0q_d600",
                                                                "w0q_d800", "w0q_d1200"]),
        (r"$\kappa_0 = 0.1$ ray, grokking paper (lower learnability)", ["grok_k0p1", "raypaper_b2", "raypaper_b8"])]
CUTOFFS = (3, 5, 7, 9, 11, 13)
MODES = (1, 3, 5, 7, 9, 11)
NORM = plt.Normalize(np.log2(75), np.log2(1200))


def colour(d):
    return plt.cm.viridis(0.9 * NORM(np.log2(d)))


def theory_fbar(a, kmax):
    """Posterior-mean He_k(x0) coefficients of the theory with the odd tower cut at He_kmax."""
    ks = tuple(range(1, kmax + 1, 2))
    _, _, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
    K = KernelQ(a["d"], a["s0"], ks=ks, nq=int(500 * max(1, np.sqrt(a["d"] / 600))))
    D, _ = solve_loop_q(Chat, lam, a["eps"], K)
    J = K.stats(D)[2]
    y = np.zeros(len(ks)); y[0], y[1] = 1.0, a["eps"]
    return J @ np.linalg.solve(J + lam * np.eye(len(ks)), y)


def load(groups):
    rows = []
    for g in groups:
        a = group_args(g)
        if a is None or not sample_files(g, 1):
            print(f"  {g}: no samples, skipped")
            continue
        fb = {k: theory_fbar(a, k) for k in CUTOFFS}
        fbc = chain_fbar(g, ks=MODES)
        y0 = np.zeros(len(MODES)); y0[0], y0[1] = 1.0, a["eps"]
        dh, edh = mean_se([np.abs(y0 - f) for f in fbc.values()])
        fh = mean_se(list(fbc.values()))[0]
        y = np.zeros(len(MODES)); y[0], y[1] = 1.0, a["eps"]
        rows.append(dict(d=a["d"], L1={k: f[0] for k, f in fb.items()}, L3={k: f[1] / a["eps"] for k, f in fb.items()},
                         dth=np.abs(y - fb[CUTOFFS[-1]][:len(MODES)]), dhmc=dh, edhmc=edh,
                         L1h=fh[0], L3h=fh[1] / a["eps"]))
        r = rows[-1]
        print(f"{g}: d={a['d']}  L3 theory He3 {r['L3'][3]:.4f} -> He13 {r['L3'][13]:.4f}, HMC {r['L3h']:.4f};  "
              f"|Δ| HMC {np.round(r['dhmc'], 4)}")
    return rows


def main():
    loaded = [(name, load(groups)) for name, groups in RAYS]
    fig, axes = plt.subplots(2, len(loaded), figsize=(7 * len(loaded), 9), sharey="row")
    for j, (name, rows) in enumerate(loaded):
        ax = axes[0, j]
        for r in rows:
            col = colour(r["d"])
            ax.plot(CUTOFFS, [100 * (r["L3"][k] / r["L3"][CUTOFFS[-1]] - 1) for k in CUTOFFS], "o-", color=col, lw=2)
            ax.plot(CUTOFFS, [100 * (r["L1"][k] / r["L1"][CUTOFFS[-1]] - 1) for k in CUTOFFS], "s--", color=col, lw=1.4,
                    ms=4)
        ax.axhline(0, color="0.6", lw=1)
        ax.set_xticks(CUTOFFS)
        ax.set_xticklabels([rf"He$_{{{k}}}$" for k in CUTOFFS])
        ax.set_xlabel(r"theory solved with modes He$_1$ ... He$_{n_{\max}}$")
        ax.set_title(name + "\n" + r"truncation error vs cutoff $n_{\max}$", fontsize=12)

        ax = axes[1, j]
        for r in rows:
            col = colour(r["d"])
            ax.errorbar(MODES, r["dhmc"], yerr=r["edhmc"], fmt="o", color=col, ms=7, zorder=3, elinewidth=1, capsize=2)
            ax.set_yscale("log")
            ax.semilogy(MODES, r["dth"], "o-", color=col, mfc="white", lw=1.2, ms=5, zorder=2)
        ax.set_xticks(MODES)
        ax.set_xticklabels([rf"He$_{{{k}}}$" for k in MODES])
        ax.set_xlabel(r"Hermite mode $n$ of the residual $y - \bar f$")
        ax.set_title(r"residual spectrum (one solution per $d$; theory with the full tower to He$_{13}$)", fontsize=11)
    axes[0, 0].set_ylabel(r"$L(\mathrm{He}_{n_{\max}}) \,/\, L(\mathrm{He}_{13}) - 1$  (%)")
    axes[1, 0].set_ylabel(r"per-mode discrepancy $|\Delta_n| = |y_n - \bar f_n|$")
    ds = sorted({r["d"] for _, rows in loaded for r in rows})
    hd = [axes[0, 0].plot([], [], "s", color=colour(d), label=rf"$d={d}$")[0] for d in ds]
    hk = [axes[0, 0].plot([], [], "o-", color="0.3", lw=2, label=r"$L_3$")[0],
          axes[0, 0].plot([], [], "s--", color="0.3", lw=1.4, ms=4, label=r"$L_1$")[0],
          axes[0, 0].plot([], [], "o", color="0.3", ms=7, label="HMC")[0],
          axes[0, 0].plot([], [], "o-", color="0.3", mfc="white", lw=1.2, ms=5, label=r"theory (full tower, He$_{13}$)")[0]]
    leg = fig.legend(handles=hd + hk, frameon=False, fontsize=10, loc="upper center", ncol=len(hd) + len(hk),
                     bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "tower_cutoff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
