#!/usr/bin/env python3
"""
Read-in action -log p(w0) along the large-d sequence (launch_large_d.sh; d=150 is pscan_d150_P3000):
all on one axis, coloured by d (viridis): HMC histogram (points), original theory (frozen perp norm, He1-He3;
dashed) and corrected theory (per-neuron perp norm + He1..He7 of x0; solid). Every curve is a normalized density on the full line,
shifted only by its own log normalization.
CPU only (samples and theory; no training data needed).

    python plot_large_d_actions.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_readin_action import load_group, theory_curve
from plot_readin_action_sweep import hist_nlp
from readin_action_theory_q import analyse_q

GROUPS = ["pscan_d150_P3000", "larged_d300", "larged_d600"]


def main():
    rows = []
    for g in GROUPS:
        loaded = load_group(g, live=True)
        if loaded is None:
            continue
        W, a, _ = loaded
        d = a["d"]
        w = np.linspace(-1.9, 1.9, 3801)
        nlp_orig, _ = theory_curve(a, 1 + 2 * a["s0"] * (d - 1) / d, w)
        q = analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], ks=(1, 3, 5, 7))
        wq = np.concatenate([-q["w"][::-1], q["w"][1:]])
        nq = np.concatenate([q["nlp"][::-1], q["nlp"][1:]])
        m = np.abs(wq) <= 1.9
        rows.append(dict(d=d, P=a["P"], w0=W[:, 0], w=w, orig=nlp_orig, wq=wq[m], corr=nq[m]))
        print(f"d={d}: {len(W) // a['N']} snapshots")

    norm = plt.Normalize(np.log2(150), np.log2(600))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    edges = np.linspace(-1.9, 1.9, 161)
    for r in rows:
        col = plt.cm.viridis(0.9 * norm(np.log2(r["d"])))
        ctr, h, err = hist_nlp(r["w0"], edges)
        ax.errorbar(ctr, h, yerr=err, fmt="o", ms=3, color=col, elinewidth=0.6, mec="white", mew=0.4, zorder=3)
        ax.plot(r["wq"], r["corr"], "-", color=col, lw=2.2, zorder=2)
        ax.plot(r["w"], r["orig"], "--", color=col, lw=1.4, alpha=0.8, zorder=1)
    for r in rows:
        ax.plot([], [], "s", color=plt.cm.viridis(0.9 * norm(np.log2(r["d"]))), label=rf"$d={r['d']}$, $P={r['P']}$")
    ax.plot([], [], "o", color="0.3", label="HMC")
    ax.plot([], [], "-", color="0.3", lw=2.2, label="corrected theory (relaxed perp norm + He1..He7)")
    ax.plot([], [], "--", color="0.3", lw=1.4, label="original theory (frozen perp norm, He1-He3)")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-3, 14)
    ax.set_xlabel(r"$w_0$", fontsize=13)
    ax.set_ylabel(r"$-\log p(w_0)$  shifted by log normalization", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
    ax.set_title(r"large-$d$ limit at fixed theory point ($\lambda=9.375\times10^{-4}$, $\hat C=474.1$; "
                 r"$N=1500$, $P\propto d^2$, $\epsilon=-0.3\sqrt{6}$, $\sigma_w^2=0.5$)", fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "large_d_actions.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
