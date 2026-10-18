#!/usr/bin/env python3
"""Plot decoupled He1–He2 Laplace V(w) (Q12=G12=0)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "action_h0_activation_plots/PureCubicChiN"
CSV = BASE / "he12_decoupled_actions.csv"
OUT = BASE / "he12_decoupled_action.png"


def main():
    pts: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    notes: dict[tuple[float, float], str] = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            key = (float(row["z"]), float(row["rho"]))
            pts[key].append((float(row["w"]), float(row["V"])))
            notes[key] = row["note"]
    zs = sorted({z for z, _ in pts})
    # pick a few rho for clarity
    want = [0.15, 0.10, 0.05, 0.03, 0.02]

    fig, axes = plt.subplots(1, len(zs), figsize=(4.2 * len(zs), 3.8))
    if len(zs) == 1:
        axes = [axes]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(want)))

    for ax, z in zip(axes, zs):
        for rho, c in zip(want, cmap):
            key = (z, rho)
            if key not in pts:
                continue
            arr = sorted(pts[key])
            w = np.array([p[0] for p in arr])
            V = np.array([p[1] for p in arr])
            i0 = int(np.argmin(np.abs(w)))
            V = V - V[i0]
            ax.plot(w, V, color=c, lw=1.7, label=rf"$\rho={rho:g}$ ({notes[key]})")
            for j in range(1, len(w) - 1):
                if V[j] < V[j - 1] and V[j] < V[j + 1]:
                    ax.plot(w[j], V[j], "o", color=c, ms=4, zorder=5)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_xlim(-3.5, 3.5)
        ax.set_xlabel(r"$w$")
        if z == 0:
            ax.set_title(rf"$z={z:g}$: $c_2\equiv 0$")
            ax.set_ylim(-2, 40)
        else:
            ax.set_title(rf"$z={z:g}$: decoupled He$_1$–He$_2$")
            ax.set_ylim(-30, 40)
        ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    axes[0].set_ylabel(r"$V(w)-V(0)$")
    fig.suptitle(
        r"Decoupled Laplace ($Q_{12}=G_{12}=0$), teacher $y=\varepsilon\,\mathrm{He}_2$",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
