#!/usr/bin/env python3
"""Plot V(w) for teacher y(x)=ε He₂(t̂·x) with biased-erf student features.

Reads he2_target_actions.csv from probe_he2_target_teacher.jl.
"""
from __future__ import annotations

from pathlib import Path

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "action_h0_activation_plots/PureCubicChiN"
CSV = BASE / "he2_target_actions.csv"
OUT = BASE / "he2_target_teacher_action.png"


def load_actions(path: Path):
    # (z, rho) -> list[(w,V)], note
    pts: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    notes: dict[tuple[float, float], str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (float(row["z"]), float(row["rho"]))
            pts[key].append((float(row["w"]), float(row["V"])))
            notes[key] = row["note"]
    return pts, notes


def main():
    pts, notes = load_actions(CSV)
    zs = sorted({z for z, _ in pts})
    rhos = sorted({r for _, r in pts}, reverse=True)

    fig, axes = plt.subplots(1, len(zs), figsize=(4.2 * len(zs), 3.8), sharey=False)
    if len(zs) == 1:
        axes = [axes]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(rhos)))

    for ax, z in zip(axes, zs):
        for rho, c in zip(rhos, cmap):
            key = (z, rho)
            if key not in pts:
                continue
            arr = sorted(pts[key])
            w = np.array([p[0] for p in arr])
            V = np.array([p[1] for p in arr])
            i0 = int(np.argmin(np.abs(w)))
            V = V - V[i0]
            note = notes[key]
            ax.plot(w, V, color=c, lw=1.7, label=rf"$\rho={rho:g}$ ({note})")
            for j in range(1, len(w) - 1):
                if V[j] < V[j - 1] and V[j] < V[j + 1]:
                    ax.plot(w[j], V[j], "o", color=c, ms=4.5, zorder=5)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.set_xlim(-3.5, 3.5)
        ax.set_xlabel(r"student read-in $w$")
        if z == 0:
            ax.set_title(rf"$z={z:g}$: $c_2\equiv 0$ (He$_2$ invisible)")
            ax.set_ylim(-2, 40)
        else:
            ax.set_title(rf"$z={z:g}$: biased erf, $y=\varepsilon\,\mathrm{{He}}_2$")
            ax.set_ylim(-40, 40)
        ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    axes[0].set_ylabel(r"$V(w)-V(0)$")
    fig.suptitle(
        r"Teacher $y(x)=\varepsilon\,\mathrm{He}_2(\hat t\cdot x)$"
        r"  (student $\mathrm{erf}(\alpha(w)\,\xi+z)$)",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
