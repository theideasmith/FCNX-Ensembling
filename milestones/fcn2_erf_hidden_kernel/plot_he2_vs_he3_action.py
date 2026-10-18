#!/usr/bin/env python3
"""Plot cavity action V(w) for pure He2 (even) vs He3 (odd) teachers.

  V(w) = (d/2) w^2 + (G/2) φ(w)^2
  φ₂ = w² - 1,   φ₃ = w³ - 3w
  saturating even: φ_sat = (w²-1)/(1+w²)

Usage:
  python -u plot_he2_vs_he3_action.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "action_h0_activation_plots/PureCubicChiN"
OUT.mkdir(parents=True, exist_ok=True)

D = 150.0


def phi_he2(w):
    return w * w - 1.0


def phi_he3(w):
    return w * w * w - 3.0 * w


def phi_sat(w):
    return (w * w - 1.0) / (1.0 + w * w)


def V(w, phi, G, d=D):
    return 0.5 * d * w * w + 0.5 * G * phi(w) ** 2


def shift0(w, v):
    i0 = int(np.argmin(np.abs(w)))
    return v - v[i0]


def main():
    w = np.linspace(-4.0, 4.0, 2001)
    Gs = [0.0, -50.0, -200.0, -1000.0]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    configs = [
        (axes[0], phi_he2, r"He$_2$ even: $\varphi=w^2-1$", False),
        (axes[1], phi_he3, r"He$_3$ odd: $\varphi=w^3-3w$", True),
        (axes[2], phi_sat, r"saturating even: $\varphi=\frac{w^2-1}{1+w^2}$", False),
    ]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(Gs)))

    for ax, phi, title, mark_inner in configs:
        for G, c in zip(Gs, cmap):
            v = shift0(w, V(w, phi, G))
            ax.plot(w, v, color=c, lw=1.6, label=rf"$G={G:g}$")
            # mark local minima for w>=0
            vv = v.copy()
            for j in range(1, len(w) - 1):
                if w[j] < 0:
                    continue
                if vv[j] < vv[j - 1] and vv[j] < vv[j + 1]:
                    ax.plot(w[j], vv[j], "o", color=c, ms=4.5, zorder=5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlim(-3.2, 3.2)
        ax.set_xlabel(r"$w$")
        ax.set_title(title, fontsize=11)
        if mark_inner:
            ax.annotate(
                "inner split\nwhen $V''(0)<0$",
                xy=(0.9, 0.15),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="0.35",
            )
        else:
            ax.annotate(
                r"$V''(0)=d-2G>0$" + "\n(center stays)",
                xy=(0.72, 0.82),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="0.35",
            )
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    axes[0].set_ylabel(r"$V(w)-V(0)$")
    # He3 with large |G| goes deep — clip for readability
    axes[1].set_ylim(-80, 40)
    axes[0].set_ylim(-5, 80)
    axes[2].set_ylim(-5, 80)

    fig.suptitle(
        rf"Pure teacher cavity actions ($d={D:g}$): even vs odd",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = OUT / "he2_vs_he3_action.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
