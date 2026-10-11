#!/usr/bin/env python3
"""Overlay V(w) along a ρ=κ/(sa0 P) ray, colored by ρ (viridis), zoomed on wells."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wmax", type=float, default=8.0, help="x half-width for zoom")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=5.0, help="y max for well zoom")
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    rhos = np.unique(data["rho"])
    rhos.sort()

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_", "").replace(".csv", ".png")
    )

    logρ = np.log10(rhos)
    norm = mpl.colors.Normalize(vmin=float(logρ.min()), vmax=float(logρ.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    Vmins = []
    for ρ in rhos:
        m = data["rho"] == ρ
        w = data["w"][m]
        V = data["V"][m]
        order = np.argsort(w)
        ax.plot(w[order], V[order], color=cmap(norm(np.log10(ρ))), lw=1.7)
        Vmins.append(float(np.min(V)))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_ticks(logρ)
    cbar.set_ticklabels([f"{r:.3g}" for r in rhos])
    cbar.ax.set_ylabel(r"ray parameter $\rho=\kappa/(\mathrm{sa0}\,P)$", rotation=270, labelpad=16)

    ax.axhline(0.0, color="gray", lw=0.7, alpha=0.5)
    ax.set_xlim(-args.wmax, args.wmax)
    ymin = args.ymin if args.ymin is not None else min(Vmins) - 0.5
    ax.set_ylim(ymin, args.ymax)
    ax.set_xlabel(r"read-in weight $w$")
    ax.set_ylabel(r"action $V(w)$ (raw)")
    ax.set_title(
        r"Pure cubic $\chi=N$: $V(w)$ along $\rho$-ray through outer-well FO transition"
        "\n"
        r"fixed $P$, $\mathrm{sa0}$; $\kappa=\rho\,\mathrm{sa0}\,P$; color = $\rho$ (viridis)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"rhos: {rhos.tolist()}")
    print(f"Vmin over ray: {min(Vmins):.4g}")


if __name__ == "__main__":
    main()
