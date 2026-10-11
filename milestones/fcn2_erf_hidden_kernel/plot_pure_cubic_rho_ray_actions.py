#!/usr/bin/env python3
"""Overlay a few V(w) curves along a ρ-ray (viridis), zoomed on the FO."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_KEEP = [0.20, 0.15, 0.1334, 0.10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--summary", default=None, help="optional summary CSV with phase")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--keep",
        default=",".join(str(x) for x in DEFAULT_KEEP),
        help="comma-separated ρ values to plot",
    )
    ap.add_argument("--wmax", type=float, default=4.0)
    ap.add_argument("--ymin", type=float, default=-3.0)
    ap.add_argument("--ymax", type=float, default=2.0)
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    all_rhos = np.unique(data["rho"])

    def nearest(target: float) -> float:
        return float(all_rhos[np.argmin(np.abs(all_rhos - target))])

    targets = [float(x) for x in args.keep.split(",") if x.strip()]
    rhos = [nearest(t) for t in targets]

    phase: dict[float, str] = {}
    if args.summary and Path(args.summary).is_file():
        summary = np.genfromtxt(args.summary, delimiter=",", names=True, dtype=None, encoding=None)
        for row in np.atleast_1d(summary):
            phase[float(row["rho"])] = str(row["phase"]).replace("_", " ")

    logρ = np.log10(np.asarray(rhos))
    norm = mpl.colors.Normalize(vmin=float(logρ.min()), vmax=float(logρ.max()))
    cmap = plt.get_cmap("viridis")

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_", "").replace(".csv", ".png")
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for ρ in rhos:
        m = data["rho"] == ρ
        w = data["w"][m]
        V = data["V"][m]
        order = np.argsort(w)
        key = min(phase.keys(), key=lambda x: abs(x - ρ)) if phase else None
        label = rf"$\rho={ρ:.3g}$" + (f" ({phase[key]})" if key is not None else "")
        ax.plot(w[order], V[order], color=cmap(norm(np.log10(ρ))), lw=2.0, label=label)

    ax.axhline(0.0, color="0.5", lw=0.8, alpha=0.6)
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel(r"read-in weight $w$")
    ax.set_ylabel(r"action $V(w)$ (raw)")
    ax.set_title(
        r"Pure cubic $\chi=N$: outer-well FO along $\rho=\kappa/(\mathrm{sa0}\,P)$"
        "\n"
        r"$P=10^3$, $\mathrm{sa0}=0.03$ — selected $\rho$ only"
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_ticks(logρ)
    cbar.set_ticklabels([f"{r:.3g}" for r in rhos])
    cbar.ax.set_ylabel(r"$\rho$", rotation=270, labelpad=12)

    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
