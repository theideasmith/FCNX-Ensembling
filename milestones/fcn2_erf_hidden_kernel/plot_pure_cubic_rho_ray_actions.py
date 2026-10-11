#!/usr/bin/env python3
"""Single-panel V(w) overlay along a ρ-ray through the outer-well FO."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_KEEP = [0.20, 0.15, 0.1334, 0.10]


def local_minima(w: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if w.size < 3:
        return np.array([]), np.array([])
    mask = (V[1:-1] < V[:-2]) & (V[1:-1] < V[2:])
    idx = np.where(mask)[0] + 1
    return w[idx], V[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--keep",
        default=",".join(str(x) for x in DEFAULT_KEEP),
        help="comma-separated ρ values to plot",
    )
    ap.add_argument("--wmax", type=float, default=3.2)
    # Defaults cover FO wells + barriers; pass null-like via omitting for auto
    ap.add_argument("--ymin", type=float, default=-1.9)
    ap.add_argument("--ymax", type=float, default=5.8)
    ap.add_argument("--auto-ylim", action="store_true", help="derive ylim from wells/barriers")
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    all_rhos = np.unique(data["rho"])

    def nearest(target: float) -> float:
        return float(all_rhos[np.argmin(np.abs(all_rhos - target))])

    targets = [float(x) for x in args.keep.split(",") if x.strip()]
    rhos = [nearest(t) for t in targets]

    phase: dict[float, str] = {}
    if args.summary and Path(args.summary).is_file():
        summary = np.genfromtxt(
            args.summary, delimiter=",", names=True, dtype=None, encoding=None
        )
        for row in np.atleast_1d(summary):
            phase[float(row["rho"])] = str(row["phase"]).replace("_", " ")

    # Collect series for limits + plot
    series = []
    mins: list[float] = []
    barriers: list[float] = []
    for ρ in rhos:
        m = data["rho"] == ρ
        w = data["w"][m]
        V = data["V"][m]
        order = np.argsort(w)
        w, V = w[order], V[order]
        series.append((ρ, w, V))
        wm, Vm = local_minima(w, V)
        in_win = np.abs(wm) <= args.wmax
        mins.extend(Vm[in_win].tolist())
        # local maxima = inter-well barriers (ignore prior walls: keep V < 20)
        if w.size >= 3:
            mask = (V[1:-1] > V[:-2]) & (V[1:-1] > V[2:])
            idx = np.where(mask)[0] + 1
            for wi, Vi in zip(w[idx], V[idx]):
                if abs(wi) <= args.wmax and Vi < 20.0:
                    barriers.append(float(Vi))

    if args.auto_ylim:
        ymin = (min(mins) - 0.25) if mins else -2.0
        ymax = (max(barriers) + 0.4) if barriers else 1.5
    else:
        ymin = args.ymin
        ymax = args.ymax

    logρ = np.log10(np.asarray(rhos))
    norm = mpl.colors.Normalize(vmin=float(logρ.min()), vmax=float(logρ.max()))
    cmap = plt.get_cmap("viridis")

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_", "").replace(".csv", ".png")
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for ρ, w, V in series:
        color = cmap(norm(np.log10(ρ)))
        key = min(phase.keys(), key=lambda x: abs(x - ρ)) if phase else None
        ph = f" ({phase[key]})" if key is not None else ""
        ax.plot(w, V, color=color, lw=2.0, label=rf"$\rho={ρ:.3g}${ph}")
        wm, Vm = local_minima(w, V)
        keep = (np.abs(wm) <= args.wmax) & (Vm >= ymin - 0.1) & (Vm <= ymax + 0.1)
        ax.scatter(
            wm[keep],
            Vm[keep],
            s=26,
            color=color,
            edgecolors="k",
            linewidths=0.45,
            zorder=5,
        )

    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"read-in weight $w$")
    ax.set_ylabel(r"action $V(w)$ (raw)")
    ax.set_title(
        r"Pure cubic $\chi=N$, $P=10^3$, $\mathrm{sa0}=0.03$"
        "\n"
        r"outer-well FO along $\rho=\kappa/(\mathrm{sa0}\,P)$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_ticks(logρ)
    cbar.set_ticklabels([f"{r:.3g}" for r in rhos])
    cbar.ax.set_ylabel(r"$\rho$", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}  ylim=({ymin:.3g}, {ymax:.3g})")


if __name__ == "__main__":
    main()
