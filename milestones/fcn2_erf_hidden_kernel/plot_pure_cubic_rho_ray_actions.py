#!/usr/bin/env python3
"""Small-multiples V(w) along a ρ-ray through the outer-well FO."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_KEEP = [0.20, 0.15, 0.1334, 0.10]


def local_minima(w: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior local minima (strict)."""
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
    ap.add_argument("--wmax", type=float, default=3.5)
    ap.add_argument("--ymin", type=float, default=-2.0)
    ap.add_argument("--ymax", type=float, default=1.5)
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    all_rhos = np.unique(data["rho"])

    def nearest(target: float) -> float:
        return float(all_rhos[np.argmin(np.abs(all_rhos - target))])

    targets = [float(x) for x in args.keep.split(",") if x.strip()]
    rhos = [nearest(t) for t in targets]

    phase: dict[float, str] = {}
    v_out: dict[float, float] = {}
    if args.summary and Path(args.summary).is_file():
        summary = np.genfromtxt(
            args.summary, delimiter=",", names=True, dtype=None, encoding=None
        )
        for row in np.atleast_1d(summary):
            ρ = float(row["rho"])
            phase[ρ] = str(row["phase"]).replace("_", " ")
            v_out[ρ] = float(row["V_out"])

    logρ = np.log10(np.asarray(rhos))
    norm = mpl.colors.Normalize(vmin=float(logρ.min()), vmax=float(logρ.max()))
    cmap = plt.get_cmap("viridis")

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_", "").replace(".csv", ".png")
    )

    n = len(rhos)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9.0, 3.8 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, ρ in enumerate(rhos):
        ax = axes_flat[i]
        m = data["rho"] == ρ
        w = data["w"][m]
        V = data["V"][m]
        order = np.argsort(w)
        w, V = w[order], V[order]
        color = cmap(norm(np.log10(ρ)))

        ax.plot(w, V, color=color, lw=2.2)
        ax.axhline(0.0, color="0.65", lw=0.7, ls="--")
        ax.axvline(0.0, color="0.85", lw=0.6)

        wm, Vm = local_minima(w, V)
        keep = (np.abs(wm) <= args.wmax) & (Vm >= args.ymin - 0.5) & (Vm <= args.ymax + 0.5)
        ax.scatter(
            wm[keep],
            Vm[keep],
            s=28,
            color=color,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )

        key = min(phase.keys(), key=lambda x: abs(x - ρ)) if phase else None
        ph = phase[key] if key is not None else ""
        vo = v_out.get(key, np.nan) if key is not None else np.nan
        if ph == "no outer":
            subtitle = "no outer well"
        elif np.isfinite(vo) and vo != 0.0:
            subtitle = f"{ph}\n$V_{{\\mathrm{{out}}}}={vo:+.2f}$"
        else:
            subtitle = ph

        ax.set_title(rf"$\rho={ρ:.3g}$" + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
        ax.set_xlim(-args.wmax, args.wmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.grid(True, alpha=0.22)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"read-in weight $w$")
        if i % ncols == 0:
            ax.set_ylabel(r"action $V(w)$")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat[:n].tolist(), pad=0.02, fraction=0.04)
    cbar.set_ticks(logρ)
    cbar.set_ticklabels([f"{r:.3g}" for r in rhos])
    cbar.ax.set_ylabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$", rotation=270, labelpad=14)

    fig.suptitle(
        r"Pure cubic $\chi=N$, $P=10^3$, $\mathrm{sa0}=0.03$"
        "\n"
        r"outer-well first-order sequence along $\rho$",
        fontsize=12,
    )
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
