#!/usr/bin/env python3
"""Amalgamate FO-window + deep-ρ cavity actions V(w) into one figure.

Two stacked panels share the same viridis(ρ) coloring:
  top  — FO / catch-up window (readable y-scale)
  bottom — deep continuation (full well deepening)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Storytelling ρ ladder: pre-FO → soft open → outer FO → catch-up → deep.
DEFAULT_KEEP = [
    0.15,
    0.12,
    0.10,
    0.095,
    0.09,
    0.085,
    0.08,
    0.075,
    0.07,
    0.05,
    0.02,
    0.005,
    0.001,
    1e-5,
]


def local_minima(w: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if w.size < 3:
        return np.array([]), np.array([])
    mask = (V[1:-1] < V[:-2]) & (V[1:-1] < V[2:])
    idx = np.where(mask)[0] + 1
    return w[idx], V[idx]


def load_actions(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def nearest(rhos: np.ndarray, target: float) -> float:
    return float(rhos[np.argmin(np.abs(rhos - target))])


def curve_at(data: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    m = data["rho"] == rho
    w = np.asarray(data["w"][m], float)
    V = np.asarray(data["V"][m], float)
    o = np.argsort(w)
    return w[o], V[o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fo",
        default="action_h0_activation_plots/PureCubicChiN/tmp_pure_cubic_rho_ray_actions_equilibrium.csv",
        help="FO-window actions CSV",
    )
    ap.add_argument(
        "--deep",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/tmp_pure_cubic_rho_ray_actions_equilibrium.csv",
        help="deep-ρ actions CSV",
    )
    ap.add_argument(
        "--out",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/actions_V_vs_w_amalgam.png",
    )
    ap.add_argument("--keep", default=",".join(str(x) for x in DEFAULT_KEEP))
    ap.add_argument("--wmax", type=float, default=11.0)
    ap.add_argument("--fo-ymin", type=float, default=-25.0)
    ap.add_argument("--fo-ymax", type=float, default=8.0)
    args = ap.parse_args()

    fo = load_actions(Path(args.fo))
    deep = load_actions(Path(args.deep))
    fo_rhos = np.unique(fo["rho"])
    deep_rhos = np.unique(deep["rho"])
    targets = [float(x) for x in args.keep.split(",") if x.strip()]

    # Prefer deep CSV when both have the ρ (wider |w|); else FO.
    meta = []
    for t in targets:
        use_deep = np.min(np.abs(deep_rhos - t)) <= 0.02 * max(t, 1e-6) + 1e-9
        # stricter: only prefer deep if within relative/absolute tolerance
        d_near = nearest(deep_rhos, t)
        f_near = nearest(fo_rhos, t)
        if abs(d_near - t) <= abs(f_near - t) and abs(d_near - t) / max(t, 1e-6) < 0.35:
            src, rho = "deep", d_near
            data = deep
        elif abs(f_near - t) / max(t, 1e-6) < 0.35 or abs(f_near - t) < 0.003:
            src, rho = "fo", f_near
            data = fo
        else:
            continue
        # de-dupe identical ρ from adjacent targets
        if meta and abs(meta[-1]["rho"] - rho) < 1e-12:
            continue
        meta.append({"rho": rho, "src": src, "data": data})

    color_vals = np.asarray([m["rho"] for m in meta], float)
    c_plot = np.log10(np.clip(color_vals, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.0]}
    )

    for ax, panel in zip(axes, ("fo", "deep")):
        for m, c in zip(meta, c_plot):
            # FO panel: skip ultra-deep curves that crush the scale
            if panel == "fo" and m["rho"] < 0.055:
                continue
            # deep panel: skip the flattest pre-FO curves (keep a few for continuity)
            if panel == "deep" and m["rho"] > 0.12:
                continue
            w, V = curve_at(m["data"], m["rho"])
            keep = np.abs(w) <= args.wmax
            w, V = w[keep], V[keep]
            color = cmap(norm(c))
            ax.plot(w, V, color=color, lw=2.0)
            wm, Vm = local_minima(w, V)
            if panel == "fo":
                ylo, yhi = args.fo_ymin, args.fo_ymax
            else:
                ylo, yhi = float(np.nanmin(V)) - 50, 20.0
            sel = (np.abs(wm) <= args.wmax) & (Vm >= ylo - 1) & (Vm <= yhi + 1)
            ax.scatter(
                wm[sel],
                Vm[sel],
                s=22,
                color=color,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
            )

        ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-args.wmax, args.wmax)
        ax.set_ylabel(r"action $V(w)$ (raw)")

    axes[0].set_ylim(args.fo_ymin, args.fo_ymax)
    # deep panel auto from plotted deep curves
    deep_vmins = []
    for m in meta:
        if m["rho"] > 0.12:
            continue
        _, V = curve_at(m["data"], m["rho"])
        deep_vmins.append(float(np.nanmin(V)))
    ymin = min(deep_vmins) if deep_vmins else -2600
    pad = 0.04 * abs(ymin)
    axes[1].set_ylim(ymin - pad, 30)

    # annotate the three stages on the FO panel
    ax = axes[0]
    ax.axvline(0.0, color="0.85", lw=0.6)
    ax.text(
        0.02,
        0.97,
        "1) soft open of center  ρ≈0.10\n"
        "2) outer FO deepens      ρ≈0.09–0.08\n"
        "3) center catch-up FO    ρ≈0.076",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.9),
    )
    axes[0].set_title(
        r"Pure cubic $\chi=N$: amalgamated $V(w)$ across the FO $\to$ deep sequence"
        "\n"
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$; color = $\rho$ (viridis)"
    )
    axes[1].set_xlabel(r"read-in weight $w$")
    axes[0].set_ylabel(r"$V(w)$  (FO / catch-up window)")
    axes[1].set_ylabel(r"$V(w)$  (deep continuation)")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.035)
    n_tick = min(len(meta), 10)
    tick_idx = np.unique(np.round(np.linspace(0, len(meta) - 1, n_tick)).astype(int))
    cbar.set_ticks(c_plot[tick_idx])
    cbar.set_ticklabels([f"{meta[i]['rho']:.3g}" for i in tick_idx])
    cbar.ax.set_ylabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$", rotation=270, labelpad=14)

    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    for m in meta:
        print(f"  ρ={m['rho']:.6g}  src={m['src']}")


if __name__ == "__main__":
    main()
