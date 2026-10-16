#!/usr/bin/env python3
"""χ=N pure-cubic V(w): full He1–He3 coupling vs --decouple (Q13=G13=0).

Usage:
  python -u plot_pure_cubic_chiN_full_vs_decouple.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "action_h0_activation_plots/PureCubicChiN"


def local_minima(w: np.ndarray, V: np.ndarray):
    if w.size < 3:
        return np.array([]), np.array([])
    mask = (V[1:-1] < V[:-2]) & (V[1:-1] < V[2:])
    idx = np.where(mask)[0] + 1
    return w[idx], V[idx]


def load_actions(path: Path):
    """Return {rho: (w, V0)} with V shifted so V(0)=0, plus summary rows."""
    by_rho: dict[float, list[tuple[float, float]]] = defaultdict(list)
    meta: dict[float, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rho = float(row["rho"])
            by_rho[rho].append((float(row["w"]), float(row["V"])))
            if rho not in meta:
                meta[rho] = {k: row[k] for k in (
                    "nwells", "dV_raw", "mu_out", "pi_out", "phase"
                ) if k in row}
    out = {}
    for rho, pts in by_rho.items():
        pts = sorted(pts)
        w = np.array([p[0] for p in pts])
        V = np.array([p[1] for p in pts])
        i0 = int(np.argmin(np.abs(w)))
        out[rho] = (w, V - V[i0], meta.get(rho, {}))
    return out


def panel(ax, data: dict, title: str, keep: list[float], wmax: float):
    rhos = sorted(data)
    keep_set = {round(r, 6) for r in keep}
    use = [r for r in rhos if round(float(r), 6) in keep_set] or rhos
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(use), max(use))
    for rho in use:
        w, V, meta = data[rho]
        m = np.abs(w) <= wmax
        phase = meta.get("phase", "")
        ax.plot(w[m], V[m], color=cmap(norm(rho)), lw=1.8,
                label=rf"$\rho={rho:g}$ [{phase}]")
        wm, Vm = local_minima(w[m], V[m])
        ax.plot(wm, Vm, "o", color=cmap(norm(rho)), ms=4, zorder=3)
    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.set_xlim(-wmax, wmax)
    ax.set_xlabel(r"read-in weight $w$")
    ax.set_ylabel(r"action $V(w)$ (shifted $V(0)=0$)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
    return norm, cmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path,
                    default=BASE / "actions_chiN_full/tmp_pure_cubic_rho_ray_actions_equilibrium.csv")
    ap.add_argument("--decouple", type=Path,
                    default=BASE / "actions_chiN_decouple/tmp_pure_cubic_rho_ray_actions_equilibrium.csv")
    ap.add_argument("--keep", default="0.15,0.12,0.10,0.09,0.08,0.07,0.06,0.05")
    ap.add_argument("--wmax", type=float, default=3.5)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--out", type=Path, default=BASE / "actions_chiN_full_vs_decouple.png")
    args = ap.parse_args()
    keep = [float(x) for x in args.keep.split(",")]

    for p in (args.full, args.decouple):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    full = load_actions(args.full)
    dec = load_actions(args.decouple)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    panel(axes[0], full, r"full He1–He3 ($\chi=N$)", keep, args.wmax)
    norm, cmap = panel(axes[1], dec, r"decoupled $Q_{13}=G_{13}=0$ ($\chi=N$)", keep, args.wmax)
    if args.ymin is not None or args.ymax is not None:
        for ax in axes:
            ax.set_ylim(args.ymin, args.ymax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\rho=\kappa/(\mathrm{sa0}\,P)$")
    fig.suptitle(
        r"Pure cubic $\chi=N$: read-in action $V(w)$ — full vs shut-off off-diagonals"
        "\n"
        r"$d=16$, $N=\chi=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $y=(0,\varepsilon)$",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}")

    for tag, data in ("full", full), ("decouple", dec):
        print(f"\n=== {tag} ===")
        print(f"{'rho':>8} {'nwells':>6} {'pi_out':>8} {'mu_out':>8} {'dV_raw':>10}  phase")
        for rho in sorted(data, reverse=True):
            _, _, m = data[rho]
            print(f"{rho:8g} {m.get('nwells','?'):>6} {float(m.get('pi_out', 'nan')):8.4f} "
                  f"{float(m.get('mu_out', 'nan')):8.3f} {float(m.get('dV_raw', 'nan')):10.4g}  "
                  f"{m.get('phase', '')}")


if __name__ == "__main__":
    main()
