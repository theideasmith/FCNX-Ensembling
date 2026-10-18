#!/usr/bin/env python3
"""χ=N eigenframe-cubic V(w): y_lab = ε u_*(Q) ⇒ ỹ = (0, ε).

Usage:
  python -u plot_pure_cubic_chiN_eigenframe.py
  python -u plot_pure_cubic_chiN_eigenframe.py --compare-full
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


def _phase_from_meta(m: dict) -> str:
    if m.get("phase"):
        return m["phase"]
    try:
        nw = int(float(m.get("nwells", "1")))
        dv = float(m.get("dV_raw", "nan"))
    except ValueError:
        return ""
    if nw < 2:
        return "no_outer"
    if not np.isfinite(dv):
        return ""
    return "inner_deeper" if dv > 0 else "outer_deeper"


def load_actions(path: Path):
    by_rho: dict[float, list[tuple[float, float]]] = defaultdict(list)
    meta: dict[float, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rho = float(row["rho"])
            by_rho[rho].append((float(row["w"]), float(row["V"])))
            if rho not in meta:
                meta[rho] = {k: row[k] for k in (
                    "nwells", "dV_raw", "mu_out", "pi_out", "phase",
                    "he1_lab", "he3_lab", "yrot1", "yrot2",
                ) if k in row}
    # Merge phase from sibling summary CSV when actions lack it.
    sum_path = path.with_name(path.name.replace("_actions_", "_summary_"))
    if sum_path.is_file():
        with sum_path.open() as f:
            for row in csv.DictReader(f):
                rho = float(row["rho"])
                if rho in meta:
                    for k in ("phase", "he1_lab", "he3_lab", "yrot1", "yrot2",
                              "nwells", "dV_raw", "mu_out", "pi_out"):
                        if k in row and row[k]:
                            meta[rho][k] = row[k]
    out = {}
    for rho, pts in by_rho.items():
        pts = sorted(pts)
        w = np.array([p[0] for p in pts])
        V = np.array([p[1] for p in pts])
        i0 = int(np.argmin(np.abs(w)))
        m = meta.get(rho, {})
        m["phase"] = _phase_from_meta(m)
        out[rho] = (w, V - V[i0], m)
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
    ap.add_argument("--eigenframe", type=Path,
                    default=BASE / "actions_chiN_eigenframe/tmp_pure_cubic_rho_ray_actions_equilibrium.csv")
    ap.add_argument("--full", type=Path,
                    default=BASE / "actions_chiN_full/tmp_pure_cubic_rho_ray_actions_equilibrium.csv")
    ap.add_argument("--compare-full", action="store_true")
    ap.add_argument("--keep", default="0.15,0.12,0.10,0.09,0.08,0.07,0.06,0.05")
    ap.add_argument("--wmax", type=float, default=3.5)
    ap.add_argument("--out", type=Path,
                    default=BASE / "actions_chiN_eigenframe.png")
    args = ap.parse_args()
    keep = [float(x) for x in args.keep.split(",")]

    if not args.eigenframe.is_file():
        raise SystemExit(f"missing {args.eigenframe}")
    eig = load_actions(args.eigenframe)

    if args.compare_full:
        if not args.full.is_file():
            raise SystemExit(f"missing {args.full}")
        full = load_actions(args.full)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
        panel(axes[0], full, r"lab $y=(0,\varepsilon)$ ($\chi=N$)", keep, args.wmax)
        norm, cmap = panel(
            axes[1], eig,
            r"eigenframe $y=\varepsilon u_*(Q)$ $\Rightarrow$ $\tilde y=(0,\varepsilon)$",
            keep, args.wmax,
        )
        title = (
            r"Pure cubic $\chi=N$: $V(w)$ — fixed lab $y=(0,\varepsilon)$ vs eigenframe"
            "\n"
            r"$d=16$, $N=\chi=200$, $P=10^3$, $\mathrm{sa0}=0.03$"
        )
        args.out = args.out if args.out.name != "actions_chiN_eigenframe.png" else (
            BASE / "actions_chiN_full_vs_eigenframe.png"
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8))
        axes = [ax]
        norm, cmap = panel(
            ax, eig,
            r"eigenframe $y=\varepsilon u_{\mathrm{cubic}}(Q)$ $\Rightarrow$ $\tilde y=(0,\varepsilon)$",
            keep, args.wmax,
        )
        title = (
            r"Eigenframe cubic $\chi=N$: read-in action $V(w)$"
            "\n"
            r"$d=16$, $N=\chi=200$, $P=10^3$, $\mathrm{sa0}=0.03$"
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\rho=\kappa/(\mathrm{sa0}\,P)$")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}")

    print(f"\n=== eigenframe ===")
    print(f"{'rho':>8} {'nwells':>6} {'pi_out':>8} {'mu_out':>8} {'dV_raw':>10} "
          f"{'he1':>8} {'he3':>8} {'yrot1':>10} {'yrot2':>8}  phase")
    for rho in sorted(eig, reverse=True):
        _, _, m = eig[rho]
        def f(k, default="nan"):
            return float(m.get(k, default))
        print(f"{rho:8g} {m.get('nwells','?'):>6} {f('pi_out'):8.4f} "
              f"{f('mu_out'):8.3f} {f('dV_raw'):10.4g}  "
              f"{f('he1_lab'):8.4f} {f('he3_lab'):8.4f} "
              f"{f('yrot1'):10.2e} {f('yrot2'):8.4f}  {m.get('phase','')}")


if __name__ == "__main__":
    main()
