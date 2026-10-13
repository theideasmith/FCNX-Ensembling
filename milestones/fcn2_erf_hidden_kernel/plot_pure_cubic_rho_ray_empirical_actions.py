#!/usr/bin/env python3
"""Thin empirical W0 action overlay for the pure-cubic χ=N ρ-ray.

Loads teacher-axis read-in weights W0[..., 0] from each run:
  - pool A_snapshots/ when present (posterior), else model.pt / checkpoint.pt
Plots -log p(w) histograms overlaid, colored by ρ = κ/(sa0 P) (viridis),
with χ=N cavity theory V(w) as thin dashed curves (same colors).

Default x half-width is 6.5 so outer wells (≈±2.5…±5) are in frame.
Theory CSVs default to the FO-window + deep-ρ equilibrium dumps; each
empirical ρ picks the nearest available theory ρ. Theory is shifted so its
min on the plot window matches the empirical action min (additive gauge).

Example::

    python plot_pure_cubic_rho_ray_empirical_actions.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_SWEEP = HERE / (
    "red_robin_pure_cubic_rho_ray"
    "_d16_N200_P1000_chiN_sa00.03_eps1_he10"
    "_lr0.01_ep60M_sched2_3_5"
)
DEFAULT_THEORY = [
    HERE
    / "action_h0_activation_plots/PureCubicChiN/tmp_pure_cubic_rho_ray_actions_equilibrium.csv",
    HERE
    / "action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/tmp_pure_cubic_rho_ray_actions_equilibrium.csv",
]
DIR_RE = re.compile(
    r"purecubic_rho(?P<rho>[\d.eE+-]+)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)"
    r"_chi(?P<chi>[\d.]+)_sa0(?P<sa0>[\d.]+)_kappa(?P<kappa>[\d.]+)_seed(?P<seed>\d+)$"
)


def parse_run(path: Path) -> dict | None:
    m = DIR_RE.match(path.name)
    if not m:
        return None
    return {
        "path": path,
        "rho": float(m.group("rho")),
        "kappa": float(m.group("kappa")),
        "d": int(m.group("d")),
        "P": int(m.group("P")),
        "N": int(m.group("N")),
        "sa0": float(m.group("sa0")),
        "seed": int(m.group("seed")),
    }


def load_w0_target(model_dir: Path) -> tuple[np.ndarray, str]:
    """Return flat W0[..., 0] and a short source tag."""
    snap_dir = model_dir / "A_snapshots"
    snaps = sorted(snap_dir.glob("epoch_*.pt")) if snap_dir.is_dir() else []
    if snaps:
        use = snaps[len(snaps) // 2 :]
        chunks = []
        for p in use:
            w = torch.load(p, map_location="cpu", weights_only=False)["W0"]
            chunks.append(w[..., 0].detach().cpu().numpy().reshape(-1))
        return np.concatenate(chunks), f"A_snapshots[{len(use)}/{len(snaps)}]"

    for name in ("model_final.pt", "model.pt", "checkpoint.pt"):
        ckpt = model_dir / name
        if not ckpt.is_file():
            continue
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = (
            obj["model_state_dict"]
            if isinstance(obj, dict) and "model_state_dict" in obj
            else obj
        )
        w0 = state["W0"][..., 0].detach().cpu().numpy().reshape(-1)
        return w0, name
    raise FileNotFoundError(f"no W0 in {model_dir}")


def action_curve(
    values: np.ndarray, bins: int = 60, min_samples: int = 3, eps: float = 1e-9
):
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(values, bins=bins)
    dens, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_samples
    action = np.where(mask, -np.log(dens + eps), np.nan)
    return centers, action


def load_theory_catalog(paths: list[Path]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """ρ → (w, V) from one or more long action CSVs (later files override)."""
    catalog: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for path in paths:
        if not path.is_file():
            print(f"theory skip (missing): {path}")
            continue
        data = np.genfromtxt(path, delimiter=",", names=True)
        for rho in np.unique(data["rho"]):
            m = data["rho"] == rho
            w = np.asarray(data["w"][m], float)
            V = np.asarray(data["V"][m], float)
            o = np.argsort(w)
            catalog[float(rho)] = (w[o], V[o])
        print(f"theory loaded {path.name}: {len(np.unique(data['rho']))} ρ curves")
    return catalog


def nearest_theory(
    catalog: dict[float, tuple[np.ndarray, np.ndarray]], rho: float
) -> tuple[float, np.ndarray, np.ndarray] | None:
    if not catalog:
        return None
    keys = np.asarray(sorted(catalog.keys()), float)
    j = int(np.argmin(np.abs(keys - rho)))
    r = float(keys[j])
    w, V = catalog[r]
    return r, w, V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <sweep>/plots/empirical_weight_actions_vs_rho.png",
    )
    ap.add_argument(
        "--theory",
        type=Path,
        nargs="*",
        default=DEFAULT_THEORY,
        help="cavity V(w) CSVs (FO + deep). Empty disables theory overlay.",
    )
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument(
        "--wmax",
        type=float,
        default=7.0,
        help="x-axis half-width (wide enough for outer wells; was emp-auto)",
    )
    ap.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="y-axis max (default: show outer-well walls without barrier blow-up)",
    )
    ap.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="y-axis min (default: below emp/theory well floors)",
    )
    args = ap.parse_args()

    models = args.sweep / "models"
    if not models.is_dir():
        raise SystemExit(f"missing models dir: {models}")

    runs = []
    for d in sorted(models.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run(d)
        if meta is None:
            continue
        try:
            w0, src = load_w0_target(d)
        except FileNotFoundError as e:
            print(f"skip {d.name}: {e}")
            continue
        meta["w0"] = w0
        meta["src"] = src
        runs.append(meta)
    if not runs:
        raise SystemExit(f"no purecubic_rho* runs with W0 under {models}")

    catalog = load_theory_catalog(list(args.theory)) if args.theory else {}

    runs.sort(key=lambda r: -r["rho"])  # high → low (yellow → purple)
    rhos = np.asarray([r["rho"] for r in runs], float)
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = args.out or (args.sweep / "plots" / "empirical_weight_actions_vs_rho.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    wmax = float(args.wmax)
    ymin_auto, ymax_auto = np.inf, -np.inf
    theory_labeled = False
    for r, c in zip(runs, c_plot):
        centers, action = action_curve(r["w0"], bins=args.bins)
        mask = np.isfinite(action)
        color = cmap(norm(c))
        ax.plot(
            centers[mask],
            action[mask],
            color=color,
            lw=2.0,
            label=rf"$\rho={r['rho']:.3g}$ emp ({r['src']})",
        )
        if mask.any():
            ymin_auto = min(ymin_auto, float(np.nanmin(action[mask])))
            ymax_auto = max(ymax_auto, float(np.nanpercentile(action[mask], 98)))

        th = nearest_theory(catalog, r["rho"])
        if th is not None and mask.any():
            rho_th, w_th, V_th = th
            in_win = np.abs(w_th) <= wmax
            if not np.any(in_win):
                in_win = np.ones_like(w_th, dtype=bool)
            emp_min = float(np.nanmin(action[mask]))
            V_win = V_th[in_win]
            if np.any(np.isfinite(V_win)):
                V_shift = V_th - float(np.nanmin(V_win)) + emp_min
                Vs = V_shift[in_win]
                # ylim: well floor + wall height; ignore high spurious local mins
                v_floor = float(np.nanmin(Vs))
                ymin_auto = min(ymin_auto, v_floor)
                ymax_auto = max(ymax_auto, v_floor + 24.0)
                lbl = None
                if not theory_labeled:
                    lbl = r"theory $V(w)$ (dashed, shifted)"
                    theory_labeled = True
                ax.plot(
                    w_th,
                    V_shift,
                    color=color,
                    lw=1.15,
                    ls="--",
                    alpha=0.95,
                    label=lbl,
                    zorder=2,
                )
                print(
                    f"  ρ={r['rho']:.4g}  emp={r['src']}  "
                    f"theory ρ={rho_th:.4g}  (Δρ={rho_th - r['rho']:+.4g})"
                )
            else:
                print(f"  ρ={r['rho']:.4g}  emp={r['src']}  theory empty window")
        else:
            print(
                f"  ρ={r['rho']:.4g}  κ={r['kappa']:.4g}  n={r['w0'].size}  "
                f"src={r['src']}  var(w0)={r['w0'].var():.4g}"
            )

    ax.set_xlim(-wmax, wmax)
    if args.ymin is not None or args.ymax is not None or np.isfinite(ymin_auto):
        y0 = float(args.ymin) if args.ymin is not None else float(ymin_auto)
        y1 = float(args.ymax) if args.ymax is not None else float(ymax_auto)
        pad = 0.08 * (y1 - y0 + 1e-9)
        if args.ymin is None:
            y0 -= pad
        if args.ymax is None:
            y1 += pad
        ax.set_ylim(y0, y1)
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel(r"read-in weight $w_0$ (teacher axis)")
    ax.set_ylabel(r"action (emp $-\log p$; theory $V$ shifted)")
    ax.set_title(
        r"Pure cubic $\chi=N$ ρ-ray: empirical $W_0$ + theory $V(w)$"
        "\n"
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$, $\mathrm{he1}=0$; "
        r"color = $\rho$ (viridis)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_ticks(c_plot)
    cbar.set_ticklabels([f"{r['rho']:.3g}" for r in runs])
    cbar.ax.set_ylabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$", rotation=270, labelpad=14)

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
