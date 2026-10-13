#!/usr/bin/env python3
"""Thin empirical W0 action overlay for the pure-cubic χ=N ρ-ray.

Loads teacher-axis read-in weights W0[..., 0] from each run:
  - pool A_snapshots/ when present (posterior), else model.pt / checkpoint.pt
Plots -log p(w) histograms overlaid, colored by ρ = κ/(sa0 P) (viridis).

Example::

    python plot_pure_cubic_rho_ray_empirical_actions.py
    python plot_pure_cubic_rho_ray_empirical_actions.py --replot   # after more training
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
        # second half of the snapshot window (equilibrated posterior)
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
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        w0 = state["W0"][..., 0].detach().cpu().numpy().reshape(-1)
        return w0, name
    raise FileNotFoundError(f"no W0 in {model_dir}")


def action_curve(values: np.ndarray, bins: int = 60, min_samples: int = 3, eps: float = 1e-9):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <sweep>/plots/empirical_weight_actions_vs_rho.png",
    )
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--wmax", type=float, default=None, help="x-axis half-width (auto if omit)")
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

    runs.sort(key=lambda r: -r["rho"])  # high → low (yellow → purple)
    rhos = np.asarray([r["rho"] for r in runs], float)
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = args.out or (args.sweep / "plots" / "empirical_weight_actions_vs_rho.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xmax_auto = 0.0
    ymin_auto, ymax_auto = np.inf, -np.inf
    for r, c in zip(runs, c_plot):
        centers, action = action_curve(r["w0"], bins=args.bins)
        mask = np.isfinite(action)
        color = cmap(norm(c))
        ax.plot(
            centers[mask],
            action[mask],
            color=color,
            lw=2.0,
            label=rf"$\rho={r['rho']:.3g}$ ({r['src']})",
        )
        if mask.any():
            xmax_auto = max(xmax_auto, float(np.max(np.abs(centers[mask]))))
            ymin_auto = min(ymin_auto, float(np.nanmin(action[mask])))
            ymax_auto = max(ymax_auto, float(np.nanpercentile(action[mask], 98)))
        print(
            f"  ρ={r['rho']:.4g}  κ={r['kappa']:.4g}  n={r['w0'].size}  "
            f"src={r['src']}  var(w0)={r['w0'].var():.4g}"
        )

    wmax = float(args.wmax) if args.wmax is not None else max(1.05 * xmax_auto, 0.8)
    ax.set_xlim(-wmax, wmax)
    if np.isfinite(ymin_auto):
        pad = 0.08 * (ymax_auto - ymin_auto + 1e-9)
        ax.set_ylim(ymin_auto - pad, ymax_auto + pad)
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel(r"read-in weight $w_0$ (teacher axis)")
    ax.set_ylabel(r"empirical action $-\log p(w_0)$")
    ax.set_title(
        r"Pure cubic $\chi=N$ ρ-ray: empirical $W_0$ actions"
        "\n"
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$, $\mathrm{he1}=0$; "
        r"color = $\rho$ (viridis)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

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
