#!/usr/bin/env python3
"""Overlay mode-Laplace −log p (self-consistent f*) on pure-cubic χ=N emp −log p.

Reads caches from action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac/
(written by tmp_mode_laplace_pure_cubic_rho.py). Emp: second half of precond
Langevin (M2000 > M200) when present under PureCubicChiNRhoRay/precond_langevin/,
else A_snapshots.

Usage:
  python plot_pure_cubic_mode_laplace_actions.py
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
DEFAULT_THEORY = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
PRECOND = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/precond_langevin"
DIR_RE = re.compile(
    r"purecubic_rho(?P<rho>[\d.eE+-]+)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)"
    r"_chi(?P<chi>[\d.]+)_sa0(?P<sa0>[\d.]+)_kappa(?P<kappa>[\d.]+)_seed(?P<seed>\d+)$"
)


def parse_run(path: Path):
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


def _rho_tag(rho: float) -> str:
    return f"rho{rho:g}".replace(".", "p")


def load_w0(model_dir: Path, rho: float | None = None):
    """Prefer precond Langevin second half; else A_snapshots second half."""
    if rho is None:
        meta = parse_run(model_dir)
        rho = meta["rho"] if meta else None
    if rho is not None:
        t = _rho_tag(rho)
        for mw in ("M2000", "M200"):
            snap_dir = PRECOND / f"{t}_{mw}" / "snapshots"
            snaps = (
                sorted(snap_dir.glob("step_*.pt"), key=lambda s: int(s.stem.split("_")[1]))
                if snap_dir.is_dir()
                else []
            )
            if len(snaps) < 4:
                continue
            use = snaps[len(snaps) // 2 :]
            chunks = [
                torch.load(p, map_location="cpu", weights_only=False)["W0"][..., 0]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                for p in use
            ]
            return np.concatenate(chunks), f"precond {mw}[{len(use)}/{len(snaps)}]"
    snap_dir = model_dir / "A_snapshots"
    snaps = sorted(snap_dir.glob("epoch_*.pt")) if snap_dir.is_dir() else []
    if snaps:
        use = snaps[len(snaps) // 2 :]
        chunks = [
            torch.load(p, map_location="cpu", weights_only=False)["W0"][..., 0]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            for p in use
        ]
        return np.concatenate(chunks), f"A_snapshots[{len(use)}/{len(snaps)}]"
    raise FileNotFoundError(f"no precond or A_snapshots for {model_dir}")


def emp_action(values, bins=72, min_samples=3, eps=1e-9):
    values = np.asarray(values, float).reshape(-1)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    dens, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_samples
    action = np.where(mask, -np.log(dens + eps), np.nan)
    return centers, action


def theory_file(theory_dir: Path, rho: float):
    tag = f"rho{rho:g}".replace(".", "p")
    p = theory_dir / f"{tag}_scfrac.npz"
    if p.is_file():
        return p
    cands = []
    for f in theory_dir.glob("rho*_scfrac.npz"):
        try:
            r = float(np.load(f)["rho"])
        except Exception:
            continue
        cands.append((abs(r - rho), f, r))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--theory", type=Path, default=DEFAULT_THEORY)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wmax", type=float, default=5.5)
    ap.add_argument("--bins", type=int, default=72)
    args = ap.parse_args()

    models = args.sweep / "models"
    runs = []
    for d in sorted(models.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run(d)
        if meta is None:
            continue
        try:
            w0, src = load_w0(d, rho=meta["rho"])
        except FileNotFoundError as e:
            print(f"skip {d.name}: {e}")
            continue
        meta["w0"], meta["src"] = w0, src
        runs.append(meta)
        print(f"ρ={meta['rho']:g}: emp from {src}", flush=True)
    if not runs:
        raise SystemExit(f"no runs under {models}")
    runs.sort(key=lambda r: -r["rho"])

    out = args.out or (args.theory / "empirical_vs_mode_laplace_scfrac.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    rhos = np.asarray([r["rho"] for r in runs], float)
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ymin, ymax = np.inf, -np.inf
    theory_labeled = False

    for r, c in zip(runs, c_plot):
        centers, action = emp_action(r["w0"], bins=args.bins)
        mask = np.isfinite(action) & (np.abs(centers) <= args.wmax)
        color = cmap(norm(c))
        ax.plot(
            centers[mask],
            action[mask],
            color=color,
            lw=2.0,
            label=rf"$\rho={r['rho']:.3g}$ emp ({r['src']})",
            zorder=3,
        )
        if mask.any():
            ymin = min(ymin, float(np.nanmin(action[mask])))
            ymax = max(ymax, float(np.nanpercentile(action[mask], 98)))

        tf = theory_file(args.theory, r["rho"])
        if tf is None:
            print(f"  ρ={r['rho']:.4g}: no theory cache")
            continue
        th = np.load(tf)
        x = np.asarray(th["x"], float)
        nlogp = np.asarray(th["nlogp"], float)
        in_win = np.abs(x) <= args.wmax
        lbl = None
        if not theory_labeled:
            lbl = r"mode-Laplace $-\log p$ (dashed; $S=I$, SELF=full, FIX_TRACE, self-consistent $f^*$)"
            theory_labeled = True
        ax.plot(
            x[in_win],
            nlogp[in_win],
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.95,
            label=lbl,
            zorder=2,
        )
        ymin = min(ymin, float(np.nanmin(nlogp[in_win])))
        ymax = max(ymax, float(np.nanmin(nlogp[in_win])) + 8.0)
        f_net = float(np.mean(np.abs(r["w0"]) > 1.0))
        print(
            f"  ρ={float(th['rho']):.4g}: f*={float(th['f_s']):.4f} "
            f"(emp |w0|>1: {f_net:.3f}) "
            f"bulk μ={float(th['mu_b']):.3f} spec μ={float(th['mu_s']):.3f}",
            flush=True,
        )

    ax.set_xlim(-args.wmax, args.wmax)
    pad = 0.08 * (ymax - ymin + 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel(r"read-in weight $w_0$ (teacher axis)")
    ax.set_ylabel(r"$-\log p(w_0)$ (density-normalized)")
    ax.set_title(
        r"Pure cubic $\chi=N$ ρ-ray: emp (precond Langevin when available) vs mode-Laplace"
        "\n"
        r"154 modes, off-data $S=I$, SELF=full + FIX_TRACE; "
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$"
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
