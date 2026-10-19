#!/usr/bin/env python3
"""Empirical W0 −log p vs density-normalized cavity theory −log π.

Loads teacher-axis read-in weights W0[..., 0] from each run:
  - pool A_snapshots/ when present (posterior), else model.pt / checkpoint.pt

Empirical action is −log of a *normalized* histogram density (gauge fixed).
Theory CSVs store raw cavity V (defined up to an additive constant). For overlay
we convert each curve to the density-normalized action

    −log π(w) = V(w) + log ∫ e^{−V} dw

(same convention as emp). Emp is not shifted.

Theory CSV selection matches the amalgam (FO CSV for high ρ, deep for low ρ).

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
DEFAULT_FO = (
    HERE
    / "action_h0_activation_plots/PureCubicChiN/tmp_pure_cubic_rho_ray_actions_equilibrium.csv"
)
DEFAULT_DEEP = (
    HERE
    / "action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3"
    / "tmp_pure_cubic_rho_ray_actions_equilibrium.csv"
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
    """Density-normalized −log p from a 1d sample (histogram density)."""
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


def density_normalized_action(w: np.ndarray, V: np.ndarray) -> np.ndarray:
    """−log π with π ∝ e^{−V} and ∫ π = 1 (trapezoid on the given grid)."""
    w = np.asarray(w, float)
    V = np.asarray(V, float)
    Vn = V - np.nanmin(V)
    # −log π = Vn + log ∫ e^{−Vn} dw
    logZ = float(np.log(np.trapezoid(np.exp(-Vn), w)))
    return Vn + logZ


def load_actions(path: Path) -> np.ndarray | None:
    if not path.is_file():
        print(f"theory skip (missing): {path}")
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    print(f"theory loaded {path.name}: {len(np.unique(data['rho']))} ρ curves")
    return data


def nearest_rho(rhos: np.ndarray, target: float) -> float:
    return float(rhos[np.argmin(np.abs(rhos - target))])


def curve_at(data: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    m = data["rho"] == rho
    w = np.asarray(data["w"][m], float)
    V = np.asarray(data["V"][m], float)
    o = np.argsort(w)
    return w[o], V[o]


def pick_theory(
    fo: np.ndarray | None, deep: np.ndarray | None, rho: float
) -> tuple[float, np.ndarray, np.ndarray, str] | None:
    """Same FO/deep preference as plot_pure_cubic_rho_actions_amalgam.py."""
    fo_rhos = np.unique(fo["rho"]) if fo is not None else np.array([])
    deep_rhos = np.unique(deep["rho"]) if deep is not None else np.array([])
    if fo_rhos.size == 0 and deep_rhos.size == 0:
        return None
    d_near = nearest_rho(deep_rhos, rho) if deep_rhos.size else None
    f_near = nearest_rho(fo_rhos, rho) if fo_rhos.size else None
    if d_near is not None and (
        f_near is None
        or (
            abs(d_near - rho) <= abs(f_near - rho)
            and abs(d_near - rho) / max(rho, 1e-6) < 0.35
        )
    ):
        w, V = curve_at(deep, d_near)
        return d_near, w, V, "deep"
    if f_near is not None and (
        abs(f_near - rho) / max(rho, 1e-6) < 0.35 or abs(f_near - rho) < 0.003
    ):
        w, V = curve_at(fo, f_near)
        return f_near, w, V, "fo"
    cands = []
    if f_near is not None:
        cands.append(("fo", f_near, fo))
    if d_near is not None:
        cands.append(("deep", d_near, deep))
    src, r, data = min(cands, key=lambda t: abs(t[1] - rho))
    w, V = curve_at(data, r)
    return r, w, V, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <sweep>/plots/empirical_weight_actions_vs_rho.png",
    )
    ap.add_argument("--fo", type=Path, default=DEFAULT_FO)
    ap.add_argument("--deep", type=Path, default=DEFAULT_DEEP)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--wmax", type=float, default=11.0)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
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

    fo = load_actions(args.fo)
    deep = load_actions(args.deep)

    runs.sort(key=lambda r: -r["rho"])
    rhos = np.asarray([r["rho"] for r in runs], float)
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = args.out or (args.sweep / "plots" / "empirical_weight_actions_vs_rho.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
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
            zorder=3,
        )
        if mask.any():
            ymin_auto = min(ymin_auto, float(np.nanmin(action[mask])))
            ymax_auto = max(ymax_auto, float(np.nanpercentile(action[mask], 98)))

        th = pick_theory(fo, deep, r["rho"])
        if th is None:
            print(f"  ρ={r['rho']:.4g}  emp={r['src']}  no theory")
            continue
        rho_th, w_th, V_th, src_th = th
        in_win = np.abs(w_th) <= wmax
        if not np.any(in_win):
            in_win = np.ones_like(w_th, dtype=bool)
        w_use, V_use = w_th[in_win], V_th[in_win]
        V_norm = density_normalized_action(w_use, V_use)
        lbl = None
        if not theory_labeled:
            lbl = r"theory $-\log\pi$ (dashed; $V$ density-normalized)"
            theory_labeled = True
        ax.plot(
            w_use,
            V_norm,
            color=color,
            lw=1.15,
            ls="--",
            alpha=0.95,
            label=lbl,
            zorder=2,
        )
        ymin_auto = min(ymin_auto, float(np.nanmin(V_norm)))
        # walls of outer wells, not barrier blow-up
        ymax_auto = max(ymax_auto, float(np.nanmin(V_norm)) + 6.0)
        print(
            f"  ρ={r['rho']:.4g}  emp={r['src']}  "
            f"theory ρ={rho_th:.4g} ({src_th})  "
            f"−logπ min={float(np.nanmin(V_norm)):.4g}  "
            f"(Δρ={rho_th - r['rho']:+.4g})"
        )

    ax.set_xlim(-wmax, wmax)
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
    ax.set_ylabel(r"action $-\log p$ / $-\log\pi$ (density-normalized)")
    ax.set_title(
        r"Pure cubic $\chi=N$ ρ-ray: emp $-\log p$ vs theory $-\log\pi$"
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
