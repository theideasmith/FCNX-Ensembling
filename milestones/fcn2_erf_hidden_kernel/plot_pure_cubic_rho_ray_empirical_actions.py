#!/usr/bin/env python3
"""Empirical W0 action overlay on the same raw V(w) scale as the theory amalgam.

Loads teacher-axis read-in weights W0[..., 0] from each run:
  - pool A_snapshots/ when present (posterior), else model.pt / checkpoint.pt

Theory is the same cavity CSVs / raw V(w) as
`action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/actions_V_vs_w_amalgam.png`
(FO CSV for high ρ, deep CSV for low ρ; no additive shift of theory).

Empirical −log p is shifted onto that raw gauge (emp min → theory min on the
plot window) so well depths are comparable. Two stacked panels share the
amalgam axis limits: FO window (−25…8) and deep continuation.

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
    # fallback: absolute nearest across available
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
    # Match amalgam defaults so overlays are visually comparable.
    ap.add_argument("--wmax", type=float, default=11.0)
    ap.add_argument("--fo-ymin", type=float, default=-25.0)
    ap.add_argument("--fo-ymax", type=float, default=8.0)
    ap.add_argument(
        "--fo-rho-min",
        type=float,
        default=0.055,
        help="FO panel: only curves with ρ ≥ this (same cut as amalgam)",
    )
    ap.add_argument(
        "--deep-rho-max",
        type=float,
        default=0.12,
        help="deep panel: only curves with ρ ≤ this (same cut as amalgam)",
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

    fo = load_actions(args.fo)
    deep = load_actions(args.deep)

    runs.sort(key=lambda r: -r["rho"])  # high → low (yellow → purple)
    rhos = np.asarray([r["rho"] for r in runs], float)
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = args.out or (args.sweep / "plots" / "empirical_weight_actions_vs_rho.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Precompute emp curves + raw theory (amalgam gauge).
    prepared = []
    deep_vmins = []
    for r, c in zip(runs, c_plot):
        centers, action = action_curve(r["w0"], bins=args.bins)
        mask = np.isfinite(action)
        th = pick_theory(fo, deep, r["rho"])
        entry = {
            "run": r,
            "c": c,
            "centers": centers,
            "action": action,
            "mask": mask,
            "th": th,
        }
        if th is not None:
            rho_th, w_th, V_th, src_th = th
            in_win = np.abs(w_th) <= args.wmax
            if not np.any(in_win):
                in_win = np.ones_like(w_th, dtype=bool)
            V_min = float(np.nanmin(V_th[in_win]))
            entry["V_min"] = V_min
            entry["w_th"] = w_th
            entry["V_th"] = V_th
            entry["rho_th"] = rho_th
            entry["src_th"] = src_th
            if r["rho"] <= args.deep_rho_max:
                deep_vmins.append(V_min)
            print(
                f"  ρ={r['rho']:.4g}  emp={r['src']}  "
                f"theory ρ={rho_th:.4g} ({src_th})  "
                f"Vmin={V_min:.4g}  (Δρ={rho_th - r['rho']:+.4g})"
            )
        else:
            print(f"  ρ={r['rho']:.4g}  emp={r['src']}  no theory")
        prepared.append(entry)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    wmax = float(args.wmax)
    emp_labeled = False
    theory_labeled = False

    for ax, panel in zip(axes, ("fo", "deep")):
        for e in prepared:
            r = e["run"]
            if panel == "fo" and r["rho"] < args.fo_rho_min:
                continue
            if panel == "deep" and r["rho"] > args.deep_rho_max:
                continue

            color = cmap(norm(e["c"]))
            mask = e["mask"]
            if e["th"] is not None and mask.any():
                # Emp onto raw theory gauge (theory unshifted = amalgam).
                emp_min = float(np.nanmin(e["action"][mask]))
                emp_plot = e["action"] - emp_min + e["V_min"]
                lbl_e = None
                if not emp_labeled:
                    lbl_e = rf"emp $-\log p$ (shifted $\to$ theory gauge)"
                    emp_labeled = True
                ax.plot(
                    e["centers"][mask],
                    emp_plot[mask],
                    color=color,
                    lw=2.0,
                    label=lbl_e,
                    zorder=3,
                )
                lbl_t = None
                if not theory_labeled:
                    lbl_t = r"theory $V(w)$ raw (dashed; same as amalgam)"
                    theory_labeled = True
                ax.plot(
                    e["w_th"],
                    e["V_th"],
                    color=color,
                    lw=1.15,
                    ls="--",
                    alpha=0.95,
                    label=lbl_t,
                    zorder=2,
                )
            elif mask.any():
                ax.plot(
                    e["centers"][mask],
                    e["action"][mask],
                    color=color,
                    lw=2.0,
                    zorder=3,
                )

        ax.axhline(0.0, color="0.55", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-wmax, wmax)

    axes[0].set_ylim(args.fo_ymin, args.fo_ymax)
    if deep_vmins:
        ymin = min(deep_vmins)
        pad = 0.04 * abs(ymin)
        axes[1].set_ylim(ymin - pad, 30.0)
    else:
        axes[1].set_ylim(-50.0, 30.0)

    axes[0].set_title(
        r"Pure cubic $\chi=N$ ρ-ray: empirical $W_0$ on amalgam theory scale"
        "\n"
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$, $\mathrm{he1}=0$; "
        r"color = $\rho$ (viridis)"
    )
    axes[0].set_ylabel(r"$V(w)$  (FO / catch-up window)")
    axes[1].set_ylabel(r"$V(w)$  (deep continuation)")
    axes[1].set_xlabel(r"read-in weight $w_0$ (teacher axis)")
    axes[0].legend(frameon=False, fontsize=8.0, loc="upper right")

    # Per-ρ color legend via colorbar (same style as amalgam).
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.035)
    cbar.set_ticks(c_plot)
    cbar.set_ticklabels([f"{r['rho']:.3g}" for r in runs])
    cbar.ax.set_ylabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$", rotation=270, labelpad=14)

    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
