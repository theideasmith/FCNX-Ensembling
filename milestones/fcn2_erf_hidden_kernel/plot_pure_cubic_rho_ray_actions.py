#!/usr/bin/env python3
"""Single-panel V(w) overlay through the outer-well FO, colored by β (or α/ρ)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_KEEP = [0.20, 0.18, 0.16, 0.15, 0.145, 0.14, 0.135, 0.13]
# Steepwell α–β relation used for labeling when α is not in the CSV.
GAMMA = 1.25
ALPHA0 = 1.0


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
        help="comma-separated ρ values to keep (FO window)",
    )
    ap.add_argument(
        "--color-by",
        choices=("beta", "alpha", "rho"),
        default="beta",
        help="colorbar / line color key (default: beta)",
    )
    ap.add_argument(
        "--kappa-ref",
        type=float,
        default=None,
        help="κ at β=1 when deriving β=κ_ref/κ from a ρ-ray CSV (default: max κ in keep)",
    )
    ap.add_argument("--wmax", type=float, default=3.2)
    ap.add_argument("--ymin", type=float, default=-3.2)
    ap.add_argument("--ymax", type=float, default=4.5)
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    names = set(data.dtype.names or ())

    has_beta = "beta" in names
    has_alpha = "alpha" in names
    has_rho = "rho" in names
    has_kappa = "kappa" in names

    # Select curves ----------------------------------------------------------
    if has_rho:
        all_rhos = np.unique(data["rho"])

        def nearest_rho(target: float) -> float:
            return float(all_rhos[np.argmin(np.abs(all_rhos - target))])

        targets = [float(x) for x in args.keep.split(",") if x.strip()]
        keys_rho = [nearest_rho(t) for t in targets]
        # one representative row per ρ for meta
        meta = []
        for ρ in keys_rho:
            row = data[data["rho"] == ρ][0]
            κ = float(row["kappa"]) if has_kappa else float("nan")
            meta.append({"rho": ρ, "kappa": κ})
        κ_ref = (
            float(args.kappa_ref)
            if args.kappa_ref is not None
            else max(m["kappa"] for m in meta)
        )
        for m in meta:
            if has_beta:
                # pull true β from matching rows if present
                m["beta"] = float(data[data["rho"] == m["rho"]]["beta"][0])
            else:
                m["beta"] = κ_ref / m["kappa"]
            if has_alpha:
                m["alpha"] = float(data[data["rho"] == m["rho"]]["alpha"][0])
            else:
                m["alpha"] = ALPHA0 * (m["beta"] ** GAMMA)
            m["key"] = m["rho"]
            m["mask"] = data["rho"] == m["rho"]
    elif has_beta:
        betas = np.unique(data["beta"])
        betas.sort()
        meta = []
        for β in betas:
            row = data[data["beta"] == β][0]
            m = {
                "beta": float(β),
                "alpha": float(row["alpha"]) if has_alpha else ALPHA0 * (float(β) ** GAMMA),
                "rho": float(row["kappa"] / (row["sa0"] * row["P"]))
                if ("kappa" in names and "sa0" in names and "P" in names)
                else float("nan"),
                "kappa": float(row["kappa"]) if has_kappa else float("nan"),
                "key": float(β),
                "mask": data["beta"] == β,
            }
            meta.append(m)
    else:
        raise SystemExit("CSV needs a 'rho' or 'beta' column")

    phase: dict[float, str] = {}
    if args.summary and Path(args.summary).is_file():
        summary = np.genfromtxt(
            args.summary, delimiter=",", names=True, dtype=None, encoding=None
        )
        for row in np.atleast_1d(summary):
            if "rho" in summary.dtype.names:
                phase[float(row["rho"])] = str(row["phase"]).replace("_", " ")
            elif "beta" in summary.dtype.names:
                phase[float(row["beta"])] = str(row["phase"]).replace("_", " ")

    color_vals = np.asarray([m[args.color_by] for m in meta], dtype=float)
    # log scale for β/α (ray-like); linear for ρ
    if args.color_by in ("beta", "alpha"):
        c_plot = np.log10(color_vals)
        cbar_label = rf"$\{args.color_by}$"
    else:
        c_plot = color_vals
        cbar_label = r"$\rho=\kappa/(\mathrm{sa0}\,P)$"
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))
    cmap = plt.get_cmap("viridis")

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_", "").replace(".csv", ".png")
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for m, c in zip(meta, c_plot):
        w = data["w"][m["mask"]]
        V = data["V"][m["mask"]]
        order = np.argsort(w)
        w, V = w[order], V[order]
        color = cmap(norm(c))
        ph = ""
        if phase:
            # match on rho if available else beta
            if has_rho and m["rho"] == m["rho"]:
                key = min(phase.keys(), key=lambda x: abs(x - m["rho"]))
            else:
                key = min(phase.keys(), key=lambda x: abs(x - m["beta"]))
            ph = f" ({phase[key]})"
        if args.color_by == "beta":
            label = rf"$\beta={m['beta']:.3g}${ph}"
        elif args.color_by == "alpha":
            label = rf"$\alpha={m['alpha']:.3g}${ph}"
        else:
            label = rf"$\rho={m['rho']:.3g}${ph}"
        ax.plot(w, V, color=color, lw=2.0, label=label)
        wm, Vm = local_minima(w, V)
        keep = (np.abs(wm) <= args.wmax) & (Vm >= args.ymin - 0.1) & (Vm <= args.ymax + 0.1)
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
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_xlabel(r"read-in weight $w$")
    ax.set_ylabel(r"action $V(w)$ (raw)")
    ax.set_title(
        r"Pure cubic $\chi=N$, $P=10^3$, $\mathrm{sa0}=0.03$"
        "\n"
        + rf"outer-well FO; color = $\{args.color_by}$ (viridis)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_ticks(c_plot)
    if args.color_by == "beta":
        cbar.set_ticklabels([f"{m['beta']:.3g}" for m in meta])
    elif args.color_by == "alpha":
        cbar.set_ticklabels([f"{m['alpha']:.3g}" for m in meta])
    else:
        cbar.set_ticklabels([f"{m['rho']:.3g}" for m in meta])
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    for m in meta:
        print(
            f"  β={m['beta']:.4g}  α={m['alpha']:.4g}  ρ={m['rho']:.4g}  κ={m['kappa']:.4g}"
        )


if __name__ == "__main__":
    main()
