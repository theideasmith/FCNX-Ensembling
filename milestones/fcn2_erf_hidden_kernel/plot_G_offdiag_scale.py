#!/usr/bin/env python3
"""Plot G-matrix off-diagonal strength vs d on the scaled pure-cubic ρ-ray."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "action_h0_activation_plots/PureCubicChiN/G_offdiag_scale"


def load_table(directory: Path) -> np.ndarray:
    for name in ("G_offdiag_vs_d.csv", "G_offdiag_vs_d_from_log.csv"):
        csv = directory / name
        if csv.is_file() and csv.stat().st_size > 10:
            data = np.genfromtxt(csv, delimiter=",", names=True)
            if data.dtype.names and "d" in data.dtype.names:
                return data
    raise SystemExit(f"no usable CSV in {directory}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT)
    args = ap.parse_args()
    data = load_table(args.dir)
    names = set(data.dtype.names or ())
    rhos = np.unique(data["rho"])
    rhos = rhos[np.argsort(-rhos)]
    cmap = plt.get_cmap("viridis")
    c_plot = np.log10(np.clip(rhos, 1e-30, None))
    norm = mpl.colors.Normalize(vmin=float(c_plot.min()), vmax=float(c_plot.max()))

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.4), sharex=True)
    for rho, c in zip(rhos, c_plot):
        m = data["rho"] == rho
        o = np.argsort(data["d"][m])
        d = data["d"][m][o]
        color = cmap(norm(c))
        lbl = rf"$\rho={rho:.3g}$"
        rG = np.abs(data["rG"][m][o]) if "rG" in names else data["abs_rG"][m][o]
        axes[0, 0].plot(d, rG, color=color, lw=2.0, marker="o", ms=4, label=lbl)
        axes[0, 1].plot(
            d, data["absG13_over_G33"][m][o], color=color, lw=2.0, marker="o", ms=4
        )
        axes[1, 0].plot(
            d, data["absG13_over_G11"][m][o], color=color, lw=2.0, marker="o", ms=4
        )
        axes[1, 1].plot(d, data["L3"][m][o], color=color, lw=2.0, marker="o", ms=4)

    axes[0, 0].axhline(1.0, color="0.6", ls="--", lw=0.8)
    axes[0, 0].set_ylabel(r"$|r_G|=|G_{13}|/\sqrt{|G_{11}G_{33}|}$")
    axes[0, 0].set_title(r"fully coupled if $|r_G|\approx 1$")
    axes[0, 1].set_ylabel(r"$|G_{13}|/|G_{33}|$")
    axes[0, 1].set_title(r"offdiag vs cubic diagonal (driven channel)")
    axes[0, 1].set_ylim(0.0, 0.35)
    axes[1, 0].set_ylabel(r"$|G_{13}|/|G_{11}|$")
    axes[1, 0].set_title(r"offdiag vs linear diagonal (induced channel)")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_ylabel(r"$L_3$")
    axes[1, 1].set_title("cubic learnability along the scale")
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$d$")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        r"Pure cubic $\chi=N$ ρ-ray: $G_{13}$ does not close under Finv-flat large-$d$ scale"
        "\n"
        r"hold $\rho$, Finv (steepwell exponents from $d{=}16$, $N{=}200$, $P{=}10^3$, sa0${=}0.03$)",
        fontsize=11,
    )
    fig.tight_layout()
    out = args.dir / "G_offdiag_vs_d.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")

    m16 = data["d"] == 16
    print("\n=== d=16 anchor (experimental ρ-ray) ===")
    print(
        f"{'ρ':>8} {'|rG|':>8} {'|G13|/G11':>10} {'|G13|/G33':>10} {'L3':>8}"
    )
    for rho in sorted(np.unique(data["rho"][m16]), reverse=True):
        row = data[m16 & (data["rho"] == rho)][0]
        rg = abs(float(row["rG"])) if "rG" in names else float(row["abs_rG"])
        print(
            f"{rho:8.4g} {rg:8.3f} {row['absG13_over_G11']:10.3f} "
            f"{row['absG13_over_G33']:10.3f} {row['L3']:8.3f}"
        )


if __name__ == "__main__":
    main()
