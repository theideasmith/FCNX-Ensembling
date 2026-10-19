#!/usr/bin/env python3
"""Plot deep-ρ self-consistent χ=N ray, with optional NNGP L3 overlay."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path):
    return np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/summary_deep_equilibrium.csv",
        help="χ=N self-consistent deep ray",
    )
    ap.add_argument(
        "--nngp",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/summary_deep_nngp_frozen.csv",
        help="NNGP L3 (ridge κ/(sa0 P), prior kernel; no N on κ/sa0)",
    )
    ap.add_argument(
        "--chi1",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/summary_deep_chi1.csv",
        help="optional self-consistent χ=1 deep ray",
    )
    ap.add_argument(
        "--outdir",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = _load(Path(args.summary))
    rho = np.asarray(d["rho"], float)
    L3 = np.asarray(d["L3"], float)
    pi = np.asarray(d["pi_out"], float)
    Vout = np.asarray(d["V_out"], float)
    Vin = np.asarray(d["V_in"], float)
    nwells = np.asarray(d["nwells"], float)
    order = np.argsort(-rho)
    rho, L3, pi, Vout, Vin, nwells = (
        rho[order], L3[order], pi[order], Vout[order], Vin[order], nwells[order]
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.8), sharex=True)
    ax = axes[0]
    ax.plot(rho, L3, "o-", color="C0", lw=2.2, ms=5, label=r"$\chi=N$ self-consistent")
    nngp_path = Path(args.nngp)
    if nngp_path.is_file():
        n = _load(nngp_path)
        rn, Ln = np.asarray(n["rho"], float), np.asarray(n["L3"], float)
        on = np.argsort(-rn)
        ax.plot(rn[on], Ln[on], "s--", color="C3", lw=2.0, ms=4.5,
                label=r"NNGP (prior, $\rho'=\kappa/(\mathrm{sa0}\,P)$)")
    chi1_path = Path(args.chi1)
    if chi1_path.is_file() and chi1_path.stat().st_size > 50:
        c1 = _load(chi1_path)
        if "L3" in (c1.dtype.names or ()):
            r1, L1 = np.asarray(c1["rho"], float), np.asarray(c1["L3"], float)
            o1 = np.argsort(-r1)
            ax.plot(r1[o1], L1[o1], "^:", color="C2", lw=1.6, ms=4,
                    label=r"$\chi=1$ self-consistent")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.9)
    ax.set_ylabel(r"$L_3$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(
        r"Pure-cubic deep $\rho$-ray: $\chi=N$ vs NNGP"
        "\n"
        r"$d=16,\ N=200,\ P=10^3,\ \mathrm{sa0}=0.03$"
    )

    ax = axes[1]
    ax.plot(rho, np.maximum(-Vin, 1e-3), "o--", color="C0", lw=1.4, ms=4, label=r"$\chi=N$ $-V_{\mathrm{in}}$")
    ax.plot(rho, np.maximum(-Vout, 1e-3), "s-", color="C0", lw=1.8, ms=4, label=r"$\chi=N$ $-V_{\mathrm{out}}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"well depth $-V$ (log)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(rho, pi, "o-", color="k", lw=1.8, ms=4, label=r"$\chi=N$ $\pi_{\mathrm{out}}$")
    ax2 = ax.twinx()
    ax2.plot(rho, nwells, "s--", color="C2", lw=1.2, ms=4, label="nwells")
    ax.set_ylabel(r"$\pi_{\mathrm{out}}$")
    ax2.set_ylabel("nwells")
    ax.set_xlabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$ (decreasing $\rightarrow$)")
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="best")

    fig.tight_layout()
    out = outdir / "L3_deep_vs_rho.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
