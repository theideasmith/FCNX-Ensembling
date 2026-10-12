#!/usr/bin/env python3
"""Plot deep-ρ self-consistent ray toward L3 → 1."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3/summary_deep_equilibrium.csv",
    )
    ap.add_argument(
        "--outdir",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_deep_l3",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = np.genfromtxt(args.summary, delimiter=",", names=True, dtype=None, encoding=None)
    d = np.atleast_1d(d)
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

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.6), sharex=True)
    ax = axes[0]
    ax.plot(rho, L3, "o-", color="C0", lw=2.0, ms=5)
    ax.axhline(1.0, color="0.5", ls="--", lw=0.9)
    ax.set_ylabel(r"$L_3$")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        r"Self-consistent pure-cubic deep $\rho$-ray"
        "\n"
        r"$d=16,\ N=\chi=200,\ P=10^3,\ \mathrm{sa0}=0.03$"
    )

    ax = axes[1]
    # wells plunge to O(10^3); plot −V on log scale
    ax.plot(rho, np.maximum(-Vin, 1e-3), "o--", color="C0", lw=1.4, ms=4, label=r"$-V_{\mathrm{in}}$")
    ax.plot(rho, np.maximum(-Vout, 1e-3), "s-", color="C3", lw=1.8, ms=4, label=r"$-V_{\mathrm{out}}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"well depth $-V$ (log)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(rho, pi, "o-", color="k", lw=1.8, ms=4, label=r"$\pi_{\mathrm{out}}$")
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
    for r, l, p, vo, nw in zip(rho[::-1], L3[::-1], pi[::-1], Vout[::-1], nwells[::-1]):
        print(f"  ρ={r:.4g}  L3={l:.4g}  π={p:.4g}  V_out={vo:.4g}  nwells={int(nw)}")


if __name__ == "__main__":
    main()
