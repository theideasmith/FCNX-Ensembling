#!/usr/bin/env python3
"""Plot inner/outer well depths along a pure-cubic steepwell-exponent ray."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray", required=True, help="CSV from --ray-scan")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = np.genfromtxt(args.ray, delimiter=",", names=True)
    beta = r["beta"]
    out = Path(args.out) if args.out else Path(args.ray).with_suffix(".wells.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    axV, axDV, axMu, axPi = axes.ravel()

    # absolute well depths at equilibrium (own scale — not crushed by π=0)
    axV.plot(beta, r["V_in"], "o-", color="C0", label=r"$V_{\mathrm{in}}$ (eq)")
    axV.plot(beta, r["V_out"], "s-", color="C3", label=r"$V_{\mathrm{out}}$ (eq)")
    axV.set_ylabel(r"raw $V$ at well (eq)")
    axV.set_title("eq well depths (both drop; outer stays ~3 above)")
    axV.legend(frameon=False, fontsize=8)
    axV.grid(True, alpha=0.3)

    # gap: negative means outer below inner — eq on left axis, π=0 on right
    axDV.axhline(0.0, color="k", lw=0.8)
    axDV.plot(beta, r["dV_raw"], "o-", color="C2", label=r"$dV$ (eq)")
    axDV.set_ylabel(r"$dV_{\mathrm{raw}}$ (eq)", color="C2")
    axDV.tick_params(axis="y", labelcolor="C2")
    axDV2 = axDV.twinx()
    axDV2.plot(beta, r["dV_raw_pi0"], "s--", color="C3", label=r"$dV$ ($\pi=0$)")
    axDV2.set_ylabel(r"$dV_{\mathrm{raw}}$ ($\pi=0$)", color="C3")
    axDV2.tick_params(axis="y", labelcolor="C3")
    axDV.set_title(r"$dV<0$ ⇒ outer below inner (only at $\pi=0$ here)")
    h1, l1 = axDV.get_legend_handles_labels()
    h2, l2 = axDV2.get_legend_handles_labels()
    axDV.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    axDV.grid(True, alpha=0.3)

    axMu.plot(beta, r["mu_out"], "s-", color="C3", label=r"$\mu_{\mathrm{out}}$ (eq)")
    if "mu_out_pi0" in r.dtype.names:
        axMu.plot(beta, np.abs(r["mu_out_pi0"]), "s--", color="C3", alpha=0.7, label=r"$|\mu_{\mathrm{out}}|$ ($\pi=0$)")
    axMu.plot(beta, np.abs(r["mu_in"]), "o-", color="C0", label=r"$|\mu_{\mathrm{in}}|$ (eq)")
    axMu.set_ylabel(r"well location $|\mu|$")
    axMu.set_title("well locations")
    axMu.legend(frameon=False, fontsize=8)
    axMu.grid(True, alpha=0.3)

    axPi.plot(beta, r["pi_out"], "o-", color="k", label=r"$\pi_{\mathrm{out}}$ (eq)")
    axPi.set_ylabel(r"$\pi_{\mathrm{out}}$")
    axPi.set_xlabel(r"ray parameter $\beta$")
    axPi.set_title("outer population")
    axPi.set_ylim(0, 1)
    axPi.legend(frameon=False, fontsize=8)
    axPi.grid(True, alpha=0.3)

    for ax in (axV, axDV, axMu):
        ax.set_xlabel(r"ray parameter $\beta$")
        ax.set_xscale("log")
    axPi.set_xscale("log")

    finv = float(np.nanmedian(r["Finv"]))
    fig.suptitle(
        rf"Pure-cubic $\chi=N$ ray at Finv$\approx${finv:.0f}: outer vs inner wells",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    # quick verdict
    eq_below = np.any(r["dV_raw"] < 0)
    pi0_below = np.any(r["dV_raw_pi0"] < 0)
    print(f"outer below inner at eq? {eq_below}")
    print(f"outer below inner at pi=0? {pi0_below}")
    print(f"dV_raw(eq) range [{np.nanmin(r['dV_raw']):.4g}, {np.nanmax(r['dV_raw']):.4g}]")
    print(f"dV_raw(pi0) range [{np.nanmin(r['dV_raw_pi0']):.4g}, {np.nanmax(r['dV_raw_pi0']):.4g}]")


if __name__ == "__main__":
    main()
