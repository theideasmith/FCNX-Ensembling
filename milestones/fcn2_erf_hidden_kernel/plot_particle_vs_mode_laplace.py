#!/usr/bin/env python3
"""Overlay mode-Laplace −log p (self-consistent f*) on particle-theory −log p.

Particle caches: action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory/
Laplace caches:  action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac/

Usage:
  python plot_particle_vs_mode_laplace.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PARTICLE = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory"
LAPLACE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
RHOS = (0.15, 0.08, 0.03, 0.015)


def tag(rho: float) -> str:
    return f"rho{rho:g}".replace(".", "p")


def particle_nlogp(w0: np.ndarray, wmax: float, bins: int):
    h, e = np.histogram(w0, bins=bins, range=(-wmax, wmax), density=True)
    xc = 0.5 * (e[1:] + e[:-1])
    ok = h > 0
    return xc[ok], -np.log(h[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle", type=Path, default=PARTICLE)
    ap.add_argument("--laplace", type=Path, default=LAPLACE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--bins", type=int, default=120)
    args = ap.parse_args()

    out = args.out or (args.particle / "particle_vs_mode_laplace.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharey=False)
    for ax, rho in zip(axes.ravel(), RHOS):
        pz = np.load(args.particle / f"{tag(rho)}_particle.npz")
        th = np.load(args.laplace / f"{tag(rho)}_scfrac.npz")
        x = np.asarray(th["x"], float)
        nlogp = np.asarray(th["nlogp"], float)
        win = np.abs(x) <= args.wmax

        for start, col in (("prior", "C3"), ("outer", "C0")):
            xc, nl = particle_nlogp(pz[f"w0_{start}"], args.wmax, args.bins)
            fo = float(pz[f"f_out_{start}"])
            ax.plot(xc, nl, color=col, lw=1.25, label=rf"particle {start} ($f_o$={fo:.3f})")

        f_s = float(th["f_s"])
        mu_s = float(th["mu_s"])
        ax.plot(
            x[win],
            nlogp[win],
            color="k",
            lw=1.4,
            ls="--",
            label=rf"mode-Laplace $f^*$={f_s:.3f}, $\mu_s$={mu_s:.2f}",
        )

        # mark specialist locations
        ax.axvline(mu_s, color="0.45", ls=":", lw=0.9)
        ax.axvline(-mu_s, color="0.45", ls=":", lw=0.9)

        ymin = min(float(np.nanmin(nlogp[win])), *(particle_nlogp(pz[f"w0_{s}"], args.wmax, args.bins)[1].min() for s in ("prior", "outer")))
        ax.set_xlim(-args.wmax, args.wmax)
        ax.set_ylim(ymin - 0.3, ymin + 6.5)
        ax.set_title(rf"$\rho={rho:g}$", fontsize=11)
        ax.set_xlabel(r"$w_0$")
        ax.set_ylabel(r"$-\log p(w_0)$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.suptitle(
        r"Particle theory on data vs mode-Laplace (self-consistent $f^*$)"
        "\n"
        r"pure cubic $\chi=N$: $d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
