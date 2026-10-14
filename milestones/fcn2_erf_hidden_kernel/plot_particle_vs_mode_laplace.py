#!/usr/bin/env python3
"""Three-way −log p overlay: network, particle theory, mode-Laplace.

Particle caches: action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory/
Laplace caches:  action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac/
Network: second half of precond Langevin (M2000 > M200) when present, else
particle npz w_net (A_snapshots 2nd half).

Usage:
  python plot_particle_vs_mode_laplace.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARTICLE = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory"
PRECOND = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/precond_langevin"
LAPLACE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
RHOS = (0.15, 0.08, 0.03, 0.015)
W_CUT = 0.6


def tag(rho: float) -> str:
    return f"rho{rho:g}".replace(".", "p")


def dens_nlogp(w0: np.ndarray, wmax: float, bins: int, min_count: int = 2):
    w0 = np.asarray(w0, float).reshape(-1)
    w0 = w0[np.isfinite(w0)]
    counts, e = np.histogram(w0, bins=bins, range=(-wmax, wmax))
    dens, _ = np.histogram(w0, bins=e, density=True)
    xc = 0.5 * (e[1:] + e[:-1])
    ok = counts >= min_count
    return xc[ok], -np.log(dens[ok] + 1e-300)


def load_network_w0(rho: float):
    """Prefer precond Langevin second half; fall back to particle npz w_net."""
    t = tag(rho)
    for mw in ("M2000", "M200"):
        snap_dir = PRECOND / f"{t}_{mw}" / "snapshots"
        snaps = sorted(snap_dir.glob("step_*.pt"), key=lambda s: int(s.stem.split("_")[1])) if snap_dir.is_dir() else []
        if len(snaps) < 4:
            continue
        use = snaps[len(snaps) // 2 :]
        w0 = np.concatenate(
            [
                torch.load(s, map_location="cpu", weights_only=False)["W0"][..., 0]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                for s in use
            ]
        )
        return w0, f"precond {mw} [{len(use)}/{len(snaps)}]"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle", type=Path, default=PARTICLE)
    ap.add_argument("--laplace", type=Path, default=LAPLACE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    out = args.out or (args.particle / "particle_theory_network_overlay.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), sharey=False)
    for ax, rho in zip(axes.ravel(), RHOS):
        pz = np.load(args.particle / f"{tag(rho)}_particle.npz")
        th = np.load(args.laplace / f"{tag(rho)}_scfrac.npz")
        x = np.asarray(th["x"], float)
        nlogp = np.asarray(th["nlogp"], float)
        win = np.abs(x) <= args.wmax

        w_net, src = load_network_w0(rho)
        if w_net is None:
            w_net = np.asarray(pz["w_net"], float)
            src = "A_snapshots"
        f_net = float(np.mean(np.abs(w_net) > W_CUT))
        xc_n, nl_n = dens_nlogp(w_net, args.wmax, bins=max(60, args.bins // 2), min_count=2)
        ax.plot(
            xc_n, nl_n, "o", ms=2.8, color="0.45", alpha=0.85,
            label=rf"network {src} ($f_o$={f_net:.3f})",
            zorder=2,
        )

        # particles: if prior≈outer, one curve; else both
        fo_p = float(pz["f_out_prior"])
        fo_o = float(pz["f_out_outer"])
        if abs(fo_p - fo_o) < 0.025:
            # average the two starts' histograms for a cleaner GT curve
            w_both = np.concatenate([pz["w0_prior"], pz["w0_outer"]])
            xc, nl = dens_nlogp(w_both, args.wmax, args.bins)
            ax.plot(
                xc, nl, color="C3", lw=1.35,
                label=rf"particle ($f_o$={0.5 * (fo_p + fo_o):.3f})",
                zorder=3,
            )
            ymins = [nl.min()]
        else:
            ymins = []
            for start, col in (("prior", "C3"), ("outer", "C0")):
                xc, nl = dens_nlogp(pz[f"w0_{start}"], args.wmax, args.bins)
                fo = float(pz[f"f_out_{start}"])
                ax.plot(xc, nl, color=col, lw=1.25, label=rf"particle {start} ($f_o$={fo:.3f})", zorder=3)
                ymins.append(nl.min())

        f_s = float(th["f_s"])
        mu_s = float(th["mu_s"])
        ax.plot(
            x[win],
            nlogp[win],
            color="k",
            lw=1.5,
            ls="--",
            label=rf"mode-Laplace $f^*$={f_s:.3f}, $\mu_s$={mu_s:.2f}",
            zorder=4,
        )
        ax.axvline(mu_s, color="0.55", ls=":", lw=0.8)
        ax.axvline(-mu_s, color="0.55", ls=":", lw=0.8)

        ymin = min(float(np.nanmin(nlogp[win])), float(nl_n.min()), *ymins)
        ax.set_xlim(-args.wmax, args.wmax)
        ax.set_ylim(ymin - 0.3, ymin + 6.5)
        ax.set_title(rf"$\rho={rho:g}$", fontsize=11)
        ax.set_xlabel(r"$w_0$")
        ax.set_ylabel(r"$-\log p(w_0)$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.2, frameon=False, loc="upper right")

    fig.suptitle(
        r"Network (precond Langevin when available) vs particle vs off-data mode-Laplace"
        r" (154 modes, $S=I$, SELF=full + FIX_TRACE)"
        "\n"
        r"pure cubic $\chi=N$: $d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    # also keep the old filename as an alias
    alias = args.particle / "particle_vs_mode_laplace.png"
    if out.resolve() != alias.resolve():
        fig.savefig(alias, dpi=160, bbox_inches="tight")
        print(f"saved {alias}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
