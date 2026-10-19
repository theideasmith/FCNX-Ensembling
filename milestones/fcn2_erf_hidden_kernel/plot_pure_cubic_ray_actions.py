#!/usr/bin/env python3
"""Overlay raw cavity actions V(w) along a ray, colored by beta (viridis)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, help="CSV from --dump-ray-actions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.genfromtxt(args.actions, delimiter=",", names=True)
    betas = np.unique(data["beta"])
    betas.sort()

    out = Path(args.out) if args.out else Path(args.actions).with_name(
        Path(args.actions).name.replace("tmp_pure_cubic_ray_actions_", "pure_cubic_ray_actions_").replace(".csv", ".png")
    )

    # log-normalize beta onto viridis (ray is log-spaced)
    logb = np.log10(betas)
    norm = mpl.colors.Normalize(vmin=float(logb.min()), vmax=float(logb.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for β in betas:
        m = data["beta"] == β
        w = data["w"][m]
        V = data["V"][m]
        # sort by w in case of any disorder
        order = np.argsort(w)
        color = cmap(norm(np.log10(β)))
        ax.plot(w[order], V[order], color=color, lw=1.6, label=rf"$\beta={β:g}$")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10} \beta$")
    # tick labels as actual beta values
    cbar.set_ticks(logb)
    cbar.set_ticklabels([f"{b:g}" for b in betas])
    cbar.ax.set_ylabel(r"ray parameter $\beta$", rotation=270, labelpad=14)

    ax.set_xlabel(r"read-in weight $w$ (teacher axis)")
    ax.set_ylabel(r"action $V(w)$  (= $-\log p$, raw)")
    finv = float(np.nanmedian(data["Finv"]))
    ax.set_title(
        rf"Pure-cubic $\chi=N$ cavity actions along ray (Finv$\approx${finv:.0f})"
        "\n"
        r"each curve = eq $V(w)$ at one $\beta$; color = $\beta$ (viridis)"
    )
    ax.set_xlim(float(np.min(data["w"])), float(np.max(data["w"])))
    pad = 0.04 * (float(np.max(data["V"])) - float(np.min(data["V"])) + 1.0)
    ax.set_ylim(float(np.min(data["V"])) - pad, float(np.max(data["V"])) + pad)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"betas: {betas.tolist()}")
    print(f"V range [{np.min(data['V']):.4g}, {np.max(data['V']):.4g}]")


if __name__ == "__main__":
    main()
