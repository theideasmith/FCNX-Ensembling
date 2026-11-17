#!/usr/bin/env python3
"""Where the bottom of the outer read-in well lies, theory against the sampler, along the kappa0 = 0.015 ray.

    python plot_well_minima.py      ->  plots/ray/well_minima_ray.png
"""
from pathlib import Path

import numpy as np

from plot_outer_well import DS, HERE, chains, theory

C_THEORY, C_HMC, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b66", "#e4e3df"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    z = np.load(HERE / "plots" / "cache" / "theory_readin_ray.npz")
    fig, ax = plt.subplots(figsize=(7, 5))
    th = [theory(d, z)[3] for d in DS]
    ax.plot(DS, th, "-o", color=C_THEORY, lw=2, ms=8, mec="white", label="theory", zorder=3)
    first, first_fixed = True, True
    for d in DS:
        runs = chains(d)
        free = [c for c in runs if not c["fixed"]]
        fixed = [c for c in runs if c["fixed"]]
        y = np.array([c["wmin"] for c in free])
        x = d * np.exp(np.linspace(-0.04, 0.04, len(y)))
        ax.plot(x, y, "o", color=C_HMC, ms=6, mec="white", mew=0.6, alpha=0.9,
                label="sampler, number of outer neurons as it fell (one point per run)" if first else None, zorder=2)
        first = False
        for c in fixed:
            ax.plot([d], [c["wmin"]], "*", color=INK, ms=16, mec="white", mew=0.8, zorder=4,
                    label="sampler, number of outer neurons set to the theory's" if first_fixed else None)
            first_fixed = False
        print(f"d = {d}: theory {theory(d, z)[3]:.3f}, free runs {np.round(np.sort(y), 3)}, "
              f"set to theory's count {[round(c['wmin'], 3) for c in fixed]}")
    ax.set_xscale("log")
    ax.set_xticks(DS)
    ax.set_xticklabels([str(d) for d in DS])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("input dimension d")
    ax.set_ylabel("|w$_0$| at the bottom of the outer well")
    ax.set_title("Bottom of the outer read-in well: theory vs sampler", color=INK, loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / "well_minima_ray.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
