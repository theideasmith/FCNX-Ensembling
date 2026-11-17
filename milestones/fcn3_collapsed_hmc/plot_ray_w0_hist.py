#!/usr/bin/env python3
"""FCN3 read-in w0 along the ray (d = 75, 150, 300, 600): full theory, theory with the perpendicular directions
locked, and the sampler histogram in wide bins, all d on one panel (colour = d, magma), each divided by
beta = d / (2 sigma_w^2) so that all d share one scale (0.08 where there are > 10^5 samples, else 0.16) with Poisson errors.
Theory from the caches of plot_ray_readin_fcn2_vs_fcn3.py and plot_locked_perp.py (not re-solved).

    python plot_ray_w0_hist.py     -> plots/locked_perp/locked_perp_ray_fcn3_w0.png
"""
import numpy as np

import plot_onset as po
import plot_ray_readin_fcn2_vs_fcn3 as rr
from plot_locked_perp import C_LOCK, load_or_solve, sym

OUT = po.HERE / "plots" / "locked_perp" / "locked_perp_ray_fcn3_w0.png"
LIM = 1.6
SW2 = 0.5


def main():
    plt = po.style()
    d3 = rr.fcn3_data()
    _, f3l = load_or_solve("cuda:0", False)
    fig, ax = plt.subplots(figsize=(9, 6))
    lo, hi = np.inf, -np.inf
    for d in (75, 150, 300, 600):
        r, lk = d3[d], f3l[f"ray_{d}"]
        beta = d / (2 * SW2)                          # every term of the action grows like d: divide by beta
        c_ = rr.colour(d, plt)
        full, locked = (r["wq"], r["th"] / beta), sym(lk["w"], lk["nlp"] / beta, LIM)
        x = r["w0"]
        bw = 0.08 if len(x) > 1e5 else 0.16
        edges = np.arange(-LIM, LIM + bw / 2, bw)
        c, _ = np.histogram(x, edges)
        k = c > 0
        mid = 0.5 * (edges[1:] + edges[:-1])[k]
        y = -np.log(c[k] / (c.sum() * bw)) / beta
        ax.plot(*full, "-", color=c_, lw=2, zorder=2)
        ax.plot(*locked, "--", color=c_, lw=1.6, zorder=2)
        ax.errorbar(mid, y, yerr=1 / np.sqrt(c[k]) / beta, fmt="o", ms=5.5, color=c_, mec="white", mew=0.4,
                    elinewidth=0.8, capsize=0, zorder=3)
        lo = min(lo, y.min(), np.nanmin(full[1]), np.nanmin(locked[1]))
        hi = max(hi, max(np.nanmax(v[np.abs(u) < 1.0]) for u, v in (full, locked)))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(lo - 0.02, lo + 1.42 * (hi - lo))
    ax.set_xlabel("w$_0$  (read-in neuron)")
    ax.set_ylabel("−log p(w$_0$) / β,   β = d / (2σ$_w^2$)")
    hd = [plt.Line2D([], [], color=rr.colour(d, plt), lw=2, marker="o", mec="white", ms=6) for d in (75, 150, 300, 600)]
    hs = [plt.Line2D([], [], color=po.INK, lw=2), plt.Line2D([], [], color=po.INK, lw=1.6, ls="--"),
          plt.Line2D([], [], color=po.INK, marker="o", ls="", mec="white", ms=6)]
    ax.legend(hd + hs, [f"d = {d}, P = {3000 * d * d // 22500}" for d in (75, 150, 300, 600)] +
              ["full theory", "locked theory", "sampler (bin width 0.08, 0.16 at d = 75, 600)"],
              fontsize=9, loc="upper center", ncol=2)
    ax.set_title("FCN3 read-in w$_0$ along the ray (P = 3000 (d/150)$^2$, κ = 0.015 d/150, N$_1$ = N$_2$ = 1500)",
                 color=po.INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
