#!/usr/bin/env python3
"""Second hidden layer on the onset line (d = 150, kappa = 0.015) on a wide x range: does the theory have further
wells in h|He1 or h|He3 beyond the range of plot_onset's per-P figure?

The theory's marginals are recomputed at the cached solutions (plots/cache/theory_onset_d150_v6.npz) with
plot_onset.hidden_marginals on h1 in [-8, 8] and h3 in [-3, 3]; the plot shows h1 in [-3, 3] and h3 in [-1, 1] and
40 nats of action. Every local minimum of the theory's action over the full computed range is printed.

    python plot_hidden_wide.py [--recompute]   ->  plots/onset/onset_d150_hidden_wide.png
"""
import argparse
from pathlib import Path

import numpy as np

import plot_onset as po

HERE = Path(__file__).resolve().parent
CACHE = HERE / "plots" / "cache" / "theory_onset_d150_hidden_wide.npz"
PS = [750, 1500, 3000, 6000, 12000, 24000, 48000]


def wells(g, a):
    """Local minima and maxima of the action a on the grid g, for g >= 0, as (position, height above the minimum)."""
    a = a - a.min()
    mins = [(g[j], a[j]) for j in range(1, len(g) - 1) if g[j] >= 0 and a[j] < a[j - 1] and a[j] <= a[j + 1]]
    maxs = [(g[j], a[j]) for j in range(1, len(g) - 1) if g[j] >= 0 and a[j] > a[j - 1] and a[j] >= a[j + 1]
            and a[j] < 600]                                   # beyond ~690 the density has underflowed
    return mins, maxs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    groups = po.by_P(po.sampler_runs())
    if CACHE.exists() and not a.recompute:
        w = dict(np.load(CACHE))
    else:
        z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
        w = {}
        for P in PS:
            r = {k: z[f"{k}_{P}"] for k in ("G", "t", "ks", "H", "hs", "qs")}
            r["gt"] = float(z[f"gt_{P}"])
            w[f"h1g_{P}"], w[f"lp1_{P}"], w[f"h3g_{P}"], w[f"lp3_{P}"] = po.hidden_marginals(
                r, a.device, h1max=8.0, n1=4001, n3=6001, h3max=3.0)
        np.savez(CACHE, **w)
    hid = {P: po.sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()])[0] for P in PS}

    for P in PS:
        for j, i in ((1, 0), (3, 1)):
            g, lp = w[f"h{j}g_{P}"], w[f"lp{j}_{P}"]
            mins, maxs = wells(g, -lp)
            x = np.abs(hid[P][:, i])
            print(f"P={P} h|He{j} (theory over |h| <= {g.max():.0f}): minima "
                  + (", ".join(f"{m:.3f} ({h:.1f})" for m, h in mins) or "only at 0")
                  + "; maxima " + (", ".join(f"{m:.3f} ({h:.1f})" for m, h in maxs) or "none")
                  + f";  sampler: largest |h| {x.max():.3f}")

    plt = po.style()
    fig, axs = plt.subplots(2, len(PS), figsize=(3.6 * len(PS), 7.4), squeeze=False)
    for c, P in enumerate(PS):
        for i, (j, lim, xl) in enumerate(((1, 3.0, "h | He$_1$"), (3, 1.0, "h | He$_3$"))):
            ax = axs[i, c]
            g, yt = w[f"h{j}g_{P}"], -w[f"lp{j}_{P}"]
            k = np.abs(g) <= lim
            ax.plot(g[k], yt[k], color=po.C_THEORY, lw=2, zorder=2)
            lo = yt.min()
            x = hid[P][:, i]
            edges = np.linspace(-lim, lim, 201)
            mid = 0.5 * (edges[1:] + edges[:-1])
            cnt, _ = np.histogram(x, edges)
            kk = cnt > 0
            y = -np.log(cnt[kk] / (cnt.sum() * np.diff(edges)[kk]))
            ax.errorbar(mid[kk], y, yerr=1 / np.sqrt(cnt[kk]), xerr=0.5 * np.diff(edges)[kk], fmt="o", ms=2.6,
                        color=po.C_HMC, mec="white", mew=0.3, elinewidth=0.7, capsize=0, zorder=3)
            lo = min(lo, y.min())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(lo - 1, lo + 40)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"−log p({xl})")
            if i == 0:
                ax.set_title(f"P = {P}", color=po.INK, loc="left", fontsize=12)
    fig.legend([plt.Line2D([], [], color=po.C_THEORY, lw=2),
                plt.Line2D([], [], color=po.C_HMC, marker="o", ls="none", ms=4)],
               ["theory (computed out to |h| = 8 for He$_1$ and 3 for He$_3$; no further wells)",
                "sampler (every hidden neuron of the saved networks, all datasets pooled)"],
               loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Second hidden layer on a wide range, d = {po.D}, N$_1$ = N$_2$ = {po.N}, κ = {po.KAPPA}",
                 color=po.INK, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    out = HERE / "plots" / "onset" / "onset_d150_hidden_wide.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
