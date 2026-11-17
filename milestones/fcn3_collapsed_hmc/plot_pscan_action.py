#!/usr/bin/env python3
"""Read-in action at d = 150 for growing training-set size P at a fixed theory point (kappa ∝ P), against the theory.

Sampler: every post-burn read-in snapshot of one chain per P on data seed 2 (P = 3000: queue_hist.sh chain; P = 6000,
12000: pscan_d150 chains, possibly still running). Theory: fcn3_fixed.solve_fcn3 at n_max = 11 (the same for every P on
this line). Plotted: -log p(w0).  Error bars: Poisson on the counts, inflated by sqrt(2) for the correlation of snapshots 5 iterations
apart; points at a single count mark the counting floor.

    python plot_pscan_action.py      ->  plots/ray/pscan_action_d150.png
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIGW2, D = 0.5, 150
C_THEORY, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6b66", "#e4e3df"
RAMP = {3000: "#f0965f", 6000: "#eb6834", 12000: "#a8401a"}      # one hue, light -> dark with P
MARK = {3000: "o", 6000: "s", 12000: "^"}


def chains():
    out = {}
    pats = [f"runs/*_ray_d{D}_P3000_N1500x1500_prior_r12/snaps.npz", f"runs/*_pscan_d{D}_*/snaps.npz"]
    for pat in pats:
        for f in sorted(glob.glob(str(HERE / pat))):
            c = json.loads((Path(f).parent / "config.json").read_text())["args"]
            if c["seed"] != 2:
                continue
            z = np.load(f)
            post = z["iter"] >= c["burn"]
            if post.sum() < 5:
                continue
            out[c["P"]] = dict(w0=z["w0"][post].ravel(), q=z["q"][post].ravel(), n=int(post.sum()),
                               last=int(z["iter"].max()), done=(Path(f).parent / "summary.json").exists())
    return dict(sorted(out.items()))


def nlog_hist(x, edges, n_eff_scale=2.0):
    cnt, _ = np.histogram(x, edges)
    mid, wd = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    ok = cnt > 0
    return mid[ok], -np.log(cnt[ok] / (len(x) * wd[ok])), np.sqrt(n_eff_scale / cnt[ok]), 0.5 * wd[ok]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    ch = chains()
    th1 = np.load(HERE / "plots" / "cache" / "theory_readin_ray.npz")                 # marginal p(|w0|), n_max = 11
    fig, ax = plt.subplots(figsize=(9, 5.4))

    # left: -log p(w0), signed
    w0t, dt_ = th1[f"w0_{D}"], th1[f"dens_{D}"]
    keep = (dt_ > 1e-30) & (w0t > 0)
    xs = np.concatenate([-w0t[keep][::-1], w0t[keep]])
    ys = -np.log(0.5 * np.concatenate([dt_[keep][::-1], dt_[keep]]))
    ax.plot(xs, ys, color=C_THEORY, lw=2.2, label="theory (n$_{max}$ = 11), the same for every P", zorder=1)
    edges = np.arange(-1.4, 1.4001, 0.04)
    for P, c in ch.items():
        m, y, e, xe = nlog_hist(c["w0"], edges)
        lab = f"P = {P}" + ("" if c["done"] else f" (running, to it {c['last']})") + f", {c['n']} snapshots"
        ax.errorbar(m, y, xerr=xe, yerr=e, fmt=MARK[P], ms=4.5, color=RAMP[P], mec="white", mew=0.6, elinewidth=0.9,
                    capsize=0, label=lab, zorder=2 + P / 1e4)
    lo = float(ys.min())
    ax.set_ylim(lo - 1, lo + 14)
    ax.set_xlim(-1.4, 1.4)
    ax.set_xlabel("w$_0$")
    ax.set_ylabel("−log p(w$_0$)")
    ax.set_title("read-in action, d = 150, κ ∝ P (theory point fixed)", color=INK, loc="left")
    ax.legend(loc="upper center", fontsize=8.5)


    fig.suptitle("Neurons between the wells disappear as the training set grows (data seed 2, N = 1500)", color=INK,
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / f"pscan_action_d{D}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    for P, c in ch.items():
        a = np.abs(c["w0"])
        print(f"P={P}: {c['n']} snapshots, between (0.35-0.75) {np.mean((a > .35) & (a < .75)):.2e}, "
              f"outer {np.mean(a > .75):.4f}")


if __name__ == "__main__":
    main()
