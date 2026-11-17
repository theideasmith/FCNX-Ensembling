#!/usr/bin/env python3
"""Outer read-in well along the kappa0 = 0.015 ray: does the location of its minimum converge to the theory's?

Top row (d = 150, 300, 600): -log p(|w0| | outer well), the distribution of |w0| WITHIN the outer well (|w0| > 0.6),
for every chain on the ray (P = 3000 (d/150)^2), against the theory's (n_max = 11). Conditioning on the well removes
the occupancy from the depth, so the curves compare shape and position only. Chains are coloured by their outer
occupancy, which is frozen in each chain (no read-in neuron changes well).
Bottom: the location of the outer-well minimum relative to the theory's, against the run's outer occupancy, with a
straight-line fit per d, read at the theory's occupancy (diamonds).

    python plot_outer_well.py      ->  plots/ray/outer_well_ray.png
"""
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIGW2 = 0.5
DS = (150, 300, 600)
P_RAY = {150: 3000, 300: 12000, 600: 48000}
C_THEORY, INK, MUTED, GRID = "#2a78d6", "#1f1f1e", "#6b6b66", "#e4e3df"
DCOL = {150: "#eb6834", 300: "#1baf7a", 600: "#7b5ce0"}
W_OUT = 0.6


def theory(d, z):
    """Theory's density of |w0|, its outer occupancy, and the |w0| of its outer-well minimum."""
    x, dens = z[f"w0_{d}"], z[f"dens_{d}"]
    p = dens * np.gradient(x)
    m = x >= W_OUT
    frac = p[m].sum() / p.sum()
    xmin = x[m][np.argmax(dens[m])]
    return x, dens, frac, xmin


def well_min(w, h=None):
    """|w0| of the outer-well minimum of -log p from samples: the peak of a Gaussian-smoothed histogram."""
    w = np.asarray(w, float)
    if h is None:
        h = max(0.01, 1.06 * w.std() * len(w) ** -0.2)
    g = np.linspace(W_OUT, 1.3, 1401)
    if len(w) > 20000:
        w = np.random.default_rng(0).choice(w, 20000, replace=False)
    dens = np.exp(-0.5 * ((g[:, None] - w[None, :]) / h) ** 2).sum(1)
    return float(g[np.argmax(dens)])


def chains(d):
    out = []
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_d{d}_P{P_RAY[d]}_*" / "config.json"))):
        c = json.loads(Path(f).read_text())["args"]
        r = Path(f).parent
        if "aborted" in str(r) or c.get("tag") != "ray":
            continue
        if (r / "snaps.npz").exists():
            s = np.load(r / "snaps.npz")
            post = s["iter"] >= c["burn"]
            if post.sum() < 3:
                continue
            w = np.abs(s["w0"][post]).astype(float)
            src = f"{int(post.sum())} snapshots"
        elif (r / "checkpoint.pt").exists():
            ck = torch.load(r / "checkpoint.pt", map_location="cpu", weights_only=False)
            w = (ck["ZW"][:, 0].double().abs() * math.sqrt(SIGW2 / d)).numpy()[None]
            src = "final state"
        else:
            continue
        o = w[w >= W_OUT]
        if len(o) == 0:
            continue
        out.append(dict(N=c["N1"], seed=c.get("seed", 0), init=c["init"], frac=float((w >= W_OUT).mean()),
                        fixed=str(c.get("group", "")).startswith("fixedn"),
                        wmin=well_min(o), w=o, src=src))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    z = np.load(HERE / "plots" / "cache" / "theory_readin_ray.npz")
    fig = plt.figure(figsize=(15, 9.2))
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1])
    axt = top.subplots(1, 3)
    axb = [bot.subplots(1, 1)]
    norm = Normalize(0.016, 0.028)
    cmap = plt.cm.Oranges
    corrected = {}
    for ax, d in zip(axt, DS):
        x, dens, tf, tm = theory(d, z)
        m = x >= W_OUT
        pc = dens[m] / (dens[m] * np.gradient(x)[m]).sum()
        ok = pc > 1e-12
        ax.plot(x[m][ok], -np.log(pc[ok]), color=C_THEORY, lw=2.4, label=f"theory: outer occupancy {tf:.4f}", zorder=5)
        ax.axvline(tm, color=C_THEORY, lw=1, ls=":")
        ch = chains(d)
        edges = np.arange(W_OUT, 1.3001, 0.02)
        mid = 0.5 * (edges[1:] + edges[:-1])
        for c in sorted(ch, key=lambda c: c["frac"]):
            cnt, _ = np.histogram(c["w"], edges)
            k = cnt > 0
            y = -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k]))
            col = cmap(0.35 + 0.65 * norm(c["frac"]))
            if c["fixed"]:
                ax.plot(mid[k], y, "-*", ms=8, lw=2.0, color="#111111",
                        label=f"sampler, outer occupancy set to {c['frac']:.4f}")
            else:
                ax.plot(mid[k], y, "-o", ms=3, lw=1.1, color=col, mec="white", mew=0.4, alpha=0.95,
                        label=f"sampler, N={c['N']}, data set {c['seed']}, {c['init']} start: {c['frac']:.4f}")
            ax.axvline(c["wmin"], color=col, lw=0.8, ls=":")
        ax.set_xlim(W_OUT, 1.25)
        ax.set_ylim(-3, 6)
        ax.set_xlabel("|w$_0$|")
        ax.set_title(f"d = {d}", color=INK, loc="left", fontsize=11)
        if d == DS[0]:
            ax.set_ylabel("−log p(|w$_0$| | outer well)")
        ax.legend(fontsize=6.5, loc="upper left")

        # bottom left: position offset vs occupancy, per d
        X = np.array([c["frac"] for c in ch])
        Y = np.array([100 * (c["wmin"] / tm - 1) for c in ch])
        b, a = np.polyfit(X, Y, 1)
        corrected[d] = (a + b * tf, len(X), X.min(), X.max(), tf)
        axb[0].scatter(X, Y, s=[40 if c["N"] == 1500 else 70 for c in ch],
                       marker=None, color=DCOL[d], edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"d = {d}: at the theory's occupancy {a + b * tf:+.1f}%")
        for c, xx, yy in zip(ch, X, Y):
            if c["fixed"]:
                axb[0].scatter([xx], [yy], s=260, marker="*", color=DCOL[d], edgecolor=INK, linewidth=0.8, zorder=6)
            if c["N"] == 3000:
                axb[0].scatter([xx], [yy], s=90, facecolor="none", edgecolor=DCOL[d], linewidth=1.2, zorder=4)
        xx = np.linspace(min(X.min(), tf) - 0.001, X.max() + 0.001, 50)
        axb[0].plot(xx, a + b * xx, color=DCOL[d], lw=1.2)
        axb[0].plot([tf], [a + b * tf], marker="D", ms=8, color=DCOL[d], mec=INK, mew=0.8, zorder=5)
    axb[0].axhline(0, color=C_THEORY, lw=1.5)
    axb[0].text(0.0158, 0.4, "theory", color=C_THEORY, fontsize=9)
    axb[0].set_xlabel("outer occupancy: fraction of read-in neurons in the outer well (|w$_0$| > 0.6); "
                      "it stays fixed during a sampler run")
    axb[0].set_ylabel("location of the outer-well minimum,\nsampler relative to theory (%)")
    axb[0].set_title("one point per sampler run; diamond = the line read at the theory's occupancy",
                     color=INK, loc="left", fontsize=10)
    axb[0].legend(fontsize=8.5, loc="lower left")

    top.suptitle("FCN3 read-in outer well along the κ$_0$ = 0.015 ray (dotted lines: location of each minimum)",
                 color=INK,
                 x=0.01, ha="left", fontsize=12)
    out = HERE / "plots" / "ray" / "outer_well_ray.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for d in DS:
        c, n, lo, hi, tf = corrected[d]
        print(f"d={d}: {n} sampler runs, occupancy {lo:.4f}-{hi:.4f} (theory {tf:.4f}) -> offset at theory occupancy {c:+.2f}%"
              f"{'  (extrapolated)' if tf < lo or tf > hi else ''}")


if __name__ == "__main__":
    main()
