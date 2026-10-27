#!/usr/bin/env python3
"""
Sweep of the read-in action over κ or σ_w²: -log p(w0) per group (collapsed HMC vs mean-field
theory, prior perp norm, no fitted offset), plus L1, L3, w0², fraction aligned against the swept parameter.

Default: all actions overlaid on one axis as normalized -log p(w0) with no shift (theory
V(w) + log Z; HMC -log of the normalized histogram), window sized to show every theory well and barrier, coloured by the swept
parameter (viridis; theory lines, HMC points; with --cavity the exact single-neuron action, averaged over neurons
and snapshots, is added dashed; it resolves barriers the histograms cannot reach). --panels gives one action panel per group instead. With --live, groups without
post-burn samples yet still get their theory curve in the overlay.

    python plot_readin_action_sweep.py --x kappa --groups rubin_em0p3_k0p1 ... rubin_em0p3_k0p3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_readin_action import load_group, theory_curve
from diag_cavity_action import cavity_analysis, group_args

OBS = (("L1", r"$L_1$"), ("L3", r"$L_3$"), ("w0_sq", r"$\langle w_0^2\rangle$"),
       ("frac_aligned", r"frac $|w_0|>2.5\sigma_W$"))


def hist_nlp(w0, edges):
    """Bin centres, -log density and Poisson error bars of the nonempty bins."""
    cnt, _ = np.histogram(w0, edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ok = cnt > 0
    return ctr[ok], -np.log(cnt[ok] / (cnt.sum() * np.diff(edges)[ok])), 1 / np.sqrt(cnt[ok])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--x", choices=["kappa", "s0", "d"], default="kappa", help="swept parameter")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live", action="store_true", help="include runs still in progress (samples so far)")
    ap.add_argument("--hist-lines", action="store_true", help="overlay: draw histograms as connected curves")
    ap.add_argument("--panels", action="store_true", help="one action panel per group instead of the overlay")
    ap.add_argument("--cavity", type=int, default=0, metavar="NSNAP",
                    help="overlay: draw HMC as the exact cavity action from NSNAP snapshots (dashed) instead of histograms")
    args = ap.parse_args()

    rows = []
    for g in args.groups:
        loaded = load_group(g, live=args.live)
        if loaded is None:
            a = group_args(g)
            print(f"  {g}: no post-burn samples yet")
            if a is None or args.panels:
                continue
            rows.append(dict(a=a, group=g, w0=None, n_snap=0))
            continue
        W, a, post = loaded
        w0 = W[:, 0]
        thr = 2.5 * np.sqrt(a["s0"] / a["d"])
        rows.append(dict(a=a, group=g, w0=w0, n_snap=len(w0) // a["N"],
                         hmc=dict(L1=post["L1_he1"], L3=post["L3"], w0_sq=np.mean(w0 ** 2),
                                  frac_aligned=np.mean(np.abs(w0) > thr))))
    if not rows:
        raise SystemExit("no groups with runs")
    gmax = max((np.abs(r["w0"]).max() for r in rows if r["w0"] is not None), default=0.0) * 1.05
    if not args.panels:
        # window from the theory: past the outermost well until the action is 8 nats above it, and up to
        # the tallest barrier, so no curve is cut off
        wf = np.linspace(0, 3, 3001)
        for r in rows:
            a = r["a"]
            nlp_f, _ = theory_curve(a, 1 + 2 * a["s0"] * (a["d"] - 1) / a["d"], wf)
            i_out = max([i for i in range(1, len(wf) - 1) if nlp_f[i] <= nlp_f[i - 1] and nlp_f[i] < nlp_f[i + 1]],
                        default=0)
            beyond = np.where((wf > wf[i_out]) & (nlp_f > nlp_f[i_out] + 8))[0]
            r["w_end"] = wf[beyond[0]] if len(beyond) else wf[-1]
            r["y_top"] = nlp_f[: i_out + 1].max()
        gmax = max(gmax, max(r["w_end"] for r in rows) * 1.03)
    for r in rows:
        a = r["a"]
        r["wmax"] = np.abs(r["w0"]).max() * 1.05 if (args.panels and r["w0"] is not None) else gmax
        r["w"] = np.linspace(-r["wmax"], r["wmax"], 2001)
        r["nlp"], r["th"] = theory_curve(a, 1 + 2 * a["s0"] * (a["d"] - 1) / a["d"], r["w"])
    if args.cavity and not args.panels:
        for r in rows:
            r["cav"] = cavity_analysis(r["group"], args.cavity, wmax=gmax, nw=301)
    xval = {"kappa": lambda a: a["T"] / 2, "s0": lambda a: a["s0"], "d": lambda a: a["d"]}[args.x]
    xlab = {"kappa": r"\kappa", "s0": r"\sigma_w^2", "d": "d"}[args.x]
    for r in rows:
        r["x"] = xval(r["a"])
    rows.sort(key=lambda r: r["x"])
    a0 = rows[0]["a"]
    sampled = [r for r in rows if r["w0"] is not None]

    print(f"\n{args.x:>6s} " + " ".join(f"{k + ' HMC':>11s} {k + ' th':>10s}" for k in ("L1", "L3", "w0_sq", "frac")))
    for r in sampled:
        print(f"{r['x']:>6g} " + " ".join(f"{r['hmc'][k]:>11.4f} {r['th'][k]:>10.4f}"
                                                for k in ("L1", "L3", "w0_sq", "frac_aligned"))
              + "   basins: " + ", ".join(f"{b['basin']}={b['weight']:.2f}" for b in r["th"]["basins"]))

    if args.panels:
        fig = plot_panels(sampled, args, xlab)
    else:
        fig = plot_overlay(rows, args, xlab)

    fixed = {"kappa": rf"$\sigma_w^2={a0['s0']:g}$", "s0": rf"$\kappa={a0['T'] / 2:.4g}$",
             "d": rf"invariant ray, $\sigma_w^2={a0['s0']:g}$"}[args.x]
    if args.x == "d":
        th0 = rows[0]["th"]
        fig.suptitle(rf"invariant ray: $d={rows[0]['a']['d']}\ldots{rows[-1]['a']['d']}$, $\lambda={th0['lam']:.4g}$, "
                     rf"$\hat C={th0['Chat']:.1f}$, $\epsilon={a0['eps']:.3g}$, $\sigma_w^2={a0['s0']:g}$", fontsize=14)
    else:
        fig.suptitle(rf"$d={a0['d']}$, $P={a0['P']}$, $N={a0['N']}$, {fixed}, "
                     rf"$\epsilon={a0['eps']:.3g}$, $\sigma_a^2={a0['sa0'] * a0['N']:.3g}/N$", fontsize=14)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / f"readin_action_sweep_{args.groups[0]}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


def plot_obs(axes, rows, xlab, colors=None):
    """L1, L3, w0², fraction aligned against the swept parameter: HMC points, theory line."""
    xs = [r["x"] for r in rows]
    sampled = [r for r in rows if r["w0"] is not None]
    for ax, (k, lab) in zip(axes, OBS):
        ax.plot(xs, [r["th"][k] for r in rows], "-", color="0.35" if colors else "#d1495b", lw=2, label="theory",
                zorder=1)
        ax.scatter([r["x"] for r in sampled], [r["hmc"][k] for r in sampled], s=40, zorder=2, label="HMC",
                   c=[colors[r["x"]] for r in sampled] if colors else "#2a6fdb", edgecolors="white", linewidths=1)
        ax.set_xlabel(rf"${xlab}$", fontsize=12)
        ax.set_title(lab, fontsize=13)
    axes[0].legend(frameon=False, fontsize=10)


def plot_overlay(rows, args, xlab):
    """One action axis with every group, coloured by the swept parameter, above the observables row."""
    xs = np.array([r["x"] for r in rows])
    norm = plt.Normalize(xs.min(), xs.max() if xs.max() > xs.min() else xs.min() + 1)
    cmap = plt.cm.viridis
    colors = {r["x"]: cmap(0.9 * norm(r["x"])) for r in rows}   # stop short of viridis' pale yellow end
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1], hspace=0.35, wspace=0.3, top=0.94)
    ax = fig.add_subplot(gs[0, :])
    edges = np.linspace(-rows[0]["wmax"], rows[0]["wmax"], args.bins + 1)
    ymin, ytop = 0.0, max(max(r["y_top"] for r in rows) + 1.5, 6.0)
    for r in rows:
        ax.plot(r["w"], r["nlp"], color=colors[r["x"]], lw=2, zorder=1)
        ymin = min(ymin, r["nlp"].min())
        if r.get("cav") is not None:
            ax.plot(r["cav"]["w"], r["cav"]["nlp_exact"], color=colors[r["x"]], lw=2, ls=(0, (4, 2)), zorder=2)
            ymin = min(ymin, r["cav"]["nlp_exact"].min())
        if r["w0"] is not None:
            ctr, nlp, err = hist_nlp(r["w0"], edges)
            if args.hist_lines:
                # one connected curve per run of adjacent nonempty bins (gaps where the histogram is empty)
                gap = np.where(np.diff(ctr) > 1.5 * (edges[1] - edges[0]))[0] + 1
                for seg_c, seg_n in zip(np.split(ctr, gap), np.split(nlp, gap)):
                    ax.plot(seg_c, seg_n, "-", color=colors[r["x"]], lw=1.3, alpha=0.9, zorder=2)
            else:
                ax.errorbar(ctr, nlp, yerr=err, fmt="o", ms=3.5, color=colors[r["x"]], elinewidth=0.7,
                            mec="white", mew=0.5, zorder=2)
            ytop = max(ytop, np.percentile(nlp, 98) + 1)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(ymin - 0.5, ytop)
    ax.set_xlabel(r"$w_0$ (read-in weight along the teacher)", fontsize=13)
    ax.set_ylabel(r"$-\log p(w_0)$  shifted by log normalization", fontsize=13)
    ax.plot([], [], "-", color="0.3", lw=2, label="theory")
    if args.hist_lines:
        ax.plot([], [], "-", color="0.3", lw=1.3, label="collapsed HMC histogram, $-\\log$ normalized density")
    else:
        ax.plot([], [], "o", color="0.3", ms=4, label="collapsed HMC histogram")
    if args.cavity:
        ax.plot([], [], color="0.3", lw=2, ls=(0, (4, 2)), label="collapsed HMC: exact single-neuron action")
    ax.legend(frameon=False, fontsize=11, loc="upper center")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(norm.vmin, norm.vmin + (norm.vmax - norm.vmin) / 0.9))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label(rf"${xlab}$", fontsize=13)
    cb.set_ticks(xs)
    cb.set_ticklabels([f"{x:g}" + ("" if any(r["x"] == x and r["w0"] is not None for r in rows) else " (pending)")
                       for x in xs])
    plot_obs([fig.add_subplot(gs[1, i]) for i in range(4)], rows, xlab, colors)
    return fig


def plot_panels(rows, args, xlab):
    """One action panel per group (HMC vs theory) above the observables row."""
    n = len(rows)
    fig, axes = plt.subplots(2, max(n, 4), figsize=(3.6 * max(n, 4), 7.6))
    for ax, r in zip(axes[0], rows):
        edges = np.linspace(-r["wmax"], r["wmax"], args.bins + 1)
        ctr, nlp_hist, err = hist_nlp(r["w0"], edges)
        shift = r["nlp"].min()
        ax.errorbar(ctr, nlp_hist - shift, yerr=err, fmt="o", ms=2.5, color="#2a6fdb", elinewidth=0.8, label="HMC")
        ax.plot(r["w"], r["nlp"] - shift, color="#d1495b", lw=2, label="theory")
        ax.set_xlim(-r["wmax"], r["wmax"])
        ax.set_ylim(-0.5, max(nlp_hist.max() - shift + 1.5, 6))
        ax.set_title(rf"${xlab}={r['x']:g}$" + (f"  ({r['n_snap']} snapshots)" if args.live else ""), fontsize=13)
        ax.set_xlabel(r"$w_0$", fontsize=12)
    axes[0, 0].set_ylabel(r"$-\log p(w_0)$ + const", fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=10)
    for ax in axes[0, n:]:
        ax.axis("off")
    plot_obs(axes[1], rows, xlab)
    for ax in axes[1, 4:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
