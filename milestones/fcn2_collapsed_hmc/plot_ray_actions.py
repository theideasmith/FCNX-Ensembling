#!/usr/bin/env python3
"""
Read-in action -log p(w0) along an invariant α/β ray (not rescaled by β, so the wells deepen with d). All d on one axis, coloured by d:
HMC histogram (points), corrected theory (relaxed perp norm + He1..He7; solid), original theory (dashed).
Each -log p is a normalized density shifted only by its log normalization.
Also prints HMC vs both theories per d (outer-well position and width, <w0²>, aligned fraction).

    python plot_ray_actions.py --groups grok_k0p1 raypaper_b2 raypaper_b4 raypaper_b8 raypaper_b16 --out ray_paper
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_readin_action import load_group, theory_curve
from readin_action_theory_q import analyse_q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--out", default="ray_actions")
    ap.add_argument("--wmax", type=float, default=1.9)
    args = ap.parse_args()

    rows = []
    for g in args.groups:
        loaded = load_group(g, live=True)
        if loaded is None:
            print(f"  {g}: no samples yet")
            continue
        W, a, _ = loaded
        d = a["d"]
        beta = d / (2 * a["s0"])
        w = np.linspace(-args.wmax, args.wmax, 3801)
        nlp_o, th_o = theory_curve(a, 1 + 2 * a["s0"] * (d - 1) / d, w)
        q = analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], a["chi"], ks=(1, 3, 5, 7))
        wq = np.concatenate([-q["w"][::-1], q["w"][1:]]); nq = np.concatenate([q["nlp"][::-1], q["nlp"][1:]])
        m = np.abs(wq) <= args.wmax
        w0 = W[:, 0]
        thr = 2.5 * np.sqrt(a["s0"] / d)
        outer = np.abs(w0) > 0.5
        rows.append(dict(d=d, P=a["P"], beta=beta, w0=w0, w=w, orig=nlp_o, wq=wq[m], corr=nq[m],
                         hmc=dict(well=np.median(np.abs(w0[outer])) if outer.sum() > 20 else np.nan,
                                  width=np.abs(w0[outer]).std() if outer.sum() > 20 else np.nan,
                                  w0_sq=np.mean(w0 ** 2), frac=np.mean(np.abs(w0) > thr)),
                         orig_s=dict(w0_sq=th_o["w0_sq"], frac=th_o["frac_aligned"]),
                         corr_s=dict(well=q["outer_well"], w0_sq=q["w0_sq"], frac=q["frac_aligned"])))
    rows.sort(key=lambda r: r["d"])

    print(f"\n{'d':>5s} {'P':>6s} | {'HMC well':>8s} {'width':>6s} {'w0²':>8s} {'frac':>6s} | {'corr well':>9s} {'w0²':>8s} {'frac':>6s} | {'orig w0²':>8s} {'frac':>6s}")
    for r in rows:
        h, c, o = r["hmc"], r["corr_s"], r["orig_s"]
        print(f"{r['d']:>5d} {r['P']:>6d} | {h['well']:>8.3f} {h['width']:>6.3f} {h['w0_sq']:>8.5f} {h['frac']:>6.3f} | "
              f"{c['well']:>9.3f} {c['w0_sq']:>8.5f} {c['frac']:>6.3f} | {o['w0_sq']:>8.5f} {o['frac']:>6.3f}")

    ds = [r["d"] for r in rows]
    norm = plt.Normalize(np.log2(min(ds)), np.log2(max(ds)) if max(ds) > min(ds) else np.log2(min(ds)) + 1)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    edges = np.linspace(-args.wmax, args.wmax, 161)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ymin, ymax = np.inf, -np.inf
    for r in rows:
        col = plt.cm.viridis(0.9 * norm(np.log2(r["d"])))
        cnt, _ = np.histogram(r["w0"], edges)
        ok = cnt > 0
        nlp = -np.log(cnt[ok] / (cnt.sum() * np.diff(edges)[ok]))
        ax.plot(ctr[ok], nlp, "o", ms=3, color=col, mec="white", mew=0.4, zorder=3)
        ax.plot(r["wq"], r["corr"], "-", color=col, lw=2.2, zorder=2)
        ax.plot(r["w"], r["orig"], "--", color=col, lw=1.3, alpha=0.8, zorder=1)
        ins = np.abs(r["wq"]) <= r["corr_s"]["well"]          # top: corrected-theory barrier between the wells
        ymin, ymax = min(ymin, nlp.min(), r["corr"].min()), max(ymax, nlp.max(), r["corr"][ins].max())
        ax.plot([], [], "s", color=col, label=rf"$d={r['d']}$, $P={r['P']}$")
    ax.plot([], [], "o", color="0.3", label="HMC")
    ax.plot([], [], "-", color="0.3", lw=2.2, label="corrected theory")
    ax.plot([], [], "--", color="0.3", lw=1.3, label="original theory")
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
    ax.set_xlabel(r"$w_0$", fontsize=13)
    ax.set_ylabel(r"$-\log p(w_0)$  (log-normalized)", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("invariant α/β ray: read-in action", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / f"{args.out}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
