#!/usr/bin/env python3
"""
Fixed-d P-scan (launch_pscan_d150.sh): the read-in theory is identical at every P (λ, Ĉ, β fixed), only
the finite-P leakage changes. If the HMC-vs-theory gap is leakage, HMC → theory as 1/P → 0.

Top: -log p(w0) (normalized; shifted only by log normalization) for every P, coloured by P, against the
single theory curve. Bottom, against 1/P: w0², aligned fraction, outer-well position, L3 (50k fresh test
points), and the training-set He3 absorption (f3_train - f3_pop) / f3_pop; theory as a horizontal line
(absorption: 0). Per-chain means are printed for convergence.

    python plot_pscan.py --Ps 3000 6000 12000 24000
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, hermite_basis, sample_files
from plot_readin_action import load_group, theory_curve
from plot_ray_trend import hmc_stats
from runs import RUNS_ROOT


@torch.no_grad()
def he3_absorption(group, a, n_snap, device):
    """(f3_train - f3_pop)/f3_pop for the posterior-mean predictor, averaged over snapshots."""
    dev = torch.device(device)
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    H = hermite_basis(X[:, 0], (0, 1, 2, 3, 4, 5, 6, 7))
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    out = []
    for f in sample_files(group, n_snap):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        s = W.norm(dim=1)
        a3 = -np.sqrt(8 / (3 * np.pi)) * s ** 3 / (1 + 2 * s * s) ** 1.5
        f3_pop = ((A * a3) * (W[:, 0] / s) ** 3).sum().item()
        f3_tr = torch.linalg.lstsq(H, (Phi @ A)[:, None]).solution[3, 0].item()
        out.append((f3_tr - f3_pop) / f3_pop)
    return float(np.mean(out))


def outer_well(w0, lo=0.5):
    """Mode of |w0| beyond lo (histogram peak) — the HMC aligned-well position."""
    x = np.abs(w0[np.abs(w0) > lo])
    if len(x) < 50:
        return float("nan")
    h, e = np.histogram(x, 60)
    return float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))


def per_chain_means(group, keys=("w0_sq", "frac_aligned")):
    import csv
    res = {}
    for cp in sorted(RUNS_ROOT.glob("*/config.json")):
        a = json.loads(cp.read_text())["args"]
        if a.get("group") != group or not (cp.parent / "obs.csv").exists():
            continue
        rows = [r for r in csv.DictReader(open(cp.parent / "obs.csv")) if r["burn"] == "0"]
        if rows:
            res[a["init"]] = {k: np.mean([float(r[k]) for r in rows]) for k in keys}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[3000, 6000, 12000, 24000])
    ap.add_argument("--prefix", default="pscan_d150_P")
    ap.add_argument("--snapshots", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for P in args.Ps:
        g = f"{args.prefix}{P}"
        loaded = load_group(g, live=True)
        if loaded is None:
            print(f"  {g}: no samples yet")
            continue
        W, a, _ = loaded
        h = hmc_stats(g, a, args.snapshots, 50000, args.device)
        rows.append(dict(P=P, a=a, w0=W[:, 0], h=h, well=outer_well(W[:, 0]),
                         absorb=he3_absorption(g, a, min(args.snapshots, 20), args.device),
                         chains=per_chain_means(g)))
    a0 = rows[0]["a"]
    T = 1 + 2 * a0["s0"] * (a0["d"] - 1) / a0["d"]
    w = np.linspace(-1.9, 1.9, 3801)
    nlp_th, th = theory_curve(a0, T, w)
    th_well = float(np.abs(w[(w > 0.5)][np.argmin(nlp_th[w > 0.5])]))
    thr = 2.5 * np.sqrt(a0["s0"] / a0["d"])

    print(f"\ntheory (same at every P): L3={th['L3']:.4f} w0²={th['w0_sq']:.5f} frac={th['frac_aligned']:.4f} outer well={th_well:.3f}")
    print(f"{'P':>6s} {'κ':>6s} | {'w0²':>8s} {'frac':>7s} {'well':>6s} {'L3':>7s} {'L1':>7s} | {'He3 absorb':>10s} | per-chain w0² / frac")
    for r in rows:
        h = r["h"]
        fr = float(np.mean(np.abs(r["w0"]) > thr))
        print(f"{r['P']:>6d} {r['a']['T'] / 2:>6.3f} | {h['w0_sq']:>8.5f} {fr:>7.4f} {r['well']:>6.3f} {h['L3']:>7.4f} {h['L1']:>7.4f} | "
              f"{r['absorb']:>+10.3f} | " + "  ".join(f"{k}: {v['w0_sq']:.5f}/{v['frac_aligned']:.4f}" for k, v in r["chains"].items()))

    norm = plt.Normalize(np.log2(min(args.Ps)), np.log2(max(args.Ps)))
    col = {r["P"]: plt.cm.viridis(0.9 * norm(np.log2(r["P"]))) for r in rows}
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.35, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(w, nlp_th, color="#d1495b", lw=2.5, label="theory (identical at every P)", zorder=3)
    edges = np.linspace(w[0], w[-1], 151)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for r in rows:
        cnt, _ = np.histogram(r["w0"], edges)
        ok = cnt > 0
        nlp = np.full(len(ctr), np.nan)
        nlp[ok] = -np.log(cnt[ok] / (cnt.sum() * np.diff(edges)[ok]))
        ax.plot(ctr, nlp, "-", color=col[r["P"]], lw=1.6, label=rf"HMC $P={r['P']}$, $\kappa={r['a']['T'] / 2:g}$")
    ax.set_ylim(np.nanmin(nlp_th) - 0.5, 14)
    ax.set_xlim(w[0], w[-1])
    ax.set_xlabel(r"$w_0$", fontsize=13)
    ax.set_ylabel(r"$-\log p(w_0)$  shifted by log normalization", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=3)
    invP = np.array([1 / r["P"] for r in rows])
    panels = [("w0²", [r["h"]["w0_sq"] for r in rows], th["w0_sq"]),
              ("aligned fraction", [float(np.mean(np.abs(r["w0"]) > thr)) for r in rows], th["frac_aligned"]),
              ("outer-well position", [r["well"] for r in rows], th_well),
              (r"$L_3$ (50k test)", [r["h"]["L3"] for r in rows], th["L3"]),
              ("He3 absorption on train", [r["absorb"] for r in rows], 0.0)]
    for i, (lab, ys, ref) in enumerate(panels):
        a_ = fig.add_subplot(gs[1, i])
        a_.axhline(ref, color="#d1495b", lw=2, label="theory")
        a_.scatter(invP, ys, c=[col[r["P"]] for r in rows], s=50, edgecolors="0.2", zorder=3)
        a_.plot(invP, ys, color="0.6", lw=1, zorder=2)
        a_.set_xlim(0, invP.max() * 1.1)
        a_.set_xlabel(r"$1/P$", fontsize=12)
        a_.set_title(lab, fontsize=12)
        a_.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(rf"fixed-$d$ P-scan at fixed theory point: $d={a0['d']}$, $N={a0['N']}$, $\lambda={th['lam']:.4g}$, "
                 rf"$\hat C={th['Chat']:.1f}$, $\epsilon={a0['eps']:.3g}$", fontsize=14)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / "pscan_d150.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
