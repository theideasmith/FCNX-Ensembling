#!/usr/bin/env python3
"""
Drive term of the read-in action across a κ sweep: theory β (D·c(w0))² against the exact drive
⟨½ σ_A² r² / (1 + G)⟩ measured from collapsed-HMC samples (diag_cavity_action.py), all groups on
one axis coloured by κ (viridis; theory solid, HMC dashed).

The right panel fits β (D·c(w0))² to the measured drive on w0 > 0 (theory features, prior perp norm)
and compares the fitted (D1, D3) with the theory's self-consistent D. The printed table also gives
the share of the drive carried by the He1+He3 part of the cavity residual.

    python plot_cavity_drive.py --groups grok_k0p07 grok_k0p1 grok_k0p15 grok_k0p2 grok_k0p25 grok_k0p3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diag_cavity_action import cavity_analysis, group_args
from readin_action_theory import Kernel, features, reduced_params, solve_loop


def theory_D(a):
    T = 1 + 2 * a["s0"] * (a["d"] - 1) / a["d"]
    _, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
    D, _ = solve_loop(Chat, lam, a["eps"], Kernel(T, beta))
    return D, T, beta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--snapshots", type=int, default=40)
    ap.add_argument("--wmax", type=float, default=1.4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for g in args.groups:
        cav = cavity_analysis(g, args.snapshots, wmax=args.wmax, nw=301)
        if cav is None:
            print(f"  {g}: no samples yet")
            continue
        a = group_args(g)
        D, T, beta = theory_D(a)
        w = cav["w"]
        c1, c3 = features(np.abs(w), T)
        pos = w > 0.02
        # fit sqrt(drive/β) = D1 c1 + D3 c3 on w0 > 0 (D·c > 0 there)
        Dfit, *_ = np.linalg.lstsq(np.stack([c1[pos], c3[pos]], 1), np.sqrt(cav["drive_exact"][pos] / beta), rcond=None)
        rows.append(dict(k=a["T"] / 2, w=w, a=a, D=D, Dfit=Dfit, n=int(cav["n_snap"]),
                         th=beta * (D[0] * c1 + D[1] * c3) ** 2,
                         fit=beta * (Dfit[0] * c1 + Dfit[1] * c3) ** 2,
                         ex=cav["drive_exact"], he13=cav["drive_He13"], rest=cav["drive_rest"]))
    rows.sort(key=lambda r: r["k"])

    print(f"\n{'κ':>5s} {'snaps':>5s} {'D theory':>18s} {'D fit to HMC':>18s} {'He13/exact @|w0|=0.5,1':>24s} {'rest/exact':>10s}")
    for r in rows:
        i5, i10 = (np.argmin(abs(r["w"] - x)) for x in (0.5, 1.0))
        print(f"{r['k']:>5g} {r['n']:>5d} {np.array2string(r['D'], precision=3):>18s} "
              f"{np.array2string(r['Dfit'], precision=3):>18s} "
              f"{r['he13'][i5] / r['ex'][i5]:>11.3f} {r['he13'][i10] / r['ex'][i10]:>11.3f} "
              f"{r['rest'][i10] / r['ex'][i10]:>10.3f}")

    ks = np.array([r["k"] for r in rows])
    norm = plt.Normalize(ks.min(), ks.max() if ks.max() > ks.min() else ks.min() + 1)
    col = {r["k"]: plt.cm.viridis(0.9 * norm(r["k"])) for r in rows}
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    for r in rows:
        ax.plot(r["w"], r["th"], color=col[r["k"]], lw=2)
        ax.plot(r["w"], r["ex"], color=col[r["k"]], lw=2, ls=(0, (4, 2)))
    ax.plot([], [], color="0.3", lw=2, label=r"theory $\beta\,(D\cdot c(w_0))^2$")
    ax.plot([], [], color="0.3", lw=2, ls=(0, (4, 2)), label=r"HMC exact drive $\langle \frac{1}{2}\sigma_A^2 r^2/(1+G)\rangle$")
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_ylim(0, max(max(r["th"].max(), r["ex"].max()) for r in rows) * 1.05)
    ax.set_xlabel(r"$w_0$", fontsize=13)
    ax.set_ylabel("drive (nats)", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper center")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(norm.vmin, norm.vmin + (norm.vmax - norm.vmin) / 0.9))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label(r"$\kappa$", fontsize=13)
    cb.set_ticks(ks)

    for j, (lab, mk) in enumerate(((r"$D_1$", "o"), (r"$|D_3|$", "s"))):
        th = [abs(r["D"][j]) for r in rows]
        fit = [abs(r["Dfit"][j]) for r in rows]
        axd.plot(ks, th, "-", color="0.35", lw=2, marker=mk, ms=5, mfc="white", label=f"{lab} theory")
        axd.scatter(ks, fit, marker=mk, s=45, c=[col[k] for k in ks], edgecolors="0.2", linewidths=0.6, zorder=3,
                    label=f"{lab} fit to HMC drive")
    axd.set_xlabel(r"$\kappa$", fontsize=13)
    axd.set_yscale("log")
    axd.set_title("effective drive vector D", fontsize=13)
    axd.legend(frameon=False, fontsize=9)
    a0 = rows[0]["a"]
    fig.suptitle(rf"$d={a0['d']}$, $P={a0['P']}$, $N={a0['N']}$, $\sigma_w^2={a0['s0']:g}$, "
                 rf"$\epsilon={a0['eps']:.3g}$, $\sigma_a^2={a0['sa0'] * a0['N']:.3g}/N$", fontsize=14)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / f"cavity_drive_{args.groups[0]}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
