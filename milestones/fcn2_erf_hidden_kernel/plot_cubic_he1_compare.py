#!/usr/bin/env python3
"""Plot L3(P) for multiwell matrix Laplace: y=(0,ε) vs y=(1,ε)."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eq(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he1-0", required=True)
    ap.add_argument("--he1-1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    a = load_eq(args.he1_0)
    b = load_eq(args.he1_1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(a["P"], a["L3"], "o-", color="#1f77b4", label=r"$y=(0,\varepsilon)$", lw=1.8)
    ax.plot(b["P"], b["L3"], "s-", color="#d62728", label=r"$y=(1,\varepsilon)$", lw=1.8)
    ax.axhline(args.thresh, color="0.4", ls="--", lw=1, label=fr"$L_3={args.thresh:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$P$")
    ax.set_ylabel(r"cubic learnability $L_3=f_3/\varepsilon$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(r"Multiwell matrix Laplace, $\chi=N$: cubic mode vs $P$")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)

    def onset(rows, thr):
        for P, L in zip(rows["P"], rows["L3"]):
            if L >= thr:
                return float(P), float(L)
        return None, None

    p0, l0 = onset(a, args.thresh)
    p1, l1 = onset(b, args.thresh)
    print(f"L3>={args.thresh}: y=(0,eps) at P={p0}, y=(1,eps) at P={p1}")

if __name__ == "__main__":
    main()
