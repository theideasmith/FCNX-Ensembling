#!/usr/bin/env python3
"""Plot V(w) for the diagonal-Λ eigenframe action (ỹ=(0,ε))."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wmax", type=float, default=6.0)
    args = ap.parse_args()
    base = Path(args.dir)
    files = sorted(base.glob("action_saddle_P*_d*.csv"),
                   key=lambda p: float(p.name.split("_")[2][1:]))
    if not files:
        raise SystemExit(f"no action_saddle_P*.csv in {base}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(files)))
    for f, c in zip(files, cmap):
        P = float(f.name.split("_")[2][1:])
        rows = np.genfromtxt(f, delimiter=",", names=True)
        m = rows["w"] <= args.wmax
        ax.plot(rows["w"][m], rows["V"][m], color=c, lw=1.6, label=fr"$P={P:g}$")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$V(w_0)$")
    ax.set_title(r"Eigenframe saddle action: $\tilde Q=\Lambda$, $\tilde y=(0,\varepsilon)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)

if __name__ == "__main__":
    main()
