#!/usr/bin/env python3
"""Plot V(w) on the full line (both ± wells) for eigenframe + mixture saddle weights."""
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
    files = sorted(base.glob("action_mix_P*_d*.csv"),
                   key=lambda p: float(p.name.split("_")[2][1:]))
    if not files:
        raise SystemExit(f"no action_mix_P*.csv in {base}")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(files)))
    for f, c in zip(files, cmap):
        P = float(f.name.split("_")[2][1:])
        rows = np.genfromtxt(f, delimiter=",", names=True)
        m = np.abs(rows["w"]) <= args.wmax
        ax.plot(rows["w"][m], rows["V"][m], color=c, lw=1.6, label=fr"$P={P:g}$")
    ax.axvline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$V(w_0)$")
    ax.set_title(r"Eigenframe + mixture saddle weights: $\tilde Q=\Lambda$, $\tilde y=(0,\varepsilon)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("saved", args.out)

if __name__ == "__main__":
    main()
