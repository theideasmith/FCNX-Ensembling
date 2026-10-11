#!/usr/bin/env python3
"""Plot the explicit cavity action V(w) = -log p from --dump-action (raw, unshifted)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, help="CSV with w,V,...")
    ap.add_argument("--wells", default=None, help="optional wells CSV")
    ap.add_argument("--meta", default=None, help="optional meta txt")
    ap.add_argument("--out", default=None, help="output PNG")
    args = ap.parse_args()

    act = np.genfromtxt(args.action, delimiter=",", names=True)
    w = act["w"]
    V = act["V"]  # raw action

    wells = None
    if args.wells and Path(args.wells).is_file():
        wells = np.genfromtxt(args.wells, delimiter=",", names=True, dtype=None, encoding=None)

    meta_lines = []
    if args.meta and Path(args.meta).is_file():
        meta_lines = Path(args.meta).read_text().strip().splitlines()

    out = Path(args.out) if args.out else Path(args.action).with_name(
        Path(args.action).name.replace("tmp_pure_cubic_action_", "pure_cubic_action_").replace(".csv", ".png")
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(w, V, color="k", lw=1.8, label=r"$V(w)=-\log p$ (raw)")

    if wells is not None:
        for row in np.atleast_1d(wells):
            mu = float(row["mu"])
            Vw = float(row["V"])
            role = str(row["role"])
            mass = float(row["mass"])
            col = "C3" if role == "outer" else "C0"
            for x in ([mu, -mu] if abs(mu) > 1e-12 else [mu]):
                ax.axvline(x, color=col, ls="--", lw=1.0, alpha=0.7)
                ax.scatter([x], [Vw], color=col, zorder=5, s=40)
            ax.annotate(
                f"{role}: π={mass:.3f}\nV={Vw:.2f}",
                xy=(mu, Vw),
                xytext=(14, 90 if role == "outer" else -90),
                textcoords="offset points",
                fontsize=8,
                color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            )

    ax.set_xlabel(r"read-in weight $w$ (teacher axis)")
    ax.set_ylabel(r"action $V(w)$  (= $-\log p$, raw)")
    title = r"Pure-cubic matrix Laplace action $V(w)$ RAW, $\chi=N$"
    if meta_lines:
        title += "\n" + meta_lines[0]
        if len(meta_lines) > 2:
            title += "\n" + meta_lines[2]
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_xlim(float(w[0]), float(w[-1]))
    pad = 0.04 * (float(V.max()) - float(V.min()) + 1.0)
    ax.set_ylim(float(V.min()) - pad, float(V.max()) + pad)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"raw V range [{V.min():.4g}, {V.max():.4g}]")


if __name__ == "__main__":
    main()
