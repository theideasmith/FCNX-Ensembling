#!/usr/bin/env python3
"""Plot the explicit cavity action V(w) = -log p (+ const) from --dump-action."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, help="CSV with w,V,V_shifted,prior,data")
    ap.add_argument("--wells", default=None, help="optional wells CSV")
    ap.add_argument("--meta", default=None, help="optional meta txt")
    ap.add_argument("--out", default=None, help="output PNG")
    args = ap.parse_args()

    act = np.genfromtxt(args.action, delimiter=",", names=True)
    w = act["w"]
    V = act["V_shifted"]  # V - min V; same shape as -log p up to a constant

    wells = None
    if args.wells and Path(args.wells).is_file():
        wells = np.genfromtxt(args.wells, delimiter=",", names=True, dtype=None, encoding=None)

    meta_lines = []
    if args.meta and Path(args.meta).is_file():
        meta_lines = Path(args.meta).read_text().strip().splitlines()

    out = Path(args.out) if args.out else Path(args.action).with_name(
        Path(args.action).name.replace("tmp_pure_cubic_action_", "pure_cubic_action_").replace(".csv", ".png")
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(w, V, color="k", lw=1.8, label=r"$V(w)-\min V$")

    if wells is not None:
        V0 = float(np.min(act["V"]))
        for row in np.atleast_1d(wells):
            mu = float(row["mu"])
            Vw = float(row["V"]) - V0
            role = str(row["role"])
            mass = float(row["mass"])
            col = "C3" if role == "outer" else "C0"
            xs = [mu, -mu] if abs(mu) > 1e-12 else [mu]
            for x in xs:
                ax.axvline(x, color=col, ls="--", lw=1.0, alpha=0.75)
                ax.scatter([x], [Vw], color=col, zorder=5, s=40)
            ax.annotate(
                f"{role}: π={mass:.3f}\nV={float(row['V']):.1f}",
                xy=(mu, Vw),
                xytext=(10, 22 if role == "outer" else -28),
                textcoords="offset points",
                fontsize=8,
                color=col,
            )

    ax.set_xlabel(r"read-in weight $w$ (teacher axis)")
    ax.set_ylabel(r"action $V(w)-\min V$   (= $-\log p$ + const)")
    title = r"Pure-cubic matrix Laplace action $V(w)$, $\chi=N$"
    if meta_lines:
        title += "\n" + meta_lines[0]
        if len(meta_lines) > 2:
            title += "\n" + meta_lines[2]
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_xlim(w[0], w[-1])
    if wells is not None:
        V0 = float(np.min(act["V"]))
        Vm = [float(r["V"]) - V0 for r in np.atleast_1d(wells)]
        # show barrier between wells without letting far walls dominate
        mid = 0.5 * (abs(float(np.atleast_1d(wells)[0]["mu"])) + abs(float(np.atleast_1d(wells)[-1]["mu"])))
        mask = np.abs(w) <= mid + 1.5
        local_max = float(np.max(V[mask])) if np.any(mask) else float(np.percentile(V, 80))
        ax.set_ylim(-2.0, max(local_max * 1.15, max(Vm) + 20.0))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
