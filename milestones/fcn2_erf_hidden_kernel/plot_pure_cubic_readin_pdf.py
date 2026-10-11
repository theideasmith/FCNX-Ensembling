#!/usr/bin/env python3
"""Plot theoretical read-in density from tmp_multiwell_pure_cubic.jl --dump-pdf."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="CSV with columns w,p,p_inner,p_outer")
    ap.add_argument("--meta", default=None, help="optional meta txt from --dump-pdf")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    data = np.genfromtxt(pdf_path, delimiter=",", names=True)
    w, p, p_in, p_out = data["w"], data["p"], data["p_inner"], data["p_outer"]

    trap = getattr(np, "trapezoid", np.trapz)
    mass_out = float(trap(p_out, w))
    mass_in = float(trap(p_in, w))

    meta_lines = []
    if args.meta and Path(args.meta).is_file():
        meta_lines = Path(args.meta).read_text().strip().splitlines()

    out = Path(args.out) if args.out else pdf_path.with_suffix(".png")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)

    for ax in (ax0, ax1):
        ax.fill_between(w, 0.0, p_in, alpha=0.35, color="C0", label=f"inner  mass={mass_in:.3f}")
        ax.fill_between(w, 0.0, p_out, alpha=0.55, color="C3", label=f"outer (±μ)  mass={mass_out:.3f}")
        ax.plot(w, p, color="k", lw=1.2)
        ax.set_xlabel(r"read-in weight $w$ (teacher axis)")
        ax.set_xlim(w[0], w[-1])

    ax0.set_ylabel(r"density $p(w)$")
    ax0.set_title("full scale (inner dominates height)")
    ax0.set_ylim(0.0, 1.05 * float(np.max(p)))
    ax0.legend(frameon=False, loc="upper right", fontsize=9)

    # right panel: scale so outer peaks are visible
    p_out_max = float(np.max(p_out)) if np.max(p_out) > 0 else 1.0
    ax1.set_title("outer-visible scale")
    ax1.set_ylim(0.0, 1.25 * p_out_max)
    ax1.legend(frameon=False, loc="upper right", fontsize=9)

    title = "Pure-cubic matrix Laplace read-in mixture (χ=N)"
    if meta_lines:
        title += "\n" + meta_lines[0]
        if len(meta_lines) > 2:
            title += "  |  " + meta_lines[2]
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"integrated masses: inner={mass_in:.6f}  outer={mass_out:.6f}  total={mass_in+mass_out:.6f}")


if __name__ == "__main__":
    main()
