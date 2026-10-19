#!/usr/bin/env python3
"""Learnability / well curves along the pure-cubic FO ρ-ray.

Global argmin_π F jumps at the first-order transition — that discontinuity is
physical. This plotter:

1. Draws each warm-started F(π) branch as its own continuous curve (from
   ``branch_local_mins.csv`` when present).
2. Draws the global-equilibrium series with line breaks wherever |Δπ| exceeds
   ``--jump-pi`` (and drops walk-edge points), so a FO jump is a gap, not a
   steep fake connector.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def stitch_tracks(rho, pi, sep: float = 0.035, max_gap: float = 0.02):
    """Continuous tracks with at most one point per ρ level.

    At each distinct ρ (descending), bipartite-match local mins to existing
    tracks by (ρ-gap, |Δπ|). A track may not jump more than ``max_gap`` in ρ
    (prevents a dead high-π branch from reattaching across the FO). Unmatched
    points spawn new tracks.
    """
    rho = np.asarray(rho, dtype=float)
    pi = np.asarray(pi, dtype=float)
    levels = sorted({float(r) for r in rho if np.isfinite(r)}, reverse=True)
    tracks: list[list[int]] = []
    for level in levels:
        idxs = [i for i, r in enumerate(rho) if np.isfinite(r) and abs(float(r) - level) < 1e-12
                and np.isfinite(pi[i])]
        if not idxs:
            continue
        pairs = []
        for j, t in enumerate(tracks):
            prev = t[-1]
            prev_pi = float(pi[prev])
            gap = abs(level - float(rho[prev]))
            if gap > max_gap:
                continue
            for i in idxs:
                dpi = abs(float(pi[i]) - prev_pi)
                if dpi <= sep:
                    pairs.append((gap, dpi, j, i))
        pairs.sort()
        used_t, used_i = set(), set()
        for gap, dpi, j, i in pairs:
            if j in used_t or i in used_i:
                continue
            tracks[j].append(i)
            used_t.add(j)
            used_i.add(i)
        for i in idxs:
            if i not in used_i:
                tracks.append([i])
    tracks.sort(key=len, reverse=True)
    return tracks


def broken_line(ax, rho, y, pi, thr, markers=True, **kwargs):
    """Plot y(ρ) with line breaks at |Δπ| jumps; markers on all finite points."""
    rho = np.asarray(rho, dtype=float)
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    order = np.argsort(-rho)
    rho, y, pi = rho[order], y[order], pi[order]
    seg_r, seg_y = [], []
    prev_pi = None
    first = True
    for i in range(len(rho)):
        if not np.isfinite(y[i]) or not np.isfinite(pi[i]):
            if seg_r:
                ax.plot(seg_r, seg_y, **(kwargs if first else {k: v for k, v in kwargs.items() if k != "label"}))
                first = False
                seg_r, seg_y = [], []
            prev_pi = None
            continue
        if prev_pi is not None and abs(pi[i] - prev_pi) > thr:
            ax.plot(seg_r, seg_y, **(kwargs if first else {k: v for k, v in kwargs.items() if k != "label"}))
            first = False
            seg_r, seg_y = [], []
        seg_r.append(rho[i])
        seg_y.append(y[i])
        prev_pi = pi[i]
    if seg_r:
        ax.plot(seg_r, seg_y, **(kwargs if first else {k: v for k, v in kwargs.items() if k != "label"}))
    if markers:
        mk = {k: v for k, v in kwargs.items() if k != "label"}
        ax.plot(rho[np.isfinite(y)], y[np.isfinite(y)], "o", ms=4.5, **mk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin/summary_equilibrium.csv",
    )
    ap.add_argument(
        "--branches",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin/branch_local_mins.csv",
    )
    ap.add_argument(
        "--outdir",
        default="action_h0_activation_plots/PureCubicChiN/fo_rho_ray_fmin",
    )
    ap.add_argument("--jump-pi", type=float, default=0.03,
                    help="|Δπ| threshold for breaking global-eq line segments")
    ap.add_argument("--track-sep", type=float, default=0.05,
                    help="|Δπ| threshold when stitching local-min points into tracks")
    ap.add_argument("--fo-lo", type=float, default=0.09)
    ap.add_argument("--fo-hi", type=float, default=0.095)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = _load_csv(Path(args.summary))
    summary = np.atleast_1d(summary)
    names = set(summary.dtype.names or ())

    rho = np.asarray(summary["rho"], dtype=float)
    L3 = np.asarray(summary["L3"], dtype=float)
    pi = np.asarray(summary["pi_out"], dtype=float)
    Vin = np.asarray(summary["V_in"], dtype=float)
    Vout = np.asarray(summary["V_out"], dtype=float)
    dV = np.asarray(summary["dV_raw"], dtype=float)
    edge = np.asarray(summary["at_edge"], dtype=float) if "at_edge" in names else np.zeros_like(rho)

    # Drop walk-edge points from the connected global series (keep as x markers).
    keep = edge < 0.5
    rho_k, L3_k, pi_k = rho[keep], L3[keep], pi[keep]
    Vin_k, Vout_k, dV_k = Vin[keep], Vout[keep], dV[keep]
    edge_rho, edge_L3 = rho[edge >= 0.5], L3[edge >= 0.5]

    branches = None
    bpath = Path(args.branches)
    if bpath.is_file() and bpath.stat().st_size > 0:
        raw = _load_csv(bpath)
        if raw is not None and np.size(raw) > 0:
            branches = np.atleast_1d(raw)

    # Stitch all local-min points into continuous π-tracks (branch ids may split
    # a single basin; proximity in π restores continuity for plotting).
    tracks = []
    br_rho = br_L3 = br_pi = br_Vin = br_Vout = br_glob = None
    if branches is not None:
        br_rho = np.asarray(branches["rho"], dtype=float)
        br_L3 = np.asarray(branches["L3"], dtype=float)
        br_pi = np.asarray(branches["pi_out"], dtype=float)
        br_Vin = np.asarray(branches["V_in"], dtype=float)
        br_Vout = np.asarray(branches["V_out"], dtype=float)
        br_glob = np.asarray(branches["is_global"], dtype=float) > 0.5
        tracks = [t for t in stitch_tracks(br_rho, br_pi, sep=args.track_sep) if len(t) >= 3]

    def plot_tracks_y(ax, y, *, lw=1.8, alpha=0.9, ylabel_prefix="", markers=True):
        """Plot π-stitched tracks as continuous curves (no FO-style breaks)."""
        cmap = plt.get_cmap("tab10")
        for i, t in enumerate(tracks):
            idx = np.asarray(t, dtype=int)
            order = np.argsort(-br_rho[idx])
            rr, yy = br_rho[idx][order], np.asarray(y)[idx][order]
            lab = f"track {i + 1}" if ylabel_prefix == "" else f"track {i + 1} {ylabel_prefix}"
            ax.plot(rr, yy, "-", color=cmap(i % 10), lw=lw, alpha=alpha, label=lab)
            if markers:
                ax.plot(rr, yy, "o", color=cmap(i % 10), ms=4.0, alpha=alpha)

    # ---- L3 alone ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvspan(args.fo_lo, args.fo_hi, color="salmon", alpha=0.25, lw=0, zorder=0)
    if tracks:
        plot_tracks_y(ax, br_L3, lw=2.0)
        broken_line(
            ax,
            br_rho[br_glob],
            br_L3[br_glob],
            br_pi[br_glob],
            args.jump_pi,
            color="k",
            lw=2.6,
            label=r"global $\min_\pi F$",
            zorder=4,
        )
    else:
        broken_line(ax, rho_k, L3_k, pi_k, args.jump_pi,
                    color="C0", lw=2.0, label=r"global $\min_\pi F$ (edge dropped)")
    if edge_rho.size:
        ax.plot(edge_rho, edge_L3, "x", color="0.45", ms=7, label="walk edge", zorder=5)
    ax.set_xlabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$ (decreasing $\rightarrow$ more learnable)")
    ax.set_ylabel(r"learnability $L_3$")
    ax.set_title(r"Cubic learnability along FO $\rho$-ray (continuous $\pi$-tracks)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    p1 = outdir / "L3_vs_rho.png"
    fig.savefig(p1, dpi=170, bbox_inches="tight")
    print(f"saved {p1}")
    plt.close(fig)

    # ---- 3-panel ----------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.2), sharex=True)
    for ax in axes:
        ax.axvspan(args.fo_lo, args.fo_hi, color="salmon", alpha=0.25, lw=0, zorder=0)
        ax.grid(True, alpha=0.3)

    ax = axes[0]
    if tracks:
        plot_tracks_y(ax, br_L3, lw=1.8)
        broken_line(
            ax,
            br_rho[br_glob],
            br_L3[br_glob],
            br_pi[br_glob],
            args.jump_pi,
            color="k",
            lw=2.6,
            label=r"global $\min_\pi F$",
            zorder=4,
        )
    else:
        broken_line(ax, rho_k, L3_k, pi_k, args.jump_pi, color="C0", lw=2.0,
                    label=r"global $\min_\pi F$")
    if edge_rho.size:
        ax.plot(edge_rho, edge_L3, "x", color="0.45", ms=7, label="walk edge")
    ax.set_ylabel(r"$L_3$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    if tracks:
        cmap = plt.get_cmap("tab10")
        for i, t in enumerate(tracks):
            idx = np.asarray(t, dtype=int)
            order = np.argsort(-br_rho[idx])
            rr = br_rho[idx][order]
            ax.plot(rr, br_Vout[idx][order], "-", color=cmap(i % 10), lw=1.8, alpha=0.9,
                    label=rf"track {i + 1} $V_{{\mathrm{{out}}}}$")
            ax.plot(rr, br_Vout[idx][order], "o", color=cmap(i % 10), ms=4.0, alpha=0.9)
            ax.plot(rr, br_Vin[idx][order], "--", color=cmap(i % 10), lw=1.0, alpha=0.45)
    else:
        broken_line(ax, rho_k, Vin_k, pi_k, args.jump_pi, color="C0", lw=2.0, label=r"$V_{\mathrm{in}}$ global")
        broken_line(ax, rho_k, Vout_k, pi_k, args.jump_pi, color="C3", lw=2.0, label=r"$V_{\mathrm{out}}$ global")
    ax.set_ylabel(r"well depth $V$")
    ax.set_ylim(-8, 8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)

    ax = axes[2]
    ax2 = ax.twinx()
    if tracks:
        cmap = plt.get_cmap("tab10")
        br_dV = br_Vout - br_Vin
        for i, t in enumerate(tracks):
            idx = np.asarray(t, dtype=int)
            order = np.argsort(-br_rho[idx])
            rr = br_rho[idx][order]
            ax.plot(rr, br_dV[idx][order], "-", color=cmap(i % 10), lw=1.8, alpha=0.9,
                    label=rf"track {i + 1} $dV$")
            ax2.plot(rr, br_pi[idx][order], "--", color=cmap(i % 10), lw=1.4, alpha=0.85,
                     label=rf"track {i + 1} $\pi$")
        broken_line(ax2, br_rho[br_glob], br_pi[br_glob], br_pi[br_glob], args.jump_pi,
                    color="k", lw=2.2, label=r"global $\pi_{\mathrm{out}}$", zorder=4)
    else:
        broken_line(ax, rho_k, dV_k, pi_k, args.jump_pi, color="C2", lw=2.0, label=r"$dV_{\mathrm{raw}}$")
        broken_line(ax2, rho_k, pi_k, pi_k, args.jump_pi, color="k", lw=1.6, label=r"$\pi_{\mathrm{out}}$")
    ax.set_ylabel(r"$V_{\mathrm{out}}-V_{\mathrm{in}}$")
    ax.set_ylim(-8, 8)
    ax2.set_ylabel(r"$\pi_{\mathrm{out}}$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")

    axes[-1].set_xlabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$ (decreasing $\rightarrow$ more learnable)")
    axes[-1].invert_xaxis()
    fig.suptitle(
        r"Pure-cubic FO $\rho$-ray at $\min_\pi F(\pi)$"
        "\n"
        r"$d=16,\ N=\chi=200,\ P=10^3,\ \mathrm{sa0}=0.03$"
        "\n"
        r"colored = continuous $\pi$-tracks; black gaps = FO jump of global $\min F$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = outdir / "learnability_and_wells_vs_rho.png"
    fig.savefig(p2, dpi=170, bbox_inches="tight")
    print(f"saved {p2}")
    plt.close(fig)


if __name__ == "__main__":
    main()
