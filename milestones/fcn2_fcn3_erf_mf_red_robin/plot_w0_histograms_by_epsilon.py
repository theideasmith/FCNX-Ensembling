#!/usr/bin/env python3
"""Plot W0 histograms for each epsilon: FCN2 vs FCN3 side-by-side.

Usage example:
    python plot_w0_histograms_by_epsilon.py --results-root results_epsilon_sweep --out-dir w0_histograms --seed 0

The script expects the structure:
  <results-root>/fcn2/<run_dir>/_eps_<eps>/seed{seed}/model_final.pt
  <results-root>/fcn3/<run_dir>/_eps_<eps>/seed{seed}/model_final.pt

It will search for "_eps_<value>" in run directory names to match epsilons.
"""

from pathlib import Path
import argparse
import re
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def find_runs_by_eps(root: Path, family: str):
    by_eps = {}
    base = root / family
    if not base.exists():
        return by_eps
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        m = re.search(r"_eps_([0-9]+\.?[0-9]*)", p.name)
        if m:
            eps = m.group(1)
            by_eps[eps] = p
    return by_eps


def load_w0_from_run(run_dir: Path, seed: int):
    seed_dir = run_dir / f"seed{seed}"
    if not seed_dir.exists():
        # fallback to first seed dir
        seed_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
        if not seed_dirs:
            return None
        seed_dir = seed_dirs[0]

    for fname in ("model_final.pt", "model.pt", "model.pt.tar"):
        ck = seed_dir / fname
        if not ck.exists():
            continue
        try:
            st = torch.load(str(ck), map_location="cpu")
        except Exception:
            continue

        # Some checkpoints may be nested under 'state_dict'
        if isinstance(st, dict) and "state_dict" in st and isinstance(st["state_dict"], dict):
            st = st["state_dict"]

        if isinstance(st, dict):
            # Direct key
            if "W0" in st:
                w0 = st["W0"].cpu().numpy()
                return w0[:,:,0]
            # Try keys that end with .W0 or /W0
            for k in st.keys():
                if k.endswith(".W0") or k.endswith("W0"):
                    try:
                        return st[k].cpu().numpy()
                    except Exception:
                        pass
        # not found or unknown format
    return None


def plot_pair(w0_2: np.ndarray, w0_3: np.ndarray, eps_label: str, out_path: Path, bins: int = 100):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # compute common bin edges
    combined = np.concatenate([w0_2.ravel(), w0_3.ravel()])
    vmin, vmax = np.percentile(combined, [0.5, 99.5])
    edges = np.linspace(float(vmin), float(vmax), bins)

    axes[0].hist(w0_2.ravel(), bins=edges, color="#1f77b4", alpha=0.8)
    axes[0].set_title("FCN2 W0")
    axes[0].set_xlabel("W0 value")

    axes[1].hist(w0_3.ravel(), bins=edges, color="#d62728", alpha=0.8)
    axes[1].set_title("FCN3 W0")
    axes[1].set_xlabel("W0 value")

    fig.suptitle(f"W0 histograms | eps={eps_label}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot W0 histograms for FCN2 vs FCN3 by epsilon")
    p.add_argument("--results-root", type=str, default="results_epsilon_sweep", help="Root results directory")
    p.add_argument("--out-dir", type=str, default="w0_histograms", help="Output directory for PNGs")
    p.add_argument("--seed", type=int, default=0, help="Seed to use (falls back to first available seed)")
    p.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    args = p.parse_args()

    root = Path(__file__).resolve().parent / args.results_root
    out_root = Path(__file__).resolve().parent / args.out_dir

    fcn2 = find_runs_by_eps(root, "fcn2")
    fcn3 = find_runs_by_eps(root, "fcn3")

    common_eps = sorted(set(fcn2.keys()).intersection(set(fcn3.keys())), key=lambda s: float(s))
    if not common_eps:
        print("No common eps found between fcn2 and fcn3 under", root)
        return

    print(f"Found {len(common_eps)} matching eps: {common_eps}")

    for eps in common_eps:
        run2 = fcn2[eps]
        run3 = fcn3[eps]
        w0_2 = load_w0_from_run(run2, args.seed)
        w0_3 = load_w0_from_run(run3, args.seed)
        if w0_2 is None or w0_3 is None:
            print(f"Skipping eps={eps}: missing W0 for FCN2 or FCN3 (paths: {run2}, {run3})")
            continue

        out_path = out_root / f"w0_hist_eps_{eps}.png"
        plot_pair(w0_2, w0_3, eps, out_path, bins=args.bins)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
