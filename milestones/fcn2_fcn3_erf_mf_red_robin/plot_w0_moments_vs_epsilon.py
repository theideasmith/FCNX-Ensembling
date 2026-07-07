#!/usr/bin/env python3
"""Plot first three moments of W0 histograms vs epsilon: FCN2 vs FCN3.

Computes mean, variance, and skewness of W0 distributions for each epsilon
and creates a publication-ready plot comparing FCN2 and FCN3.

Usage example:
    python plot_w0_moments_vs_epsilon.py --results-root results_epsilon_sweep --out-dir plots --seed 0
"""

from pathlib import Path
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


def find_runs_by_eps(root: Path, family: str):
    """Find run directories indexed by epsilon value."""
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
    """Load W0 weights from a model checkpoint."""
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
                return w0[:, :, 0]
            # Try keys that end with .W0 or /W0
            for k in st.keys():
                if k.endswith(".W0") or k.endswith("W0"):
                    try:
                        return st[k].cpu().numpy()
                    except Exception:
                        pass
    return None


def compute_moments(w0: np.ndarray):
    """Compute first three moments: mean, variance, skewness."""
    w0_flat = w0.ravel()
    mean = np.mean(w0_flat)
    var = np.var(w0_flat)
    skewness = stats.skew(w0_flat)
    return mean, var, skewness


def plot_moments(eps_values, fcn2_moments, fcn3_moments, out_path: Path):
    """Create publication-ready plot of moments vs epsilon."""
    eps_floats = np.array([float(e) for e in eps_values])
    
    # Extract moments
    fcn2_mean, fcn2_var, fcn2_skew = fcn2_moments
    fcn3_mean, fcn3_var, fcn3_skew = fcn3_moments
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(r"$W_0$ Moments vs Regularization Strength ($\epsilon$)", fontsize=13, fontweight='bold')
    
    # Color scheme
    color_fcn2 = "#1f77b4"  # Blue
    color_fcn3 = "#d62728"  # Red
    
    # Plot 1: Mean
    axes[0].plot(eps_floats, fcn2_mean, 'o-', color=color_fcn2, label='FCN2', linewidth=2, markersize=6)
    axes[0].plot(eps_floats, fcn3_mean, 's-', color=color_fcn3, label='FCN3', linewidth=2, markersize=6)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\epsilon$', fontsize=11)
    axes[0].set_ylabel(r'Mean', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('First Moment', fontsize=11, fontweight='bold')
    
    # Plot 2: Variance
    axes[1].plot(eps_floats, fcn2_var, 'o-', color=color_fcn2, label='FCN2', linewidth=2, markersize=6)
    axes[1].plot(eps_floats, fcn3_var, 's-', color=color_fcn3, label='FCN3', linewidth=2, markersize=6)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\epsilon$', fontsize=11)
    axes[1].set_ylabel(r'Variance', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Second Moment', fontsize=11, fontweight='bold')
    
    # Plot 3: Skewness
    axes[2].plot(eps_floats, fcn2_skew, 'o-', color=color_fcn2, label='FCN2', linewidth=2, markersize=6)
    axes[2].plot(eps_floats, fcn3_skew, 's-', color=color_fcn3, label='FCN3', linewidth=2, markersize=6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel(r'$\epsilon$', fontsize=11)
    axes[2].set_ylabel(r'Skewness', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Third Moment', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot W0 moments for FCN2 vs FCN3 over epsilon")
    p.add_argument("--results-root", type=str, default="results_epsilon_sweep", help="Root results directory")
    p.add_argument("--out-dir", type=str, default="plots", help="Output directory for plots")
    p.add_argument("--seed", type=int, default=0, help="Seed to use (falls back to first available seed)")
    p.add_argument("--out-filename", type=str, default="w0_moments_vs_epsilon.png", help="Output filename")
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

    # Collect moments for all epsilon values
    fcn2_moments = {"mean": [], "var": [], "skew": []}
    fcn3_moments = {"mean": [], "var": [], "skew": []}

    for eps in common_eps:
        run2 = fcn2[eps]
        run3 = fcn3[eps]
        w0_2 = load_w0_from_run(run2, args.seed)
        w0_3 = load_w0_from_run(run3, args.seed)
        
        if w0_2 is None or w0_3 is None:
            print(f"Skipping eps={eps}: missing W0 for FCN2 or FCN3")
            continue

        mean2, var2, skew2 = compute_moments(w0_2)
        mean3, var3, skew3 = compute_moments(w0_3)
        
        fcn2_moments["mean"].append(mean2)
        fcn2_moments["var"].append(var2)
        fcn2_moments["skew"].append(skew2)
        
        fcn3_moments["mean"].append(mean3)
        fcn3_moments["var"].append(var3)
        fcn3_moments["skew"].append(skew3)
        
        print(f"eps={eps}: FCN2 (μ={mean2:.4f}, σ²={var2:.4f}, γ={skew2:.4f}), "
              f"FCN3 (μ={mean3:.4f}, σ²={var3:.4f}, γ={skew3:.4f})")

    # Convert lists to arrays
    fcn2_data = (np.array(fcn2_moments["mean"]), np.array(fcn2_moments["var"]), np.array(fcn2_moments["skew"]))
    fcn3_data = (np.array(fcn3_moments["mean"]), np.array(fcn3_moments["var"]), np.array(fcn3_moments["skew"]))

    # Create plot
    out_path = out_root / args.out_filename
    plot_moments(common_eps, fcn2_data, fcn3_data, out_path)


if __name__ == "__main__":
    main()
