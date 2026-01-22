#!/usr/bin/env python3
"""Compute histograms for h1_preactivation using a streaming approach for large sample counts.

This script samples 10 million (or user-specified) x points and processes them in batches
to compute histograms of the preactivation values without loading all data into memory at once.

- Scans for model.pt files with the pattern d<d>_P<P>_N<N>_chi<chi>[/...]/seed<seed>/model.pt
- Streams batches of random input data through the network
- Accumulates histogram counts across all batches
- Saves histogram data (counts, bin edges) to JSON
- Optionally plots the histogram

Usage:
    python compute_preactivation_histogram_streaming.py --base-dir . --dims 150 --n-samples 10000000 --batch-size 100000
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories that contain model.pt and match the naming convention."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)"

    model_files = list(base.glob(f"**/*{suffix}*/model.pt")) if suffix else list(base.glob("**/model.pt"))
    for model_file in model_files:
        seed_dir = model_file.parent
        seed_name = seed_dir.name
        m_seed = re.match(r"seed(\d+)", seed_name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        cfg_dir = seed_dir.parent
        cfg_name = cfg_dir.name
        m_cfg = re.match(pattern, cfg_name)
        if not m_cfg:
            continue

        d = int(m_cfg.group(1))
        if dims and d not in dims:
            continue
        P = int(m_cfg.group(2))
        N = int(m_cfg.group(3))
        chi = int(float(m_cfg.group(4)))

        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))

    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    print(f"Found {len(selected)} runs with model.pt")
    return selected


def load_model(run_dir: Path, config: Dict[str, int], device: torch.device) -> Optional[FCN3NetworkActivationGeneric]:
    """Load a trained FCN3NetworkActivationGeneric model from disk."""
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(config.get("d"))
    P = int(config.get("P"))
    N = int(config.get("N"))
    chi = int(config.get("chi", N))

    state_dict = torch.load(model_path, map_location=device)
    ens = int(state_dict['W0'].shape[0])

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return model


def compute_preactivation_histogram_streaming(
    model: FCN3NetworkActivationGeneric,
    d: int,
    device: torch.device,
    n_samples: int = 10_000_000,
    batch_size: int = 100_000,
    n_bins: int = 200,
    bin_range: Optional[Tuple[float, float]] = None,
    projection_type: str = "linear_target",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of h1_preactivation projection onto Hermite polynomials.
    
    Computes projections of h1_preactivation onto normalized Hermite polynomial directions:
    - linear_target: x[:,0] (target direction)
    - linear_perp: x[:,3] (perpendicular direction)
    - cubic_target: He3(x[:,0]) = (x[:,0]^3 - 3*x[:,0]) / sqrt(6)
    - cubic_perp: He3(x[:,3]) = (x[:,3]^3 - 3*x[:,3]) / sqrt(6)
    
    Uses matmul for efficient computation of projections.
    
    Args:
        model: The trained network model
        d: Input dimension
        device: Torch device to use
        n_samples: Total number of input samples to generate
        batch_size: Number of samples per batch
        n_bins: Number of histogram bins
        bin_range: Optional (min, max) for histogram bins. If None, computed from all data.
        projection_type: Type of projection ("linear_target", "linear_perp", "cubic_target", "cubic_perp")
        seed: Random seed for reproducibility
    
    Returns:
        density: Probability density (n_bins,)
        bin_edges: Histogram bin edges (n_bins+1,)
    """
    model.to(device)
    model.device = device
    model.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize projection accumulators for each ensemble member (ens, n1)
    projection = None
    
    # Accumulate projections across batches
    print(f"Computing h1_preactivation {projection_type} projection for {n_samples:,} samples in {n_batches} batches of {batch_size:,}...")
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Computing projection"):
            # Generate batch
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            X = torch.randn(current_batch_size, d, device=device)
            
            # Compute h1_preactivation: (batch_size, ens, n1)
            h1 = model.h1_preactivation(X)
            batch_size_actual, ens, n1 = h1.shape
            
            # Create projection direction based on type
            if projection_type == "linear_target":
                direction = X[:, 0]  # (batch_size,)
            elif projection_type == "linear_perp":
                direction = X[:, 3] if d > 3 else torch.randn_like(X[:, 0])
            elif projection_type == "cubic_target":
                x0 = X[:, 0]
                direction = (x0**3 - 3.0 * x0)  # He3(x[:,0])
            elif projection_type == "cubic_perp":
                x3 = X[:, 3] if d > 3 else torch.randn_like(X[:, 0])
                direction = (x3**3 - 3.0 * x3)  # He3(x[:,3])
            else:
                raise ValueError(f"Unknown projection_type: {projection_type}")
            
            # Normalize direction
            direction_normed = direction 
            
            # Project h1 onto direction using matmul: (batch_size, ens, n1) @ (batch_size,) -> (ens, n1)
            # Reshape for matmul: direction is (batch_size,), h1 is (batch_size, ens*n1)
            h1_flat = h1.reshape(batch_size_actual, -1)  # (batch_size, ens*n1)
            batch_projection = (direction_normed @ h1_flat).reshape(ens, n1) / n_samples
            
            # Accumulate projection
            if projection is None:
                projection = batch_projection.cpu().numpy()
            else:
                projection += batch_projection.cpu().numpy()
    
    # Flatten projection for histogram
    h1_all = projection.flatten()
    print(f"Projection shape: {projection.shape}, Total values for histogram: {len(h1_all):,}")
    
    # Determine bin range if not provided
    if bin_range is None:
        print("Computing histogram range from all data...")
        bin_range = (float(np.percentile(h1_all, 0.1)), float(np.percentile(h1_all, 99.9)))
        print(f"  Bin range: [{bin_range[0]:.4f}, {bin_range[1]:.4f}]")
    
    # Compute histogram once
    print("Computing histogram...")
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    counts, _ = np.histogram(h1_all, bins=bin_edges)
    
    # Convert to density
    bin_width = bin_edges[1] - bin_edges[0]
    density = counts / (counts.sum() ) 
    
    print(f"Samples in histogram: {counts.sum():,}")
    print(f"Samples outside range: {len(h1_all) - counts.sum():,}")
    variance = np.var(h1_all)
    print(f"Variance of projection values: {variance:.4f}")
    return density, bin_edges, variance


def save_histogram_data(
    run_dir: Path,
    cfg: Dict[str, int],
    density: np.ndarray,
    bin_edges: np.ndarray,
    n_samples: int,
) -> Path:
    """Save histogram data to JSON file."""
    out_dir = run_dir / "histograms"
    out_dir.mkdir(exist_ok=True)
    
    out_path = out_dir / f"h1_preactivation_histogram_{n_samples}.json"
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Compute action (negative log probability density)
    action = -np.log(density + 1e-30)  # Add small constant to avoid log(0)
    
    data = {
        "config": cfg,
        "n_samples": n_samples,
        "n_bins": len(density),
        "density": density.tolist(),
        "action": action.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
    }
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved histogram data to {out_path}")
    return out_path


def plot_histogram(
    run_dir: Path,
    cfg: Dict[str, int],
    density: np.ndarray,
    bin_edges: np.ndarray,
    n_samples: int,
) -> Path:
    """Plot and save action (negative log probability density) visualization."""
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    
    out_path = out_dir / f"h1_preactivation_action_{n_samples}.png"
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Compute action (negative log probability density)
    action = -np.log(density + 1e-30)  # Add small constant to avoid log(0)
    
    fig, ax =  plt.subplots(figsize=(12, 6))
    
    # Plot action (-log probability) as a line or scatter
    mask = np.isfinite(action) & (density > 1e-10)  # Only plot where density is non-negligible
    ax.plot(bin_centers[mask], action[mask], 'o-', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_xlabel('h1 Preactivation Value')
    ax.set_ylabel('Action = -log(probability density)')
    ax.set_title(
        f"h1_preactivation Action (-Log Probability Density)\n"
        f"d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}, seed={cfg['seed']}\n"
        f"{n_samples:,} samples"
    )
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.sum(density * bin_centers * bin_width)
    var_val = np.sum(density * (bin_centers - mean_val)**2 * bin_width)
    std_val = np.sqrt(var_val)
    
    ax.text(
        0.98, 0.98,
        f"Mean: {mean_val:.2e}\nVar: {var_val:.2e}",
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved action plot to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compute h1_preactivation histogram using streaming")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory to scan")
    parser.add_argument("--dims", type=int, nargs="+", help="Filter by dimension(s)")
    parser.add_argument("--suffix", type=str, default="", help="Directory suffix filter")
    parser.add_argument("--n-samples", type=int, default=10_000_000, help="Total number of samples")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Batch size for streaming")
    parser.add_argument("--n-bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--bin-range", type=float, nargs=2, metavar=("MIN", "MAX"), 
                        help="Fixed bin range [min, max]. Auto-detected if not provided.")
    parser.add_argument("--projection-type", type=str, default="all",
                        choices=["linear_target", "linear_perp", "cubic_target", "cubic_perp", "all"],
                        help="Projection type to compute (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    print(f"Using device: {device}")

    run_dirs = find_run_dirs(base_dir, dims=args.dims, suffix=args.suffix)
    if not run_dirs:
        print("No runs found")
        return

    bin_range = tuple(args.bin_range) if args.bin_range else None

    # --- Aggregate all runs for each projection type ---
    if args.projection_type == "all":
        projection_types = ["linear_target", "linear_perp", "cubic_target", "cubic_perp"]
    else:
        projection_types = [args.projection_type]

    # Store all densities, variances, bin_centers for each projection type
    all_results = {pt: [] for pt in projection_types}
    all_bin_centers = {}
    for run_dir, cfg in run_dirs:
        print(f"\n{'='*80}")
        print(f"Processing: {run_dir}")
        print(f"Config: d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}, seed={cfg['seed']}")

        # Load model
        model = load_model(run_dir, cfg, device)
        if model is None:
            print("  Error: Could not load model")
            continue

        print(f"  Model loaded: ens={model.ens}, n1={model.n1}")

        for proj_type in projection_types:
            print(f"\n  Computing {proj_type} projection...")
            try:
                density, bin_edges, variance = compute_preactivation_histogram_streaming(
                    model=model,
                    d=cfg['d'],
                    device=device,
                    n_samples=args.n_samples,
                    batch_size=args.batch_size,
                    n_bins=args.n_bins,
                    bin_range=bin_range,
                    projection_type=proj_type,
                    seed=args.seed if args.seed is not None else cfg.get('seed'),
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                all_results[proj_type].append({
                    "density": density,
                    "bin_centers": bin_centers,
                    "variance": variance,
                })
                all_bin_centers[proj_type] = bin_centers
            except Exception as e:
                print(f"    Error processing projection {proj_type}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # --- Plot aggregated results with error bars ---
    if not args.no_plot:
        plot_out_dir = Path(args.base_dir).resolve() / "plots"
        plot_out_dir.mkdir(exist_ok=True)
        plot_path = plot_out_dir / f"h1_preactivation_action_all_{args.n_samples}.png"

        import matplotlib
        matplotlib.rcParams['text.usetex'] = False  # Use mathtext for $\sigma^2$

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        # Linear subplot
        for proj_type in ["linear_target", "linear_perp"]:
            if proj_type in all_results and all_results[proj_type]:
                densities = np.array([r["density"] for r in all_results[proj_type]])
                means = np.mean(densities, axis=0)
                stds = np.std(densities, axis=0)
                bin_centers = all_bin_centers[proj_type]
                # Compute mean variance for legend
                variances = [r["variance"] for r in all_results[proj_type]]
                mean_var = np.mean(variances)
                axes[0].errorbar(
                    bin_centers, -np.log(means + 1e-30), yerr=stds/(means+1e-30),
                    label=f"{proj_type} ($\\sigma^2$={mean_var:.2e})", linewidth=2, alpha=0.7
                )
        axes[0].set_xlabel('Projection Value')
        axes[0].set_ylabel('Action = -log(probability density)')
        axes[0].set_title('Linear Projections')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Cubic subplot
        for proj_type in ["cubic_target", "cubic_perp"]:
            if proj_type in all_results and all_results[proj_type]:
                densities = np.array([r["density"] for r in all_results[proj_type]])
                means = np.mean(densities, axis=0)
                stds = np.std(densities, axis=0)
                bin_centers = all_bin_centers[proj_type]
                variances = [r["variance"] for r in all_results[proj_type]]
                mean_var = np.mean(variances)
                axes[1].errorbar(
                    bin_centers, -np.log(means + 1e-30), yerr=stds/(means+1e-30),
                    label=f"{proj_type} ($\\sigma^2$={mean_var:.2e})", linewidth=2, alpha=0.7
                )
        axes[1].set_xlabel('Projection Value')
        axes[1].set_title('Cubic Projections')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle(
            f"h1_preactivation Action (-Log Probability Density)\n"
            f"d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']}\n"
            f"{args.n_samples:,} samples (mean $\\pm$ std over seeds)"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved combined action plot to {plot_path}")

    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()
