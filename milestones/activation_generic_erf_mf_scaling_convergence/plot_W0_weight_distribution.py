#!/usr/bin/env python3
"""Plot W1 weight distributions for all discovered runs.

Scans run folders (d<d>_P<P>_N<N>_chi<chi>/seed<seed>/model.pt), loads each model,
and saves a two-panel histogram to <run>/plots/W1_weight_distribution.png:
- Left: flattened W1[:, :, 0] (first input column into layer 2)
- Right: mean across columns 1: (or all columns if only one), flattened

Also computes theoretical lWT and lWP eigenvalues and overlays Gaussian distributions.
Produces an aggregate plot combining all seed histograms.

Usage:
    python plot_W1_weight_distribution.py --base-dir <runs_root> [--dims 150] [--suffix "_something"]
"""

import argparse
import re
import subprocess
import tempfile
import json as json_lib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_theory_eigenvalues(d: int, N: int, P: int, chi: float, kappa: float, epsilon: float = 0.0):
    """Compute theoretical lWT and lWP eigenvalues via Julia."""
    try:
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
        n2 = N
        b = 4.0 / (3.0 * np.pi)
        lr = 1e-6
        max_iter = 6_000_000
        anneal = True
        anneal_steps = 30000
        tol = 1e-12
        precision = 8
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cmd = [
                "julia",
                str(julia_script),
                "--d", str(d),
                "--kappa", str(kappa),
                "--epsilon", str(epsilon),
                "--P", str(P),
                "--n1", str(N),
                "--n2", str(n2),
                "--chi", str(chi),
                "--b", str(b),
                "--lr", str(lr),
                "--max-iter", str(max_iter),
                "--anneal-steps", str(anneal_steps),
                "--tol", str(tol),
                "--precision", str(precision),
                "--to", str(tmp_path),
                "--quiet"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(tmp_path, "r") as f:
                result = json_lib.load(f)
            
            # Extract lWT and lWP from result (use correct keys from Julia output)
            target_block = result.get("target", {})
            perp_block = result.get("perpendicular", {})
            lW1T = target_block.get("lWT")
            lW1P = perp_block.get("lWP")
            
            if lW1T is not None:
                lW1T = float(lW1T)
            if lW1P is not None:
                lW1P = float(lW1P)
            
            print(f"  Theory: d={d}, P={P}, N={N}, chi={chi}, kappa={kappa} => lW1T={lW1T}, lW1P={lW1P}")
            return lW1T, lW1P
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"Warning: could not compute theory eigenvalues: {e}")
        import traceback
        traceback.print_exc()
        return None, None



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


def plot_W1_weight_distribution(
    run_dir: Path,
    cfg: Dict[str, int],
    model: FCN3NetworkActivationGeneric,
    out_path: Path,
    lWT: Optional[float] = None,
    lWP: Optional[float] = None,
) -> None:
    """Plot distribution of W1 weights analogous to W0 plot in plot_full_spectrum."""
    try:
        W1 = model.W0.detach().cpu().numpy()  # shape (ens, n2, n1)

        w1_feature0 = W1[:, :, 0].flatten()
        
        # For lWP: compute variance per d channel and average them
        if W1.shape[2] > 1:
            # Compute variance for each of the perpendicular dimensions (1 through d-1)
            perp_variances = np.array([W1[:, :, i].flatten().var() for i in range(1, W1.shape[2])])
            empirical_lWP = np.mean(perp_variances)
            # For histogram, use all perpendicular weights flattened
            w1_mean_across_input = W1[:, :, 1:].flatten()
            mean_label = f'W1 perpendicular (cols 1:), avg_var={empirical_lWP:.4e}'
        else:
            empirical_lWP = 0.0
            w1_mean_across_input = np.array([])
            mean_label = 'No perpendicular dimensions'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot W1[:, :, 0] with Gaussian overlay
        counts0, bins0, _ = axes[0].hist(w1_feature0, bins=50, color='tab:blue', alpha=0.7, edgecolor='black', density=True)
        axes[0].set_xlabel('Weight value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(
            f'W1[:, :, 0] distribution\nVar={w1_feature0.var():.4e}, Mean={w1_feature0.mean():.4e}'
        )
        axes[0].grid(True, alpha=0.3)
        
        # Overlay Gaussian if lWT is available
        print(f"  Plotting {run_dir.name}: lWT={lWT}, lWP={lWP}")
        if lWT is not None and lWT > 0:
            print(f"    Adding Gaussian overlay for lWT={lWT}")
            x_range = np.linspace(w1_feature0.min(), w1_feature0.max(), 200)
            gaussian_wt = (1.0 / np.sqrt(2 * np.pi * lWT)) * np.exp(-x_range**2 / (2 * lWT))
            axes[0].plot(x_range, gaussian_wt, 'r-', linewidth=2.5, label=f'Theory Gaussian (σ²={lWT:.4e})', zorder=5)
            axes[0].legend(fontsize=10)
        else:
            print(f"    Skipping Gaussian overlay: lWT={lWT}")

        # Plot perpendicular with Gaussian overlay
        if len(w1_mean_across_input) > 0:
            counts1, bins1, _ = axes[1].hist(w1_mean_across_input, bins=50, color='tab:orange', alpha=0.7, edgecolor='black', density=True)
            axes[1].set_xlabel('Weight value')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(
                f'{mean_label}\nVar={empirical_lWP:.4e}, Mean={w1_mean_across_input.mean():.4e}'
            )
        else:
            axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[1].set_title(mean_label)
        axes[1].grid(True, alpha=0.3)
        
        # Overlay Gaussian if lWP is available
        if lWP is not None and lWP > 0:
            print(f"    Adding Gaussian overlay for lWP={lWP}")
            x_range = np.linspace(w1_mean_across_input.min(), w1_mean_across_input.max(), 200)
            gaussian_wp = (1.0 / np.sqrt(2 * np.pi * lWP)) * np.exp(-x_range**2 / (2 * lWP))
            axes[1].plot(x_range, gaussian_wp, 'r-', linewidth=2.5, label=f'Theory Gaussian (σ²={lWP:.4e})', zorder=5)
            axes[1].legend(fontsize=10)
        else:
            print(f"    Skipping Gaussian overlay: lWP={lWP}")

        fig.suptitle(
            f'W1 Weight Distributions (d={cfg["d"]}, P={cfg["P"]}, N={cfg["N"]}, chi={cfg["chi"]})',
            fontsize=12
        )
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W1 weight distribution plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot W1 weight distribution for {run_dir}: {e}")


def plot_W1_weight_distribution_aggregate(
    runs: List[Tuple[Path, Dict[str, int]]],
    models_data: Dict[Tuple[int, int, int, int], list],
    out_dir: Path,
    lWT: Optional[float] = None,
    lWP: Optional[float] = None,
) -> None:
    """Create aggregate histogram combining all seed data for a config."""
    if not models_data:
        return
    
    try:
        # Extract first config from models_data keys (all should be the same)
        config_key = list(models_data.keys())[0]
        d, P, N, chi = config_key
        print(f"Creating aggregate plot for config {config_key}: lWT={lWT}, lWP={lWP}")
        
        # Collect all weights across seeds
        all_w1_feature0 = []
        all_w1_perpendicular = []
        all_channel_variances = []
        
        for seed_weights in models_data[config_key]:
            w1_feature0, w1_perpendicular = seed_weights
            all_w1_feature0.extend(w1_feature0.flatten())
            
            # Compute per-channel variances for this seed if perpendicular data exists
            if w1_perpendicular.size > 0:
                channel_vars = np.array([w1_perpendicular[:, :, i].flatten().var() for i in range(w1_perpendicular.shape[2])])
                all_channel_variances.append(channel_vars)
            
            # Also collect raw perpendicular weights for histogram
            all_w1_perpendicular.extend(w1_perpendicular.flatten())
        
        all_w1_feature0 = np.array(all_w1_feature0)
        all_w1_perpendicular = np.array(all_w1_perpendicular)
        
        # Compute empirical lWP as average of per-channel variances across all seeds
        if len(all_channel_variances) > 0:
            empirical_lWP_aggregate = np.mean(np.concatenate(all_channel_variances))
        else:
            empirical_lWP_aggregate = 0.0
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Aggregate plot for W1[:, :, 0]
        axes[0].hist(all_w1_feature0, bins=100, color='tab:blue', alpha=0.7, edgecolor='black', density=True)
        axes[0].set_xlabel('Weight value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(
            f'Aggregate W1[:, :, 0] distribution (all seeds)\nVar={all_w1_feature0.var():.4e}, Mean={all_w1_feature0.mean():.4e}'
        )
        axes[0].grid(True, alpha=0.3)
        
        if lWT is not None and lWT > 0:
            x_range = np.linspace(all_w1_feature0.min(), all_w1_feature0.max(), 200)
            gaussian_wt = (1.0 / np.sqrt(2 * np.pi * lWT)) * np.exp(-x_range**2 / (2 * lWT))
            axes[0].plot(x_range, gaussian_wt, 'r-', linewidth=2.5, label=f'Theory Gaussian (σ²={lWT:.4e})', zorder=5)
            axes[0].legend(fontsize=10)
        
        # Aggregate plot for mean across input columns
        axes[1].hist(all_w1_perpendicular, bins=100, color='tab:orange', alpha=0.7, edgecolor='black', density=True)
        axes[1].set_xlabel('Weight value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(
            f'Aggregate W1 perpendicular (cols 1:, all seeds)\nAvg_var={empirical_lWP_aggregate:.4e}, Mean={all_w1_perpendicular.mean():.4e}'
        )
        axes[1].grid(True, alpha=0.3)
        
        if lWP is not None and lWP > 0:
            if len(all_w1_perpendicular) > 0:
                x_range = np.linspace(all_w1_perpendicular.min(), all_w1_perpendicular.max(), 200)
                gaussian_wp = (1.0 / np.sqrt(2 * np.pi * lWP)) * np.exp(-x_range**2 / (2 * lWP))
                axes[1].plot(x_range, gaussian_wp, 'r-', linewidth=2.5, label=f'Theory Gaussian (σ²={lWP:.4e})', zorder=5)
                axes[1].legend(fontsize=10)
        
        fig.suptitle(
            f'Aggregate W1 Weight Distributions (d={d}, P={P}, N={N}, chi={chi})',
            fontsize=12
        )
        fig.tight_layout()
        
        out_dir.mkdir(parents=True, exist_ok=True)
        aggregate_plot_path = out_dir / "W1_weight_distribution_aggregate.png"
        fig.savefig(aggregate_plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved aggregate W1 weight distribution plot to {aggregate_plot_path}")
    except Exception as e:
        print(f"Warning: failed to plot aggregate W1 distribution: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot W1 weight distributions for runs.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to plot.")
        return

    # Group runs by config (d, P, N, chi) and compute theory once per config
    config_to_theory = {}  # (d, P, N, chi, kappa) -> (lWT, lWP)
    config_to_runs = {}  # (d, P, N, chi) -> [list of (run_dir, cfg)]
    config_to_weights = {}  # (d, P, N, chi) -> [(w1_feature0, w1_mean_input), ...]
    
    for run_dir, cfg in runs:
        config_key = (cfg["d"], cfg["P"], cfg["N"], cfg["chi"])
        
        if config_key not in config_to_runs:
            config_to_runs[config_key] = []
            config_to_weights[config_key] = []
        
        config_to_runs[config_key].append((run_dir, cfg))
        
        # Load model and extract weights
        model = load_model(run_dir, cfg, device)
        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue
        
        W1 = model.W0.detach().cpu().numpy()
        w1_feature0 = W1[:, :, 0]
        
        # Store full perpendicular weight matrix for per-channel variance computation in aggregate
        if W1.shape[2] > 1:
            w1_perpendicular = W1[:, :, 1:]
        else:
            w1_perpendicular = np.array([])
        
        config_to_weights[config_key].append((w1_feature0, w1_perpendicular))
        
        # Compute theory and plot individual run
        config_theory_key = (cfg["d"], cfg["P"], cfg["N"], cfg["chi"], cfg.get("kappa", 0.07))
        if config_theory_key not in config_to_theory:
            print(f"Computing theory for config {config_key}...")
            kappa = cfg.get("kappa", 0.07)
            lWT, lWP = compute_theory_eigenvalues(cfg["d"], cfg["N"], cfg["P"], cfg["chi"], kappa)
            config_to_theory[config_theory_key] = (lWT, lWP)
        else:
            lWT, lWP = config_to_theory[config_theory_key]
        
        # Plot individual model
        out_path = run_dir / "plots" / "W1_weight_distribution.png"
        plot_W1_weight_distribution(run_dir, cfg, model, out_path, lWT=lWT, lWP=lWP)
    
    # Create aggregate plots per config
    for config_key, runs_list in config_to_runs.items():
        if config_key not in config_to_weights or not config_to_weights[config_key]:
            continue
        
        d, P, N, chi = config_key
        config_theory_key = (d, P, N, chi, 0.07)  # Default kappa
        lWT, lWP = config_to_theory.get(config_theory_key, (None, None))
        
        # Create output directory for aggregate plots
        agg_out_dir = runs_list[0][0].parent / "plots"
        plot_W1_weight_distribution_aggregate(runs_list, {config_key: config_to_weights[config_key]}, 
                                             agg_out_dir, lWT=lWT, lWP=lWP)


if __name__ == "__main__":
    main()
