#!/usr/bin/env python3
"""
Analyze eigenvalue convergence with different sample sizes.

Usage:
    python analyze_eig_convergence.py --run-dir d50_P200_N200_chi_200.0_lr_5e-07_T_2.0_seed_0
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

torch.set_default_dtype(torch.float32)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def load_model_and_config(run_dir: Path):
    """Load trained model and configuration from run directory."""
    # Load config
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    
    with open(cfg_path, "r") as f:
        config = json.load(f)
    
    d = int(config["d"])
    P = int(config["P"])
    N = int(config["N"])
    ens = int(config.get("ens", 5))
    
    # Create model
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N)
    )
    
    # Load weights (try checkpoint first, then fallbacks)
    checkpoint_path = run_dir / "checkpoint.pt"
    final_path = run_dir / "model_final.pt"
    model_path = run_dir / "model.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
    elif final_path.exists():
        state_dict = torch.load(final_path, map_location="cpu")
        model.load_state_dict(state_dict)
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No model weights found in {run_dir}")
    
    model.eval()
    return model, config


def compute_eigenvalues_at_size(model, d, sample_size, dataset_seed=42):
    """Compute H eigenvalues using a dataset of given sample size."""
    torch.manual_seed(dataset_seed)
    X = torch.randn(sample_size, d)
    
    with torch.no_grad():
        result = model.H_eig(X, X, std=True)
        if isinstance(result, tuple):
            eigenvalues, eigenvalues_std = result
        else:
            eigenvalues = result
            eigenvalues_std = None
    
    eigenvalues = eigenvalues.cpu().numpy()
    if eigenvalues_std is not None:
        eigenvalues_std = eigenvalues_std.cpu().numpy()
    
    return eigenvalues, eigenvalues_std


def main():
    parser = argparse.ArgumentParser(
        description='Analyze eigenvalue convergence with different sample sizes'
    )
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to run directory with trained model')
    parser.add_argument('--sizes', type=int, nargs='+',
                       default=[1000, 2000, 3000, 4000, 5000, 8000],
                       help='Sample sizes to test')
    parser.add_argument('--dataset-seed', type=int, default=42,
                       help='Random seed for generating test data')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for plot (default: run_dir/eig_convergence.png)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    print(f"Loading model from: {run_dir}")
    model, config = load_model_and_config(run_dir)
    d = config["d"]
    
    print(f"\nModel configuration:")
    print(f"  d={d}, P={config['P']}, N={config['N']}")
    print(f"  ens={config.get('ens', 5)}")
    
    print(f"\nComputing eigenvalues for sample sizes: {args.sizes}")
    print("="*60)
    
    results = []
    for size in args.sizes:
        print(f"Sample size: {size:5d} ... ", end='', flush=True)
        eigenvalues, eigenvalues_std = compute_eigenvalues_at_size(
            model, d, size, args.dataset_seed
        )
        
        max_eig = eigenvalues.max()
        mean_eig = eigenvalues.mean()
        min_eig = eigenvalues.min()
        
        if eigenvalues_std is not None:
            max_std = eigenvalues_std[eigenvalues.argmax()]
        else:
            max_std = 0.0
        
        results.append({
            'size': size,
            'max': max_eig,
            'mean': mean_eig,
            'min': min_eig,
            'max_std': max_std,
            'all_eigs': eigenvalues,
            'all_stds': eigenvalues_std
        })
        
        print(f"max={max_eig:.6f}, mean={mean_eig:.6f}, min={min_eig:.6f}")
    
    print("="*60)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Max eigenvalue vs sample size
    ax = axes[0, 0]
    sizes = [r['size'] for r in results]
    max_eigs = [r['max'] for r in results]
    max_stds = [r['max_std'] for r in results]
    
    ax.errorbar(sizes, max_eigs, yerr=max_stds, fmt='o-', linewidth=2, 
                markersize=8, capsize=5, capthick=2)
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Max Eigenvalue', fontsize=12)
    ax.set_title('Max Eigenvalue vs Sample Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean eigenvalue vs sample size
    ax = axes[0, 1]
    mean_eigs = [r['mean'] for r in results]
    ax.plot(sizes, mean_eigs, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Mean Eigenvalue', fontsize=12)
    ax.set_title('Mean Eigenvalue vs Sample Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Min eigenvalue vs sample size
    ax = axes[1, 0]
    min_eigs = [r['min'] for r in results]
    ax.plot(sizes, min_eigs, 'o-', linewidth=2, markersize=8, color='#d62728')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Min Eigenvalue', fontsize=12)
    ax.set_title('Min Eigenvalue vs Sample Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: All eigenvalues for largest sample size
    ax = axes[1, 1]
    largest = results[-1]
    eigs_sorted = np.sort(largest['all_eigs'])[::-1]
    if largest['all_stds'] is not None:
        sort_idx = np.argsort(largest['all_eigs'])[::-1]
        stds_sorted = largest['all_stds'][sort_idx]
        ax.errorbar(range(1, len(eigs_sorted) + 1), eigs_sorted, 
                   yerr=stds_sorted, fmt='o-', markersize=4, 
                   capsize=3, alpha=0.7)
    else:
        ax.plot(range(1, len(eigs_sorted) + 1), eigs_sorted, 'o-', 
               markersize=4, alpha=0.7)
    ax.set_xlabel('Eigenvalue Index (sorted)', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(f'Full Spectrum (size={largest["size"]})', 
                fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Eigenvalue Convergence Analysis\n{run_dir.name}', 
                fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    # Save plot
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "eig_convergence.png"
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Save data to JSON
    json_path = output_path.with_suffix('.json')
    save_data = {
        'sizes': sizes,
        'max_eigenvalues': max_eigs,
        'mean_eigenvalues': mean_eigs,
        'min_eigenvalues': min_eigs,
        'max_stds': max_stds,
        'config': config
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Data saved to: {json_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
