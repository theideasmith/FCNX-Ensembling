"""
Plot GPR vs EK loss comparison results.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_gpr_vs_ek(results: Dict[str, Any], output_dir: Path):
    """Plot GPR loss vs EK predicted loss."""
    d_values = sorted([int(d) for d in results.keys()])
    ek_losses = [results[str(d)]['ek_loss'] for d in d_values]
    gpr_means = [results[str(d)]['gpr_loss_mean'] for d in d_values]
    gpr_stds = [results[str(d)]['gpr_loss_std'] for d in d_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot EK prediction
    ax.plot(d_values, ek_losses, 'o-', label='EK Prediction', linewidth=2, markersize=8, color='blue')
    
    # Plot GPR mean with error bars
    ax.errorbar(d_values, gpr_means, yerr=gpr_stds, fmt='s-', label='GPR (mean ± std)', 
                linewidth=2, markersize=8, capsize=5, color='red', alpha=0.7)
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('GPR vs EK Loss Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(d_values)
    ax.set_yscale('log')
    plt.tight_layout()
    output_path = output_dir / 'gpr_vs_ek_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_loss_ratio(results: Dict[str, Any], output_dir: Path):
    """Plot ratio of GPR loss to EK predicted loss."""
    d_values = sorted([int(d) for d in results.keys()])
    ratios = []
    
    for d in d_values:
        ek = results[str(d)]['ek_loss']
        gpr = results[str(d)]['gpr_loss_mean']
        ratio = gpr / ek if ek > 0 else np.nan
        ratios.append(ratio)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(d_values, ratios, 'o-', linewidth=2, markersize=10, color='green')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='GPR = EK')
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('Loss Ratio (GPR / EK)', fontsize=12)
    ax.set_title('GPR Loss Ratio vs EK Prediction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(d_values)
    
    plt.tight_layout()
    output_path = output_dir / 'gpr_ek_loss_ratio.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_ek_components(results: Dict[str, Any], output_dir: Path):
    """Plot EK bias and variance components."""
    d_values = sorted([int(d) for d in results.keys()])
    biases = [results[str(d)]['ek_bias'] for d in d_values]
    variances = [results[str(d)]['ek_variance'] for d in d_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(d_values, biases, 'o-', label='Bias', linewidth=2, markersize=8, color='purple')
    ax.plot(d_values, variances, 's-', label='Variance', linewidth=2, markersize=8, color='orange')
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('Component Value', fontsize=12)
    ax.set_title('EK Bias and Variance Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(d_values)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'ek_components.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gpr_distribution(results: Dict[str, Any], output_dir: Path):
    """Plot distribution of GPR losses for each dimension."""
    d_values = sorted([int(d) for d in results.keys()])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    box_data = [np.array(results[str(d)]['gpr_losses']) for d in d_values]
    bp = ax.boxplot(box_data, labels=d_values, patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('GPR Loss', fontsize=12)
    ax.set_title('Distribution of GPR Losses Across Datasets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'gpr_loss_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(results: Dict[str, Any], output_dir: Path):
    """Create and save a summary table as an image."""
    d_values = sorted([int(d) for d in results.keys()])
    
    table_data = []
    for d in d_values:
        p = results[str(d)]['P']
        ek = results[str(d)]['ek_loss']
        gpr = results[str(d)]['gpr_loss_mean']
        ratio = gpr / ek if ek > 0 else np.nan
        
        table_data.append([
            f"{d}",
            f"{p}",
            f"{ek:.4e}",
            f"{gpr:.4e}",
            f"{ratio:.3f}"
        ])
    
    fig, ax = plt.subplots(figsize=(10, len(d_values) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=['d', 'P', 'EK Loss', 'GPR Loss', 'Ratio'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.1, 0.25, 0.25, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(d_values) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('GPR vs EK Loss Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'gpr_ek_summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gpr_vs_ek_predictions(results: Dict[str, Any], output_dir: Path):
    """Plot GPR predictions vs EK predictions: (1/d) / (1/d + kappa) * x[0]."""
    d_values = sorted([int(d) for d in results.keys()])
    
    # Load first dataset predictions for each d
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, d in enumerate(d_values[:9]):  # Plot up to 9 dimensions
        ax = axes[idx]
        
        gpr_preds = results[str(d)].get('gpr_predictions', [])
        ek_factor = results[str(d)].get('ek_prediction_factor', 0.0)
        
        if gpr_preds:
            # Generate corresponding x[0] values (we need to regenerate first dataset)
            P = results[str(d)]['P']
            # Use same seed as in gpr_ek_comparison.py for first dataset
            import torch
            torch.manual_seed(1)  # seeds[0] + 0 * 1000 = 1
            X = torch.randn(P, d)
            x0_values = X[:, 0].numpy()
            
            # EK prediction: (1/d) / (1/d + kappa) * x[0]
            ek_preds = ek_factor * x0_values
            
            # Take only training portion (80%)
            split_idx = int(0.8 * P)
            x0_train = x0_values[:split_idx]
            ek_preds_train = ek_preds[:split_idx]
            gpr_preds_array = np.array(gpr_preds[:split_idx])
            
            # Scatter plot
            ax.scatter(ek_preds_train, gpr_preds_array, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(ek_preds_train.min(), gpr_preds_array.min())
            max_val = max(ek_preds_train.max(), gpr_preds_array.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
            
            ax.set_xlabel(f'EK Prediction', fontsize=10)
            ax.set_ylabel(f'GPR Prediction', fontsize=10)
            ax.set_title(f'd={d}, P={P}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Compute and display correlation
            corr = np.corrcoef(ek_preds_train, gpr_preds_array)[0, 1]
            ax.text(0.05, 0.95, f'ρ = {corr:.4f}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(d_values), 9):
        axes[idx].axis('off')
    
    plt.suptitle('GPR vs EK Prediction Factor Comparison\nEK: (1/d)/(1/d + κ) × x₀', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'gpr_vs_ek_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_factors(results: Dict[str, Any], output_dir: Path):
    """Plot EK prediction factor (1/d)/(1/d + kappa) vs d."""
    d_values = sorted([int(d) for d in results.keys()])
    ek_factors = [results[str(d)].get('ek_prediction_factor', 0.0) for d in d_values]
    theoretical = [1.0 / d for d in d_values]  # lH = 1/d
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(d_values, ek_factors, 'o-', label='EK Prediction Factor: (1/d)/(1/d + κ)', 
            linewidth=2, markersize=8, color='blue')
    ax.plot(d_values, theoretical, 's--', label='Theoretical lH = 1/d', 
            linewidth=2, markersize=8, color='green', alpha=0.7)
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('Prediction Factor', fontsize=12)
    ax.set_title('EK Prediction Factor vs Input Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(d_values)
    
    plt.tight_layout()
    output_path = output_dir / 'ek_prediction_factor.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "data" / "results_gpr"
    plots_dir = base_dir / "plots_gpr"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "gpr_vs_ek_loss.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run gpr_ek_comparison.py first to generate results.")
        return
    
    print(f"Loading results from {results_file}")
    results = load_results(results_file)
    
    print(f"\nGenerating plots in {plots_dir}...\n")
    
    plot_gpr_vs_ek(results, plots_dir)
    plot_loss_ratio(results, plots_dir)
    plot_ek_components(results, plots_dir)
    plot_gpr_distribution(results, plots_dir)
    plot_gpr_vs_ek_predictions(results, plots_dir)
    plot_prediction_factors(results, plots_dir)
    plot_summary_table(results, plots_dir)
    
    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
