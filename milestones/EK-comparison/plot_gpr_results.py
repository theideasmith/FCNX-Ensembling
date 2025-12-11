"""
Plot GPR vs EK loss comparison results.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from gpr_ek_comparison import generate_synthetic_data


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_gpr_vs_ek(results: Dict[str, Any], output_dir: Path):
    """Plot GPR loss vs EK predicted loss."""
    d_values = sorted([int(d) for d in results.keys()])
    ek_losses = [results[str(d)]['ek_loss'] for d in d_values]
    gpr_losses = [results[str(d)]['gpr_loss'] for d in d_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot EK prediction
    ax.plot(d_values, ek_losses, 'o-', label='EK Prediction', linewidth=2, markersize=8, color='blue')
    
    # Plot GPR
    ax.plot(d_values, gpr_losses, 's-', label='GPR (Bias + Variance)', 
            linewidth=2, markersize=8, color='red', alpha=0.7)
    
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
        gpr = results[str(d)]['gpr_loss']
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
    """Plot EK and GPR bias and variance components."""
    d_values = sorted([int(d) for d in results.keys()])
    ek_biases = [results[str(d)]['ek_bias'] for d in d_values]
    ek_variances = [results[str(d)]['ek_variance'] for d in d_values]
    gpr_biases = [results[str(d)]['gpr_bias'] for d in d_values]
    gpr_variances = [results[str(d)]['gpr_variance'] for d in d_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(d_values, ek_biases, 'o-', label='EK Bias', linewidth=2, markersize=8, color='purple')
    ax.plot(d_values, ek_variances, 's-', label='EK Variance', linewidth=2, markersize=8, color='orange')
    ax.plot(d_values, gpr_biases, 'o--', label='GPR Bias', linewidth=2, markersize=8, color='purple', alpha=0.6)
    ax.plot(d_values, gpr_variances, 's--', label='GPR Variance', linewidth=2, markersize=8, color='orange', alpha=0.6)
    
    ax.set_xlabel('Input Dimension (d)', fontsize=12)
    ax.set_ylabel('Component Value', fontsize=12)
    ax.set_title('EK vs GPR Bias and Variance Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(d_values)
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'ek_gpr_components.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_bias_variance_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot separate comparison of bias and variance between EK and GPR."""
    d_values = sorted([int(d) for d in results.keys()])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bias comparison
    ek_biases = [results[str(d)]['ek_bias'] for d in d_values]
    gpr_biases = [results[str(d)]['gpr_bias'] for d in d_values]
    
    ax1.plot(d_values, ek_biases, 'o-', label='EK Bias', linewidth=2, markersize=8, color='blue')
    ax1.plot(d_values, gpr_biases, 's-', label='GPR Bias', linewidth=2, markersize=8, color='red', alpha=0.7)
    ax1.set_xlabel('Input Dimension (d)', fontsize=12)
    ax1.set_ylabel('Bias', fontsize=12)
    ax1.set_title('Bias Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(d_values)
    ax1.set_yscale('log')
    
    # Variance comparison
    ek_variances = [results[str(d)]['ek_variance'] for d in d_values]
    gpr_variances = [results[str(d)]['gpr_variance'] for d in d_values]
    
    ax2.plot(d_values, ek_variances, 'o-', label='EK Variance', linewidth=2, markersize=8, color='blue')
    ax2.plot(d_values, gpr_variances, 's-', label='GPR Variance', linewidth=2, markersize=8, color='red', alpha=0.7)
    ax2.set_xlabel('Input Dimension (d)', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Variance Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(d_values)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / 'bias_variance_comparison.png'
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
        gpr = results[str(d)]['gpr_loss']
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
    plot_bias_variance_comparison(results, plots_dir)
    plot_summary_table(results, plots_dir)
    
    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
