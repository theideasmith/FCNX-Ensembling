"""
Plotting script for EK-Comparison results.

Generates three main plots:
1. EK prediction factor vs empirical (actual network output)
2. Sum-reduction loss averaged over datasets vs EK predicted loss
3. Individual component analysis (bias, variance decomposition)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import seaborn as sns

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12


def load_results(results_dir: Path) -> Tuple[Dict, Dict]:
    """Load configuration and comparison data from results directory."""
    with open(results_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    with open(results_dir / 'ek_comparison.json', 'r') as f:
        comparison = json.load(f)
    
    return config, comparison


def plot_predictions(config: Dict, comparison: Dict, output_path: Path):
    """
    Plot 1: EK prediction factor vs empirical network outputs.
    
    The EK prediction factor is: (1/d) / (1/d + kappa)
    This represents the theoretical prediction of network output at initialization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    d_values = sorted([int(d) for d in comparison.keys()])
    ek_factors = []
    
    for d in d_values:
        ek_factors.append(comparison[str(d)]['ek_prediction_factor'])
    
    ax.plot(d_values, ek_factors, 'o-', linewidth=2.5, markersize=8, 
            label=f'EK Prediction (κ={config["kappa_ek"]})', color='steelblue')
    
    ax.set_xlabel('Input Dimension (d)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prediction Factor: (1/d) / (1/d + κ)', fontsize=13, fontweight='bold')
    ax.set_title('EK Prediction Factor vs Input Dimension\n' + 
                 f'Network: N={config["N"]}, χ={config["chi"]}, κ={config["kappa_ek"]}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ek_prediction_factor.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: ek_prediction_factor.png")
    plt.close()


def plot_loss_comparison(config: Dict, comparison: Dict, output_path: Path):
    """
    Plot 2: Sum-reduction loss comparison.
    
    - EK Predicted Loss: bias + variance (from NNGP theory)
    - Empirical Loss: mean empirical MSE averaged over all datasets/ensembles
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    d_values = sorted([int(d) for d in comparison.keys()])
    ek_losses = []
    empirical_losses = []
    empirical_stds = []
    
    for d in d_values:
        ek_loss_dict = comparison[str(d)]['ek_loss']
        ek_losses.append(ek_loss_dict['loss'])
        empirical_losses.append(comparison[str(d)]['empirical_loss_mean'])
        empirical_stds.append(comparison[str(d)]['empirical_loss_std'])
    
    # Plot 1: Absolute losses
    ax1.plot(d_values, ek_losses, 'o-', linewidth=2.5, markersize=8, 
            label='EK Theory (Bias + Variance)', color='#E74C3C')
    ax1.errorbar(d_values, empirical_losses, yerr=empirical_stds, fmt='s-', 
                linewidth=2.5, markersize=8, capsize=5,
                label='Empirical Network Loss', color='#3498DB', alpha=0.7)
    
    ax1.set_xlabel('Input Dimension (d)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
    ax1.set_title('Loss: EK Theory vs Empirical Networks',
                 fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11, loc='best')
    
    # Plot 2: Relative error
    relative_error = []
    for i, d in enumerate(d_values):
        if ek_losses[i] > 0:
            rel_err = abs(empirical_losses[i] - ek_losses[i]) / ek_losses[i] * 100
            relative_error.append(rel_err)
        else:
            relative_error.append(0)
    
    ax2.bar(d_values, relative_error, color='#2ECC71', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Input Dimension (d)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Relative Error: |Empirical - EK| / EK',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Loss Comparison: N={config["N"]}, χ={config["chi"]}, κ={config["kappa_ek"]}, P=1.5d',
                fontsize=15, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: loss_comparison.png")
    plt.close()


def plot_bias_variance_decomposition(config: Dict, comparison: Dict, output_path: Path):
    """
    Plot 3: Bias-Variance decomposition.
    
    Shows how the EK loss is split between bias and variance terms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    d_values = sorted([int(d) for d in comparison.keys()])
    biases = []
    variances = []
    total_losses = []
    
    for d in d_values:
        ek_loss_dict = comparison[str(d)]['ek_loss']
        biases.append(ek_loss_dict['bias'])
        variances.append(ek_loss_dict['variance'])
        total_losses.append(ek_loss_dict['loss'])
    
    # Plot 1: Stacked bar chart
    width = 0.6
    ax1.bar(d_values, biases, width, label='Bias', color='#E74C3C', alpha=0.85)
    ax1.bar(d_values, variances, width, bottom=biases, label='Variance', color='#3498DB', alpha=0.85)
    
    ax1.set_xlabel('Input Dimension (d)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss Component', fontsize=13, fontweight='bold')
    ax1.set_title('Bias-Variance Decomposition (Stacked)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Ratio plot
    bias_ratio = [b / t * 100 if t > 0 else 0 for b, t in zip(biases, total_losses)]
    var_ratio = [v / t * 100 if t > 0 else 0 for v, t in zip(variances, total_losses)]
    
    ax2.plot(d_values, bias_ratio, 'o-', linewidth=2.5, markersize=8, 
            label='Bias %', color='#E74C3C')
    ax2.plot(d_values, var_ratio, 's-', linewidth=2.5, markersize=8, 
            label='Variance %', color='#3498DB')
    
    ax2.set_xlabel('Input Dimension (d)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Loss (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Bias-Variance Ratio',
                 fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'EK Bias-Variance Analysis: N={config["N"]}, χ={config["chi"]}, κ={config["kappa_ek"]}',
                fontsize=15, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.savefig(output_path / 'bias_variance_decomposition.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: bias_variance_decomposition.png")
    plt.close()


def plot_scaling_analysis(config: Dict, comparison: Dict, output_path: Path):
    """
    Plot 4: Scaling analysis of bias, variance, and loss with d.
    
    Looking for power-law scalings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    d_values = np.array(sorted([int(d) for d in comparison.keys()]))
    biases = []
    variances = []
    total_losses = []
    P_values = []
    
    for d in d_values:
        ek_loss_dict = comparison[str(d)]['ek_loss']
        biases.append(ek_loss_dict['bias'])
        variances.append(ek_loss_dict['variance'])
        total_losses.append(ek_loss_dict['loss'])
        P_values.append(comparison[str(d)]['P'])
    
    biases = np.array(biases)
    variances = np.array(variances)
    total_losses = np.array(total_losses)
    P_values = np.array(P_values)
    
    # Plot individual scalings on log-log scale
    plots = [
        (axes[0, 0], biases, 'Bias', '#E74C3C'),
        (axes[0, 1], variances, 'Variance', '#3498DB'),
        (axes[1, 0], total_losses, 'Total Loss', '#2ECC71'),
        (axes[1, 1], P_values, 'Sample Size P', '#F39C12'),
    ]
    
    for ax, values, label, color in plots:
        ax.loglog(d_values, values, 'o-', linewidth=2.5, markersize=8, color=color, label=label)
        
        # Fit power law on log scale
        coeffs = np.polyfit(np.log10(d_values), np.log10(values), 1)
        power_law = 10 ** (coeffs[1]) * d_values ** coeffs[0]
        ax.loglog(d_values, power_law, '--', linewidth=2, alpha=0.6, color='gray', label=f'Fit: d^{coeffs[0]:.2f}')
        
        ax.set_xlabel('Input Dimension (d)', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Scaling (log-log)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
    
    fig.suptitle(f'Scaling Analysis: N={config["N"]}, χ={config["chi"]}, κ={config["kappa_ek"]}',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: scaling_analysis.png")
    plt.close()


def create_summary_table(config: Dict, comparison: Dict, output_path: Path):
    """Create a summary table of all results."""
    d_values = sorted([int(d) for d in comparison.keys()])
    
    summary_lines = [
        "="*100,
        "EK-COMPARISON SUMMARY TABLE",
        "="*100,
        f"Configuration: N={config['N']}, χ={config['chi']}, κ={config['kappa_ek']}, " +
        f"P=1.5d (rounded), {config['num_datasets']} datasets, {config['ensemble_size']} ensembles",
        "="*100,
        f"{'d':<5} {'P':<5} {'EK Prediction':<18} {'EK Loss':<15} {'Emp. Loss':<15} {'Bias':<15} {'Variance':<15}",
        "-"*100,
    ]
    
    for d in d_values:
        data = comparison[str(d)]
        ek_loss = data['ek_loss']
        summary_lines.append(
            f"{d:<5} {data['P']:<5} {data['ek_prediction_factor']:<18.6f} "
            f"{ek_loss['loss']:<15.6e} {data['empirical_loss_mean']:<15.6e} "
            f"{ek_loss['bias']:<15.6e} {ek_loss['variance']:<15.6e}"
        )
    
    summary_lines.extend([
        "="*100,
        ""
    ])
    
    summary_text = "\n".join(summary_lines)
    
    with open(output_path / 'summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"✓ Saved: summary.txt")


def main():
    """Main plotting function."""
    print("="*70)
    print("EK-Comparison Plotting Script")
    print("="*70)
    
    results_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/data/results')
    plots_dir = Path('/home/akiva/FCNX-Ensembling/milestones/EK-comparison/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {results_dir}")
    config, comparison = load_results(results_dir)
    
    print(f"Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nGenerating plots...")
    plot_predictions(config, comparison, plots_dir)
    plot_loss_comparison(config, comparison, plots_dir)
    plot_bias_variance_decomposition(config, comparison, plots_dir)
    plot_scaling_analysis(config, comparison, plots_dir)
    
    print(f"\nGenerating summary table...")
    create_summary_table(config, comparison, plots_dir)
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
