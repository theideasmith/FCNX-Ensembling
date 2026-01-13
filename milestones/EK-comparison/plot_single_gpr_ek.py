"""
Generate a single dataset and plot GPR vs EK predictions with respect to x[0].
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from GPKit import gpr_dot_product_explicit


def generate_dataset(d: int, P: int, seed: int = 42, device='cpu'):
    """Generate a single synthetic dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(P, d, device=device)
    y = X[:, 0].unsqueeze(-1)  # Target is first component
    
    return X, y


def compute_gpr_predictions(X, y, kappa: float = 1.0):
    """Compute GPR predictions using dot-product kernel."""
    try:
        y_pred = gpr_dot_product_explicit(X, y, X, kappa)
        return y_pred.cpu().numpy().flatten()
    except Exception as e:
        print(f"GPR computation failed: {e}")
        return None


def compute_ek_predictions(X, d: int, kappa: float = 1.0):
    """Compute EK predictions: (1/d) / (1/d + kappa) * x[0]."""
    ek_factor = (1.0 / d) / (1.0 / d + kappa / X.shape[0])
    x0 = X[:, 0].cpu().numpy()
    return ek_factor * x0


def plot_predictions_vs_x0(x0, y_true, y_gpr, y_ek, d, P, kappa, output_dir):
    """Plot GPR and EK predictions vs x[0]."""
    # Sort by x0 for cleaner line plots
    sort_idx = np.argsort(x0)
    x0_sorted = x0[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_gpr_sorted = y_gpr[sort_idx]
    y_ek_sorted = y_ek[sort_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: All predictions vs x[0]
    ax1 = axes[0]
    ax1.scatter(x0, y_true, alpha=0.3, s=20, label='True Target (x₀)', color='black')
    ax1.plot(x0_sorted, y_gpr_sorted, '-', linewidth=2, label='GPR Prediction', color='red', alpha=0.8)
    ax1.plot(x0_sorted, y_ek_sorted, '--', linewidth=2, label='EK Prediction', color='blue', alpha=0.8)
    
    ax1.set_xlabel('x₀ (First Input Component)', fontsize=12)
    ax1.set_ylabel('Prediction', fontsize=12)
    ax1.set_title(f'Predictions vs x₀\n(d={d}, P={P}, κ={kappa})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: GPR vs EK directly
    ax2 = axes[1]
    ax2.scatter(y_ek, y_gpr, alpha=0.6, s=30, color='green')
    
    # Perfect match line
    min_val = min(y_ek.min(), y_gpr.min())
    max_val = max(y_ek.max(), y_gpr.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    
    ax2.set_xlabel('EK Prediction', fontsize=12)
    ax2.set_ylabel('GPR Prediction', fontsize=12)
    ax2.set_title('GPR vs EK Predictions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Compute correlation
    corr = np.corrcoef(y_ek, y_gpr)[0, 1]
    mse = np.mean((y_gpr - y_ek) ** 2)
    ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}\nMSE: {mse:.4e}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / f'gpr_vs_ek_single_dataset_d{d}_P{P}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return corr, mse


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    d = 10
    P = 50
    kappa = 1.0
    seed = 42
    
    print(f"\nGenerating dataset: d={d}, P={P}, κ={kappa}, seed={seed}")
    
    # Generate data
    X, y = generate_dataset(d, P, seed, device)
    x0 = X[:, 0].cpu().numpy()
    y_true = y.cpu().numpy().flatten()
    
    # Compute predictions
    print("Computing GPR predictions...")
    y_gpr = compute_gpr_predictions(X, y, kappa)
    
    print("Computing EK predictions...")
    y_ek = compute_ek_predictions(X, d, kappa)
    
    if y_gpr is None:
        print("Failed to compute GPR predictions")
        return
    
    # Create output directory
    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots_gpr"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    print("Generating plots...")
    corr, mse = plot_predictions_vs_x0(x0, y_true, y_gpr, y_ek, d, P, kappa, plots_dir)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Dimension (d): {d}")
    print(f"Sample Size (P): {P}")
    print(f"Kappa (κ): {kappa}")
    print(f"EK Factor: (1/d)/(1/d + κ) = {(1/d)/(1/d + kappa):.6f}")
    print(f"Correlation (GPR vs EK): {corr:.6f}")
    print(f"MSE (GPR vs EK): {mse:.6e}")
    print(f"\nGPR mean prediction: {y_gpr.mean():.6f} ± {y_gpr.std():.6f}")
    print(f"EK mean prediction:  {y_ek.mean():.6f} ± {y_ek.std():.6f}")
    print(f"True mean: {y_true.mean():.6f} ± {y_true.std():.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
