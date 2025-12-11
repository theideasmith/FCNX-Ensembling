"""
Compute exact GPR on same parameters as EK-comparison networks and compare with EK loss.
Uses the NNGP kernel (dot-product) and compares test loss with EK predictions.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch
from typing import Dict
from tqdm import tqdm

sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from GPKit import gpr_dot_product_explicit, gpr_dot_product_variance
from ramanujan_seeds import get_ramanujan_partition_seeds


def generate_synthetic_data(d: int, P: int, seed: int, device='cpu'):
    """Generate synthetic regression data with train/test split (80/20)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(P, d, device=device)
    y = X[:, 0].unsqueeze(-1)
    
    # 80/20 train/test split
    split_idx = int(0.8 * P)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def generate_large_test_set(d: int, P_test: int, seed: int = 999999, device='cpu'):
    """Generate a large test dataset for bias computation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_test = torch.randn(P_test, d, device=device)
    y_test = X_test[:, 0].unsqueeze(-1)  # True target function
    
    return X_test, y_test


def compute_gpr_bias_and_variance(datasets, d, P, sigma_0_sq, P_test=100, device='cpu'):
    """
    Compute GPR bias and variance following theoretical definition.
    
    BIAS: 
    1. Generate large test set
    2. For each dataset, compute GP predictions on test set
    3. Average predictions across datasets
    4. Compute MSE between averaged predictor and true target function
    
    VARIANCE:
    1. For each dataset, compute variance diagonal: K(X_test, X_train) @ K_inv @ K(X_train, X_test)
    2. Average variance diagonals across datasets
    3. Take mean of the averaged diagonal
    
    Args:
        datasets: List of (X_train, y_train) tuples
        d: Input dimension
        P: Training set size per dataset
        sigma_0_sq: Kernel hyperparameter (kappa)
        P_test: Size of test set for bias computation
        device: Computation device
    
    Returns:
        Dict with 'bias', 'variance', and 'loss' (bias + variance)
    """
    # Generate large test set for bias computation
    X_test_large, y_test_large = generate_large_test_set(d, P_test, device=device)
    
    # Collect predictions on test set from each dataset
    predictions_on_test = []
    variance_diagonals = []
    
    for X_train, y_train in datasets:
        # Compute GP prediction on test set
        try:
            y_pred_test = gpr_dot_product_explicit(X_train, y_train, X_test_large, sigma_0_sq)
            predictions_on_test.append(y_pred_test)
            
        except Exception as e:
            print(f"Warning: GPR computation failed for a dataset: {e}")
            continue
    
    if not predictions_on_test:
        return {'bias': np.nan, 'variance': np.nan, 'loss': np.nan}

    # BIAS: Average predictions across datasets, then compute MSE vs true targets
    predictions_stacked = torch.stack(predictions_on_test, dim=0) # (num_datasets, P_test, 1)
    bias = ((predictions_stacked.mean(dim=0) - y_test_large)**2).mean().item()  # (P_test, 1)

    
    # # VARIANCE: Average variance diagonals across datasets, then take mean
    var_empirical = torch.mean(torch.var(predictions_stacked, dim=0)).item()

    return {
        'bias': bias,
        'variance': var_empirical,
        'loss': bias + var_empirical
    }


def compute_ek_loss(d: int, P: int, kappa: float = 0.5, chi: float = 1.0) -> Dict[str, float]:
    """
    Compute EK predicted loss = bias + variance.
    
    With lH = 1/d (NNGP)
    """
    lH = 1.0 / d
    
    bias = ((kappa/P) / (lH + kappa/P)) ** 2
    variance = (kappa / (P)) * (lH / (lH + kappa/P))
    loss = (bias) +  (variance) 
    
    return {
        'bias': bias,
        'variance': variance,
        'loss': loss,
        'lH': lH
    }


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration matching ek_comparison.py
    chi = 1.0
    N = 256
    P_factor = 5
    num_datasets = 50
    kappa_ek = 1.0
    d_range = list(range(2, 80, 5))
    
    seeds = get_ramanujan_partition_seeds(10)
    
    # Create output directories
    base_dir = Path(__file__).resolve().parent
    res_dir = base_dir / "data" / "results_gpr"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("\nComputing GPR losses and comparing with EK predictions...\n")
    
    for d in d_range:
        P = int(np.round(P_factor * d))
        kappa_ek = 2.0 * d
        ek_loss_dict = compute_ek_loss(d, P, kappa=kappa_ek, chi=chi)
        ek_loss = ek_loss_dict['loss']
        
        # Collect all datasets for this dimension
        datasets = []
        dataset_seeds = []
        x0_train_first_dataset = []
        gpr_predictions = []
        
        print(f"\nGenerating {num_datasets} datasets for d={d}, P={P}...")
        with tqdm(total=num_datasets, desc=f"Datasets for d={d}") as pbar:
            for dataset_idx in range(num_datasets):
                seed = seeds[dataset_idx % len(seeds)] + dataset_idx * 1000
                dataset_seeds.append(seed)
                X_train, X_test, y_train, y_test = generate_synthetic_data(d, P, seed, device)
                
                datasets.append((X_train, y_train))
                
                # Save first dataset info for plotting
                if dataset_idx == 0:
                    x0_train_first_dataset = X_train[:, 0].cpu().numpy().flatten().tolist()
                    try:
                        y_pred_gpr = gpr_dot_product_explicit(X_train, y_train, X_train, kappa_ek)
                        gpr_predictions = y_pred_gpr.cpu().numpy().flatten().tolist()
                    except Exception:
                        gpr_predictions = []
                
                pbar.update(1)
        
        # Compute GPR bias and variance using all datasets
        print(f"Computing GPR bias and variance for d={d}...")
        gpr_result = compute_gpr_bias_and_variance(datasets, d, P, kappa_ek, P_test=1000, device=device)
        gpr_bias = gpr_result['bias']
        gpr_variance = gpr_result['variance']
        gpr_loss = gpr_result['loss']
        
        print(f"d={d:>2}, P={P:>2}: EK loss={ek_loss:.6e}, GPR loss={gpr_loss:.6e}")
        print(f"  EK:  Bias={ek_loss_dict['bias']:.6e}, Variance={ek_loss_dict['variance']:.6e}")
        print(f"  GPR: Bias={gpr_bias:.6e}, Variance={gpr_variance:.6e}")
        
        results[d] = {
            'P': P,
            'ek_loss': ek_loss,
            'ek_bias': ek_loss_dict['bias'],
            'ek_variance': ek_loss_dict['variance'],
            'gpr_loss': float(gpr_loss),
            'gpr_bias': float(gpr_bias),
            'gpr_variance': float(gpr_variance),
            'num_gpr_samples': len(datasets),
            'gpr_predictions': gpr_predictions,
            'dataset_seeds': dataset_seeds,
            'x0_train_first_dataset': x0_train_first_dataset,
            'ek_prediction_factor': (1.0 / d) / (1.0 / d + kappa_ek / P),
        }
    
    # Save results
    out_path = res_dir / "gpr_vs_ek_loss.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGPR vs EK comparison saved to: {out_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary: GPR loss vs EK loss prediction")
    print(f"{'='*70}")
    for d in d_range:
        ek = results[d]['ek_loss']
        gpr = results[d]['gpr_loss']
        ratio = gpr / ek if ek > 0 else np.nan
        print(f"d={d:>2}: EK={ek:.6e}, GPR={gpr:.6e}, Ratio={ratio:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
