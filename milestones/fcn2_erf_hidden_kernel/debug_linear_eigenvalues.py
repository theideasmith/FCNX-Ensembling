#!/usr/bin/env python3
"""
Debug script to understand linear FCN2 eigenvalues.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

torch.set_default_dtype(torch.float32)

def test_linear_eigenvalues():
    """Test linear network eigenvalues against theoretical 1/d."""
    print("\nLinear FCN2 Eigenvalue Analysis")
    print("="*60)
    
    d = 10
    P = 500
    N = 200
    ens = 20
    device = torch.device("cpu")
    
    # Theoretical value for linear network
    theoretical_value = 1.0 / d
    
    print(f"Input dimension d: {d}")
    print(f"Samples P: {P}")
    print(f"Hidden width N: {N}")
    print(f"Ensembles: {ens}")
    print(f"Theoretical eigenvalue (1/d): {theoretical_value:.6f}")
    print()
    
    # Create linear model
    torch.manual_seed(42)
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="linear",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    # Generate standard Gaussian data
    X = torch.randn(P, d, device=device)
    
    # Compute eigenvalues using X as eigenfunctions
    eigenvalues = model.H_eig(X, X)
    
    print("Results:")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Mean empirical eigenvalue: {eigenvalues.mean().item():.6f}")
    print(f"  Median empirical eigenvalue: {eigenvalues.median().item():.6f}")
    print(f"  Std of empirical eigenvalues: {eigenvalues.std().item():.6f}")
    print(f"  Min eigenvalue: {eigenvalues.min().item():.6f}")
    print(f"  Max eigenvalue: {eigenvalues.max().item():.6f}")
    print()
    
    ratio = eigenvalues.mean().item() / theoretical_value
    print(f"  Empirical/Theoretical ratio: {ratio:.3f}")
    print(f"  Relative error: {abs(ratio - 1.0):.2%}")
    print()
    
    # Let's manually compute what the kernel should be
    print("Manual kernel computation:")
    with torch.no_grad():
        # Compute h0 = W0 @ X for each ensemble member
        h0 = model.h0_preactivation(X)  # (P, ens, N)
        print(f"  h0 shape: {h0.shape}")
        print(f"  Mean h0^2: {(h0**2).mean().item():.6f}")
        
        # Kernel per ensemble: K_q[u,v] = (1/N) * sum_k h0[u,q,k] * h0[v,q,k]
        K_per_ens = torch.einsum('uqk,vqk->quv', h0, h0) / N  # (ens, P, P)
        K = K_per_ens.mean(dim=0)  # Average over ensembles: (P, P)
        
        print(f"  K shape: {K.shape}")
        print(f"  Mean diagonal K[i,i]: {torch.diagonal(K).mean().item():.6f}")
        
        # Compute eigenvalues of K
        K_eigenvals = torch.linalg.eigvalsh(K)
        print(f"  Mean eigenvalue of K: {K_eigenvals.mean().item():.6f}")
        print(f"  Max eigenvalue of K: {K_eigenvals.max().item():.6f}")
        
        # Now compute Rayleigh quotients with X as eigenfunctions
        # For each column j of X: lambda_j = X[:,j]^T @ K @ X[:,j] / ||X[:,j]||^2
        manual_eigenvals = []
        for j in range(d):
            x_j = X[:, j]  # (P,)
            Kx_j = K @ x_j  # (P,)
            numerator = (x_j * Kx_j).sum()
            denominator = (x_j * x_j).sum()
            eig_j = numerator / denominator
            manual_eigenvals.append(eig_j.item() / P) 
        
        manual_eigenvals = torch.tensor(manual_eigenvals)
        print(f"\nManual Rayleigh quotient computation:")
        print(f"  Mean: {manual_eigenvals.mean().item():.6f}")
        print(f"  Matches H_eig: {torch.allclose(manual_eigenvals, eigenvalues, rtol=1e-4)}")        
        # Let's also check what normalization gives 1/d
        print(f"\nInvestigating normalization:")
        # If we normalize by P^2
        normalized_by_P2 = manual_eigenvals / (P * P)
        print(f"  Manual / P^2: {normalized_by_P2.mean().item():.6f}")
        # If we normalize by P  
        normalized_by_P = manual_eigenvals / P
        print(f"  Manual / P: {normalized_by_P.mean().item():.6f}")
        # Check H_eig * P^2
        heig_times_P2 = eigenvalues * (P * P)
        print(f"  H_eig * P^2: {heig_times_P2.mean().item():.6f}")
        # Check H_eig * P
        heig_times_P = eigenvalues * P
        print(f"  H_eig * P: {heig_times_P.mean().item():.6f}")
if __name__ == "__main__":
    test_linear_eigenvalues()
