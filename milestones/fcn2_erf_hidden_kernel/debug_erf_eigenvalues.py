#!/usr/bin/env python3
"""
Debug script to understand erf FCN2 eigenvalues.
Diagnose the arcsin kernel (from GPR) and compare with network preactivations.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

torch.set_default_dtype(torch.float32)


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X)
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    return (2 / torch.pi) * torch.arcsin(arg)


def test_erf_eigenvalues():
    """Test erf network eigenvalues against GPR arcsin kernel."""
    print("\nErf FCN2 Eigenvalue Analysis")
    print("="*60)
    
    d = 50
    P = 80
    N = 1028
    ens = 20
    device = torch.device("cuda:0")
    
    print(f"Input dimension d: {d}")
    print(f"Samples P: {P}")
    print(f"Hidden width N: {N}")
    print(f"Ensembles: {ens}")
    print()
    
    # Create erf model
    torch.manual_seed(42)
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    # Generate standard Gaussian data
    X = torch.randn(3000, d, device=device)
    
    # Compute eigenvalues using X as eigenfunctions
    eigenvalues = model.H_eig(X, X)
    
    print("FCN2 H_eig Results (using X as eigenfunctions):")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Mean empirical eigenvalue: {eigenvalues.mean().item():.6e}")
    print(f"  Median empirical eigenvalue: {eigenvalues.median().item():.6e}")
    print(f"  Std of empirical eigenvalues: {eigenvalues.std().item():.6e}")
    print(f"  Min eigenvalue: {eigenvalues.min().item():.6e}")
    print(f"  Max eigenvalue: {eigenvalues.max().item():.6e}")
    print()
    
    # Compute arcsin kernel eigenvalues for comparison
    print("Arcsin Kernel (GPR) Analysis:")
    with torch.no_grad():
        K_arcsin = arcsin_kernel(X)  # (P, P)
        K_arcsin_eigs = (torch.linalg.eigvalsh(K_arcsin) / P)[:d]
        print(f"  Arcsin kernel eig shape: {K_arcsin_eigs.shape}")
        print(f"  Mean eigenvalue: {K_arcsin_eigs.mean().item():.6e}")
        print(f"  Max eigenvalue: {K_arcsin_eigs.max().item():.6e}")
        print(f"  Min eigenvalue: {K_arcsin_eigs.min().item():.6e}")
        print(f"  Trace (sum of eigs): {K_arcsin_eigs.sum().item():.6e}")
        print()
    
    # Manual h0 (preactivation) kernel computation
    print("Manual Preactivation (h0) Kernel Computation:")
    with torch.no_grad():
        h0 = model.h0_preactivation(X)  # (P, ens, N)
        print(f"  h0 shape: {h0.shape}")
        print(f"  h0 mean: {h0.mean().item():.6e}")
        print(f"  h0 std: {h0.std().item():.6e}")
        print(f"  h0 max: {h0.max().item():.6e}")
        print(f"  h0 min: {h0.min().item():.6e}")
        
        # Kernel per ensemble: K_q[u,v] = (1/N) * sum_k h0[u,q,k] * h0[v,q,k]
        K_per_ens = torch.einsum('uqk,vqk->quv', h0, h0) / N  # (ens, P, P)
        K_h0 = K_per_ens.mean(dim=0)  # Average over ensembles: (P, P)
        
        print(f"  K_h0 shape: {K_h0.shape}")
        print(f"  K_h0 diag mean: {torch.diagonal(K_h0).mean().item():.6e}")
        print(f"  K_h0 diag std: {torch.diagonal(K_h0).std().item():.6e}")
        
        # Eigenvalues of preactivation kernel
        K_h0_eigs = torch.linalg.eigvalsh(K_h0)
        print(f"  K_h0 mean eigenvalue: {K_h0_eigs.mean().item():.6e}")
        print(f"  K_h0 max eigenvalue: {K_h0_eigs.max().item():.6e}")
        print(f"  K_h0 trace: {K_h0_eigs.sum().item():.6e}")
        print()
    
    # Manual erf activation kernel
    print("Manual Activation (h0_erf) Kernel Computation:")
    with torch.no_grad():
        h0_erf = torch.erf(h0)  # (P, ens, N)
        print(f"  h0_erf shape: {h0_erf.shape}")
        print(f"  h0_erf mean: {h0_erf.mean().item():.6e}")
        print(f"  h0_erf std: {h0_erf.std().item():.6e}")
        
        # Kernel per ensemble from erf activations
        K_erf_per_ens = torch.einsum('uqk,vqk->quv', h0_erf, h0_erf) / N  # (ens, P, P)
        K_erf = K_erf_per_ens.mean(dim=0)  # Average over ensembles: (P, P)
        
        print(f"  K_erf shape: {K_erf.shape}")
        print(f"  K_erf diag mean: {torch.diagonal(K_erf).mean().item():.6e}")
        print(f"  K_erf diag std: {torch.diagonal(K_erf).std().item():.6e}")
        
        # Eigenvalues of erf kernel
        K_erf_eigs = torch.linalg.eigvalsh(K_erf)
        print(f"  K_erf mean eigenvalue: {K_erf_eigs.mean().item():.6e}")
        print(f"  K_erf max eigenvalue: {K_erf_eigs.max().item():.6e}")
        print(f"  K_erf trace: {K_erf_eigs.sum().item():.6e}")
        print()
    
    # Compare with Rayleigh quotients on X
    print("Rayleigh Quotient Analysis (using X as eigenfunctions):")
    with torch.no_grad():
        # Rayleigh quotient with K_erf
        rayleigh_erf = []
        for j in range(d):
            x_j = X[:, j]  # (P,)
            K_erf_x = K_erf @ x_j  # (P,)
            numerator = (x_j * K_erf_x).sum()
            denominator = (x_j * x_j).sum()
            rayleigh_erf.append((numerator / denominator).item())
        
        rayleigh_erf = torch.tensor(rayleigh_erf)
        print(f"  Rayleigh quotient (K_erf) mean: {rayleigh_erf.mean().item():.6e}")
        print(f"  Rayleigh quotient (K_erf) std: {rayleigh_erf.std().item():.6e}")
        print()
        
        # Rayleigh quotient with arcsin kernel
        rayleigh_arcsin = []
        for j in range(d):
            x_j = X[:, j]  # (P,)
            K_arcsin_x = K_arcsin @ x_j  # (P,)
            numerator = (x_j * K_arcsin_x).sum()
            denominator = (x_j * x_j).sum()
            rayleigh_arcsin.append((numerator / denominator).item())
        
        rayleigh_arcsin = torch.tensor(rayleigh_arcsin)
        print(f"  Rayleigh quotient (K_arcsin) mean: {rayleigh_arcsin.mean().item():.6e}")
        print(f"  Rayleigh quotient (K_arcsin) std: {rayleigh_arcsin.std().item():.6e}")
        print()
    
    # Compare H_eig with the various kernels
    print("Summary Comparison:")
    print(f"  H_eig mean:                        {eigenvalues.mean().item():.6e}")
    print(f"  Rayleigh (K_erf) mean:             {rayleigh_erf.mean().item():.6e}")
    print(f"  Rayleigh (K_arcsin) mean:          {rayleigh_arcsin.mean().item():.6e}")
    print()
    print(f"  H_eig * P mean:                    {(eigenvalues * P).mean().item():.6e}")
    print(f"  Rayleigh (K_erf) / P mean:         {(rayleigh_erf / P).mean().item():.6e}")
    print(f"  Rayleigh (K_arcsin) / P mean:      {(rayleigh_arcsin / P).mean().item():.6e}")
    print()
    
  


if __name__ == "__main__":
    test_erf_eigenvalues()
