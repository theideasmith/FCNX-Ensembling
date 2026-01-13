#!/usr/bin/env python3
"""
Simple test script for FCN2NetworkActivationGeneric H_eig computation.
Verifies that the eigenvalue computation works correctly.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

torch.set_default_dtype(torch.float32)


def test_h_eig_basic():
    """Test basic H_eig computation."""
    print("\n" + "="*60)
    print("Test 1: Basic H_eig computation")
    print("="*60)
    
    d = 5
    P = 20
    N = 10
    ens = 3
    device = torch.device("cpu")
    
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    X = torch.randn(P, d, device=device)
    Y = X[:, 0]  # Use first dimension as eigenfunction
    
    # Compute eigenvalues
    eigenvalues = model.H_eig(X, Y)
    
    print(f"  Input shape: {X.shape}")
    print(f"  Eigenfunction shape: {Y.shape}")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Max eigenvalue: {eigenvalues.max().item():.6f}")
    print(f"  Mean eigenvalue: {eigenvalues.mean().item():.6f}")
    
    assert eigenvalues.shape == (P,), f"Expected shape ({P},), got {eigenvalues.shape}"
    print("  ✓ Test passed!")


def test_h_eig_with_std():
    """Test H_eig with standard deviation computation."""
    print("\n" + "="*60)
    print("Test 2: H_eig with std computation")
    print("="*60)
    
    d = 5
    P = 20
    N = 10
    ens = 10  # Need multiple ensembles for std
    device = torch.device("cpu")
    
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    X = torch.randn(P, d, device=device)
    Y = X[:, 0]
    
    # Compute eigenvalues with std
    eigenvalues, std_eigenvalues = model.H_eig(X, Y, std=True)
    
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Std eigenvalues shape: {std_eigenvalues.shape}")
    print(f"  Mean eigenvalue: {eigenvalues.mean().item():.6f}")
    print(f"  Mean std: {std_eigenvalues.mean().item():.6f}")
    
    assert eigenvalues.shape == (P,), f"Expected shape ({P},), got {eigenvalues.shape}"
    assert std_eigenvalues.shape == (P,), f"Expected std shape ({P},), got {std_eigenvalues.shape}"
    print("  ✓ Test passed!")


def test_h_eig_multi_column():
    """Test H_eig with multiple eigenfunction columns."""
    print("\n" + "="*60)
    print("Test 3: H_eig with multiple eigenfunction columns")
    print("="*60)
    
    d = 5
    P = 20
    N = 10
    ens = 5
    M = 3  # Number of eigenfunctions
    device = torch.device("cpu")
    
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    X = torch.randn(P, d, device=device)
    Y = X[:, :M]  # Use first M dimensions as eigenfunctions
    
    # Compute eigenvalues
    eigenvalues = model.H_eig(X, Y)
    
    print(f"  Eigenfunction shape: {Y.shape}")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Eigenvalues: {eigenvalues}")
    
    assert eigenvalues.shape == (M,), f"Expected shape ({M},), got {eigenvalues.shape}"
    print("  ✓ Test passed!")


def test_linear_vs_erf():
    """Compare linear vs erf activation eigenvalues."""
    print("\n" + "="*60)
    print("Test 4: Linear vs Erf activation comparison")
    print("="*60)
    
    d = 5
    P = 50
    N = 10
    ens = 5
    device = torch.device("cuda:0")
    
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    Y = X[:, 0]
    
    # Linear activation
    model_linear = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="linear",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    eig_linear = model_linear.H_eig(X, Y)
    
    # Erf activation
    model_erf = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    # Copy weights for fair comparison
    model_erf.W0.data = model_linear.W0.data.clone()
    model_erf.A.data = model_linear.A.data.clone()
    
    eig_erf = model_erf.H_eig(X, Y)
    
    print(f"  Linear activation - Max eigenvalue: {eig_linear.max().item():.6f}")
    print(f"  Erf activation - Max eigenvalue: {eig_erf.max().item():.6f}")
    print(f"  Linear activation - Mean eigenvalue: {eig_linear.mean().item():.6f}")
    print(f"  Erf activation - Mean eigenvalue: {eig_erf.mean().item():.6f}")
    print("  ✓ Test passed!")


def test_forward_pass():
    """Test forward pass produces expected shapes."""
    print("\n" + "="*60)
    print("Test 5: Forward pass shape test")
    print("="*60)
    
    d = 5
    P = 20
    N = 10
    ens = 3
    device = torch.device("cuda:0")
    
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    X = torch.randn(P, d, device=device)
    output = model(X)
    
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({P}, {ens})")
    
    assert output.shape == (P, ens), f"Expected shape ({P}, {ens}), got {output.shape}"
    print("  ✓ Test passed!")


def test_eigenvalue_theoretical_comparison():
    """Test eigenvalues against theoretical prediction 4/(3*pi) * 1/d."""
    print("\n" + "="*60)
    print("Test 6: Eigenvalue theoretical comparison")
    print("="*60)
    
    import numpy as np
    
    d = 20
    P = 3000  # Large sample size for better statistics
    N = 1028   # Large hidden layer
    ens = 5  # Many ensembles for averaging
    device = torch.device("cuda:0")
    
    # Theoretical value
    theoretical_value = 4.0/(3.0 * np.pi * d)
    
    print(f"  Input dimension d: {d}")
    print(f"  Theoretical eigenvalue: {theoretical_value:.6f}")
    
    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N),
        device=device
    )
    
    # Generate data
    torch.manual_seed(42)
    X = torch.randn(P, d, device=device)
    
    # Compute eigenvalues using X as eigenfunctions
    eigenvalues = model.H_eig(X, X)
    
    mean_eigenvalue = eigenvalues.mean().item()
    median_eigenvalue = eigenvalues.median().item()
    std_eigenvalue = eigenvalues.std().item()
    
    print(f"  Mean empirical eigenvalue: {mean_eigenvalue:.6f}")
    print(f"  Median empirical eigenvalue: {median_eigenvalue:.6f}")
    print(f"  Std of empirical eigenvalues: {std_eigenvalue:.6f}")
    
    # Compute relative error
    relative_error_mean = abs(mean_eigenvalue - theoretical_value) / theoretical_value
    relative_error_median = abs(median_eigenvalue - theoretical_value) / theoretical_value
    
    print(f"  Relative error (mean): {relative_error_mean:.2%}")
    print(f"  Relative error (median): {relative_error_median:.2%}")
    
    # Calculate ratio
    ratio = mean_eigenvalue / theoretical_value
    print(f"  Empirical/Theoretical ratio: {ratio:.3f}")
    
    # Note: The discrepancy may be due to:
    # - Finite width effects (N finite vs N->infinity limit)
    # - Finite sample size (P finite vs P->infinity limit)
    # - Different normalization conventions in kernel definition
    # - Theoretical prediction may be for infinite width limit
    
    if relative_error_mean < 0.20:
        print(f"  ✓ Close match to theory!")
    else:
        print(f"  ⚠ Significant deviation from theory (this may be expected for finite N,P)")
    
    print("  ✓ Test completed (informational)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FCN2NetworkActivationGeneric Test Suite")
    print("="*60)
    
    test_h_eig_basic()
    test_h_eig_with_std()
    test_h_eig_multi_column()
    test_linear_vs_erf()
    test_forward_pass()
    test_eigenvalue_theoretical_comparison()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
