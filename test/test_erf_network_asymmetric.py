#!/usr/bin/env python3
"""
Test FCN3NetworkEnsembleErf with asymmetric hidden layer widths (N1=40, N2=50).
"""

import sys
from pathlib import Path
import torch
import pytest

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
from FCN3Network import FCN3NetworkEnsembleErf


def test_create_erf_network_asymmetric():
    """Test creating an erf network with N1=40 and N2=50."""
    d = 10
    N1 = 40
    N2 = 50
    P = 30
    ens = 5
    
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=N1,
        n2=N2,
        P=P,
        ens=ens,
        weight_initialization_variance=(1/d, 1/N1, 1/N2)
    )
    
    # Check that model is created
    assert model is not None
    assert model.d == d
    assert model.num_samples == P
    assert model.n1 == N1
    assert model.n2 == N2
    assert model.ens == ens


def test_forward_erf_network_asymmetric():
    """Test forward pass with asymmetric hidden layers."""
    d = 10
    N1 = 40
    N2 = 50
    P = 30
    ens = 5
    
    device = torch.device("cpu")
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=N1,
        n2=N2,
        P=P,
        ens=ens,
        weight_initialization_variance=(1/d, 1/N1, 1/N2),
        device=device
    ).to(device)
    
    # Create random input
    X = torch.randn(P, d, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model.forward(X)
    
    # Check output shape: (P, ens)
    assert output.shape == (P, ens), f"Expected shape ({P}, {ens}), got {output.shape}"
    
    # Check that output is finite
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    # Check output values are reasonable (erf activation bounds output)
    # With random init, outputs should be relatively small
    assert output.abs().max() < 100, "Output values unexpectedly large"


def test_forward_single_sample():
    """Test forward pass with a single sample."""
    d = 10
    N1 = 40
    N2 = 50
    ens = 3
    
    device = torch.device("cpu")
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=N1,
        n2=N2,
        P=1,  # Single sample
        ens=ens,
        weight_initialization_variance=(1/d, 1/N1, 1/N2),
        device=device
    ).to(device)
    
    # Single input sample
    X = torch.randn(1, d, device=device)
    
    with torch.no_grad():
        output = model.forward(X)
    
    assert output.shape == (1, ens)
    assert torch.isfinite(output).all()


def test_weight_shapes():
    """Verify weight tensor shapes for asymmetric network."""
    d = 10
    N1 = 40
    N2 = 50
    P = 30
    ens = 5
    
    model = FCN3NetworkEnsembleErf(d=d, n1=N1, n2=N2, P=P, ens=ens)
    
    # Check W0: (ens, N1, d)
    assert model.W0.shape == (ens, N1, d), f"W0 shape mismatch: {model.W0.shape}"
    
    # Check W1: (ens, N2, N1)
    assert model.W1.shape == (ens, N2, N1), f"W1 shape mismatch: {model.W1.shape}"
    
    # Check A: (ens, N2)
    assert model.A.shape == (ens, N2), f"A shape mismatch: {model.A.shape}"


def test_mean_field_weight_initialization():
    """Test mean field scaling weight initialization with chi parameter."""
    d = 10
    N1 = 40
    N2 = 50
    P = 30
    ens = 100  # Large ensemble for better statistics
    chi = 2.0  # Test with non-unity chi
    
    # Mean field scaling: weight variances are 1/(fan_in * fan_out) with chi for output layer
    sigma_w0_sq = 1.0 / (d)
    sigma_w1_sq = 1.0 / (N1)
    sigma_a_sq = 1.0 / (N2 * chi)
    
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=N1,
        n2=N2,
        P=P,
        ens=ens,
        weight_initialization_variance=(sigma_w0_sq, sigma_w1_sq, sigma_a_sq)
    )
    

    
    # Check actual weight variances (with tolerance for finite sampling)
    with torch.no_grad():
        # W0 variance
        w0_var = model.W0.var().item()
        expected_w0_var = sigma_w0_sq
        rel_error_w0 = abs(w0_var - expected_w0_var) / expected_w0_var
        assert rel_error_w0 < 0.15, f"W0 variance mismatch: expected {expected_w0_var:.6f}, got {w0_var:.6f} (rel error {rel_error_w0:.2%})"
        
        # W1 variance
        w1_var = model.W1.var().item()
        expected_w1_var = sigma_w1_sq
        rel_error_w1 = abs(w1_var - expected_w1_var) / expected_w1_var
        assert rel_error_w1 < 0.15, f"W1 variance mismatch: expected {expected_w1_var:.6f}, got {w1_var:.6f} (rel error {rel_error_w1:.2%})"
        
        # A variance (most important for chi test)
        a_var = model.A.var().item()
        expected_a_var = sigma_a_sq
        rel_error_a = abs(a_var - expected_a_var) / expected_a_var
        assert rel_error_a < 0.15, f"A variance mismatch: expected {expected_a_var:.6f}, got {a_var:.6f} (rel error {rel_error_a:.2%})"
        
        print(f"  W0 variance: {w0_var:.6f} (expected {expected_w0_var:.6f}, error {rel_error_w0:.2%})")
        print(f"  W1 variance: {w1_var:.6f} (expected {expected_w1_var:.6f}, error {rel_error_w1:.2%})")
        print(f"  A variance: {a_var:.6f} (expected {expected_a_var:.6f}, error {rel_error_a:.2%})")
        
        # Check that weights are approximately normally distributed (check mean close to 0)
        w0_mean = model.W0.mean().item()
        w1_mean = model.W1.mean().item()
        a_mean = model.A.mean().item()
        
        assert abs(w0_mean) < 0.01, f"W0 mean should be ~0, got {w0_mean}"
        assert abs(w1_mean) < 0.01, f"W1 mean should be ~0, got {w1_mean}"
        assert abs(a_mean) < 0.01, f"A mean should be ~0, got {a_mean}"


def test_mean_field_with_different_chi_values():
    """Test that different chi values produce correct A weight variances."""
    d = 10
    N1 = 40
    N2 = 50
    P = 30
    ens = 100
    
    chi_values = [0.5, 1.0, 2.0, 5.0]
    
    for chi in chi_values:
        sigma_a_sq = 1.0 / (N2 * chi)
        
        model = FCN3NetworkEnsembleErf(
            d=d,
            n1=N1,
            n2=N2,
            P=P,
            ens=ens,
            weight_initialization_variance=(1.0/(d), 1.0/(N1), sigma_a_sq)
        )
        
        # Verify chi is set
        
        # Check A variance scales inversely with chi
        with torch.no_grad():
            a_var = model.A.var().item()
            expected_var = sigma_a_sq
            rel_error = abs(a_var - expected_var) / expected_var
            
            assert rel_error < 0.15, \
                f"For chi={chi}: A variance {a_var:.6f} doesn't match expected {expected_var:.6f} (error {rel_error:.2%})"
            
            print(f"  chi={chi}: A var={a_var:.6f}, expected={expected_var:.6f}, error={rel_error:.2%}")


def test_gradients_per_ensemble_are_isolated():
    """Gradients from one ensemble member do not leak to others."""
    torch.manual_seed(0)
    d, n1, n2, P, ens = 4, 6, 5, 3, 3
    model = FCN3NetworkEnsembleErf(
        d=d,
        n1=n1,
        n2=n2,
        P=P,
        ens=ens,
        weight_initialization_variance=(1.0 / d, 1.0 / n1, 1.0 / n2),
    )

    X = torch.randn(P, d)
    model.zero_grad()

    output = model(X)  # (P, ens)
    mask = torch.zeros_like(output)
    mask[:, 0] = 1.0  # isolate ensemble 0
    loss = ((output * mask) ** 2).sum()

    loss.backward()

    # Ensemble 0 should have nonzero gradients; others should be (near) zero
    assert torch.any(model.W0.grad[0].abs() > 0)
    assert torch.any(model.W1.grad[0].abs() > 0)
    assert torch.any(model.A.grad[0].abs() > 0)

    zero = torch.zeros_like
    assert torch.allclose(model.W0.grad[1], zero(model.W0.grad[1]), atol=1e-9)
    assert torch.allclose(model.W0.grad[2], zero(model.W0.grad[2]), atol=1e-9)
    assert torch.allclose(model.W1.grad[1], zero(model.W1.grad[1]), atol=1e-9)
    assert torch.allclose(model.W1.grad[2], zero(model.W1.grad[2]), atol=1e-9)
    assert torch.allclose(model.A.grad[1], zero(model.A.grad[1]), atol=1e-9)
    assert torch.allclose(model.A.grad[2], zero(model.A.grad[2]), atol=1e-9)


if __name__ == "__main__":
    print("Running tests for asymmetric erf network (N1=40, N2=50)...")
    test_create_erf_network_asymmetric()
    print("✓ Creation test passed")
    
    test_forward_erf_network_asymmetric()
    print("✓ Forward pass test passed")
    
    test_forward_single_sample()
    print("✓ Single sample test passed")
    
    test_weight_shapes()
    print("✓ Weight shape test passed")
    
    print("\nTesting mean field weight initialization...")
    test_mean_field_weight_initialization()
    print("✓ Mean field initialization test passed")
    
    print("\nTesting different chi values...")
    test_mean_field_with_different_chi_values()
    print("✓ Chi scaling test passed")
    
    print("\nAll tests passed!")
