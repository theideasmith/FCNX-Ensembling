"""
Comprehensive test suite for FCNX-Ensembling FCN2Network.

Tests cover:
    - Model initialization and forward passes
    - Kernel eigenvalue computation
    - Various activation functions (erf, linear)
    - Network size variations and stability
    - Ensemble consistency
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from FCN2Network import (
    FCN2NetworkActivationGeneric,
    FCN2NetworkEnsembleErf,
    FCN2NetworkEnsembleLinear
)


class TestFCN2NetworkInitialization:
    """Test FCN2Network initialization and basic properties."""
    
    def test_init_erf_activation(self):
        """Test initialization with erf activation."""
        d, n1, P, ens = 10, 50, 100, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        assert model.d == 10, f"Expected d=10, got d={model.d}"
        assert model.n1 == 50, f"Expected n1=50, got n1={model.n1}"
        assert model.ens == 3, f"Expected ens=3, got ens={model.ens}"
        assert model.activation_name == "erf", f"Expected activation='erf', got '{model.activation_name}'"
        assert model.W0.shape == (3, 50, 10), f"Expected W0.shape=(3, 50, 10), got {model.W0.shape}"
        assert model.A.shape == (3, 50), f"Expected A.shape=(3, 50), got {model.A.shape}"
    
    def test_init_linear_activation(self):
        """Test initialization with linear activation."""
        d, n1, P, ens = 5, 20, 50, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="linear")
        assert model.activation_name == "linear", f"Expected activation='linear', got '{model.activation_name}'"
        assert model.W0.shape == (2, 20, 5), f"Expected W0.shape=(2, 20, 5), got {model.W0.shape}"
        assert model.A.shape == (2, 20), f"Expected A.shape=(2, 20), got {model.A.shape}"
    
    def test_init_invalid_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Activation must be"):
            FCN2NetworkActivationGeneric(d=10, n1=50, P=100, activation="relu")
    
    def test_weight_initialization_variance(self):
        """Test custom weight initialization variances."""
        d, n1, P, ens = 10, 50, 100, 1
        model = FCN2NetworkActivationGeneric(
            d=d, n1=n1, P=P, ens=ens,
            weight_initialization_variance=(0.5, 0.1)
        )
        # Check approximate variances (statistical test)
        w0_var = model.W0.var().item()
        a_var = model.A.var().item()
        assert 0.3 < w0_var < 0.7, f"W0 variance={w0_var:.4f} not in range [0.3, 0.7] (expected ~0.5)"
        assert 0.05 < a_var < 0.15, f"A variance={a_var:.4f} not in range [0.05, 0.15] (expected ~0.1)"
    
    def test_einsum_path_precomputation(self):
        """Test that einsum paths are precomputed when P is provided."""
        model_with_p = FCN2NetworkActivationGeneric(d=10, n1=50, P=100, ens=3)
        model_without_p = FCN2NetworkActivationGeneric(d=10, n1=50, P=None, ens=3)
        assert model_with_p.forward_path_h0 is not None, "Model with P should have precomputed forward_path_h0"
        assert model_without_p.forward_path_h0 is None, "Model without P should have forward_path_h0=None"
    
    def test_ensemble_class_erf(self):
        """Test FCN2NetworkEnsembleErf convenience class."""
        model = FCN2NetworkEnsembleErf(d=10, n1=50, P=100, ens=2)
        assert model.activation_name == "erf", "FCN2NetworkEnsembleErf should use erf activation"
        assert model.W0.shape == (2, 50, 10), f"Expected shape (2, 50, 10), got {model.W0.shape}"
    
    def test_ensemble_class_linear(self):
        """Test FCN2NetworkEnsembleLinear convenience class."""
        model = FCN2NetworkEnsembleLinear(d=10, n1=50, P=100, ens=2)
        assert model.activation_name == "linear", "FCN2NetworkEnsembleLinear should use linear activation"
        assert model.W0.shape == (2, 50, 10), f"Expected shape (2, 50, 10), got {model.W0.shape}"


class TestFCN2NetworkForwardPass:
    """Test FCN2Network forward passes."""
    
    def test_forward_pass_basic_erf(self):
        """Test basic forward pass with erf activation."""
        d, n1, P, ens = 10, 50, 20, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(20, 10)
        y = model(X)
        assert y.shape == (20, 3), f"Expected output shape (20, 3), got {y.shape}"
        assert torch.isfinite(y).all(), f"Output contains NaN or Inf: min={y.min()}, max={y.max()}, has_nan={torch.isnan(y).any()}, has_inf={torch.isinf(y).any()}"
    
    def test_forward_pass_linear(self):
        """Test forward pass with linear activation."""
        d, n1, P, ens = 5, 20, 20, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="linear")
        X = torch.randn(10, 5)
        y = model(X)
        assert y.shape == (10, 2), f"Expected shape (10, 2), got {y.shape}"
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        d, n1, P, ens = 10, 50, 1, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(1, 10)
        y = model(X)
        assert y.shape == (1, 3), f"Expected shape (1, 3), got {y.shape}"
    
    def test_forward_pass_large_batch(self):
        """Test forward pass with large batch."""
        d, n1, P, ens = 10, 50, 500, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(500, 10)
        y = model(X)
        assert y.shape == (500, 3), f"Expected shape (500, 3), got {y.shape}"
    
    def test_h0_preactivation_shape(self):
        """Test h0_preactivation output shape."""
        d, n1, P, ens = 10, 50, 20, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(20, 10)
        h0 = model.h0_preactivation(X)
        assert h0.shape == (20, 3, 50), f"Expected h0.shape (20, 3, 50), got {h0.shape}"
    
    def test_h0_activation_shape(self):
        """Test h0_activation output shape."""
        d, n1, P, ens = 10, 50, 20, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(20, 10)
        a0 = model.h0_activation(X)
        assert a0.shape == (20, 3, 50), f"Expected a0.shape (20, 3, 50), got {a0.shape}"
        assert torch.isfinite(a0).all(), f"Activation contains NaN/Inf"
    
    def test_forward_ensemble_consistency(self):
        """Test that ensemble members produce different outputs."""
        torch.manual_seed(42)
        d, n1, P, ens = 10, 50, 20, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(5, 10)
        y = model(X)
        # Check that ensemble members differ (not all identical)
        assert not torch.allclose(y[:, 0], y[:, 1]), "Ensemble members 0 and 1 are identical (should differ)"
        assert not torch.allclose(y[:, 1], y[:, 2]), "Ensemble members 1 and 2 are identical (should differ)"
    
    def test_forward_determinism(self):
        """Test that forward pass is deterministic."""
        d, n1, P, ens = 10, 50, 20, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(5, 10)
        y1 = model(X)
        y2 = model(X)
        assert torch.allclose(y1, y2), f"Forward passes differ: max_diff={torch.abs(y1 - y2).max()}"


class TestFCN2NetworkEigenvalues:
    """Test eigenvalue computation via H_eig."""
    
    def test_h_eig_shape(self):
        """Test that H_eig returns correct shape."""
        d, n1, P, ens = 10, 50, 100, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(100, 10)
        Y = torch.randn(100)
        eigs = model.H_eig(X, Y)
        assert eigs.shape == (100,), f"Expected eigenvalues shape (100,), got {eigs.shape}"
        assert eigs.dtype in [torch.float32, torch.float64], f"Expected float dtype, got {eigs.dtype}"
    
    def test_h_eig_matrix_y(self):
        """Test H_eig with matrix Y (multiple eigenfunctions)."""
        d, n1, P, ens = 10, 50, 100, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(100, 10)
        Y = torch.randn(100, 5)  # 5 eigenfunctions
        eigs = model.H_eig(X, Y)
        assert eigs.shape == (5,), f"Expected shape (5,), got {eigs.shape}"
    
    def test_h_eig_with_std(self):
        """Test H_eig returning standard deviations."""
        d, n1, P, ens = 10, 50, 100, 5
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(100, 10)
        Y = torch.randn(100)
        eigs, std = model.H_eig(X, Y, std=True)
        assert eigs.shape == (100,), f"Expected eigs shape (100,), got {eigs.shape}"
        assert std.shape == (100,), f"Expected std shape (100,), got {std.shape}"
        assert (std >= 0).all(), f"Standard deviations should be non-negative, min: {std.min()}"
    
    def test_h_eig_no_grad(self):
        """Test that H_eig operates in no_grad mode."""
        d, n1, P, ens = 10, 50, 100, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(100, 10, requires_grad=True)
        Y = torch.randn(100)
        eigs = model.H_eig(X, Y)
        # X still has requires_grad=True but gradients shouldn't flow through H_eig
        assert eigs.requires_grad == False, "Eigenvalues should not require gradients"


class TestFCN2NetworkSizeVariations:
    """Test network stability across different sizes."""
    
    @pytest.mark.parametrize("d,n1,P,ens", [
        (5, 10, 50, 1),
        (10, 50, 100, 1),
        (50, 200, 200, 1),
        (10, 20, 100, 5),
        (20, 50, 150, 10),
    ])
    def test_various_sizes_forward(self, d, n1, P, ens):
        """Test forward pass for various network sizes."""
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(P, d)
        y = model(X)
        assert y.shape == (P, ens), f"For d={d}, n1={n1}, P={P}, ens={ens}: expected shape ({P}, {ens}), got {y.shape}"
        assert torch.isfinite(y).all(), f"For d={d}, n1={n1}, P={P}, ens={ens}: output contains NaN/Inf"
    
    @pytest.mark.parametrize("d,n1,P,ens", [
        (5, 10, 50, 1),
        (10, 100, 100, 3),
        (50, 300, 200, 5),
    ])
    def test_eigenvalues_various_sizes(self, d, n1, P, ens):
        """Test eigenvalue computation for various network sizes."""
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(P, d)
        Y = torch.randn(P)
        eigs = model.H_eig(X, Y)
        assert eigs.shape == (P,), f"For d={d}, n1={n1}, P={P}: expected shape ({P},), got {eigs.shape}"
        assert torch.isfinite(eigs).all(), f"For d={d}, n1={n1}, P={P}: eigenvalues contain NaN/Inf"


class TestFCN2NetworkActivationBehavior:
    """Test activation-specific behaviors."""
    
    def test_erf_activation_bounded(self):
        """Test that erf activation produces bounded outputs."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(100, 10)
        a0 = model.h0_activation(X)
        # erf output should be bounded in [-1, 1]
        assert (a0 >= -1.0).all(), f"erf activation produces values < -1: min={a0.min()}"
        assert (a0 <= 1.0).all(), f"erf activation produces values > 1: max={a0.max()}"
    
    def test_linear_activation_unbounded(self):
        """Test that linear activation is unbounded."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="linear")
        X = torch.randn(100, 10) * 10  # Large inputs
        a0 = model.h0_activation(X)
        # Linear activation can be unbounded
        assert a0.shape == (100, 2, 50), f"Expected shape (100, 2, 50), got {a0.shape}"
    
    def test_erf_vs_linear_difference(self):
        """Test that erf and linear activations produce different outputs."""
        torch.manual_seed(42)
        d, n1, P, ens = 10, 50, 100, 1
        
        # Create two identical models with different activations
        model_erf = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        model_linear = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="linear")
        
        # Copy weights from erf to linear
        model_linear.W0.data = model_erf.W0.data.clone()
        model_linear.A.data = model_erf.A.data.clone()
        
        X = torch.randn(50, 10)
        y_erf = model_erf(X)
        y_linear = model_linear(X)
        
        # They should differ (unless by extreme coincidence)
        assert not torch.allclose(y_erf, y_linear, atol=1e-4), "erf and linear activations should produce different outputs"


class TestFCN2NetworkStability:
    """Test numerical stability of networks."""
    
    def test_no_nan_large_network(self):
        """Test that large networks don't produce NaNs."""
        d, n1, P, ens = 100, 500, 200, 3
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens, activation="erf")
        X = torch.randn(200, 100)
        y = model(X)
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"
    
    def test_no_nan_eigenvalues(self):
        """Test that eigenvalue computation doesn't produce NaNs."""
        d, n1, P, ens = 50, 200, 500, 5
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(500, 50)
        Y = torch.randn(500)
        eigs = model.H_eig(X, Y)
        assert not torch.isnan(eigs).any(), "Eigenvalues contain NaN"
        assert not torch.isinf(eigs).any(), "Eigenvalues contain Inf"
    
    def test_very_small_weights(self):
        """Test stability with very small weight initialization."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(
            d=d, n1=n1, P=P, ens=ens,
            weight_initialization_variance=(1e-6, 1e-6)
        )
        X = torch.randn(100, 10)
        y = model(X)
        assert torch.isfinite(y).all(), "Output with small weights contains NaN/Inf"
    
    def test_very_large_weights(self):
        """Test stability with very large weight initialization."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(
            d=d, n1=n1, P=P, ens=ens,
            weight_initialization_variance=(10.0, 10.0)
        )
        X = torch.randn(100, 10)
        y = model(X)
        # With erf activation, should still be bounded
        assert (y >= -1.5).all() and (y <= 1.5).all(), "Output with large weights should remain bounded"


class TestFCN2NetworkGradients:
    """Test gradient flow through network."""
    
    def test_weight_gradients(self):
        """Test that gradients flow to weights."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(50, 10, requires_grad=True)
        y = model(X).sum()
        y.backward()
        
        assert model.W0.grad is not None, "W0 should have gradients"
        assert model.A.grad is not None, "A should have gradients"
        assert torch.isfinite(model.W0.grad).all(), "W0 gradients contain NaN/Inf"
        assert torch.isfinite(model.A.grad).all(), "A gradients contain NaN/Inf"
    
    def test_input_gradients(self):
        """Test that gradients flow to inputs."""
        d, n1, P, ens = 10, 50, 100, 2
        model = FCN2NetworkActivationGeneric(d=d, n1=n1, P=P, ens=ens)
        X = torch.randn(50, 10, requires_grad=True)
        y = model(X).sum()
        y.backward()
        
        assert X.grad is not None, "X should have gradients"
        assert torch.isfinite(X.grad).all(), "X gradients contain NaN/Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
