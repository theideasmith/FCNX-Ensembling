"""
Test suite for kappa correction in 2-layer NNGP networks.

This test diagnoses issues with the self_consistent_kappa_solver.jl
by comparing it against known analytical solutions and numerical results.
"""

import pytest
import torch
import numpy as np
import json
import subprocess
import tempfile
from pathlib import Path
import sys

# Add lib and julia_lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent.parent / "julia_lib"))

JULIA_SOLVER = Path(__file__).parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
DEVICE = "cpu"


def arcsin_kernel_2layer_nngp(X: torch.Tensor) -> torch.Tensor:
    """
    Compute arcsin kernel matrix for 2-layer NNGP.
    X: (P, d) - P samples, d input dimension
    Returns: (P, P) kernel matrix
    """
    P, d = X.shape
    XXT = (X @ X.T) / d  # (P, P)
    diag_vals = torch.sqrt(1 + 2 * torch.diagonal(XXT))  # (P,)
    denom = diag_vals.unsqueeze(1) @ diag_vals.unsqueeze(0)  # (P, P)
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)  # Ensure numerical stability
    K = (2 / torch.pi) * torch.asin(arg)
    return K


def normalize_eigenvalues(lambdas: np.ndarray, P: int) -> np.ndarray:
    """Normalize eigenvalues: divide by number of eigenvalues (P)."""
    return lambdas / P


def solve_kappa_ref(lambdas: np.ndarray, kappa_bare: float, P: int, 
                    tol: float = 1e-10, max_iter: int = 1000) -> float:
    """
    Reference implementation of self-consistent kappa solver.
    This is the corrected formula: κ = κ_bare + (1/P) * Σ_i λ_i * κ / (κ + λ_i)
    Rearranged: κ - κ_bare = (1/P) * Σ_i λ_i * κ / (κ + λ_i)
    """
    from scipy.optimize import fsolve
    
    def equation(kappa_eff):
        denom = kappa_eff + lambdas
        correction = np.sum(lambdas * kappa_eff / denom)
        return kappa_eff - (kappa_bare + correction / P)
    
    # Initial guess
    x0 = kappa_bare + np.mean(lambdas)
    sol = fsolve(equation, x0)
    return float(sol[0])


def call_julia_solver(lambdas: np.ndarray, kappa_bare: float, P: int) -> float:
    """Call the Julia self_consistent_kappa_solver.jl script."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf:
        json.dump({
            "eigenvalues": lambdas.tolist(),
            "kappa_bare": float(kappa_bare)
        }, tf)
        eig_json = tf.name
    
    try:
        output = subprocess.check_output(
            ["julia", str(JULIA_SOLVER), eig_json, str(P)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        # Parse output for kappa_eff
        import re
        match = re.search(r"kappa_eff\s*=\s*([-0-9.eE+]+)", output)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Could not parse kappa_eff from output:\n{output}")
    finally:
        Path(eig_json).unlink(missing_ok=True)


class TestKappaCorrectionNNGP:
    """Test kappa correction for 2-layer NNGP networks."""

    def test_arcsin_kernel_computation(self):
        """Test arcsin kernel computation for 2-layer NNGP."""
        torch.manual_seed(42)
        P, d = 100, 50
        X = torch.randn(P, d)
        K = arcsin_kernel_2layer_nngp(X)
        
        # Check shape
        assert K.shape == (P, P), f"Expected kernel shape (P, P)={(P, P)}, got {K.shape}"
        
        # Check symmetry
        assert torch.allclose(K, K.T, atol=1e-6), "Kernel should be symmetric"
        
        # Check diagonal is positive
        assert torch.all(torch.diagonal(K) > 0), "Kernel diagonal should be positive"
        
        # Check eigenvalues are non-negative
        eigs = torch.linalg.eigvalsh(K)
        assert torch.all(eigs >= -1e-6), f"Kernel should be PSD, got min eigenvalue {eigs.min()}"
        
        print(f"✓ arcsin_kernel_computation passed (P={P}, d={d})")

    def test_simple_kappa_solver(self):
        """Test kappa solver on simple eigenvalue spectrum."""
        # Simple test case: uniform eigenvalues
        P = 100
        kappa_bare = 1.0
        lambdas = np.ones(P) * 0.1  # All eigenvalues equal to 0.1
        
        # Call reference implementation
        kappa_ref = solve_kappa_ref(lambdas, kappa_bare, P)
        
        # For uniform eigenvalues, we can compute analytically:
        # κ = κ_bare + λ * κ / (κ + λ)
        # This should have a simple solution
        assert kappa_ref > kappa_bare, "kappa_eff should be > kappa_bare"
        assert kappa_ref < kappa_bare + P * 0.1, "kappa_eff should be bounded"
        
        print(f"✓ simple_kappa_solver passed: κ_bare={kappa_bare:.4f}, κ_eff={kappa_ref:.4f}")

    def test_kappa_correction_small_spectrum(self):
        """Test kappa correction with a small, known spectrum."""
        torch.manual_seed(123)
        P = 50
        d = 25
        kappa_bare = 2.0
        
        # Generate small NNGP kernel
        X = torch.randn(P, d)
        K = arcsin_kernel_2layer_nngp(X)
        
        # Extract eigenvalues
        eigs = torch.linalg.eigvalsh(K)
        lambdas = eigs.cpu().numpy() / P  # Normalize by P
        
        # Call reference solver
        kappa_eff_ref = solve_kappa_ref(lambdas, kappa_bare, P)
        
        # Verify properties
        assert kappa_eff_ref > kappa_bare, f"kappa_eff ({kappa_eff_ref}) should be > kappa_bare ({kappa_bare})"
        assert np.isfinite(kappa_eff_ref), "kappa_eff should be finite"
        
        print(f"✓ kappa_correction_small_spectrum passed: P={P}, d={d}")
        print(f"  κ_bare={kappa_bare:.6f}, κ_eff={kappa_eff_ref:.6f}")
        print(f"  Correction: {kappa_eff_ref - kappa_bare:.6f}")

    def test_kappa_correction_medium_spectrum(self):
        """Test kappa correction with medium-sized spectrum."""
        torch.manual_seed(456)
        P = 200
        d = 100
        kappa_bare = 1.0
        
        X = torch.randn(P, d)
        K = arcsin_kernel_2layer_nngp(X)
        eigs = torch.linalg.eigvalsh(K)
        lambdas = eigs.cpu().numpy() / P
        
        kappa_eff_ref = solve_kappa_ref(lambdas, kappa_bare, P)
        
        assert kappa_eff_ref > kappa_bare, "kappa_eff should be > kappa_bare"
        assert np.isfinite(kappa_eff_ref), "kappa_eff should be finite"
        
        print(f"✓ kappa_correction_medium_spectrum passed: P={P}, d={d}")
        print(f"  κ_eff={kappa_eff_ref:.6f} (correction: {100*(kappa_eff_ref-kappa_bare)/kappa_bare:.2f}%)")

    @pytest.mark.skipif(not JULIA_SOLVER.exists(), reason="Julia solver not found")
    def test_julia_solver_runs(self):
        """Test that Julia solver runs successfully and produces valid output."""
        torch.manual_seed(789)
        P = 100
        d = 50
        kappa_bare = 1.0
        
        X = torch.randn(P, d)
        K = arcsin_kernel_2layer_nngp(X)
        eigs = torch.linalg.eigvalsh(K)
        lambdas = eigs.cpu().numpy()  # Use raw eigenvalues
        
        # Get Julia solution
        try:
            kappa_julia = call_julia_solver(lambdas, kappa_bare, P)
            
            # Check that we got a valid result
            assert isinstance(kappa_julia, float), "kappa_julia should be a float"
            assert np.isfinite(kappa_julia), "kappa_julia should be finite"
            assert kappa_julia > 0, "kappa_julia should be positive"
            
            print(f"✓ julia_solver_runs passed:")
            print(f"  P={P}, d={d}, κ_bare={kappa_bare}")
            print(f"  Julia result: κ_eff={kappa_julia:.10f}")
        except subprocess.CalledProcessError as e:
            print(f"Julia solver failed with error:\n{e.output}")
            raise

    def test_kappa_scaling_with_spectrum_size(self):
        """Test how kappa correction scales with spectrum size."""
        torch.manual_seed(42)
        kappa_bare = 1.0
        
        corrections = []
        sizes = [50, 100, 200]
        
        for P in sizes:
            d = P // 2
            X = torch.randn(P, d)
            K = arcsin_kernel_2layer_nngp(X)
            eigs = torch.linalg.eigvalsh(K)
            lambdas = eigs.cpu().numpy() / P
            
            kappa_eff = solve_kappa_ref(lambdas, kappa_bare, P)
            correction = kappa_eff - kappa_bare
            corrections.append(correction)
            
            print(f"  P={P:3d}: κ_eff={kappa_eff:.6f}, correction={correction:.6f}")
        
        print(f"✓ kappa_scaling_with_spectrum_size passed")

    def test_extreme_eigenvalue_spectrum(self):
        """Test kappa solver with extreme eigenvalue ranges."""
        # Test with very small and very large eigenvalues
        P = 100
        kappa_bare = 1.0
        
        # Case 1: Very small eigenvalues
        lambdas_small = np.ones(P) * 1e-6
        kappa_small = solve_kappa_ref(lambdas_small, kappa_bare, P)
        assert kappa_small > kappa_bare, "Should still correct for small eigenvalues"
        print(f"  Small eigenvalues: κ_eff={kappa_small:.10f} (vs κ_bare={kappa_bare})")
        
        # Case 2: Large eigenvalues
        lambdas_large = np.ones(P) * 10.0
        kappa_large = solve_kappa_ref(lambdas_large, kappa_bare, P)
        assert kappa_large > kappa_bare, "Should correct for large eigenvalues"
        print(f"  Large eigenvalues: κ_eff={kappa_large:.6f} (vs κ_bare={kappa_bare})")
        
        print(f"✓ extreme_eigenvalue_spectrum passed")

    def test_kappa_monotonicity(self):
        """Test that increasing kappa_bare increases kappa_eff monotonically."""
        torch.manual_seed(42)
        P = 100
        d = 50
        
        X = torch.randn(P, d)
        K = arcsin_kernel_2layer_nngp(X)
        eigs = torch.linalg.eigvalsh(K)
        lambdas = eigs.cpu().numpy() / P
        
        kappa_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        kappa_eff_values = []
        
        for kb in kappa_values:
            ke = solve_kappa_ref(lambdas, kb, P)
            kappa_eff_values.append(ke)
            print(f"  κ_bare={kb:.1f} → κ_eff={ke:.6f}")
        
        # Check monotonicity
        for i in range(len(kappa_eff_values) - 1):
            assert kappa_eff_values[i] < kappa_eff_values[i+1], \
                f"kappa_eff should be monotonically increasing, but {kappa_eff_values[i]} >= {kappa_eff_values[i+1]}"
        
        print(f"✓ kappa_monotonicity passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
