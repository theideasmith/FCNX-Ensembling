"""
Diagnostic test to verify self_consistent_kappa_solver.jl runs correctly
on 2-layer NNGP networks.
"""

import numpy as np
import json
import subprocess
import tempfile
from pathlib import Path
import torch

JULIA_SOLVER = Path(__file__).parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"


def test_julia_solver_with_nngp_spectrum():
    """Test Julia solver with actual NNGP eigenvalue spectrum."""
    
    print("=" * 70)
    print("TESTING JULIA SOLVER WITH 2-LAYER NNGP SPECTRUM")
    print("=" * 70)
    print()
    
    # Generate 2-layer NNGP kernel eigenvalues
    torch.manual_seed(42)
    P = 100
    d = 50
    kappa_bare = 1.0
    
    # Create random input
    X = torch.randn(P, d)
    
    # Compute arcsin kernel (2-layer NNGP)
    XXT = (X @ X.T) / d
    diag_vals = torch.sqrt(1 + 2 * torch.diagonal(XXT))
    denom = diag_vals.unsqueeze(1) @ diag_vals.unsqueeze(0)
    arg = 2 * XXT / denom
    arg = torch.clamp(arg, -1.0, 1.0)
    K = (2 / torch.pi) * torch.asin(arg)
    
    # Get eigenvalues
    eigs = torch.linalg.eigvalsh(K)
    lambdas = eigs.cpu().numpy()
    
    print(f"Generated 2-layer NNGP kernel:")
    print(f"  P={P}, d={d}")
    print(f"  Kernel shape: {K.shape}")
    print(f"  Number of eigenvalues: {len(lambdas)}")
    print(f"  λ_max: {lambdas.max():.6f}")
    print(f"  λ_min: {lambdas.min():.6e}")
    print(f"  λ_mean: {lambdas.mean():.6f}")
    print()
    
    # Prepare input for Julia solver
    print(f"Parameters for Julia solver:")
    print(f"  κ_bare={kappa_bare}")
    print(f"  P={P}")
    print()
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf:
        json.dump({
            "eigenvalues": lambdas.tolist(),
            "kappa_bare": float(kappa_bare)
        }, tf)
        eig_json = tf.name
    
    try:
        print(f"Calling Julia solver...")
        output = subprocess.check_output(
            ["julia", str(JULIA_SOLVER), eig_json, str(P)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        print(f"Julia output:")
        print(output)
        
        import re
        match = re.search(r"kappa_eff\s*=\s*([-0-9.eE+]+)", output)
        if match:
            kappa_julia = float(match.group(1))
            print(f"✓ SUCCESS!")
            print(f"  κ_eff = {kappa_julia:.10f}")
            print(f"  Correction: {kappa_julia - kappa_bare:.10f} ({100*(kappa_julia-kappa_bare)/kappa_bare:.2f}%)")
            return True
        else:
            print("✗ FAILED: Could not parse kappa_eff from output")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: Julia solver error:\n{e.output}")
        return False
    finally:
        Path(eig_json).unlink(missing_ok=True)


if __name__ == "__main__":
    test_julia_solver_with_nngp_spectrum()
