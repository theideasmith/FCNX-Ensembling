#!/usr/bin/env python3
"""
Script to validate theory predictions against NNGP initialization.

Checks that theoretical lH eigenvalues from eos_fcn3erf.jl closely match
the empirical variances of He1 and He3 projections at initialization.

Usage:
    python validate_theory_nngp.py --P 600 --d 50 --N 800 --chi 80 --kappa 1.0 --device cuda:0
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import json
import subprocess
import tempfile
from typing import Dict, Optional

torch.set_default_dtype(torch.float32)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric


def compute_theory(d: int, P: int, N: int, chi: float, kappa: float, eps: float) -> Dict[str, Optional[float]]:
    """Get theoretical predictions by calling Julia eos_fcn3erf.jl and reading JSON output."""
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--n2={N}",
        f"--chi={chi}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={to_path}",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(to_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Julia theory solver failed: {e}")
        data = {}
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass

    tgt = data.get("target", {}) if isinstance(data, dict) else {}
    perp = data.get("perpendicular", {}) if isinstance(data, dict) else {}

    return {
        "lH1T": tgt.get("lH1T"),
        "lH1P": perp.get("lH1P"),
        "lH3T": tgt.get("lH3T"),
        "lH3P": perp.get("lH3P"),
    }


def validate_nngp_initialization(d, P, N, chi, kappa, eps=0.03, device_str="cpu", seed=42):
    """
    Initialize a network and check agreement between theory and empirical projections.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"NNGP Validation: d={d}, P={P}, N={N}, chi={chi}, kappa={kappa}")
    print(f"{'='*70}")
    
    # Data
    torch.manual_seed(seed)
    X = torch.randn(10000, d, device=device)
    X0 = X[:, 0].squeeze(-1).unsqueeze(-1)
    Y = X0 + eps * (X0**3 - 3 * X0)
    
    # Initialize model
    torch.manual_seed(70)
    model = FCN3NetworkActivationGeneric(
        d, N, N, P, ens=1,  # Use ens=1 for cleaner validation
        activation="erf",
        weight_initialization_variance=(1/d, 1/N, 1/(N * chi))
    ).to(device)
    
    model.eval()
    
    # Get theory predictions
    print("\nComputing theory predictions...")
    theory_H = compute_theory(d, P, N, chi, kappa, eps)
    print(f"Theory: lH1P={theory_H['lH1P']:.6f}, lH3P={theory_H['lH3P']:.6f}")
    
    # Compute empirical projections at initialization
    print("\nComputing empirical projections at initialization...")
    with torch.no_grad():
        output = model.h1_preactivation(X)  # shape: (P, ensemble, n2)
        P_dim = output.shape[0]
        
        # Compute projection directions
        x3_perp = X[:, 3] if d > 3 else torch.randn_like(X[:, 0])
        x3_perp_normed = x3_perp / x3_perp.norm() 
        
        # Hermite cubic polynomial for perp: (x^3 - 3x)/sqrt(6)
        h3_perp = (x3_perp**3 - 3.0 * x3_perp) / 6**0.5
        h3_perp_normed = h3_perp / h3_perp.norm() 
        
        # Project outputs onto perp directions per ensemble
        proj_lin_perp = torch.einsum('pqn,p->qn', output, x3_perp_normed) / P_dim
        proj_cubic_perp = torch.einsum('pqn,p->qn', output, h3_perp_normed) / P_dim
        
        # Compute variances
        var_lin_perp = float(torch.var(proj_lin_perp).item())
        var_cubic_perp = float(torch.var(proj_cubic_perp).item())
        
        # Alternative method: Compute kernel and sandwich with normalized vectors
        print("\nComputing kernel sandwich method...")
        f_inf = model.h1_preactivation(X)  # (P, ens, n2)
        
        # Compute kernel per ensemble: K_i[u,v] = sum_m f_inf[u,i,m] * f_inf[v,i,m]
        # Result: (ens, P, P)
        hh_inf_i = torch.einsum('uqm,vqm->quv', f_inf, f_inf) / (model.n1 * P_dim)
        
        # Average over ensembles: K[u,v] = mean_q K_i[q,u,v]
        hh_inf = torch.mean(hh_inf_i, dim=0)  # (P, P)
        
        # Normalize projection vectors
        x3_perp_norm = x3_perp / x3_perp.norm()
        h3_perp_norm = h3_perp / h3_perp.norm()
        
        # Compute eigenvalues via sandwich: v^T K v / (v^T v) = v^T K v (since v normalized)
        kernel_lin_perp = torch.einsum('u,uv,v->', x3_perp_norm, hh_inf, x3_perp_norm).item()
        kernel_cubic_perp = torch.einsum('u,uv,v->', h3_perp_norm, hh_inf, h3_perp_norm).item()
    
    print(f"Empirical (projection): lin_perp={var_lin_perp:.6f}, cubic_perp={var_cubic_perp:.6f}")
    print(f"Empirical (kernel sandwich): lin_perp={kernel_lin_perp:.6f}, cubic_perp={kernel_cubic_perp:.6f}")
    
    # Compute relative errors
    print("\n" + "="*70)
    print("Comparison (relative error - projection method):")
    print("="*70)
    
    rel_err_h1p_proj = abs(var_lin_perp - theory_H['lH1P']) / (abs(theory_H['lH1P']) + 1e-10) * 100
    rel_err_h3p_proj = abs(var_cubic_perp - theory_H['lH3P']) / (abs(theory_H['lH3P']) + 1e-10) * 100
    
    print(f"He1 Perp (proj):   theory={theory_H['lH1P']:.6f}, empirical={var_lin_perp:.6f}, rel_err={rel_err_h1p_proj:.2f}%")
    print(f"He3 Perp (proj):   theory={theory_H['lH3P']:.6f}, empirical={var_cubic_perp:.6f}, rel_err={rel_err_h3p_proj:.2f}%")
    
    print("\n" + "="*70)
    print("Comparison (relative error - kernel sandwich method):")
    print("="*70)
    
    rel_err_h1p_kernel = abs(kernel_lin_perp - theory_H['lH1P']) / (abs(theory_H['lH1P']) + 1e-10) * 100
    rel_err_h3p_kernel = abs(kernel_cubic_perp - theory_H['lH3P']) / (abs(theory_H['lH3P']) + 1e-10) * 100
    
    print(f"He1 Perp (kernel): theory={theory_H['lH1P']:.6f}, empirical={kernel_lin_perp:.6f}, rel_err={rel_err_h1p_kernel:.2f}%")
    print(f"He3 Perp (kernel): theory={theory_H['lH3P']:.6f}, empirical={kernel_cubic_perp:.6f}, rel_err={rel_err_h3p_kernel:.2f}%")
    
    # Check if agreement is good (< 5% relative error)
    tolerance = 5.0
    all_good_proj = all([rel_err_h1p_proj < tolerance, rel_err_h3p_proj < tolerance])
    all_good_kernel = all([rel_err_h1p_kernel < tolerance, rel_err_h3p_kernel < tolerance])
    
    print("\n" + "="*70)
    if all_good:
        print("✓ EXCELLENT AGREEMENT! All relative errors < 5%")
    else:
        print("⚠ SOME DISCREPANCIES - Check which eigenvalues have large relative errors")
    print("="*70 + "\n")
    
    return {
        "theory": theory_H,
        "empirical": {
            "lin_perp": var_lin_perp,
            "cubic_perp": var_cubic_perp,
        },
        "relative_errors": {
            "lin_perp": rel_err_h1p,
            "cubic_perp": rel_err_h3p,
        },
        "agreement": all_good,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate theory predictions against NNGP initialization')
    parser.add_argument('--P', type=int, required=True, help='Number of data points')
    parser.add_argument('--d', type=int, required=True, help='Input dimension')
    parser.add_argument('--N', type=int, required=True, help='Hidden layer size')
    parser.add_argument('--chi', type=int, required=True, help='Chi parameter')
    parser.add_argument('--kappa', type=float, required=True, help='Kappa parameter')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon parameter for cubic target generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = validate_nngp_initialization(
        d=args.d,
        P=args.P,
        N=args.N,
        chi=args.chi,
        kappa=args.kappa,
        eps=args.eps,
        device_str=args.device,
        seed=args.seed,
    )
    
    # Save results
    results_path = Path("nngp_validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
