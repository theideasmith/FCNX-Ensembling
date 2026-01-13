#!/usr/bin/env python3
"""
Validate FCN2 (2-layer) theory predictions against NNGP initialization.

Checks that theoretical lJ eigenvalues (linear and cubic) from FCS2Erf_Cubic 
closely match the empirical variances of He1 and He3 projections of the 
hidden layer activation at initialization.

For FCN2, we examine h0_activation (first layer after erf), which should
match the J kernel predictions.

Usage:
    python validate_fcn2_theory.py --P 600 --d 50 --N 800 --kappa 1.0 --device cuda:0
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
from FCN2Network import FCN2NetworkEnsembleErf


def compute_theory_fcn2(d: int, P: int, N: int, chi: float, kappa: float, eps: float) -> Dict[str, Optional[float]]:
    """Get theoretical predictions by calling Julia FCS2Erf_Cubic solver."""
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_cubic_eigs.jl"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--chi={chi}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={to_path}",
        "--quiet",
    ]
    print("Running Julia command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
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
    print(data)
    tgt = data.get("target", {}) if isinstance(data, dict) else {}
    perp = data.get("perpendicular", {}) if isinstance(data, dict) else {}

    return {
        "lJ1T": tgt.get("lJ1T"),
        "lJ1P": perp.get("lJ1P"),
        "lJ3T": tgt.get("lJ3T"),
        "lJ3P": perp.get("lJ3P"),
        "lWT": tgt.get("lWT"),
        "lWP": perp.get("lWP"),
    }


def validate_fcn2_initialization(d, P, N, chi, kappa, eps=0.03, device_str="cpu", seed=42):
    """
    Initialize a 2-layer network and check agreement between theory and empirical projections.
    For FCN2, we examine the J kernel (hidden layer activations).
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"FCN2 NNGP Validation: d={d}, P={P}, N={N}, χ={chi}, κ={kappa}")
    print(f"{'='*70}")
    
    # Data
    torch.manual_seed(seed)
    Pinf = 5000
    X = torch.randn(Pinf, d, device=device)
    
    # Initialize model - FCN2 has 2 layers: input -> hidden (erf) -> output
    torch.manual_seed(70)
    # For FCN2 in standard scaling: σ_W0^2 = 1/d, σ_A^2 = 1/(N*chi)
    model = FCN2NetworkEnsembleErf(
        d, N, P, ens=1,  # ens=1 for clean validation
        weight_initialization_variance=(1/d, 1/(N * chi)),
        device=device
    )
    
    model.eval()
   
    # Get theory predictions
    print("\nComputing theory predictions from FCS2Erf_Cubic...")
    theory_J = compute_theory_fcn2(d, P, N, chi, kappa, eps)
    print(f"Theory (perp only): lJ1P={theory_J['lJ1P']:.6f}, lJ3P={theory_J['lJ3P']:.6f}")


    theory_J['lWP'] = 1 / d  # Theoretical W0 variance for FCN2 in standard scaling
    print("Dataset dimension is: ", d)
    theory_J['lJ3P'] = 16 / (np.pi * (3)**3) * (15 * (theory_J['lWP']**3))  # Corrected cubic term
    print(f"The lJ3P corrected value is: {theory_J['lJ3P']:.8f}")
    # Compute empirical W0 variance
    print("\nComputing empirical W0 variance...")
    with torch.no_grad():
        W0_var = float(torch.var(model.W0).item())
    print(f"Empirical W0 variance: {W0_var:.6f}")
    print(f"Theory lWP: {theory_J.get('lWP', float('nan')):.6f}")
    
    # Compare W0 variance to theory
    if not np.isnan(theory_J.get('lWP', float('nan'))):
        rel_err_wt = abs(W0_var - theory_J['lWP']) / (abs(theory_J['lWP']) + 1e-10) * 100
        print(f"W0 variance relative error: {rel_err_wt:.2f}%")
    else:
        rel_err_wt = float('nan')
    
    # Compute empirical projections at initialization
    print("\nComputing empirical projections of h0_activation (J kernel) averaged over all input dimensions...")
    with torch.no_grad():
        # h0_activation is erf(W0 @ X) -> (P, ens, n1)
        h0 = model.h0_activation(X)  # shape: (P, ens, n1)
        P_dim = h0.shape[0]
        # Method: Projection method averaged over all input dims
        lin_vars = []
        cubic_vars = []
        for i in range(d):
            xi = X[:, i]
            xi_norm = xi / xi.norm() * np.sqrt(P_dim)
            he3 = (xi**3 - 3.0 * xi) / (6.0**0.5)
            he3_norm = he3 / he3.norm() * np.sqrt(P_dim)

            proj_lin = torch.einsum('pqk,p->qk', h0, xi_norm) / P_dim
            proj_cubic = torch.einsum('pqk,p->qk', h0, he3_norm) / P_dim

            lin_vars.append(torch.var(proj_lin))
            cubic_vars.append(torch.var(proj_cubic))

        var_lin_mean = float(torch.mean(torch.stack(lin_vars)).item())
        var_lin_std = float(torch.std(torch.stack(lin_vars)).item())
        var_cubic_mean = float(torch.mean(torch.stack(cubic_vars)).item())
        var_cubic_std = float(torch.std(torch.stack(cubic_vars)).item())

    print(f"Empirical (projection, avg over dims): lin_mean={var_lin_mean:.6f}±{var_lin_std:.6f}, cubic_mean={var_cubic_mean:.6f}±{var_cubic_std:.6f}")

    # Compute eigenvalues via H_eig method (sandwiching)
    print("\nComputing eigenvalues via H_eig (kernel sandwiching method)...")
    with torch.no_grad():
        # For each input dimension, compute eigenvalues for linear and cubic directions
        eig_lin_sandwich = []
        eig_cubic_sandwich = []
        
        for i in range(d):
            xi = X[:, i]  # shape: (P,)
            he3 = (xi**3 - 3.0 * xi) / (6.0**0.5)
            
            # Use model.H_eig with the direction vectors
            eig_lin = float(model.H_eig(X, xi.unsqueeze(1)).item())
            eig_cubic = float(model.H_eig(X, he3.unsqueeze(1)).item())
            
            eig_lin_sandwich.append(eig_lin)
            eig_cubic_sandwich.append(eig_cubic)
        
        eig_lin_mean = float(np.mean(eig_lin_sandwich))
        eig_lin_std = float(np.std(eig_lin_sandwich))
        eig_cubic_mean = float(np.mean(eig_cubic_sandwich))
        eig_cubic_std = float(np.std(eig_cubic_sandwich))
        
    print(f"Eigenvalues (H_eig): lin={eig_lin_mean:.6f}±{eig_lin_std:.6f}, cubic={eig_cubic_mean:.6f}±{eig_cubic_std:.6f}")
    
    # Compare to theory
    rel_err_j1_sandwich = abs(eig_lin_mean - theory_J['lJ1P']) / (abs(theory_J['lJ1P']) + 1e-10) * 100
    rel_err_j3_sandwich = abs(eig_cubic_mean - theory_J['lJ3P']) / (abs(theory_J['lJ3P']) + 1e-10) * 100
    
    print(f"lJ1 (H_eig): theory={theory_J['lJ1P']:.6f}, empirical={eig_lin_mean:.6f}, rel_err={rel_err_j1_sandwich:.2f}%")
    print(f"lJ3 (H_eig): theory={theory_J['lJ3P']:.6f}, empirical={eig_cubic_mean:.6f}, rel_err={rel_err_j3_sandwich:.2f}%")

    # Compute relative errors vs perp theory (isotropic average)
    print("\n" + "="*70)
    print("Comparison - Projection Method (averaged over all input dimensions):")
    print("="*70)

    rel_err_j1_proj = abs(var_lin_mean - theory_J['lJ1P']) / (abs(theory_J['lJ1P']) + 1e-10) * 100
    rel_err_j3_proj = abs(var_cubic_mean - theory_J['lJ3P']) / (abs(theory_J['lJ3P']) + 1e-10) * 100

    print(f"lJ1 (proj avg): theory={theory_J['lJ1P']:.6f}, empirical={var_lin_mean:.6f}, rel_err={rel_err_j1_proj:.2f}%")
    print(f"lJ3 (proj avg): theory={theory_J['lJ3P']:.6f}, empirical={var_cubic_mean:.6f}, rel_err={rel_err_j3_proj:.2f}%")

    # Check if agreement is good (< 5% relative error)
    tolerance = 5.0
    all_good_proj = all([rel_err_j1_proj < tolerance, rel_err_j3_proj < tolerance])

    print("\n" + "="*70)
    if all_good_proj:
        print("✓ PROJECTION METHOD: Good agreement (< 5% error)")
    else:
        print("⚠ PROJECTION METHOD: Some discrepancies")
    print("="*70 + "\n")

    return {
        "theory": theory_J,
        "empirical_projection": {
            "lJ1_mean": var_lin_mean,
            "lJ1_std": var_lin_std,
            "lJ3_mean": var_cubic_mean,
            "lJ3_std": var_cubic_std,
        },
        "empirical_sandwich": {
            "lJ1_mean": eig_lin_mean,
            "lJ1_std": eig_lin_std,
            "lJ3_mean": eig_cubic_mean,
            "lJ3_std": eig_cubic_std,
        },
        "empirical_W0": {
            "W0_variance": W0_var,
            "theory_lWT": theory_J.get('lWP'),
            "relative_error": rel_err_wt,
        },
        "relative_errors_projection": {
            "lJ1": rel_err_j1_proj,
            "lJ3": rel_err_j3_proj,
        },
        "relative_errors_sandwich": {
            "lJ1": rel_err_j1_sandwich,
            "lJ3": rel_err_j3_sandwich,
        },
        "agreement_projection": all_good_proj,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate FCN2 theory predictions against NNGP initialization')
    parser.add_argument('--P', type=int, required=True, help='Number of data points')
    parser.add_argument('--d', type=int, required=True, help='Input dimension')
    parser.add_argument('--N', type=int, required=True, help='Hidden layer size')
    parser.add_argument('--chi', type=int, default=1, help='Chi parameter (default: 1 for standard scaling)')
    parser.add_argument('--kappa', type=float, required=True, help='Kappa parameter')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--eps', type=float, default=0.03, help='Epsilon parameter for cubic target generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = validate_fcn2_initialization(
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
    results_path = Path("fcn2_nngp_validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
