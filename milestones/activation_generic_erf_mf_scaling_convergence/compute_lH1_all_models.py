#!/usr/bin/env python3
"""
Compute lH1 eigenvalues for all models in the results folder.

This script iterates through all model directories and computes:
- Empirical lH1 eigenvalues using the H_eig_random_svd method
- Theoretical lH1 eigenvalues by solving the EOS with kappa_eff
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))

from FCN3Network import FCN3NetworkActivationGeneric
from kappa_eff_solver import compute_kappa_eff


RESULTS_DIR = Path(__file__).parent / "results"
print(RESULTS_DIR)
JULIA_SCRIPT = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"


def compute_theoretical_lH1(
    d: int,
    P: int,
    n1: int,
    n2: int,
    chi: float,
    kappa_eff: float,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Compute theoretical eigenvalues by solving EOS with kappa_eff.
    
    Args:
        d: Input dimension
        P: Dataset size
        n1: Hidden layer 1 width
        n2: Hidden layer 2 width
        chi: Regularization weight
        kappa_eff: Effective ridge parameter
        verbose: Verbose output
    
    Returns:
        Dict with lH1T (largest) and lH3T (dth eigenvalue) or None if computation fails
    """
    if not JULIA_SCRIPT.exists():
        if verbose:
            print(f"    Julia script not found: {JULIA_SCRIPT}")
        return None
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as tf:
        output_file = tf.name
    
    try:
        cmd = [
            "julia", str(JULIA_SCRIPT),
            f"--d={d}", f"--P={P}",
            f"--n1={n1}", f"--n2={n2}",
            f"--chi={chi}", f"--kappa={kappa_eff}",
            f"--epsilon={1e-3}", f"--to={output_file}", "--quiet"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        target = results.get("target", {})
        lH1T = float(target.get("lH1T", np.nan))
        lH3T = float(target.get("lH3T", np.nan))
        
        return {"lH1T": lH1T, "lH3T": lH3T}
    
    except Exception as e:
        if verbose:
            print(f"    Error computing theoretical eigenvalues: {e}")
        return None
    
    finally:
        try:
            Path(output_file).unlink()
        except:
            pass
JULIA_SCRIPT = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"


def compute_lH1_for_model(
    model_dir: Path,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False
) -> Optional[Dict]:
    """
    Compute lH1 eigenvalue for a single model.
    
    Args:
        model_dir: Path to model directory
        device: Device for computation
        verbose: Verbose output
    
    Returns:
        Dict with results or None if computation fails
    """
    config_path = model_dir / "config.json"
    model_path = model_dir / "model_final.pt"
    
    if not config_path.exists():
        if verbose:
            print(f"  Skipping {model_dir.name}: no config.json")
        return None
    
    if not model_path.exists():
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            if verbose:
                print(f"  Skipping {model_dir.name}: no model file")
            return None
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        
        # Load model state dict
        sd = torch.load(model_path, map_location="cpu")
        
        # Extract dimensions
        d = sd['W0'].shape[-1]
        n1 = sd['W0'].shape[-2]
        n2 = sd['W1'].shape[-2]
        P = cfg["P"]
        ens = sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1]
        
        # Initialize and load model
        model = FCN3NetworkActivationGeneric(
            d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
            weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2)),
            device=device
        )
        model.load_state_dict(
            {k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v for k, v in sd.items()},
            strict=False
        )
        model = model.to(device)
        model.eval()
        
        # Compute kappa_eff
        kappa_eff = compute_kappa_eff(
            d=d,
            P=P,
            kappa_bare=cfg['kappa'],
            n1=n1,
            n2=n2,
            chi=cfg.get('chi', 10.0),
            device=device,
            verbose=False
        )
        
        # Generate random data and compute eigenvalues
        torch.manual_seed(0)
        X = torch.randn(3000, d, device=device)
        
        # Compute H eigenvalues
        eigs = model.H_eig_random_svd(X, k=400)
        
        # Compute empirical lH1 (1st eigenvalue)
        lH1_emp = eigs[0]#float((eigs[0] / (eigs[0] + kappa_eff / P)).detach().cpu().numpy())
        
        # Compute empirical lH3 (dth eigenvalue)
        lH3_emp = eigs[d]#float((eigs[d] / (eigs[d] + kappa_eff / P)).detach().cpu().numpy())
        
        # Compute theoretical eigenvalues
        theo_eigs = compute_theoretical_lH1(
            d=d, P=P, n1=n1, n2=n2,
            chi=cfg.get('chi'), kappa_eff=kappa_eff,
            verbose=verbose
        )
        
        result = {
            "model_dir": str(model_dir),
            "d": d,
            "P": P,
            "N": n1,
            "chi": cfg.get('chi'),
            "kappa_bare": cfg['kappa'],
            "kappa_eff": kappa_eff,
            "eig_H_0": float(eigs[0].detach().cpu().numpy()),
            "eig_H_d": float(eigs[d].detach().cpu().numpy()),
            "lH1_emp": lH1_emp,
            "lH3_emp": lH3_emp,
            "lH1T_theory": theo_eigs.get('lH1T') if theo_eigs else None,
            "lH3T_theory": theo_eigs.get('lH3T') if theo_eigs else None,
        }
        
        if verbose:
            lh1t = theo_eigs.get('lH1T') if theo_eigs else None
            lh3t = theo_eigs.get('lH3T') if theo_eigs else None
            print(f"  {model_dir.name}: lH1_emp={lH1_emp:.6f}, lH3_emp={lH3_emp:.6f}, lH1T={lh1t:.6f if lh1t else 'N/A'}, lH3T={lh3t:.6f if lh3t else 'N/A'}")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"  Error computing lH1 for {model_dir.name}: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute lH1 eigenvalues for all models")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if args.verbose:
        print(f"Using device: {device}")
    
    # Find all model directories
    model_dirs = sorted(RESULTS_DIR.glob("d*/*seed*"))
    
    if not model_dirs:
        print(f"No model directories found in {RESULTS_DIR}")
        return 1
    
    if args.verbose:
        print(f"Found {len(model_dirs)} model directories")
    
    # Compute lH1 for each model
    results = []
    for i, model_dir in enumerate(model_dirs):
        if args.verbose:
            print(f"[{i+1}/{len(model_dirs)}] {model_dir.relative_to(RESULTS_DIR)}")
        
        result = compute_lH1_for_model(model_dir, device=device, verbose=args.verbose)
        if result:
            results.append(result)
    
    # Group by configuration for summary
    print("\n" + "="*120)
    print("SUMMARY OF lH1 EIGENVALUES (EMPIRICAL vs THEORY)")
    print("="*120)
    
    configs = defaultdict(list)
    for r in results:
        key = (r['d'], r['P'], r['N'], r['chi'], r['kappa_bare'])
        configs[key].append(r)
    
    for (d, P, N, chi, kappa_bare), group in sorted(configs.items()):
        lH1_emp_vals = [r['lH1_emp'] for r in group]
        lH3_emp_vals = [r['lH3_emp'] for r in group]
        lH1T_theory_vals = [r['lH1T_theory'] for r in group if r['lH1T_theory'] is not None]
        lH3T_theory_vals = [r['lH3T_theory'] for r in group if r['lH3T_theory'] is not None]
        
        print(f"d={d}, P={P}, N={N}, chi={chi}, kappa_bare={kappa_bare:.4f}")
        print(f"  Empirical lH1 (1st):    mean={np.mean(lH1_emp_vals):.6f}, std={np.std(lH1_emp_vals):.6f}, n={len(lH1_emp_vals)}")
        if lH1T_theory_vals:
            print(f"  Theory lH1T (1st):      mean={np.mean(lH1T_theory_vals):.6f}, std={np.std(lH1T_theory_vals):.6f}, n={len(lH1T_theory_vals)}")
        else:
            print(f"  Theory lH1T (1st):      no results")
        print(f"  Empirical lH3 (d-th):   mean={np.mean(lH3_emp_vals):.6f}, std={np.std(lH3_emp_vals):.6f}, n={len(lH3_emp_vals)}")
        if lH3T_theory_vals:
            print(f"  Theory lH3T (d-th):     mean={np.mean(lH3T_theory_vals):.6f}, std={np.std(lH3T_theory_vals):.6f}, n={len(lH3T_theory_vals)}")
        else:
            print(f"  Theory lH3T (d-th):     no results")
    
    print("="*120)
    print(f"\nTotal models computed: {len(results)}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
