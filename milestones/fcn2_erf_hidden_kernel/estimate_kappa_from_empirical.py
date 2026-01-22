#!/usr/bin/env python3
"""
Estimate kappa from empirical eigenvalues using self-consistent solver.

Given a model directory and dataset size, this script:
1. Loads the model
2. Computes empirical eigenvalues (lJ1T, lJ1P, lJ3T, lJ3P) via the J kernel
3. Calls the self-consistent Julia solver with these eigenvalues
4. Returns the self-consistent kappa value

Usage:
    python estimate_kappa_from_empirical.py --model-dir <path> --dataset-size <N> [--device cuda:0]
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_config_from_dirname(dirname: str) -> Dict[str, float]:
    """Parse d, P, N, chi, T from directory name pattern: d<d>_P<P>_N<N>_chi<chi>_lr<lr>_T<T>..."""
    import re
    dirname_str = str(Path(dirname).name) if Path(dirname).is_dir() else str(Path(dirname).parent.name)
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi_([\d.]+).*_T_([\d.]+)"

    m = re.search(pattern, dirname_str)
    print("FOUND MATCH:", m)
    if not m:
        raise ValueError(f"Could not parse config from {dirname}")
    
    d = int(m.group(1))
    P = int(m.group(2))
    N = int(m.group(3))
    chi = float(m.group(4))
    T = float(m.group(5))
    
    return {"d": d, "P": P, "N": N, "chi": chi, "T": T}


def load_model(model_dir: Path, config: Dict[str, int], device: torch.device) -> FCN2NetworkActivationGeneric:
    """Load FCN3 model from directory."""
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        model_path = model_dir / "model_final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    d = config["d"]
    P = config["P"]
    N = config["N"]
    chi = config["chi"]
    
    state_dict = torch.load(model_path, map_location=device)
    ens = int(state_dict['W0'].shape[0])
    
    weight_var = (1.0 / d, 1.0 / (N * chi))
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return model


def compute_empirical_eigenvalues(
    model: FCN2NetworkActivationGeneric,
    d: int,
    device: torch.device,
    p_large: int = 10_000
) -> np.ndarray:
    """
    Compute empirical H kernel eigenvalues.
    
    Generates random input data X, computes the H_Kernel (pre-activation kernel),
    and returns all eigenvalues.
    
    Returns:
        Array of all eigenvalues from torch.linalg.eigvalsh
    """
    with torch.no_grad():
        model.to(device)
        model.device = device

        # Generate random input data
        X = torch.randn(p_large, d, device=device)
        
        # Compute H kernel
        K = model.H_Kernel(X, avg_ens=True)  # (P, P)
        
        # Get all eigenvalues
        eigs = torch.linalg.eigvalsh(K)  / p_large # (P,)
        
        return eigs.cpu().numpy()


def call_self_consistent_solver(
    eigenvalues: np.ndarray,
    P: int,
    kappa_bare: float = None,
) -> Optional[float]:
    """
    Call self_consistent_kappa_solver.jl and extract the resulting kappa.
    
    The solver expects:
    - A JSON file with {"eigenvalues": [...], "kappa_bare": <value>}
    - P as a command-line argument
    
    Args:
        eigenvalues: Array of eigenvalues from H_Kernel
        P: Number of samples used to compute kernel
        kappa_bare: Bare kappa value. If None, uses mean of eigenvalues.
    
    Returns:
        kappa value from solver output, or None if solver failed
    """
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
    
    if not julia_script.exists():
        print(f"Warning: Julia solver not found at {julia_script}")
        return None
    

    # Create JSON file with eigenvalues and kappa_bare
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as tf:
        json.dump({
            "eigenvalues": eigenvalues.tolist(),
            "kappa_bare": float(kappa_bare)
        }, tf)
        eig_json = Path(tf.name)
    
    try:
        cmd = [
            "julia",
            str(julia_script),
            str(eig_json),
            str(P)
        ]
        
        print(f"Running Julia command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=300, text=True)
        print(f"Julia solver output:\n{result.stdout}")
        
        # Parse kappa_eff from stdout (format: "kappa_eff = <value>")
        import re
        match = re.search(r"kappa_eff\s*=\s*([-0-9.eE+]+)", result.stdout)
        if match:
            kappa = float(match.group(1))
            print(f"Self-consistent kappa: {kappa}")
            return kappa
        else:
            print("Warning: kappa_eff not found in solver output")
            return None
    except subprocess.TimeoutExpired:
        print("Error: Julia solver timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error calling Julia solver: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return None
    except Exception as e:
        print(f"Error calling Julia solver: {e}")
        return None
    finally:
        try:
            eig_json.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Estimate kappa from empirical eigenvalues using self-consistent solver"
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--dataset-size", type=int, required=True, help="Size of original dataset (not used currently)")
    parser.add_argument("--device", type=str, default=None, help="Device string (cuda:0 or cpu)")
    parser.add_argument("--p-large", type=int, default=10000, help="Size of random dataset for eigenvalue computation")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    model_dir = args.model_dir
    
    print(f"Loading model from {model_dir}")
    config = parse_config_from_dirname(str(model_dir))
    print(f"Parsed config: {config}")
    
    # Extract kappa_bare from T/2
    kappa_bare = config["T"] / 2.0
    print(f"Extracted kappa_bare = T/2 = {config['T']}/2 = {kappa_bare}")
    
    model = load_model(model_dir, config, device)
    print(f"Model loaded: {model}")
    
    print(f"\nComputing empirical H kernel eigenvalues (p_large={args.p_large})...")
    eigenvalues = compute_empirical_eigenvalues(
        model, config["d"], device, p_large=args.p_large
    )
    print(f"Eigenvalues computed:")
    print(f"  Shape: {eigenvalues.shape}")
    print(f"  Max: {eigenvalues.max():.6e}")
    print(f"  Min: {eigenvalues.min():.6e}")
    print(f"  Mean: {eigenvalues.mean():.6e}")
    
    print(f"\nCalling self-consistent kappa solver...")
    kappa = call_self_consistent_solver(
        eigenvalues=eigenvalues,
        P=config["P"],
        kappa_bare=kappa_bare,
    )
    
    if kappa is not None:
        print(f"\n{'='*60}")
        print(f"RESULT: Self-consistent kappa = {kappa:.6f}")
        print(f"{'='*60}")
    else:
        print("\nFailed to compute self-consistent kappa")


if __name__ == "__main__":
    main()
