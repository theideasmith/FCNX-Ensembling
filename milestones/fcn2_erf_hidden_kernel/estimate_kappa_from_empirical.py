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
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_config_from_dirname(dirname: str) -> Dict[str, float]:
    """Parse d, P, N, chi from directory name pattern: d<d>_P<P>_N<N>_chi<chi>..."""
    import re
    dirname_str = str(Path(dirname).name) if Path(dirname).is_dir() else str(Path(dirname).parent.name)
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi([\d.]+)"
    m = re.search(pattern, dirname_str)
    if not m:
        raise ValueError(f"Could not parse config from {dirname}")
    
    d = int(m.group(1))
    P = int(m.group(2))
    N = int(m.group(3))
    chi = float(m.group(4))
    
    return {"d": d, "P": P, "N": N, "chi": chi}


def load_model(model_dir: Path, config: Dict[str, int], device: torch.device) -> FCN3NetworkActivationGeneric:
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
    
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return model


def j_random_QB_activation_generic(model, X, k=2000, p=10):
    """Low-rank QB approximation for J kernel using h0 activations."""
    with torch.no_grad():
        l = k + p
        h0 = model.h0_activation(X)  # (N, ens, n1)
        Omega = torch.randn((X.shape[0], l), device=model.device, dtype=h0.dtype)

        res = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        chunk_size = 4096
        N = X.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)

        Z = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        return Q, Z


def compute_empirical_j_eigenvalues(
    model: FCN3NetworkActivationGeneric,
    d: int,
    device: torch.device,
    p_large: int = 10_000
) -> Tuple[float, float, float, float]:
    """
    Compute empirical J kernel eigenvalues (lJ1T, lJ1P, lJ3T, lJ3P).
    
    Returns:
        (lJ1T, lJ1P, lJ3T, lJ3P)
    """
    with torch.no_grad():
        model.to(device)
        model.device = device

        X = torch.randn(p_large, d, device=device)
        Y1 = X
        Y3 = (X ** 3 - 3.0 * X) / 6.0**0.5

        # Low-rank approximation QB for J kernel
        Q, Z = j_random_QB_activation_generic(model, X, k=9000, p=10)
        Ut, _S, V = torch.linalg.svd(Z.T)
        m, n = Z.shape[1], Z.shape[0]
        k_eff = min(m, n)
        Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
        Sigma[:k_eff, :k_eff] = torch.diag(_S[:k_eff])
        U = torch.matmul(Q, Ut)

        # Left eigenvalues for Y1 via J_eig
        Y1_norm = Y1 / torch.norm(Y1, dim=0)
        left_eigenvaluesY1 = model.J_eig(X, Y1_norm)

        # Left eigenvalues for Y3 via projection through U, Sigma
        Y3_norm = Y3 / torch.norm(Y3, dim=0)
        proj = (Y3_norm.t() @ U)
        left_Y3_mat = proj @ torch.diag(_S[:k_eff]) @ (U.t() @ Y3_norm)
        left_eigenvaluesY3 = left_Y3_mat.diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]

        # Extract target (first) and perpendicular (rest) eigenvalues
        # For Y1: target is first, perpendicular are indices 1:d
        lJ1T = float(left_eigenvaluesY1[0].cpu().numpy())
        lJ1P = float(left_eigenvaluesY1[1].cpu().numpy()) if len(left_eigenvaluesY1) > 1 else lJ1T
        
        # For Y3: target is at index d, perpendicular are after that
        lJ3T = float(left_eigenvaluesY3[d].cpu().numpy()) if len(left_eigenvaluesY3) > d else float('nan')
        lJ3P = float(left_eigenvaluesY3[d + 1].cpu().numpy()) if len(left_eigenvaluesY3) > d + 1 else lJ3T

        return lJ1T, lJ1P, lJ3T, lJ3P


def call_self_consistent_solver(
    d: int,
    P: int,
    N: int,
    chi: float,
    lJ1T: float,
    lJ1P: float,
    lJ3T: float,
    lJ3P: float,
    epsilon: float = 0.03,
) -> Optional[float]:
    """
    Call self_consistent_kappa_solver.jl and extract the resulting kappa.
    
    Returns:
        kappa value from solver output, or None if solver failed
    """
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
    
    if not julia_script.exists():
        print(f"Warning: Julia solver not found at {julia_script}")
        return None
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = Path(tf.name)
    
    cmd = [
        "julia",
        str(julia_script),
        f"--d={d}",
        f"--P={P}",
        f"--n1={N}",
        f"--n2={N}",
        f"--chi={chi}",
        f"--epsilon={epsilon}",
        f"--lJ1T={lJ1T}",
        f"--lJ1P={lJ1P}",
        f"--lJ3T={lJ3T}",
        f"--lJ3P={lJ3P}",
        f"--to={out_path}",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=300, text=True)
        print(f"Julia solver output:\n{result.stdout}")
        
        with open(out_path, "r") as f:
            output = json.load(f)
        
        kappa = output.get("kappa")
        if kappa is not None:
            print(f"Self-consistent kappa: {kappa}")
            return float(kappa)
        else:
            print("Warning: kappa not found in solver output")
            return None
    except subprocess.TimeoutExpired:
        print("Error: Julia solver timed out")
        return None
    except Exception as e:
        print(f"Error calling Julia solver: {e}")
        return None
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Estimate kappa from empirical eigenvalues using self-consistent solver"
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--dataset-size", type=int, required=True, help="Size of original dataset (not used currently)")
    parser.add_argument("--device", type=str, default=None, help="Device string (cuda:0 or cpu)")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Cubic coupling parameter")
    parser.add_argument("--p-large", type=int, default=10000, help="Size of random dataset for eigenvalue computation")
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    model_dir = args.model_dir
    
    print(f"Loading model from {model_dir}")
    config = parse_config_from_dirname(str(model_dir))
    print(f"Parsed config: {config}")
    
    model = load_model(model_dir, config, device)
    print(f"Model loaded: {model}")
    
    print(f"\nComputing empirical J eigenvalues (p_large={args.p_large})...")
    lJ1T, lJ1P, lJ3T, lJ3P = compute_empirical_j_eigenvalues(
        model, config["d"], device, p_large=args.p_large
    )
    print(f"Empirical eigenvalues:")
    print(f"  lJ1T = {lJ1T:.6e}")
    print(f"  lJ1P = {lJ1P:.6e}")
    print(f"  lJ3T = {lJ3T:.6e}")
    print(f"  lJ3P = {lJ3P:.6e}")
    
    print(f"\nCalling self-consistent kappa solver...")
    kappa = call_self_consistent_solver(
        d=config["d"],
        P=config["P"],
        N=config["N"],
        chi=config["chi"],
        lJ1T=lJ1T,
        lJ1P=lJ1P,
        lJ3T=lJ3T,
        lJ3P=lJ3P,
        epsilon=args.epsilon,
    )
    
    if kappa is not None:
        print(f"\n{'='*60}")
        print(f"RESULT: Self-consistent kappa = {kappa:.6f}")
        print(f"{'='*60}")
    else:
        print("\nFailed to compute self-consistent kappa")


if __name__ == "__main__":
    main()
