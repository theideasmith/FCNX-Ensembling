#!/usr/bin/env python3
"""
Compute H eigenvalues for FCN2 using random SVD from a run directory.

Usage:
    python compute_h_eig_random_svd.py --run-dir <path> [--k 100] [--p 25] [--num-samples 3000]
"""

import argparse
import json
from pathlib import Path
import sys
import torch
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

# Julia solver (FCS2Erf) for FCN2 target/perp eigenvalues
try:
    import juliacall
    from juliacall import Main as jl
    jl.include(str(Path(__file__).parent.parent.parent / "julia_lib" / "FCS2Erf.jl"))
    HAS_JULIA = True
except Exception as _e:  # defer failure; report later
    HAS_JULIA = False
    _JULIA_ERR = _e


def load_model(run_dir: Path, device: torch.device):
    """Load model and config from a run directory."""
    checkpoint_path = run_dir / "checkpoint.pt"
    model_path = run_dir / "model.pt"
    config_path = run_dir / "config.json"

    config = None
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    state_dict = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if config is None:
            config = checkpoint.get("config", None)
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location=device)

    if config is None:
        raise FileNotFoundError("config.json or checkpoint config not found in run directory")

    d = int(config["d"])
    P = int(config["P"])
    N = int(config["N"])
    chi = float(config.get("chi", 1))
    ens = int(config.get("ens", 50))

    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/(N*chi)),
        device=device,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, config


def compute_fcs2_eigs(d: int, n1: int, P: int, kappa: float = 1.0, chi: float = 1.0):
    if not HAS_JULIA:
        raise RuntimeError(f"Julia unavailable: {_JULIA_ERR}")

    chi = float(chi)
    b = 4.0 / (3.0 * math.pi)

    def solve(delta: float, kappa_val: float):
        i0 = julia_init_guess(d)
        sol = jl.FCS2Erf.nlsolve_solver(
            i0,
            chi=chi,
            d=float(d),
            kappa=kappa_val,
            delta=delta,
            n1=float(n1),
            b=b,
            P=float(P),
            lr=1e-3,
            max_iter=5_000,
            tol=1e-8,
            verbose=False,
            anneal=True,
            anneal_steps=50_000
        )
        lJ = float(sol[0])
        return {"lJ": lJ}

    delta1 = solve(delta=1.0, kappa_val=kappa)
    delta0 = solve(delta=0.0, kappa_val=kappa)
    return {"delta1": delta1, "delta0": delta0}


def julia_init_guess(d: int):
    # Simple positive init to help convergence
    return juliacall.convert(jl.Vector[jl.Float64], [1.0 / d, 1.0 / d, 1.0/d])


def main():
    parser = argparse.ArgumentParser(description="Compute H eigenvalues using random SVD")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:X)")
    parser.add_argument("--k", type=int, default=100, help="Number of eigenvalues to compute")
    parser.add_argument("--p", type=int, default=25, help="Oversampling parameter")
    parser.add_argument("--num-samples", type=int, default=3000, help="Number of samples for eigenvalue computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-julia", action="store_true", help="Skip Julia FCS2 solver")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")

    # Load model
    model, config = load_model(run_dir, device)  # CPU for stability
    print("Model configuration: ")
    for k, v in config.items():
        print(f"  {k}: {v}")
    d = int(config["d"])

    print(f"\nComputing H eigenvalues for {run_dir.name}")
    print(f"  d={d}, N={config['N']}, P={config['P']}")
    print(f"  Using random SVD: k={args.k}, p={args.p}, num_samples={args.num_samples}")

    # Generate test data
    torch.manual_seed(args.seed)
    X = torch.randn(args.num_samples, d, device=device)

    # Compute eigenvalues
    print("\nComputing eigenvalues...")
    with torch.no_grad():
        eigenvalues = model.H_eig(X, X)

    julia_eigs = None
    if not args.no_julia:
        if not HAS_JULIA:
            print(f"\nJulia solver unavailable: {_JULIA_ERR}")
        else:
            try:
                print("\nComputing Julia FCS2 eigenvalues (delta=1.0 target, delta=0.0 perp)...")
                julia_eigs = compute_fcs2_eigs(d=d, n1=int(config["N"]), P=int(config["P"]), kappa=float(config.get("temperature", 1.0))/2.0, chi=config.get("chi", 1.0))
                print(f"  delta=1.0: lJ={julia_eigs['delta1']['lJ']:.6e}")
                print(f"  delta=0.0: lJ={julia_eigs['delta0']['lJ']:.6e}")
            except Exception as _e:
                print(f"  Julia FCS2 solver failed: {_e}")

    print(f"\nEigenvalues (top {args.k}):")
    print(f"  Mean: {eigenvalues.mean().item():.6e}")
    print(f"  Max:  {eigenvalues.max().item():.6e}")
    print(f"  Min:  {eigenvalues.min().item():.6e}")
    print(f"  Std:  {eigenvalues.std().item():.6e}")

    # Save results
    results = {
        "k": args.k,
        "p": args.p,
        "num_samples": args.num_samples,
        "eigenvalues": eigenvalues.cpu().numpy().tolist(),
        "mean": float(eigenvalues.mean().item()),
        "max": float(eigenvalues.max().item()),
        "min": float(eigenvalues.min().item()),
        "std": float(eigenvalues.std().item()),
        "julia_fcs2": julia_eigs,
    }
    
    output_path = run_dir / f"h_eig_random_svd_k{args.k}_p{args.p}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
