#!/usr/bin/env python3
"""
Compare linear and erf FCN2 network outputs and eigenvalues.

Loads trained linear and erf models from their run directories,
compares eigenvalue predictions from Julia solvers (FCS2Linear and FCS2Erf),
and plots side-by-side output comparisons.
"""

import argparse
import json
import sys
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

# Try to import Julia
try:
    import juliacall
    from juliacall import Main as jl
    HAS_JULIA = True
except Exception:
    HAS_JULIA = False


def load_model(run_dir: Path, activation: str, device: torch.device):
    """Load model and config from run directory."""
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
        raise FileNotFoundError("config.json or checkpoint not found")

    d = int(config["d"])
    P = int(config["P"])
    N = int(config["N"])
    ens = int(config.get("ens", 5))

    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation=activation,
        weight_initialization_variance=(1/d, 1/N),
        device=device,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, config


def compute_julia_eigs(d: int, n1: int, P: int, activation: str):
    """Compute theoretical eigenvalues from Julia solver."""
    if not HAS_JULIA:
        return None

    try:
        if activation == "linear":
            jl.include(str(Path(__file__).parent.parent.parent / "julia_lib" / "FCS2Linear.jl"))
            solver_module = jl.FCS2Linear
            b = 0.0  # no erf constant for linear
        else:  # erf
            jl.include(str(Path(__file__).parent.parent.parent / "julia_lib" / "FCS2Erf.jl"))
            solver_module = jl.FCS2Erf
            b = 4.0 / (3.0 * math.pi)

        chi = float(n1)
        kappa = 1.0 / chi

        # Target (delta=1)
        init = juliacall.convert(jl.Vector[jl.Float64], [1.0 / d])
        sol_t = solver_module.nlsolve_solver(
            init,
            chi=chi, d=float(d), kappa=kappa, delta=1.0, n1=float(n1),
            b=b, P=float(P), lr=1e-3, max_iter=50_000, tol=1e-8,
            verbose=False, anneal=True,
        )
        lJ_t = float(sol_t[1]) if sol_t is not None else None

        # Perp (delta=0)
        sol_p = solver_module.nlsolve_solver(
            init,
            chi=chi, d=float(d), kappa=1.0, delta=0.0, n1=float(n1),
            b=b, P=float(P), lr=1e-3, max_iter=50_000, tol=1e-8,
            verbose=False, anneal=True,
        )
        lJ_p = float(sol_p[1]) if sol_p is not None else None

        return {"target": lJ_t, "perp": lJ_p}

    except Exception as e:
        print(f"Warning: Julia solver failed ({activation}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare linear and erf FCN2 networks")
    parser.add_argument("--d", type=int, default=10, help="Input dimension")
    parser.add_argument("--P", type=int, default=50, help="Number of samples")
    parser.add_argument("--N", type=int, default=200, help="Hidden layer width")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    linear_dir = base_dir / f"linear_d{args.d}_P{args.P}_N{args.N}"
    erf_dir = base_dir / f"d{args.d}_P{args.P}_N{args.N}"

    if not linear_dir.exists():
        print(f"Error: Linear directory not found: {linear_dir}")
        return
    if not erf_dir.exists():
        print(f"Error: Erf directory not found: {erf_dir}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\nComparing Linear vs Erf FCN2 Networks")
    print(f"  d={args.d}, P={args.P}, N={args.N}")
    print()

    # Load models
    print("Loading models...")
    model_linear, config_linear = load_model(linear_dir, "linear", device)
    model_erf, config_erf = load_model(erf_dir, "erf", device)

    # Generate test data
    torch.manual_seed(42)
    X = torch.randn(args.P, args.d, device=device)
    Y = X[:, 0]

    # Get outputs
    with torch.no_grad():
        out_linear = model_linear(X).mean(dim=1)  # (P,) mean over ensembles
        out_erf = model_erf(X).mean(dim=1)        # (P,) mean over ensembles

    # Compute Julia eigenvalues if available
    print("\nComputing theoretical eigenvalues from Julia solvers...")
    eigs_linear = compute_julia_eigs(args.d, args.N, args.P, "linear")
    eigs_erf = compute_julia_eigs(args.d, args.N, args.P, "erf")

    if eigs_linear:
        print(f"Linear (julia):  target={eigs_linear.get('target', 'N/A'):.6e}, "
              f"perp={eigs_linear.get('perp', 'N/A'):.6e}")
    if eigs_erf:
        print(f"Erf (julia):     target={eigs_erf.get('target', 'N/A'):.6e}, "
              f"perp={eigs_erf.get('perp', 'N/A'):.6e}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Linear network
    axes[0, 0].scatter(Y.cpu().numpy(), out_linear.cpu().numpy(), alpha=0.5, s=20)
    axes[0, 0].plot([-3, 3], [-3, 3], "k--", lw=1)
    axes[0, 0].set_xlabel("Target Y")
    axes[0, 0].set_ylabel("Linear Network Output")
    axes[0, 0].set_title("Linear Network")
    axes[0, 0].grid(alpha=0.3)

    # Erf network
    axes[0, 1].scatter(Y.cpu().numpy(), out_erf.cpu().numpy(), alpha=0.5, s=20, color="orange")
    axes[0, 1].plot([-3, 3], [-3, 3], "k--", lw=1)
    axes[0, 1].set_xlabel("Target Y")
    axes[0, 1].set_ylabel("Erf Network Output")
    axes[0, 1].set_title("Erf Network")
    axes[0, 1].grid(alpha=0.3)

    # Difference
    diff = out_linear.cpu().numpy() - out_erf.cpu().numpy()
    axes[1, 0].hist(diff, bins=30, alpha=0.8)
    axes[1, 0].set_xlabel("Linear - Erf")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Difference Distribution (mean={diff.mean():.6e}, std={diff.std():.6e})")
    axes[1, 0].grid(alpha=0.3)

    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    Network Comparison
    d={args.d}, P={args.P}, N={args.N}
    
    Linear Output Stats:
      Mean: {out_linear.mean().item():.6f}
      Std:  {out_linear.std().item():.6f}
    
    Erf Output Stats:
      Mean: {out_erf.mean().item():.6f}
      Std:  {out_erf.std().item():.6f}
    
    Difference:
      Mean: {diff.mean():.6e}
      Std:  {diff.std():.6e}
      Max:  {np.abs(diff).max():.6e}
    """

    if eigs_linear:
        summary_text += f"\n    Linear Julia Eigs:\n"
        if eigs_linear.get('target'):
            summary_text += f"      target: {eigs_linear['target']:.6e}\n"
        if eigs_linear.get('perp'):
            summary_text += f"      perp:   {eigs_linear['perp']:.6e}\n"

    if eigs_erf:
        summary_text += f"\n    Erf Julia Eigs:\n"
        if eigs_erf.get('target'):
            summary_text += f"      target: {eigs_erf['target']:.6e}\n"
        if eigs_erf.get('perp'):
            summary_text += f"      perp:   {eigs_erf['perp']:.6e}\n"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
                   verticalalignment='center', transform=axes[1, 1].transAxes)

    fig.tight_layout()
    output_path = base_dir / "linear_vs_erf_comparison.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot to {output_path}")
    plt.close(fig)

    # Save results JSON
    results = {
        "d": args.d,
        "P": args.P,
        "N": args.N,
        "linear_output_mean": float(out_linear.mean().item()),
        "linear_output_std": float(out_linear.std().item()),
        "erf_output_mean": float(out_erf.mean().item()),
        "erf_output_std": float(out_erf.std().item()),
        "difference_mean": float(diff.mean()),
        "difference_std": float(diff.std()),
        "difference_max": float(np.abs(diff).max()),
    }

    if eigs_linear:
        results["julia_linear"] = eigs_linear
    if eigs_erf:
        results["julia_erf"] = eigs_erf

    results_path = base_dir / "linear_vs_erf_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
