#!/usr/bin/env python3
"""Compute the full H_eig spectrum for a saved FCN3 model."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric


def load_config(config_path: Path) -> dict:
    """Load model configuration from config.json."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_model(model_path: Path, config: dict, device: torch.device) -> FCN3NetworkActivationGeneric:
    """Load model from saved state_dict."""
    d = config["d"]
    N = config["N"]
    P = config["P"]
    ens = config.get("ens", 1)
    
    # Load state dict
    sd = torch.load(model_path, map_location="cpu")
    
    # Get dimensions from state dict
    n1 = sd['W0'].shape[-2]
    n2 = sd['W1'].shape[-2]
    
    # Create model
    model = FCN3NetworkActivationGeneric(
        d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2))
    ).to(device)
    
    # Load state dict
    model.load_state_dict({
        k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v 
        for k, v in sd.items()
    }, strict=False)
    
    model.eval()
    return model


def compute_h_eig_spectrum(
    model: FCN3NetworkActivationGeneric,
    d: int,
    n_samples: int = 10000,
    device: torch.device = None
) -> np.ndarray:
    """Compute H_eig spectrum on random data.
    
    Args:
        model: The FCN3 model
        d: Input dimension
        n_samples: Number of random samples to use
        device: Device to compute on
        
    Returns:
        Array of eigenvalues (descending order)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)
    with torch.no_grad():
        X = torch.randn(n_samples, d, device=device)
        eigs = model.H_eig(X, X)
        
    # Convert to numpy and sort descending
    eigs_np = eigs.detach().cpu().numpy()
    eigs_np = np.sort(eigs_np)[::-1]
    print(eigs_np[:10])  # Print top 10 eigenvalues for sanity check
    return eigs_np


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute H_eig spectrum for a saved FCN3 model"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory (e.g., p_scan_erf_results/d50_P49_N800_chi10_kappa0.1/seed0)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of random samples for spectrum computation"
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of the spectrum"
    )
    parser.add_argument(
        "--save-eigs",
        action="store_true",
        help="Save eigenvalues to JSON"
    )
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {model_dir}")
        sys.exit(1)
    
    config = load_config(config_path)
    print(f"Config: d={config['d']}, P={config['P']}, N={config['N']}, "
          f"chi={config['chi']}, kappa={config['kappa']}, ens={config.get('ens', 1)}")
    
    # Find model file
    model_path = model_dir / "model_final.pt"
    if not model_path.exists():
        model_path = model_dir / "model.pt"
    
    if not model_path.exists():
        print(f"Error: model file not found in {model_dir}")
        sys.exit(1)
    
    print(f"Loading model from {model_path.name}...")
    
    # Load model
    device = torch.device(args.device)
    model = load_model(model_path, config, device)
    print("Model loaded successfully")
    
    # Compute spectrum
    print(f"Computing H_eig spectrum with {args.n_samples} samples...")
    eigs = compute_h_eig_spectrum(model, config['d'], args.n_samples, device)
    
    print(f"\nSpectrum statistics:")
    print(f"  Number of eigenvalues: {len(eigs)}")
    print(f"  Max eigenvalue: {eigs[0]:.6e}")
    print(f"  Min eigenvalue: {eigs[-1]:.6e}")
    print(f"  Mean eigenvalue: {np.mean(eigs):.6e}")
    print(f"  Median eigenvalue: {np.median(eigs):.6e}")
    print(f"  Std dev: {np.std(eigs):.6e}")
    
    # Save eigenvalues if requested
    if args.save_eigs:
        output_file = model_dir / "h_eig_spectrum.json"
        with open(output_file, "w") as f:
            json.dump({"eigenvalues": eigs.tolist(), "n_samples": args.n_samples}, f)
        print(f"\nEigenvalues saved to {output_file}")
    
    # Plot if requested
    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(range(len(eigs)), eigs, 'b-', linewidth=2, label="H_eig spectrum")
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue (log scale)")
        ax.set_title(
            f"H_eig Spectrum: d={config['d']}, P={config['P']}, N={config['N']}, "
            f"χ={config['chi']}, κ={config['kappa']}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_file = model_dir / "h_eig_spectrum_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
