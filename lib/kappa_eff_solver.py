"""
Effective Ridge (kappa_eff) Solver

Computes the effective ridge parameter using arcsin kernel eigenvalues
and a self-consistent solver via Julia.
"""

import subprocess
import tempfile
import json
import re
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Tuple


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """
    Compute arcsin kernel matrix for inputs X.
    
    Args:
        X: Input tensor of shape (P, d)
    
    Returns:
        Arcsin kernel matrix of shape (P, P)
    """
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def compute_arcsin_eigenvalues(
    d: int,
    num_samples: int = 5000,
    device: Union[str, torch.device] = "cpu"
) -> np.ndarray:
    """
    Compute arcsin kernel eigenvalues for random data.
    
    Args:
        d: Input dimension
        num_samples: Number of random samples to generate
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        Normalized eigenvalues as numpy array of shape (num_samples,)
    """
    device = torch.device(device)
    np.random.seed(0)
    
    # Generate random data and compute kernel
    X = np.random.randn(num_samples, d).astype(np.float32)
    X_torch = torch.from_numpy(X).to(device)
    
    # Compute arcsin kernel
    K = arcsin_kernel(X_torch)
    
    # Get eigenvalues and normalize
    eigvals = torch.linalg.eigvalsh(K).cpu().numpy()
    eigvals_normalized = eigvals / num_samples
    
    return eigvals_normalized


def compute_kappa_eff(
    d: int,
    P: int,
    kappa_bare: float,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    chi: float = 10.0,
    num_samples: int = 5000,
    julia_script: Optional[Path] = None,
    julia_theory_script: Optional[Path] = None,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = False
) -> float:
    """
    Compute effective ridge (kappa_eff) using weight covariance from theory
    and self-consistent solver.
    
    Args:
        d: Input dimension
        P: Number of training samples
        kappa_bare: Bare ridge parameter
        n1: Hidden layer 1 width (default: P)
        n2: Hidden layer 2 width (default: P)
        chi: Regularization weight (default: 10.0)
        num_samples: Number of samples for eigenvalue computation (default: 5000)
        julia_script: Path to self_consistent_kappa_solver.jl script
        julia_theory_script: Path to eos_fcn3erf.jl theory solver script
        device: Device for computation ('cpu' or 'cuda')
        verbose: Print debug information
    
    Returns:
        Effective ridge parameter kappa_eff
    
    Raises:
        RuntimeError: If Julia solvers fail or cannot be located
    """
    if n1 is None:
        n1 = P
    if n2 is None:
        n2 = P
    
    eig_json = None
    theory_json = None
    
    try:
        device = torch.device(device)
        
        # Step 1: Call Julia theory solver to get lWT and lWP
        if julia_theory_script is None:
            julia_theory_script = Path(__file__).parent.parent / "julia_lib" / "eos_fcn3erf.jl"
        
        julia_theory_script = Path(julia_theory_script)
        if not julia_theory_script.exists():
            raise RuntimeError(f"Julia theory script not found at {julia_theory_script}")
        
        if verbose:
            print(f"Running Julia theory solver (d={d}, n1={n1}, n2={n2}, P={P}, chi={chi})...")
        
        # Create temporary output file for theory results
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            theory_json = tf.name
        
        theory_cmd = [
            "julia", str(julia_theory_script),
            f"--d={d}", f"--P={P}",
            f"--n1={n1}", f"--n2={n2}",
            f"--chi={chi}", f"--kappa={kappa_bare}",
            f"--epsilon={1e-3}", f"--to={theory_json}", "--quiet"
        ]
        subprocess.run(theory_cmd, check=True, capture_output=True, timeout=300)
        
        # Extract lWT and lWP from theory results
        with open(theory_json, 'r') as f:
            theory_results = json.load(f)
        
        lWT = theory_results.get("target", {}).get("lWT", 1.0)
        lWP = theory_results.get("target", {}).get("lWP", 1.0)
        
        if verbose:
            print(f"  Theory results: lWT={lWT:.6f}, lWP={lWP:.6f}")
        
        # Step 2: Compute weight covariance matrix Sigma and spectral properties
        if verbose:
            print(f"Computing weight covariance structure with arcsin kernel...")
        
        np.random.seed(0)
        X_np = np.random.randn(num_samples, d).astype(np.float32)
        X = torch.from_numpy(X_np).to(device)
        
        # Build diagonal weight covariance matrix: Sigma = diag(lWT, lWT, ..., lWP, lWP, ...)
        # For simplicity, we'll use a diagonal approximation with mixed eigenvalues
        sigma_diag = torch.cat([
            torch.ones(d, device=device) * lWT,  # Input->hidden1 weights contribute lWT
            torch.ones(1, device=device) * lWP   # Remainder
        ])[:d]
        
        Sigma = torch.diag(sigma_diag).to(device)
        
        # Compute refined spectral properties: X^T Sigma X
        XSX = torch.einsum('ui, ij, vj -> uv', X, Sigma, X)
        
        # Get eigenvalues of the refined kernel
        eigvals = torch.linalg.eigvalsh(XSX).cpu().numpy()
        eigvals_normalized = eigvals / num_samples
        
        if verbose:
            print(f"  Refined eigenvalues (top 5): {eigvals_normalized[-5:][::-1]}")
        
        # Step 3: Run self-consistent kappa solver with refined eigenvalues
        if julia_script is None:
            julia_script = Path(__file__).parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
        
        julia_script = Path(julia_script)
        if not julia_script.exists():
            raise RuntimeError(f"Julia solver script not found at {julia_script}")
        
        # Create temporary JSON file with refined eigenvalues and kappa_bare
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf:
            eig_json = tf.name
            json.dump({
                "eigenvalues": eigvals_normalized.tolist(),
                "kappa_bare": kappa_bare
            }, tf)
        
        if verbose:
            print(f"Running self-consistent solver with P={P}...")
        
        # Run Julia self-consistent solver
        sc_cmd = [
            "julia", str(julia_script),
            eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True, stderr=subprocess.STDOUT)
        
        if verbose:
            print("Solver output:", sc_out)
        
        # Extract kappa_eff from solver output
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
            if verbose:
                print(f"Computed kappa_eff = {kappa_eff:.6f}")
            return kappa_eff
        else:
            raise RuntimeError("Could not extract kappa_eff from Julia output")
    
    except Exception as e:
        raise RuntimeError(f"kappa_eff computation failed: {e}")
    
    finally:
        # Clean up temporary files
        for temp_file in [eig_json, theory_json]:
            try:
                if temp_file and Path(temp_file).exists():
                    Path(temp_file).unlink()
            except:
                pass


def compute_kappa_eff_batch(
    parameters: list[dict],
    julia_script: Optional[Path] = None,
    julia_theory_script: Optional[Path] = None,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = False
) -> list[Tuple[dict, float]]:
    """
    Compute kappa_eff for multiple parameter sets.
    
    Args:
        parameters: List of dicts with keys 'd', 'P', 'kappa_bare', 
                   optionally 'n1', 'n2', 'chi'
        julia_script: Path to Julia solver script
        julia_theory_script: Path to Julia theory script
        device: Device for computation
        verbose: Print debug information
    
    Returns:
        List of tuples (input_params, kappa_eff)
    """
    results = []
    for i, params in enumerate(parameters):
        if verbose:
            print(f"\n[{i+1}/{len(parameters)}] Computing kappa_eff for d={params['d']}, P={params['P']}, kappa_bare={params['kappa_bare']}")
        
        kappa_eff = compute_kappa_eff(
            d=params['d'],
            P=params['P'],
            kappa_bare=params['kappa_bare'],
            n1=params.get('n1'),
            n2=params.get('n2'),
            chi=params.get('chi', 10.0),
            julia_script=julia_script,
            julia_theory_script=julia_theory_script,
            device=device,
            verbose=verbose
        )
        results.append((params, kappa_eff))
    
    return results
