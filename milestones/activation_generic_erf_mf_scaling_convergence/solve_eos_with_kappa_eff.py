#!/usr/bin/env python3
"""
Compute kappa_eff from model folder and solve EOS with effective ridge.

This script:
1. Extracts parameters from the model folder name
2. Computes kappa_eff using the kappa_eff solver
3. Solves the equation of state (EOS) using the computed kappa_eff
4. Outputs results to JSON

Usage:
    python solve_eos_with_kappa_eff.py /path/to/d50_P499_N800_chi10_kappa0.1/seed0
    python solve_eos_with_kappa_eff.py /path/to/model/folder --output results.json --verbose
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from kappa_eff_solver import compute_kappa_eff


def extract_params_from_path(folder_path: Path) -> Dict[str, float]:
    """
    Extract parameters from folder name and config.json.
    
    Supports folder names like:
    - d50_P499_N800_chi10_kappa0.1
    - d50_P499_N800_chi10_kappa0.1/seed0
    
    Args:
        folder_path: Path to model folder
    
    Returns:
        Dict with keys: d, P, N, chi, kappa, seed (if available)
    
    Raises:
        ValueError: If cannot extract parameters
    """
    params = {}
    
    # Try to load from config.json first
    config_path = folder_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                params.update({
                    'd': int(config.get('d')),
                    'P': int(config.get('P')),
                    'N': int(config.get('N')),
                    'chi': float(config.get('chi')),
                    'kappa': float(config.get('kappa')),
                })
                if 'seed' in config:
                    params['seed'] = int(config['seed'])
                return params
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}", file=sys.stderr)
    
    # Fallback: parse from folder name
    # Look backwards through the path to find the parameter-containing folder
    current = folder_path if folder_path.is_dir() else folder_path.parent
    
    while current != current.parent:
        folder_name = current.name
        
        # Try to match pattern like d50_P499_N800_chi10_kappa0.1
        pattern = r'd(\d+).*P(\d+).*N(\d+).*chi([\d.eE+-]+).*kappa([\d.eE+-]+)'
        match = re.search(pattern, folder_name)
        
        if match:
            params = {
                'd': int(match.group(1)),
                'P': int(match.group(2)),
                'N': int(match.group(3)),
                'chi': float(match.group(4)),
                'kappa': float(match.group(5)),
            }
            
            # Try to find seed in the current folder or its name
            seed_match = re.search(r'seed(\d+)', current.name)
            if seed_match:
                params['seed'] = int(seed_match.group(1))
            
            return params
        
        current = current.parent
    
    raise ValueError(
        f"Could not extract parameters from folder path: {folder_path}\n"
        "Expected folder name like: d50_P499_N800_chi10_kappa0.1 or config.json"
    )


def solve_eos_with_julia(
    d: int,
    P: int,
    n1: int,
    n2: int,
    chi: float,
    kappa: float,
    epsilon: float = 1e-3,
    julia_script: Optional[Path] = None,
    verbose: bool = False
) -> Dict:
    """
    Solve equation of state using Julia.
    
    Args:
        d: Input dimension
        P: Dataset size
        n1: Hidden layer 1 width
        n2: Hidden layer 2 width
        chi: Regularization weight
        kappa: Ridge parameter (can be kappa_eff)
        epsilon: Numerical tolerance
        julia_script: Path to eos_fcn3erf.jl (default: standard location)
        verbose: Verbose output
    
    Returns:
        Dict with EOS results
    
    Raises:
        RuntimeError: If Julia solver fails
    """
    if julia_script is None:
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    
    julia_script = Path(julia_script)
    if not julia_script.exists():
        raise RuntimeError(f"Julia script not found: {julia_script}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as tf:
        output_file = tf.name
    
    try:
        if verbose:
            print(f"Running Julia EOS solver...")
            print(f"  Parameters: d={d}, P={P}, n1={n1}, n2={n2}, chi={chi}, kappa={kappa}")
        
        cmd = [
            "julia", str(julia_script),
            f"--d={d}", f"--P={P}",
            f"--n1={n1}", f"--n2={n2}",
            f"--chi={chi}", f"--kappa={kappa}",
            f"--epsilon={epsilon}", f"--to={output_file}", "--quiet"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        
        # Load and return results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        return results.get("target", {})
    
    finally:
        # Clean up
        try:
            Path(output_file).unlink()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Compute kappa_eff and solve EOS for a model folder"
    )
    
    parser.add_argument(
        "model_folder",
        help="Path to model folder (e.g., /path/to/d50_P499_N800_chi10_kappa0.1/seed0)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5000,
        help="Number of samples for eigenvalue computation (default: 5000)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for eigenvalue computation ('cpu' or 'cuda', default: cpu)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse folder path
    model_folder = Path(args.model_folder)
    if not model_folder.exists():
        print(f"Error: Folder not found: {model_folder}", file=sys.stderr)
        return 1
    
    try:
        # Extract parameters from folder
        if args.verbose:
            print(f"Extracting parameters from: {model_folder}")
        
        params = extract_params_from_path(model_folder)
        
        if args.verbose:
            print(f"Extracted parameters: {json.dumps(params, indent=2)}")
        
        # Compute kappa_eff
        if args.verbose:
            print(f"\nComputing kappa_eff...")
        
        kappa_eff = compute_kappa_eff(
            d=params['d'],
            P=params['P'],
            kappa_bare=params['kappa'],
            n1=params['N'],
            n2=params['N'],
            chi=params['chi'],
            device=args.device,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"kappa_bare = {params['kappa']:.6f}")
            print(f"kappa_eff  = {kappa_eff:.6f}")
        
        # Solve EOS with kappa_eff
        if args.verbose:
            print(f"\nSolving EOS with kappa_eff...")
        
        eos_results = solve_eos_with_julia(
            d=params['d'],
            P=params['P'],
            n1=params['N'],
            n2=params['N'],
            chi=params['chi'],
            kappa=kappa_eff,  # Use kappa_eff
            verbose=args.verbose
        )
        
        # Compile results
        results = {
            "model_folder": str(model_folder),
            "parameters": params,
            "kappa_bare": params['kappa'],
            "kappa_eff": kappa_eff,
            "eos_results": eos_results
        }
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Model folder: {model_folder}")
        print(f"Parameters: d={params['d']}, P={params['P']}, N={params['N']}, chi={params['chi']}")
        print(f"kappa_bare:  {params['kappa']:.6f}")
        print(f"kappa_eff:   {kappa_eff:.6f}")
        print(f"Ratio (kappa_eff/kappa_bare): {kappa_eff / params['kappa']:.4f}")
        print("\nEOS Results:")
        for key, value in eos_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print("="*70)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
