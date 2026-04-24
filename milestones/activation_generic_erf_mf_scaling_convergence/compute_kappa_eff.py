#!/usr/bin/env python3
"""
Compute effective ridge (kappa_eff) for given parameters.

Usage:
    python compute_kappa_eff.py --d 50 --P 500 --kappa 0.1
    python compute_kappa_eff.py --d 50 100 --P 500 1000 --kappa 0.1 0.5
    python compute_kappa_eff.py --config parameters.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Import the solver
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from kappa_eff_solver import compute_kappa_eff, compute_kappa_eff_batch


def main():
    parser = argparse.ArgumentParser(
        description="Compute effective ridge (kappa_eff) for neural network parameters"
    )
    
    parser.add_argument(
        "--d", type=int, nargs='+',
        help="Input dimension(s)"
    )
    parser.add_argument(
        "--P", type=int, nargs='+',
        help="Number of training samples"
    )
    parser.add_argument(
        "--kappa", type=float, nargs='+',
        help="Bare ridge parameter(s)"
    )
    parser.add_argument(
        "--n1", type=int, nargs='+',
        help="Hidden layer 1 width(s) (default: P)"
    )
    parser.add_argument(
        "--n2", type=int, nargs='+',
        help="Hidden layer 2 width(s) (default: P)"
    )
    parser.add_argument(
        "--chi", type=float, nargs='+', default=[10.0],
        help="Regularization weight(s) (default: 10.0)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--config", type=str,
        help="JSON config file with parameter sets"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5000,
        help="Number of samples for eigenvalue computation (default: 5000)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for computation ('cpu' or 'cuda', default: cpu)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config and not (args.d and args.P and args.kappa):
        parser.error("Either --config or (--d, --P, --kappa) must be provided")
    
    # Prepare parameter sets
    parameters: List[Dict] = []
    
    if args.config:
        if args.verbose:
            print(f"Loading parameters from {args.config}...")
        with open(args.config, 'r') as f:
            config = json.load(f)
            if isinstance(config, list):
                parameters = config
            elif isinstance(config, dict) and 'parameters' in config:
                parameters = config['parameters']
            else:
                parser.error("Config file must contain list or dict with 'parameters' key")
    else:
        # Create parameter combinations
        n1_vals = args.n1 if args.n1 else [None]
        n2_vals = args.n2 if args.n2 else [None]
        chi_vals = args.chi if args.chi else [10.0]
        
        for d in args.d:
            for P in args.P:
                for kappa in args.kappa:
                    for n1 in n1_vals:
                        for n2 in n2_vals:
                            for chi in chi_vals:
                                params = {
                                    'd': d,
                                    'P': P,
                                    'kappa_bare': kappa
                                }
                                if n1 is not None:
                                    params['n1'] = n1
                                if n2 is not None:
                                    params['n2'] = n2
                                params['chi'] = chi
                                parameters.append(params)
    
    if args.verbose:
        print(f"\nComputing kappa_eff for {len(parameters)} parameter set(s):")
        print(json.dumps(parameters, indent=2))
        print()
    
    # Compute kappa_eff
    try:
        results = compute_kappa_eff_batch(
            parameters,
            device=args.device,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Format and print results
    output_data = {
        "parameters": [r[0] for r in results],
        "kappa_eff": [r[1] for r in results],
        "results": [
            {**r[0], "kappa_eff": r[1]}
            for r in results
        ]
    }
    
    # Print to stdout
    print("\n" + "="*70)
    print("KAPPA_EFF COMPUTATION RESULTS")
    print("="*70)
    for params, kappa_eff in results:
        print(f"d={params['d']:3d}, P={params['P']:5d}, kappa_bare={params['kappa_bare']:.4f}  →  kappa_eff={kappa_eff:.6f}")
    print("="*70)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
