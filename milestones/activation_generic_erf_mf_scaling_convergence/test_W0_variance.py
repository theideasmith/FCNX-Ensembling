#!/usr/bin/env python3
"""Test whether W0 variance and standard deviation match theoretical expectations.

Initializes FCN3NetworkActivationGeneric models with various configurations and
compares the empirical W0 variance against the theoretical value (1/d).

Usage:
    python test_W0_variance.py --dims 50 100 150 --ens 10 50 --N 100 500
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_W0_statistics(
    d: int,
    N: int,
    ens: int,
    P: int,
    chi: int,
    device: torch.device,
    num_trials: int = 5,
) -> Tuple[float, float, float, float]:
    """Initialize models and compute W0 variance statistics.
    
    Args:
        d: Input dimension
        N: Hidden layer width
        ens: Ensemble size
        P: Output dimension
        chi: Scale parameter for second layer
        device: Torch device
        num_trials: Number of independent initializations to average over
    
    Returns:
        (mean_var, std_var, mean_std, std_std): Mean and std of variance and stddev across trials
    """
    variances = []
    stddevs = []
    
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    
    for trial in range(num_trials):
        model = FCN3NetworkActivationGeneric(
            d=d,
            n1=N,
            n2=N,
            P=P,
            ens=ens,
            activation="erf",
            weight_initialization_variance=weight_var,
        ).to(device)
        
        # W0 shape: (ens, N, d)
        W0 = model.W0.detach().cpu().numpy()
        
        # Compute variance and stddev of all W0 weights
        var = W0.var()
        std = W0.std()
        
        variances.append(var)
        stddevs.append(std)
    
    return (
        float(np.mean(variances)),
        float(np.std(variances)),
        float(np.mean(stddevs)),
        float(np.std(stddevs)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test W0 variance against theoretical expectations."
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", default=[50, 100, 150],
        help="List of input dimensions to test"
    )
    parser.add_argument(
        "--N-values", type=int, nargs="+", default=[1600, 1200],
        help="List of hidden layer widths to test"
    )
    parser.add_argument(
        "--ens-values", type=int, nargs="+", default=[10, 50],
        help="List of ensemble sizes to test"
    )
    parser.add_argument(
        "--P", type=int, default=1200,
        help="Output dimension (default: 100)"
    )
    parser.add_argument(
        "--chi", type=int, default=1,
        help="Scale parameter for second layer (default: 1)"
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Number of independent initializations per configuration (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device string, e.g., cuda:0 or cpu (default: auto)"
    )
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    
    print("Testing W0 variance and standard deviation against theory")
    print(f"Device: {device}")
    print(f"Trials per configuration: {args.trials}")
    print("=" * 80)
    
    results = []
    
    for d in args.dims:
        theory_var = 1.0 / d
        theory_std = 1.0 / np.sqrt(d)
        
        for N in args.N_values:
            for ens in args.ens_values:
                print(f"\nConfiguration: d={d}, N={N}, ens={ens}, P={args.P}, chi={args.chi}")
                print(f"  Theory: var={theory_var:.6f}, std={theory_std:.6f}")
                
                mean_var, std_var, mean_std, std_std = test_W0_statistics(
                    d=d,
                    N=N,
                    ens=ens,
                    P=args.P,
                    chi=args.chi,
                    device=device,
                    num_trials=args.trials,
                )
                
                var_rel_error = abs(mean_var - theory_var) / theory_var * 100
                std_rel_error = abs(mean_std - theory_std) / theory_std * 100
                
                print(f"  Empirical: var={mean_var:.6f} ± {std_var:.6f}, std={mean_std:.6f} ± {std_std:.6f}")
                print(f"  Relative error: var={var_rel_error:.2f}%, std={std_rel_error:.2f}%")
                
                results.append({
                    "d": d,
                    "N": N,
                    "ens": ens,
                    "theory_var": theory_var,
                    "theory_std": theory_std,
                    "mean_var": mean_var,
                    "std_var": std_var,
                    "mean_std": mean_std,
                    "std_std": std_std,
                    "var_rel_error": var_rel_error,
                    "std_rel_error": std_rel_error,
                })
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'d':>5} {'N':>5} {'ens':>5} | {'Theory Var':>12} {'Emp Var':>12} {'Var Err %':>10} | {'Theory Std':>12} {'Emp Std':>12} {'Std Err %':>10}")
    print("-" * 80)
    
    for r in results:
        print(
            f"{r['d']:>5} {r['N']:>5} {r['ens']:>5} | "
            f"{r['theory_var']:>12.6f} {r['mean_var']:>12.6f} {r['var_rel_error']:>9.2f}% | "
            f"{r['theory_std']:>12.6f} {r['mean_std']:>12.6f} {r['std_rel_error']:>9.2f}%"
        )
    
    # Check for significant discrepancies
    print("\n" + "=" * 80)
    threshold = 5.0  # 5% error threshold
    issues = [r for r in results if r['var_rel_error'] > threshold or r['std_rel_error'] > threshold]
    
    if issues:
        print(f"WARNING: {len(issues)} configuration(s) with >5% error:")
        for r in issues:
            print(f"  d={r['d']}, N={r['N']}, ens={r['ens']}: var_err={r['var_rel_error']:.2f}%, std_err={r['std_rel_error']:.2f}%")
    else:
        print("✓ All configurations match theory within 5% tolerance.")


if __name__ == "__main__":
    main()
