#!/usr/bin/env python3
"""
Compute and plot learnabilities for He1 (linear) and He3 (cubic) modes
using streaming evaluation on 50M test samples for FCN2 vs FCN3.

The script:
1. Identifies paired FCN2/FCN3 trained models from epsilon sweep
2. Streams through 50M test samples in batches
3. Computes He1 projections directly
4. Computes He3 projections using Gram matrix accumulation (streaming)
5. Plots He1 and He3 learnabilities against epsilon with publication-ready styling
"""

from pathlib import Path
import argparse
import re
import json
import math
import sys
from typing import Optional, Tuple, Dict, List
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lib.FCN3Network import FCN3NetworkActivationGeneric
from lib.FCN2Network import FCN2NetworkActivationGeneric

# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Matplotlib styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# Target function and Hermite polynomials
def target_fn(X: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Target function: f(x) = x_0 + epsilon * (x_0^3 - 3*x_0)"""
    x0 = X[:, 0]
    return x0 + epsilon * (x0**3 - 3.0 * x0)


def hermite_h1(x: torch.Tensor) -> torch.Tensor:
    """He1: x (linear component)"""
    return x


def hermite_h3(x: torch.Tensor) -> torch.Tensor:
    """He3: (x^3 - 3*x) / sqrt(6)"""
    return (x**3 - 3.0 * x) / math.sqrt(6.0)


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device = DEVICE
) -> Tuple[Dict, torch.nn.Module]:
    """Load model from checkpoint. Detects FCN2 vs FCN3 from checkpoint structure."""
    run_dir = checkpoint_path.parent
    config_path = run_dir / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        raise FileNotFoundError(f"No config.json found in {run_dir}")
    
    d = int(cfg["d"])
    n1 = int(cfg.get("N") or cfg.get("n1", 800))
    n2 = int(cfg.get("n2", n1))
    P = int(cfg["P"])
    ens = int(cfg.get("ens", 1))
    activation = str(cfg.get("activation", "erf")).lower()
    
    # Load checkpoint to detect FCN2 vs FCN3
    state_dict_raw = torch.load(checkpoint_path, map_location=device)
    state_dict = state_dict_raw
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        if "state_dict" in state_dict and "W0" not in state_dict:
            state_dict = state_dict["state_dict"]
    
    # Detect network type from checkpoint keys
    has_w1 = "W1" in state_dict
    is_fcn3 = has_w1
    
    if is_fcn3:
        model_class = FCN3NetworkActivationGeneric
        model = model_class(
            d, n1, n2, P, ens=ens, activation=activation, device=str(device)
        ).to(device).double().eval()
    else:
        model_class = FCN2NetworkActivationGeneric
        model = model_class(
            d, n1, P, ens=ens, activation=activation, device=str(device)
        ).to(device).double().eval()
    
    # Handle ensemble dimension
    for key in ["W0", "W1", "A"]:
        if key in state_dict and state_dict[key].ndim == 4:
            state_dict[key] = state_dict[key].squeeze(0)
    
    model.load_state_dict(state_dict)
    return cfg, model


def find_runs_by_eps(root: Path, family: str) -> Dict[str, Path]:
    """Find run directories indexed by epsilon value."""
    by_eps = {}
    base = root / family
    if not base.exists():
        return by_eps
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        m = re.search(r"_eps_([0-9]+\.?[0-9]*)", p.name)
        if m:
            eps = m.group(1)
            by_eps[eps] = p
    return by_eps


class StreamingLearnabilityComputer:
    """Compute He1 and He3 learnabilities using streaming evaluation."""
    
    def __init__(self, model: torch.nn.Module, d: int, epsilon: float, device: torch.device = DEVICE):
        self.model = model
        self.d = d
        self.epsilon = epsilon
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset accumulators for streaming computation."""
        self.n_samples = 0
        self.he1_sum = 0.0
        self.he3_sum = 0.0
        self.he1_target_sum = 0.0
        self.he3_target_sum = 0.0
        # For He3: accumulate Gram matrix entries
        self.he3_he3_sum = 0.0
        self.target_he3_sum = 0.0
    
    def process_batch(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Process a batch of inputs and targets."""
        n = X_batch.shape[0]
        x0 = X_batch[:, 0]
        
        # He1 component: linear in x_0
        he1 = x0
        
        # He3 component: (x_0^3 - 3*x_0) / sqrt(6)
        he3 = (x0**3 - 3.0 * x0) / math.sqrt(6.0)
        
        # Target components (already includes both He1 and He3)
        # f(x) = x_0 + epsilon * (x_0^3 - 3*x_0)
        #      = x_0 + epsilon * sqrt(6) * He3(x_0)
        target_he1_coeff = 1.0
        target_he3_coeff = self.epsilon * math.sqrt(6.0)
        
        # Model predictions
        with torch.no_grad():
            y_pred_raw = self.model(X_batch)
        
        # Handle different output shapes
        # Could be: (batch,), (batch, 1), (batch, ens), (batch, ens, 1)
        if isinstance(y_pred_raw, (tuple, list)):
            y_pred_raw = y_pred_raw[0]
        
        # Squeeze dimensions as needed
        while y_pred_raw.ndim > 1 and y_pred_raw.shape[-1] == 1:
            y_pred_raw = y_pred_raw.squeeze(-1)
        
        if y_pred_raw.ndim > 1:
            # Average over ensemble or other batch dimensions
            y_pred_raw = y_pred_raw.mean(dim=list(range(1, y_pred_raw.ndim)))
        
        y_pred = y_pred_raw.to(dtype=DTYPE)
        
        if y_pred.shape[0] != n:
            raise RuntimeError(
                f"Output shape mismatch: expected {n} samples, got {y_pred.shape}"
            )
        
        # He1 learnability: compute projection of prediction onto He1
        he1_pred_product = (y_pred * he1).sum().item()
        he1_he1 = (he1 * he1).sum().item()
        
        self.he1_sum += he1_pred_product
        self.he1_target_sum += (y_batch * he1).sum().item()
        
        # He3 learnability: use Gram matrix accumulation
        # After projecting out He1, compute He3 projection
        he3_pred_product = (y_pred * he3).sum().item()
        he3_target_product = (y_batch * he3).sum().item()
        he3_he3 = (he3 * he3).sum().item()
        
        self.he3_sum += he3_pred_product
        self.he3_target_sum += he3_target_product
        self.he3_he3_sum += he3_he3
        
        self.n_samples += n
        
        # Memory cleanup
        del y_pred_raw, y_pred, he1, he3, x0
        torch.cuda.empty_cache()
    
    def get_learnabilities(self) -> Dict[str, float]:
        """Compute He1 and He3 learnabilities from accumulated sums."""
        if self.n_samples == 0:
            return {"he1": 0.0, "he3": 0.0, "he1_proj": 0.0, "he3_proj": 0.0}
        
        # He1 learnability
        he1_avg_pred = self.he1_sum / self.n_samples
        he1_avg_target = self.he1_target_sum / self.n_samples
        he1_learnability = he1_avg_pred / he1_avg_target if abs(he1_avg_target) > 1e-10 else 0.0
        
        # He3 learnability
        he3_avg_pred = self.he3_sum / self.n_samples
        he3_avg_target = self.he3_target_sum / self.n_samples
        he3_learnability = he3_avg_pred / he3_avg_target if abs(he3_avg_target) > 1e-10 else 0.0
        
        return {
            "he1": float(he1_learnability),
            "he3": float(he3_learnability),
            "he1_proj": float(he1_avg_pred),
            "he3_proj": float(he3_avg_pred),
            "he1_target": float(he1_avg_target),
            "he3_target": float(he3_avg_target),
        }


def compute_learnabilities_streaming(
    model: torch.nn.Module,
    d: int,
    epsilon: float,
    num_test_samples: int = 50_000_000,
    batch_size: int = 50000,
    test_seed: int = 12345,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Compute He1 and He3 learnabilities by streaming through test samples.
    
    Args:
        model: trained FCN3 model
        d: input dimension
        epsilon: regularization strength / target function parameter
        num_test_samples: total number of test samples to process
        batch_size: batch size for streaming
        test_seed: random seed for generating test data
        device: torch device
    
    Returns:
        Dictionary with He1 and He3 learnabilities
    """
    computer = StreamingLearnabilityComputer(model, d, epsilon, device=device)
    
    generator = torch.Generator(device=device)
    generator.manual_seed(test_seed)
    
    num_batches = (num_test_samples + batch_size - 1) // batch_size
    
    print(f"  Computing learnabilities over {num_test_samples:,} samples ({num_batches} batches)...")
    
    for batch_idx in range(num_batches):
        # Generate batch of input samples
        X_batch = torch.randn(
            batch_size, d,
            generator=generator,
            dtype=DTYPE,
            device=device
        )
        
        # Compute target values
        y_batch = target_fn(X_batch, epsilon=computer.epsilon).to(dtype=DTYPE)
        
        # Process batch
        computer.process_batch(X_batch, y_batch)
        
        # Progress update every 10% or last batch
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
            percent = min(100, 100.0 * (batch_idx + 1) / num_batches)
            print(f"    Progress: {percent:.1f}% ({(batch_idx+1)*batch_size:,} samples)")
    
    return computer.get_learnabilities()


def evaluate_epsilon_sweep(
    results_root: Path,
    num_test_samples: int = 50_000_000,
    test_seed: int = 12345,
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
) -> Tuple[List[float], Dict, Dict]:
    """
    Evaluate both FCN2 and FCN3 models across epsilon sweep.
    
    Returns:
        (eps_values, fcn2_results, fcn3_results)
    """
    fcn2_results: Dict[str, Dict] = {}
    fcn3_results: Dict[str, Dict] = {}
    cached_eps_set = set()

    if use_cache and cache_path and cache_path.exists():
        print(f"Loading cache from {cache_path}")
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            fcn2_results = {str(k): v for k, v in cache_data.get("fcn2", {}).items()}
            fcn3_results = {str(k): v for k, v in cache_data.get("fcn3", {}).items()}
            cached_eps_set = set(fcn2_results.keys()).intersection(set(fcn3_results.keys()))
            print(f"Loaded {len(cached_eps_set)} epsilon values from cache")
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            fcn2_results = {}
            fcn3_results = {}

    fcn2_runs = find_runs_by_eps(results_root, "fcn2")
    fcn3_runs = find_runs_by_eps(results_root, "fcn3")
    
    common_eps = sorted(
        set(fcn2_runs.keys()).intersection(set(fcn3_runs.keys())),
        key=lambda s: float(s)
    )
    
    print(f"Found {len(common_eps)} matching epsilon values")

    eps_values: List[float] = []
    eps_to_compute = [eps for eps in common_eps if eps not in cached_eps_set]
    if eps_to_compute:
        print(
            f"Need to compute {len(eps_to_compute)} epsilon values "
            f"(cached: {len(common_eps) - len(eps_to_compute)})"
        )
    else:
        print(f"All {len(common_eps)} epsilon values available in cache!")

    for eps_str in common_eps:
        eps_val = float(eps_str)
        eps_values.append(eps_val)

        if eps_str in cached_eps_set:
            continue

        print(f"\nComputing epsilon = {eps_str}")

        # FCN2
        try:
            fcn2_run = fcn2_runs[eps_str]
            fcn2_checkpoint = fcn2_run / "seed0" / "model_final.pt"
            if not fcn2_checkpoint.exists():
                seed_dirs = sorted([d for d in fcn2_run.iterdir() 
                                   if d.is_dir() and d.name.startswith("seed")])
                if seed_dirs:
                    fcn2_checkpoint = seed_dirs[0] / "model_final.pt"
            
            if fcn2_checkpoint.exists():
                cfg_fcn2, model_fcn2 = load_model_from_checkpoint(fcn2_checkpoint, device=DEVICE)
                eps_fcn2 = float(cfg_fcn2.get("eps", eps_str))
                print(f"  FCN2: d={cfg_fcn2['d']}, N={cfg_fcn2.get('N', cfg_fcn2.get('n1'))}, eps={eps_fcn2}")
                
                fcn2_learn = compute_learnabilities_streaming(
                    model_fcn2,
                    d=int(cfg_fcn2["d"]),
                    epsilon=eps_fcn2,
                    num_test_samples=num_test_samples,
                    test_seed=test_seed,
                    device=DEVICE
                )
                fcn2_results[eps_str] = fcn2_learn
                del model_fcn2
                torch.cuda.empty_cache()
            else:
                print(f"  FCN2: checkpoint not found at {fcn2_checkpoint}")
        except Exception as e:
            print(f"  FCN2: Error - {e}")
        
        # FCN3
        try:
            fcn3_run = fcn3_runs[eps_str]
            fcn3_checkpoint = fcn3_run / "seed0" / "model_final.pt"
            if not fcn3_checkpoint.exists():
                seed_dirs = sorted([d for d in fcn3_run.iterdir()
                                   if d.is_dir() and d.name.startswith("seed")])
                if seed_dirs:
                    fcn3_checkpoint = seed_dirs[0] / "model_final.pt"
            
            if fcn3_checkpoint.exists():
                cfg_fcn3, model_fcn3 = load_model_from_checkpoint(fcn3_checkpoint, device=DEVICE)
                eps_fcn3 = float(cfg_fcn3.get("eps", eps_str))
                print(f"  FCN3: d={cfg_fcn3['d']}, N={cfg_fcn3.get('N', cfg_fcn3.get('n1'))}, eps={eps_fcn3}")
                
                fcn3_learn = compute_learnabilities_streaming(
                    model_fcn3,
                    d=int(cfg_fcn3["d"]),
                    epsilon=eps_fcn3,
                    num_test_samples=num_test_samples,
                    test_seed=test_seed,
                    device=DEVICE
                )
                fcn3_results[eps_str] = fcn3_learn
                del model_fcn3
                torch.cuda.empty_cache()
            else:
                print(f"  FCN3: checkpoint not found at {fcn3_checkpoint}")
        except Exception as e:
            print(f"  FCN3: Error - {e}")
    
    eps_values = sorted([float(eps) for eps in common_eps])
    return eps_values, fcn2_results, fcn3_results


def plot_learnabilities(
    eps_values: List[float],
    fcn2_results: Dict,
    fcn3_results: Dict,
    out_path: Path,
) -> None:
    """Create publication-ready plot of He1 and He3 learnabilities vs epsilon."""
    
    eps_array = np.array(eps_values)
    
    # Extract He1 and He3 learnabilities
    he1_fcn2 = np.array([fcn2_results[str(e)]["he1"] for e in eps_values if str(e) in fcn2_results])
    he1_fcn3 = np.array([fcn3_results[str(e)]["he1"] for e in eps_values if str(e) in fcn3_results])
    he3_fcn2 = np.array([fcn2_results[str(e)]["he3"] for e in eps_values if str(e) in fcn2_results])
    he3_fcn3 = np.array([fcn3_results[str(e)]["he3"] for e in eps_values if str(e) in fcn3_results])
    
    # Align arrays (some might have missing values)
    he1_eps_fcn2 = [e for e in eps_values if str(e) in fcn2_results]
    he1_eps_fcn3 = [e for e in eps_values if str(e) in fcn3_results]
    he3_eps_fcn2 = [e for e in eps_values if str(e) in fcn2_results]
    he3_eps_fcn3 = [e for e in eps_values if str(e) in fcn3_results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Learnability of $W_0$ Activations vs Regularization Strength", 
                 fontsize=13, fontweight='bold', y=1.00)
    
    # Colors
    color_fcn2 = "#1f77b4"  # Blue
    color_fcn3 = "#d62728"  # Red
    
    # He1 (linear) learnability
    ax = axes[0]
    ax.plot(he1_eps_fcn2, he1_fcn2, 'o-', color=color_fcn2, label='FCN2', 
            linewidth=2.5, markersize=7, markeredgewidth=0)
    ax.plot(he1_eps_fcn3, he1_fcn3, 's-', color=color_fcn3, label='FCN3', 
            linewidth=2.5, markersize=6.5, markeredgewidth=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Regularization Strength ($\epsilon$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'He1 Learnability', fontsize=12, fontweight='bold')
    ax.set_title('Linear Mode (He1)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    # ax.set_ylim(bottom=-0.05)
    
    # He3 (cubic) learnability
    ax = axes[1]
    ax.plot(he3_eps_fcn2, he3_fcn2, 'o-', color=color_fcn2, label='FCN2', 
            linewidth=2.5, markersize=7, markeredgewidth=0)
    ax.plot(he3_eps_fcn3, he3_fcn3, 's-', color=color_fcn3, label='FCN3', 
            linewidth=2.5, markersize=6.5, markeredgewidth=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Regularization Strength ($\epsilon$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'He3 Learnability', fontsize=12, fontweight='bold')
    ax.set_title('Cubic Mode (He3)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    # ax.set_ylim(bottom=-0.05)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot He1/He3 learnabilities with streaming (50M test samples)"
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results_epsilon_sweep",
        help="Root directory containing fcn2/ and fcn3/ subdirectories"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Output directory for plots and results"
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=1_000_000,
        help="Number of test samples to stream through"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for streaming"
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=12345,
        help="Random seed for test data generation"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached results if available, skip recomputation"
    )
    args = parser.parse_args()
    
    results_root = Path(__file__).resolve().parent / args.results_root
    out_dir = Path(__file__).resolve().parent / args.out_dir
    cache_path = out_dir / "learnability_cache.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_root.exists():
        print(f"Error: Results root not found: {results_root}")
        return
    
    print(f"Results root: {results_root}")
    print(f"Output directory: {out_dir}")
    print(f"Cache file: {cache_path}")
    print(f"Processing {args.num_test_samples:,} test samples per model")
    print(f"Use cache: {args.use_cache}")
    print()
    
    # Evaluate models
    eps_values, fcn2_results, fcn3_results = evaluate_epsilon_sweep(
        results_root,
        num_test_samples=args.num_test_samples,
        test_seed=args.test_seed,
        use_cache=args.use_cache,
        cache_path=cache_path,
    )
    
    # Save results
    results = {
        "eps_values": eps_values,
        "fcn2": {str(k): v for k, v in fcn2_results.items()},
        "fcn3": {str(k): v for k, v in fcn3_results.items()},
    }
    json_path = out_dir / "learnability_he1_he3_streaming.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")
    
    # Also save to cache for future runs.
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved cache to {cache_path}")
    
    # Create plots
    plot_learnabilities(
        eps_values,
        fcn2_results,
        fcn3_results,
        out_dir / "learnability_he1_he3_vs_epsilon.png",
    )


if __name__ == "__main__":
    main()
