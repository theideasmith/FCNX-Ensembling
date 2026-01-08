#!/usr/bin/env python3
"""
Compute self-consistent kappa_eff for each trained model by estimating its
preactivation kernel eigenvalues and invoking the Julia solver
`self_consistent_kappa_solver.jl`. Results are written to JSON.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Workspace libs
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric


def find_run_dirs(base: Path, dims: List[int]) -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories containing model.pt/model_final.pt and parse cfg."""
    import re
    selected: List[Tuple[Path, Dict[str, int]]] = []
    model_files = list(base.glob("**/model.pt")) + list(base.glob("**/model_final.pt"))

    for model_file in model_files:
        seed_dir = model_file.parent
        seed_match = re.match(r"seed(\d+)", seed_dir.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        cfg_dir = seed_dir.parent
        cfg_match = re.match(r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)", cfg_dir.name)
        if not cfg_match:
            continue
        d = int(cfg_match.group(1))
        P = int(cfg_match.group(2))
        N = int(cfg_match.group(3))
        chi = int(float(cfg_match.group(4)))

        if dims and d not in dims:
            continue

        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))

    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    return selected


def load_model(run_dir: Path, config: Dict[str, int], device: torch.device) -> Optional[FCN3NetworkActivationGeneric]:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(config.get("d"))
    P = int(config.get("P"))
    N = int(config.get("N"))
    chi = int(config.get("chi", N))

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
    return model


def compute_empirical_spectrum(
    model: FCN3NetworkActivationGeneric, d: int, P: int, device: torch.device, k: Optional[int] = None
) -> Optional[np.ndarray]:
    """Estimate spectrum via randomized SVD of H on random data."""
    try:
        with torch.no_grad():
            model.to(device)
            model.device = device
            X = torch.randn(10000, d, device=device)
            rank_k = k if k is not None else 9000
            eigs = model.H_eig_random_svd(X, rank_k)
            return eigs.detach().cpu().numpy()
    except Exception as e:
        print(f"  Warning: failed to compute spectrum via H_eig_random_svd: {e}")
        return None


def solve_kappa_eff(eigenvalues: np.ndarray, P: int, kappa_bare: float, julia_script: Path) -> Optional[float]:
    """Call Julia self_consistent_kappa_solver.jl and parse kappa_eff."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as tf:
        json.dump({"eigenvalues": eigenvalues.tolist(), "kappa_bare": kappa_bare}, tf)
        eig_json = tf.name

    try:
        output = subprocess.check_output(
            ["julia", str(julia_script), eig_json, str(P)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r"kappa_eff\s*=\s*([-0-9.eE+]+)", output)
        if match:
            return float(match.group(1))
        print("  Warning: kappa_eff not found in solver output")
        print(output)
        return None
    except Exception as e:
        print(f"  Warning: self-consistent solver failed: {e}")
        return None
    finally:
        try:
            Path(eig_json).unlink(missing_ok=True)
        except Exception:
            pass


def process_runs(base_dir: Path, dims: List[int], kappa_bare: float, k_svd: Optional[int], device: torch.device) -> Dict:
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
    runs = find_run_dirs(base_dir, dims)
    results = []

    for run_dir, cfg in runs:
        model = load_model(run_dir, cfg, device)
        if model is None:
            print(f"Skipping {run_dir} (no model file)")
            continue

        eigs = compute_empirical_spectrum(model, cfg["d"], cfg["P"], device, k=k_svd)
        if eigs is None:
            print(f"Skipping {run_dir} (spectrum failed)")
            continue

        kappa_eff = solve_kappa_eff(eigs, cfg["P"], kappa_bare, julia_script)
        results.append({
            "run_dir": str(run_dir),
            "d": cfg["d"],
            "P": cfg["P"],
            "N": cfg["N"],
            "chi": cfg["chi"],
            "seed": cfg.get("seed"),
            "kappa_bare": kappa_bare,
            "kappa_eff": kappa_eff,
            "num_eigs": int(len(eigs)),
        })

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_dir": str(base_dir),
        "kappa_bare": kappa_bare,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute effective kappa for each model using eigenvalues")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory containing d*_P*_N*_chi*/seed*/model.pt")
    parser.add_argument("--dims", type=int, nargs="*", default=[], help="Optional list of d values to include")
    parser.add_argument("--kappa-bare", type=float, default=2.0, help="Bare kappa to feed into solver")
    parser.add_argument("--rank", type=int, default=9000, help="Rank k for randomized SVD in H_eig_random_svd")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Torch device string")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: base/plots/kappa_effective.json)")
    args = parser.parse_args()

    base_dir = args.base
    output_path = args.output or (base_dir / "plots" / "kappa_effective.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    payload = process_runs(base_dir, args.dims, args.kappa_bare, args.rank, device)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote kappa_eff results to {output_path}")


if __name__ == "__main__":
    main()
