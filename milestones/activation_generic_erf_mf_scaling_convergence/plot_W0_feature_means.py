#!/usr/bin/env python3
"""Aggregate and plot W0 feature vars across runs with seed-based error bars.

Usage:
    python plot_W0_feature_vars.py --base-dir <runs_root> [--dims 150]

Finds model.pt (or model_final.pt) under run folders named d<d>_P<P>_N<N>_chi<chi>/seed<seed>/,
loads each model, averages W0 over ensemble and neuron axes to get a length-d vector, stacks across
seeds/runs, and plots mean±std per feature.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)"

    model_files = list(base.glob(f"**/*{suffix}*/model.pt")) if suffix else list(base.glob("**/model.pt"))
    for model_file in model_files:
        seed_dir = model_file.parent
        seed_name = seed_dir.name
        m_seed = re.match(r"seed(\d+)", seed_name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        cfg_dir = seed_dir.parent
        cfg_name = cfg_dir.name
        m_cfg = re.match(pattern, cfg_name)
        if not m_cfg:
            continue

        d = int(m_cfg.group(1))
        if dims and d not in dims:
            continue
        P = int(m_cfg.group(2))
        N = int(m_cfg.group(3))
        chi = int(float(m_cfg.group(4)))

        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))

    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    print(f"Found {len(selected)} runs with model.pt")
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
    model.device = device
    return model


def load_run_config(run_dir: Path) -> Dict[str, float]:
    """Load config.json for a run (if present) to retrieve kappa/epsilon/etc."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def compute_theory(base_dir: Path, cfg: Dict[str, int], kappa: float, eps: float) -> Optional[Dict[str, float]]:
    """Invoke Julia eos_fcn3erf.jl to compute theory and return lWT/lWP.

    Stores theory JSON under base_dir/theory.json and returns its parsed content.
    """
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    theory_path = base_dir / "theory.json"

    cmd = [
        "julia",
        str(julia_script),
        f"--d={cfg['d']}",
        f"--P={cfg['P']}",
        f"--n1={cfg['N']}",
        f"--n2={cfg['N']}",
        f"--chi={cfg['chi']}",
        f"--kappa={kappa}",
        f"--epsilon={eps}",
        f"--to={theory_path}",
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        print(f"  Warning: failed to run Julia theory solver: {e}")
        return None

    try:
        with open(theory_path, "r") as f:
            data = json.load(f)
        target = data.get("target", {})
        perp = data.get("perpendicular", {})
        return {
            "lWT": target.get("lWT"),
            "lWP": perp.get("lWP"),
        }
    except Exception as e:
        print(f"  Warning: failed to load theory.json: {e}")
        return None


def load_kappa_effective(base_dir: Path) -> Dict[str, float]:
    """Load kappa_effective.json if present; return map run_dir->kappa_eff."""
    path = base_dir / "plots" / "kappa_effective.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        results = data.get("results", [])
        return {entry.get("run_dir", ""): float(entry["kappa_eff"]) for entry in results if entry.get("kappa_eff") is not None}
    except Exception as e:
        print(f"  Warning: failed to load kappa_effective.json: {e}")
        return {}


def plot_W0_feature_vars_all_runs(
    runs: List[Tuple[Path, Dict[str, int]]],
    device: torch.device,
    out_path: Path,
) -> None:
    """Plot W0 vars over ensemble and neuron axes, with error bars across seeds."""
    try:
        var_list = []
        d_ref = None
        theory_bare = None
        theory_corrected = None
        theory_cache: Dict[Tuple[int, int, int, int, float, float], Dict[str, float]] = {}
        kappa_eff_map = load_kappa_effective(out_path.parent.parent)

        for run_dir, cfg in runs:
            model = load_model(run_dir, cfg, device)
            if model is None:
                continue
            run_cfg_json = load_run_config(run_dir)
            kappa_bare = float(run_cfg_json.get("kappa", 1.0))
            eps = float(run_cfg_json.get("eps", run_cfg_json.get("epsilon", 0.03)))

            # Bare theory (once per cfg)
            key_bare = (cfg['d'], cfg['P'], cfg['N'], cfg['chi'], kappa_bare, eps)
            if theory_bare is None and key_bare in theory_cache:
                theory_bare = theory_cache[key_bare]
            if theory_bare is None:
                val = compute_theory(run_dir.parent, cfg, kappa_bare, eps)
                if val:
                    theory_cache[key_bare] = val
                    theory_bare = val

            # Corrected theory using kappa_eff if available for this run_dir
            kappa_eff = kappa_eff_map.get(str(run_dir))
            if theory_corrected is None and kappa_eff is not None:
                key_eff = (cfg['d'], cfg['P'], cfg['N'], cfg['chi'], kappa_eff, eps)
                if key_eff in theory_cache:
                    theory_corrected = theory_cache[key_eff]
                else:
                    val_eff = compute_theory(run_dir.parent, cfg, kappa_eff, eps)
                    if val_eff:
                        theory_cache[key_eff] = val_eff
                        theory_corrected = val_eff
            W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
            var_d = W0.var(axis=(0, 1))
            if d_ref is None:
                d_ref = var_d.shape[0]
            elif var_d.shape[0] != d_ref:
                print(f"  Skipping {run_dir}: mismatched d")
                continue
            var_list.append(var_d)

        if not var_list:
            print("No W0 vars to plot.")
            return

        vars = np.stack(var_list, axis=0)  # (num_runs, d)
        mean_over_runs = vars.mean(axis=0)
        std_over_runs = vars.std(axis=0) / np.sqrt(vars.shape[0])  # SEM

        # Extract empirical lWT (first element of mean_over_runs)
        empirical_lWT = float(mean_over_runs[0]) if len(mean_over_runs) > 0 else float('nan')

        fig, ax = plt.subplots(figsize=(12, 5))
        x_idx = np.arange(d_ref)
        ax.errorbar(x_idx, mean_over_runs, yerr=std_over_runs, fmt='o', markersize=3,
                    ecolor='gray', elinewidth=1, capsize=2, 
                    label=f'Empirical $\\lambda_W^*$ = {empirical_lWT:.3g}')
        # Add theory lines if available
        if theory_bare:
            lWT = theory_bare.get("lWT")
            lWP = theory_bare.get("lWP")
            if lWT is not None:
                ax.axhline(lWT, color='tab:red', linestyle='--', linewidth=2.0, 
                          label=f"Theory (bare $\\kappa$) $\\lambda_W^*$ = {lWT:.3g}")
            if lWP is not None:
                ax.axhline(lWP, color='tab:orange', linestyle='-.', linewidth=2.0, 
                          label=f"Theory (bare $\\kappa$) $\\lambda_W^\\perp$ = {lWP:.3g}")
        if theory_corrected:
            lWTc = theory_corrected.get("lWT")
            lWPc = theory_corrected.get("lWP")
            if lWTc is not None:
                ax.axhline(lWTc, color='tab:purple', linestyle='--', linewidth=2.0, 
                          label=f"Theory ($\\kappa_{{\\text{{eff}}}}$) $\\lambda_W^*$ = {lWTc:.3g}")
            if lWPc is not None:
                ax.axhline(lWPc, color='tab:brown', linestyle='-.', linewidth=2.0, 
                          label=f"Theory ($\\kappa_{{\\text{{eff}}}}$) $\\lambda_W^\\perp$ = {lWPc:.3g}")
        ax.set_xlabel('Feature index (d)')
        ax.set_ylabel('Readin $W$ variance (avg over ens,N)')
        ax.set_title('Readin $W$ variances across runs with run-wise error bars')
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W0 feature vars plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot W0 feature vars across runs: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot W0 vars across runs with error bars.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for the plot (default: <base-dir>/plots/W0_feature_vars_all_runs.png)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to plot.")
        return

    out_path = args.out or (base_dir / "plots" / "W0_feature_vars_all_runs.png")
    plot_W0_feature_vars_all_runs(runs, device, out_path)


if __name__ == "__main__":
    main()
