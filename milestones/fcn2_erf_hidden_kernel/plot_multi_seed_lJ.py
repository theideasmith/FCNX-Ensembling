#!/usr/bin/env python3
"""
Compute H_eig (lJ) for multiple seed runs and plot against theory.

Example:
    python milestones/fcn2_erf_hidden_kernel/plot_multi_seed_lJ.py \
        --base-pattern milestones/fcn2_erf_hidden_kernel/d50_P200_N200_chi_200.0_lr_5e-07_T_2.0_seed_{seed} \
        --seeds 1 2 3 4 5 6 7 8 9 \
        --output multi_seed_lJ.png \
        --json-out multi_seed_lJ.json
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def parse_run_meta_from_name(name: str):
    """Parse d, P, N, chi from run directory name when config.json is absent.

    Expected pattern example: d50_P200_N200_chi_200.0_lr_5e-07_T_2.0_seed_3
    Returns dict with keys d, P, N, chi if found; missing keys omitted.
    """
    import re

    meta = {}
    patterns = {
        "d": r"d(?P<val>\d+)",
        "P": r"_P(?P<val>\d+)",
        "N": r"_N(?P<val>\d+)",
        "chi": r"_chi_(?P<val>[0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, name)
        if m:
            val = m.group("val")
            meta[key] = float(val) if key == "chi" else int(val)
    return meta


def load_model(run_dir: Path, device: torch.device):
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

    parsed = parse_run_meta_from_name(run_dir.name)
    if config is None:
        if not parsed:
            raise FileNotFoundError(f"config.json not found and could not parse hyperparameters from run dir name: {run_dir}")
        config = parsed
    config.setdefault("ens", 1)

    d = int(config.get("d", parsed.get("d")))
    P = int(config.get("P", parsed.get("P")))
    N = int(config.get("N", parsed.get("N", 0)))
    ens = int(config.get("ens", 1))

    model = FCN2NetworkActivationGeneric(
        d, N, P, ens=ens,
        activation="erf",
        weight_initialization_variance=(1/d, 1/N if N != 0 else 1.0),
        device=device,
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, config


def run_theory_solver(P: int, d: int, n1: int, chi: float, kappa: float, solver_path: Path):
    # Create temporary file for Julia solver output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cmd = [
            "julia",
            str(solver_path),
            "--P", str(P),
            "--d", str(d),
            "--n1", str(n1),
            "--chi", str(chi),
            "--kappa", str(kappa),
            "--output", tmp_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Read JSON from output file
        with open(tmp_path, 'r') as f:
            theory = json.load(f)
        lJ = float(theory.get("lJ")) if "lJ" in theory else None
        return theory, lJ
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def compute_eigs(model: FCN2NetworkActivationGeneric, d: int, P: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    X = torch.randn(P, d, device=device)
    # Use Y=X to project onto input coordinates; std=True gives ensemble std
    with torch.no_grad():
        eig_mean, eig_std = model.H_eig(X, X, std=True)
    # Standard error over ensemble members
    eig_se = eig_std / np.sqrt(float(model.ens))
    return eig_mean.cpu().numpy(), eig_se.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Compute H_eig across seeds with ensemble error bars vs theory")
    parser.add_argument("--base-pattern", type=str, required=True, help="Pattern with {seed} placeholder for run dirs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 10)), help="Seed numbers to evaluate")
    parser.add_argument("--eval-seed-offset", type=int, default=0, help="Offset added to each seed for eval data")
    parser.add_argument("--num-samples", type=int, default=None, help="Override P; default from config")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--theory-lJ", type=float, default=None, help="Theoretical lJ value (fallback if no JSON)")
    parser.add_argument("--theory-json", type=str, default=None, help="Path to theory JSON (e.g., output of compute_fcn2_erf_eigs.jl)")
    parser.add_argument("--solver", type=str, default=str(Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"), help="Path to Julia theory solver script")
    parser.add_argument("--kappa", type=float, default=1.0, help="Kappa parameter for theory solver")
    parser.add_argument("--run-theory", action="store_true", help="Run Julia theory solver if JSON/theory-lJ not provided")
    parser.add_argument("--output", type=str, default="multi_seed_eigs.png", help="Output plot path")
    parser.add_argument("--json-out", type=str, default="multi_seed_eigs.json", help="Output stats JSON path")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")

    eig_means = []
    eig_ses = []
    seeds_used = []
    run_dirs = []

    theory_json = None
    theory_lJ = args.theory_lJ
    theory_lJP = None
    if args.theory_json is not None:
        tpath = Path(args.theory_json)
        if tpath.exists():
            with open(tpath, "r") as f:
                theory_json = json.load(f)
            if theory_lJ is None and "lJ" in theory_json:
                theory_lJ = float(theory_json["lJ"])
            if "lJP" in theory_json:
                theory_lJP = float(theory_json["lJP"])
            print(f"Loaded theory from {tpath}: lJ={theory_lJ}, lJP={theory_lJP}")
        else:
            print(f"Theory JSON not found at {tpath}; will try solver if enabled")

    if theory_lJ is None and args.run_theory:
        # Use config of first available run to set solver args
        if args.theory_json is None:
            print("No theory JSON provided; running Julia solver...")
        # Load one config to get d, P, N, chi
        sample_seed = args.seeds[0]
        sample_dir = Path(str(args.base_pattern).replace("{seed}", str(sample_seed)))
        if not sample_dir.exists():
            raise FileNotFoundError(f"Cannot run solver: sample run dir missing: {sample_dir}")
        sample_meta = parse_run_meta_from_name(sample_dir.name)
        sample_model, sample_config = load_model(sample_dir, torch.device("cpu"))
        P_cfg = int(sample_config.get("P", sample_meta.get("P", 0)))
        d_cfg = int(sample_config.get("d", sample_meta.get("d", 0)))
        n1_cfg = int(sample_config.get("N", sample_meta.get("N", 0)))
        chi_cfg = float(sample_config.get("chi", sample_meta.get("chi", n1_cfg)))
        solver_path = Path(args.solver)
        theory_json, theory_lJ = run_theory_solver(P_cfg, d_cfg, n1_cfg, chi_cfg, args.kappa, solver_path)
        if "lJP" in theory_json:
            theory_lJP = float(theory_json["lJP"])
        print(f"Theory solver: lJ={theory_lJ}, lJP={theory_lJP}")

    for seed in args.seeds:
        run_dir = Path(str(args.base_pattern).replace("{seed}", str(seed)))
        if not run_dir.exists():
            print(f"Skipping missing run dir: {run_dir}")
            continue
        model, config = load_model(run_dir, torch.device("cpu"))
        meta = parse_run_meta_from_name(run_dir.name)
        P = args.num_samples if args.num_samples is not None else int(config.get("P", meta.get("P", 0)))
        d = int(config.get("d", meta.get("d", 0)))
        eval_seed = seed + args.eval_seed_offset
        eig_mean, eig_se = compute_eigs(model, d, P, eval_seed, torch.device("cpu"))
        eig_means.append(eig_mean)
        eig_ses.append(eig_se)
        seeds_used.append(seed)
        run_dirs.append(str(run_dir))
        print(f"Seed {seed}: eig_mean[0]={eig_mean[0]:.6e}")

    if not eig_means:
        raise RuntimeError("No runs processed; check base-pattern and seeds")

    # Stack eigenvalues across seeds: shape (num_seeds, P)
    eig_means_arr = np.stack(eig_means, axis=0)
    eig_ses_arr = np.stack(eig_ses, axis=0)

    num_seeds, num_eigs = eig_means_arr.shape
    x = np.arange(num_eigs)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in range(num_seeds):
        ax.errorbar(
            x,
            eig_means_arr[idx],
            yerr=eig_ses_arr[idx],
            fmt='o',
            markersize=3,
            alpha=0.6,
            elinewidth=0.8,
            capsize=2,
            label=f"seed {seeds_used[idx]}"
        )
    if theory_lJ is not None:
        ax.axhline(theory_lJ, color="orange", linestyle="--", linewidth=2, label=f"Theory lJ={theory_lJ:.4g}")
    if theory_lJP is not None:
        ax.axhline(theory_lJP, color="red", linestyle=":", linewidth=2, label=f"Theory lJP={theory_lJP:.4g}")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("H_eig mean ± SE (over ensembles)")
    ax.set_title("H_eig per eigenvalue with ensemble SE across seeds")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    stats = {
        "seeds": seeds_used,
        "run_dirs": run_dirs,
        "eig_mean": eig_means_arr.tolist(),
        "eig_se": eig_ses_arr.tolist(),
        "theory_lJ": theory_lJ,
        "theory_lJP": theory_lJP,
        "theory_json": theory_json,
        "solver_path": str(Path(args.solver)),
        "kappa": args.kappa,
    }
    with open(Path(args.json_out), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved plot to {out_path}")
    print(f"Saved stats to {args.json_out}")


if __name__ == "__main__":
    main()
