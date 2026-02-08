#!/usr/bin/env python3
"""Compute eigenvalues of the W0 weight covariance for each ensemble member.

For each run (d<d>_P<P>_N<N>_chi<chi>/seed<seed>/model.pt), load the model,
compute per-ensemble covariance eigenvalues of W0 (shape: ens x N x d), and
save the eigenvalues to <run>/plots/W0_cov_eigenvalues.npy. Also prints basic
stats (mean/min/max) for quick inspection.

Usage:
    python compute_W0_cov_eigenvalues.py --base-dir <runs_root> [--dims 150] [--suffix "_tag"]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric
from Experiment import Experiment

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories that contain model.pt and match the naming convention."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?).*"
    model_files = glob.glob(str(base / f"**/*seed*/model_final.pt"), recursive=True) 

    for model_file in model_files:

        seed_dir = Path(model_file).parent
        print(seed_dir)
        seed_name = seed_dir.name
        m_seed = re.match(r".*seed(\d+)", seed_name)
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
    """Load model from run directory."""
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


    if len(state_dict['W0'].shape) == 4:
        # 
        state_dict['W0'] = state_dict['W0'].squeeze(0)
        state_dict['W1'] = state_dict['W1'].squeeze(0)
        state_dict['A'] = state_dict['A'].squeeze(0)
    ens = int(state_dict['W0'].shape[0])
    print(state_dict['W0'].shape)
    print("Loading model from", model_path, f"with W0 shape {state_dict['W0'].shape} and ensemble size {ens}")
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


def compute_W0_cov_eigenvalues(model: FCN3NetworkActivationGeneric, d: int, N: int) -> np.ndarray:
    """Return eigenvalues of W0 covariance for each ensemble member.

    W0 shape: (ens, N, d)
    Output shape: (ens, d)
    """
    W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
    ens = W0.shape[0]
    eigvals_list = []
    for e in range(ens):
        W0_e = W0[e]  # (N, d)
        cov = np.cov(W0_e, rowvar=False)  # (d, d)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_list.append(eigvals)
    eigvals_array = np.array(eigvals_list)  # (ens, d)
    return eigvals_array


def plot_W0_cov_eigenvalues(
    eigvals: np.ndarray,
    cfg: Dict[str, int],
    out_path: Path,
    theory_lwt: Optional[float] = None,
    theory_lwp: Optional[float] = None,
) -> None:
    """Bar plot of mean eigenvalues with std shading and theory lines (lWT, lWP)."""
    try:
        eigvals_sorted = np.sort(eigvals, axis=1)[:, ::-1]
        mean_spec = eigvals_sorted.mean(axis=0)
        std_spec = eigvals_sorted.std(axis=0)
        idx = np.arange(mean_spec.shape[0])
        print(mean_spec[:10])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(idx, mean_spec, color='tab:blue', alpha=0.7, label='mean eigenvalue (ens avg)')
        ax.errorbar(idx, mean_spec, yerr=std_spec, fmt='none', ecolor='gray', elinewidth=1.0, capsize=2, label='±1 std')
        if theory_lwt is not None:
            ax.axhline(theory_lwt, color='tab:red', linestyle='--', linewidth=2.0, label=f'theory lWT={theory_lwt:.3e}')
        if theory_lwp is not None:
            ax.axhline(theory_lwp, color='tab:orange', linestyle='-.', linewidth=2.0, label=f'theory lWP={theory_lwp:.3e}')
        ax.set_yscale('log')
        ax.set_xlabel('Eigenvalue index (sorted)')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(
            f"W0 covariance eigenvalues (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})"
        )
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved W0 covariance eigenvalue plot to {out_path}")
    except Exception as e:
        print(f"  Warning: failed to plot W0 covariance eigenvalues for cfg={cfg}: {e}")


def compute_theory(run_dir: Path, cfg: Dict[str, int], device: torch.device) -> Dict[str, Optional[float]]:
    """Compute lWT/lWP theory using Experiment.eig_predictions.
    
    Loads kappa from config.json in the run directory for accuracy.
    """
    try:
        # Load config.json to get actual kappa used in training
        config_path = run_dir / "config.json"
        kappa = 1.0  # default
        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                config_data = json.load(f)
            kappa = float(config_data.get("kappa", 1.0))
        
        exp = Experiment(
            file=str(run_dir),
            kappa=kappa,
            N=int(cfg["N"]),
            d=int(cfg["d"]),
            chi=float(cfg["chi"]),
            P=int(cfg["P"]),
            ens=int(cfg.get("ens", 50)),
            device=device,
            eps=0.03,
        )

        preds = exp.eig_predictions()
        lwt = float(preds.lWT) if hasattr(preds, "lWT") and preds.lWT is not None else None
        lwp = float(preds.lWP) if hasattr(preds, "lWP") and preds.lWP is not None else None
        return {"lWT": lwt, "lWP": lwp}
    except Exception as e:
        print(f"  Warning: failed to compute theory for d={cfg.get('d')}, P={cfg.get('P')}: {e}")
        import traceback
        traceback.print_exc()
        return {"lWT": None, "lWP": None}


def main():
    parser = argparse.ArgumentParser(description="Compute W0 covariance eigenvalues for runs.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to do.")
        return

    for run_dir, cfg in runs:
        try:
            model = load_model(run_dir, cfg, device)
        except:
            print(f"Error loading model for {run_dir}, skipping. Exception:")
            import traceback
            traceback.print_exc()

        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue

        cfg_local = dict(cfg)
        cfg_local["ens"] = int(model.W0.shape[0])
        theory = compute_theory(run_dir, cfg_local, device)

        eigvals = compute_W0_cov_eigenvalues(model, cfg["d"], cfg["N"])
        out_dir = run_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "W0_cov_eigenvalues.npy"
        np.save(out_path, eigvals)

        print(
            f"Saved W0 covariance eigenvalues to {out_path} | shape {eigvals.shape}, "
            f"mean={eigvals.mean():.4e}, min={eigvals.min():.4e}, max={eigvals.max():.4e}"
        )

        plot_path = out_dir / "W0_cov_eigenvalues.png"
        plot_W0_cov_eigenvalues(eigvals, cfg, plot_path, theory_lwt=theory.get("lWT"), theory_lwp=theory.get("lWP"))


if __name__ == "__main__":
    main()
