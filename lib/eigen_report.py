import os
import json
import subprocess
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import re
import hashlib
import datetime
from tqdm import tqdm

# Assuming these imports are available in your environment
from FCN2Network import FCN2NetworkActivationGeneric
from julia_theory import compute_fcn2_erf_cubic_eigs
from spectrum_svd import randomized_svd_spectrum, qb_projection_eigenvalues
from compute_h3_projections import compute_h3_projections_streaming

def load_model_from_run(run_dir: Path, device):
    d, N, P, chi = None, None, None, 1.0
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                d = int(cfg.get("d")) if cfg.get("d") is not None else None
                P = int(cfg.get("P", 0)) or None
                N = int(cfg.get("N")) if cfg.get("N") is not None else None
                chi = float(cfg.get("chi", 1.0))
        except Exception: pass

    state_paths = [run_dir / "model_final.pt", run_dir / "model.pt", run_dir / "checkpoint.pt"]
    sd = None
    for sp in state_paths:
        if sp.exists():
            sd_raw = torch.load(sp, map_location=device)
            sd = sd_raw["model_state_dict"] if isinstance(sd_raw, dict) and "model_state_dict" in sd_raw else sd_raw
            break
    
    if sd is None: raise FileNotFoundError(f"No state found in {run_dir}")
    
    w_shape = sd["W0"].shape
    ens, n1, d_weights = w_shape[0], w_shape[1], w_shape[2]
    d = d or d_weights
    N = N or n1
    
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=None, ens=ens, activation="erf",
        weight_initialization_variance=(1.0/d, 1.0/(N*chi)), device=device
    ).to(device)
    model.load_state_dict(sd)
    return model.double().eval()

def parse_run_params(run_dir: Path):
    # Standard parsing logic for hyperparameters
    d, P, N, chi, T, eps = None, None, None, None, None, None
    # _lr can be anything, so ignore it

    regex = r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi_(?P<chi>[\d.]+)_lr_(?P<lr>[\d.eE+-]+)_T_(?P<T>[\d.]+)_seed_(?P<seed>\d+)_eps_(?P<eps>[\d.]+)"

    m = re.match(regex, run_dir.name)

    if m:
        params = {
            "d": int(m.group("d")),
            "P": int(m.group("P")),
            "N": int(m.group("N")),
            "chi": float(m.group("chi")),
            "lr": float(m.group("lr")),
            "T": float(m.group("T")),
            "seed": int(m.group("seed")),
            "eps": float(m.group("eps"))
        }
    d,P,N,chi,T,eps = params.get("d"), params.get("P"), params.get("N"), params.get("chi"), params.get("T"), params.get("eps")
    # Try config.json if regex fails or to supplement
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)
            d = d or cfg.get("d")
            chi = chi or cfg.get("chi")
            eps = cfg.get("eps")
    kappa = T / 2.0 if T is not None else None
    return d, P, N, chi, kappa, eps

def get_model_H_eig(model, d, device):
    if hasattr(model, 'H_eig'):
        X = torch.randn(10000, d, device=device)
        eig_lin = model.H_eig(X, X).detach().sort(descending=True).values.cpu().numpy()
        X3 = (X**3 - 3.0*X)
        eig_he3 = model.H_eig(X, X3).detach().sort(descending=True).values.cpu().numpy()
        return np.concatenate([eig_lin, eig_he3])
    return None

def eigen_report(train_runs: List[str], out_dir: str = None):
    # 1. Directory setup
    run_paths = [Path(r).resolve() for r in train_runs]
    model_hash = hashlib.md5("".join(sorted([str(p) for p in run_paths])).encode()).hexdigest()[:10]
    out_path = Path(out_dir or f"eigen_report_{model_hash}")
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Aggregators: { method: { run_idx: { meas_name: (mean, std) } } }
    results_map = {m: {} for m in ["h_proj", "svd_eigs", "qb_proj_eigs", "H_eig"]}
    full_spectra = {m: [] for m in ["svd_eigs", "qb_proj_eigs", "H_eig"]}
    theory_data = []

    for idx, run_dir in enumerate(tqdm(run_paths, desc="Analyzing Models")):
        model = load_model_from_run(run_dir, device)
        d, P, N, chi, kappa, eps = parse_run_params(run_dir)
        
        # --- Computations ---
        # 1. Projections
        stats = compute_h3_projections_streaming(model, d, P_total=50_000_000, device=device)
        results_map["h_proj"][idx] = {
            "lH1_T": (stats["h1"]["target"]["second_moment"], 0.0),
            "lH1_P": (stats["h1"]["perp"]["second_moment"], 0.0),
            "lH3_T": (stats["h3"]["target"]["second_moment"], 0.0), "lH3_P": (stats["h3"]["perp"]["second_moment"], 0.0)
        }

        # 2. Spectral methods
        specs = {
            "svd_eigs": randomized_svd_spectrum(model, d, device, N=10000, k=9000, p=1000, chunk_size=2048),
            "qb_proj_eigs": qb_projection_eigenvalues(model, d, device, N=10000, k=9000, p=1000, chunk_size=2048)["all_eigenvalues"],
            "H_eig": get_model_H_eig(model, d, device)
        }

        for m_name, val in specs.items():
            if val is not None:
                arr = np.array(val)
                full_spectra[m_name].append(arr)
                # Logic: 0 (H1T), 1:d (H1P), d (H3T), d+1: (H3P)
                h1_p_vec = arr[1:d]
                h3_p_vec = arr[d+1:] if len(arr) > d+1 else np.array([0.0])
                results_map[m_name][idx] = {
                    "lH1_T": (arr[0], 0.0),
                    "lH1_P": (np.mean(h1_p_vec), np.std(h1_p_vec)),
                    "lH3_T": (arr[d] if len(arr) > d else 0.0, 0.0),
                    "lH3_P": (np.mean(h3_p_vec), np.std(h3_p_vec))
                }

        # Theory (Effective Kappa via Julia solver logic - simplified here)
        theory_eff = compute_fcn2_erf_cubic_eigs(d, N, P, chi, kappa, epsilon=eps or 0.0)
        theory_data.append(theory_eff)

    # ─────────────────────────────────────────────────────────
    # PLOTTING LOGIC
    # ─────────────────────────────────────────────────────────
    meas_keys = ["lH1_T", "lH1_P", "lH3_T", "lH3_P"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Reference theory (using first run)
    ref_theory = theory_data[0]
    t_vals = {
        "lH1_T": ref_theory["target"]["lJ1T"], "lH1_P": ref_theory["perpendicular"]["lJ1P"],
        "lH3_T": ref_theory["target"]["lJ3T"], "lH3_P": ref_theory["perpendicular"]["lJ3P"]
    }
    try: 

        # Save results using pickle
        import pickle
        with open(out_path / "eigen_report_results.pkl", "wb") as f:
            pickle.dump({
                "results_map": results_map,
                "theory_data": theory_data,
                "full_spectra": full_spectra,
                "t_vals": t_vals
            }, f)
        # Make results_map json serializable
        serializable_map = {
            m: {
                r: {k: [v[0], v[1]] for k, v in results_map[m][r].items()}
                for r in results_map[m]
            }
            for m in results_map
        }
        with open(out_path / "eigen_report_results.json", "w") as f:
            json.dump({
                "results_map": serializable_map,
                "theory_data": theory_data,
                "t_vals": t_vals
            }, f, indent=4)
    except Exception as e:
        print(f"Warning: could not save results pickle file: {e}")

    for i, m_name in enumerate(results_map.keys()):
        ax = axes[i]
        data = results_map[m_name]
        if not data: continue

        means, errs = [], []
        for k in meas_keys:
            run_vals = [data[r][k][0] for r in data]
            means.append(np.mean(run_vals))
            # Aggregate across models: sigma / sqrt(# models)
            errs.append(np.std(run_vals) / np.sqrt(len(run_vals)))

        # Also scatter plot the individual runs
        for r in meas_keys:
            run_vals = [data[r][k][0] for r in data]
            ax.scatter([meas_keys.index(r)]*len(run_vals), run_vals, color='gray', alpha=0.5, zorder=1, label='Individual Runs')

        x = np.arange(len(meas_keys))
        ax.bar(x, means, yerr=errs, capsize=6, color='teal', alpha=0.7, label='Empirical (SEM)')
        
        # Plot theory lines
        for j, k in enumerate(meas_keys):
            ax.hlines(t_vals[k], j-0.3, j+0.3, colors='red', linestyles='--', lw=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(meas_keys)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel("Eigenvalue Rank (log scale)")
        ax.set_title(f"Aggregated: {m_name}")
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path / "summary_4panel_bars.png")

    # Full Spectrum Plots
    for m_name, specs_list in full_spectra.items():
        if not specs_list: continue
        stack = np.stack([s[:min(map(len, specs_list))] for s in specs_list])
        m_spec, s_spec = np.mean(stack, axis=0), np.std(stack, axis=0) / np.sqrt(len(specs_list))
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(m_spec)), m_spec, yerr=s_spec, color='black', alpha=0.5)
        plt.axhline(t_vals["lH1_T"], color='r', label='H1 Target')
        plt.axhline(t_vals["lH3_T"], color='b', label='H3 Target')
        plt.yscale('log')
        # Plot perp as well
        plt.axhline(t_vals["lH1_P"], color='r', linestyle='--', label='H1 Perpendicular')
        plt.axhline(t_vals["lH3_P"], color='b', linestyle='--', label='H3 Perpendicular')
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue Magnitude")
        plt.yscale('log')
        plt.title(f"Full Spectrum: {m_name}")
        plt.legend()
        plt.savefig(out_path / f"spectrum_{m_name}.png")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='+')
    args = parser.parse_args()
    eigen_report(args.dirs)