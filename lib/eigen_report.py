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

def plot_projections_arxiv(results_map, t_vals, out_path: Path):
    """
    Create a high-quality, arxiv/thesis-ready plot of H3 projections only.
    Extracted from the top-left panel of the 4-panel plot.
    
    Parameters:
    -----------
    results_map : dict
        Results dictionary with 'h_proj' key containing measurement data
    t_vals : dict
        Theory values for comparison
    out_path : Path
        Output directory for saving the figure
    """
    try:
        print("[DEBUG] Starting plot_projections_arxiv...")
        print(f"[DEBUG] results_map keys: {results_map.keys()}")
        print(f"[DEBUG] output path: {out_path}")
        
        # Set up publication-quality matplotlib settings
        plt.rcParams.update({
            'font.size': 13,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.labelsize': 15,
            'axes.titlesize': 16,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'figure.titlesize': 18,
            'lines.linewidth': 1.5,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
        })
        
        meas_keys = ["lH1_T", "lH1_P", "lH3_T", "lH3_P"]
        data = results_map["h_proj"]
        
        if not data:
            print("[WARNING] No h_proj data available")
            return
    
    # Compute aggregated statistics
    means, errs = [], []
    all_values = {k: [] for k in meas_keys}
    
    for k in meas_keys:
        run_vals = [data[r][k][0] for r in data]
        all_values[k] = run_vals
        means.append(np.mean(run_vals))
        errs.append(np.std(run_vals) / np.sqrt(len(run_vals)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(meas_keys))
    width = 0.6
    
    # Scatter plot individual runs with transparency
    np.random.seed(42)  # For reproducibility of jitter
    for i, k in enumerate(meas_keys):
        run_vals = all_values[k]
        # Add small jitter to x position
        jitter = np.random.normal(0, 0.02, len(run_vals))
        ax.scatter(
            x[i] + jitter, run_vals,
            color='steelblue', alpha=0.35, s=60, zorder=2,
            edgecolors='none'
        )
    
    # Bar plot with error bars (empirical)
    bars = ax.bar(
        x, means, width=width,
        yerr=errs, capsize=8,
        color='steelblue', alpha=0.75,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5},
        label='Empirical (Mean ± SEM)',
        zorder=3
    )
    
    # Theory comparison - horizontal lines
    colors_theory = {'target': '#d62728', 'perp': '#ff7f0e'}
    linestyles = {'target': '-', 'perp': '--'}
    
    for i, k in enumerate(meas_keys):
        theory_val = t_vals[k]
        style_type = 'target' if 'T' in k else 'perp'
        ax.hlines(
            theory_val, x[i] - width/2 - 0.1, x[i] + width/2 + 0.1,
            colors=colors_theory[style_type],
            linestyles=linestyles[style_type],
            lw=2.2,
            zorder=4
        )
        
        # Add percentage gap label above each bar
        if theory_val is not None and theory_val > 0:
            pct_gap = 100.0 * (means[i] - theory_val) / theory_val
            # Position text 20% higher than the bar+error in log space
            y_pos = means[i] + errs[i]
            y_pos_log = np.log10(y_pos) + 0.15  # Add 0.15 in log10 space
            y_pos = 10 ** y_pos_log
            ax.text(x[i], y_pos, f'{pct_gap:+.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='none'))
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, alpha=0.75, label='Empirical (Mean ± SEM)'),
        Line2D([0], [0], color='#d62728', lw=2.2, linestyle='-',
               label='Theory (Target)'),
        Line2D([0], [0], color='#ff7f0e', lw=2.2, linestyle='--',
               label='Theory (Perpendicular)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=False, edgecolor='black', framealpha=0.95, fontsize=13)
    
    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels([r'$H^{(1)}_T$', r'$H^{(1)}_\perp$', 
                        r'$H^{(3)}_T$', r'$H^{(3)}_\perp$'],
                       fontsize=15)
    ax.set_ylabel(r'Log Projection Eigenvalue (log scale)', fontsize=15, labelpad=10)
    ax.set_yscale('log')
    
    # Grid styling
    ax.grid(axis='y', which='major', alpha=0.3, linestyle='-', linewidth=0.7)
    ax.grid(axis='y', which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set reasonable y-axis limits
    all_vals = np.concatenate([all_values[k] for k in meas_keys])
    ymin = np.min(all_vals) * 0.5
    ymax = np.max(all_vals) * 2
    ax.set_ylim(ymin, ymax)
    
    # Tight layout
    plt.tight_layout()
    
    # Save at multiple resolutions
    out_path.mkdir(parents=True, exist_ok=True)
    
    # High-quality PDF for thesis
    plt.savefig(out_path / "projections_arxiv.pdf", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # High-quality PNG as backup
    plt.savefig(out_path / "projections_arxiv.png", dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved arxiv-quality projections plot to:")
    print(f"  {out_path / 'projections_arxiv.pdf'}")
    print(f"  {out_path / 'projections_arxiv.png'}")
    
    plt.close()

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
        stats = compute_h3_projections_streaming(model, d, P_total=200_000_000, device=device)
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
    plt.close()

    # Generate high-quality arxiv/thesis version of projections plot
    plot_projections_arxiv(results_map, t_vals, out_path)

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