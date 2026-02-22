#!/usr/bin/env python3
"""
FCN3-specific eigen-reporting utilities.

Uses `compute_h3_projections_fcn3.compute_h3_projections_streaming` and
`lib/spectrum_svd_fcn3` routines to produce projection statistics and
spectral estimates for saved FCN3 runs.

This is intentionally separate from `lib/eigen_report.py`.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import re
import subprocess
import tempfile
import json as json_lib
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
import sys
sys.path.append('/home/akiva/FCNX-Ensembling/lib')


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def compute_kappa_eff(d: int, P: int, kappa: float, n_samples: int = 5000):
    """Compute effective ridge by running self-consistent kappa solver using arcsin kernel eigenvalues."""
    try:
        # Compute n_samples x n_samples arcsin kernel eigenvalues
        np.random.seed(0)
        X = np.random.randn(n_samples, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / n_samples
        
        # Run self-consistent solver
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        eigenvalues = eigvals.tolist()
        with open(eig_json, "w") as f:
            json_lib.dump({"eigenvalues": eigenvalues, "kappa_bare": kappa}, f)
        
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
        sc_cmd = [
            "julia", str(julia_script),
            eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True)
        
        # Extract kappa_eff from output
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
            print("Kappa eff calculated: ", kappa_eff)
            return kappa_eff
        else:
            return kappa
    except Exception as e:
        print(f"Warning: kappa_eff computation failed: {e}. Using bare kappa.")
        return kappa
    finally:
        try:
            if 'eig_json' in locals() and Path(eig_json).exists():
                Path(eig_json).unlink()
        except:
            pass


def _import_compute_h3_module(base_dir: Path):
    p = base_dir / "compute_h3_projections_fcn3.py"
    if not p.exists():
        raise FileNotFoundError(f"Expected {p} to exist")
    spec = importlib.util.spec_from_file_location("compute_h3_proj_fcn3", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot_projections_arxiv_fcn3(results_map, t_vals, out_path: Path, param_title: str = ""):
    """
    Create a high-quality, arxiv/thesis-ready plot of H3 projections only.
    FCN3-specific version using theoretical values as reference.
    
    Parameters:
    -----------
    results_map : dict
        Results dictionary with 'h_proj' key containing measurement data
    t_vals : dict
        Theory values with keys "lH1T", "lH1P", "lH3T", "lH3P"
    out_path : Path
        Output directory for saving the figure
    param_title : str
        Parameter title for the figure (e.g., "d=10, P=100")
    """
    try:
        print("[DEBUG] Starting plot_projections_arxiv_fcn3...")
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
        mode_meta = {
            "lH1_T": {"mode": r"\mathrm{He}_{1}", "state": r"*", "theory_color": "#d62728"},
            "lH1_P": {"mode": r"\mathrm{He}_{1}", "state": r"\perp", "theory_color": "#ff7f0e"},
            "lH3_T": {"mode": r"\mathrm{He}_{3}", "state": r"*", "theory_color": "#9467bd"},
            "lH3_P": {"mode": r"\mathrm{He}_{3}", "state": r"\perp", "theory_color": "#2ca02c"},
        }

        def _sci_latex(val: float, digits: int = 2) -> str:
            if val == 0 or not np.isfinite(val):
                return "0"
            exponent = int(np.floor(np.log10(abs(val))))
            mantissa = val / (10 ** exponent)
            return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"

        data = results_map.get("h_proj", {})
        
        if not data:
            print("[WARNING] No h_proj data available")
            return
        
        # Compute aggregated statistics
        means, errs = [], []
        all_values = {k: [] for k in meas_keys}
        print(data)
        for k in meas_keys:
            run_vals = [float(data[r][k][0]) for r in data if k in data[r]]
            all_values[k] = run_vals
            if run_vals:
                means.append(np.mean(run_vals))
                errs.append(np.std(run_vals) / np.sqrt(len(run_vals)) if len(run_vals) > 1 else 0.0)
            else:
                means.append(0.0)
                errs.append(0.0)
        
        print(f"[DEBUG] Computed statistics: {len(means)} measurements")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(meas_keys))
        width = 0.6
        
        # Scatter plot individual runs with transparency
        np.random.seed(42)  # For reproducibility of jitter
        for i, k in enumerate(meas_keys):
            run_vals = all_values[k]
            if run_vals:
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
        linestyles = {'target': '-', 'perp': '--'}
        
        for i, k in enumerate(meas_keys):
            theory_val = t_vals.get(k)
            if theory_val is not None:
                style_type = 'target' if 'T' in k else 'perp'
                ax.hlines(
                    theory_val, x[i] - width/2 - 0.1, x[i] + width/2 + 0.1,
                    colors=mode_meta[k]["theory_color"],
                    linestyles=linestyles[style_type],
                    lw=2.2,
                    zorder=4
                )
                
                # Add percentage gap label above each bar
                if theory_val > 0:
                    pct_gap = 100.0 * (means[i] - theory_val) / theory_val
                    print("Computing Percentage for: ", k)
                    print(means[i], theory_val)
                    # Position text 20% higher than the bar+error in log space
                    y_pos = means[i] + errs[i]
                    y_pos_log = np.log10(y_pos) + 0.15  # Add 0.15 in log10 space
                    y_pos = 10 ** y_pos_log
                    ax.text(x[i], y_pos, f'{pct_gap:+.1f}%', ha='center', va='bottom', 
                           fontsize=11, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='none'))
        
        # Create custom legend with per-mode scientific notation values
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = []
        for i, k in enumerate(meas_keys):
            mode = mode_meta[k]["mode"]
            state = mode_meta[k]["state"]
            emp_val = means[i]
            th_val = t_vals.get(k)
            legend_elements.append(
                Patch(
                    facecolor='steelblue',
                    alpha=0.75,
                    label=rf"Empirical $\lambda^{{{mode}}}_{{{state}}}={_sci_latex(emp_val)}$"
                )
            )
            if th_val is not None:
                style_type = 'target' if 'T' in k else 'perp'
                legend_elements.append(
                    Line2D(
                        [0], [0],
                        color=mode_meta[k]["theory_color"],
                        lw=2.2,
                        linestyle=linestyles[style_type],
                        label=rf"Theory $\lambda^{{{mode}}}_{{{state}}}={_sci_latex(float(th_val))}$"
                    )
                )
        ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                  fancybox=False, edgecolor='black', framealpha=0.95, fontsize=11, ncol=2)
        
        # Customize axes
        ax.set_xticks(x)
        ax.set_xticklabels([
            r'$\lambda^{\mathrm{He}_{1}}_{*}$',
            r'$\lambda^{\mathrm{He}_{1}}_{\perp}$',
            r'$\lambda^{\mathrm{He}_{3}}_{*}$',
            r'$\lambda^{\mathrm{He}_{3}}_{\perp}$'
        ],
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
        all_vals_concat = np.concatenate([np.array(all_values[k]) for k in meas_keys if all_values[k]])
        if len(all_vals_concat) > 0:
            ymin = np.min(all_vals_concat) * 0.5
            ymax = np.max(all_vals_concat) * 2
            ax.set_ylim(ymin, ymax)
        
        # Add parameter title if provided
        if param_title:
            fig.suptitle(f"H³ Projections — {param_title}", fontsize=16, y=0.98)
        
        # Tight layout
        plt.tight_layout()
        
        # Save at multiple resolutions
        out_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[DEBUG] Created output directory: {out_path}")
        
        # High-quality PDF for thesis
        pdf_path = out_path / "projections_arxiv.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"[DEBUG] Saved PDF to {pdf_path}")
        
        # High-quality PNG as backup
        png_path = out_path / "projections_arxiv.png"
        plt.savefig(png_path, dpi=400, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[DEBUG] Saved PNG to {png_path}")
        
        print(f"✓ Saved arxiv-quality projections plot to:")
        print(f"  {pdf_path}")
        print(f"  {png_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"[ERROR] Failed to create projections plot: {e}")
        import traceback
        traceback.print_exc()
        return


def eigen_report_fcn3(train_runs: List[str], out_dir: str = None, force_recompute: bool = False):
    base_dir = Path(__file__).parent
    compute_mod = _import_compute_h3_module(base_dir)
    import spectrum_svd_fcn3 as spec_fcn3

    run_paths = [Path(r).resolve() for r in train_runs]
    # Default output directory: inside the first run directory
    if out_dir is not None:
        out_path = Path(out_dir)
    else:
        out_path = run_paths[0] / "eigen_report_fcn3"
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse model parameters early so we can use them in plot titles
    d0, P0, N0, chi0, kappa0, eps0 = compute_mod.parse_run_params(run_paths[0])
    param_title = ""
    if d0 is not None:
        parts = []
        parts.append(rf"$d={d0}$")
        if P0 is not None:
            parts.append(rf"$P={P0}$")
        if N0 is not None:
            parts.append(rf"$N={N0}$")
        if chi0 is not None:
            parts.append(rf"$\chi={chi0}$")
        if kappa0 is not None:
            parts.append(rf"$\kappa={kappa0}$")
        if eps0 is not None and eps0 != 0:
            parts.append(rf"$\epsilon={eps0}$")
        param_title = r",".join(parts)

    results_map = {m: {} for m in ["h_proj", "svd_eigs", "qb_proj_eigs", "H_eig"]}
    full_spectra = {m: [] for m in ["svd_eigs", "qb_proj_eigs", "H_eig"]}
    theory_data = []
    w0_emp_lwt = []
    w0_emp_lwp = []

    cache_json = out_path / "eigen_report_fcn3.json"
    cache_pkl = out_path / "eigen_report_fcn3.pkl"

    # Theory lines from EOS solver (target + perpendicular)
    theory_lh1t = None
    theory_lh3t = None
    theory_lh1p = None
    theory_lh3p = None
    theory_lwt = None
    theory_lwp = None
    kappa_eff = kappa0

    # Load cache if present and recomputation not forced
    if cache_json.exists() and not force_recompute:
        try:
            with open(cache_json, "r") as f:
                cached = json.load(f)
            results_map = cached.get("results_map", results_map)
            full_spectra = cached.get("full_spectra", full_spectra)
            theory_lines = cached.get("theory_lines", {})
            theory_lh1t = theory_lines.get("lH1T")
            theory_lh3t = theory_lines.get("lH3T")
            theory_lh1p = theory_lines.get("lH1P")
            theory_lh3p = theory_lines.get("lH3P")
            theory_lwt = theory_lines.get("lWT")
            theory_lwp = theory_lines.get("lWP")
            kappa_eff = theory_lines.get("kappa_eff", kappa0)
            w0_emp_lwt = cached.get("w0_emp_lwt", w0_emp_lwt)
            w0_emp_lwp = cached.get("w0_emp_lwp", w0_emp_lwp)
        except Exception:
            pass

    # If cached projections are used, still compute empirical W0 values if missing
    if (cache_json.exists() and not force_recompute and
        (not w0_emp_lwt and not w0_emp_lwp) and run_paths):
        for idx, run_dir in enumerate(run_paths):
            try:
                model = compute_mod.load_model_from_run(Path(run_dir), device)
                d = int(getattr(model, "d", None) or 0)
                W0 = getattr(model, "W0", None)
                if W0 is not None and d > 0:
                    W0_flat = W0.detach().double().reshape(-1, d)
                    cov = (W0_flat.T @ W0_flat) / W0_flat.shape[0]
                    eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()[::-1]
                    if eigvals.size > 0:
                        w0_emp_lwt.append(float(eigvals[0]))
                    if eigvals.size > 1:
                        w0_emp_lwp.append(float(np.mean(eigvals[1:])))
            except Exception as e:
                print(f"Warning: could not compute empirical W0 eigenvalues for cached run {idx}: {e}")

    # When using cached projections (no force_recompute), override lH1_T with H_eig
    if cache_json.exists() and not force_recompute and run_paths:
        for idx, run_dir in enumerate(run_paths):
            try:
                model = compute_mod.load_model_from_run(Path(run_dir), device)
                d = int(getattr(model, "d", None) or 0)
                if d <= 0 or not hasattr(model, "H_eig"):
                    continue
                n_samples = 5000 * d
                with torch.no_grad():
                    X = torch.randn(2000, d, device=device)
                    eig_lin = model.H_eig(X, X).detach().sort(descending=True).values
                    lH1T_val = float(eig_lin[0].cpu().item()) if eig_lin.numel() > 0 else None
                if lH1T_val is not None:
                    results_map.setdefault("h_proj", {})
                    results_map["h_proj"].setdefault(idx, {})
                    results_map["h_proj"][idx]["lH1_T"] = (lH1T_val, 0.0)
            except Exception as e:
                print(f"Warning: could not override lH1_T from H_eig for cached run {idx}: {e}")

    if force_recompute or not cache_json.exists():
        if run_paths:
            if d0 is not None and P0 is not None and N0 is not None and chi0 is not None and kappa0 is not None:
                try:
                    print("Computing kappa_eff via self-consistent solver...")
                    kappa_eff = compute_kappa_eff(d0, P0, kappa0)
                    print(f"  kappa_bare = {kappa0:.6f}, kappa_eff = {kappa_eff:.6f}")
                    print("Computing Theory with parameters: ")
                    print(f"d: {d0} N: {N0}, P:{P0}, chi: {chi0}, kappa: {kappa_eff}, epsilon: {eps0}")
                    theory = compute_mod.compute_theory_with_julia(d0, N0, P0, chi0, kappa_eff, eps0)
                    print("Theory compute:")
                    print(theory)
                    target_block = theory.get("target", {}) if isinstance(theory, dict) else {}
                    perp_block = theory.get("perpendicular", {}) if isinstance(theory, dict) else {}
                    lH1T = target_block.get("lH1T")
                    lH3T = target_block.get("lH3T")
                    lH1P = perp_block.get("lH1P")
                    lH3P = perp_block.get("lH3P")
                    lWT = target_block.get("lWT")
                    lWP = perp_block.get("lWP")
                    if lH1T is not None:
                        theory_lh1t = float(lH1T)
                    if lH3T is not None:
                        theory_lh3t = float(lH3T) #/ 6.0
                    if lH1P is not None:
                        theory_lh1p = float(lH1P)
                    if lH3P is not None:
                        theory_lh3p = float(lH3P) #/ 6.0
                    if lWT is not None:
                        theory_lwt = float(lWT)
                    if lWP is not None:
                        theory_lwp = float(lWP)
                except Exception:
                    pass

    if force_recompute or not cache_json.exists():
        for idx, run_dir in enumerate(tqdm(run_paths, desc="Analyzing FCN3 Models")):
            run_dir = Path(run_dir)
            # reuse loader from compute_h3_projections_fcn3
            model = compute_mod.load_model_from_run(run_dir, device)

            # infer d from model
            d = int(getattr(model, "d", None) or 0)
            print(f"Run {idx}: Loaded model from {run_dir} with inferred d={d}")
            print("Computing projections and spectra...")
            # Empirical W0 covariance eigenvalues (pooled across ensembles)
            try:
                W0 = getattr(model, "W0", None)
                if W0 is not None and d > 0:
                    W0_flat = W0.detach().double().reshape(-1, d)
                    cov = (W0_flat.T @ W0_flat) / W0_flat.shape[0]
                    eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()[::-1]
                    if eigvals.size > 0:
                        w0_emp_lwt.append(float(eigvals[0]))
                    if eigvals.size > 1:
                        w0_emp_lwp.append(float(np.mean(eigvals[1:])))
            except Exception as e:
                print(f"Warning: could not compute empirical W0 eigenvalues: {e}")
            # Measure how long it takes to compute the projections and spectra for this run
            timestart = torch.cuda.Event(enable_timing=True)
            timeend = torch.cuda.Event(enable_timing=True)
            timestart.record()
            # 1) large-batch H3 projections (use 200 million total samples as requested)
            stats = compute_mod.compute_h3_projections_streaming(model, d, P_total=100_000_000, batch_size=10000, device=device)
            results_map["h_proj"][idx] = {
                "lH1_T": (stats["h1"]["target"]["second_moment"], 0.0),
                "lH1_P": (stats["h1"]["perp"]["second_moment"], 0.0),
                "lH3_T": (stats["h3"]["target"]["second_moment"], 0.0),
                "lH3_P": (stats["h3"]["perp"]["second_moment"], 0.0),
            }
            print(results_map["h_proj"][idx])
            # Show how long it took
            timeend.record()
            torch.cuda.synchronize()
            elapsed = timestart.elapsed_time(timeend) / 1000.0
            print(f"Projections computed in {elapsed:.2f} seconds")

            # 2) spectral estimates using FCN3-specific routines
            try:
                svd_vals = spec_fcn3.randomized_svd_spectrum(model, d, device, N=400, k=100, p=50, chunk_size=100)
            except Exception:
                svd_vals = None
            try:
                qb_res = spec_fcn3.qb_projection_eigenvalues(model, d, device, N=400, k=100, p=50, chunk_size=100)
                qb_vals = qb_res.get("all_eigenvalues")
            except Exception:
                qb_vals = None

            specs = {"svd_eigs": svd_vals, "qb_proj_eigs": qb_vals, "H_eig": None}

            for m_name, val in specs.items():
                if val is not None:
                    arr = np.array(val)
                    full_spectra[m_name].append(arr)
                    h1_p_vec = arr[1:d] if len(arr) > 1 else np.array([0.0])
                    h3_p_vec = arr[d+1:] if len(arr) > d+1 else np.array([0.0])
                    n_h1p = len(h1_p_vec) if len(h1_p_vec) > 0 else 1
                    n_h3p = len(h3_p_vec) if len(h3_p_vec) > 0 else 1
                    results_map[m_name][idx] = {
                        "lH1_T": (arr[0], 0.0),
                        "lH1_P": (np.mean(h1_p_vec), np.std(h1_p_vec) / np.sqrt(n_h1p)),
                        "lH3_T": (arr[d] if len(arr) > d else 0.0, 0.0),
                        "lH3_P": (np.mean(h3_p_vec), np.std(h3_p_vec) / np.sqrt(n_h3p)),
                    }

    # Save results (make JSON serializable)
    try:
        import pickle
        with open(out_path / "eigen_report_fcn3.pkl", "wb") as f:
            pickle.dump({
                "results_map": results_map,
                "full_spectra": full_spectra,
                "w0_emp_lwt": w0_emp_lwt,
                "w0_emp_lwp": w0_emp_lwp,
                "theory_lines": {
                    "lH1T": theory_lh1t,
                    "lH3T": theory_lh3t,
                    "lH1P": theory_lh1p,
                    "lH3P": theory_lh3p,
                    "lWT": theory_lwt,
                    "lWP": theory_lwp,
                    "kappa_eff": kappa_eff,
                },
            }, f)

        # Convert results_map to JSON-serializable structure
        serializable_map = {}
        for m, runs in results_map.items():
            serializable_map[m] = {}
            for r_idx, metrics in runs.items():
                serializable_map[m][str(r_idx)] = {}
                for k, v in metrics.items():
                    try:
                        val0 = float(v[0])
                    except Exception:
                        val0 = v[0]
                    try:
                        val1 = float(v[1])
                    except Exception:
                        val1 = v[1]
                    serializable_map[m][str(r_idx)][k] = [val0, val1]

        # Also convert simple full_spectra to lists of lists (truncate large arrays)
        serial_full = {}
        for m, specs_list in full_spectra.items():
            serial_full[m] = [s[:1000].tolist() if hasattr(s, 'tolist') else list(s) for s in specs_list]

        with open(out_path / "eigen_report_fcn3.json", "w") as f:
            json.dump({
                "results_map": serializable_map,
                "full_spectra": serial_full,
                "w0_emp_lwt": w0_emp_lwt,
                "w0_emp_lwp": w0_emp_lwp,
                "theory_lines": {
                    "lH1T": theory_lh1t,
                    "lH3T": theory_lh3t,
                    "lH1P": theory_lh1p,
                    "lH3P": theory_lh3p,
                    "lWT": theory_lwt,
                    "lWP": theory_lwp,
                    "kappa_eff": kappa_eff,
                },
            }, f, indent=2)
    except Exception as e:
        print("Warning: could not save results:", e)

    # Minimal plotting of aggregated means and summary 4-panel bars
    meas_keys = ["lH1_T", "lH1_P", "lH3_T", "lH3_P"]

    # Summary 4-panel bars: one panel per method in results_map
    methods = list(results_map.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, m_name in enumerate(methods):
        ax = axes[i]
        data = results_map.get(m_name, {})
        if not data:
            ax.set_title(m_name + ' (no data)')
            continue
        means, errs = [], []
        for key in meas_keys:
            run_vals = [data[r][key][0] for r in data if key in data[r]]
            run_stds = [float(data[r][key][1]) for r in data if key in data[r]]
            if len(run_vals) == 0:
                means.append(0.0)
                errs.append(0.0)
            else:
                run_vals = [float(x) for x in run_vals]
                mean_val = np.mean(run_vals)
                means.append(mean_val)
                # Use degeneracy-based std for degenerate eigenvalues,
                # cross-run SEM for non-degenerate ones
                avg_degeneracy_std = np.mean(run_stds) 
                if avg_degeneracy_std > 0:
                    errs.append(avg_degeneracy_std)
                else:
                    errs.append(np.std(run_vals) / np.sqrt(len(run_vals)))

        x = np.arange(len(meas_keys))
        ax.bar(x, means, yerr=errs, capsize=6, color='teal', alpha=0.7)
        
        # Add percentage gap labels above each bar using theory_for_key
        theory_for_key = {
            "lH1_T": theory_lh1t,
            "lH1_P": theory_lh1p,
            "lH3_T": theory_lh3t,
            "lH3_P": theory_lh3p,
        }
        for xi, (key, mean_val, err_val) in enumerate(zip(meas_keys, means, errs)):
            theo_val = theory_for_key.get(key)
            if theo_val is not None and theo_val > 0:
                percent_gap = abs(mean_val - theo_val) / theo_val * 100
                y_pos = mean_val + err_val + 0.10 * max(means)
                ax.text(xi, y_pos, f'{percent_gap:.1f}%', ha='center', va='bottom', fontsize=8, color='darkred', weight='bold')
        
        # Add theory lines (target + perpendicular) across all panels
        if theory_lh1t is not None:
            ax.axhline(theory_lh1t, color="#E45756", linestyle="--", label="Theory lH1T")
        if theory_lh3t is not None:
            ax.axhline(theory_lh3t, color="#F58518", linestyle=":", label="Theory lH3T/6")
        if theory_lh1p is not None:
            ax.axhline(theory_lh1p, color="#54A24B", linestyle="-.", label="Theory lH1P")
        if theory_lh3p is not None:
            ax.axhline(theory_lh3p, color="#B279A2", linestyle=(0, (1, 2)), label="Theory lH3P/6")
        ax.set_xticks(x)
        ax.set_xticklabels(meas_keys)
        ax.margins(x=0.15)  # Add left/right margin to prevent label cutoff
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(m_name)
        ax.grid(axis='y', alpha=0.3)
        if (theory_lh1t is not None or theory_lh3t is not None or
            theory_lh1p is not None or theory_lh3p is not None):
            ax.legend()

    fig.suptitle(f"Method Summary — {param_title}", fontsize=13)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path / "summary_4panel_bars.png", dpi=300)
    print("Saved summary plot:", out_path / "summary_4panel_bars.png")
    plt.close()

    # Generate high-quality arxiv/thesis version of projections plot
    t_vals = {
        "lH1_T": theory_lh1t,
        "lH1_P": theory_lh1p,
        "lH3_T": theory_lh3t,
        "lH3_P": theory_lh3p,
    }
    plot_projections_arxiv_fcn3(results_map, t_vals, out_path, param_title)

    # Per-eigenvalue comparison plot: one subplot per eigenvalue, bars per method
    theory_for_key = {
        "lH1_T": theory_lh1t,
        "lH1_P": theory_lh1p,
        "lH3_T": theory_lh3t,
        "lH3_P": theory_lh3p,
    }
    method_colors = {
        "h_proj": "#4C78A8",
        "svd_eigs": "#F58518",
        "qb_proj_eigs": "#54A24B",
        "H_eig": "#E45756",
    }
    fig_eig, axes_eig = plt.subplots(2, 2, figsize=(14, 10))
    axes_eig = axes_eig.flatten()
    for ki, key in enumerate(meas_keys):
        ax = axes_eig[ki]
        bar_labels = []
        bar_vals = []
        bar_errs = []
        bar_colors = []
        for m_name in methods:
            data = results_map.get(m_name, {})
            run_vals = [float(data[r][key][0]) for r in data if key in data[r]]
            run_stds = [float(data[r][key][1]) for r in data if key in data[r]]
            if not run_vals:
                continue
            bar_labels.append(m_name)
            bar_vals.append(np.mean(run_vals))
            # Use degeneracy spread as error bar when available
            avg_deg_std = np.mean(run_stds)
            if avg_deg_std > 0:
                bar_errs.append(avg_deg_std)
            else:
                bar_errs.append(np.std(run_vals) / np.sqrt(len(run_vals)) if len(run_vals) > 1 else 0.0)
            bar_colors.append(method_colors.get(m_name, "gray"))
        if bar_vals:
            x = np.arange(len(bar_vals))
            ax.bar(x, bar_vals, yerr=bar_errs, capsize=6, color=bar_colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=25, ha="right", fontsize=9)
            ax.margins(x=0.15)  # Add left/right margin to prevent label cutoff
            
            # Add percentage gap labels above each bar
            theo_val = theory_for_key.get(key)
            if theo_val is not None and theo_val > 0:
                for xi, (bar_val, bar_err) in enumerate(zip(bar_vals, bar_errs)):
                    percent_gap = abs(bar_val - theo_val) / theo_val * 100
                    # Position label above the bar (including error bar)
                    y_pos = bar_val + bar_err + 0.10 * max(bar_vals)
                    ax.text(xi, y_pos, f'{percent_gap:.1f}%', ha='center', va='bottom', fontsize=8, color='darkred', weight='bold')
        
        theo_val = theory_for_key.get(key)
        if theo_val is not None:
            ax.axhline(theo_val, color="red", linestyle="--", linewidth=2, label=f"Theory {key}")
            ax.legend()
        ax.set_title(key)
        ax.set_ylabel("Eigenvalue")
        ax.grid(axis='y', alpha=0.3)
    fig_eig.suptitle(f"Per-Eigenvalue Comparison — {param_title}", fontsize=13)
    fig_eig.tight_layout(pad=1.5)
    fig_eig.savefig(out_path / "per_eigenvalue_comparison.png", dpi=300)
    print("Saved per-eigenvalue plot:", out_path / "per_eigenvalue_comparison.png")

    # Spectrum plots per method
    for m_name, specs_list in full_spectra.items():
        if not specs_list:
            continue
        stack = np.stack([s[:min(map(len, specs_list))] for s in specs_list])
        m_spec = np.mean(stack, axis=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(m_spec)), m_spec, color='steelblue', alpha=0.7, label='Empirical')
        
        # Add theory lines for qb_proj_eigs and svd_eigs
        if m_name in ("qb_proj_eigs", "svd_eigs"):
            theory_vals = [theory_lh1t, theory_lh1p, theory_lh3t, theory_lh3p]
            theory_names = ['lH1T', 'lH1P', 'lH3T', 'lH3P']
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
            
            # Add horizontal lines at the theoretical values
            if theory_lh1t is not None:
                ax.axhline(theory_lh1t, color=colors[0], linestyle='--', linewidth=2, label=f'{theory_names[0]}={theory_lh1t:.3f}', alpha=0.7)
            if theory_lh3t is not None:
                ax.axhline(theory_lh3t, color=colors[2], linestyle='--', linewidth=2, label=f'{theory_names[2]}={theory_lh3t:.3f}', alpha=0.7)
            if theory_lh1p is not None:
                ax.axhline(theory_lh1p, color=colors[1], linestyle=':', linewidth=2, label=f'{theory_names[1]}={theory_lh1p:.3f}', alpha=0.7)
            if theory_lh3p is not None:
                ax.axhline(theory_lh3p, color=colors[3], linestyle=':', linewidth=2, label=f'{theory_names[3]}={theory_lh3p:.3f}', alpha=0.7)
        
        ax.set_yscale('log')
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_title(f"Full Spectrum: {m_name}\n{param_title}", fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path / f"spectrum_{m_name}.png", dpi=300)
        plt.close()
    
    # W0 covariance eigenvalues plot (combining with spectrum visualization)
    if (theory_lwt is not None or theory_lwp is not None or
        w0_emp_lwt or w0_emp_lwp):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create placeholders for W0 eigenvalues
        x_pos = [0, d0 - 0.5] if d0 and d0 > 1 else [0, 0.5]
        labels = ['lWT (target)', f'lWP (perp avg)']
        w0_theory = [theory_lwt if theory_lwt is not None else 0, 
                     theory_lwp if theory_lwp is not None else 0]
        colors_w0 = ['#d62728', '#ff7f0e']
        emp_colors = ['#1f77b4', '#2ca02c']
        
        # Plot theory lines for W0
        if theory_lwt is not None:
            ax.axhline(theory_lwt, color=colors_w0[0], linestyle='--', linewidth=2, alpha=0.5,
                       label=f'Theory {labels[0]}={theory_lwt:.3f}')
        
        if theory_lwp is not None:
            ax.axhline(theory_lwp, color=colors_w0[1], linestyle=':', linewidth=2, alpha=0.5,
                       label=f'Theory {labels[1]}={theory_lwp:.3f}')

        # Plot empirical W0 values
        if w0_emp_lwt:
            jitter = np.random.normal(0, 0.02, len(w0_emp_lwt))
            ax.scatter(np.full(len(w0_emp_lwt), x_pos[0]) + jitter, w0_emp_lwt,
                       s=60, alpha=0.5, color=emp_colors[0], label='Empirical lWT samples')
            mean_lwt = float(np.mean(w0_emp_lwt))
            sem_lwt = float(np.std(w0_emp_lwt) / np.sqrt(len(w0_emp_lwt)))
            ax.errorbar(x_pos[0], mean_lwt, yerr=sem_lwt, fmt='o', color=emp_colors[0],
                        capsize=5, label='Empirical lWT mean')
            if theory_lwt is not None and mean_lwt > 0:
                pct_gap = 100.0 * (mean_lwt - theory_lwt) / theory_lwt
                y_pos_log = np.log10(mean_lwt + sem_lwt) + 0.12
                ax.text(x_pos[0], 10 ** y_pos_log, f'{pct_gap:+.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6, edgecolor='none'))

        if w0_emp_lwp:
            jitter = np.random.normal(0, 0.02, len(w0_emp_lwp))
            ax.scatter(np.full(len(w0_emp_lwp), x_pos[1]) + jitter, w0_emp_lwp,
                       s=60, alpha=0.5, color=emp_colors[1], label='Empirical lWP samples')
            mean_lwp = float(np.mean(w0_emp_lwp))
            sem_lwp = float(np.std(w0_emp_lwp) / np.sqrt(len(w0_emp_lwp)))
            ax.errorbar(x_pos[1], mean_lwp, yerr=sem_lwp, fmt='o', color=emp_colors[1],
                        capsize=5, label='Empirical lWP mean')
            if theory_lwp is not None and mean_lwp > 0:
                pct_gap = 100.0 * (mean_lwp - theory_lwp) / theory_lwp
                y_pos_log = np.log10(mean_lwp + sem_lwp) + 0.12
                ax.text(x_pos[1], 10 ** y_pos_log, f'{pct_gap:+.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6, edgecolor='none'))
        
        ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
        ax.set_title(f"W0 Covariance Eigenvalues\n{param_title}", fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        y_candidates = [v for v in w0_theory if v is not None and v > 0]
        y_candidates.extend([v for v in w0_emp_lwt if v > 0])
        y_candidates.extend([v for v in w0_emp_lwp if v > 0])
        if y_candidates:
            ax.set_ylim(bottom=min(y_candidates) * 0.5, top=max(y_candidates) * 2)
        fig.tight_layout()
        fig.savefig(out_path / "spectrum_w0_theory.png", dpi=300)
        plt.close()
        print("Saved W0 covariance theory plot:", out_path / "spectrum_w0_theory.png")


def eigen_report_seed_aggregate(seed_parent_dir: str, out_dir: str = None, force_recompute: bool = False):
    """
    Aggregate h3 projection eigenvalues across seed subdirectories.
    Plots lH1T, lH1P, lH3T, lH3P as scatter dots per seed (color-coded),
    with mean dots and theory lines.
    
    Args:
        seed_parent_dir: Path to parent directory containing seed1, seed2, ... subdirs
        out_dir: Output directory for plots (defaults to seed_parent_dir/eigen_report_aggregate)
        force_recompute: Whether to force recomputation of theory values
    """
    base_dir = Path(__file__).parent
    compute_mod = _import_compute_h3_module(base_dir)
    
    seed_parent = Path(seed_parent_dir).resolve()
    if not seed_parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {seed_parent}")
    
    # Find all seed<x> subdirectories
    seed_dirs = sorted([d for d in seed_parent.glob("seed*") if d.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found in {seed_parent}")
    
    print(f"Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")
    
    # Default output directory
    if out_dir is not None:
        out_path = Path(out_dir)
    else:
        out_path = seed_parent / "eigen_report_aggregate"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Parse parameters from first seed
    first_seed_model_dir = seed_dirs[0]
    d0, P0, N0, chi0, kappa0, eps0 = compute_mod.parse_run_params(first_seed_model_dir)
    
    param_title = ""
    if d0 is not None:
        parts = []
        parts.append(rf"$d={d0}$")
        if P0 is not None:
            parts.append(rf"$P={P0}$")
        if N0 is not None:
            parts.append(rf"$N={N0}$")
        if chi0 is not None:
            parts.append(rf"$\chi={chi0}$")
        if kappa0 is not None:
            parts.append(rf"$\kappa={kappa0}$")
        if eps0 is not None and eps0 != 0:
            parts.append(rf"$\epsilon={eps0}$")
        param_title = r",\ \ ".join(parts)
    
    # Compute kappa_eff and theory values (cached)
    theory_lh1t = None
    theory_lh3t = None
    theory_lh1p = None
    theory_lh3p = None
    kappa_eff = kappa0  # Default to bare kappa
    
    theory_cache = out_path / "theory_cache.json"
    if theory_cache.exists() and not force_recompute:
        try:
            with open(theory_cache, "r") as f:
                theory_cached = json.load(f)
            theory_lh1t = theory_cached.get("lH1T")
            theory_lh3t = theory_cached.get("lH3T")
            theory_lh1p = theory_cached.get("lH1P")
            theory_lh3p = theory_cached.get("lH3P")
            kappa_eff = theory_cached.get("kappa_eff", kappa0)
            print(f"Loaded cached theory values from {theory_cache}")
        except Exception as e:
            print(f"Warning: could not load theory cache: {e}")
    
    if (theory_lh1t is None or theory_lh3t is None or theory_lh1p is None or theory_lh3p is None) and d0 is not None:
        print("Computing kappa_eff via self-consistent solver...")
        try:
            kappa_eff = compute_kappa_eff(d0, P0, kappa0)
            print(f"  kappa_bare = {kappa0:.6f}, kappa_eff = {kappa_eff:.6f}")
        except Exception as e:
            print(f"  Warning: kappa_eff computation failed: {e}. Using bare kappa.")
            kappa_eff = kappa0
        
        print("Computing theory values via Julia with kappa_eff...")

        try:
            theory = compute_mod.compute_theory_with_julia(d0, N0, P0, chi0, kappa_eff, eps0)
            print("Computing theory with parameters")
            print(f"d: {d0} N: {N0}, P:{P0}, chi: {chi0}, kappa: {kappa0}, epsilon: {eps0}")

            target_block = theory.get("target", {}) if isinstance(theory, dict) else {}
            perp_block = theory.get("perpendicular", {}) if isinstance(theory, dict) else {}
            lH1T = target_block.get("lH1T")
            lH3T = target_block.get("lH3T")
            lH1P = perp_block.get("lH1P")
            lH3P = perp_block.get("lH3P")
            if lH1T is not None:
                theory_lh1t = float(lH1T)
            if lH3T is not None:
                theory_lh3t = float(lH3T) #/ 6.0
            if lH1P is not None:
                theory_lh1p = float(lH1P)
            if lH3P is not None:
                theory_lh3p = float(lH3P) #/ 6.0
            
            # Cache theory values
            with open(theory_cache, "w") as f:
                json.dump({
                    "lH1T": theory_lh1t,
                    "lH3T": theory_lh3t,
                    "lH1P": theory_lh1p,
                    "lH3P": theory_lh3p,
                    "kappa_eff": kappa_eff,

                }, f)
        except Exception as e:
            print(f"Warning: could not compute theory: {e}")
    
    # Collect h3 projections from each seed (cache per-seed JSON like main function does)
    seed_eigenvalues = {}  # seed_name -> {"lH1T": val, "lH1P": val, "lH3T": val, "lH3P": val}
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    for seed_dir in tqdm(seed_dirs, desc="Computing h3 projections"):
        seed_name = seed_dir.name
        h3_json = seed_dir / "h3_projections_fcn3.json"
        stats = None
        
        # Try to load cached projections if they exist and recompute not forced
        if h3_json.exists() and not force_recompute:
            try:
                with open(h3_json, "r") as f:
                    stats = json.load(f)
                print(f"  Loaded cached projections for {seed_name}")
            except Exception as e:
                print(f"  Warning: could not load cached projections for {seed_name}: {e}")
        
        # Compute if not loaded from cache or force_recompute is True
        if stats is None:
            try:
                model = compute_mod.load_model_from_run(seed_dir, device)
                d = int(getattr(model, "d", None) or 0)
                stats = compute_mod.compute_h3_projections_streaming(model, d, P_total=10_000_000, batch_size=1000, device=device)
                # Save the computed projections for future use
                compute_mod.save_stats(seed_dir, stats)
            except Exception as e:
                print(f"  Warning: could not compute h3 projections for {seed_name}: {e}")
                continue
        
        # Extract eigenvalues
        try:
            seed_eigenvalues[seed_name] = {
                "lH1T": stats["h1"]["target"]["second_moment"],
                "lH1P": stats["h1"]["perp"]["second_moment"],
                "lH3T": stats["h3"]["target"]["second_moment"],
                "lH3P": stats["h3"]["perp"]["second_moment"],
            }
        except Exception as e:
            print(f"  Warning: could not extract eigenvalues for {seed_name}: {e}")
    
    if not seed_eigenvalues:
        print("No seed eigenvalues collected. Exiting.")
        return
    
    # Create aggregate plot: scatter per seed (color-coded), mean dots, theory lines
    eig_types = ["lH1T", "lH1P", "lH3T", "lH3P"]
    eig_labels = [r"$\lambda_{H1}^T$", r"$\lambda_{H1}^{\perp}$", r"$\lambda_{H3}^T$", r"$\lambda_{H3}^{\perp}$"]
    theory_vals = [theory_lh1t, theory_lh1p, theory_lh3t, theory_lh3p]
    
    # Color palette for seeds
    seed_names = sorted(seed_eigenvalues.keys())
    num_seeds = len(seed_names)
    seed_colors = plt.cm.tab20(np.linspace(0, 1, num_seeds))
    seed_color_map = {name: seed_colors[i] for i, name in enumerate(seed_names)}
    
    # Create figure with 4 subplots (one per eigenvalue type) or 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax_idx, (eig_type, eig_label, theory_val) in enumerate(zip(eig_types, eig_labels, theory_vals)):
        ax = axes[ax_idx]
        
        # Collect values per seed
        seed_vals = []
        seed_names_list = []
        for seed_name in seed_names:
            if seed_name in seed_eigenvalues:
                val = seed_eigenvalues[seed_name].get(eig_type)
                if val is not None:
                    seed_vals.append(val)
                    seed_names_list.append(seed_name)
        
        if not seed_vals:
            ax.text(0.5, 0.5, f"No data for {eig_type}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(eig_label)
            continue
        
        # Plot scatter dots per seed at same x position (x=0.5) with small jitter
        jitter_strength = 0.03
        x_pos = 0.5
        np.random.seed(42)  # Consistent jitter across runs
        x_jitter = x_pos + np.random.normal(0, jitter_strength, len(seed_vals))
        
        for i, (seed_name, val) in enumerate(zip(seed_names_list, seed_vals)):
            color = seed_color_map[seed_name]
            ax.scatter(x_jitter[i], val, s=60, alpha=0.6, color=color, label=seed_name, zorder=3)
        
        # Plot mean as solid black dot
        mean_val = np.mean(seed_vals)
        ax.scatter(x_pos, mean_val, s=150, marker='D', color="black", alpha=0.95, 
                   label="Mean", zorder=4, edgecolors="white", linewidth=2)
        
        # Plot theory line if available
        if theory_val is not None:
            # Calculate percentage gap between mean and theory
            pct_gap = 100.0 * (1.0 - theory_val / mean_val) if mean_val > 0 else 0.0
            theory_label = f"Theory (κ_eff): {pct_gap:+.1f}%"
            ax.axhline(theory_val, color="red", linestyle="--", linewidth=2.5, alpha=0.8, label=theory_label, zorder=2)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([])
        ax.set_ylabel("Eigenvalue", fontsize=11)
        ax.set_title(eig_label, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    
    fig.suptitle(f"He3 Projection Eigenvalues Across Seeds — {param_title}", fontsize=13, y=0.995)
    fig.tight_layout()
    aggregate_plot = out_path / "aggregate_h3_eigenvalues.png"
    fig.savefig(aggregate_plot, dpi=300)
    print(f"Saved aggregate plot (4-panel): {aggregate_plot}")
    plt.close(fig)
    
    # Create single-subplot aggregate plot with all eigenvalues at different x positions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette for eigenvalue types
    eig_type_colors = {
        "lH1T": "#4C78A8",
        "lH1P": "#F58518", 
        "lH3T": "#54A24B",
        "lH3P": "#B279A2",
    }
    
    for x_idx, (eig_type, eig_label, theory_val) in enumerate(zip(eig_types, eig_labels, theory_vals)):
        # Collect values per seed for this eigenvalue type
        seed_vals = []
        for seed_name in seed_names:
            if seed_name in seed_eigenvalues:
                val = seed_eigenvalues[seed_name].get(eig_type)
                if val is not None:
                    seed_vals.append(val)
        
        if not seed_vals:
            continue
        
        x_pos = x_idx
        
        # Plot scatter dots per seed at this x position with small jitter
        jitter_strength = 0.05
        np.random.seed(42)
        x_jitter = x_pos + np.random.normal(0, jitter_strength, len(seed_vals))
        
        eig_color = eig_type_colors.get(eig_type, "gray")
        
        for i, (seed_name, val) in enumerate(zip(seed_names, seed_vals)):
            seed_color = seed_color_map[seed_name]
            ax.scatter(x_jitter[i], val, s=50, alpha=0.5, color=seed_color, zorder=3)
        
        # Plot mean as solid dot with eigenvalue-type color
        mean_val = np.mean(seed_vals)
        ax.scatter(x_pos, mean_val, s=200, marker='o', color=eig_color, alpha=0.9,
                   edgecolors="white", linewidth=2, zorder=4, label=eig_label)
        
        # Plot theory line if available and add error text
        if theory_val is not None:
            pct_gap = 100.0 * (1.0 - theory_val / mean_val) if mean_val > 0 else 0.0
            # Draw a short horizontal line at theory value
            ax.plot([x_pos - 0.15, x_pos + 0.15], [theory_val, theory_val],
                   color="red", linestyle="--", linewidth=2.5, alpha=0.7, zorder=2)
            # Add error text above the mean dot
            ax.text(x_pos, mean_val * 1.3, f"{pct_gap:+.1f}%", 
                   ha="center", va="bottom", fontsize=10, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="none"), zorder=5)
    
    ax.set_xticks(range(len(eig_types)))
    ax.set_xticklabels([label.replace("$", "").replace("\\", "") for label in eig_labels], fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=12)
    ax.set_xlabel("Eigenvalue Type", fontsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, title="Mean (colored dots)")
    
    fig.suptitle(f"He3 Projection Eigenvalues Across Seeds (Unified) — {param_title}", fontsize=13)
    fig.tight_layout()
    aggregate_plot_unified = out_path / "aggregate_h3_eigenvalues_unified.png"
    fig.savefig(aggregate_plot_unified, dpi=300)
    print(f"Saved aggregate plot (unified): {aggregate_plot_unified}")
    plt.close(fig)
    
    # Save seed eigenvalues to JSON for reference
    output_json = out_path / "seed_h3_eigenvalues.json"
    with open(output_json, "w") as f:
        json.dump(seed_eigenvalues, f, indent=2)
    print(f"Saved seed eigenvalues: {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='*', default=[])
    parser.add_argument('--out', default=None)
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--directory', '-d', default=None, 
                        help='Aggregate h3 projections across seed<x> subdirectories')
    args = parser.parse_args()
    
    if args.directory:
        eigen_report_seed_aggregate(args.directory, out_dir=args.out, force_recompute=args.force_recompute)
    elif args.dirs:
        eigen_report_fcn3(args.dirs, out_dir=args.out, force_recompute=args.force_recompute)
    else:
        parser.print_help()
        print("Error: Must provide either positional 'dirs' or --directory flag")
        exit(1)
