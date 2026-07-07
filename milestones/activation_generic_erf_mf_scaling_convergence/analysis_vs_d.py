import subprocess
import tempfile
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# --- arXiv Style Plotting ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 9,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 200,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

ARXIV_COLORS = ['#003399', '#CC0000', '#006400', '#800080']

RESULTS_DIR = Path(__file__).parent / "d_scan_erf_results"
CACHE_DIR = RESULTS_DIR / "analysis_cache"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

EPSILON = 1e-3

class CacheManager:
    @staticmethod
    def get_config_hash(cfg):
        relevant_keys = ["d", "P", "N", "chi", "kappa"]
        core_params = {k: cfg.get(k) for k in relevant_keys if k in cfg}
        core_params["eps"] = EPSILON
        param_str = json.dumps(core_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    @staticmethod
    def load_result(cfg_hash):
        cache_path = CACHE_DIR / f"res_{cfg_hash}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    @staticmethod
    def save_result(cfg_hash, data):
        cache_path = CACHE_DIR / f"res_{cfg_hash}.json"
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)

def get_fit(x, y):
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2: return 0.0, np.zeros_like(x)
    xlog, ylog = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(xlog, ylog, 1)
    y_fit = np.exp(intercept) * x ** slope
    return slope, y_fit

# --- Paths (Adjusted for your structure) ---
LIB_PATH = Path(__file__).parent.parent.parent / "lib"
JULIA_SCRIPT = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
sys.path.insert(0, str(LIB_PATH))

MAX_GPU_WORKERS = 1 
MAX_CPU_WORKERS = mp.cpu_count() - 2

def run_theory_task(params):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)
    cmd = [
        "julia", str(JULIA_SCRIPT),
        f"--d={params['d']}", f"--P={params['P']}",
        f"--n1={params['n1']}", f"--n2={params['n2']}",
        f"--chi={params['chi']}", f"--kappa={params['kappa']}",
        f"--epsilon={params['eps']}", f"--to={to_path}", "--quiet"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        with open(to_path, "r") as f:
            return json.load(f).get("target", {})
    except Exception as e:
        print(f"Julia error: {e}")
        return {}
    finally:
        if to_path.exists(): to_path.unlink()

def run_empirical_task(task_info):
    m_dir, cfg = Path(task_info['path']), task_info['cfg']
    pt_path = m_dir / "model_final.pt" if (m_dir / "model_final.pt").exists() else m_dir / "model.pt"
    from FCN3Network import FCN3NetworkActivationGeneric
    device = torch.device("cpu") 
    try:
        sd = torch.load(pt_path, map_location=device)
        d, n1, n2 = sd['W0'].shape[-1], sd['W0'].shape[-2], sd['W1'].shape[-2]
        model = FCN3NetworkActivationGeneric(
            d=d, n1=n1, n2=n2, P=cfg["P"], 
            ens=sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1],
            activation="erf", 
            weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2))
        ).to(device)
        model.load_state_dict({k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v for k, v in sd.items()}, strict=False)
        model.eval()
        X = torch.randn(2000, d, device=device) 
        h_eig = model.H_eig(X, X)
        with torch.no_grad():
            W0_reshaped = model.W0.view(-1, d)
            cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / W0_reshaped.shape[0]
            eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.numpy()
        return {"emp_h": float(h_eig[0].detach().numpy()), "emp_w0": float(eigvals_W0[0])}
    except Exception as e:
        print(f"Empirical Error at {m_dir}: {e}")
        return None

# --- Unified Plotting Logic ---

def render_scaling_plot(ax, regime_dict, mode='H'):
    """Unified engine to ensure visual consistency across all plot files."""
    emp_key = "emp_h" if mode == 'H' else "emp_w0"
    theo_key = "theo_h" if mode == 'H' else "theo_w"
    title_label = r"$\lambda_H$ Eigenvalue" if mode == 'H' else r"$\lambda_W$ Eigenvalue"
    
    sorted_labels = sorted(regime_dict.keys())
    for i, label in enumerate(sorted_labels):
        data = regime_dict[label]
        d_vals = np.array([r["d"] for r in data])
        idx = np.argsort(d_vals)
        d_s = d_vals[idx]
        eh_s = np.array([r[emp_key] for r in data])[idx]
        th_s = np.array([r[theo_key] for r in data])[idx]
        
        color = ARXIV_COLORS[i % len(ARXIV_COLORS)]
        
        # Power Law Fits
        s_th, f_th = get_fit(d_s, th_s)
        s_eh, f_eh = get_fit(d_s, eh_s)

        # Plot Theory (Solid line)
        ax.plot(d_s, th_s, '-', color=color, lw=2.5, 
                label=f"Theo: {label} ($d^{{{s_th:.2f}}}$)")
        
        # Plot Empirical (Dashed fit line + Scatter points)
        ax.plot(d_s, f_eh, '--', color=color, lw=1.5, alpha=0.7,
                label=f"Emp: {label} ($d^{{{s_eh:.2f}}}$)")
        ax.scatter(d_s, eh_s, color='none', edgecolor=color, s=60, alpha=0.6, zorder=3)

        # Scaling Annotation near the end of the line
        ax.annotate(f"$d^{{{s_th:.2f}}}$", xy=(d_s[-1], th_s[-1]), xytext=(5, 5), 
                    textcoords='offset points', color=color, fontsize=12, fontweight='bold')

    ax.set_title(title_label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$d$ (Input Dimension)")
    ax.set_ylabel(r"Eigenvalue Magnitude")
    ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    results_dirs = [
        Path(__file__).parent / "d_scan_erf_results", 
        Path(__file__).parent / "d_scan_erf_results_Pd32"
    ]
    
    all_dirs = []
    for rdir in results_dirs:
        if rdir.exists():
            all_dirs.extend(sorted(list(rdir.glob("d*/*seed*"))))

    final_data = []
    to_compute_tasks = []
    
    for m_dir in all_dirs:
        if not (m_dir / "config.json").exists(): continue
        with open(m_dir / "config.json") as f:
            cfg = json.load(f)
        
        c_hash = CacheManager.get_config_hash(cfg)
        cached = CacheManager.load_result(c_hash)
        
        parent_name = str(m_dir)
        label = r"$P \propto d^{1.5}$" if "Pd32" in parent_name else r"$P \propto d$"

        if cached and not args.force:
            cached['regime_label'] = label
            final_data.append(cached)
        else:
            to_compute_tasks.append({'path': str(m_dir), 'cfg': cfg, 'hash': c_hash, 'regime_label': label})

    if to_compute_tasks:
        print(f"Processing {len(to_compute_tasks)} items...")
        with ProcessPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            emp_results = list(executor.map(run_empirical_task, to_compute_tasks))
        
        theory_params = [{"d": t['cfg']["d"], "P": t['cfg']["P"], "n1": t['cfg']["N"], "n2": t['cfg']["N"],
                          "chi": t['cfg']["chi"], "kappa": t['cfg']["kappa"], "eps": EPSILON}
                         for t in to_compute_tasks]
        
        with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
            theo_results = list(executor.map(run_theory_task, theory_params))

        for i, (emp, theo) in enumerate(zip(emp_results, theo_results)):
            if emp is None: continue
            res = {**to_compute_tasks[i]['cfg'], **emp,
                   "theo_h": float(theo.get("lH1T", np.nan)), "theo_w": float(theo.get("lWT", np.nan)),
                   "regime_label": to_compute_tasks[i]['regime_label']}
            CacheManager.save_result(to_compute_tasks[i]['hash'], res)
            final_data.append(res)

    # --- Grouping ---
    per_regime = defaultdict(list)
    for r in final_data:
        per_regime[r.get('regime_label', 'Unknown')].append(r)

    # --- 1. Main Summary Plot (1x2) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    render_scaling_plot(ax1, per_regime, mode='H')
    render_scaling_plot(ax2, per_regime, mode='W')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scaling_laws_summary.png")

    # --- 2. Individual High-Res Plots ---
    for mode in ['H', 'W']:
        fig_ind, ax_ind = plt.subplots(figsize=(9, 7))
        render_scaling_plot(ax_ind, per_regime, mode=mode)
        plt.tight_layout()
        filename = f"eigenvalues_vs_d_loglog_{mode}.png"
        fig_ind.savefig(RESULTS_DIR / filename, dpi=300)
        plt.close(fig_ind)

    plt.show()