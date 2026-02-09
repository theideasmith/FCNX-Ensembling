import subprocess
import tempfile
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict

# Set publication-quality styling
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 150,
})

# --- Configuration & Paths ---
LIB_PATH = Path(__file__).parent.parent.parent / "lib"
JULIA_SCRIPT = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
RESULTS_DIR = Path(__file__).parent / "p_scan_erf_results"
CACHE_DIR = RESULTS_DIR / "analysis_cache"
sys.path.insert(0, str(LIB_PATH))

MAX_GPU_WORKERS = 2   
MAX_CPU_WORKERS = 10  
EPSILON = 1e-3

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Cache Manager ---

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

# --- 2. Worker Tasks ---

def run_theory_task(params):
    """CPU-bound Julia Task."""
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
    """GPU-bound Empirical Task."""
    m_dir, cfg = Path(task_info['path']), task_info['cfg']
    pt_path = m_dir / "model_final.pt" if (m_dir / "model_final.pt").exists() else m_dir / "model.pt"
    from FCN3Network import FCN3NetworkActivationGeneric
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        sd = torch.load(pt_path, map_location="cpu")
        d, n1, n2 = sd['W0'].shape[-1], sd['W0'].shape[-2], sd['W1'].shape[-2]
        P = cfg["P"]
        ens = sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1]
        
        model = FCN3NetworkActivationGeneric(
            d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
            weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2))
        ).to(device)
        model.load_state_dict({k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v for k, v in sd.items()}, strict=False)
        model.eval()

        torch.manual_seed(0)
        X = torch.randn(6000, d, device=device)
        h_eig = model.H_eig(X, X)
        emp_h = float(h_eig[0].detach().cpu().numpy())
        with torch.no_grad():
            W0 = model.W0  # shape: (ens, N, d)
            W0_reshaped = W0.view(model.ensembles * W0.shape[1], d)  # shape: (ens*N, d)
            cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * W0.shape[1])  # shape: (d, d)
            eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy()
        
        P_total, batch_size = 1000000, 25000
        h1_sum, h3_sum, x0_norm_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for _ in range(P_total // batch_size):
                X_batch = torch.randn(batch_size, d, device=device)
                out = model(X_batch)
                x0 = X_batch[:, 0]
                h3_comp = (x0**3 - 3*x0) / np.sqrt(6)
                x0_norm_sum += x0.norm().item()
                h1_sum += (out * x0.unsqueeze(-1)).sum().item() if out.ndim > 1 else (out * x0).sum().item()
                h3_sum += (out * h3_comp.unsqueeze(-1)).sum().item() if out.ndim > 1 else (out * h3_comp).sum().item()

        return {"emp_h": emp_h, "emp_w0": float(eigvals_W0[0]), "h1_emp": h1_sum / P_total, "h3_emp": h3_sum / P_total}
    except Exception as e:
        print(f"Empirical Error {m_dir.name}: {e}"); return None
    finally:
        if device.type == 'cuda': torch.cuda.empty_cache()

# --- 3. Main Pipeline ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=float, nargs='+', default=[80])
    parser.add_argument("--kappa", type=float, nargs='+', default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    all_dirs = sorted(list(RESULTS_DIR.glob("d*/*seed*")))
    final_data = []
    to_compute_dirs = []
    to_compute_hashes = []

    # Step 1: Filter and Cache Check
    for m_dir in all_dirs:
        if not (m_dir / "config.json").exists(): continue
        with open(m_dir / "config.json") as f:
            cfg = json.load(f)
        
        if not any(abs(float(cfg.get("chi", 0)) - c) < 1e-6 for c in args.chi): continue
        if args.kappa and not any(abs(float(cfg.get("kappa", 0)) - k) < 1e-6 for k in args.kappa): continue

        c_hash = CacheManager.get_config_hash(cfg)
        cached = CacheManager.load_result(c_hash)

        if cached and not args.force:
            final_data.append(cached)
        else:
            to_compute_dirs.append({'path': str(m_dir), 'cfg': cfg})
            to_compute_hashes.append(c_hash)

    # Step 2: Parallel Computation for Cache Misses
    if to_compute_dirs:
        print(f"Processing {len(to_compute_dirs)} cache misses...")
        
        # Parallel Empirical (GPU)
        with ProcessPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            emp_results = list(executor.map(run_empirical_task, to_compute_dirs))
        
        # Parallel Theory (CPU)
        theory_params = [{"d": d['cfg']["d"], "P": d['cfg']["P"], "n1": d['cfg']["N"], "n2": d['cfg']["N"], 
                          "chi": d['cfg']["chi"], "kappa": d['cfg']["kappa"], "eps": EPSILON} 
                         for d in to_compute_dirs]
        with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
            theo_results = list(executor.map(run_theory_task, theory_params))

        # Merge and Save to Cache
        for i, (emp, theo) in enumerate(zip(emp_results, theo_results)):
            if emp is None: continue
            res = {**to_compute_dirs[i]['cfg'], **emp, 
                   "theo_h": float(theo.get("lH1T", np.nan)), "theo_w": float(theo.get("lWT", np.nan)),
                   "h1_theory": float(theo.get("mu1", np.nan)), "h3_theory": float(theo.get("mu3", np.nan))}
            CacheManager.save_result(to_compute_hashes[i], res)
            final_data.append(res)

    # --- 4. Plotting ---
    if not final_data:
        print("No data points available."); sys.exit()

    unique_kappas = sorted(set(r["kappa"] for r in final_data))
    unique_chis = sorted(set(r["chi"] for r in final_data))
    color_by = "kappa" if len(unique_kappas) > 1 else "chi"
    unique_vals = unique_kappas if color_by == "kappa" else unique_chis
    groups = {val: [r for r in final_data if r[color_by] == val] 
              for val in unique_vals}
    
    num_colors = len(unique_vals)
    if num_colors < 5:
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # blue, orange, green, red, purple
        colors = colors_list[:num_colors]
    else:
        cmap = plt.cm.get_cmap("plasma", num_colors)
        colors = [cmap(i) for i in range(num_colors)]

    # Eigenvalues
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for i, (val, res) in enumerate(groups.items()):
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        ax1.scatter(p_vals, [r["emp_h"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        ax2.scatter(p_vals, [r["emp_w0"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        ax1.plot(unique_p, mean_emp_h, '-', color=colors[i], linewidth=3, alpha=0.8)
        ax1.plot(unique_p, mean_theo_h, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax2.plot(unique_p, mean_emp_w, '-', color=colors[i], linewidth=3, alpha=0.8)
        ax2.plot(unique_p, mean_theo_w, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    # Dummy plot for legend
    ax1.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    ax2.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    d = final_data[0]["d"] if final_data else 0
    ax1.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax2.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel("P"); ax.set_xscale('log')
    kappa = final_data[0]["kappa"] if final_data else 0
    N = final_data[0]["N"] if final_data else 0
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Separate plots
    # Lambda H
    fig_h = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_h"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        plt.plot(unique_p, mean_emp_h, '-', color=colors[i], linewidth=3, alpha=0.8)
        plt.plot(unique_p, mean_theo_h, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_H^*$)" + f" d={d}, κ={kappa}, N={N}")
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Lambda W
    fig_w = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_w0"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        # Mean lines
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        for r in res:
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
        unique_p = sorted(p_to_emp_w.keys())
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        plt.plot(unique_p, mean_emp_w, '-', color=colors[i], linewidth=3, alpha=0.8)
        plt.plot(unique_p, mean_theo_w, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_W^*$)" + f" d={d}, κ={kappa}, N={N}")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')

    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Learnability
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(14, 8))
        for i, (val, res) in enumerate(groups.items()):
            p_vals = [r["P"] for r in res]
            label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
            plt.scatter(p_vals, [r[f"{mode}_emp"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
            d = res[0]["d"] if res else 0
            plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d" if i == 0 else None)

            # Mean lines
            p_to_emp = defaultdict(list)
            p_to_theo = defaultdict(list)
            for r in res:
                p_to_emp[r["P"]].append(r[f"{mode}_emp"])
                p_to_theo[r["P"]].append(r[f"{mode}_theory"])
            unique_p = sorted(p_to_emp.keys())
            mean_emp = [np.mean(p_to_emp[p]) for p in unique_p]
            mean_theo = [np.mean(p_to_theo[p]) for p in unique_p]
            plt.plot(unique_p, mean_emp, '-', color=colors[i], linewidth=3, alpha=0.8)
            plt.plot(unique_p, mean_theo, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
        # Dummy plot for legend
        plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        N = res[0]['N'] if res else 0
        plt.title(r"Hermite-1 Learnability $\eta_{He1}$" + f", $d={d}, N={N}, \kappa={kappa}$"); plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{\| He_1 \|^2}$ (Learnability)")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # --- Additional plots with alpha on x-axis (linear scale) ---
    # Eigenvalues with alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 18))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        ax1.scatter(alpha_vals, [r["emp_h"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        ax2.scatter(alpha_vals, [r["emp_w0"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        ax1.plot(unique_alpha, mean_emp_h, '-', color=colors[i], linewidth=3, alpha=0.8)
        ax1.plot(unique_alpha, mean_theo_h, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax2.plot(unique_alpha, mean_emp_w, '-', color=colors[i], linewidth=3, alpha=0.8)
        ax2.plot(unique_alpha, mean_theo_w, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    # Dummy plot for legend
    ax1.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    ax2.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    ax1.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel(r"$\alpha$")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_alpha_linear_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Separate alpha plots
    # Lambda H alpha
    fig_h = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        plt.plot(unique_alpha, mean_emp_h, '-', color=colors[i], linewidth=3, alpha=0.8)
        plt.plot(unique_alpha, mean_theo_h, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title("Preactivation Target Eigenvalues "+ r"$\lambda^{H,He1}_*$"  + "\n" + f" d={d}, κ={kappa}, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_alpha_linear_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Lambda W alpha
    fig_w = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_w0"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
        # Mean lines
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
        unique_alpha = sorted(alpha_to_emp_w.keys())
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        plt.plot(unique_alpha, mean_emp_w, '-', color=colors[i], linewidth=3, alpha=0.8)
        plt.plot(unique_alpha, mean_theo_w, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"$\lambda_W$ Eigenvalue" + f" d={d}, κ={kappa}, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_alpha_linear_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    # Learnability with alpha
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(8, 8))
        for i, (val, res) in enumerate(groups.items()):
            alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
            label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
            plt.scatter(alpha_vals, [r[f"{mode}_emp"] for r in res], color=colors[i], label=label, alpha=0.7, s=50)
            plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$" if i == 0 else None)

            # Mean lines
            alpha_to_emp = defaultdict(list)
            alpha_to_theo = defaultdict(list)
            for r in res:
                alpha = np.log(r["P"]) / np.log(r["d"])
                alpha_to_emp[alpha].append(r[f"{mode}_emp"])
                alpha_to_theo[alpha].append(r[f"{mode}_theory"])
            unique_alpha = sorted(alpha_to_emp.keys())
            mean_emp = [np.mean(alpha_to_emp[a]) for a in unique_alpha]
            mean_theo = [np.mean(alpha_to_theo[a]) for a in unique_alpha]
            plt.plot(unique_alpha, mean_emp, '-', color=colors[i], linewidth=3, alpha=0.8)
            plt.plot(unique_alpha, mean_theo, '--', color=colors[i], linewidth=3, marker='o', markersize=8, alpha=0.8)
        # Dummy plot for legend
        plt.plot([], [], '--', color='k', linewidth=3, marker='o', markersize=8, label="Theory")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        N = res[0]['N'] if res else 0
        plt.title("Hermite-1 Learnability " +r"$\eta_{He1}$" + "\n" + f"$d={d}, N={N}, \kappa=2.0$"); plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{\| He_1 \|^2}$ (Learnability)")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_alpha_linear_d{d}_N{N}_kappa{kappa}.png", dpi=300)

    plt.show()