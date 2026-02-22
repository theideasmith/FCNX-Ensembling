import subprocess
import tempfile
import json
import sys
import re
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
import traceback
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

# --- Arcsin Kernel Function ---
def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)

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

# --- Helper function to compute kappa_eff ---

def compute_kappa_eff(d: int, P: int, kappa: float):
    """Compute effective ridge by running self-consistent kappa solver using arcsin kernel eigenvalues."""
    try:
        # Compute 5K x 5K arcsin kernel eigenvalues
        np.random.seed(0)
        X = np.random.randn(5000, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / 5000.0
        
        # Run self-consistent solver
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        eigenvalues = eigvals.tolist()
        with open(eig_json, "w") as f:
            json.dump({"eigenvalues": eigenvalues, "kappa_bare": kappa}, f)
        
        sc_cmd = [
            "julia", str(Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"),
            eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True)
        
        # Extract kappa_eff from output
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
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
    
    # Import h3 projection function
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_h3_projections_fcn3 import compute_h3_projections_streaming
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        P_total, batch_size = 3000, 500
        X = torch.randn(P_total, d, device=device)
        # eigs = model.H_eig_random_svd(X, k=700)
        # emp_h = float(eigs[0].detach().cpu().numpy())
        with torch.no_grad():
            W0 = model.W0  # shape: (ens, N, d)
            W0_reshaped = W0.view(model.ensembles * W0.shape[1], d)  # shape: (ens*N, d)
            cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * W0.shape[1])  # shape: (d, d)

            eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy()
            # eigvals_W0 = torch.var(model.W0[:,:,0]).cpu().numpy()  # Variance of first input dimension across all ensembles and neurons

        model.device = X.device  # Ensure model is on the same device as X
        eigs = model.H_eig_random_svd(X, k=700)
        kappa_eff = cfg.get('kappa_eff', cfg['kappa'])  # Use kappa_eff if available, else kappa
        lh1 = (eigs[0] / (eigs[0] + kappa_eff / cfg['P'])).item()
        lh3 = (eigs[cfg['d']] / (eigs[cfg['d']] + kappa_eff / cfg['P'])).item()
        P_total, batch_size = 30_000, 10_000
        h1_sum, h3_sum, x0_norm_sum, x3_norm_sum = 0.0, 0.0, 0.0, 0.0
        print(f"Running empirical h1/h3 estimation for d={cfg['d']}, P={cfg['P']} on device {device}...")
        print("ENSEMBLES: ", model.ensembles)
        with torch.no_grad():
            torch.manual_seed(4324)  # Reset seed for reproducibility
            # for _ in range(P_total // batch_size):
            #     X_batch = torch.randn(batch_size, d, device=device)
            #     out = model(X_batch)
            #     x0 = X_batch[:, 0]
            #     d_h1_sum = (out * x0.unsqueeze(-1)).sum().item() if out.ndim > 1 else (out * x0).sum().item() 
            #     d_h1_sum /= model.ensembles
            #     h1_sum += d_h1_sum 
            # h1_sum /= P_total
            # print("H1 Sum is: ", h1_sum)
            # # If seed exists: 
            # if 'seed' in cfg:
            #    torch.manual_seed(cfg['seed'])  # Reset seed for reproducibility
            # else:
            #     torch.manual_seed(cfg['base_seed'])  # Default seed if not specified
            # for _ in range(P_total // batch_size):
            #     X_batch = torch.randn(batch_size, d, device=device)
            #     out = model(X_batch)

            #     x0 = X_batch[:, 0]
            #     h3_comp = (x0**3 - 3*x0) / 6**0.5
            #     # Graham-Schmidt orthogonalization to get h3 component
            #     remainder = out - (h1_sum * x0).unsqueeze(-1)
            #     proj3_target_sum = torch.einsum('pq,p->q', remainder, h3_comp).sum().item() / model.ensembles  # Average over ensembles
            #     # print(d_h3_sum)
            #     h3_sum += proj3_target_sum
            # print("H3 Sum is: ", h3_sum / P_total)
            # Projection using graham schmidt orthogonalization to get h3 component
            torch.manual_seed(int(cfg['seed'] * 3.14) if 'seed' in cfg else 4324)  # Reset seed for reproducibility
            x0 = torch.randn(12000, d, device=device)  # Sample new x0 for projection
            out = model(x0)  # Get model output for x0
            linear_component = torch.einsum('pq,p->q', out, x0[:,0]).sum().item() * x0[:,0].unsqueeze(-1) / model.ensembles / x0.shape[0] # Average over ensembles
            h3_comp = (x0[:,0]**3 - 3*x0[:,0]) / 6**0.5
            h1_sum = torch.einsum('pq,p->q', out, x0[:,0]).sum().item() / model.ensembles / x0.shape[0]
            remainder = out - linear_component
            proj3_target_sum = torch.einsum('pq,p->q', remainder, h3_comp).sum().item() / model.ensembles  # Average over ensembles
            h3_sum = proj3_target_sum / x0.shape[0] / (0.03 * 6**0.5)  # Normalize by target scaling and number of samples
            print("H3 Sum is: ", h3_sum)
        
        # Compute h3 eigenvalues using high-precision streaming (P_total=200M)
        print(f"Computing h3 projections with P_total=200_000_000 for d={cfg['d']}, P={cfg['P']}...")
        h3_stats = compute_h3_projections_streaming(
            model, 
            d=cfg['d'],
            P_total=200_000_000,
            batch_size=10_000,
            device=device
        )
        h3_target_eig = h3_stats['h3']['target']['second_moment']
        h3_perp_eig = h3_stats['h3']['perp']['second_moment']
        print(f"h3_target={h3_target_eig}, h3_perp={h3_perp_eig}")
        
        # return {"emp_h": emp_h, "emp_w0": float(eigvals_W0[0]), "h1_emp": h1_sum / x0_norm_sum, "h3_emp": h3_sum / x3_norm_sum}
        return {"emp_h": float(eigs[0].detach().cpu().numpy().item()), "emp_w0": float(eigvals_W0[0]), "h1_emp": h1_sum , "h3_emp": h3_sum, "h3_target_eig": h3_target_eig, "h3_perp_eig": h3_perp_eig}
    except Exception as e:
        print(f"Empirical Error {m_dir.name}: {e}"); 
        traceback.print_exc()
        return None
    finally:
        if device.type == 'cuda': torch.cuda.empty_cache()

# --- 3. Main Pipeline ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=float, nargs='+', default=[80])
    parser.add_argument("--kappa", type=float, nargs='+', default=None)
    # Added d argument here
    parser.add_argument("--d", type=int, nargs='+', default=None, help="Filter by input dimension d")
    parser.add_argument("--ignore-seeds", type=int, nargs='+', default=[], help="Seed values to exclude from analysis")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Directory containing result subfolders")
    args = parser.parse_args()
    ignored_seeds = set(args.ignore_seeds)
    results_dir = Path(args.results_dir)
    RESULTS_DIR = results_dir  # Update global variable with argument value
    mp.set_start_method('spawn', force=True)
    all_dirs = sorted(list(results_dir.glob("d*/*seed*")))

    print(all_dirs)
    final_data = []
    to_compute_dirs = []
    to_compute_hashes = []

    # Step 1: Filter and Cache Check
    for m_dir in all_dirs:
        
        if not (m_dir / "config.json").exists(): 
            print("NO CONFIG", m_dir)
            continue
        with open(m_dir / "config.json") as f:
            cfg = json.load(f)
        print(cfg['d'], cfg['P'], cfg['chi'], cfg['kappa'])
        if cfg['P'] > 10000:
            print("Skipping due to large P: ", cfg['P'])
            continue
        # Filtering logic
        if not any(abs(float(cfg.get("chi", 0)) - c) < 1e-6 for c in args.chi): continue
        if args.kappa and not any(abs(float(cfg.get("kappa", 0)) - k) < 1e-6 for k in args.kappa): continue
        

        # Filter by d if the argument is provided
        if args.d and int(cfg.get("d", 0)) not in args.d: 
            continue

        run_seed = None
        for seed_key in ["seed", "base_seed", "torch_seed", "rng_seed"]:
            if seed_key in cfg:
                try:
                    run_seed = int(cfg[seed_key])
                    break
                except (TypeError, ValueError):
                    pass

        if run_seed is None:
            seed_match = re.search(r"seed(\d+)", m_dir.name)
            if seed_match:
                run_seed = int(seed_match.group(1))

        if run_seed is not None and run_seed in ignored_seeds:
            print(f"Skipping ignored seed {run_seed}: {m_dir}")
            continue

        c_hash = CacheManager.get_config_hash(cfg)
        cached = CacheManager.load_result(c_hash)

        if cached and not args.force:
            # Ensure kappa_eff is computed even for cached results
            if "kappa_eff" not in cached:
                cached["kappa_eff"] = compute_kappa_eff(cached["d"], cached["P"], cached["kappa"])
                CacheManager.save_result(c_hash, cached)
            final_data.append(cached)
        else:
            # Even when --force, reuse cached kappa_eff if available
            cached_kappa_eff = None
            if cached and "kappa_eff" in cached:
                cached_kappa_eff = cached["kappa_eff"]
            to_compute_dirs.append({'path': str(m_dir), 'cfg': cfg, 'cached_kappa_eff': cached_kappa_eff})
            to_compute_hashes.append(c_hash)

    # Step 2: Parallel Computation for Cache Misses
    if to_compute_dirs:
        print(f"Processing {len(to_compute_dirs)} cache misses...")
        
        # Step 2a: Compute kappa_eff for all configurations
        print("Computing effective ridge (kappa_eff)...")
        kappa_effs = []
        for task_info in to_compute_dirs:
            cfg = task_info['cfg']
            d = cfg["d"]
            P = cfg["P"]
            kappa = cfg["kappa"]

            # Reuse cached kappa_eff if available (even under --force)
            cached_kappa_eff = task_info.get('cached_kappa_eff')
            if cached_kappa_eff is not None:
                kappa_eff = cached_kappa_eff
                print(f"  d={d}, P={P}: kappa_eff={kappa_eff:.6f} (from cache)")
            else:
                kappa_eff = compute_kappa_eff(d, P, kappa)
                print(f"  d={d}, P={P}: kappa_bare={kappa:.6f} -> kappa_eff={kappa_eff:.6f}")
            kappa_effs.append(kappa_eff)
            # Update task_info cfg with kappa_eff for use in empirical task
            task_info['cfg']['kappa_eff'] = kappa_eff
            print(f"  d={d}, P={P}: kappa_bare={kappa:.6f} -> kappa_eff={kappa_eff:.6f}")
        
        # Parallel Empirical (GPU)
        with ProcessPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            emp_results = list(executor.map(run_empirical_task, to_compute_dirs))
        
        # Parallel Theory (CPU) - using kappa_eff
        theory_params = [{"d": to_compute_dirs[i]['cfg']["d"], "P": to_compute_dirs[i]['cfg']["P"], 
                          "n1": to_compute_dirs[i]['cfg']["N"], "n2": to_compute_dirs[i]['cfg']["N"], 
                          "chi": to_compute_dirs[i]['cfg']["chi"], "kappa": kappa_effs[i], "eps": EPSILON} 
                         for i in range(len(to_compute_dirs))]
        with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
            theo_results = list(executor.map(run_theory_task, theory_params))

        # Merge and Save to Cache
        for i, (emp, theo) in enumerate(zip(emp_results, theo_results)):
            if emp is None: continue
            
            res = {**to_compute_dirs[i]['cfg'], **emp, 
                   "theo_h": float(theo.get("lH1T", np.nan)), "theo_w": float(theo.get("lWT", np.nan)),
                   "h1_theory": float(theo.get("mu1", np.nan)), "h3_theory": float(theo.get("mu3", np.nan)),
                   "kappa_eff": kappa_effs[i]}
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
        ax1.scatter(p_vals, [r["emp_h"] for r in res], color='red', label=label, alpha=0.7, s=50)
        ax2.scatter(p_vals, [r["emp_w0"] for r in res], color='red', label=label, alpha=0.7, s=50)
        
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
        ax1.plot(unique_p, mean_emp_h, '-', color='red', linewidth=3, alpha=0.8)
        ax1.plot(unique_p, mean_theo_h, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax2.plot(unique_p, mean_emp_w, '-', color='red', linewidth=3, alpha=0.8)
        ax2.plot(unique_p, mean_theo_w, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
    # Dummy plot for legend
    ax1.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    ax1.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    ax2.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    ax2.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    d = final_data[0]["d"] if final_data else 0
    ax1.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax2.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel("P"); ax.set_xscale('log')
    kappa = final_data[0]["kappa"] if final_data else 0
    kappa_eff = final_data[0]["kappa_eff"] if final_data else 0
    N = final_data[0]["N"] if final_data else 0
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Separate plots
    # Lambda H
    fig_h = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_h"] for r in res], color='red', label=label, alpha=0.7, s=50)
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        plt.plot(unique_p, mean_emp_h, '-', color='red', linewidth=3, alpha=0.8)
        plt.plot(unique_p, mean_theo_h, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_H^*$)" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Lambda W
    fig_w = plt.figure(figsize=(10, 6))
    for i, (val, res) in enumerate(groups.items()):
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(p_vals, [r["emp_w0"] for r in res], color='red', label=label, alpha=0.7, s=50)
        # Mean lines
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        for r in res:
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
        unique_p = sorted(p_to_emp_w.keys())
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        plt.plot(unique_p, mean_emp_w, '-', color='red', linewidth=3, alpha=0.8)
        plt.plot(unique_p, mean_theo_w, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_W^*$)" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.ylim(0,None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Learnability
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(14, 8))
        for i, (val, res) in enumerate(groups.items()):
            p_vals = [r["P"] for r in res]
            label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
            plt.scatter(p_vals, [r[f"{mode}_emp"] for r in res], color='red', label=label, alpha=0.7, s=50)
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
            plt.plot(unique_p, mean_emp, '-', color='red', linewidth=3, alpha=0.8)
            plt.plot(unique_p, mean_theo, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
        # Dummy plot for legend
        plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
        plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0

        title_mode = "Hermite-1" if mode == "h1" else "Hermite-3"
        plt.title(rf"{title_mode} Learnability $\eta_{{He1}}$" + f", $d={d}, N={N}, \\kappa_{{eff}}={kappa_eff:.2g}$"); plt.xlabel("P (dataset size)"); plt.legend(); plt.grid(True, alpha=0.3); plt.xscale('log')
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{\| He_1 \|^2}$ (Learnability)")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
        plt.ylim(0, None)
    # --- Additional plots with alpha on x-axis (linear scale) ---
    # Eigenvalues with alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        ax1.scatter(alpha_vals, [r["emp_h"] for r in res], color='red', label=label, alpha=0.7, s=50)
        ax2.scatter(alpha_vals, [r["emp_w0"] for r in res], color='red', label=label, alpha=0.7, s=50)
        
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
        ax1.plot(unique_alpha, mean_emp_h, '-', color='red', linewidth=3, alpha=0.8)
        ax1.plot(unique_alpha, mean_theo_h, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax2.plot(unique_alpha, mean_emp_w, '-', color='red', linewidth=3, alpha=0.8)
        ax2.plot(unique_alpha, mean_theo_w, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
        ax1.set_ylim(0, None); ax2.set_ylim(0, None)
    # Dummy plot for legend
    ax1.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    ax1.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    ax2.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    ax2.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    ax1.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel(r"$\alpha$")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Separate alpha plots
    # Lambda H alpha
    fig_h = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color='red', label=label, alpha=0.7, s=50)
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
        plt.plot(unique_alpha, mean_emp_h, '-', color='red', linewidth=3, alpha=0.8)
        plt.plot(unique_alpha, mean_theo_h, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
    plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title("Preactivation Target Eigenvalues "+ r"$\lambda^{H,He1}_*$"  + "\n" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
    # Lambda W alpha
    fig_w = plt.figure(figsize=(8, 8))
    for i, (val, res) in enumerate(groups.items()):
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        plt.scatter(alpha_vals, [r["emp_w0"] for r in res], color='red', label=label, alpha=0.7, s=50)
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
        plt.plot(unique_alpha, mean_emp_w, '-', color='red', linewidth=3, alpha=0.8)
        plt.plot(unique_alpha, mean_theo_w, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
    # Y minimum is 0
    plt.ylim(0, None)
    plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
    plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"$\lambda_W$ Eigenvalue" + f" d={d}, $\\kappa_{{eff}}={kappa_eff:.2g}$, N={N}")
    plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)

    # Learnability with alpha
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(8, 8))
        for i, (val, res) in enumerate(groups.items()):
            alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
            label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
            plt.scatter(alpha_vals, [r[f"{mode}_emp"] for r in res], color='red', label=label, alpha=0.7, s=50)
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
            plt.plot(unique_alpha, mean_emp, '-', color='red', linewidth=3, alpha=0.8)
            plt.plot(unique_alpha, mean_theo, '--', color='blue', linewidth=3, marker='o', markersize=8, alpha=0.8)
        # Dummy plot for legend
        plt.plot([], [], '-', color='red', linewidth=3, label="Experiment")
        plt.plot([], [], '--', color='blue', linewidth=3, marker='o', markersize=8, label="Theory")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0
        title_mode = "Hermite-1" if mode == "h1" else "Hermite-3"
        plt.title(rf"{title_mode} Learnability $\eta_{{He1}}$" + f"\n$d={d}, N={N}, \\kappa_{{eff}}={kappa_eff:.2g}$"); plt.xlabel(r"$\alpha$"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{\| He_1 \|^2}$ (Learnability)")
        plt.ylim(0, None)

        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_alpha_linear_d{d}_N{N}_kappa_bare{kappa:.2g}_kappa_eff{kappa_eff:.2g}.png", dpi=300)
    # plt.show()