import os
import json
import glob
import subprocess
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from multiprocessing import Pool
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

matplotlib.use('Agg')

# --- Constants ---
DEVICE = 'cuda:0'
CACHE_NAME = "analysis_results_cache.json"
RESULTS_DIR = Path("/home/akiva/FCNX-Ensembling/milestones/activation_generic_erf_mf_scaling_convergence/p_scan_erf_results")
HISTS_DIR = RESULTS_DIR / "W0_hists"

# --- Theory Worker (Top Level for Pickling) ---

def theory_worker_task(task_info):
    """Runs the Julia solver for a specific parameter set."""
    P, d, N, chi, kappa, eps = task_info
    
    # Path to your Julia script
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)
    
    cmd = [
        "julia", str(julia_script), f"--d={d}", f"--P={P}", f"--n1={N}", f"--n2={N}",
        f"--chi={chi}", f"--kappa={kappa}", f"--epsilon={eps}", f"--to={to_path}", "--quiet"
    ]
    print(' '.join(cmd))
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        with open(to_path, "r") as f:
            data = json.load(f)
        
        tgt = data.get("target", {})
        lK1T = tgt.get("lK1T")
        lH3T = tgt.get("lH3T")
        lWT = tgt.get("lWT")  # Weight variance eigenvalue
        
        # Calculate learnability immediately
        h1_learn = lK1T / (lK1T + kappa / P) if lK1T is not None else 0.0
        h3_learn = lH3T / (lH3T + kappa / P) if lH3T is not None else 0.0
        
        return P, {"h1_learn": h1_learn, "h3_learn": h3_learn, "lWT": lWT}
    except Exception as e:
        print(f"Theory failed for P={P}: {e}")
        return P, None
    finally:
        to_path.unlink(missing_ok=True)

# --- Projection Worker (Unchanged but included for context) ---

def projection_worker_task(args):
    model_path, config, P_total, batch_size, device = args
    import sys
    sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
    from FCN3Network import FCN3NetworkErfOptimized
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model = FCN3NetworkErfOptimized(
            d=config['d'], n1=config['N'], n2=config['N'], 
            P=config['P'], num_seeds=config['num_seeds'], 
            ens=config['ens'], device=device
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        torch.manual_seed(config['base_seed'])
        h1_sum, h3_sum = torch.zeros((config['num_seeds'], model.ens), device=device, dtype=torch.float32), torch.zeros((config['num_seeds'], model.ens), device=device, dtype=torch.float32)
        num_batches = P_total // batch_size
        # Compute W0 variance for first dimension
        W0 = model.W0# shape: (num_seeds, ens, n1, d)
        W0_first_dim = W0[:, :, :, 0]  # shape: (num_seeds, ens, n1)
        W0_var = torch.var(W0_first_dim).item()


        with torch.no_grad():
            for _ in range(num_batches):
                print("BATCH!")
                X = torch.randn(config['num_seeds'], batch_size, config['d'], device=device)
                out = model(X)
                x0 = X[:, :, 0]
                h1_sum += torch.einsum('sbe,sb->se', out, x0)
                h3_sum += torch.einsum('sbe,sb->se', out, (x0**3 - 3*x0))
        
        norm = (P_total)
        return config['P'], {"h1": h1_sum.mean().item() / norm, "h3": h3_sum.mean().item() / norm, "W0_var": W0_var}
    except Exception as e:
        print(e)
        return config['P'], f"Error: {str(e)}"

# --- Main Logic Class ---

class ModelAnalyzer:
    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        self.cache_path = RESULTS_DIR / CACHE_NAME
        HISTS_DIR.mkdir(exist_ok=True)

    def get_models(self) -> List[Path]:
        return sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('d')])
    
    def extract_chi(self, model_dir: Path) -> Optional[float]:
        """Extract chi value from directory name like d50_P20_N1000_chi80_kappa0.1_nseeds10_ens20"""
        parts = model_dir.name.split('_')
        for part in parts:
            if part.startswith('chi'):
                try:
                    return float(part[3:])
                except ValueError:
                    return None
        return None

    def run_analysis(self, model_dirs, P_total=50_000):
        if self.use_cache and self.cache_path.exists():
            print("Using cached results.")
            with open(self.cache_path, 'r') as f: return json.load(f)

        proj_tasks = []
        theory_tasks = []
        model_config_map = {}  # Store config info for each model

        for mdir in model_dirs:
            config_path = mdir / "base_seed0" / "config.json"
            model_path = mdir / "base_seed0" / "model_final.pt"
            if not config_path.exists(): continue
            
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            chi = self.extract_chi(mdir)
            if chi is None:
                print(f"Warning: Could not extract chi from {mdir.name}")
                continue
            
            # Setup Empirical tasks (GPU/CPU bound)
            proj_tasks.append((str(model_path), cfg, P_total, 5_000, DEVICE))
            
            # Setup Theory tasks (Subprocess bound)
            theory_tasks.append((cfg['P'], cfg['d'], cfg['N'], chi, cfg['kappa'], 0.03))
            
            # Store config info for later
            model_config_map[str(cfg['P'])] = {
                "d": cfg['d'],
                "N": cfg['N'],
                "chi": chi,
                "kappa": cfg['kappa']
            }

        # 1. Parallel Projections
        print(f"Starting Projections on {DEVICE}...")
        with Pool(processes=4) as pool:
            proj_results = dict(list(tqdm(pool.imap(projection_worker_task, proj_tasks), total=len(proj_tasks))))

        # 2. Parallel Theory
        print("Starting Theory Computations (Julia Parallel)...")
        with Pool(processes=os.cpu_count() // 2) as pool:
            theory_results = dict(list(tqdm(pool.imap(theory_worker_task, theory_tasks), total=len(theory_tasks))))

        # 3. Merge
        final_data = {}
        for p_val in proj_results.keys():
            if isinstance(proj_results[p_val], str): continue # Handle errors
            final_data[str(p_val)] = {
                "empirical": proj_results[p_val],
                "theory": theory_results.get(p_val),
                "config": model_config_map.get(str(p_val))
            }

        with open(self.cache_path, 'w') as f:
            json.dump(final_data, f, indent=4)
        return final_data

    def log_tensorboard(self, data):
        writer = SummaryWriter(str(RESULTS_DIR / "tensorboard" / "hermite_analysis"))
        print("logging results to TensorBoard: ", RESULTS_DIR / "tensorboard" / "hermite_analysis")

        # Collect data for W0 variance plot, organized by chi
        chi_data = {}

        for p_str in sorted(data.keys(), key=lambda x: float(x)):
            P = float(p_str)
            vals = data[p_str]
            cfg = vals.get("config", {})
            
            print(f"Logging results for P={P}, empirical: ", vals['empirical']['h1'], "theory: ", vals['theory']['h1_learn'])
            if not vals['empirical']: raise ValueError(f"Empirical results missing for P={P}")
            if not vals['theory']: raise ValueError(f"Theory results missing for P={P}")

            writer.add_scalars('Projection/H1', {'emp': vals['empirical']['h1'], 'theory': vals['theory']['h1_learn']}, P)
            writer.add_scalars('Projection/H3', {'emp': vals['empirical']['h3'], 'theory': vals['theory']['h3_learn']}, P)
            
            # Add W0 variance comparison
            if 'W0_var' in vals['empirical'] and vals['theory']['lWT'] is not None:
                writer.add_scalars('W0_Variance', {'emp': vals['empirical']['W0_var'], 'theory': vals['theory']['lWT']}, P)
                
                chi = cfg.get('chi')
                if chi not in chi_data:
                    chi_data[chi] = {"P_vals": [], "W0_vars": [], "lWT_vals": [], "configs": []}
                
                chi_data[chi]["P_vals"].append(P)
                chi_data[chi]["W0_vars"].append(vals['empirical']['W0_var'])
                chi_data[chi]["lWT_vals"].append(vals['theory']['lWT'])
                chi_data[chi]["configs"].append(cfg)
        
        # Create separate W0 variance scatter plots for each chi value
        for chi, plot_data in sorted(chi_data.items()):
            P_vals = plot_data["P_vals"]
            W0_vars = plot_data["W0_vars"]
            lWT_vals = plot_data["lWT_vals"]
            configs = plot_data["configs"]
            
            if P_vals:
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.scatter(lWT_vals, W0_vars, s=100, alpha=0.7, edgecolors='k')
                
                # Add diagonal reference line
                min_val = min(min(lWT_vals), min(W0_vars))
                max_val = max(max(lWT_vals), max(W0_vars))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
                
                # Label points with d, P, chi
                for p, lwt, w0v, cfg in zip(P_vals, lWT_vals, W0_vars, configs):
                    d = cfg.get('d')
                    label = f'd={d}, P={int(p)}, χ={chi}'
                    ax.annotate(label, (lwt, w0v), fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
                
                ax.set_xlabel('Theory lWT', fontsize=12)
                ax.set_ylabel('Empirical W0[:,:,0] Variance', fontsize=12)
                ax.set_title(f'W0 Weight Variance vs Theory (χ={chi})', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                writer.add_figure(f'W0_Variance/Scatter_chi_{chi}', fig)
                plt.close(fig)
        
        writer.flush()

# --- Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()

    analyzer = ModelAnalyzer(use_cache=args.use_cache)
    model_dirs = analyzer.get_models()
    results = analyzer.run_analysis(model_dirs)
    analyzer.log_tensorboard(results)

if __name__ == "__main__":
    main()