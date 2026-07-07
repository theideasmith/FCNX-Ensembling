import os
import json
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

# --- Theory Worker (Must be Top Level for Pickling) ---

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
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        with open(to_path, "r") as f:
            data = json.load(f)
        
        tgt = data.get("target", {})
        lWT = tgt.get("lWT")
        
        h1_learn = tgt.get('mu1')
        h3_learn = tgt.get('mu3')
        
        return P, {"h1_learn": h1_learn, "h3_learn": h3_learn, "lWT": lWT}
    except Exception as e:
        print(f"Theory error for P={P}: {e}")
        return P, None
    finally:
        if to_path.exists():
            to_path.unlink()

# --- Projection Worker (Must be Top Level) ---

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
        
        W0 = model.W0
        W0_first_dim = W0[:, :, :, 0]
        W0_var = torch.var(W0_first_dim).item()

        h1_sum, h3_sum = 0.0, 0.0
        num_batches = max(1, P_total // batch_size)
        
        with torch.no_grad():
            for _ in range(num_batches):
                X = torch.randn(config['num_seeds'], batch_size, config['d'], device=device)
                out = model(X) 
                x0 = X[:, :, 0]
                h1_sum += torch.mean(torch.einsum('sbe,sb->se', out, x0)).item()
                h3_sum += torch.mean(torch.einsum('sbe,sb->se', out, (x0**3 - 3*x0))).item()
        
        return config['P'], {"h1": h1_sum / num_batches, "h3": h3_sum / num_batches, "W0_var": W0_var}
    except Exception as e:
        return config['P'], {"error": str(e)}

# --- Analyzer Class ---

class ModelAnalyzer:
    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        self.cache_path = RESULTS_DIR / CACHE_NAME
        self.results_cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.use_cache and self.cache_path.exists():
            print(f"Loading existing cache from {self.cache_path}")
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self, results: Dict):
        # Update internal cache with new results and write to disk
        self.results_cache.update(results)
        with open(self.cache_path, "w") as f:
            json.dump(self.results_cache, f, indent=4)
        print(f"Cache updated at {self.cache_path}")

    def get_models(self) -> List[Path]:
        return sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('d')])
    
    def parse_config(self, model_dir: Path) -> Dict:
        parts = model_dir.name.split('_')
        config = {'num_seeds': 1, 'ens': 1}
        for p in parts:
            try:
                if p.startswith('d'): config['d'] = int(p[1:])
                elif p.startswith('P'): config['P'] = int(p[1:])
                elif p.startswith('N'): config['N'] = int(p[1:])
                elif p.startswith('chi'): config['chi'] = float(p[3:])
                elif p.startswith('kappa'): config['kappa'] = float(p[5:])
                elif p.startswith('nseeds'): config['num_seeds'] = int(p[6:])
                elif p.startswith('ens'): config['ens'] = int(p[3:])
            except: continue
        config['base_seed'] = 42
        return config

    def run_analysis(self, model_dirs: List[Path]) -> Dict:
        final_results = {}
        theory_tasks = []
        proj_tasks = []
        
        # Track which P values actually need processing
        to_process_p = []

        for md in model_dirs:
            config = self.parse_config(md)
            p_key = str(config['P'])

            # 1. Check if we already have this in the main cache
            if self.use_cache and p_key in self.results_cache:
                final_results[p_key] = self.results_cache[p_key]
                continue

            # 2. If not cached, prepare for computation
            checkpoint = next(md.rglob("model_final.pt"), None)
            if not checkpoint:
                print(f"Skipping {md}: No checkpoint found.")
                continue

            to_process_p.append(p_key)
            theory_tasks.append((config['P'], config['d'], config['N'], config['chi'], config['kappa'], 1e-3))
            proj_tasks.append((checkpoint, config, config['P'], 128, DEVICE))

        if not theory_tasks and not proj_tasks:
            if not final_results:
                print("No models found to analyze.")
            else:
                print("All models loaded from cache.")
            return final_results

        # Execute Theory tasks in parallel
        t_results_map = {}
        if theory_tasks:
            print(f"Processing {len(theory_tasks)} new theory models...")
            with Pool() as pool:
                t_results_map = dict(list(tqdm(pool.imap(theory_worker_task, theory_tasks), 
                                             total=len(theory_tasks), desc="Theory")))

        # Execute Empirical tasks sequentially (to avoid GPU OOM)
        e_results_map = {}
        if proj_tasks:
            print(f"Processing {len(proj_tasks)} new empirical models...")
            for task in tqdm(proj_tasks, desc="Empirical"):
                p_val, res = projection_worker_task(task)
                e_results_map[p_val] = res

        # 3. Combine new results and add to final set
        new_batch = {}
        for p_key in to_process_p:
            p_int = int(p_key)
            # Find the original config for this P
            cfg = next(self.parse_config(md) for md in model_dirs if str(self.parse_config(md)['P']) == p_key)
            
            combined = {
                "config": cfg,
                "empirical": e_results_map.get(p_int),
                "theory": t_results_map.get(p_int)
            }
            final_results[p_key] = combined
            new_batch[p_key] = combined

        # 4. Save the new results to the persistent cache
        if new_batch:
            self._save_cache(new_batch)

        return final_results

    def log_tensorboard(self, data: Dict):
        log_path = RESULTS_DIR / "tensorboard" / "hermite_analysis"
        writer = SummaryWriter(str(log_path))
        print(f"Logging results to: {log_path}")

        group_map = {}
        for p_str, vals in data.items():
            if not vals.get('empirical') or not vals.get('theory') or 'error' in vals['empirical']:
                continue
            P = float(p_str)
            cfg = vals.get('config', {})
            chi = cfg.get('chi')
            d = cfg.get('d')
            key = (chi, d)
            if key not in group_map:
                group_map[key] = []
            group_map[key].append({
                'P': P,
                'h1_emp': vals['empirical'].get('h1'),
                'h1_theory': vals['theory'].get('h1_learn'),
                'w0_emp': vals['empirical'].get('W0_var'),
                'w0_theory': vals['theory'].get('lWT')
            })
        
        for (chi, d), points in group_map.items():
            sorted_points = sorted(points, key=lambda x: x['P'])
            for pt in sorted_points:
                P_val = pt['P']
                writer.add_scalars(
                    f'Learnability/H1/chi_{chi}_d_{d}',
                    {'empirical': pt['h1_emp'], 'theory': pt['h1_theory']},
                    global_step=int(P_val)
                )
                writer.add_scalars(
                    f'W0_Variance/chi_{chi}_d_{d}',
                    {'empirical': pt['w0_emp'], 'theory': pt['w0_theory']},
                    global_step=int(P_val)
                )
        writer.flush()
        writer.close()
        print("Done logging.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cache', action='store_true', help="Use the master JSON cache to skip re-computation")
    args = parser.parse_args()

    analyzer = ModelAnalyzer(use_cache=args.use_cache)
    model_dirs = analyzer.get_models()
    results = analyzer.run_analysis(model_dirs)
    
    if results:
        analyzer.log_tensorboard(results)

if __name__ == "__main__":
    main()