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

# --- Theory Worker ---

def theory_worker_task(task_info):
    """Runs the Julia solver for a specific parameter set."""
    P, d, N, chi, kappa, eps, cache_key = task_info
    
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
        return cache_key, {
            "h1_learn": tgt.get('mu1'), 
            "h3_learn": tgt.get('mu3'), 
            "lWT": tgt.get("lWT")
        }
    except Exception as e:
        print(f"Theory error for {cache_key}: {e}")
        return cache_key, None
    finally:
        if to_path.exists():
            to_path.unlink()

# --- Projection Worker ---

def projection_worker_task(args):
    model_path, config, P_total, batch_size, device, cache_key = args
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
        num_seeds, ens, N, d = config['num_seeds'], config['ens'], config['N'], config['d']

        with torch.no_grad():
            W0 = model.W0  # shape: (num_seeds, ens, N, d)
            W0_reshaped = W0.view(num_seeds * ens * N, d)  # shape: (num_seeds*ens*N, d)
            cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (num_seeds * ens * N)
            W0_var = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy().max()

            h1_sum, h3_sum = torch.zeros(config['num_seeds'], config['ens'], device=device), torch.zeros(config['num_seeds'], config['ens'], device=device)
            num_batches = P_total // batch_size
            x0_norm = torch.zeros(config['num_seeds'], device=device)
            for _ in range(num_batches):
                X = torch.randn(config['num_seeds'], batch_size, config['d'], device=device)
                out = model(X) 
                x0 = X[:, :, 0]
                x0_norm+=x0.norm(dim=1)
                h1_sum += torch.einsum('sbe,sb->se', out, x0) 
                h3_sum += torch.einsum('sbe,sb->se', out, (x0**3 - 3*x0))
        
        return cache_key, {"h1": (h1_sum.mean() / P_total).item(), "h3": (h3_sum.mean(dim=1) / x0_norm).mean().item(), "W0_var": W0_var}
    except Exception as e:
        return cache_key, {"error": str(e)}

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

    def _save_cache(self, new_results: Dict):
        self.results_cache.update(new_results)
        with open(self.cache_path, "w") as f:
            json.dump(self.results_cache, f, indent=4)
        print(f"Cache updated at {self.cache_path}")

    def _generate_cache_key(self, config: Dict) -> str:
        """Creates a unique ID based on P, chi, d, and kappa."""
        return f"P{config['P']}_chi{config['chi']}_d{config['d']}_kappa{config['kappa']}"

    def get_models(self) -> List[Path]:
        return sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith('d')])
    
    def parse_config(self, model_dir: Path) -> Dict:
        parts = model_dir.name.split('_')
        config = {'num_seeds': 1, 'ens': 1, 'chi': 1.0, 'kappa': 0.0} # Defaults
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
        return config

    def run_analysis(self, model_dirs: List[Path]) -> Dict:
        final_results = {}
        theory_tasks = []
        proj_tasks = []
        configs_to_process = {}

        for md in model_dirs:
            print(md.name)
            config = self.parse_config(md)
            cache_key = self._generate_cache_key(config)
            print(cache_key)
            if self.use_cache and cache_key in self.results_cache:
                final_results[cache_key] = self.results_cache[cache_key]
                continue
            
            checkpoint = next(md.rglob("model_final.pt"), None)
            if not checkpoint:
                continue


            configs_to_process[cache_key] = config
            theory_tasks.append((config['P'], config['d'], config['N'], config['chi'], config['kappa'], 1e-3, cache_key))
            proj_tasks.append((checkpoint, config, 1000000, 10000, DEVICE, cache_key))

        if not theory_tasks and not proj_tasks:
            print("No new models to analyze (all loaded from cache or none found).")
            return final_results

        # Execute Theory tasks
        t_results_map = {}
        if theory_tasks:
            print(f"Processing {len(theory_tasks)} theory tasks...")
            with Pool() as pool:
                t_results_map = dict(list(tqdm(pool.imap(theory_worker_task, theory_tasks), 
                                             total=len(theory_tasks), desc="Theory")))

        # Execute Empirical tasks
        e_results_map = {}
        if proj_tasks:
            print(f"Processing {len(proj_tasks)} empirical tasks...")
            for task in tqdm(proj_tasks, desc="Empirical"):
                ckey, res = projection_worker_task(task)
                e_results_map[ckey] = res

        # Combine and Update
        new_batch = {}
        for ckey, cfg in configs_to_process.items():
            combined = {
                "config": cfg,
                "empirical": e_results_map.get(ckey),
                "theory": t_results_map.get(ckey)
            }
            final_results[ckey] = combined
            new_batch[ckey] = combined

        if new_batch:
            try:
                self._save_cache(new_batch)
            except Exception as e:

                print(f"  Warning: failed to save cache: {e}")

        return final_results

    def log_tensorboard(self, data: Dict):
        log_path = RESULTS_DIR / "tensorboard" / "hermite_analysis"
        writer = SummaryWriter(str(log_path))

        group_map = {}
        for ckey, vals in data.items():
            if not vals.get('empirical') or not vals.get('theory') or 'error' in vals['empirical']:
                continue
            
            cfg = vals['config']
            group_key = (cfg['chi'], cfg['d'], cfg['kappa'])
            if group_key not in group_map:
                group_map[group_key] = []
            
            group_map[group_key].append({
                'P': cfg['P'],
                'h1_emp': vals['empirical'].get('h1'),
                'h1_theory': vals['theory'].get('h1_learn'),
                'w0_emp': vals['empirical'].get('W0_var'),
                'w0_theory': vals['theory'].get('lWT')
            })
        
        for (chi, d, kappa), points in group_map.items():
            print(f"Logging for chi={chi}, d={d}, kappa={kappa} with {len(points)} points")
            sorted_points = sorted(points, key=lambda x: x['P'])
            tag_suffix = f"chi_{chi}_d_{d}_kappa_{kappa}"

            for pt in sorted_points:
                try:
                    writer.add_scalars(f'Learnability/H1/{tag_suffix}', 
                                    {'empirical': pt['h1_emp'], 'theory': pt['h1_theory']}, pt['P'])
                    writer.add_scalars(f'W0_Variance/{tag_suffix}', 
                                    {'empirical': pt['w0_emp'], 'theory': pt['w0_theory']}, pt['P'])
                except Exception as e:
                    print(f"  Warning: failed to log point P={pt['P']} for {tag_suffix}: {e}")

        
        writer.flush()
        writer.close()
        print(f"TensorBoard logging complete: {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()

    analyzer = ModelAnalyzer(use_cache=args.use_cache)
    model_dirs = analyzer.get_models()
    results = analyzer.run_analysis(model_dirs)
    print(f"Analysis complete for {len(results)} configurations.")
    if results:
        analyzer.log_tensorboard(results)

if __name__ == "__main__":
    main()