import subprocess
import tempfile
import json
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.multiprocessing as mp
import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import traceback
from matplotlib.markers import MarkerStyle

torch.set_float32_matmul_precision('high')

# Set publication-quality styling
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
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
        relevant_keys = ["d", "P", "N", "chi", "kappa", "seed", "base_seed"]
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
        np.random.seed(0)
        X = np.random.randn(P, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / P
        
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
        
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            return float(match.group(1))
        else:
            return kappa
    except Exception as e:
        print(f"Warning: kappa_eff computation failed: {e}. Using bare kappa.")
        return kappa
    finally:
        try:
            if 'eig_json' in locals() and Path(eig_json).exists():
                Path(eig_json).unlink()
        except: pass

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
            ret = json.load(f)
        return ret
    except Exception as e:
        print(f"Julia error: {e}"); return {}
    finally:
        if to_path.exists(): to_path.unlink()

def split_theory_result(theo):
    if not isinstance(theo, dict): return {}, {}
    target = theo.get("target", {}) or {}
    perpendicular = theo.get("perpendicular", {}) or {}
    return (
        {k: safe_float(v) for k, v in target.items()},
        {k: safe_float(v) for k, v in perpendicular.items()},
    )

def safe_float(value):
    return float(value) if value is not None else float(np.nan)

def cached_theory_from_result(cached):
    if not isinstance(cached, dict): return {}, {}
    return (
        {
            "lH1T": safe_float(cached.get("theo_h", np.nan)),
            "lH3T": safe_float(cached.get("theo_h3", np.nan)),
            "lWT": safe_float(cached.get("theo_w", np.nan)),
            "mu1": safe_float(cached.get("h1_theory", np.nan)),
            "mu3": safe_float(cached.get("h3_theory", np.nan)),
        },
        {
            "lH1P": safe_float(cached.get("theo_h_nngp", np.nan)),
            "lH3P": safe_float(cached.get("theo_h3_nngp", np.nan)),
            "lWP": safe_float(cached.get("theo_w_nngp", np.nan)),
            "mu1": safe_float(cached.get("h1_nngp_theory", np.nan)),
            "mu3": safe_float(cached.get("h3_nngp_theory", np.nan)),
        },
    )

def theory_fields_are_finite(theory_pair):
    target_theory, perp_theory = theory_pair
    needed = [
        target_theory.get("lH1T", np.nan),
        target_theory.get("lH3T", np.nan),
        target_theory.get("lWT", np.nan),
        target_theory.get("mu1", np.nan),
        target_theory.get("mu3", np.nan),
        perp_theory.get("lH1P", np.nan),
        perp_theory.get("lH3P", np.nan),
        perp_theory.get("lWP", np.nan),
    ]
    return all(np.isfinite(x) for x in needed)

def theory_job_key(task_info, kappa_eff):
    cfg = task_info["cfg"]
    return (int(cfg["d"]), int(cfg["P"]), int(cfg["N"]), float(cfg["chi"]), float(kappa_eff))

def run_empirical_task(task_info):
    """GPU-bound Empirical Task."""
    m_dir, cfg = Path(task_info['path']), task_info['cfg']
    pt_path = m_dir / "model_final.pt" if (m_dir / "model_final.pt").exists() else m_dir / "model.pt"
    from FCN3Network import FCN3NetworkActivationGeneric
    sys.path.insert(0, str(Path(__file__).parent))
    from compute_h3_projections_fcn3 import compute_h3_projections_streaming
    
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
        X = torch.randn(3000, d, device=device) # DEVICE FIX
        with torch.no_grad():
            W0 = model.W0 
            W0_reshaped = W0.view(model.ensembles * W0.shape[1], d)
            cov_W0 = torch.matmul(W0_reshaped.t(), W0_reshaped) / (model.ensembles * W0.shape[1])
            eigvals_W0 = torch.linalg.eigvalsh(cov_W0).sort(descending=True).values.cpu().numpy()

        eigs = model.H_eig_random_svd(X, k=700)
        
        with torch.no_grad():
            x0_test = torch.randn(12000, d, device=device)
            out = model(x0_test)
            h1_sum = torch.einsum('pq,p->q', out, x0_test[:,0]).sum().item() / model.ensembles / x0_test.shape[0]

        h3_stats = compute_h3_projections_streaming(model, d=cfg['d'], P_total=1_000_000, batch_size=10_000, device=device)
        h3_target_eig = h3_stats['h3']['target']['second_moment']
        h3_perp_eig = h3_stats['h3']['perp']['second_moment']
        
        return {"emp_h": float(eigs[0].item()), "emp_w0": float(eigvals_W0[0]), "h1_emp": h1_sum, "h3_target_eig": h3_target_eig, "h3_perp_eig": h3_perp_eig}
    except Exception as e:
        print(f"Empirical Error {m_dir.name}: {e}"); traceback.print_exc(); return None
    finally:
        if device.type == 'cuda': torch.cuda.empty_cache()

# --- 3. Main Pipeline ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi", type=float, nargs='+', default=[80])
    parser.add_argument("--kappa", type=float, nargs='+', default=None)
    parser.add_argument("--d", type=int, nargs='+', default=None)
    parser.add_argument("--ignore-seeds", type=int, nargs='+', default=[])
    parser.add_argument("--truncate-last", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recompute-theory", action="store_true")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()
    
    ignored_seeds = set(args.ignore_seeds)
    results_dir = Path(args.results_dir)
    RESULTS_DIR = results_dir
    mp.set_start_method('spawn', force=True)
    
    multi_d_mode = args.d and len(args.d) > 1
    param_tuples = list(zip(args.d, args.chi, args.kappa or ([0.0]*len(args.d)))) if multi_d_mode else None
    
    all_dirs = sorted(list(results_dir.glob("d*/*seed*")))
    final_data = []
    to_compute_dirs = []
    to_compute_hashes = []

    # Step 1: Cache Check
    for m_dir in all_dirs:
        if not (m_dir / "config.json").exists(): continue
        with open(m_dir / "config.json") as f: cfg = json.load(f)
        if cfg['P'] > 10000: continue
        
        if multi_d_mode:
            c_d, c_chi, c_kappa = int(cfg.get("d",0)), float(cfg.get("chi",0)), float(cfg.get("kappa",0))
            if not any(c_d==dv and abs(c_chi-cv)<1e-6 and abs(c_kappa-kv)<1e-6 for dv, cv, kv in param_tuples): continue
        else:
            if not any(abs(float(cfg.get("chi",0))-c)<1e-6 for c in args.chi): continue
            if args.kappa and not any(abs(float(cfg.get("kappa",0))-k)<1e-6 for k in args.kappa): continue
            if args.d and int(cfg.get("d",0)) not in args.d: continue

        c_hash = CacheManager.get_config_hash(cfg)
        cached = CacheManager.load_result(c_hash)

        if cached and not args.force:
            if "kappa_eff" not in cached:
                cached["kappa_eff"] = compute_kappa_eff(cached["d"], cached["P"], cached["kappa"])
                CacheManager.save_result(c_hash, cached)
            
            if args.recompute_theory or "theo_h_nngp" not in cached:
                ct_all = run_theory_task({"d": cached["d"], "P": cached["P"], "n1": cached["N"], "n2": cached["N"], "chi": cached["chi"], "kappa": cached["kappa_eff"], "eps": EPSILON})
                _, ct_perp = split_theory_result(ct_all)
                cached.update({"theo_h_nngp": safe_float(ct_perp.get("lH1P", np.nan)), "theo_h3_nngp": safe_float(ct_perp.get("lH3P", np.nan))})
                CacheManager.save_result(c_hash, cached)
            final_data.append(cached)
        else:
            cached_kappa_eff = cached.get("kappa_eff") if cached else None
            cached_theory_pair = cached_theory_from_result(cached) if (cached and not args.recompute_theory) else ({}, {})
            to_compute_dirs.append({'path': str(m_dir), 'cfg': cfg, 'cached_kappa_eff': cached_kappa_eff, 'cached_theory_pair': cached_theory_pair})
            to_compute_hashes.append(c_hash)

    # Step 2: Computation logic exactly as original (deduplicated)
    if to_compute_dirs:
        kappa_map = {}
        for task_info in to_compute_dirs:
            cfg = task_info['cfg']
            key = (int(cfg["d"]), int(cfg["P"]), float(cfg["kappa"]))
            if task_info.get('cached_kappa_eff') is not None:
                kappa_map[key] = task_info['cached_kappa_eff']
            elif key not in kappa_map:
                kappa_map[key] = compute_kappa_eff(*key)

        kappa_effs = [kappa_map[(int(t['cfg']["d"]), int(t['cfg']["P"]), float(t['cfg']["kappa"]))] for t in to_compute_dirs]
        for idx, task in enumerate(to_compute_dirs): task['cfg']['kappa_eff'] = kappa_effs[idx]

        with ProcessPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            emp_results = list(executor.map(run_empirical_task, to_compute_dirs))
        
        theory_job_groups = {}
        theo_results = [None] * len(to_compute_dirs)
        for i, task in enumerate(to_compute_dirs):
            if not args.recompute_theory and theory_fields_are_finite(task['cached_theory_pair']):
                theo_results[i] = {"target": task['cached_theory_pair'][0], "perpendicular": task['cached_theory_pair'][1]}
            else:
                theory_job_groups.setdefault(theory_job_key(task, kappa_effs[i]), []).append(i)

        if theory_job_groups:
            keys = list(theory_job_groups.keys())
            params = [{"d": k[0], "P": k[1], "n1": k[2], "n2": k[2], "chi": k[3], "kappa": k[4], "eps": EPSILON} for k in keys]
            with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
                comp_theos = list(executor.map(run_theory_task, params))
            for key, theo in zip(keys, comp_theos):
                for idx in theory_job_groups[key]: theo_results[idx] = theo

        for i, (emp, theo) in enumerate(zip(emp_results, theo_results)):
            if emp is None: continue
            t_target, t_perp = split_theory_result(theo)
            res = {**to_compute_dirs[i]['cfg'], **emp,
                   "theo_h": safe_float(t_target.get("lH1T", np.nan)), "theo_w": safe_float(t_target.get("lWT", np.nan)), "theo_h3": safe_float(t_target.get("lH3T", np.nan)),
                   "theo_h_nngp": safe_float(t_perp.get("lH1P", np.nan)), "theo_h3_nngp": safe_float(t_perp.get("lH3P", np.nan)),
                   "h1_theory": safe_float(t_target.get("mu1", np.nan)), "h3_theory": safe_float(t_target.get("mu3", np.nan)),
                   "h1_nngp_theory": safe_float(t_perp.get("mu1", np.nan)), "h3_nngp_theory": safe_float(t_perp.get("mu3", np.nan)),
                   "kappa_eff": kappa_effs[i]}
            CacheManager.save_result(to_compute_hashes[i], res)
            final_data.append(res)

    # --- 4. Plotting Helper Functions ---

    def draw_top_errorbar(ax, *args, **kwargs):
        eb = ax.errorbar(*args, **kwargs)
        data_line, caplines, barlinecols = eb
        if data_line is not None: data_line.set_zorder(3000)
        for cap in caplines: cap.set_zorder(3001)
        for bar in barlinecols: bar.set_zorder(3001)
        return eb

    def add_d_legend_if_multi_d(ax=None):
        if multi_d_mode:
            if ax is None: ax = plt.gca()
            h, l = [], []
            for i, d_v in enumerate(unique_ds):
                s = groups[d_v][0]
                h.append(plt.Line2D([0], [0], color=colors[i], lw=3))
                l.append(f"d={int(d_v)}, χ={s['chi']}, κ={s['kappa']:.2g}, N={s['N']}")
            leg = ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=10)
            ax.add_artist(leg)

    def collapse_rows_to_seed_means(rows):
        grouped = defaultdict(list)
        for idx, row in enumerate(rows):
            s_id = row.get("seed") or row.get("base_seed") or idx
            grouped[(row.get("d"), row.get("P"), s_id)].append(row)
        collapsed = []
        for items in grouped.values():
            if len(items) == 1: collapsed.append(items[0]); continue
            merged = dict(items[0])
            for key in items[0].keys():
                if key in ["d", "P", "N", "chi", "kappa"]: continue
                vals = [it[key] for it in items if isinstance(it.get(key), (int, float))]
                if len(vals) == len(items): merged[key] = float(np.mean(vals))
            collapsed.append(merged)
        return collapsed

    # --- Plotting Implementation ---
    if not final_data: sys.exit("No data points.")
    unique_ds = sorted(set(r["d"] for r in final_data))
    unique_chis = sorted(set(r["chi"] for r in final_data))
    color_by = "d" if multi_d_mode else "chi"
    unique_vals = unique_ds if multi_d_mode else unique_chis
    groups = {v: [r for r in final_data if r[color_by] == v] for v in unique_vals}
    colors = [plt.colormaps["viridis"](v) for v in np.linspace(0, 0.8, len(unique_vals))]

    # 1. Main Eigenvalues Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    for i, (val, res_orig) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res_orig)
        c, u_p = colors[i], sorted({r["P"] for r in res})
        for ax, key, t_key in zip([ax1, ax2], ["emp_h", "emp_w0"], ["theo_h", "theo_w"]):
            means = [np.mean([r[key] for r in res if r["P"]==p]) for p in u_p]
            stds = [np.std([r[key] for r in res if r["P"]==p], ddof=1)/np.sqrt(len([r for r in res if r["P"]==p])) if len([r for r in res if r["P"]==p])>1 else 0 for p in u_p]
            draw_top_errorbar(ax, u_p, means, yerr=stds, color=c, marker='o', ls='-', lw=3)
            ax.plot(u_p, [np.mean([r[t_key] for r in res if r["P"]==p]) for p in u_p], '--', color=c, lw=3)
    for ax in [ax1, ax2]:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.set_xlabel("P")
        ax.legend([plt.Line2D([0],[0], color='black', marker='o'), plt.Line2D([0],[0], color='black', ls='--')], 
                  ["Model", "Theory"], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "eigenvalues_scan.png", bbox_inches='tight')

    # 2. Learnability Plots
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(10, 8))
        for i, (val, res_orig) in enumerate(groups.items()):
            res = collapse_rows_to_seed_means(res_orig)
            c, u_p = colors[i], sorted({r["P"] for r in res})
            key = "h1_emp" if mode=="h1" else "h3_target_eig"
            t_key, n_key = f"{mode}_theory", f"{mode}_nngp_theory"
            means = [np.mean([r[key] for r in res if r["P"]==p]) for p in u_p]
            draw_top_errorbar(plt, u_p, means, color=c, marker='o', lw=3)
            plt.plot(u_p, [np.mean([r[t_key] for r in res if r["P"]==p]) for p in u_p], '--', color=c, lw=3)
            plt.plot(u_p, [np.nanmean([r.get(n_key, np.nan) for r in res if r["P"]==p]) for p in u_p], ':', color=c, lw=3)
        plt.xscale('log');  plt.yscale('log'); plt.grid(True, alpha=0.3); plt.xlabel("P"); plt.title(f"{mode.upper()} Learnability")
        plt.legend([plt.Line2D([0],[0], color='black', marker='o'), plt.Line2D([0],[0], color='black', ls='--'), plt.Line2D([0],[0], color='black', ls=':')], 
                   ["Model", "Theory", "NNGP"], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
        add_d_legend_if_multi_d()
        plt.savefig(RESULTS_DIR / f"learnability_{mode}.png", bbox_inches='tight')

    # 3. Alpha Eigenvalues
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    for i, (val, res_orig) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res_orig)
        c = colors[i]
        for ax, key in zip([ax1, ax2], ["emp_h", "emp_w0"]):
            a_dict = defaultdict(list)
            for r in res: a_dict[np.log(r["P"])/np.log(r["d"])].append(r[key])
            u_a = sorted(a_dict.keys())
            ax.plot(u_a, [np.mean(a_dict[a]) for a in u_a], color=c, marker='o', lw=3)
    for ax in [ax1, ax2]:
        ax.set_xscale('log');ax.set_yscale('log');
        ax.grid(True, alpha=0.3); ax.set_xlabel(r"$\alpha$")
        ax.legend([plt.Line2D([0],[0], color='black', marker='o')], ["Model"], loc='upper center', bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "alpha_eigenvalues.png", bbox_inches='tight')

    print(f"Complete. Plots saved to {RESULTS_DIR}")