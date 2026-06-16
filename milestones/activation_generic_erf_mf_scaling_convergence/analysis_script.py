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
        relevant_keys = ["d", "P", "N", "chi", "kappa", "seed", "base_seed"]
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
        # Compute P x P arcsin kernel eigenvalues
        # Compute P x P arcsin kernel eigenvalues
        np.random.seed(0)
        X = np.random.randn(P, d).astype(np.float32)
        X = np.random.randn(P, d).astype(np.float32)
        X_torch = torch.from_numpy(X)
        K = arcsin_kernel(X_torch)
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / P
        eigvals = torch.linalg.eigvalsh(K).cpu().numpy() / P
        
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
            ret = json.load(f)
        return ret
            ret = json.load(f)
        return ret
    except Exception as e:
        print(f"Julia error: {e}")
        return {}
    finally:
        if to_path.exists(): to_path.unlink()


def split_theory_result(theo):
    if not isinstance(theo, dict):
        return {}, {}
    target = theo.get("target", {}) or {}
    perpendicular = theo.get("perpendicular", {}) or {}
    return (
        {k: safe_float(v) for k, v in target.items()},
        {k: safe_float(v) for k, v in perpendicular.items()},
    )


def safe_float(value):
    return float(value) if value is not None else float(np.nan)


def cached_theory_from_result(cached):
    if not isinstance(cached, dict):
        return {}, {}
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
        perp_theory.get("mu1", np.nan),
        perp_theory.get("mu3", np.nan),
    ]
    return all(np.isfinite(x) for x in needed)


def theory_job_key(task_info, kappa_eff):
    cfg = task_info["cfg"]
    return (
        int(cfg["d"]),
        int(cfg["P"]),
        int(cfg["N"]),
        float(cfg["chi"]),
        float(kappa_eff),
    )


def split_theory_result(theo):
    if not isinstance(theo, dict):
        return {}, {}
    target = theo.get("target", {}) or {}
    perpendicular = theo.get("perpendicular", {}) or {}
    return (
        {k: safe_float(v) for k, v in target.items()},
        {k: safe_float(v) for k, v in perpendicular.items()},
    )


def safe_float(value):
    return float(value) if value is not None else float(np.nan)


def cached_theory_from_result(cached):
    if not isinstance(cached, dict):
        return {}, {}
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
        perp_theory.get("mu1", np.nan),
        perp_theory.get("mu3", np.nan),
    ]
    return all(np.isfinite(x) for x in needed)


def theory_job_key(task_info, kappa_eff):
    cfg = task_info["cfg"]
    return (
        int(cfg["d"]),
        int(cfg["P"]),
        int(cfg["N"]),
        float(cfg["chi"]),
        float(kappa_eff),
    )

def run_empirical_task(task_info):
    """GPU-bound Empirical Task."""
    m_dir, cfg = Path(task_info['path']), task_info['cfg']
    pt_path = m_dir / "model_final.pt" if (m_dir / "model_final.pt").exists() else m_dir / "model.pt"
    from FCN3Network import FCN3NetworkActivationGeneric
    
    # Import h3 projection function
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
        P_total, batch_size = 10_000, 10_000#2_000_000, 10_000
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
            for _ in range(P_total // batch_size):
                X_batch = torch.randn(batch_size, d, device=device)
                out = model(X_batch)
                x0 = X_batch[:, 0]
                h3_comp = (x0**3 - 3*x0) / 6**0.5
                # Graham-Schmidt orthogonalization to get h3 component
                remainder = out - (h1_sum * x0).unsqueeze(-1)
                proj3_target_sum = torch.einsum('pq,p->q', remainder, h3_comp).sum().item() / model.ensembles  # Average over ensembles
                # print(d_h3_sum)
                h3_sum += proj3_target_sum
            # print("H3 Sum is: ", h3_sum / P_total)
            # Projection using graham schmidt orthogonalization to get h3 component
            torch.manual_seed(int(cfg['seed'] * 3.14) if 'seed' in cfg else 4324)  # Reset seed for reproducibility
            x0 = torch.randn(12000, d, device=device)  # Sample new x0 for projection
            out = model(x0)  # Get model output for x0
            linear_component = torch.einsum('pq,p->q', out, x0[:,0]).sum().item() * x0[:,0].unsqueeze(-1) / model.ensembles / x0.shape[0]  # Average over ensembles
            h3_comp = (x0[:,0]**3 - 3*x0[:,0]) / 6**0.5
            h1_sum = torch.einsum('pq,p->q', out, x0[:,0]).sum().item() / model.ensembles / x0.shape[0]  # Average over ensembles and samples
            remainder = out - linear_component
            # proj3_target_sum = torch.einsum('pq,p->q', remainder, h3_comp).sum().item() / model.ensembles  # Average over ensembles
            # # y_k is the projection of the target h3 component onto the normalized target, which is 1/sqrt(6) for the standard normal distribution   
            # y_k = torch.einsum('p,p->', h3_comp, x0[:,0] + 0.03 * h3_comp).item() / x0.shape[0]  # Average over samples
            # h3_sum = proj3_target_sum / x0.shape[0] / 0.03  # Normalize by target scaling and number of samples
            # print("y_k is: ", y_k)
            # print("H3 Sum is: ", h3_sum)
        P_total = 1_000_000
        # Compute h3 eigenvalues using high-precision streaming (P_total=200M)
        print(f"Computing h3 projections with P_total={P_total} for d={cfg['d']}, P={cfg['P']}...")
        h3_stats = compute_h3_projections_streaming(
            model, 
            d=cfg['d'],
            P_total=10_000_000,
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
    parser.add_argument("--d", type=int, nargs='+', default=None, help="Filter by input dimension d. If multiple d values provided, must match length of chi and kappa.")
    parser.add_argument("--d", type=int, nargs='+', default=None, help="Filter by input dimension d. If multiple d values provided, must match length of chi and kappa.")
    parser.add_argument("--ignore-seeds", type=int, nargs='+', default=[], help="Seed values to exclude from analysis")
    parser.add_argument("--truncate-last", type=int, default=0, help="Drop the last N measurements by largest unique P before plotting")
    parser.add_argument("--truncate-last", type=int, default=0, help="Drop the last N measurements by largest unique P before plotting")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recompute-theory", action="store_true", help="Recompute theory while keeping empirical cache")
    parser.add_argument("--recompute-theory", action="store_true", help="Recompute theory while keeping empirical cache")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Directory containing result subfolders")
    args = parser.parse_args()
    ignored_seeds = set(args.ignore_seeds)
    results_dir = Path(args.results_dir)
    RESULTS_DIR = results_dir  # Update global variable with argument value
    mp.set_start_method('spawn', force=True)
    
    # Validate and prepare (d, chi, kappa) tuples
    # If multiple d values are provided, they pair with chi/kappa in order
    multi_d_mode = args.d and len(args.d) > 1
    param_tuples = None  # List of (d, chi, kappa) tuples to match
    
    if multi_d_mode:
        # Validate that d, chi, kappa have the same length
        num_d = len(args.d)
        if len(args.chi) != num_d:
            print(f"Error: --d has {num_d} values but --chi has {len(args.chi)} values. They must match.")
            sys.exit(1)
        if args.kappa and len(args.kappa) != num_d:
            print(f"Error: --d has {num_d} values but --kappa has {len(args.kappa)} values. They must match.")
            sys.exit(1)
        # Default kappa to first value if not provided
        if not args.kappa:
            args.kappa = [args.kappa[0]] * num_d if args.kappa else [0.0] * num_d
        param_tuples = list(zip(args.d, args.chi, args.kappa))
    
    # Validate and prepare (d, chi, kappa) tuples
    # If multiple d values are provided, they pair with chi/kappa in order
    multi_d_mode = args.d and len(args.d) > 1
    param_tuples = None  # List of (d, chi, kappa) tuples to match
    
    if multi_d_mode:
        # Validate that d, chi, kappa have the same length
        num_d = len(args.d)
        if len(args.chi) != num_d:
            print(f"Error: --d has {num_d} values but --chi has {len(args.chi)} values. They must match.")
            sys.exit(1)
        if args.kappa and len(args.kappa) != num_d:
            print(f"Error: --d has {num_d} values but --kappa has {len(args.kappa)} values. They must match.")
            sys.exit(1)
        # Default kappa to first value if not provided
        if not args.kappa:
            args.kappa = [args.kappa[0]] * num_d if args.kappa else [0.0] * num_d
        param_tuples = list(zip(args.d, args.chi, args.kappa))
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
        
        # Filtering logic: check against parameter tuples if multi_d_mode
        if multi_d_mode:
            # Check if this config matches any of the (d, chi, kappa) tuples
            cfg_d = int(cfg.get("d", 0))
            cfg_chi = float(cfg.get("chi", 0))
            cfg_kappa = float(cfg.get("kappa", 0))
            matches_tuple = False
            for d_val, chi_val, kappa_val in param_tuples:
                if (cfg_d == d_val and 
                    abs(cfg_chi - chi_val) < 1e-6 and 
                    abs(cfg_kappa - kappa_val) < 1e-6):
                    matches_tuple = True
                    break
            if not matches_tuple:
                continue
        else:
            # Original filtering logic
            if not any(abs(float(cfg.get("chi", 0)) - c) < 1e-6 for c in args.chi): 
                continue
            if args.kappa and not any(abs(float(cfg.get("kappa", 0)) - k) < 1e-6 for k in args.kappa): 
                continue
            if args.d and int(cfg.get("d", 0)) not in args.d: 
                continue
        
        # Filtering logic: check against parameter tuples if multi_d_mode
        if multi_d_mode:
            # Check if this config matches any of the (d, chi, kappa) tuples
            cfg_d = int(cfg.get("d", 0))
            cfg_chi = float(cfg.get("chi", 0))
            cfg_kappa = float(cfg.get("kappa", 0))
            matches_tuple = False
            for d_val, chi_val, kappa_val in param_tuples:
                if (cfg_d == d_val and 
                    abs(cfg_chi - chi_val) < 1e-6 and 
                    abs(cfg_kappa - kappa_val) < 1e-6):
                    matches_tuple = True
                    break
            if not matches_tuple:
                continue
        else:
            # Original filtering logic
            if not any(abs(float(cfg.get("chi", 0)) - c) < 1e-6 for c in args.chi): 
                continue
            if args.kappa and not any(abs(float(cfg.get("kappa", 0)) - k) < 1e-6 for k in args.kappa): 
                continue
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
            needs_theory_refresh = args.recompute_theory or (
                "theo_h_nngp" not in cached or not np.isfinite(cached.get("theo_h_nngp", np.nan)) or
                "theo_h3_nngp" not in cached or not np.isfinite(cached.get("theo_h3_nngp", np.nan))
            )
            if needs_theory_refresh:
                cached_theory_all = run_theory_task({
                    "d": cached["d"],
                    "P": cached["P"],
                    "n1": cached["N"],
                    "n2": cached["N"],
                    "chi": cached["chi"],
                    "kappa": cached["kappa_eff"],
                    "eps": EPSILON,
                })
                cached_theory, cached_perp = split_theory_result(cached_theory_all)
                cached["theo_h_nngp"] = safe_float(cached_perp.get("lH1P", np.nan))
                cached["theo_h3_nngp"] = safe_float(cached_perp.get("lH3P", np.nan))
                cached["theo_w_nngp"] = safe_float(cached_perp.get("lWP", np.nan))
                cached["h1_nngp_theory"] = safe_float(cached_perp.get("mu1", np.nan))
                cached["h3_nngp_theory"] = safe_float(cached_perp.get("mu3", np.nan))
                CacheManager.save_result(c_hash, cached)
            needs_theory_refresh = args.recompute_theory or (
                "theo_h_nngp" not in cached or not np.isfinite(cached.get("theo_h_nngp", np.nan)) or
                "theo_h3_nngp" not in cached or not np.isfinite(cached.get("theo_h3_nngp", np.nan))
            )
            if needs_theory_refresh:
                cached_theory_all = run_theory_task({
                    "d": cached["d"],
                    "P": cached["P"],
                    "n1": cached["N"],
                    "n2": cached["N"],
                    "chi": cached["chi"],
                    "kappa": cached["kappa_eff"],
                    "eps": EPSILON,
                })
                cached_theory, cached_perp = split_theory_result(cached_theory_all)
                cached["theo_h_nngp"] = safe_float(cached_perp.get("lH1P", np.nan))
                cached["theo_h3_nngp"] = safe_float(cached_perp.get("lH3P", np.nan))
                cached["theo_w_nngp"] = safe_float(cached_perp.get("lWP", np.nan))
                cached["h1_nngp_theory"] = safe_float(cached_perp.get("mu1", np.nan))
                cached["h3_nngp_theory"] = safe_float(cached_perp.get("mu3", np.nan))
                CacheManager.save_result(c_hash, cached)
            final_data.append(cached)
        else:
            # Even when --force, reuse cached kappa_eff if available
            cached_kappa_eff = None
            if cached and "kappa_eff" in cached:
                cached_kappa_eff = cached["kappa_eff"]
            cached_theory_pair = ({}, {}) if args.recompute_theory else (cached_theory_from_result(cached) if cached else ({}, {}))
            to_compute_dirs.append({'path': str(m_dir), 'cfg': cfg, 'cached_kappa_eff': cached_kappa_eff, 'cached_theory_pair': cached_theory_pair})
            to_compute_hashes.append(c_hash)

    # Step 2: Parallel Computation for Cache Misses
    if to_compute_dirs:
        print(f"Processing {len(to_compute_dirs)} cache misses...")
        
        # Step 2a: Compute kappa_eff for all configurations (deduplicated)
        print("Computing effective ridge (kappa_eff) with deduplication...")
        # Build unique keys for kappa computation: (d, P, kappa)
        kappa_map = {}
        keys_to_compute = []
        # Step 2a: Compute kappa_eff for all configurations (deduplicated)
        print("Computing effective ridge (kappa_eff) with deduplication...")
        # Build unique keys for kappa computation: (d, P, kappa)
        kappa_map = {}
        keys_to_compute = []
        for task_info in to_compute_dirs:
            cfg = task_info['cfg']
            key = (int(cfg["d"]), int(cfg["P"]), float(cfg["kappa"]))
            key = (int(cfg["d"]), int(cfg["P"]), float(cfg["kappa"]))
            cached_kappa_eff = task_info.get('cached_kappa_eff')
            if cached_kappa_eff is not None:
                kappa_map[key] = cached_kappa_eff
                kappa_map[key] = cached_kappa_eff
            else:
                # Mark key for computation once
                if key not in kappa_map:
                    kappa_map[key] = None
                    keys_to_compute.append(key)

        # Compute kappa_eff once per unique key
        for d_key, P_key, kappa_bare in keys_to_compute:
            try:
                kappa_val = compute_kappa_eff(d_key, P_key, kappa_bare)
                print(f"  d={d_key}, P={P_key}: kappa_bare={kappa_bare:.6f} -> kappa_eff={kappa_val:.6f}")
            except Exception as e:
                print(f"  d={d_key}, P={P_key}: kappa_eff computation failed: {e}. Using bare kappa.")
                kappa_val = kappa_bare
            kappa_map[(d_key, P_key, kappa_bare)] = kappa_val

        # Assign kappa_eff to each task_info and build kappa_effs list preserving order
        kappa_effs = []
        for task_info in to_compute_dirs:
            cfg = task_info['cfg']
            key = (int(cfg["d"]), int(cfg["P"]), float(cfg["kappa"]))
            kappa_eff = kappa_map.get(key, cfg.get('kappa'))
            kappa_effs.append(kappa_eff)
            task_info['cfg']['kappa_eff'] = kappa_eff
            src = "cache" if task_info.get('cached_kappa_eff') is not None else "computed"
            print(f"  d={key[0]}, P={key[1]}: kappa_eff={kappa_eff:.6f} ({src})")
        
        # Parallel Empirical (GPU)
        with ProcessPoolExecutor(max_workers=MAX_GPU_WORKERS) as executor:
            emp_results = list(executor.map(run_empirical_task, to_compute_dirs))
        
        # Parallel Theory (CPU) - using kappa_eff
        theory_job_groups = {}
        theo_results: list[dict[str, dict[str, object]] | None] = [None] * len(to_compute_dirs)
        for i, task_info in enumerate(to_compute_dirs):
            cached_theory_pair = task_info.get('cached_theory_pair', ({}, {}))
            if not args.recompute_theory and cached_theory_pair and theory_fields_are_finite(cached_theory_pair):
                theo_results[i] = {
                    "target": cached_theory_pair[0],
                    "perpendicular": cached_theory_pair[1],
                }
            else:
                key = theory_job_key(task_info, kappa_effs[i])
                theory_job_groups.setdefault(key, []).append(i)

        if theory_job_groups:
            theory_job_keys = list(theory_job_groups.keys())
            theory_params = [{"d": key[0], "P": key[1], 
                              "n1": key[2], "n2": key[2], 
                              "chi": key[3], "kappa": key[4], "eps": EPSILON} 
                             for key in theory_job_keys]
            with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
                computed_theories = list(executor.map(run_theory_task, theory_params))
            for key, theo in zip(theory_job_keys, computed_theories):
                for idx in theory_job_groups[key]:
                    theo_results[idx] = theo

        # Merge and Save to Cache
        for i, (emp, theo) in enumerate(zip(emp_results, theo_results)):
            if emp is None: continue
            target_theory, perp_theory = split_theory_result(theo)
            res = {**to_compute_dirs[i]['cfg'], **emp,
                   "theo_h": safe_float(target_theory.get("lH1T", np.nan)), "theo_w": safe_float(target_theory.get("lWT", np.nan)), "theo_h3": safe_float(target_theory.get("lH3T", np.nan)),
                   "theo_h_nngp": safe_float(perp_theory.get("lH1P", np.nan)), "theo_h3_nngp": safe_float(perp_theory.get("lH3P", np.nan)), "theo_w_nngp": safe_float(perp_theory.get("lWP", np.nan)),
                   "h1_theory": safe_float(target_theory.get("mu1", np.nan)), "h3_theory": safe_float(target_theory.get("mu3", np.nan)),
                   "h1_nngp_theory": safe_float(perp_theory.get("mu1", np.nan)), "h3_nngp_theory": safe_float(perp_theory.get("mu3", np.nan)),
                   "kappa_eff": kappa_effs[i]}
            CacheManager.save_result(to_compute_hashes[i], res)
            if args.force:
                print(f"  refreshed cache with NNGP fields for d={res['d']}, P={res['P']}")
            final_data.append(res)

    # --- 4. Plotting ---
    if not final_data:
        print("No data points available."); sys.exit()

    if args.truncate_last:
        unique_ps = sorted({r["P"] for r in final_data})
        if args.truncate_last >= len(unique_ps):
            print(f"Truncating all data because truncate-last={args.truncate_last} and only {len(unique_ps)} unique P values are available.")
            final_data = []
        else:
            kept_ps = set(unique_ps[:-args.truncate_last])
            dropped_ps = unique_ps[-args.truncate_last:]
            final_data = [r for r in final_data if r["P"] in kept_ps]
            print(f"Truncated last {args.truncate_last} unique P values: {dropped_ps}")

    if not final_data:
        print("No data points available after truncation."); sys.exit()

    unique_kappas = sorted(set(r["kappa"] for r in final_data))
    unique_chis = sorted(set(r["chi"] for r in final_data))
    unique_ds = sorted(set(r["d"] for r in final_data))
    
    # Determine coloring strategy: multi_d_mode > multi_chi_mode > multi_kappa_mode
    multi_chi_mode = len(args.chi) > 1 and len(unique_chis) > 1
    
    if multi_d_mode:
        color_by = "d"
        unique_vals = unique_ds
    else:
        color_by = "chi" if multi_chi_mode else ("kappa" if len(unique_kappas) > 1 else "chi")
        unique_vals = unique_kappas if color_by == "kappa" else unique_chis
    
    groups = {val: [r for r in final_data if r[color_by] == val] 
              for val in unique_vals}
    
    num_colors = len(unique_vals)
    if multi_d_mode:
        # Use viridis colormap for multiple d values with darker color range
        cmap = plt.colormaps["viridis"]
        color_positions = np.linspace(0.0, 0.8, num_colors)
        colors = [cmap(v) for v in color_positions]
    elif multi_chi_mode:
        cmap = plt.colormaps["viridis"]
        color_positions = np.linspace(0.1, 0.9, max(num_colors, 1))
        colors = [cmap(v) for v in color_positions]
    elif num_colors < 5:
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = colors_list[:num_colors]
    else:
        cmap = plt.colormaps["plasma"]
        color_positions = np.linspace(0.1, 0.9, max(num_colors, 1))
        colors = [cmap(v) for v in color_positions]

    exp_marker = MarkerStyle("o")
    theo_marker = MarkerStyle("s") if (multi_chi_mode or multi_d_mode) else MarkerStyle("o")

    # Fixed colors for consistency: orange (empirical), blue (theory), green (NNGP)
    color_empirical = '#FF7F0E'
    color_theory = '#1F77B4'
    color_nngp = '#2CA02C'
    single_d_mode = len(unique_ds) == 1

    def series_color(i):
        return colors[i]

    def role_color(role: str, group_color):
        if not single_d_mode:
            return group_color
        if role == "empirical":
            return color_empirical
        if role == "theory":
            return color_theory
        if role == "nngp":
            return color_nngp
        return group_color

    def add_d_legend_if_multi_d():
        """Add a second legend for d values with χ, κ, N when multi_d_mode is True."""
        if multi_d_mode and color_by == "d":
            d_handles = []
            d_labels = []
            for i, d_val in enumerate(unique_ds):
                # Get sample values from this d's group
                d_group = groups.get(d_val, [])
                if d_group:
                    sample = d_group[0]
                    chi_val = sample.get("chi", "?")
                    kappa_val = sample.get("kappa", "?")
                    n_val = sample.get("N", "?")
                    label = f"d={int(d_val) if float(d_val).is_integer() else d_val}, $\chi={chi_val:.2g}, \kappa$={kappa_val:.2g}, N={n_val}"
                else:
                    label = f"d={int(d_val) if float(d_val).is_integer() else d_val}"
                h = plt.Line2D([0], [0], color=series_color(i), linewidth=3)
                d_handles.append(h)
                d_labels.append(label)
            leg2 = plt.legend(d_handles, d_labels, loc='lower right', framealpha=0.95, fontsize=5)
            plt.gca().add_artist(leg2)

    def get_d_str_for_filename():
        """Generate a d-list string for filenames when multi_d_mode."""
        if multi_d_mode:
            d_str = "_".join(str(int(d) if float(d).is_integer() else d) for d in unique_ds)
            return f"d{d_str}"
        else:
            d = final_data[0]["d"] if final_data else 0
            return f"d{d}"

    def get_data_label(val, is_first=False):
        """Generate a label with d, χ, κ, N for multi_d_mode."""
        if multi_d_mode and color_by == "d":
            d_group = groups.get(val, [])
            if d_group and not is_first:
                sample = d_group[0]
                chi_val = sample.get("chi", "?")
                kappa_val = sample.get("kappa", "?")
                n_val = sample.get("N", "?")
                return f"d={int(val) if float(val).is_integer() else val}, $\chi={chi_val:.2g}, \kappa$={kappa_val:.2g}, N={n_val}"
            elif d_group:
                sample = d_group[0]
                chi_val = sample.get("chi", "?")
                kappa_val = sample.get("kappa", "?")
                n_val = sample.get("N", "?")
                return f"d={int(val) if float(val).is_integer() else val}, $\chi={chi_val:.2g}, \kappa$={kappa_val:.2g}, N={n_val}"
        else:
            if color_by == "chi":
                return r"$\chi = $" + str(val) if not is_first else ""
            else:
                return f"{color_by}={val}" if not is_first else ""
        return ""

    def draw_top_errorbar(ax, *args, **kwargs):
        """Draw an errorbar and force all bar/cap artists above other plot elements."""
        eb = ax.errorbar(*args, **kwargs)
        try:
            data_line, caplines, barlinecols = eb
            if data_line is not None:
                data_line.set_zorder(3000)
                data_line.set_clip_on(False)
            for cap in caplines:
                cap.set_zorder(3001)
                cap.set_clip_on(False)
            for barcol in barlinecols:
                barcol.set_zorder(3001)
                barcol.set_clip_on(False)
        except Exception:
            pass
        return eb

    def get_seed_identifier(row, fallback_index):
        for seed_key in ("seed", "base_seed", "torch_seed", "rng_seed"):
            if seed_key in row and row[seed_key] is not None:
                try:
                    return int(row[seed_key])
                except (TypeError, ValueError):
                    pass
        for path_key in ("run_dir", "path", "checkpoint"):
            raw_path = row.get(path_key)
            if isinstance(raw_path, str):
                match = re.search(r"seed(\d+)", raw_path)
                if match:
                    return int(match.group(1))
        return f"row_{fallback_index}"

    def collapse_rows_to_seed_means(rows):
        grouped = defaultdict(list)
        for idx, row in enumerate(rows):
            seed_id = get_seed_identifier(row, idx)
            key = (row.get("d"), row.get("P"), seed_id)
            grouped[key].append(row)

        id_like_keys = {
            "d", "P", "N", "chi", "kappa", "kappa_eff",
            "seed", "base_seed", "torch_seed", "rng_seed",
        }
        collapsed = []
        for items in grouped.values():
            if len(items) == 1:
                collapsed.append(items[0])
                continue

            merged = dict(items[0])
            all_keys = set().union(*(it.keys() for it in items))
            for key in all_keys:
                if key in id_like_keys:
                    continue
                vals = []
                for it in items:
                    value = it.get(key)
                    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                        vals.append(float(value))
                if len(vals) == len(items):
                    merged[key] = float(np.mean(vals))
            merged["ensemble_rows_collapsed"] = len(items)
            collapsed.append(merged)
        return collapsed

    def plot_empirical_mean_with_errorbars(ax, xs, grouped_values, color, marker):
        means = [float(np.mean(grouped_values[x])) for x in xs]
        stds = [float(np.std(grouped_values[x], ddof=1) / np.sqrt(len(grouped_values[x]))) if len(grouped_values[x]) > 1 else 0.0 for x in xs]
        draw_top_errorbar(
            ax,
            xs,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            linestyle='-',
            linewidth=3,
            markersize=6,
            alpha=0.85,
            capsize=4,
            elinewidth=1.5,
            ecolor='black',
            zorder=1000, clip_on=False,
        )

    # Eigenvalues
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        ax1.scatter(p_vals, [r["emp_h"] for r in res], color=role_color("empirical", c), label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        ax2.scatter(p_vals, [r["emp_w0"] for r in res], color=role_color("empirical", c), label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        p_to_nngp_w = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
            p_to_nngp_w[r["P"]].append(r.get("theo_w_nngp", np.nan))
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        mean_nngp_w = [np.mean([v for v in p_to_nngp_w[p] if np.isfinite(v)]) if any(np.isfinite(p_to_nngp_w[p])) else np.nan for p in unique_p]
        draw_top_errorbar(ax1, unique_p, mean_emp_h, yerr=[float(np.std(p_to_emp_h[p], ddof=1) / np.sqrt(len(p_to_emp_h[p]))) if len(p_to_emp_h[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax1.plot(unique_p, mean_theo_h, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        draw_top_errorbar(ax2, unique_p, mean_emp_w, yerr=[float(np.std(p_to_emp_w[p], ddof=1) / np.sqrt(len(p_to_emp_w[p]))) if len(p_to_emp_w[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax2.plot(unique_p, mean_theo_w, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        ax2.plot(unique_p, mean_nngp_w, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    # Dummy plot for legend
    ax1.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    ax1.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax2.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    ax2.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax2.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    d = final_data[0]["d"] if final_data else 0
    ax1.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax2.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel("P"); ax.set_xscale('log')
    N = final_data[0]["N"] if final_data else 0
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Separate plots
    # Lambda H
    fig_h = plt.figure(figsize=(10, 10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        label = get_data_label(val, i == 0)
        plt.scatter(p_vals, [r["emp_h"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        p_to_emp_h = defaultdict(list)
        p_to_theo_h = defaultdict(list)
        p_to_nngp_h = defaultdict(list)
        for r in res:
            p_to_emp_h[r["P"]].append(r["emp_h"])
            p_to_theo_h[r["P"]].append(r["theo_h"])
            p_to_nngp_h[r["P"]].append(r.get("theo_h_nngp", np.nan))
        unique_p = sorted(p_to_emp_h.keys())
        mean_emp_h = [np.mean(p_to_emp_h[p]) for p in unique_p]
        mean_theo_h = [np.mean(p_to_theo_h[p]) for p in unique_p]
        mean_nngp_h = [np.mean(p_to_nngp_h[p]) for p in unique_p]
        draw_top_errorbar(plt, unique_p, mean_emp_h, yerr=[float(np.std(p_to_emp_h[p], ddof=1) / np.sqrt(len(p_to_emp_h[p]))) if len(p_to_emp_h[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_h, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    plt.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda^H_{1,*}$)")
    plt.ylim(0,None)  
    plt.xlabel("P (dataset size)"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Lambda W
    fig_w = plt.figure(figsize=(10,10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        label = get_data_label(val, i == 0)
        plt.scatter(p_vals, [r["emp_w0"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        p_to_emp_w = defaultdict(list)
        p_to_theo_w = defaultdict(list)
        p_to_nngp_w = defaultdict(list)
        for r in res:
            p_to_emp_w[r["P"]].append(r["emp_w0"])
            p_to_theo_w[r["P"]].append(r["theo_w"])
            p_to_nngp_w[r["P"]].append(r.get("theo_w_nngp", np.nan))
        unique_p = sorted(p_to_emp_w.keys())
        mean_emp_w = [np.mean(p_to_emp_w[p]) for p in unique_p]
        mean_theo_w = [np.mean(p_to_theo_w[p]) for p in unique_p]
        mean_nngp_w = [np.mean([v for v in p_to_nngp_w[p] if np.isfinite(v)]) if any(np.isfinite(p_to_nngp_w[p])) else np.nan for p in unique_p]
        draw_top_errorbar(plt, unique_p, mean_emp_w, yerr=[float(np.std(p_to_emp_w[p], ddof=1) / np.sqrt(len(p_to_emp_w[p]))) if len(p_to_emp_w[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_w, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_w, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    plt.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")
    plt.title(r"Linear Target Eigenvalues ($\lambda_W^*$)")
    plt.ylabel(r"$\lambda_W^* = v^T \Sigma_w v$");
    plt.xlabel("P (dataset size)"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.xscale('log')
    plt.ylim(0,None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Learnability
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(15, 10))
        for i, (val, res) in enumerate(groups.items()):
            res = collapse_rows_to_seed_means(res)
            c = series_color(i)
            p_vals = [r["P"] for r in res]
            label = get_data_label(val, i == 0)
            plt.scatter(p_vals, [r[f"{mode}_emp"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
            d = res[0]["d"] if res else 0
            plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d" if i == 0 else None)

            # Mean lines
            p_to_emp = defaultdict(list)
            p_to_theo = defaultdict(list)
            p_to_nngp = defaultdict(list)
            for r in res:
                p_to_emp[r["P"]].append(r[f"{mode}_emp"])
                p_to_theo[r["P"]].append(r[f"{mode}_theory"])
                p_to_nngp[r["P"]].append(r.get(f"{mode}_nngp_theory", np.nan))
            unique_p = sorted(p_to_emp.keys())
            mean_emp = [np.mean(p_to_emp[p]) for p in unique_p]
            mean_theo = [np.mean(p_to_theo[p]) for p in unique_p]
            mean_nngp = [np.mean(p_to_nngp[p]) for p in unique_p]
            draw_top_errorbar(plt, unique_p, mean_emp, yerr=[float(np.std(p_to_emp[p], ddof=1) / np.sqrt(len(p_to_emp[p]))) if len(p_to_emp[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
            plt.plot(unique_p, mean_theo, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
            plt.plot(unique_p, mean_nngp, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        # Dummy plot for legend (line-style legend; colors indicate d)
        plt.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
        plt.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
        plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0

        title_mode = "He1" if mode == "h1" else "He3"
        plt.title(rf"Learnability of {title_mode}$(v_* \cdot x)$"); plt.xlabel("P (dataset size)"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3); plt.xscale('log')
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{y_{He1}}$ (Learnability)")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_{get_d_str_for_filename()}_N{N}.png", dpi=300)
        plt.ylim(0, None)
    # --- Additional plots with alpha on x-axis (linear scale) ---
    # Eigenvalues with alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = get_data_label(val, i == 0)
        ax1.scatter(alpha_vals, [r["emp_h"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        ax2.scatter(alpha_vals, [r["emp_w0"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        alpha_to_nngp_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
            alpha_to_nngp_w[alpha].append(r.get("theo_w_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        mean_nngp_h = [np.mean([v for v in alpha_to_nngp_h[a] if np.isfinite(v)]) if any(np.isfinite(alpha_to_nngp_h[a])) else np.nan for a in unique_alpha]
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        mean_nngp_w = [np.mean([v for v in alpha_to_nngp_w[a] if np.isfinite(v)]) if any(np.isfinite(alpha_to_nngp_w[a])) else np.nan for a in unique_alpha]
        draw_top_errorbar(ax1, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax1.plot(unique_alpha, mean_theo_h, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        ax1.plot(unique_alpha, mean_nngp_h, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        draw_top_errorbar(ax2, unique_alpha, mean_emp_w, yerr=[float(np.std(alpha_to_emp_w[a], ddof=1) / np.sqrt(len(alpha_to_emp_w[a]))) if len(alpha_to_emp_w[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        ax2.plot(unique_alpha, mean_theo_w, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        ax2.plot(unique_alpha, mean_nngp_w, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        # ax1.set_ylim(0, None); ax2.set_ylim(0, None)
    # Dummy plot for legend
    ax1.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    ax1.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax1.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    ax2.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    ax2.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    ax2.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    ax1.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    ax1.set_title(r"$\lambda_H$ Eigenvalue"); ax2.set_title(r"$\lambda_W$ Eigenvalue")
    for ax in [ax1, ax2]: ax.legend(loc='lower right'); ax.grid(True, alpha=0.3); ax.set_xlabel(r"$\alpha$")
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Separate alpha plots
    # Lambda H alpha
    fig_h = plt.figure(figsize=(10,10))
    fig_h = plt.figure(figsize=(10,10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = get_data_label(val, i == 0)
        plt.yscale('log')
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_theo_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_theo_h[alpha].append(r["theo_h"])
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_theo_h = [np.mean(alpha_to_theo_h[a]) for a in unique_alpha]
        mean_nngp_h = [np.mean(alpha_to_nngp_h[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)

    plt.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        mean_nngp_h = [np.mean(alpha_to_nngp_h[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)

    plt.plot([], [], '-', color=color_empirical if single_d_mode else 'black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color=color_theory if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title("He1 Target Eigenvalues ($\lambda^{H, He1}_*$)"); plt.ylabel(r"$\lambda^{H, He1}_*$",fontsize=25)
    plt.xlabel(r"$\alpha$"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.title("He1 Target Eigenvalues ($\lambda^{H, He1}_*$)"); plt.ylabel(r"$\lambda^{H, He1}_*$",fontsize=25)
    plt.xlabel(r"$\alpha$"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)
    # He3 Lambda H (target) vs P
    fig_h3 = plt.figure(figsize=(15, 10))
    plt.yscale('log')
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"

        res = collapse_rows_to_seed_means(res)
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"

        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        p_vals = [r["P"] for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        if multi_d_mode:
            label = get_data_label(val, True)
            # if val==150:
                # continue
        if multi_d_mode:
            label = get_data_label(val, True)
            # if val==150:
                # continue
        # scatter of empirical He3 target eigenvalues
        plt.scatter(p_vals, [r["h3_target_eig"] for r in res],
                color=c, label=label , alpha=0.7, s=50, marker=exp_marker)
                color=c, label=label , alpha=0.7, s=50, marker=exp_marker)

        # mean lines over seeds
        p_to_emp_h3 = defaultdict(list)
        p_to_theo_h3 = defaultdict(list)
        p_to_nngp_h3 = defaultdict(list)
        p_to_nngp_h3 = defaultdict(list)
        for r in res:
            p_to_emp_h3[r["P"]].append(r["h3_target_eig"])
            p_to_theo_h3[r["P"]].append(r["theo_h3"])
            p_to_nngp_h3[r["P"]].append(r.get("theo_h3_nngp", np.nan))
            p_to_nngp_h3[r["P"]].append(r.get("theo_h3_nngp", np.nan))
        unique_p = sorted(p_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(p_to_emp_h3[p]) for p in unique_p]
        mean_theo_h3 = [np.mean(p_to_theo_h3[p]) for p in unique_p]
        mean_nngp_h3 = [np.mean(p_to_nngp_h3[p]) for p in unique_p]
        draw_top_errorbar(plt, unique_p, mean_emp_h3, yerr=[float(np.std(p_to_emp_h3[p], ddof=1) / np.sqrt(len(p_to_emp_h3[p]))) if len(p_to_emp_h3[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_h3, '--', color=role_color("theory", c), linewidth=3,
            marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h3, ':', color=role_color("nngp", c), linewidth=3,
            marker=theo_marker, markersize=6, alpha=0.7)
    plt.yscale('log')
    plt.ylim(0.0, None)
    # dummy handles for legend (line-style legend; colors indicate d)
    plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        mean_nngp_h3 = [np.mean(p_to_nngp_h3[p]) for p in unique_p]
        draw_top_errorbar(plt, unique_p, mean_emp_h3, yerr=[float(np.std(p_to_emp_h3[p], ddof=1) / np.sqrt(len(p_to_emp_h3[p]))) if len(p_to_emp_h3[p]) > 1 else 0.0 for p in unique_p], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_p, mean_theo_h3, '--', color=role_color("theory", c), linewidth=3,
            marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_p, mean_nngp_h3, ':', color=role_color("nngp", c), linewidth=3,
            marker=theo_marker, markersize=6, alpha=0.7)
    plt.yscale('log')
    plt.ylim(0.0, None)
    # dummy handles for legend (line-style legend; colors indicate d)
    plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(d, color='gray', linestyle='--', alpha=0.5, linewidth=2, label="P=d")

    plt.title(r"He3 Target Eigenvalues ($\lambda^{H,He3}_*$)")
    plt.title(r"He3 Target Eigenvalues ($\lambda^{H,He3}_*$)")
    plt.xlabel("P (dataset size)")
    plt.ylabel(r"$\lambda^{H,He3}_*$", fontsize=25)
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.ylabel(r"$\lambda^{H,He3}_*$", fontsize=25)
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"eigenvalues_He3_{get_d_str_for_filename()}_N{N}.png", dpi=300)
    plt.savefig(RESULTS_DIR / f"eigenvalues_He3_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # He3 Lambda H (target) vs alpha
    fig_h3_alpha = plt.figure(figsize=(10,10))
    plt.yscale('log')
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        # if val==150: continue
        res = collapse_rows_to_seed_means(res)
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        # if val==150: continue
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else f"{color_by}={val}"
        
        # scatter of empirical He3 target eigenvalues
        plt.scatter(alpha_vals, [r["h3_target_eig"] for r in res],
                color=c, label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)

        # mean lines over seeds
        alpha_to_emp_h3 = defaultdict(list)
        alpha_to_theo_h3 = defaultdict(list)
        alpha_to_nngp_h3 = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h3[alpha].append(r["h3_target_eig"])
            alpha_to_theo_h3[alpha].append(r["theo_h3"])
            alpha_to_nngp_h3[alpha].append(r.get("theo_h3_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h3.keys())
        mean_emp_h3 = [np.mean(alpha_to_emp_h3[a]) for a in unique_alpha]
        mean_theo_h3 = [np.mean(alpha_to_theo_h3[a]) for a in unique_alpha]
        mean_nngp_h3 = [np.mean(alpha_to_nngp_h3[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h3, yerr=[float(np.std(alpha_to_emp_h3[a], ddof=1) / np.sqrt(len(alpha_to_emp_h3[a]))) if len(alpha_to_emp_h3[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_h3, '--', color=role_color("theory", c), linewidth=3,
            marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_h3, ':', color=role_color("nngp", c), linewidth=3,
            marker=theo_marker, markersize=6, alpha=0.7)

    # dummy handles for legend (line-style legend; colors indicate d)
    plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")

    plt.title(r"He3 Target Eigenvalues ($\lambda^{H,He3}_*$)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\lambda^{H,He3}_*$", fontsize=25)
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"eigenvalues_He3_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Lambda W alpha
    fig_w = plt.figure(figsize=(10,10))
    plt.yscale('log')
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        label = r"$\chi = $" + str(val) if color_by == "chi" else get_data_label(val, multi_d_mode)
        plt.scatter(alpha_vals, [r["emp_w0"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_w = defaultdict(list)
        alpha_to_theo_w = defaultdict(list)
        alpha_to_nngp_w = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_w[alpha].append(r["emp_w0"])
            alpha_to_theo_w[alpha].append(r["theo_w"])
            alpha_to_nngp_w[alpha].append(r.get("theo_w_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_w.keys())
        mean_emp_w = [np.mean(alpha_to_emp_w[a]) for a in unique_alpha]
        mean_theo_w = [np.mean(alpha_to_theo_w[a]) for a in unique_alpha]
        mean_nngp_w = [np.mean([v for v in alpha_to_nngp_w[a] if np.isfinite(v)]) if any(np.isfinite(alpha_to_nngp_w[a])) else np.nan for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_w, yerr=[float(np.std(alpha_to_emp_w[a], ddof=1) / np.sqrt(len(alpha_to_emp_w[a]))) if len(alpha_to_emp_w[a]) > 1 else 0.0 for a in unique_alpha], color=role_color("empirical", c), marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_theo_w, '--', color=role_color("theory", c), linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
        plt.plot(unique_alpha, mean_nngp_w, ':', color=role_color("nngp", c), linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
    # Y minimum is 0
    plt.ylim(0, None)
    plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
    plt.plot([], [], ':', color=color_nngp if single_d_mode else 'black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
    plt.ylabel(r"$\lambda^\Sigma_* = v^T \Sigma_w v$", fontsize=25);
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"$\lambda^\Sigma_*$ Eigenvalue")
    plt.xlabel(r"$\alpha$"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_W_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # NNGP Eigenvalues with alpha
    # Lambda H (NNGP alpha)
    fig_nngp_h_alpha = plt.figure(figsize=(10,10))
    for i, (val, res) in enumerate(groups.items()):
        res = collapse_rows_to_seed_means(res)
        c = series_color(i)
        if color_by == "chi":
            label = r"$\chi = $" + str(val)
        elif color_by == "d":
            label = f"d={val}"
        else:
            label = f"{color_by}={val}"
        alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
        plt.scatter(alpha_vals, [r["emp_h"] for r in res], color=role_color("empirical", c), label=label if i == 0 else "", alpha=0.7, s=50, marker=exp_marker)
        # Mean lines
        alpha_to_emp_h = defaultdict(list)
        alpha_to_nngp_h = defaultdict(list)
        for r in res:
            alpha = np.log(r["P"]) / np.log(r["d"])
            alpha_to_emp_h[alpha].append(r["emp_h"])
            alpha_to_nngp_h[alpha].append(r.get("theo_h_nngp", np.nan))
        unique_alpha = sorted(alpha_to_emp_h.keys())
        mean_emp_h = [np.mean(alpha_to_emp_h[a]) for a in unique_alpha]
        mean_nngp_h = [np.mean(alpha_to_nngp_h[a]) for a in unique_alpha]
        draw_top_errorbar(plt, unique_alpha, mean_emp_h, yerr=[float(np.std(alpha_to_emp_h[a], ddof=1) / np.sqrt(len(alpha_to_emp_h[a]))) if len(alpha_to_emp_h[a]) > 1 else 0.0 for a in unique_alpha], color=c, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
        plt.plot(unique_alpha, mean_nngp_h, '--', color=c, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
    plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
    plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="NNGP")
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$")
    plt.title(r"NNGP Hidden Kernel Eigenvalues ($\lambda_H^{NNGP}$)")
    plt.xlabel(r"$\alpha$"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / f"eigenvalues_H_NNGP_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)

    # Learnability with alpha
    for mode in ["h1", "h3"]:
        plt.figure(figsize=(10,10))
        plt.yscale('log')
        for i, (val, res) in enumerate(groups.items()):
            res = collapse_rows_to_seed_means(res)
            res = collapse_rows_to_seed_means(res)
            c = series_color(i)
            alpha_vals = [np.log(r["P"]) / np.log(r["d"]) for r in res]
            if color_by == "chi":
                label = r"$\chi = $" + str(val)
            elif color_by == "d":
                label = f"d={val}"
            else:
                label = f"{color_by}={val}"
            if multi_d_mode:
                label = get_data_label(val, True)
            plt.scatter(alpha_vals, [r[f"{mode}_emp"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
            if color_by == "chi":
                label = r"$\chi = $" + str(val)
            elif color_by == "d":
                label = f"d={val}"
            else:
                label = f"{color_by}={val}"
            if multi_d_mode:
                label = get_data_label(val, True)
            plt.scatter(alpha_vals, [r[f"{mode}_emp"] for r in res], color=role_color("empirical", c), label=label if (i == 0 or multi_d_mode) else "", alpha=0.7, s=50, marker=exp_marker)
            plt.axvline(1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label=r"$\alpha=1$" if i == 0 else None)

            # Mean lines
            alpha_to_emp = defaultdict(list)
            alpha_to_theo = defaultdict(list)
            alpha_to_nngp = defaultdict(list)
            for r in res:
                alpha = np.log(r["P"]) / np.log(r["d"])
                alpha_to_emp[alpha].append(r[f"{mode}_emp"])
                alpha_to_theo[alpha].append(r[f"{mode}_theory"])
                alpha_to_nngp[alpha].append(r.get(f"{mode}_nngp_theory", np.nan))
            unique_alpha = sorted(alpha_to_emp.keys())
            mean_emp = [np.mean(alpha_to_emp[a]) for a in unique_alpha]
            mean_theo = [np.mean(alpha_to_theo[a]) for a in unique_alpha]
            mean_nngp = [np.mean(alpha_to_nngp[a]) for a in unique_alpha]
            draw_top_errorbar(plt, unique_alpha, mean_emp, yerr=[float(np.std(alpha_to_emp[a], ddof=1) / np.sqrt(len(alpha_to_emp[a]))) if len(alpha_to_emp[a]) > 1 else 0.0 for a in unique_alpha], color=c, marker=exp_marker, linestyle='-', linewidth=3, markersize=6, alpha=0.85, capsize=4, elinewidth=1.5, ecolor='black', zorder=1000, clip_on=False)
            plt.plot(unique_alpha, mean_theo, '--', color=c, linewidth=3, marker=theo_marker, markersize=8, alpha=0.8)
            plt.plot(unique_alpha, mean_nngp, ':', color=c, linewidth=3, marker=theo_marker, markersize=6, alpha=0.7)
        # Dummy plot for legend (line-style legend; colors indicate d)
        plt.plot([], [], '-', color='black', linewidth=3, marker=exp_marker, markersize=6, label="Model (empirical)")
        plt.plot([], [], '--', color='black', linewidth=3, marker=theo_marker, markersize=8, label="Theory (Mean-Field)")
        plt.plot([], [], ':', color='black', linewidth=3, marker=theo_marker, markersize=6, label="NNGP")
        d = res[0]["d"] if res else 0
        kappa = res[0]['kappa'] if res else 0
        kappa_eff = res[0]['kappa_eff'] if res else 0
        N = res[0]['N'] if res else 0
        title_mode = "He1" if mode == "h1" else "He3"
        plt.title(rf"Learnability of {title_mode}$(v_* \cdot x)$"); plt.xlabel(r"$\alpha$"); plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
        plt.ylabel(r"$\eta_{He1} = \frac{\langle f \mid He_1 \rangle}{y_{He1}}$ (Learnability)")
        plt.ylim(0, None)

        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"learnability_{mode}_alpha_linear_{get_d_str_for_filename()}_N{N}.png", dpi=300)
    
    # Plot eigenvalues vs chi if multiple chi values are present
    if len(unique_chis) > 1:
        fig_chi = plt.figure(figsize=(10, 8))
        
        d_val = final_data[0]["d"] if final_data else 0
        kappa_val = final_data[0]["kappa"] if final_data else 0
        N_val = final_data[0]["N"] if final_data else 0
        
        # Group data by P to support multi-P plots if present
        p_groups = defaultdict(list)
        for r in final_data:
            p_groups[r["P"]].append(r)
            
        for p_val, rows in sorted(p_groups.items()):
            # Group by chi
            chi_to_emp = defaultdict(list)
            chi_to_theo = defaultdict(list)
            for r in rows:
                if "chi" in r and "emp_h" in r and "theo_h" in r:
                    chi_to_emp[r["chi"]].append(r["emp_h"])
                    chi_to_theo[r["chi"]].append(r["theo_h"])
            
            chis = sorted(chi_to_emp.keys())
            mean_emp = [np.mean(chi_to_emp[c]) for c in chis]
            mean_theo = [np.mean(chi_to_theo[c]) for c in chis]
            
            p_suffix = f" (P={p_val})" if len(p_groups) > 1 else ""
            
            # Scatter individual experiment points
            all_chis = []
            all_emps = []
            for c in chis:
                for val in chi_to_emp[c]:
                    all_chis.append(c)
                    all_emps.append(val)
            plt.scatter(all_chis, all_emps, color=color_empirical, marker='o', alpha=0.4, s=30, zorder=2)
            
            # Experiment mean line: solid circles connected by solid line
            plt.plot(chis, mean_emp, linestyle='-', marker='o', color=color_empirical, 
                     linewidth=2.5, markersize=8, label=f"Experiment{p_suffix}", zorder=3)
            
            # Theory line: dashed line and squares (scatter)
            plt.plot(chis, mean_theo, linestyle='--', marker='s', color=color_theory, 
                     linewidth=2.5, markersize=8, label=f"Theory{p_suffix}", zorder=4)
            
        plt.xlabel("chi")
        plt.ylabel("lH")
        plt.title(f"d={d_val}, kappa={kappa_val}, N={N_val}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = RESULTS_DIR / f"eigenvalues_vs_chi_{get_d_str_for_filename()}_N{N_val}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig_chi)
        print(f"Saved eigenvalues vs chi plot to {plot_path}")

    # plt.show()

