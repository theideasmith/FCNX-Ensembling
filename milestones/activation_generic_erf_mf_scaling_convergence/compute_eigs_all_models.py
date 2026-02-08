import subprocess
import tempfile
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import normaltest, shapiro, kstest, norm

# --- Configuration & Paths ---
LIB_PATH = Path(__file__).parent.parent.parent / "lib"
JULIA_SCRIPT = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
RESULTS_DIR = Path(__file__).parent / "p_scan_erf_results"
sys.path.insert(0, str(LIB_PATH))

TARGET_CHI = 80.0
MAX_WORKERS = 4
EPSILON = 1e-3

# --- 1. Theory Backend (Julia) ---

class JuliaSolver:
    @staticmethod
    def query(d, P, n1, n2, chi, kappa, eps):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            to_path = Path(tf.name)
        cmd = [
            "julia", str(JULIA_SCRIPT), 
            f"--d={d}", f"--P={P}", f"--n1={n1}", f"--n2={n2}",
            f"--chi={chi}", f"--kappa={kappa}", f"--epsilon={eps}", 
            f"--to={to_path}", "--quiet"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            with open(to_path, "r") as f:
                data = json.load(f)
            return data.get("target", {})
        except Exception as e:
            print(f"Julia solver error [P={P}]: {e}")
            return {}
        finally:
            if to_path.exists(): to_path.unlink()

# --- 2. Model Utilities ---

def load_model(model_path, device="cpu"):
    from FCN3Network import FCN3NetworkActivationGeneric
    sd = torch.load(model_path, map_location=device)
    d, n1, n2 = sd['W0'].shape[-1], sd['W0'].shape[-2], sd['W1'].shape[-2]
    P = sd['A'].shape[-1] if sd['A'].ndim > 1 else 1
    ens = sd['W0'].shape[0] if sd['W0'].ndim == 3 else sd['W0'].shape[1]
    
    model = FCN3NetworkActivationGeneric(
        d=d, n1=n1, n2=n2, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1.0/d, 1.0/n1, 1.0/(n1*n2))
    ).to(device)
    
    cleaned_sd = {k: v.squeeze(0) if v.ndim > (3 if 'W' in k else 2) else v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd, strict=False)
    return model.eval()

# --- 3. Analysis Tasks ---

def run_gaussianity_check(model_dir):
    """Performs statistical tests on the W0 weight distribution."""
    config_path = model_dir / "config.json"
    model_path = model_dir / "model_final.pt"
    
    if not (config_path.exists() and model_path.exists()):
        return None

    model = load_model(model_path)
    # Extract weights along the target direction
    W0 = model.W0.detach().cpu().numpy()
    W0_target = W0[..., 0].reshape(-1)

    # Statistical Tests
    _, p_norm = normaltest(W0_target)
    _, p_shap = shapiro(W0_target[:5000]) # Shapiro limit
    _, p_ks   = kstest(W0_target, 'norm', args=(np.mean(W0_target), np.std(W0_target)))

    # Visualization
    plt.figure(figsize=(6, 4))
    plt.hist(W0_target, bins=50, density=True, alpha=0.6, color='skyblue')
    x = np.linspace(np.min(W0_target), np.max(W0_target), 200)
    plt.plot(x, norm.pdf(x, np.mean(W0_target), np.std(W0_target)), 'r--', lw=2)
    plt.title(f"Gaussianity: {model_dir.name}\n(Norm P: {p_norm:.2e})")
    plt.savefig(model_dir / "W0_target_gaussianity.png")
    plt.close()
    config = json.load(open(config_path))
    return {
        "model": model_dir.name,
        "p_normal": float(p_norm),
        "p_shapiro": float(p_shap),
        "p_ks": float(p_ks),
        "mean": float(np.mean(W0_target)),
        "std": float(np.std(W0_target)),
        "d": int(model.W0.shape[-1]),
        "P": int(config.get("P", 1))
    }

def process_eigenvalues(model_dir):
    """Calculates empirical and theoretical eigenvalues and learnabilities."""
    cfg_path, pt_path = model_dir / "config.json", model_dir / "model_final.pt"

    # IF pt path doesnt exist, try model.pt (for older runs)
    if not pt_path.exists():
        pt_path = model_dir / "model.pt"
        if not pt_path.exists():
            print(f"No model file found in {model_dir.name}")
            return None
    with open(cfg_path) as f:
        cfg = json.load(f)

    if float(cfg.get("chi", 0)) != TARGET_CHI:
        return None

    d, P, N, kappa = cfg["d"], cfg["P"], cfg["N"], cfg["kappa"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(pt_path, device)
        torch.manual_seed(0)
        X = torch.randn(6000, d, device=device)
        
        # Hessian & Weight Covariance
        h_eig = model.H_eig(X, X)
        emp_h = float(h_eig[0].detach().cpu().numpy())
        
        w0_flat = model.W0.detach().cpu().numpy().reshape(-1, d)
        emp_w0 = float(np.max(np.linalg.eigvalsh((w0_flat.T @ w0_flat) / w0_flat.shape[0])))
        
        # Learnability Computation (H1 and H3 projections)
        P_total = 1000000
        batch_size = 10000
        num_batches = P_total // batch_size
        
        h1_sum = 0.0
        h3_sum = 0.0
        x0_norm_sum = 0.0
        
        with torch.no_grad():
            for _ in range(num_batches):
                X_batch = torch.randn(batch_size, d, device=device)
                out = model(X_batch)  # Shape: (batch_size, ens) or (batch_size,)
                x0 = X_batch[:, 0]
                x0_norm_sum += x0.norm()
                
                # H1 projection: <out, x0>
                if out.ndim == 2:  # ensemble case
                    h1_sum += torch.einsum('be,b->e', out, x0).mean().item()
                else:  # single model
                    h1_sum += (out * x0).sum().item()
                
                # H3 projection: <out, x0^3 - 3*x0>
                h3_component = x0**3 - 3*x0
                if out.ndim == 2:
                    h3_sum += torch.einsum('be,b->e', out, h3_component).mean().item()
                else:
                    h3_sum += (out * h3_component).sum().item()
        
        emp_h1 = h1_sum / P_total
        emp_h3 = h3_sum / x0_norm_sum
        
    except Exception as e:
        print(f"Error in {model_dir.name}: {e}")
        return None

    theory = JuliaSolver.query(d, P, N, N, TARGET_CHI, kappa, EPSILON)
    
    return {
        "P": P, "emp_h": emp_h, "emp_w0": emp_w0,
        "theo_h": float(theory.get("lH1T", np.nan)),
        "theo_w": float(theory.get("lWT", np.nan)),
        "h1_emp": emp_h1,
        "h1_theory": float(theory.get("mu1", np.nan)),
        "h3_emp": emp_h3,
        "h3_theory": float(theory.get("mu3", np.nan)),
        "kappa": kappa  # Store kappa for annotation
    }

# --- 4. Main Pipeline ---

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    model_dirs = sorted(list(RESULTS_DIR.glob("d*/*seed*")))
    
    # Filter by directories which contain both config and model files
    model_dirs = [d for d in model_dirs if (d / "config.json").exists() and ((d / "model_final.pt").exists() or (d/"model.pt").exists())]

    # Filter model directories by chi value in config
    valid_model_dirs = []
    for m_dir in model_dirs:
        try:
            cfg = json.load(open(m_dir / "config.json"))
            if float(cfg.get("chi", 0)) == TARGET_CHI:
                valid_model_dirs.append(m_dir)
        except Exception as e:
            print(f"Skipping {m_dir.name} due to config error: {e}")
    model_dirs = valid_model_dirs

    # Step A: Run Eigenvalue Scan in Parallel
    print(f"--- Running Eigenvalue Scan ({len(model_dirs)} models) ---")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        eig_results = [r for r in executor.map(process_eigenvalues, model_dirs) if r]

    # Step B: Run Gaussianity Tests (Post-processing)
    print("--- Running Gaussianity Tests ---")
    gauss_summary = []
    for m_dir in model_dirs:
        res = run_gaussianity_check(m_dir)
        if res: gauss_summary.append(res)
    
    with open(RESULTS_DIR / "gaussianity_summary.json", "w") as f:
        json.dump(gauss_summary, f, indent=2)

    # Step C: Plotting
    if eig_results:
        eig_results.sort(key=lambda x: x["P"])
        ps = [r["P"] for r in eig_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(ps, [r["emp_h"] for r in eig_results], 'o-', label="Empirical")
        ax1.plot(ps, [r["theo_h"] for r in eig_results], 's--', label="Theory")
        ax1.set_title("H Leading Eig"); ax1.legend()

        ax2.plot(ps, [r["emp_w0"] for r in eig_results], 'o-', label="Empirical")
        ax2.plot(ps, [r["theo_w"] for r in eig_results], 's--', label="Theory")
        ax2.set_title("W0 Covariance Top Eig"); ax2.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "final_analysis.png")

        # --- Learnability Plot ---
        h1_emp = [r.get("h1_emp", float('nan')) for r in eig_results]
        h1_theory = [r.get("h1_theory", float('nan')) for r in eig_results]
        plt.figure(figsize=(8,6))
        plt.plot(ps, h1_emp, 'o-', label="Empirical Learnability (h1_emp)")
        plt.plot(ps, h1_theory, 's--', label="Theory Learnability (mu1)")
        
        # Add annotation for P=1200 with its kappa value
        for i, r in enumerate(eig_results):
            if r["P"] == 1200:
                kappa_val = r.get("kappa", "unknown")
                plt.annotate(f'P=1200\nκ={kappa_val}', 
                           xy=(r["P"], h1_emp[i]), 
                           xytext=(10, 20), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                break
        
        plt.xlabel("P")
        plt.ylabel("Learnability")
        plt.title("Empirical vs Theory Learnability vs P")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "learnability_vs_P.png")
        plt.show()
        print("Done. Results saved to", RESULTS_DIR)