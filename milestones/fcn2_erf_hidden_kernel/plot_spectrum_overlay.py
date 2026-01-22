import json
import tempfile
import subprocess
from typing import Optional
def j_random_QB_activation_generic_streaming(model, d, device, N, k=2000, p=10, chunk_size=4096):
    """Low-rank QB approximation for J kernel using h0 activations, streaming X in chunks."""
    with torch.no_grad():
        l = k + p
        dtype = torch.float32
        # Set a fixed seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        Omega = torch.randn((N, l), device=device, dtype=dtype)
        res = torch.zeros((N, l), device=device, dtype=dtype)
        h0_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # Use the same seed for each chunk to ensure reproducibility
            torch.manual_seed(seed + start)
            np.random.seed(seed + start)
            X_chunk = torch.randn(end - start, d, device=device, dtype=dtype)
            batch_h0 = model.h0_activation(X_chunk)
            h0_chunks.append(batch_h0)
        h0 = torch.cat(h0_chunks, dim=0)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)
        Q, _ = torch.linalg.qr(res)
        Z = torch.zeros((N, l), device=device, dtype=dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)
        return Q, Z, seed
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys

# Add lib path for FCN2NetworkActivationGeneric
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

# List of model directories
model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_0_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_3_eps_0.03']

def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    return d, P, N, chi

def load_model(model_dir, device):
    d, P, N, chi = parse_config_from_dirname(model_dir)
    model_path = Path(model_dir) / "model.pt"
    if not model_path.exists():
        model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        print(f"Model not found in {model_dir}")
        return None, None, None, None
    state_dict = torch.load(model_path, map_location=device)
    ens = state_dict['W0'].shape[0]
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1/d, 1/(N*chi)), device=device
    )
    model.to(torch.float32)

    model.load_state_dict(state_dict)
    model.eval()
    return model, d, P, N
from contextlib import contextmanager
import time
@contextmanager
def timed(msg: str, print_it: bool = True):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1e3   # seconds
    else:
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0

    if print_it:
        print(f"[{msg}] {elapsed:.4f}s")

def H_random_QB(self, X, k = 100, p = 25, verbose=False, chunk_size=4096):

    xtype = torch.float32

    # Returns a low rank QB decomposition of A 
    # using Halko et. al. 2011's random SVD algorithm
    with torch.no_grad():
        l = k + p
        h1 = self.h0_activation(X, optimize=False)                     # (N, ens, n1)
        if verbose:
            print("Computing H_random_QB on device: ", self.device)
        # ----- Random projections ------------------------------------------------
        with timed("Random Omega generation"):
            Omega = torch.randn((X.shape[0], l),
                                device=self.device,
                                dtype=xtype)

        res = torch.zeros((X.shape[0], l),
                        device=self.device,
                        dtype=xtype)

        # ----- Build `res` (the random-projection matrix) ----------------------
        chunk_size = min(chunk_size, X.shape[0])          # 2048 * 2; feel free to tune
        N = X.shape[0]

        with timed(f"res computation (chunks of {chunk_size})"):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch_h1 = h1[start:end]                     # (b, ens, n1)

                with timed(f"  res chunk [{start}:{end}]"):
                    # einsum: b q k,  N q k,  N l  -> b l
                    res[start:end] = torch.einsum(
                        'bqk,Nqk,Nl->bl',
                        batch_h1, h1, Omega
                    ) / (self.ens * self.n1)

        with timed("QR factorisation"):
            Q, _ = torch.linalg.qr(res)                     # (m, l)

        Z = torch.zeros((X.shape[0], l),
                        device=self.device,
                        dtype=xtype)

        # ----- Build `Z` (kernel projected onto Q) ------------------------------
        with timed(f"Z computation (chunks of {chunk_size})"):
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch_h1 = h1[start:end]

                with timed(f"  Z chunk [{start}:{end}]"):
                    # K_uv  : b x N
                    K_uv = torch.einsum(
                        'bqk,Nqk->bN',
                        batch_h1, h1
                    ) / (self.ens * self.n1)

                    # matmul: (k, m) @ (m, l) -> (k, l)
                    Z[start:end] = torch.matmul(K_uv, Q)

            return Q, Z

def compute_empirical_j_spectrum_streaming(model, d, device, p_large=10000, k=9000, p=25, chunk_size=4096):
    # Use the user's logic, but stream X/h0 in chunks
    model.to(device)
    model.device = device
    torch.manual_seed(42)
    seed = 42
    X = torch.randn((p_large,d), device=device, dtype=torch.float32)

    Q, Z = H_random_QB(model, X, k, p, chunk_size=chunk_size)

    # CRITICAL FIX 1: Check for NaN/Inf in Q and Z before SVD
    if torch.isnan(Q).any() or torch.isinf(Q).any():
        print(f"Warning: Q contains NaN or Inf values")
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
    
    if torch.isnan(Z).any() or torch.isinf(Z).any():
        print(f"Warning: Z contains NaN or Inf values")
        Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # CRITICAL FIX 2: Apply nan_to_num to Z.T, not just Z
    Z_T = Z.T
    Z_T = torch.nan_to_num(Z_T, nan=0.0, posinf=0.0, neginf=0.0)
    
    # OPTIONAL FIX 3: Add error handling for SVD
    try:
        Ut, _S, V = torch.linalg.svd(Z_T)
    except torch._C._LinAlgError as e:
        print(f"SVD failed with error: {e}")
        print(f"Z_T shape: {Z_T.shape}")
        print(f"Z_T contains NaN: {torch.isnan(Z_T).any()}")
        print(f"Z_T contains Inf: {torch.isinf(Z_T).any()}")
        print(f"Z_T min/max: {Z_T.min()}, {Z_T.max()}")
        # Try with CPU as fallback
        print("Attempting SVD on CPU...")
        Z_T_cpu = Z_T.cpu()
        Ut, _S, V = torch.linalg.svd(Z_T_cpu)
        Ut, _S, V = Ut.to(device), _S.to(device), V.to(device)
    
    # SVD on Z.T for spectrum
    # handle Nans in Z
    Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Ut, _S, V = torch.linalg.svd(Z.T)
    m, n = Z.shape[1], Z.shape[0]
    k_eff = min(m, n)
    Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
    Sigma[:k_eff, :k_eff] = torch.diag(_S[:k_eff])
    U = torch.matmul(Q, Ut)
    # Use the same seed for X as in QB
    torch.manual_seed(seed)
    np.random.seed(seed)
    Y1 = X[:,:]
    Y3 = (X[:,:]**3 - 3.0 * X[:,:])

    # Left eigenvalues for Y1 via J_eig
    Y1_norm = Y1 / torch.norm(Y1, dim=0)

    # Left eigenvalues for Y3 via projection through U, Sigma
    Y3_norm = Y3 / torch.norm(Y3, dim=0)
    left_eigenvalues_Y1 = (torch.matmul(Y1_norm.t(), U) @ torch.diag(_S[:k_eff]) @ torch.matmul(U.T, Y1_norm)).diagonal() / torch.norm(Y1_norm, dim=0)/ X.shape[0]
    
    left_eigenvaluesY3 = (torch.matmul(Y3_norm.t(), U) @ torch.diag(_S[:k_eff]) @ torch.matmul(U.T, Y3_norm)).diagonal() / torch.norm(Y3_norm, dim=0)/ X.shape[0]
    # Extract target (first) and perpendicular (rest) eigenvalues
    lJ1T = float(left_eigenvalues_Y1[0].cpu().numpy())
    lJ1P = float(left_eigenvalues_Y1[1].cpu().numpy()) 
    lJ3T = float(left_eigenvaluesY3[0].cpu().numpy()) 
    lJ3P = float(left_eigenvaluesY3[1:].mean().cpu().numpy())

    # Return summary and all eigenvalues (sorted descending)
    all_eigvals = np.sort(_S.detach().cpu().numpy())[::-1]
    return {
        "summary": np.array([lJ1T, lJ1P, lJ3T, lJ3P]),
        "all_eigenvalues": all_eigvals/ X.shape[0],
    }
 # Import theory function from compute_h3_projections if available
def compute_theory_with_julia(d: int, n1: int, P: int, chi: float, kappa: float, epsilon: Optional[float] = None):
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_cubic_eigs.jl"
    if epsilon is None:
        epsilon = 0.03
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "julia",
            str(julia_script),
            "--d", str(d),
            "--n1", str(n1),
            "--P", str(P),
            "--chi", str(chi),
            "--kappa", str(kappa),
            "--epsilon", str(epsilon),
            "--to", str(tmp_path),
            "--quiet",
        ]
        print("Running Julia command:", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp_path, "r") as f:
            result = json.load(f)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
def main():
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Output directory for saving plot and config
    output_dir = Path(__file__).parent / "spectrum_overlay_output"
    output_dir.mkdir(exist_ok=True)

    # Fig path should be named based on the P in the model_dirs list


    config_path = output_dir / "spectrum_overlay_config.txt"

    plt.figure(figsize=(10, 6))
    colors = ['C0', 'C1', 'C2', 'C3']
    used_model_dirs = []
    # Cache for theory results by unique (d, P, N, chi, kappa, epsilon)
    theory_cache = {}
    Ps = []

    for i, model_dir in enumerate(model_dirs):
        print(f"Processing: {model_dir}")
        model, d, P, N = load_model(model_dir, device)
        if model is None:
            continue
        result = compute_empirical_j_spectrum_streaming(model, d, device, p_large=35000, k=9000, p=25, chunk_size=500)
        eigvals_summary = result["summary"]
        all_eigenvalues = result["all_eigenvalues"]
        # Scatter plot (as before)
        plt.scatter(np.arange(len(eigvals_summary)), np.sort(eigvals_summary)[::-1], label=f'Empirical Summary Seed {i}', color=colors[i % len(colors)], marker='o', s=40, alpha=0.7)
        # Bar plot of all eigenvalues (sorted descending)
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(all_eigenvalues)), all_eigenvalues, color=colors[i % len(colors)], alpha=0.5)
        # Overlay theory lines if available
        parts = Path(model_dir).name.split('_')
        epsilon = None
        for p in parts:
            if p.startswith('eps'):
                try:
                    epsilon = float(p[4:])
                except Exception:
                    pass
        d_th, P_th, N_th, chi_th = parse_config_from_dirname(model_dir)
        kappa_th = 8.2
        cache_key = (d_th, P_th, N_th, chi_th, kappa_th, epsilon)
        if cache_key not in theory_cache:
            if kappa_th is None:
                print(f"No kappa specified for theory for {model_dir}, skipping theory overlay.")
                theory = None
            else:
                theory = compute_theory_with_julia(d_th, N_th, P_th, chi_th, kappa_th, epsilon)
            theory_cache[cache_key] = theory
        else:
            theory = theory_cache[cache_key]
        if theory is not None:
            lJ1T = theory["target"]["lJ1T"]
            lJ1P = theory["perpendicular"]["lJ1P"]
            lJ3T = theory["target"]["lJ3T"]
            lJ3P = theory["perpendicular"]["lJ3P"]
            plt.axhline(lJ1T, linestyle='--', color='k', alpha=0.7, label='Theory $\\lambda_{{He1}}^T$')
            plt.axhline(lJ1P, linestyle='--', color='k', alpha=0.4, label='Theory $\\lambda_{{He1}}^P$')
            plt.axhline(lJ3T, linestyle='--', color='r', alpha=0.7, label='Theory $\\lambda_{{He3}}^T$')
            plt.axhline(lJ3P, linestyle='--', color='r', alpha=0.4, label='Theory $\\lambda_{{He3}}^P$')
        plt.xlabel('Eigenvalue index (sorted)')
        plt.ylabel('Eigenvalue')
        plt.title(f'All Empirical Eigenvalues (Seed {i})')
        plt.legend()
        plt.tight_layout()
        plt.show()
        used_model_dirs.append(model_dir)
        Ps.append(P)

    fig_path = output_dir / f"spectrum_overlay_P{Ps}.png"
    config_path = output_dir / f"spectrum_overlay_config_P{Ps}.txt"
    plt.xlabel('Eigenvalue index (sorted)')
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Eigenvalue')
    plt.title('Empirical J Kernel Spectrum for FCN2 Networks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved spectrum plot to {fig_path}")

    # Save config file with model directories
    with open(config_path, 'w') as f:
        f.write("# Model directories used for spectrum overlay plot\n")
        for model_dir in used_model_dirs:
            f.write(model_dir + "\n")
    print(f"Saved config file to {config_path}")

    plt.show()

if __name__ == "__main__":
    main()
