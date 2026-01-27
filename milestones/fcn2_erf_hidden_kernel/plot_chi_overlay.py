import torch
import numpy as np
from pathlib import Path
import sys
import argparse
import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import os
from collections import defaultdict
import subprocess
import json
from scipy.stats import norm

MODELDIR = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults'
THEORY_CACHE_FILE = os.path.join(MODELDIR, 'theory_cache.json')


def j_random_QB_activation_generic(model, X, k=2000, p=10):
    """Low-rank QB approximation for J kernel using h0 activations."""
    with torch.no_grad():
        l = k + p
        h0 = model.h0_activation(X)  # (N, ens, n1)
        Omega = torch.randn((X.shape[0], l), device=model.device, dtype=h0.dtype)

        res = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        chunk_size = 4096
        N = X.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            res[start:end] = torch.einsum('bqk,Nqk,Nl->bl', batch_h0, h0, Omega) / (model.ens * model.n1)

        Q, _ = torch.linalg.qr(res)

        Z = torch.zeros((X.shape[0], l), device=model.device, dtype=h0.dtype)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_h0 = h0[start:end]
            K_uv = torch.einsum('bqk,Nqk->bN', batch_h0, h0) / (model.ens * model.n1)
            Z[start:end] = torch.matmul(K_uv, Q)

        return Q, Z


def compute_empirical_j_eigenvalues(model, d, device, p_large=10_000):
    """Compute empirical J kernel eigenvalues (lJ1T, lJ1P, lJ3T, lJ3P)."""
    with torch.no_grad():
        model.to(device)
        model.device = device

        X = torch.randn(p_large, d, device=device)
        Y1 = X
        Y3 = (X ** 3 - 3.0 * X) / 6.0**0.5

        # Low-rank approximation QB for J kernel
        Q, Z = j_random_QB_activation_generic(model, X, k=9000, p=10)
        Ut, _S, V = torch.linalg.svd(Z.T)
        m, n = Z.shape[1], Z.shape[0]
        k_eff = min(m, n)
        Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
        Sigma[:k_eff, :k_eff] = torch.diag(_S[:k_eff])
        U = torch.matmul(Q, Ut)

        # Left eigenvalues for Y1 via J_eig
        Y1_norm = Y1 / torch.norm(Y1, dim=0)
        left_eigenvaluesY1 = model.J_eig(X, Y1_norm)

        # Left eigenvalues for Y3 via projection through U, Sigma
        Y3_norm = Y3 / torch.norm(Y3, dim=0)
        proj = (Y3_norm.t() @ U)
        left_Y3_mat = proj @ torch.diag(_S[:k_eff]) @ (U.t() @ Y3_norm)
        left_eigenvaluesY3 = left_Y3_mat.diagonal() / torch.norm(Y3_norm, dim=0) / X.shape[0]

        # Extract target (first) and perpendicular (rest) eigenvalues
        lJ1T = float(left_eigenvaluesY1[0].cpu().numpy())
        lJ1P = float(left_eigenvaluesY1[1].cpu().numpy()) if len(left_eigenvaluesY1) > 1 else lJ1T
        
        lJ3T = float(left_eigenvaluesY3[d].cpu().numpy()) if len(left_eigenvaluesY3) > d else float('nan')
        lJ3P = float(left_eigenvaluesY3[d + 1].cpu().numpy()) if len(left_eigenvaluesY3) > d + 1 else lJ3T

        return lJ1T, lJ1P, lJ3T, lJ3P


def get_self_consistent_kappa(model_dir, d, P, N, chi):
    """Call estimate_kappa_from_empirical.py script to get self-consistent kappa."""
    script_path = Path(__file__).parent / "estimate_kappa_from_empirical.py"
    
    if not script_path.exists():
        print(f"Warning: estimate_kappa_from_empirical.py not found at {script_path}, using original kappa")
        return None
    
    try:
        cmd = [
            "python",
            str(script_path),
            "--model-dir", str(model_dir),
            "--dataset-size", "10000",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=600, text=True)
        
        # Parse output to find "Self-consistent kappa = X.XXXXXX"
        for line in result.stdout.split('\n'):
            if 'Self-consistent kappa' in line and '=' in line:
                # Extract the kappa value
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        kappa = float(parts[-1].strip())
                        print(f"  → Self-consistent kappa: {kappa:.6f}")
                        return kappa
                    except ValueError:
                        pass
        
        print("  Warning: Could not parse kappa from script output")
        print(f"  Output: {result.stdout[-500:]}")  # Print last 500 chars of output
        return None
    except subprocess.TimeoutExpired:
        print("  Warning: estimate_kappa_from_empirical.py timed out")
        return None
    except Exception as e:
        print(f"  Warning: Failed to call estimate_kappa_from_empirical.py: {e}")
        return None

def parse_config_from_dirname(dirname):
    """Parse configuration from directory name"""
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    
    # Extract T (temperature) from filename to compute kappa
    T = None
    for i, part in enumerate(parts):
        if part == 'T' and i + 1 < len(parts):
            T = float(parts[i + 1])
            break
    
    # kappa = T / 2
    kappa = T / 2.0 if T is not None else None
    print("Parsed config from {}: d={}, P={}, N={}, chi={}, T={}, kappa={}".format(
        dirname, d, P, N, chi, T, kappa))
    return d, P, N, chi, T, kappa

def load_model(model_dir, device):
    """Load model from directory"""
    try:
        d, P, N, chi, T, kappa = parse_config_from_dirname(model_dir)
    except (IndexError, ValueError):
        return None, None, None, None, None, None, None
    
    model_path = Path(model_dir) / "model.pt"
    # if N > 1000:
        # return None, None, None, None, None, None, None
    if not model_path.exists():
        model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        print(f"Model not found in {model_dir}")
        return None, None, None, None, None, None, None

    try:
        state_dict = torch.load(model_path, map_location=device)
        ens = state_dict['W0'].shape[0]
        model = FCN2NetworkActivationGeneric(
            d=d, n1=N, P=P, ens=ens, activation="erf",
            weight_initialization_variance=(1/d, 1/(N*chi)), device=device
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model, d, P, N, chi, T, kappa
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        return None, None, None, None, None, None, None

def compute_neg_log_prob_weights(model):
    """Compute -log P for target and perpendicular weights"""
    W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
    
    # Target weights: W[:, :, 0]
    target_weights = W0[:, :, 0].flatten()
    
    # Perpendicular weights: W[:, :, 1:]
    if W0.shape[2] > 1:
        perp_weights = W0[:, :, 1:].flatten()
    else:
        perp_weights = np.array([])
    
    def compute_neg_log_probs(data, bins=50):
        """Compute -log P(w) values from histogram where P is probability density"""
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = hist > 0
        # hist is already probability density (from density=True), so take -log directly
        return bin_centers[mask], -np.log(hist[mask])
    
    target_centers, target_neg_log_probs = compute_neg_log_probs(target_weights)
    
    if len(perp_weights) > 0:
        perp_centers, perp_neg_log_probs = compute_neg_log_probs(perp_weights)
    else:
        perp_centers, perp_neg_log_probs = np.array([]), np.array([])
    
    # Compute variances for empirical distributions
    target_var = np.var(target_weights)
    perp_var = np.var(perp_weights) if len(perp_weights) > 0 else None
    
    return target_centers, target_neg_log_probs, perp_centers, perp_neg_log_probs, target_var, perp_var

def compute_theory_weights(d, P, N, chi, kappa):
    """Compute theoretical weight variances using Julia"""
    julia_lib_path = Path(__file__).parent.parent.parent / "julia_lib"
    script_path = julia_lib_path / "compute_fcn2_erf_eigs.jl"
    
    # Create temporary output file
    output_file = f"/tmp/fcn2_erf_eigs_{d}_{P}_{N}_{chi}_{kappa}.json"
    
    try:
        # Call Julia script with kappa parameter
        cmd = [
            "julia", str(script_path),
            "--d", str(d),
            "--n1", str(N),
            "--P", str(P),
            "--chi", str(chi),
            "--kappa", str(kappa),
            "--output", output_file
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        
        # Read results
        with open(output_file, 'r') as f:
            result = json.load(f)
        
        # Extract variances
        lWT = result.get('lWT', None)
        lWTP = result.get('lWTP', None)
        
        return lWT, lWTP
    except Exception as e:
        print(f"Error computing theory for d={d}, P={P}, N={N}, chi={chi}: {e}")
        return None, None

def load_theory_cache(cache_file):
    """Load theory data from cache file"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"Loaded theory cache from {cache_file} with {len(cache)} entries")
            # Convert string keys back to tuples
            theory_data = {}
            for key_str, value in cache.items():
                parts = key_str.split(',')
                # d and P are ints, chi is float
                key = (int(float(parts[0])), int(float(parts[1])), float(parts[2]))
                theory_data[key] = value
            return theory_data
        except Exception as e:
            print(f"Error loading theory cache: {e}")
            return {}
    return {}

def save_theory_cache(theory_data, cache_file):
    """Save theory data to cache file"""
    try:
        # Convert tuple keys to strings for JSON serialization
        cache = {}
        for (d, P, chi), value in theory_data.items():
            key_str = f"{d},{P},{chi}"
            cache[key_str] = value
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Saved theory cache to {cache_file} with {len(cache)} entries")
    except Exception as e:
        print(f"Error saving theory cache: {e}")

def main(recompute_theory=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Configure matplotlib for publication-quality figures
    plt.rcParams['text.usetex'] = False  # Use mathtext instead of LaTeX
    plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for nice math
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    # Load theory cache if it exists and not recomputing
    if recompute_theory:
        theory_data = {}
        print("Recomputing theory (cache will be overwritten)")
    else:
        theory_data = load_theory_cache(THEORY_CACHE_FILE)
    
    # Load all model directories - both sets
    # Model set A: from MiniGrokkingMFScalingResults (chi=800)
    model_dirs_A = [os.path.join(MODELDIR, d) for d in os.listdir(MODELDIR)]
    model_dirs_A = [d for d in model_dirs_A if os.path.isdir(d)]
    model_dirs_A = model_dirs_A[:1]
    # Model set B: chi=80 models
    model_dirs_B = [
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_0',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_1',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_2',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_3'
    ]


    model_dirs_B = [
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_1_eps_0.0',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.0'
    ]

    model_dirs_C = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_0',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_1',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_2',
        '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_3']

    model_dirs_C = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P7000_N600_chi_60.0_lr_3e-05_T_8.0_seed_0_eps_0.0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/DivergingLossDuetoLRChange/d50_P7000_N600_chi_60.0_lr_3e-05_T_8.0_seed_1_eps_0.0',  
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/DivergingLossDuetoLRChange/d50_P7000_N600_chi_60.0_lr_3e-05_T_8.0_seed_2_eps_0.0']

    model_dirs_C += ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/DivergingLossDuetoLRChange/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_0_eps_0.0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/DivergingLossDuetoLRChange/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_1_eps_0.0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/DivergingLossDuetoLRChange/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_2_eps_0.0']
    model_dirs_C += ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d20_P100_N400_chi_400.0_lr_1e-05_T_1.0_seed_42_eps_0.03']
    # Combine both sets
    model_dirs = model_dirs_A + model_dirs_B + model_dirs_C
    

    model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_0_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_3_eps_0.03']
    model_dirs = model_dirs_A
    # Parse and organize by (P, chi)
    model_data = defaultdict(list)
    theory_data = {}  # Store theoretical variances by (d, P, chi)
    all_P_values = set()
    all_chi_values = set()
    
    for model_dir in model_dirs:
        print(model_dir)
        model, d, P, N, chi, T, kappa = load_model(model_dir, device)
        if model is None:
            continue
        
        target_centers, target_neg_log_probs, perp_centers, perp_neg_log_probs, target_var, perp_var = compute_neg_log_prob_weights(model)
        
        # Compute empirical J eigenvalues and self-consistent kappa
        print(f"  Computing self-consistent kappa via empirical eigenvalues...")
        # kappa_sc = get_self_consistent_kappa(model_dir, d, P, N, chi)
        kappa_sc = kappa
        if kappa_sc is not None:
            kappa_corrected = kappa_sc
            print(f"  Using self-consistent kappa={kappa_corrected:.6f}")
        else:
            kappa_corrected = kappa
            print(f"  Self-consistent solver failed, using original kappa={kappa_corrected:.6f}")
        
        model_data[(P, chi)].append({
            'model_dir': model_dir,
            'd': d,
            'N': N,
            'T': T,
            'kappa': kappa,
            'kappa_corrected': kappa_corrected,
            'target_centers': target_centers,
            'target_neg_log_probs': target_neg_log_probs,
            'target_var': target_var,
            'perp_centers': perp_centers,
            'perp_neg_log_probs': perp_neg_log_probs,
            'perp_var': perp_var
        })
        
        all_P_values.add(P)
        all_chi_values.add(chi)
        
        # Compute theoretical variances if not already done
        if (d, P, chi) not in theory_data:
            lWT, lWTP = compute_theory_weights(d, P, N, chi, kappa_corrected)
            theory_data[(d, P, chi)] = {'lWT': lWT, 'lWTP': lWTP}
            if lWT is not None:
                print(f"Theory for d={d}, P={P}, chi={chi}: " + "$\\lambda_W^*$="+f'{lWT:.6f}, '+"$\\lambda_W^\\perp$"+f'={lWTP:.6f}')
    all_P_values = sorted(list(all_P_values))
    all_chi_values = sorted(list(all_chi_values))
    
    print(f"Found P values: {all_P_values}")
    print(f"Found chi values: {all_chi_values}")
    print(f"Model data keys: {list(model_data.keys())}")
    
    # Save theory cache
    save_theory_cache(theory_data, THEORY_CACHE_FILE)
    
    # Generate colors dynamically for all unique parameter combinations
    # Sort keys to ensure consistent coloring
    sorted_keys = sorted(model_data.keys())
    
    # Use a colormap with enough distinct colors
    n_unique = len(sorted_keys)
    if n_unique <= 10:
        cmap = cm.get_cmap('tab10')
        param_colors = {key: cmap(i / max(n_unique - 1, 1)) for i, key in enumerate(sorted_keys)}
    else:
        cmap = cm.get_cmap('hsv')
        param_colors = {key: cmap(i / n_unique) for i, key in enumerate(sorted_keys)}
    
    # Convert to hex for easier handling
    param_colors_hex = {key: plt.matplotlib.colors.rgb2hex(param_colors[key]) for key in param_colors}
    
    print(f"\nParameter combinations and their colors:")
    for key, color in param_colors_hex.items():
        print(f"  {key}: {color}")
    
    # Create overlay plots with professional sizing
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0))
    fig.suptitle('Action for Readin Layer Weights \n'+ r'$S[w] = -\log P(w\cdot v_k)$ for FCN2 $f(x) = \mathbf{a}\cdot \mathrm{erf}(\mathbf{W}\cdot \mathbf{x}),\; x\in \mathbb{R}^d,\; d=100$', fontsize=14, fontweight='normal', y=1.00)
    
    # Use dynamically generated colors
    chi_colors = param_colors_hex
    
    # Plot target weights (W[:, :, 0])
    ax_target = axes[0]
    
    for (P, chi), data_list in sorted(model_data.items()):
        color = param_colors_hex.get((P, chi), '#000000')  # default to black if not in dict
        linestyle = '-' if chi < data_list[0]['N'] else '--'
        
        # Get N from first data element
        N = data_list[0]['N']
        
        # Count number of seeds
        num_seeds = len(data_list)
        
        # Compute average variance across ensemble members
        target_vars = [data['target_var'] for data in data_list]
        avg_target_var = np.mean(target_vars)
        
        # Average over ensemble members for this (P, chi) pair
        all_centers = []
        all_probs = []
        for data in data_list:
            all_centers.append(data['target_centers'])
            all_probs.append(data['target_neg_log_probs'])
        
        # Combine and sort for smoother plotting
        combined_centers = np.concatenate(all_centers)
        combined_probs = np.concatenate(all_probs)
        
        # Sort by centers for plotting
        sort_idx = np.argsort(combined_centers)
        combined_centers = combined_centers[sort_idx]
        combined_probs = combined_probs[sort_idx]
        
        # Get kappa from first data element
        kappa = data_list[0]['kappa']
        kappa_corrected = data_list[0]['kappa_corrected']
        label = f'P={P}, $\chi={chi}$, $\kappa={kappa_corrected:.2f}$\nN={N}, σ²={avg_target_var:.4f}'
        if chi < N:
            label += f'\n{num_seeds} seeds (Grokking)'
        elif chi != N:
            label += f'\n{num_seeds} seeds (GFL)'
        else:
            label += '\n(Grokking)'
        ax_target.plot(combined_centers, combined_probs, color=color, linestyle=linestyle, 
                      linewidth=2, label=label, alpha=0.8)
        
        # Overlay theoretical Gaussian if available
        d = data_list[0]['d']
        theory_key = (d, P, chi)
        if theory_key in theory_data and theory_data[theory_key]['lWT'] is not None:
            lWT = theory_data[theory_key]['lWT']
            sigma = np.sqrt(lWT)
            x_range = np.linspace(-3*sigma, 3*sigma, 200)
            # Gaussian: p(w) = 1/(sigma*sqrt(2*pi)) * exp(-w^2/(2*sigma^2))
            # -log(p(w)) = 0.5*log(2*pi*var) + w^2/(2*var) = 0.5*log(2*pi) + log(sigma) + w^2/(2*sigma^2)
            gaussian_neg_log_p = 0.5*np.log(2*np.pi) + np.log(sigma) + x_range**2 / (2*sigma**2)
            ax_target.plot(x_range, gaussian_neg_log_p, color=color, linestyle=':', 
                          linewidth=1.5, alpha=0.6, label=f'$\lambda_W^*={lWT:.4f}$')

    
    ax_target.set_xlabel(r'$w\cdot w^*$', fontsize=12)
    ax_target.set_ylabel(r'$-\log P(w)$', fontsize=12)
    ax_target.set_title(r'Target Weights: $W_{:,:,0}$', fontsize=12, fontweight='normal', pad=10)
    ax_target.grid(True, alpha=0.3)
    
    # Do not create individual legend yet - will create shared legend below
    
    # Plot perpendicular weights (W[:, :, 1:])
    ax_perp = axes[1]
    
    for (P, chi), data_list in sorted(model_data.items()):
        color = param_colors_hex.get((P, chi), '#000000')  # default to black if not in dict
        linestyle = '-' if chi == 80.0 else '--'
        
        # Get N and d from first data element
        N = data_list[0]['N']
        d = data_list[0]['d']
        
        # Count number of seeds
        num_seeds = len(data_list)
        
        # Compute average variance across ensemble members
        perp_vars = [data['perp_var'] for data in data_list if data['perp_var'] is not None]
        avg_perp_var = np.mean(perp_vars) if len(perp_vars) > 0 else None
        
        # Average over ensemble members for this (P, chi) pair
        all_centers = []
        all_probs = []
        for data in data_list:
            if len(data['perp_centers']) > 0:
                all_centers.append(data['perp_centers'])
                all_probs.append(data['perp_neg_log_probs'])
        
        if len(all_centers) == 0:
            continue
        
        # Combine and sort for smoother plotting
        combined_centers = np.concatenate(all_centers)
        combined_probs = np.concatenate(all_probs)
        
        # Sort by centers for plotting
        sort_idx = np.argsort(combined_centers)
        combined_centers = combined_centers[sort_idx]
        combined_probs = combined_probs[sort_idx]
        
        # Get kappa from first data element
        kappa = data_list[0]['kappa']
        kappa_corrected = data_list[0]['kappa_corrected']
        label_base = f'P={P}, $\chi={chi}$, $\kappa={kappa_corrected:.2f}$\nN={N}'
        if avg_perp_var is not None:
            label_base += f', σ²={avg_perp_var:.4f}'
        if chi < N:
            label_base += f'\n{num_seeds} seeds (Grokking)'
        elif chi != N:
            label_base += f'\n{num_seeds} seeds (GFL)'
        else:
            label_base += '\n(Grokking)'
        ax_perp.plot(combined_centers, combined_probs, color=color, linestyle=linestyle, 
                    linewidth=2, label=label_base, alpha=0.8)
        
        # Overlay theoretical Gaussian if available
        theory_key = (d, P, chi)
        if theory_key in theory_data and theory_data[theory_key]['lWTP'] is not None:
            lWTP = theory_data[theory_key]['lWTP']
            sigma = np.sqrt(lWTP)
            x_range = np.linspace(-3*sigma, 3*sigma, 200)
            # Gaussian: p(w) = 1/(sigma*sqrt(2*pi)) * exp(-w^2/(2*sigma^2))
            # -log(p(w)) = 0.5*log(2*pi*var) + w^2/(2*var) = 0.5*log(2*pi) + log(sigma) + w^2/(2*sigma^2)
            gaussian_neg_log_p = 0.5*np.log(2*np.pi) + np.log(sigma) + x_range**2 / (2*sigma**2)
            ax_perp.plot(x_range, gaussian_neg_log_p, color=color, linestyle=':', 
                        linewidth=1.5, alpha=0.6, label=f'$\\lambda_W^\\perp={lWTP:.4f}$')
    
    ax_perp.set_xlabel(r'$w\cdot w^\perp$', fontsize=12)
    ax_perp.set_ylabel(r'$-\log P(w)$', fontsize=12)
    ax_perp.set_title(r'Perpendicular Weights: $W_{:,:,1:}$', fontsize=12, fontweight='normal', pad=10)
    ax_target.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    
    legend_kwargs = dict(
        loc='lower right',
        fontsize=9,
        ncol=1,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='black',
        framealpha=0.95,
        borderpad=0.5,
        labelspacing=0.4,
        handlelength=1.2,
        columnspacing=0.7,
        handletextpad=0.5,
    )
    ax_target.legend(**legend_kwargs)
    ax_perp.legend(**legend_kwargs)
    
    fig.subplots_adjust(bottom=0.12, top=0.90, right=0.98, left=0.08)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(MODELDIR, 'chi_overlay_neg_log_prob.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved overlay plot to {output_path}")
    plt.show()
    
    # Create single subplot plot with only target weights
    fig_target, ax_target_single = plt.subplots(1, 1, figsize=(9.0, 6.0))
    fig_target.suptitle('Action for Readin Layer Weights \n'+ r'$S[w] = -\log P(w\cdot v_k)$ for FCN2 $f(x) = \mathbf{a}\cdot \mathrm{erf}(\mathbf{W}\cdot \mathbf{x}),\; x\in \mathbb{R}^d,\; d=100$', fontsize=14, fontweight='normal', y=0.98)
    
    for (P, chi), data_list in sorted(model_data.items()):
        color = param_colors_hex.get((P, chi), '#000000')  # default to black if not in dict
        linestyle = '-' if chi < data_list[0]['N'] else '--'
        
        # Get N from first data element
        N = data_list[0]['N']
        
        # Count number of seeds
        num_seeds = len(data_list)
        
        # Compute average variance across ensemble members
        target_vars = [data['target_var'] for data in data_list]
        avg_target_var = np.mean(target_vars)
        
        # Average over ensemble members for this (P, chi) pair
        all_centers = []
        all_probs = []
        for data in data_list:
            all_centers.append(data['target_centers'])
            all_probs.append(data['target_neg_log_probs'])
        
        # Combine and sort for smoother plotting
        combined_centers = np.concatenate(all_centers)
        combined_probs = np.concatenate(all_probs)
        
        # Sort by centers for plotting
        sort_idx = np.argsort(combined_centers)
        combined_centers = combined_centers[sort_idx]
        combined_probs = combined_probs[sort_idx]
        
        # Get kappa from first data element
        kappa = data_list[0]['kappa']
        kappa_corrected = data_list[0]['kappa_corrected']
        label = f'P={P}, $\chi={chi}$, $\kappa={kappa_corrected:.2f}$\nN={N}, σ²={avg_target_var:.4f}'
        if chi < N:
            label += f'\n{num_seeds} seeds (Grokking)'
        elif chi != N:
            label += f'\n{num_seeds} seeds (GFL)'
        else:
            label += '\n(Grokking)'
        ax_target_single.plot(combined_centers, combined_probs, color=color, linestyle=linestyle, 
                      linewidth=2, label=label, alpha=0.8)
        
        # Overlay theoretical Gaussian if available
        d = data_list[0]['d']
        theory_key = (d, P, chi)
        if theory_key in theory_data and theory_data[theory_key]['lWT'] is not None:
            lWT = theory_data[theory_key]['lWT']
            sigma = np.sqrt(lWT)
            x_range = np.linspace(-3*sigma, 3*sigma, 200)
            # Gaussian: p(w) = 1/(sigma*sqrt(2*pi)) * exp(-w^2/(2*sigma^2))
            # -log(p(w)) = 0.5*log(2*pi) + log(sigma) + w^2/(2*sigma^2)
            gaussian_neg_log_p = 0.5*np.log(2*np.pi) + np.log(sigma) + x_range**2 / (2*sigma**2)
            ax_target_single.plot(x_range, gaussian_neg_log_p, color=color, linestyle=':', 
                          linewidth=1.5, alpha=0.6, label=f'$\lambda_W^*={lWT:.4f}$')
    
    ax_target_single.set_xlabel(r'$w\cdot w^*$', fontsize=12)
    ax_target_single.set_ylabel(r'$-\log P(w)$', fontsize=12)
    ax_target_single.set_title(r'Target Weights: $W_{:,:,0}$', fontsize=12, fontweight='normal', pad=10)
    ax_target_single.grid(True, alpha=0.3)
    ax_target_single.legend(**legend_kwargs)
    
    fig_target.subplots_adjust(bottom=0.12, top=0.90, left=0.10, right=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path_target = os.path.join(MODELDIR, 'chi_overlay_target_only.png')
    plt.savefig(output_path_target, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved target-only plot to {output_path_target}")
    plt.show()

if __name__ == "__main__":
    main()
