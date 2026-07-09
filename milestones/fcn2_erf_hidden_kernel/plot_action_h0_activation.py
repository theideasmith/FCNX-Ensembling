import torch
import numpy as np
from pathlib import Path
import sys
import json
import subprocess
import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf

model_dirs = [
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_2',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_2'
]
# model_dirs = [
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_1e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_2'
# ]

import os
MODELDIR = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults'
model_dirs = [os.path.join(MODELDIR, d) for d in os.listdir(MODELDIR)]
# Only take directories (not files)
model_dirs = [d for d in model_dirs if os.path.isdir(d)]

model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_80.0_lr_0.0003_T_2.0_seed_42']

model_dirs = [
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_2',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_3']
model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.03', 
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_3_eps_0.03']

model_dirs = [
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_1',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_0',
    '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_1'
]

model_dirs = [
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_3e-05_T_0.2_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_3e-05_T_0.2_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_3e-05_T_0.2_seed_1_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_3e-05_T_0.2_seed_1_eps_0.03'
]

def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4])

    T = None
    if 'T' in parts:
        T = float(parts[parts.index('T') + 1])

    seed = None
    if 'seed' in parts:
        seed = int(parts[parts.index('seed') + 1])

    epsilon = None
    if 'eps' in parts:
        epsilon = float(parts[parts.index('eps') + 1])

    return d, P, N, chi, seed, T, epsilon

def load_model(model_dir, device):
    d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
    model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        print(f"Model not found in {model_dir}")
        return None, None, None
    state_dict = torch.load(model_path, map_location=device)
    ens = state_dict['W0'].shape[0]
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1/d, 1/(N*chi)), device=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, d, P, seed

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt
    # Group models by parameter set (d, P, N, chi)
    from collections import defaultdict
    grouped = defaultdict(list)
    param_labels = {}
    for model_dir in model_dirs:
        d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
        key = (d, P, N, chi)
        grouped[key].append((model_dir, seed))
        param_labels[key] = f"d={d}, P={P}, N={N}, chi={chi}"

    param_keys = sorted(grouped.keys(), key=lambda x: (x[1], x[3]))  # sort by P, chi
    ncols = 4
    nrows = (len(param_keys) + ncols - 1) // ncols
    fig_scatter, axes_scatter = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, key in enumerate(param_keys):
        ax = axes_scatter[idx//ncols, idx%ncols]
        for i, (model_dir, seed) in enumerate(sorted(grouped[key], key=lambda x: x[1])):
            print(f"Loading model from {model_dir}")
            d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
            model, *_ = load_model(model_dir, device)
            if model is None:
                continue
            # Generate dataset X with correct seed
            if seed is not None:
                torch.manual_seed(seed)
                import numpy as np
                np.random.seed(seed)
            X = torch.randn(1000, d, device=device)
            f_full = model.forward(X).detach().cpu().numpy()  # (P, ens)
            X0 = X[:,0].cpu().numpy()
            if f_full.ndim == 2:
                model_output = f_full.mean(axis=1)  # (P,)
            else:
                model_output = f_full
            color = color_cycle[i % len(color_cycle)]
            ax.scatter(X0, model_output, s=2, alpha=0.7, label=f'seed={seed}', color=color)
        # Add y = x reference line
        min_val = ax.get_xlim()[0]
        max_val = ax.get_xlim()[1]
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='y = x')
        ax.set_xlabel('X[:,0] (True Target)')
        ax.set_ylabel('Model Output')
        ax.set_title(param_labels[key])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- New: Per-model h0_activation projection log-action plots ---
    n_models = len(model_dirs)
    ncols_hist = 4
    nrows_hist = (n_models + ncols_hist - 1) // ncols_hist
    fig_hist, axes_hist = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)

    def log_density_curve(values, bins=40, min_samples=3, eps=1e-9):
        values_np = np.asarray(values, dtype=np.float64)
        counts_raw, bin_edges = np.histogram(values_np, bins=bins)
        density, _ = np.histogram(values_np, bins=bin_edges, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = counts_raw >= min_samples
        log_density = np.where(mask, -np.log(density + eps), np.nan)
        return centers, log_density

    def symmetric_bimodal_action(x, mu, sigma, eps=1e-12):
        x_np = np.asarray(x, dtype=np.float64)
        mu = float(abs(mu))
        sigma = float(max(sigma, 1e-12))
        density = 0.5 * np.exp(-0.5 * ((x_np - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        density += 0.5 * np.exp(-0.5 * ((x_np + mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        return -np.log(density + eps)

    def fit_bimodal_gaussian_action(values, eps=1e-12):
        values_np = np.asarray(values, dtype=np.float64).reshape(-1)
        if values_np.size < 10:
            return None

        mean = float(values_np.mean())
        std = float(values_np.std())
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0

        median = float(np.median(values_np))
        lower = values_np[values_np <= median]
        upper = values_np[values_np > median]
        mu1_init = float(lower.mean()) if lower.size else mean - 0.5 * std
        mu2_init = float(upper.mean()) if upper.size else mean + 0.5 * std
        sigma_init = max(std / 2.0, 1e-3)
        counts_raw, bin_edges = np.histogram(values_np, bins=40)
        empirical = counts_raw.astype(np.float64)
        empirical_sum = float(empirical.sum())
        if empirical_sum <= 0.0:
            return None
        empirical = empirical / empirical_sum

        def normal_cdf(x, mu, sigma):
            sigma = np.maximum(sigma, 1e-12)
            return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

        def model_bin_probs(pi, mu1, mu2, sigma1, sigma2):
            left = bin_edges[:-1]
            right = bin_edges[1:]
            probs = (
                pi * (normal_cdf(right, mu1, sigma1) - normal_cdf(left, mu1, sigma1))
                + (1.0 - pi) * (normal_cdf(right, mu2, sigma2) - normal_cdf(left, mu2, sigma2))
            )
            probs = np.clip(probs, eps, None)
            return probs / probs.sum()

        def unpack(params):
            logit_pi, mu1, mu2, log_sigma1, log_sigma2 = params
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            sigma1 = np.exp(log_sigma1)
            sigma2 = np.exp(log_sigma2)
            return pi, mu1, mu2, sigma1, sigma2

        def objective(params):
            pi, mu1, mu2, sigma1, sigma2 = unpack(params)
            model_probs = model_bin_probs(pi, mu1, mu2, sigma1, sigma2)
            return float(np.sum(empirical * (np.log(empirical + eps) - np.log(model_probs + eps))))

        initial_params = np.array([
            0.0,
            mu1_init,
            mu2_init,
            np.log(sigma_init),
            np.log(sigma_init),
        ], dtype=np.float64)

        result = minimize(objective, initial_params, method='L-BFGS-B')
        if not result.success:
            return None

        pi, mu1, mu2, sigma1, sigma2 = unpack(result.x)
        if mu1 > mu2:
            pi = 1.0 - pi
            mu1, mu2 = mu2, mu1
            sigma1, sigma2 = sigma2, sigma1

        return {
            'pi': float(pi),
            'mu1': float(mu1),
            'sigma1': float(sigma1),
            'mu2': float(mu2),
            'sigma2': float(sigma2),
            'success': bool(result.success),
            'kl': float(result.fun),
        }

    def bimodal_gaussian_action(x, fit_params):
        x_np = np.asarray(x, dtype=np.float64)
        pi = fit_params['pi']
        mu1 = fit_params['mu1']
        sigma1 = fit_params['sigma1']
        mu2 = fit_params['mu2']
        sigma2 = fit_params['sigma2']
        density = (
            pi * np.exp(-0.5 * ((x_np - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2.0 * np.pi))
            + (1.0 - pi) * np.exp(-0.5 * ((x_np - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2.0 * np.pi))
        )
        return -np.log(density + 1e-12)

    from compute_h3_projections import compute_theory_with_julia

    theory_cache = {}
    vga_cache = {}

    def get_theory_for_model(model_dir, d, P, N, chi, T, epsilon):
        epsilon = 0.0
        if model_dir in theory_cache:
            return theory_cache[model_dir]
        kappa = (T / 2.0) if T is not None else (1.0 / chi if chi else 0.0)
        try:
            theory_cache[model_dir] = compute_theory_with_julia(d, N, P, chi, kappa, epsilon)
        except Exception as exc:
            print(f"Could not compute theory for {model_dir}: {exc}")
            theory_cache[model_dir] = None
        return theory_cache[model_dir]

    def get_vga_for_model(model_dir, d, P, N, chi, T, epsilon):
        if model_dir in vga_cache:
            return vga_cache[model_dir]
        kappa = (T / 2.0) if T is not None else (1.0 / chi if chi else 0.0)
        print("Using kappa: ", kappa)
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "fcn2_vga_erf.jl"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cmd = [
                "julia",
                str(julia_script),
                "--d", str(d),
                "--n1", str(N),
                "--P", str(P),
                "--chi", str(chi),
                "--kappa", str(kappa),
                "--epsilon", str(epsilon if epsilon is not None else 0.03),
                "--to", str(tmp_path),
                "--quiet",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(tmp_path, "r") as f:
                vga_cache[model_dir] = json.load(f)
        except Exception as exc:
            print(f"Could not compute VGA theory for {model_dir}: {exc}")
            vga_cache[model_dir] = None
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return vga_cache[model_dir]

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        # Generate x from seed
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(10000, d, dtype=torch.float32, device=device)
        # Compute h0_activation (assume shape (N,))
        if hasattr(model, 'h0_activation'):
            with torch.no_grad():
                h0 = model.h0_activation(x)
        else:
            # fallback: use first layer pre-activation if available
            h0 = x[:,0]
        del model
        # Hermite3 and linear projections
        x0 = x[:,0]
        h3 = x0**3 - 3.0 * x0
        lin = x0
        # Project h0 onto Hermite3 and linear directions
        proj_h3 = torch.einsum('pqn,p->qn', h0, h3) / x0.shape[0]
        proj_lin = torch.einsum('pqn,p->qn', h0, lin) / x0.shape[0]
        del x0, h0, x
        axh = axes_hist[idx//ncols_hist, idx%ncols_hist]
        var_h3 = float(proj_h3.flatten().var().item())
        var_lin = float(proj_lin.flatten().var().item())
        for v, label, color, var in zip(
            [proj_h3, proj_lin],
            ['Hermite3', 'Linear'],
            ['royalblue', 'orange'],
            [var_h3, var_lin],
        ):
            centers, log_density = log_density_curve(v.flatten().cpu().numpy(), bins=40, min_samples=3)
            mask = np.isfinite(log_density)
            axh.plot(centers[mask], log_density[mask], label=rf'{label} ($\sigma^2={var:.3g}$)', color=color, linewidth=1.2, marker='x', ms=4)
        # breakpoint()
        theory = get_theory_for_model(model_dir, d, P, N, chi, T, epsilon)
        lK1T = theory.get("target", {}).get("lK1T") if theory is not None else None

        if lK1T is not None and lK1T > 0:
            lin_centers, lin_action = log_density_curve(proj_lin.flatten().cpu().numpy(), bins=40, min_samples=3)
            lin_mask = np.isfinite(lin_action)
            if np.any(lin_mask):
                x_min = float(lin_centers[lin_mask].min())
                x_max = float(lin_centers[lin_mask].max())
                x_theory = np.linspace(x_min, x_max, 1000)
                gaussian_action = 0.5 * x_theory**2 / float(lK1T) + 0.5 * np.log(2.0 * np.pi * float(lK1T))
                axh.plot(x_theory, gaussian_action, '--', color='black', linewidth=1.4, label=rf'Gaussian theory ($lK1T={float(lK1T):.3g}$)')

        axh.set_title(Path(model_dir).name)
        axh.set_xlabel('Projection value')
        axh.set_ylabel('Action: -log P')
        axh.legend()
        axh.grid(True, alpha=0.3)
    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_hist[j//ncols_hist, j%ncols_hist].axis('off')
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'h0_activation_projection_histograms.png'), dpi=150)
    plt.close(fig_hist)
    print('Saved h0_activation projection histograms to h0_activation_projection_histograms.png')

    # --- New: Per-model first-layer target weight action plots ---
    fig_weights, axes_weights = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)
    fig_wsq, axes_wsq = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue

        weights_target = model.W0[:, :, 0].detach().cpu().numpy().reshape(-1)
        counts_raw, bin_edges = np.histogram(weights_target, bins=40)
        density, _ = np.histogram(weights_target, bins=bin_edges, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mask = counts_raw >= 3
        action = np.where(mask, -np.log(density + 1e-9), np.nan)
        var_w0t = float(np.var(weights_target))

        theory = theory_cache.get(model_dir)

        lWT = theory.get("target").get("lWT", None) if theory is not None else None

        axw = axes_weights[idx//ncols_hist, idx%ncols_hist]
        axw.plot(bin_centers[mask], action[mask], color='royalblue', linewidth=1.2, marker='x', ms=4, label=rf'$W_{{:,:,0}}$ action ($\sigma^2={var_w0t:.3g}$)')

        print("lWT:", lWT)
        if lWT is not None and lWT > 0:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_theory = np.linspace(x_min, x_max, 1000)
            gaussian_action = 0.5 * x_theory**2 / float(lWT) + 0.5 * np.log(2.0 * np.pi * float(lWT))
            axw.plot(x_theory, gaussian_action, '--', color='black', linewidth=1.4, label=rf'Gaussian theory ($lWT={float(lWT):.3g}$)')

        vga_theory = get_vga_for_model(model_dir, d, P, N, chi, T, epsilon)
        vga_target = vga_theory.get("vga", {}).get("target") if vga_theory is not None else None
        if vga_target is not None:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_fit = np.linspace(x_min, x_max, 1000)
            action_fit = symmetric_bimodal_action(x_fit, vga_target.get("muW", 0.0), vga_target.get("sigS", 1.0))
            axw.plot(
                x_fit,
                action_fit,
                color='crimson',
                linewidth=1.8,
                label=(
                    rf"VGA bimodal $\mu={float(abs(vga_target.get('muW', 0.0))):.3g}$, "
                    rf"$\sigma={float(vga_target.get('sigS', 1.0)):.3g}$"
                ),
            )
            print(
                f"VGA bimodal parameters for {Path(model_dir).name}: "
                f"muW={float(vga_target.get('muW', 0.0)):.6g}, sigS={float(vga_target.get('sigS', 1.0)):.6g}, "
                f"lWT={float(vga_target.get('lWT', float('nan'))):.6g}"
            )

        bimodal_fit = fit_bimodal_gaussian_action(weights_target)
        if bimodal_fit is not None:
            x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
            x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
            x_fit = np.linspace(x_min, x_max, 1000)
            action_fit = bimodal_gaussian_action(x_fit, bimodal_fit)
            axw.plot(
                x_fit,
                action_fit,
                color='magenta',
                linewidth=1.2,
                linestyle='--',
                label=(
                    rf"Empirical fit $\pi={bimodal_fit['pi']:.3g}$, "
                    rf"$\mu_1={bimodal_fit['mu1']:.3g}$, $\sigma_1={bimodal_fit['sigma1']:.3g}$, "
                    rf"$\mu_2={bimodal_fit['mu2']:.3g}$, $\sigma_2={bimodal_fit['sigma2']:.3g}$"
                ),
            )
            print(
                f"Bimodal W1 fit for {Path(model_dir).name}: "
                f"pi={bimodal_fit['pi']:.6g}, mu1={bimodal_fit['mu1']:.6g}, sigma1={bimodal_fit['sigma1']:.6g}, "
                f"mu2={bimodal_fit['mu2']:.6g}, sigma2={bimodal_fit['sigma2']:.6g}, kl={bimodal_fit['kl']:.6g}"
            )

        axw.set_title(Path(model_dir).name, fontsize=26)
        axw.set_xlabel('Target weight value', fontsize=22)
        axw.set_ylabel('Action: -log P', fontsize=22)
        axw.tick_params(axis='both', labelsize=18)
        axw.legend(fontsize=18)
        axw.grid(True, alpha=0.3)

        axsq = axes_wsq[idx//ncols_hist, idx%ncols_hist]
        weights_target_sq = weights_target**2
        weights_perp = model.W0[:, :, 1:].detach().cpu().numpy().reshape(-1)
        weights_perp_sq = weights_perp**2
        axsq.hist(weights_target_sq, bins=40, density=True, color='royalblue', alpha=0.55, label=r'Target $W_{:, :, 0}^2$')
        axsq.hist(weights_perp_sq, bins=40, density=True, color='seagreen', alpha=0.45, label=r'Perp $W_{:, :, 1:}^2$')
        axsq.set_title(Path(model_dir).name, fontsize=26)
        axsq.set_xlabel('Squared weight value', fontsize=22)
        axsq.set_ylabel('Density', fontsize=22)
        axsq.tick_params(axis='both', labelsize=18)
        axsq.legend(fontsize=18)
        axsq.grid(True, alpha=0.3)
        del model

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_weights[j//ncols_hist, j%ncols_hist].axis('off')
    fig_weights.tight_layout()
    fig_weights.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'weight_action_target_histograms.png'), dpi=150)
    plt.close(fig_weights)
    print('Saved target weight action plots to weight_action_target_histograms.png')

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_wsq[j//ncols_hist, j%ncols_hist].axis('off')
    fig_wsq.tight_layout()
    fig_wsq.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'weight_sq_target_histograms.png'), dpi=150)
    plt.close(fig_wsq)
    print('Saved squared target weight plots to weight_sq_target_histograms.png')

    # # --- Output projection histograms (onto target x[:,0] and perp x[:,3]) ---
    # fig_output, axes_output = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)
    # from compute_h3_projections import compute_theory_with_julia
    # for idx, model_dir in enumerate(model_dirs):
    #     d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
    #     model, *_ = load_model(model_dir, device)
    #     if model is None:
    #         continue
    #     # Generate x from seed
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #     # Streaming batch computation for projections
    #     total_samples = 50000000
    #     batch_size = 5000
    #     num_batches = total_samples // batch_size
    #     remainder = total_samples % batch_size
    #     dtype = torch.float32
    #     ens = model.ens
    #     n1 = model.n1
    #     # Accumulators
    #     proj_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    #     proj_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
    #     for i in range(num_batches + (1 if remainder > 0 else 0)):
    #         bs = batch_size if i < num_batches else remainder
    #         if bs == 0:
    #             break
    #         X_batch = torch.randn(bs, d, dtype=dtype, device=device)

    #         x0 = X_batch[:, 0]
    #         phi3_target = x0**3 - 3.0 * x0
    #         # Perpendicular projections: average over all x[:,1:]
    #         phi3_perp_sum = torch.zeros(bs, dtype=dtype, device=device)
    #         for j in [1]:
    #             xj = X_batch[:, j]
    #             phi3_perp_sum += xj**3 - 3.0 * xj
    #         phi3_perp = phi3_perp_sum
    #         with torch.no_grad():
    #             a0 = model.h0_activation(X_batch)
    #         proj_target_sum += torch.einsum('pqn,p->qn', a0, phi3_target)
    #         proj_perp_sum += torch.einsum('pqn,p->qn', a0, phi3_perp)

    #         del X_batch, a0, phi3_target, phi3_perp, phi3_perp_sum, x0
    #     del model
    #     torch.cuda.empty_cache()
    #     # Normalize
    #     proj_target = proj_target_sum / total_samples
    #     proj_perp = proj_perp_sum / total_samples
    #     # Compute variances
    #     var_target = proj_target.var().item()
    #     var_perp = proj_perp.var().item()
    #     axo = axes_output[idx//ncols_hist, idx%ncols_hist]
    #     # Histogram and action plot for target

    #     # --- Overlay theoretical Gaussian action curves ---
    #     # Parse config for chi, kappa, epsilon if present
    #     d_cfg, P_cfg, N_cfg, chi, seed_cfg, T_cfg, epsilon_cfg = parse_config_from_dirname(model_dir)
    #     # Estimate kappa as 1/chi if not present
    #     kappa = T_cfg
    #     # Use N as n1
    #     epsilon = epsilon_cfg if epsilon_cfg is not None else 0.0
    #     theory = compute_theory_with_julia(d_cfg, N_cfg, P_cfg, chi, kappa, epsilon)
    #     lJ3T = theory["target"]["lJ3T"]
    #     lJ3P = theory["perpendicular"]["lJ3P"]

    #     # Compute histogram and mask before using for theory overlay x-range
    #     v_flat = proj_target.flatten().cpu()
    #     hist_range = (v_flat.min().item(), v_flat.max().item())
    #     bins = 200
    #     hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
    #     bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     bin_width = (hist_range[1] - hist_range[0]) / bins
    #     probs_density = (hist / hist.sum()) / bin_width
    #     mask = probs_density > 0
    #     bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_cpu = mask.cpu() if mask.is_cuda else mask

    #     # Plot theoretical Gaussian action for target and perp, using a smooth line over the same x-range as experiment (after mask)
    #     import numpy as np
    #     bin_centers_np = bin_centers_cpu[mask_cpu].numpy() if hasattr(bin_centers_cpu, 'numpy') else np.array(bin_centers_cpu)[mask_cpu]
    #     if len(bin_centers_np) > 1:
    #         x_min = float(bin_centers_np.min())
    #         x_max = float(bin_centers_np.max())
    #     else:
    #         x_min = float(bin_edges[0])
    #         x_max = float(bin_edges[-1])
    #     x_theory = np.linspace(x_min, x_max, 1000)
    #     for lJ3, color, label in [
    #         (lJ3T, 'royalblue', 'Theory Target'),
    #         # (lJ3P, 'orange', 'Theory Perp')
    #     ]:
    #         var = lJ3
    #         action = 0.5 * x_theory**2 / var + 0.5 * np.log(2 * np.pi * var)
    #         axo.plot(x_theory, action, '--', color=color, label=f'{label} $\\sigma^2={var:.2e}$')
    #     v_flat = proj_target.flatten().cpu()
    #     hist_range = v_flat.min().item(), v_flat.max().item()
    #     bins = 200
    #     hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
    #     bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
    #     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     bin_width = (hist_range[1] - hist_range[0]) / bins
    #     probs_density = (hist / hist.sum()) / bin_width
    #     mask = probs_density > 0
    #     bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_cpu = mask.cpu() if mask.is_cuda else mask
    #     axo.plot(
    #         bin_centers_cpu[mask_cpu].numpy(),
    #         (-probs_density[mask].log()).cpu().numpy(),
    #         label=f'Target $x_0$ ($\\sigma^2={var_target:.2e}$)',
    #         color='royalblue'
    #     )
    #     # Histogram and action plot for averaged perp
    #     v_flat_perp = proj_perp.flatten().cpu()
    #     hist_perp = torch.histc(v_flat_perp, bins=bins, min=hist_range[0], max=hist_range[1])
    #     probs_perp_density = (hist_perp / hist_perp.sum()) / bin_width
    #     mask_perp = probs_perp_density > 0
    #     bin_centers_perp_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
    #     mask_perp_cpu = mask_perp.cpu() if mask_perp.is_cuda else mask_perp
    #     d_val = d if d is not None else 0
    #     axo.plot(
    #         bin_centers_perp_cpu[mask_perp_cpu].numpy(),
    #         (-probs_perp_density[mask_perp].log()).cpu().numpy(),
    #         label=f'Perp avg $x_{{1...{d_val-1}}}$ ($\\sigma^2={var_perp:.2e}$)',
    #         color='orange'
    #     )
        
    #     # Parse config parameters for title
    #     d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
    #     axo.set_title(f"Action: $-\\log P$ | d={d}, P={P}, N={N}, $\\chi$={chi}", fontsize=11)
    #     axo.set_xlabel('Output projection value')
    #     axo.set_ylabel('Action: -log P')
    #     axo.legend()
    #     axo.grid(True, alpha=0.3)
    
    # for j in range(idx+1, nrows_hist*ncols_hist):
    #     axes_output[j//ncols_hist, j%ncols_hist].axis('off')
    # fig_output.tight_layout()
    # fig_output.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'output_projection_histograms.png'), dpi=150)
    # plt.close(fig_output)
    # print('Saved output projection histograms to output_projection_histograms.png')

    # Hide unused subplots
    for j in range(len(param_keys), nrows*ncols):
        axes_scatter[j//ncols, j%ncols].axis('off')
    fig_scatter.tight_layout()
    fig_scatter.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'grid_X0_vs_model_output_grouped.png'), dpi=150)
    print('Saved grouped grid scatter plot to grid_X0_vs_model_output_grouped.png')

if __name__ == "__main__":
    main()
