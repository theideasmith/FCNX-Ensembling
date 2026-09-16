import argparse
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

from plot_action_h0_activation import (
    EXPERIMENT_GROUP_BY_NAME,
    OUTPUT_BASE_DIR,
    load_model,
    parse_config_from_dirname,
    slugify,
)

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

import os
MODELDIR = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults'
model_dirs = [os.path.join(MODELDIR, d) for d in os.listdir(MODELDIR)]
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
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_2.0_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_0.0001_T_2.0_seed_0_eps_0.03',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_0.0001_T_2.0_seed_0_eps_0.03'
]

model_dirs = [
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_1_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d30_P3000_N600_chi_60.0_lr_3e-05_T_2.0_seed_2_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N60_chi_60.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P400_N120_chi_120.0_lr_1e-05_T_0.2_seed_1_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N300_chi_300.0_lr_0.0001_T_0.1_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_0.1_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_5e-06_T_0.1_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1400_chi_1400.0_lr_5e-06_T_0.1_seed_42_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_0.0001_T_2.0_seed_0_eps_0.03",
    "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_5e-06_T_0.1_seed_0_eps_0.03",
    # "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N3800_chi_3800.0_lr_5e-06_T_0.1_seed_42_eps_0.03"
]


def extract_theory_params(data):
    if not isinstance(data, dict):
        return {}
    if "vga" in data and isinstance(data["vga"], dict):
        data = data["vga"]
    elif "naive" in data and isinstance(data["naive"], dict):
        data = data["naive"]
    
    if "target" in data and isinstance(data["target"], dict):
        target = data["target"]
    else:
        target = data

    return {
        "lJ1": target.get("lJ1", target.get("lJ1T")),
        "lJ3": target.get("lJ3", target.get("lJ3T")),
        "lWT": target.get("lWT"),
        "muW": target.get("muW"),
        "sigS": target.get("sigS"),
    }

def main():
    parser = argparse.ArgumentParser(description='Compare empirical actions to VGA and naive theory')
    parser.add_argument(
        '--group',
        choices=sorted(EXPERIMENT_GROUP_BY_NAME),
        help='Experiment group from plot_action_h0_activation.py',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory for output plots (default: action_h0_activation_plots/<group>)',
    )
    args = parser.parse_args()

    if args.group:
        model_dirs = list(EXPERIMENT_GROUP_BY_NAME[args.group].model_dirs)
        output_dir = args.output_dir or (OUTPUT_BASE_DIR / slugify(args.group))
    else:
        output_dir = args.output_dir or Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt

    if not model_dirs:
        print('No model directories to plot.')
        return

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
            d, P, N, chi, seed, T, epsilon, *_ = parse_config_from_dirname(model_dir)
            model, *_ = load_model(model_dir, device)
            if model is None:
                continue
            if seed is not None:
                torch.manual_seed(seed)
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
        min_val = ax.get_xlim()[0]
        max_val = ax.get_xlim()[1]
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='y = x')
        ax.set_xlabel('X[:,0] (True Target)')
        ax.set_ylabel('Model Output')
        ax.set_title(param_labels[key])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- Per-model h0_activation projection log-action plots ---
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

    vga_cache = {}
    naive_cache = {}

    def get_vga_for_model(model_dir, d, P, N, chi, T, epsilon):
        if model_dir in vga_cache:
            return vga_cache[model_dir]
        kappa = (T / 2.0) if T is not None else (1.0 / chi if chi else 0.0)
        print("Using kappa (VGA): ", kappa)
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

    def get_naive_theory_for_model(model_dir, d, P, N, chi, T, epsilon):
        if model_dir in naive_cache:
            return naive_cache[model_dir]
        kappa = (T / 2.0) if T is not None else (1.0 / chi if chi else 0.0)
        print("Using kappa (naive): ", kappa)
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_cubic_eigs.jl"
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
                naive_cache[model_dir] = json.load(f)
        except Exception as exc:
            print(f"Could not compute Naive theory for {model_dir}: {exc}")
            naive_cache[model_dir] = None
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return naive_cache[model_dir]

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon, *_ = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        if seed is not None:
            torch.manual_seed(seed)
        total_draws = 1000000
        batch_size = 10000
        proj_h3_sum = 0
        proj_lin_sum = 0

        with torch.no_grad():
            for _ in range(total_draws // batch_size):
                x = torch.randn(batch_size, d, dtype=torch.float32, device=device)
                if hasattr(model, 'h0_activation'):
                    h0 = model.h0_activation(x)
                else:
                    h0 = x[:, 0].unsqueeze(-1).unsqueeze(-1)

                x0 = x[:, 0]
                h3 = (x0**3 - 3.0 * x0) / 6**0.5
                lin = x0

                proj_h3_sum = proj_h3_sum + torch.einsum('pqn,p->qn', h0, h3)
                proj_lin_sum = proj_lin_sum + torch.einsum('pqn,p->qn', h0, lin)
                del x, h0, x0, h3, lin

        proj_h3 = proj_h3_sum / total_draws
        proj_lin = proj_lin_sum / total_draws
        del model

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

        # Theoretical curves: VGA & Naive
        vga_theory = get_vga_for_model(model_dir, d, P, N, chi, T, epsilon)
        vga_params = extract_theory_params(vga_theory)
        lJ1T_vga = vga_params.get("lJ1")
        lJ3T_vga = vga_params.get("lJ3")

        naive_theory = get_naive_theory_for_model(model_dir, d, P, N, chi, T, epsilon)
        naive_params = extract_theory_params(naive_theory)
        lJ1T_naive = naive_params.get("lJ1")
        lJ3T_naive = naive_params.get("lJ3")

        # Linear projections theory comparison
        lin_centers, lin_action = log_density_curve(proj_lin.flatten().cpu().numpy(), bins=40, min_samples=3)
        lin_mask = np.isfinite(lin_action)
        if np.any(lin_mask):
            x_min = float(lin_centers[lin_mask].min())
            x_max = float(lin_centers[lin_mask].max())
            x_theory = np.linspace(x_min, x_max, 1000)
            if lJ1T_vga is not None and lJ1T_vga > 0:
                gaussian_action_vga = 0.5 * x_theory**2 / float(lJ1T_vga) + 0.5 * np.log(2.0 * np.pi * float(lJ1T_vga))
                axh.plot(x_theory, gaussian_action_vga, '--', color='black', linewidth=1.4, label=rf'VGA theory ($lJ1T={float(lJ1T_vga):.3g}$)')
            if lJ1T_naive is not None and lJ1T_naive > 0:
                gaussian_action_naive = 0.5 * x_theory**2 / float(lJ1T_naive) + 0.5 * np.log(2.0 * np.pi * float(lJ1T_naive))
                axh.plot(x_theory, gaussian_action_naive, ':', color='purple', linewidth=1.4, label=rf'Naive theory ($lJ1T={float(lJ1T_naive):.3g}$)')

        # Hermite3 projections theory comparison
        h3_centers, h3_action = log_density_curve(proj_h3.flatten().cpu().numpy(), bins=40, min_samples=3)
        h3_mask = np.isfinite(h3_action)
        if np.any(h3_mask):
            x_min = float(h3_centers[h3_mask].min())
            x_max = float(h3_centers[h3_mask].max())
            x_theory = np.linspace(x_min, x_max, 1000)
            if lJ3T_vga is not None and lJ3T_vga > 0:
                gaussian_action_vga = 0.5 * x_theory**2 / float(lJ3T_vga) + 0.5 * np.log(2.0 * np.pi * float(lJ3T_vga))
                axh.plot(x_theory, gaussian_action_vga, '--', color='forestgreen', linewidth=1.4, label=rf'VGA theory ($lJ3T={float(lJ3T_vga):.3g}$)')
            if lJ3T_naive is not None and lJ3T_naive > 0:
                gaussian_action_naive = 0.5 * x_theory**2 / float(lJ3T_naive) + 0.5 * np.log(2.0 * np.pi * float(lJ3T_naive))
                axh.plot(x_theory, gaussian_action_naive, ':', color='teal', linewidth=1.4, label=rf'Naive theory ($lJ3T={float(lJ3T_naive):.3g}$)')

        axh.set_title(Path(model_dir).name)
        axh.set_xlabel('Projection value')
        axh.set_ylabel('Action: -log P')
        axh.legend()
        axh.grid(True, alpha=0.3)

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_hist[j//ncols_hist, j%ncols_hist].axis('off')
    fig_hist.tight_layout()
    fig_hist.savefig(output_dir / 'h0_activation_projection_histograms_vga_compare.png', dpi=150)
    plt.close(fig_hist)
    print(f'Saved h0_activation projection histograms to {output_dir / "h0_activation_projection_histograms_vga_compare.png"}')

    # --- Per-model first-layer target weight action plots ---
    fig_weights, axes_weights = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)
    fig_wsq, axes_wsq = plt.subplots(nrows_hist, ncols_hist, figsize=(15*ncols_hist, 12*nrows_hist), squeeze=False)

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon, *_ = parse_config_from_dirname(model_dir)
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

        vga_theory = get_vga_for_model(model_dir, d, P, N, chi, T, epsilon)
        vga_params = extract_theory_params(vga_theory)
        lWT_vga = vga_params.get("lWT")
        muW_vga = vga_params.get("muW")
        sigS_vga = vga_params.get("sigS")

        naive_theory = get_naive_theory_for_model(model_dir, d, P, N, chi, T, epsilon)
        naive_params = extract_theory_params(naive_theory)
        lWT_naive = naive_params.get("lWT")
        muW_naive = naive_params.get("muW")
        sigS_naive = naive_params.get("sigS")

        axw = axes_weights[idx//ncols_hist, idx%ncols_hist]
        axw.plot(bin_centers[mask], action[mask], color='royalblue', linewidth=1.2, marker='x', ms=4, label=rf'$W_{{:,:,0}}$ action ($\sigma^2={var_w0t:.3g}$)')

        print(f"[{Path(model_dir).name}] lWT VGA: {lWT_vga}, lWT Naive: {lWT_naive}")
        
        x_min = float(bin_centers[mask].min()) if np.any(mask) else float(bin_edges[0])
        x_max = float(bin_centers[mask].max()) if np.any(mask) else float(bin_edges[-1])
        x_theory = np.linspace(x_min, x_max, 1000)

        # Plot VGA Gaussian action
        if lWT_vga is not None and lWT_vga > 0:
            gaussian_action_vga = 0.5 * x_theory**2 / float(lWT_vga) + 0.5 * np.log(2.0 * np.pi * float(lWT_vga))
            axw.plot(x_theory, gaussian_action_vga, '--', color='black', linewidth=1.4, label=rf'VGA Gaussian theory ($lWT={float(lWT_vga):.3g}$)')

        # Plot Naive Gaussian action
        if lWT_naive is not None and lWT_naive > 0:
            gaussian_action_naive = 0.5 * x_theory**2 / float(lWT_naive) + 0.5 * np.log(2.0 * np.pi * float(lWT_naive))
            axw.plot(x_theory, gaussian_action_naive, ':', color='purple', linewidth=1.4, label=rf'Naive Gaussian theory ($lWT={float(lWT_naive):.3g}$)')

        # Plot VGA Bimodal action
        if muW_vga is not None and sigS_vga is not None:
            action_fit_vga = symmetric_bimodal_action(x_theory, muW_vga, sigS_vga)
            axw.plot(
                x_theory,
                action_fit_vga,
                color='crimson',
                linewidth=1.8,
                label=(
                    rf"VGA bimodal $\mu={float(abs(muW_vga)):.3g}$, "
                    rf"$\sigma={float(sigS_vga):.3g}$"
                ),
            )

        # Plot Naive Bimodal action (if present)
        if muW_naive is not None and sigS_naive is not None and (muW_naive != 0 or sigS_naive != 1.0):
            action_fit_naive = symmetric_bimodal_action(x_theory, muW_naive, sigS_naive)
            axw.plot(
                x_theory,
                action_fit_naive,
                ':',
                color='darkorange',
                linewidth=1.6,
                label=(
                    rf"Naive bimodal $\mu={float(abs(muW_naive)):.3g}$, "
                    rf"$\sigma={float(sigS_naive):.3g}$"
                ),
            )

        bimodal_fit = fit_bimodal_gaussian_action(weights_target)
        if bimodal_fit is not None:
            action_fit_emp = bimodal_gaussian_action(x_theory, bimodal_fit)
            axw.plot(
                x_theory,
                action_fit_emp,
                color='magenta',
                linewidth=1.2,
                linestyle='--',
                label=(
                    rf"Empirical fit $\pi={bimodal_fit['pi']:.3g}$, "
                    rf"$\mu_1={bimodal_fit['mu1']:.3g}$, $\sigma_1={bimodal_fit['sigma1']:.3g}$, "
                    rf"$\mu_2={bimodal_fit['mu2']:.3g}$, $\sigma_2={bimodal_fit['sigma2']:.3g}$"
                ),
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
    fig_weights.savefig(output_dir / 'weight_action_target_histograms_vga_compare.png', dpi=150)
    plt.close(fig_weights)
    print(f'Saved target weight action plots to {output_dir / "weight_action_target_histograms_vga_compare.png"}')

    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_wsq[j//ncols_hist, j%ncols_hist].axis('off')
    fig_wsq.tight_layout()
    fig_wsq.savefig(output_dir / 'weight_sq_target_histograms_vga_compare.png', dpi=150)
    plt.close(fig_wsq)
    print(f'Saved squared target weight plots to {output_dir / "weight_sq_target_histograms_vga_compare.png"}')

    for j in range(len(param_keys), nrows*ncols):
        axes_scatter[j//ncols, j%ncols].axis('off')
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / 'grid_X0_vs_model_output_grouped_vga_compare.png', dpi=150)
    print(f'Saved grouped grid scatter plot to {output_dir / "grid_X0_vs_model_output_grouped_vga_compare.png"}')

if __name__ == "__main__":
    main()