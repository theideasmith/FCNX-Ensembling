import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
import matplotlib.pyplot as plt

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

def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')

    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    # Format of the dirname is /home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.03
    # Extract temperature if present
    T = float(parts[8][2:]) if len(parts) > 8 and parts[8].startswith('T') else None
    # Extract seed if present
    seed = int(parts[10]) if len(parts) > 10 else None
    # Extract epsilon if present
    epsilon = None
    for p in parts:
        if p.startswith('eps'):
            try:
                epsilon = float(p[4:])
            except Exception:
                pass
    return d, P, N, chi, seed, T, epsilon

def load_model(model_dir, device):
    d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
    model_path = Path(model_dir) / "model.pt"
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

    # --- New: Per-model h0_activation projection histograms ---
    n_models = len(model_dirs)
    ncols_hist = 4
    nrows_hist = (n_models + ncols_hist - 1) // ncols_hist
    fig_hist, axes_hist = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)
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
        h3_raw = x0**3 - 3*x0
        h3 = h3_raw 
        lin = x0 
        # Project h0 onto Hermite3 and linear directions
        proj_h3 = torch.einsum('pqn, p->qn', h0, h3) / (x0.shape[0])
        proj_lin = torch.einsum('pqn, p->qn', h0, lin) / (x0.shape[0])
        del x0, h0, x
        vals = [proj_h3, proj_lin]
        axh = axes_hist[idx//ncols_hist, idx%ncols_hist]
        # Compute histogram and convert to probabilities, plot -log P as line (torch only)
        for v, label, color in zip(vals, ['Hermite3', 'Linear'], ['royalblue', 'orange']):
            v_flat = v.flatten().cpu()
            hist_range = (-.1, .1)
            bins = 200
            hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
            bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            probs = hist / hist.sum()
            mask = probs > 0
            axh.plot(bin_centers[mask].numpy(), (-probs[mask].log()).numpy(), label=label, color=color)
        axh.set_title(Path(model_dir).name)
        axh.set_xlabel(r'$-\log <|Projection|>$')
        axh.set_ylabel('Action: -log P')
        axh.legend()
        axh.grid(True, alpha=0.3)
    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_hist[j//ncols_hist, j%ncols_hist].axis('off')
    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'h0_activation_projection_histograms.png'), dpi=150)
    plt.close(fig_hist)
    print('Saved h0_activation projection histograms to h0_activation_projection_histograms.png')

    # --- Output projection histograms (onto target x[:,0] and perp x[:,3]) ---
    fig_output, axes_output = plt.subplots(nrows_hist, ncols_hist, figsize=(5*ncols_hist, 4*nrows_hist), squeeze=False)
    from compute_h3_projections import compute_theory_with_julia
    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        # Generate x from seed
        if seed is not None:
            torch.manual_seed(seed)
        # Streaming batch computation for projections
        total_samples = 50000000
        batch_size = 5000
        num_batches = total_samples // batch_size
        remainder = total_samples % batch_size
        dtype = torch.float32
        ens = model.ens
        n1 = model.n1
        # Accumulators
        proj_target_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
        proj_perp_sum = torch.zeros(ens, n1, dtype=dtype, device=device)
        for i in range(num_batches + (1 if remainder > 0 else 0)):
            bs = batch_size if i < num_batches else remainder
            if bs == 0:
                break
            X_batch = torch.randn(bs, d, dtype=dtype, device=device)

            x0 = X_batch[:, 0]
            phi3_target = x0**3 - 3.0 * x0
            # Perpendicular projections: average over all x[:,1:]
            phi3_perp_sum = torch.zeros(bs, dtype=dtype, device=device)
            for j in [1]:
                xj = X_batch[:, j]
                phi3_perp_sum += xj**3 - 3.0 * xj
            phi3_perp = phi3_perp_sum
            with torch.no_grad():
                a0 = model.h0_activation(X_batch)
            proj_target_sum += torch.einsum('pqn,p->qn', a0, phi3_target)
            proj_perp_sum += torch.einsum('pqn,p->qn', a0, phi3_perp)

            del X_batch, a0, phi3_target, phi3_perp, phi3_perp_sum, x0
        del model
        torch.cuda.empty_cache()
        # Normalize
        proj_target = proj_target_sum / total_samples
        proj_perp = proj_perp_sum / total_samples
        # Compute variances
        var_target = proj_target.var().item()
        var_perp = proj_perp.var().item()
        axo = axes_output[idx//ncols_hist, idx%ncols_hist]
        # Histogram and action plot for target

        # --- Overlay theoretical Gaussian action curves ---
        # Parse config for chi, kappa, epsilon if present
        d_cfg, P_cfg, N_cfg, chi, seed_cfg, T_cfg, epsilon_cfg = parse_config_from_dirname(model_dir)
        # Estimate kappa as 1/chi if not present
        kappa = 4.2
        # Use N as n1
        epsilon = epsilon_cfg if epsilon_cfg is not None else 0.0
        theory = compute_theory_with_julia(d_cfg, N_cfg, P_cfg, chi, kappa, epsilon)
        lJ3T = theory["target"]["lJ3T"]
        lJ3P = theory["perpendicular"]["lJ3P"]

        # Compute histogram and mask before using for theory overlay x-range
        v_flat = proj_target.flatten().cpu()
        hist_range = (v_flat.min().item(), v_flat.max().item())
        bins = 200
        hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
        bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = (hist_range[1] - hist_range[0]) / bins
        probs_density = (hist / hist.sum()) / bin_width
        mask = probs_density > 0
        bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
        mask_cpu = mask.cpu() if mask.is_cuda else mask

        # Plot theoretical Gaussian action for target and perp, using a smooth line over the same x-range as experiment (after mask)
        import numpy as np
        bin_centers_np = bin_centers_cpu[mask_cpu].numpy() if hasattr(bin_centers_cpu, 'numpy') else np.array(bin_centers_cpu)[mask_cpu]
        if len(bin_centers_np) > 1:
            x_min = float(bin_centers_np.min())
            x_max = float(bin_centers_np.max())
        else:
            x_min = float(bin_edges[0])
            x_max = float(bin_edges[-1])
        x_theory = np.linspace(x_min, x_max, 1000)
        for lJ3, color, label in [
            (lJ3T, 'royalblue', 'Theory Target'),
            (lJ3P, 'orange', 'Theory Perp')
        ]:
            var = lJ3
            action = 0.5 * x_theory**2 / var + 0.5 * np.log(2 * np.pi * var)
            axo.plot(x_theory, action, '--', color=color, label=f'{label} $\\sigma^2={var:.2e}$')
        v_flat = proj_target.flatten().cpu()
        hist_range = v_flat.min().item(), v_flat.max().item()
        bins = 200
        hist = torch.histc(v_flat, bins=bins, min=hist_range[0], max=hist_range[1])
        bin_edges = torch.linspace(hist_range[0], hist_range[1], bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = (hist_range[1] - hist_range[0]) / bins
        probs_density = (hist / hist.sum()) / bin_width
        mask = probs_density > 0
        bin_centers_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
        mask_cpu = mask.cpu() if mask.is_cuda else mask
        axo.plot(
            bin_centers_cpu[mask_cpu].numpy(),
            (-probs_density[mask].log()).cpu().numpy(),
            label=f'Target $x_0$ ($\\sigma^2={var_target:.2e}$)',
            color='royalblue'
        )
        # Histogram and action plot for averaged perp
        v_flat_perp = proj_perp.flatten().cpu()
        hist_perp = torch.histc(v_flat_perp, bins=bins, min=hist_range[0], max=hist_range[1])
        probs_perp_density = (hist_perp / hist_perp.sum()) / bin_width
        mask_perp = probs_perp_density > 0
        bin_centers_perp_cpu = bin_centers.cpu() if bin_centers.is_cuda else bin_centers
        mask_perp_cpu = mask_perp.cpu() if mask_perp.is_cuda else mask_perp
        d_val = d if d is not None else 0
        axo.plot(
            bin_centers_perp_cpu[mask_perp_cpu].numpy(),
            (-probs_perp_density[mask_perp].log()).cpu().numpy(),
            label=f'Perp avg $x_{{1...{d_val-1}}}$ ($\\sigma^2={var_perp:.2e}$)',
            color='orange'
        )
        
        # Parse config parameters for title
        d, P, N, chi, seed, T, epsilon = parse_config_from_dirname(model_dir)
        axo.set_title(f"Action: $-\\log P$ | d={d}, P={P}, N={N}, $\\chi$={chi}", fontsize=11)
        axo.set_xlabel('Output projection value')
        axo.set_ylabel('Action: -log P')
        axo.legend()
        axo.grid(True, alpha=0.3)
    
    for j in range(idx+1, nrows_hist*ncols_hist):
        axes_output[j//ncols_hist, j%ncols_hist].axis('off')
    fig_output.tight_layout()
    fig_output.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'output_projection_histograms.png'), dpi=150)
    plt.close(fig_output)
    print('Saved output projection histograms to output_projection_histograms.png')

    # Hide unused subplots
    for j in range(len(param_keys), nrows*ncols):
        axes_scatter[j//ncols, j%ncols].axis('off')
    fig_scatter.tight_layout()
    fig_scatter.savefig(os.path.join('/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel', 'grid_X0_vs_model_output_grouped.png'), dpi=150)
    print('Saved grouped grid scatter plot to grid_X0_vs_model_output_grouped.png')

if __name__ == "__main__":
    main()
