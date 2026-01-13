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

def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    # Extract seed if present
    seed = int(parts[-1])
 
    return d, P, N, chi, seed

def load_model(model_dir, device):
    d, P, N, chi, seed = parse_config_from_dirname(model_dir)
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt
    # Group models by parameter set (d, P, N, chi)
    from collections import defaultdict
    grouped = defaultdict(list)
    param_labels = {}
    for model_dir in model_dirs:
        d, P, N, chi, seed = parse_config_from_dirname(model_dir)
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
            model, d, P, seed = load_model(model_dir, device)
            if model is None:
                continue
            # Generate dataset X with correct seed
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
        model, d, N, seed = load_model(model_dir, device)
        if model is None:
            continue
        # Generate x from seed
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(1000, d, dtype=torch.float32, device=device)
        # Compute h0_activation (assume shape (N,))
        if hasattr(model, 'h0_activation'):
            with torch.no_grad():
                h0 = model.h0_activation(x)
        else:
            # fallback: use first layer pre-activation if available
            h0 = x[:,0]
        # Hermite3 and linear projections
        x0 = x[:,0]
        h3_raw = x0**3 - 3*x0
        h3 = h3_raw / 6**0.5
        lin = x0 
        # Project h0 onto Hermite3 and linear directions
        proj_h3 = torch.einsum('pqn, p', h0, h3) / (P * N * model.W0.shape[0])
        proj_lin = torch.einsum('pqn, p', h0, lin) / (P * N * model.W0.shape[0])
        vals = torch.stack([proj_h3, proj_lin]).cpu().numpy()
        axh = axes_hist[idx//ncols_hist, idx%ncols_hist]
        # Compute histogram and convert to probabilities, plot -log P as line
        for v, label, color in zip(vals, ['Hermite3', 'Linear'], ['royalblue', 'orange']):
            hist, bin_edges = np.histogram(v, bins=200, density=True)
            bin_widths = np.diff(bin_edges)
            probs = -np.log(hist * bin_widths)  # Convert density to probability
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = probs > 0
            axh.plot(bin_centers[mask], -np.log(probs[mask]), label=label, color=color)
        axh.set_title(Path(model_dir).name)
        axh.set_xlabel('$-\log <|Projection|>$')
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
    for idx, model_dir in enumerate(model_dirs):
        model, d, P, seed = load_model(model_dir, device)
        if model is None:
            continue
        # Generate x from seed
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(1000, d, dtype=torch.float32, device=device)
        
        # Compute model output
        with torch.no_grad():
            output = model.forward(x)  # shape: (P, ens) or (P,)
            if output.ndim == 2:
                output = output.mean(dim=1)  # average over ensemble: (P,)
        
        # Target and perpendicular directions
        x0_target = x[:, 0]
        x3_perp = x[:, 3] if d > 3 else torch.randn_like(x[:, 0])
        
        # Project output onto target and perp directions
        # Normalize by dataset size
        proj_target = torch.einsum('p, p -> ', output, x0_target) / P
        proj_perp = torch.einsum('p, p -> ', output, x3_perp) / P
        
        # For histogram, we need distributions - compute projections per sample
        proj_target_samples = (output * x0_target).cpu().numpy()
        proj_perp_samples = (output * x3_perp).cpu().numpy()
        
        axo = axes_output[idx//ncols_hist, idx%ncols_hist]
        # Compute histogram and convert to action plot
        for samples, label, color in zip(
            [proj_target_samples, proj_perp_samples],
            ['Target (x[:,0])', 'Perp (x[:,3])'],
            ['royalblue', 'orange']
        ):
            hist, bin_edges = np.histogram(samples, bins=200, density=True)
            bin_widths = np.diff(bin_edges)
            probs = hist * bin_widths
            # Avoid log(0)
            probs = np.clip(probs, 1e-12, None)
            action = -np.log(probs)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = np.isfinite(action) & (action > 0)
            axo.plot(bin_centers[mask], action[mask], label=label, color=color)
        
        axo.set_title(Path(model_dir).name)
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
