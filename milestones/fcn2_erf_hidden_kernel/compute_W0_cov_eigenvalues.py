import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

# model_dirs = [
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/PScaledLR/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_2'
# ]

# model_dirs = [
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N800_chi_800.0_lr_0.0003_T_2.0_seed_1',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1600_chi_1600.0_lr_0.0003_T_2.0_seed_1',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N800_chi_800.0_lr_0.0003_T_5.0_seed_1',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N1600_chi_1600.0_lr_0.0003_T_5.0_seed_1'
# ]

# model_dirs = [
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_1e-06_T_2.0_seed_0',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_0',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_1',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_2',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_0',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_1',
#     # '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P600_N1400_chi_1400.0_lr_3e-06_T_2.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P1200_N1600_chi_1600.0_lr_3e-06_T_4.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_0',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_1',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/sum reduction/d150_P3000_N1600_chi_1600.0_lr_3e-06_T_10.0_seed_2',
#     '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42_VALID',
# ]

import os
MODELDIR = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/MiniGrokkingMFScalingResults'
model_dirs = [os.path.join(MODELDIR, d) for d in os.listdir(MODELDIR)]
# Only take directories (not files)
model_dirs = [d for d in model_dirs if os.path.isdir(d)]
model_dirs_A = model_dirs
#model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_80.0_lr_0.0003_T_2.0_seed_42']

# model_dirs_B  =[
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_1',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_2',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1200_N800_chi_80.0_lr_3e-05_T_4.0_seed_3']

# model_dirs_B = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_1_eps_0.0',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.0']
model_dirs = [f'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_{i}_eps_0.03' for i in range(4)]
model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/Chi=10/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/Chi=10/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/Chi=10/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/Chi=10/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_3_eps_0.03'] + \
                ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_3e-05_T_16.0_seed_0_eps_0.03',
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
    if N > 3000:return None, None, None
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
    return model, d, N

def compute_W0_cov_eigenvalues(model, d, N):
    # W0 shape: (ens, N, d)
    W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
    ens = W0.shape[0]
    eigvals_list = []
    for e in range(ens):
        W0_e = W0[e]  # (N, d)
        cov = np.cov(W0_e, rowvar=False)  # (d, d)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_list.append(eigvals)
    eigvals_array = np.array(eigvals_list)  # (ens, d)
    return eigvals_array

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    import matplotlib.pyplot as plt
    n_models = len(model_dirs)
    results = {}
    # Prepare storage for per-model data
    largest_eigs_list = []
    target_weight_list = []
    other_weight_list = []
    model_names = []
    d_values = []  # Store d values for each model
    for model_dir in model_dirs:
        print(f"Loading model from {model_dir}")
        model, d, N = load_model(model_dir, device)
        if model is None:
            continue
        eigvals_array = compute_W0_cov_eigenvalues(model, d, N)
        results[model_dir] = eigvals_array
        print(model.W0.shape)
        largest_eigs = eigvals_array.max(axis=1)
        W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
        target_weight_vals = W0[:,:,0].flatten()
        perp_weight_vals = W0[:,:,1:].flatten() if W0.shape[2] > 1 else []
        largest_eigs_list.append(largest_eigs)
        target_weight_list.append(target_weight_vals)
        other_weight_list.append(perp_weight_vals)
        model_names.append(Path(model_dir).name)
        d_values.append(d)
    np.savez("W0_cov_eigenvalues.npz", **results)
    print("Saved all eigenvalues to W0_cov_eigenvalues.npz")

    # Plot grids for each type
    from scipy.stats import skew, kstest, norm
    import json
    def cumulants(data):
        mean = float(np.mean(data))
        var = float(np.var(data))
        std = float(np.std(data))
        skewness = float(skew(data))
        return {'mean': mean, 'var': var, 'std': std, 'skew': skewness}

    # Compute cumulants for all models
    cumulants_json = {}
    print("\nCumulants Comparison Table:")
    print(f"{'Model':<40} | {'Target Mean':>10} | {'Target Var':>10} | {'Target Skew':>10} | {'Other Mean':>10} | {'Other Var':>10} | {'Other Skew':>10}")
    print('-'*100)
    for i, name in enumerate(model_names):
        target_cum = cumulants(target_weight_list[i])
        other_cum = cumulants(other_weight_list[i]) if len(other_weight_list[i]) > 0 else {'mean': None, 'var': None, 'std': None, 'skew': None}
        cumulants_json[name] = {
            'target': target_cum,
            'other': other_cum
        }
        print(f"{name:<40} | {target_cum['mean']:>10.4g} | {target_cum['var']:>10.4g} | {target_cum['skew']:>10.4g} | "
              f"{other_cum['mean'] if other_cum['mean'] is not None else '':>10} | {other_cum['var'] if other_cum['var'] is not None else '':>10} | {other_cum['skew'] if other_cum['skew'] is not None else '':>10}")
    # Kolmogorov-Smirnov normality test for W0[:,:,0] and W0[:,:,1:] (all nonzero dims combined)
    print("\nKolmogorov-Smirnov Normality Test Results (W0[:,:,0] vs W0[:,:,1:]):")
    print(f"{'Model':<40} | {'W0[:,0] D':>10} | {'W0[:,0] p':>10} | {'W0[:,0] Normal?':>12} | {'W0[:,1:] D':>10} | {'W0[:,1:] p':>10} | {'W0[:,1:] Normal?':>12}")
    print('-'*120)
    alpha = 0.05
    for i, name in enumerate(model_names):
        model_dir = model_dirs[i]
        model, d, N = load_model(model_dir, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if model is None:
            continue
        W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
        # W0[:,:,0]
        data_0 = W0[:,:,0].flatten()
        data_0_std = (data_0 - np.mean(data_0)) / np.std(data_0)
        try:
            D_0, p_0 = kstest(data_0_std, 'norm')
            normal_0 = 'YES' if p_0 > alpha else 'NO'
        except Exception as e:
            D_0, p_0, normal_0 = float('nan'), float('nan'), 'ERR'
        # W0[:,:,1:]
        if W0.shape[2] > 1:
            data_rest = W0[:,:,1:].flatten()
            data_rest_std = (data_rest - np.mean(data_rest)) / np.std(data_rest)
            try:
                D_rest, p_rest = kstest(data_rest_std, 'norm')
                normal_rest = 'YES' if p_rest > alpha else 'NO'
            except Exception as e:
                D_rest, p_rest, normal_rest = float('nan'), float('nan'), 'ERR'
        else:
            D_rest, p_rest, normal_rest = float('nan'), float('nan'), 'N/A'
        print(f"{name:<40} | {D_0:>10.4g} | {p_0:>10.4g} | {normal_0:>12} | {D_rest:>10.4g} | {p_rest:>10.4g} | {normal_rest:>12}")
    with open('W0_cumulants.json', 'w') as f:
        json.dump(cumulants_json, f, indent=2)
    print("Cumulants saved to W0_cumulants.json")

    def plot_grid(data_list, model_names, d_vals, bins, color, xlabel, title, filename):
        n = len(data_list)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        for i, (data, name, d_val) in enumerate(zip(data_list, model_names, d_vals)):
            ax = axes[i//ncols, i%ncols]
            mean = np.mean(data)
            std = np.std(data)
            ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
            ax.set_title(f'{name} (d={d_val})' + '\n $\mu$='+f'{mean:.4g}, $\sigma^2$=' + f'{std**2:.4g}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.4g}')
            ax.axvline(mean + std, color='green', linestyle=':', label=f'+ $\sigma$')
            ax.axvline(mean - std, color='green', linestyle=':', label=f'- $\sigma$')
            ax.legend()
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for j in range(i+1, nrows*ncols):
            axes[j//ncols, j%ncols].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join('.', filename), dpi=150)
        plt.close()
        print(f'Saved grid plot to {filename}')

    def plot_neg_log_prob_grid(data_list, model_names, d_vals, filename):
        """Plot -log probabilities for histograms"""
        n = len(data_list)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        
        for i, (data, name, d_val) in enumerate(zip(data_list, model_names, d_vals)):
            ax = axes[i//ncols, i%ncols]
            
            # Create histogram
            hist, bin_edges = np.histogram(data, density=True)
            bin_widths = np.diff(bin_edges)
            probs = hist * bin_widths  # Convert density to probability
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Filter positive probabilities
            mask = probs > 0
            neg_log_probs = -np.log(probs[mask])
            
            ax.plot(bin_centers[mask], neg_log_probs, linewidth=2)
            ax.set_title(f'{name} (d={d_val})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('-log(P)')
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for j in range(i+1, nrows*ncols):
            axes[j//ncols, j%ncols].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELDIR, filename), dpi=150)
        plt.close()
        print(f'Saved grid plot to {filename}')
    
    def plot_combined_neg_log_prob_grid(target_list, perp_list, model_names, d_vals, filename):
        """Plot -log probabilities for both target and perp histograms on same axes"""
        n = len(target_list)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        
        for i, (target_data, perp_data, name, d_val) in enumerate(zip(target_list, perp_list, model_names, d_vals)):
            ax = axes[i//ncols, i%ncols]
            
            # Target histogram
            hist, bin_edges = np.histogram(target_data, bins=200, density=True)
            bin_widths = np.diff(bin_edges)
            probs = hist * bin_widths
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = probs > 0
            ax.plot(bin_centers[mask], -np.log(probs[mask]), label=f'Target (d={d_val})', color='salmon', linewidth=2)
            
            # Perp histogram
            if len(perp_data) > 0:
                hist, bin_edges = np.histogram(perp_data, bins=200, density=True)
                bin_widths = np.diff(bin_edges)
                probs = hist * bin_widths
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mask = probs > 0
                ax.plot(bin_centers[mask], -np.log(probs[mask]), label=f'Perp (d={d_val})', color='mediumseagreen', linewidth=2)
            
            ax.set_title(f'{name} (d={d_val})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('-log(P)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for j in range(i+1, nrows*ncols):
            axes[j//ncols, j%ncols].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join('.', filename), dpi=150)
        plt.close()
        print(f'Saved grid plot to {filename}')
    
    plt.suptitle('W0 Covariance Largest Eigenvalue and Weight Distributions', fontsize=16)
    plot_grid(largest_eigs_list, model_names, d_values, bins=20, color='skyblue', xlabel='Largest Eigenvalue', title='Largest Eigenvalues', filename='grid_largest_eigenvalue_distribution.png')
    plot_grid(target_weight_list, model_names, d_values, bins=40, color='salmon', xlabel='W0[:, :, 0] Value (Target Weight)', title='Target Weight', filename='grid_target_weight_distribution.png')
    if any(len(ow) > 0 for ow in other_weight_list):
        plot_grid(other_weight_list, model_names, d_values, bins=40, color='mediumseagreen', xlabel='W0[:, :, 1:] Value (Perp Weights)', title='Perp Weights', filename='grid_perp_weight_distribution.png')
    

    # Plot -log probability distributions
    plot_combined_neg_log_prob_grid(target_weight_list, other_weight_list, model_names, d_values, filename='grid_neg_log_prob_combined.png')


if __name__ == "__main__":
    main()
