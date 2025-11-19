import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPKit import GPR, compute_gpr_nngp_torch
from FCN2Network import FCN_2_Ensemble

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

def main():
    N_train = 50
    N_test = 100
    input_dim = 3

    torch.manual_seed(0)
    X_train = torch.randn(N_train, input_dim).to(DEVICE)
    X_test = torch.randn(N_test, input_dim).to(DEVICE)
    # Create a one-hot weight vector w
    w = torch.zeros(input_dim, device=DEVICE)
    w[0] = 1.0  # Set first dimension to 1, others to 0

    def true_func(x):
        return x @ w  # Linear function with one-hot weight
    y_train = true_func(X_train)
    y_test = true_func(X_test)

    # GPR prediction (dot product kernel)
    def dot_product_kernel(X1, X2):
        a = X1 / X1.shape[1]**0.5
        b = X2 / X2.shape[1]**0.5
        return a @ b.T

    sigma_0_sq = 1e-2
    K_xx = dot_product_kernel(X_train, X_train) + sigma_0_sq * torch.eye(N_train, device=DEVICE)
    K_xstar_x = dot_product_kernel(X_test, X_train)
    K_xx_inv = torch.linalg.inv(K_xx)
    gpr_pred = K_xstar_x @ K_xx_inv @ y_train

    # FCN2 Ensemble prediction for various ensemble sizes, in batches
    n_hidden = 1000  # Large width for GP-like behavior
    batch_size = 5000  # Number of networks in each batch
    n_batches = 5      # 5 batches of 5000 = 25000 total
    total_ensembles = batch_size * n_batches
    activation = lambda x: x  # Identity activation for dot product kernel equivalence
    s2W = 1.0
    s2A = 1.0

    all_preds = []
    for batch in range(n_batches):
        print(f"Processing batch {batch+1}/{n_batches}...")
        ensemble = FCN_2_Ensemble(
            d=input_dim,
            n1=n_hidden,
            s2W=s2W,
            s2A=s2A,
            ensembles=batch_size
        ).to(DEVICE)
        with torch.no_grad():
            batch_preds = []
            for i in range(batch_size):
                out = ensemble.forward(X_test)[:, i]  # shape: (N_test, )
                batch_preds.append(out.cpu())  # Move to CPU to save GPU memory
            batch_preds = torch.stack(batch_preds, dim=1)  # shape: (N_test, batch_size)
            all_preds.append(batch_preds)
        del ensemble
        torch.cuda.empty_cache()
    all_preds = torch.cat(all_preds, dim=1)  # shape: (N_test, total_ensembles)

    # Compute MAD for increasing ensemble sizes
    ensemble_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 15000, 20000, 25000]
    mad_values = []
    for nens in ensemble_sizes:
        mean_pred = all_preds[:, :nens].mean(dim=1)
        mad = torch.mean(torch.abs(mean_pred - gpr_pred.cpu().squeeze())).item()
        mad_values.append(mad)
        print(f"Ensemble size: {nens}, MAD: {mad:.4e}")

    # Plot MAD vs. ensemble size (semilog-x and log-log)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Semilog-x plot
    axs[0].plot(ensemble_sizes, mad_values, marker='o')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Number of Ensemble Members (log scale)')
    axs[0].set_ylabel('Mean Absolute Deviation (MAD) vs. GPR')
    axs[0].set_title('MAD vs. Ensemble Size (Semilog-x)')
    axs[0].grid(True, which='both', ls='--', alpha=0.5)

    # Log-log plot
    axs[1].plot(ensemble_sizes, mad_values, marker='o')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Number of Ensemble Members (log scale)')
    axs[1].set_ylabel('MAD vs. GPR (log scale)')
    axs[1].set_title('MAD vs. Ensemble Size (Log-Log)')
    axs[1].grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'fcn2_ensemble_vs_gpr_mad.png'))
    print('MAD plot saved to', os.path.join(os.path.dirname(__file__), 'fcn2_ensemble_vs_gpr_mad.png'))

if __name__ == '__main__':
    main() 