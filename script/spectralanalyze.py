import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
import scipy.stats as stats
sys.path.insert(0, '/home/akiva/FCNX-Ensembling/lib')
from Experiment import Experiment 

import numpy as np


def same_order_mask(a, b, eps=1):
    """
    Returns True where |floor(log10|a|) - floor(log10|b|)| <= eps
    eps=0  → exact same order (your original)
    eps=1  → allows one order of magnitude difference (most common soft version)
    """
    a, b = np.asarray(a), np.asarray(b)
    zero = (a == 0) & (b == 0)
    both_nonzero = (a != 0) & (b != 0)
    exp_a = np.log10(np.abs(a))
    exp_b = np.log10(np.abs(b))
    close = np.abs(exp_a - exp_b) <= eps
    return zero | (both_nonzero & close)
    
experimentMF = Experiment(
        eps=0.03,
        file = '/home/akiva/exp/fcn3erf/erf_cubic_eps_0.03_P_250_D_40_N_400_epochs_25000000_lrA_1.33e-06_time_20251118_155401',
        N=400,
        d=40,
        chi=50,
        P=250,
        ens=3,
        device = torch.device('cpu')
    )
# experimentMF = Experiment(
#         eps=0.00,
#         file = '/home/akiva/exp/fcn3erf/erf_P_30_D_25_N_250_epochs_250000000_lrA_1.11e-07_time_20250922_144935',
#         N=250,
#         d=25,
#         chi=250,
#         P=30,
#         ens=7,
#     )
experimentMF.load()
experimentMF.model.to(experimentMF.device)
experimentMF.predictions = experimentMF.eig_predictions()
X, Y1,Y3 = experimentMF.large_dataset(p_large = 3000, flat=True)
experimentMF.model.device = experimentMF.device
print("Model device is: ", experimentMF.model.device)
# Available CPU memory:
import psutil
mem = psutil.virtual_memory()
print(f"Available memory: {mem.available / (1024**3):.2f} GB")
# Available memory not in use:
print(f"Free memory: {mem.free / (1024**3):.2f} GB")
# compute the svd of the low rank approximaton to H
# Random svd goes from H ∈ ℝ^m*m to its low rank approximation QB ∈ ℝ^m*m with Q ∈ ℝ^m*l and B ∈ ℝ^l*k. 
Q, Z = experimentMF.model.H_random_QB(X, k = 2000, p=10)
Ut, _S, V = torch.linalg.svd(Z.T)
m, n = Z.shape[1], Z.shape[0]
k = min(m, n)
Sigma = torch.zeros(m, n, device=Z.device, dtype=Z.dtype)
Sigma[:k, :k] = torch.diag(_S) 
U = torch.matmul(Q, Ut)
ls = Sigma.diagonal() / X.shape[0]
Y = Y1 / torch.norm(Y1, dim=0) 
left_eigenvalues = (torch.matmul(Y.t(), U) @ _S.diag() @ torch.matmul(U.T, Y)).diagonal() / torch.norm(Y, dim=0)/ X.shape[0]
ls = experimentMF.model.H_eig_random_svd(X, k=2000)

import matplotlib.pyplot as plt
import numpy as np
import torch
MF_eigenvalues = experimentMF.predictions
# ------------------------------------------------------------------
# 1. Choose colormap
# ------------------------------------------------------------------
cmap_name = "viridis"          # or "magma"
cmap = plt.get_cmap(cmap_name)
colors = cmap(np.linspace(0, 1, 4))

# ------------------------------------------------------------------
# 2. Move ls3 to CPU and convert to NumPy early
# ------------------------------------------------------------------
ls3 = ls # assuming ls is your eigenvalue tensor
ls3_np = ls3.detach().cpu().numpy()
eig_upto = ls3.shape[0]  # or set to desired number of eigenvalues



# ------------------------------------------------------------------
# 3. Create boolean masks → **convert to NumPy immediately**
# ------------------------------------------------------------------
idx_big_np = (ls3_np == ls3_np.max()
              ) & (np.arange(len(ls3_np)) < eig_upto)
idx_mid1_np = same_order_mask(experimentMF.predictions.lH1P, ls3_np) & (ls3_np < ls3_np.max())  & (np.arange(len(ls3_np)) < eig_upto)
idx_mid_np = ~idx_mid1_np & same_order_mask(experimentMF.predictions.lH3T, ls3_np, eps=0.1) & (np.arange(len(ls3_np)) < eig_upto)
idx_small_np = same_order_mask(experimentMF.predictions.lH3P, ls3_np, eps=0.1) & (
    np.arange(len(ls3_np)) < eig_upto)

# Count sizes
n_big   = np.sum(idx_big_np)
n_mid1 = np.sum(idx_mid1_np)
n_mid   = np.sum(idx_mid_np)
n_small = np.sum(idx_small_np)

# ------------------------------------------------------------------
# 4. Build bar positions (NumPy)
# ------------------------------------------------------------------
pos_big   = np.arange(0, n_big)
pos_mid1 = np.arange(n_big, n_big + n_mid1)
pos_mid   = np.arange(n_big + n_mid1, n_big + n_mid + n_mid1)
pos_small = np.arange(n_big + n_mid + n_mid1, n_big + n_mid + n_mid1 + n_small)

# ------------------------------------------------------------------
# 5. Horizontal target lines
# ------------------------------------------------------------------
plt.axhline(y=MF_eigenvalues.lH1T, color=colors[0], linestyle='--', label='$\mathbb{E}\;[\lambda^{H1}_T]=$' + f'{MF_eigenvalues.lH1T:.5f}')
plt.axhline(y=MF_eigenvalues.lH1P, color=colors[1], linestyle='-',
            label='$\mathbb{E}\;[\lambda^{H1}_P]=$' + f'{MF_eigenvalues.lH1P:.5f}')
plt.axhline(y=MF_eigenvalues.lH3T, color=colors[2], linestyle='--',
            label='$\mathbb{E}\;[\lambda^{H3}_T]=$' + f'{MF_eigenvalues.lH3T:.5f}')
plt.axhline(y=MF_eigenvalues.lH3P, color=colors[3], linestyle='-',  label='$\mathbb{E}\;[\lambda^{H3}_P]$='
            + f'{MF_eigenvalues.lH3P:.5f}')

# ------------------------------------------------------------------
# 6. Bar plots – **all NumPy**
# ------------------------------------------------------------------
if n_big > 0:
    plt.bar(pos_big, ls3_np[idx_big_np], color=colors[0], label='$\lambda^{H1}_T$')

if n_mid > 0:
    plt.bar(pos_mid, ls3_np[idx_mid_np], color=colors[2], label='$\lambda^{H1}_P$')

if n_mid1 > 0:
    mid1_vals = ls3_np[idx_mid1_np]
    mid1_pos  = pos_mid1

    plt.bar(pos_mid1, mid1_vals, color=colors[1], label='$\lambda^{H3}_T$')

if n_small > 0:
    small_vals = ls3_np[idx_small_np]
    small_pos  = pos_small

    plt.bar(small_pos, small_vals, color=colors[3], label='$\lambda^{H3}_P$')

# ------------------------------------------------------------------
# 7. Finalize
# ------------------------------------------------------------------
plt.title(f"FCN3-Erf on y = He1 + 4.0 He3 Eigenspectrum \n $N=400, P=2500, d=50, \chi=100, \epsilon=4.0$")
plt.yscale('log')
# plt.xscale('log')
plt.xlabel("Eigenvalue Index (log scale)")
plt.ylabel("Magnitude (log scale)")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.show()

project_dir = '/home/akiva/FCNX-Ensembling/plots/'
plt.savefig(project_dir + 'spectral_analysis_fcn3erf_he1_he3.png', dpi=300)