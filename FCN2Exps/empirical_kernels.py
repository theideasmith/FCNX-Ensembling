import math
import sys
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn as nn
INIT_SEED = 222
import argparse
from scipy.stats import zscore
import torch
import matplotlib.pyplot as plt # For visualization, not part of the core function


parser = argparse.ArgumentParser(description="Post computation of FCN2 NN trains in MF and STD scaling")
parser.add_argument('filename', help='The name of the file to process.')
args = parser.parse_args()
fname = args.filename
def gpr_dot_product_explicit(train_x, train_y, test_x, noise_var):
    """
    Computes Gaussian Process Regression mean and standard deviation using explicit formulas
    with a Dot Product Kernel.

    Args:
        train_x (torch.Tensor): N_train x D tensor of training input features.
        train_y (torch.Tensor): N_train x 1 tensor of training target values.
        test_x (torch.Tensor): N_test x D tensor of test input features.
        sigma_0_sq (float or torch.Tensor): The sigma_0^2 hyperparameter for the DotProduct kernel.
                                            This is NOT trained, you provide its value.
        noise_var (float or torch.Tensor): The observational noise variance (added to diagonal).
                                           This is NOT trained, you provide its value.

    Returns:
        tuple: (mu_pred, sigma_pred)
            mu_pred (torch.Tensor): N_test x 1 tensor of predicted means.
            sigma_pred (torch.Tensor): N_test x 1 tensor of predicted standard deviations.
    """

    # Ensure hyperparameters are tensors and on the correct device
    # This makes the function flexible to accept floats or pre-existing tensors
    sigma_0_sq = torch.tensor(sigma_0_sq, dtype=torch.float32, device=train_x.device)
    noise_var = torch.tensor(noise_var, dtype=torch.float32, device=train_x.device)

    # 1. Define the DotProduct kernel function (helper within the main function)
    def dot_product_kernel_torch(X1, X2, sigma_0_sq_param):
        """
        Computes the DotProduct kernel matrix K(X1, X2) using PyTorch.
        k(xi, xj) = xi @ xj.T
        """
        return X1 @ X2.T

    K_xx = dot_product_kernel_torch(train_x, train_x) + noise_var * torch.eye(train_x.shape[0], device=train_x.device)

    K_xstar_x = dot_product_kernel_torch(test_x, train_x, sigma_0_sq)

    K_xstar_xstar = dot_product_kernel_torch(test_x, test_x, sigma_0_sq)

    jitter = 1e-6 * torch.eye(train_x.shape[0], device=train_x.device)
    try:
        K_xx_inv = torch.linalg.inv(K_xx + jitter)
    except torch.linalg.LinAlgError as e:
        print(f"Error: K_xx is singular or ill-conditioned even with jitter: {e}")
        raise

    # 4. Predict Mean (mu_pred)
    # Formula: mu_pred = K(X_test, X_train) @ K(X_train, X_train)^-1 @ y_train
    mu_pred = K_xstar_x @ K_xx_inv @ train_y

    return mu_pred

class FCN_2_layers(nn.Module):
      def __init__(self, d,N,init_out,init_hidden,activation, init_seed=None):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(INIT_SEED)
        self.lin1 = nn.Linear(d,N, bias=False)
        self.lin2 = nn.Linear(N,1, bias=False)
        self.activation=activation
        self.N = N
        self.d = d
        #np.random.seed(5)
        nn.init.normal_(self.lin1.weight,0,(init_hidden)**0.5)
        nn.init.normal_(self.lin2.weight,0,(init_out)**0.5)

      def forward(self, x):
        x = self.lin1(x)

        res = self.lin2(torch.flatten(x, start_dim=1))
        return res

class FCN_2_Ensemble(nn.Module):
    def __init__(self, d, n1, s2W, s2A,ensembles=1,init_seed=None):
        super().__init__()        
        if init_seed is not None: 
            torch.manual_seed(INIT_SEED)

        self.arch = [d, n1]
        self.d = d
        self.n1 = n1
        self.W0 = nn.Parameter(torch.normal(mean=0.0,
            std=torch.full((ensembles,n1,d),s2W**0.5)).to(DEVICE),
            requires_grad=True)
        self.A = nn.Parameter(torch.normal(
            mean=0.0,
            std=torch.full((ensembles,n1), s2A**0.5)).to(DEVICE),
            requires_grad=True)

    def forward(self,X):
        """                                                       
                                                                  
        Efficiently computes the outputs of a three layer network 
        using opt_einsum                                          
                                                                  
        f : P*d -> P*e*1                                          
        C1_ui = W1_ijk*x_uk                                       
        C3_ui = A_ij*C2_uij                                       
        """                                                       
        Xp = X.squeeze()
        return contract(                                          
           'ik,ikl,ul->ui',                                  
            self.A, self.W0, Xp,                                         
          backend='torch'                                         
        )                                                         

    def h_activation(self,X):
        return contract(
            'ikl,ul->uik',
            W1, X,
            backend='torch'
        )

def activation(x):
    return x

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#   # I trained one network per set of hyperparameters
#   def find_net0_files_os_walk(parent_dir):
#       found_files = []
#       for root, dir, files in os.walk(parent_dir):
#           for file in files:
#               if file == 'netnum_0':  # Or 'net₀' if that's the exact character
#                   found_files.append(os.path.join(root, file))
#       return found_files

def find_net0_files_os_walk(base_dir: str = "gptnettrain") -> list[str]:
    """
    Finds files named 'netnum_0' within subdirectories of a base directory
    (defaulting to 'gptnettrain') that start with 'network_ensemble_300000'.

    Args:
        base_dir (str): The base directory to start the search from (e.g., 'gptnettrain').

    Returns:
        list[str]: A list of full paths to all found 'netnum_0' files
                   that meet the specified subdirectory condition.
                   Returns an empty list if the base directory does not exist
                   or no such files are found.
    """
    found_files = []
    target_subdir_prefix = fname

    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return []

    # First, identify the direct subdirectories that match the prefix
    qualifying_subdirs = []
    try:
        for entry in os.listdir(base_dir):
            full_path = os.path.join(base_dir, entry)
            if os.path.isdir(full_path) and entry.startswith(target_subdir_prefix):
                qualifying_subdirs.append(full_path)
    except OSError as e:
        print(f"Error accessing directory '{base_dir}': {e}")
        return []

    if not qualifying_subdirs:
        print(f"No subdirectories starting with '{target_subdir_prefix}' found directly under '{base_dir}'.")
        return []

    # Now, walk through each of these specific subdirectories
    for subdir_to_explore in qualifying_subdirs:
        for root, _, files in os.walk(subdir_to_explore):
            for filename in files:
                if filename.startswith("GP_ondata_True_ensparallel_5"):
                    full_path = os.path.join(root, filename)
                    found_files.append(full_path)

    return found_files


def get_data(d,n,seed):
    np.random.seed(seed)
    X = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n,1,d))).to(dtype=torch.float32)
    return X

Ps = [10, 20, 50, 100, 500, 600, 800, 1000, 2000, 4000]
Ns = [int(P * 0.7) for P in Ps]
Ds = [int(H / 3) for H in Ns]



train_seed = 563

theory_eigs_path = './theory_eigs.json'


if __name__ == '__main__':
    Menagerie_dir = os.path.join('/home/akiva/gpnettrain')
    theory = None
    with open(theory_eigs_path,'r') as f:
        theory = json.load(f)
    lH_diffs = []
    nets = find_net0_files_os_walk(Menagerie_dir)
    nets = sorted(nets, key=lambda x: os.path.getctime(x))
    eigs = []
    Kemp_thr_ratios = []
    Kstd = []
    Ns = []

    dats=[]
    """
       dats: [ 
       {N:
        ratios}
    ]


    """
    print(f'{len(nets)} nets identified. Loading now....')
    for i in range(len(nets)): 
        print(f'Loading net {i} @ {nets[i]}')
        n = torch.load(nets[i])

        
        W1 = n.lin1.weight.detach()
        P = 30
        N = W1.shape[0]
        d = W1.shape[1]
#       print(f'EVALUATING: {nets[i]}')

        X = get_data(d, N, train_seed).to('cuda') # this is the train seed used in net.py
        X = X.squeeze()
        W1 = n.lin1.weight.detach()


        print('-----------------------')
#       print('Some representative K(x,x\')')
        f = (X @ W1.T)
        fm = torch.mean(f, dim=1)
        hh = torch.einsum('ji,ki->jk', f, f)/N #- np.einsum('j,k->jk', fm,fm)
        real_hh = torch.einsum('ui,vi->uv',X,X)/d
        torch.set_printoptions(precision=3, sci_mode=False)
        a0 = hh.flatten()
        b0 = real_hh.flatten()
        krat = (a0/b0).cpu().numpy()
        # krat = krat[np.abs(zscore(krat)) < 1.5]
        krat = krat.tolist()
        Ls = torch.einsum('uj,uv,vj->j', X, hh, X) / N
        norm = torch.einsum('ij,ij->j',X, X) / N
        ls = (Ls/norm).cpu().numpy().tolist()
        # eigs.append(ls)

        dats.append({'width':N,'ratio': krat, 'eig': ls})

    

#       theorylHp = float(theory["perps"][i][1])
#       theorylHT = float(theory["targ"][i][1])
#       lH_diff = ls[0] - theorylHp
#       lH_diffs.append(lH_diff)
#       print(f'P: {P}, N:{N}, d:{d}')
#       print(f'Emp   : {np.mean(ls): .4f}, std: {np.std(ls): .4f}')
#       print(f'Theory: {(1/d): .4f}')
#       print(f'FLT   : {theorylHT: .4f}')
#       print(f'FLp   : {theorylHp: .4f}')
#       print(f'Diff  : {lH_diff: .4f}')
#       print("------------------------")
#
        # What do I actually want to compute?
        # The centering of the eigenvalues around 
        # their expected value, as P, N, d grow
#   print(lH_diffs)
    
    kks = {}
    all_eigs = {}
    for d in dats:


        n = d['width']
        kk = d['ratio']
        if not n in all_eigs:
            all_eigs[n] = d['eig']
        else:
            all_eigs[n].extend(d['eig'])
        
        if not n in kks:
            kks[n] = kk

        else: 
            kks[n].extend(kk)

    Ns = []
    kkm = []
    kkstd = []

    eigs_m = []
    eigs_std = []

    for n,l in all_eigs.items():
        eigs_m.append(np.mean(l))
        eigs_std.append(np.std(l))

    for n,kk in kks.items():

        kk = np.array(kk)
        # z_scores = np.abs(zscore(kk))
        # threshold_z = 2.0 # Common threshold: 2.5 or 3 standard deviations
        # kk_filtered = kk[z_scores < threshold_z]

        kkm.append(np.mean(kk))
        kkstd.append(np.std(kk))
        Ns.append(n)
        plt.figure()
        plt.title(f'N: {N} Mean: {np.mean(krat)}, Std: {np.std(krat)}')
        plt.hist(kk, density=True, bins=20)                                   
        plt.savefig(os.path.join(Menagerie_dir, fname, f'{i}_kratio_hist')+'.png', dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.scatter(np.log10(Ns), np.log10(np.array(kkstd)**2))
    plt.xlabel('$\log N$')
    plt.ylabel('$\log\sigma^2$')
    plt.savefig(os.path.join(Menagerie_dir, fname, 'loglogvariance.png'), dpi=300, bbox_inches='tight')

    # Create the plot with error bars
    plt.figure()
    plt.scatter(Ns, kkm)    
    plt.errorbar(Ns, kkm, yerr=kkstd, fmt='o', capsize=4, label='Kernel Ratio')

    for i, txt in enumerate(kkm):
        plt.text(Ns[i], kkm[i]+kkstd[i], f'({Ns[i]}, {kkm[i]:.2f})', color='black', ha='center', va='bottom', fontsize=8)
    # Add labels and title for clarity
    plt.xlabel("N (width)")
    plt.ylabel("Ratio $K_{uv}$ Exp/Theory")
    plt.title( f'Ensemble of {len(nets)} FCN2s @ 30/N On Data | d:3 P:30 | '+ '$\\hat{K}_{uv}/K^\\text{theory}_{uv}$')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(Menagerie_dir, fname, 'kernel_ratio_scaling')+'.png', dpi=300, bbox_inches='tight')
    plt.close()





            
