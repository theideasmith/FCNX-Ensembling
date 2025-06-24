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
from opt_einsum import contract

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

DEVICE = 'cpu'

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
            self.W0, X,
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

def processdats(dats):
    ratio_means = {}
    ratio_stds = {}
    
    diff_means = {}
    diff_stds = {}
    
    all_eigs = {}
    
    for d in dats:
        n = d['width']
        km = d['ratio_mean'].tolist()
        ks = d['ratio_std'].tolist()
    
        Dm = d['diff_mean'].tolist()
        Ds = d['diff_std'].tolist()
        if not n in all_eigs:
            all_eigs[n] = d['eig'].tolist()
        else:
            all_eigs[n].extend(d['eig'])
        
        if not n in ratio_means:
            ratio_means[n] = [km]
            ratio_stds[n] = [km]
        else: 
            ratio_means[n].append(km)
            ratio_stds[n].append(ks)
    
        if not n in diff_means:
            diff_means[n] = [Dm]
            diff_stds[n]= [Ds]
        else: 
            diff_means[n].append(Dm)
            diff_stds[n].append(Ds)
    
    def stdstd(xs):
        return np.sqrt(np.mean(np.array(xs)**2))
        
    Ns = np.array([i for i in ratio_means.keys()])
    kkm = [np.mean(i) for k,i in ratio_means.items()]
    kks = [stdstd(i) for k,i in ratio_stds.items()]
    Dkm = np.array([np.mean(i) for k,i in diff_means.items()])
    Dks = np.array([stdstd(i) for k,i in diff_stds.items()])

    return Ns, kkm, kks, Dkm, Dks

def find_net0_files_os_walk(base_dir) -> list[str]:
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
                if filename.startswith("GP_ondata_True_ensparallel"):
                    full_path = os.path.join(root, filename)
                    found_files.append(full_path)

    return found_files


def get_data(d,n,seed):
    np.random.seed(seed)
    X = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n,1,d))).to(dtype=torch.float32)
    return X

train_seed = 563

theory_eigs_path = './theory_eigs.json'

def run(fname):
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
        model = torch.load(nets[i])

        # W: d, N, ensembles
        W = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1))

        d = W.shape[0] # Input dimension
        N = W.shape[1] # Width of W0
        q = W.shape[2] # Number of ensembles
        
        torch.set_printoptions(precision=3, sci_mode=False)

        X = get_data(d, N, train_seed).to('cuda') # this is the train seed used in net.py
        X = X.squeeze()

        P = X.shape[0]

        Wm = torch.mean(W, axis=1)
        Wm2 = torch.einsum('ai,bi->abi', Wm,Wm)
        cov_W = torch.einsum('aji,bji->abi', W, W) / N - Wm2
        # Average over the ensemble dimension
        cov_W_m = torch.mean(cov_W, axis=2)
        cov_W = cov_W_m.cpu().detach().numpy()
        lH = cov_W_m.diagonal().squeeze().cpu().detach().numpy()
        print(f'-- d: {d} --- N: {N} ----- P: {P} ------')

        print("Representative Preactivation Kernel Values")
        print("X[:5,:]")
        print(X[:5,:])
        print("Empirical (ensemble averaged; q=5)")
        print("K_Emp[:5,:5]")
        f = torch.einsum('ui,ijk->ujk', X, W) # P * N * ensembles
        fm = torch.mean(f, dim=1) # P * ensembles

        # Tensor product over the output dimensions
        # and average over the internal neurons
        # and the ensemble dimension
        hh = torch.einsum('uim,vim->uv', f, f)/(N * q)
        print(hh[:5,:5])

        print("Theoretical")
        print("K_theory[:5,:5]")
        real_hh = torch.einsum('ui,vi->uv',X,X)/d
        print(real_hh[:5,:5])

        a0 = hh.flatten()
        b0 = real_hh.flatten()
        krat = (a0/b0).cpu().detach().numpy()
        krat = krat
        kdiff = (a0-b0).cpu().detach().numpy()

        dats.append({'width':N,
                    'ratio_mean': krat.mean(), 
                    'ratio_std': krat.std(),
                    'eig': lH,
                    'diff_mean': kdiff.mean(),
                    'diff_std': kdiff.std()})

    return dats 

def kkrender(dats,ens):
    Ns, kkm, kks, Dkm, Dks = dats
    # plt.figure()
    plt.scatter(np.log10(Ns), np.log10(np.array(kks)**2), label=f'{ens} ensemble')
    plt.xlabel('$\log N$')
    plt.ylabel('$\log\sigma^2_{\hat{K}/K}$')
    plt.title(f'd:3 P:30'+'– $\sigma_{\\hat{K}/K}$ loglog')
    plt.legend()


def dkrender(dats,ens):
    Ns, kkm, kks, Dkm, Dks = dats
    # plt.figure()
    plt.scatter(np.log10(Ns), np.log10(np.array(Dks)**2), label=f'{ens} ensemble')
    plt.xlabel('$\log N$')
    plt.ylabel('$\log\sigma^2_{\hat{K} - K}$')
    plt.title("log $\sigma^2_{\hat{K} - K}$ against log N (width) ")
    plt.legend()

def pl(dats,ens):
        # Create the plot with error bars
    Ns, kkm, kks, Dkm, Dks = dats
    plt.scatter(Ns, Dkm)    
    plt.errorbar(Ns, Dkm, yerr=Dks, fmt='o', capsize=4, label=f'{ens} ensemble')
    
    for i, txt in enumerate(kkm):
        plt.text(Ns[i], Dkm[i]+Dks[i], f'({Ns[i]}, {Dkm[i]:.2f})', color='black', ha='center', va='bottom', fontsize=8)
    # Add labels and title for clarity
    plt.xlabel("N (width)")
    plt.ylabel("Diff $K_{uv}$ Exp-Theory")
    plt.title( f'Ensemble of {len(dats)} FCN2s @ 30/N On Data | d:3 P:30 | '+ '$\hat{K}_{uv}-K^t_{uv}$')
    plt.legend()
    plt.tight_layout()




if __name__=='__main__':
    Menagerie_dir = os.path.join('/home/akiva/gpnettrain')
    dats = run(fname)
    plt.figure()
    kkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_ratiok.png'), dpi=300, bbox_inches='tight')

    plt.figure()
    dkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_diffk.png'), dpi=300, bbox_inches='tight')

    plt.figure()
    pl(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'kernel_discrepancy_scaling')+'.png', dpi=300, bbox_inches='tight')







            
