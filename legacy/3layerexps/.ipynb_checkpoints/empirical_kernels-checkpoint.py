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
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from GPKit import compute_gpr_nngp_torch


parser = argparse.ArgumentParser(description="Post computation of FCN2 NN trains in MF and STD scaling")
parser.add_argument('filename', help='The name of the file to process.')
args = parser.parse_args()
fname = args.filename
def gpr_dot_product_explicit(train_x, train_y, test_x, sigma_0_sq):
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

    D = train_x.shape[1]

    # 1. Define the DotProduct kernel function (helper within the main function)
    def dot_product_kernel_torch(X1, X2):
        """
        Computes the DotProduct kernel matrix K(X1, X2) using PyTorch.
        k(xi, xj) = xi @ xj.T
        """
        return X1 @ X2.T / D

    K_xx = dot_product_kernel_torch(train_x, train_x) + sigma_0_sq * torch.eye(train_x.shape[0], device=train_x.device)

    K_xstar_x = dot_product_kernel_torch(test_x, train_x)

    K_xstar_xstar = dot_product_kernel_torch(test_x, test_x)

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
    Finds files named 'netnum_0' within subdirectories of a base directory.

    Args:
        base_dir (str): The base directory to start the search from (e.g., 'gptnettrain').

    Returns:
        list[str]: A list of full paths to all found 'netnum_0' files
                   that meet the specified subdirectory condition.
                   Returns an empty list if the base directory does not exist
                   or no such files are found.
    """
    found_files = []
    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return []

    # Walk through all subdirectories of base_dir
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.startswith("netnum_"):
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

if __name__ == '__main__':
    Menagerie_dir = fname
    lH_diffs = []
    nets = find_net0_files_os_walk(Menagerie_dir)
    nets = sorted(nets, key=lambda x: os.path.getctime(x))
    eigs = []
    Kemp_thr_ratios = []
    Kstd = []
    Ns = []

    dats=[]
    print(f'{len(nets)} nets identified. Loading now....')
    all_networks_discrepancy_data = []

    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    P_inf = 10_000
    test_seed = 10
    train_seed = 563
    X_inf = None



    for i in range(len(nets[:(7*3)])):
        print(f'Loading net {i} @ {nets[i]}')
        n = torch.load(nets[i]).to(DEVICE)
        W1 = n.lin1.weight.detach().to(DEVICE)
        
        P = 30
        N = W1.shape[0]
        d = W1.shape[1]
        torch.set_printoptions(precision=3, sci_mode=False)

        print(f'Std of a: {n.lin2.weight.std()}, theory: {1.0/N**0.5}')
        print(f'Std of w: {n.lin1.weight.std()}, theory: {1.0/d**0.5}')

        X = get_data(d, P, train_seed).to(DEVICE)
        X = X.squeeze()
        W = W1.T
        Wm = torch.mean(W, axis=1)
        Wm2 = torch.einsum('a,b->ab', Wm,Wm)
        cov_W = torch.einsum('aj,bj->ab', W, W) / N - Wm2

        lH = cov_W.diagonal().cpu().detach().numpy()

        f = (X @ W1.T)

        fm = torch.mean(f, dim=1)
        hh = torch.einsum('ji,ki->jk', f, f)/N
        real_hh = torch.einsum('ui,vi->uv',X,X)/d
        a0 = hh.flatten()
        b0 = real_hh.flatten()
        krat = (a0/b0).cpu().detach().numpy()
        kdiff = (a0-b0).cpu().detach().numpy()
        Ls = torch.einsum('uj,uv,vj->j', X, hh, X) / N
        norm = torch.einsum('ij,ij->j',X, X) / N
        ls = (Ls/norm).cpu().numpy()

        # Operator projection for large P
        if X_inf is None:
            X_inf = get_data(d, P_inf, test_seed).squeeze().to(DEVICE)
        f_inf = (X_inf @ W1.T)
        hh_inf = torch.einsum('ji,ki->jk', f_inf, f_inf)/(N * P_inf)
        Ls_inf = torch.einsum('uj,uv,vj->j', X_inf, hh_inf, X_inf) / P_inf
        norm_inf = torch.einsum('ij,ij->j',X_inf, X_inf) / P_inf
        lsT = (Ls_inf/norm_inf).cpu().numpy()

        true_gpr = gpr_dot_product_explicit(X, X[:,0], X, 1.0).cpu().detach().numpy()
        projection_gpr = compute_gpr_nngp_torch(X, X[:,0], X, torch.tensor(lsT, device=DEVICE), 1.0).cpu().detach().numpy()
        weight_cov_gpr = compute_gpr_nngp_torch(X, X[:,0], X, torch.tensor(lH, device=DEVICE), 1.0).cpu().detach().numpy()
        nngp_gpr = compute_gpr_nngp_torch(X, X[:,0], X, torch.ones_like(torch.tensor(lsT, device=DEVICE)) * (1.0/d), 1.0).cpu().detach().numpy()
        tru_y = X[:,0].cpu().numpy()

        print(f"True GPR: {true_gpr.mean()}, Projection GPR: {projection_gpr.mean()}, Weight Cov. GPR: {weight_cov_gpr.mean()}, NNGP Theoretical: {nngp_gpr.mean()}")
        print(f"Statistics of output - Mean: {n(X).mean(axis=1).mean()}, Std: {n(X).mean(axis=1).std()}")        # Discrepancies
        print(f"Statistics of GPR – Mean: {true_gpr.mean()}, Std: {true_gpr.std()}")

        discrepancy_network_output = np.abs(n(X).mean(axis=1).cpu().detach().numpy() - true_gpr)
        discrepancy_projection_gpr = np.abs(projection_gpr - true_gpr)
        discrepancy_weight_cov_gpr = np.abs(weight_cov_gpr - true_gpr)
        discrepancy_nngp_gpr = np.abs(nngp_gpr - true_gpr)

        all_discrepancies = [
            discrepancy_network_output,
            discrepancy_projection_gpr,
            discrepancy_weight_cov_gpr,
            discrepancy_nngp_gpr
        ]
        means = [np.mean(d) for d in all_discrepancies]
        stds = [np.std(d) for d in all_discrepancies]
        labels = [
            'Network Output',
            'Projection GPR',
            'Weight Cov. GPR',
            'NNGP Theoretical'
        ]
        # fig, ax = plt.subplots(figsize=(10, 6))
        # x_pos = np.arange(len(labels))
        # ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
        # ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y_\mu^\text{GPR}|$')
        # ax.set_xticks(x_pos)
        # ax.set_xticklabels(labels, rotation=45, ha="right")
        # ax.set_title(f'Mean Discrepancy of GPR Approximations vs. $y(x)$ (N={N},d={d},P=30)')
        # ax.yaxis.grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(Menagerie_dir, fname, f'gpr_discrepancy_P_{P}_N_{N}_d_{d}_chi_1.png'), dpi=300, bbox_inches='tight')

        dats.append({'width':N,
                    'ratio_mean': krat.mean(),
                    'ratio_std': krat.std(),
                    'eig': lH,
                    'diff_mean': kdiff.mean(),
                    'diff_std': kdiff.std()})
        current_net_discrepancies = {
            'N': N,
            'Network Output': {'mean': np.mean(discrepancy_network_output), 'std': np.std(discrepancy_network_output)},
            'Projection GPR': {'mean': np.mean(discrepancy_projection_gpr), 'std': np.std(discrepancy_projection_gpr)},
            'Weight Cov. GPR': {'mean': np.mean(discrepancy_weight_cov_gpr), 'std': np.std(discrepancy_weight_cov_gpr)},
            'NNGP Theoretical': {'mean': np.mean(discrepancy_nngp_gpr), 'std': np.std(discrepancy_nngp_gpr)},
        }
        all_networks_discrepancy_data.append(current_net_discrepancies)
    # Plot the distribution of second layer weights for one of the networks
    # (Assume the last trained network is called 'net' or similar; if not, use the last one in a list if available)
    # We'll use matplotlib to plot a histogram of the weights in the second layer

    # If you have a list of networks, e.g., nets, you can pick one, e.g., nets[0] or nets[-1]
    # Here, we assume 'net' is the current network object

    if 'n' in locals():
        
        second_layer_weights = n.lin2.weight.detach().cpu().numpy().flatten()
        mean_w = np.mean(second_layer_weights)
        std_w = np.std(second_layer_weights)
        s = n.lin1.out_features  # output dimension of first layer
        theoretical_std = 1.0 / s**0.5
        x_vals = np.linspace(
            min(second_layer_weights.min(), -4*theoretical_std),
            max(second_layer_weights.max(), 4*theoretical_std),
            200
        )
        from scipy.stats import norm

        plt.figure(figsize=(7, 4))
        # Plot histogram with density=True
        plt.hist(second_layer_weights, bins=30, color='C1', alpha=0.7, density=True, label='Empirical')
        # Plot theoretical normal (mean=0, std=1/s)
        plt.plot(x_vals, norm.pdf(x_vals, loc=0, scale=theoretical_std), 'k--', lw=2, label=f'Normal(0, {theoretical_std})')
        # Plot empirical normal (mean, std)
        plt.plot(x_vals, norm.pdf(x_vals, loc=mean_w, scale=std_w), 'C2', lw=2, label='Empirical Normal')
        plt.title('Distribution of Second Layer Weights (lin2) for One Network')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Menagerie_dir, fname, f'second_layer_weights_histogram_N_{N}_d_{d}.png'), dpi=200)
        plt.close()

    # Consolidated discrepancy plot (now as 4 subplots)
    all_networks_discrepancy_data_sorted = sorted(all_networks_discrepancy_data, key=lambda x: x['N'])
    Ns = [d['N'] for d in all_networks_discrepancy_data_sorted]
    discrepancy_types = [
        'Network Output',
        'Projection GPR',
        'Weight Cov. GPR',
        'NNGP Theoretical'
    ]
    discrepancy_titles = {
        'Network Output': 'Network Output',
        'Projection GPR': 'Projection NNGP',
        'Weight Cov. GPR': 'Weight Covariance NNGP',
        'NNGP Theoretical': 'Theoretical $\\lambda$ NNGP'
    }
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
    fig.suptitle('Mean Absolute Discrepancy with GPR $y^\\text{GPR}(x)$ (d=3, P=30)', fontsize=16)
    for idx, disc_type in enumerate(discrepancy_types):
        # Aggregate means and stds for each unique N, then average over duplicates
        from collections import defaultdict

        means_dict = defaultdict(list)
        stds_dict = defaultdict(list)
        for d in all_networks_discrepancy_data_sorted:
            N_val = d['N']
            means_dict[N_val].append(d[disc_type]['mean'])
            stds_dict[N_val].append(d[disc_type]['std'])
        # Now, for each unique N, take the average of the means and stds
        unique_Ns = sorted(means_dict.keys())
        means = [np.mean(means_dict[N]) for N in unique_Ns]
        stds = [np.mean(stds_dict[N]) for N in unique_Ns]
        Ns = unique_Ns  # Redefine Ns to be the unique, sorted N values
        axes[idx].scatter(Ns, means, color='C0', label='Mean')
        axes[idx].fill_between(Ns, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color='C0', alpha=0.2, label='±1 std')
        axes[idx].set_title(discrepancy_titles[disc_type])
        axes[idx].set_xlabel('N (width)')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_ylim(bottom=0, top=2 * np.max(np.array(means) + np.array(stds)))
        if idx == 0:
            axes[idx].set_ylabel('Mean Absolute Discrepancy')
        axes[idx].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    final_plot_path = os.path.join(Menagerie_dir, fname, 'consolidated_gpr_discrepancy_plot.png')
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')

    plt.figure()
    kkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_ratiok.png'), dpi=300, bbox_inches='tight')

    plt.figure()
    dkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_diffk.png'), dpi=300, bbox_inches='tight')

    plt.figure()
    pl(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'kernel_discrepancy_scaling')+'.png', dpi=300, bbox_inches='tight')



