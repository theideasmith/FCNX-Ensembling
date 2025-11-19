"""

THIS FILE IS MEANT TO COMPARE AN FCN2 IN STANDARD SCALING
TO THE GPR

"""
import matplotlib.pyplot as plt
import math
import torch
import sys
import os
import numpy as np
import json
import torch.nn as nn
INIT_SEED = 222
import argparse
from scipy.stats import zscore
import torch

from opt_einsum import contract

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GPKit import *

parser = argparse.ArgumentParser(description="Post computation of FCN2 NN trains in MF and STD scaling")
parser.add_argument('filename', help='The name of the file to process.')
args = parser.parse_args()
fname = args.filename

DEVICE = 'cuda:1'


X_inf = None
P_inf = 10_000
test_seed = 10

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

def find_net0_files_os_walk(base_dir):
    """
    Finds files named 'netnum_0' within subdirectories of a base directory

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

    # Now, walk through each of these specific subdirectories
    for root, _, files in os.walk(fname):
        for filename in files:
            if filename.startswith("netnum_"):
                full_path = os.path.join(root, filename)
                found_files.append(full_path)

    return found_files


def get_data(d,n,seed):
    np.random.seed(seed)
    X = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n,1,d))).to(dtype=torch.float32)
    return X

train_seed = 563

def run(fname):
    Menagerie_dir = os.path.dirname(fname)

    nets = find_net0_files_os_walk(fname)
    nets = sorted(nets, key=lambda x: os.path.getctime(x))
    dats=[]
    """
       dats: [ 
       {N:
        ratios}
    ]
    """
    print(f'{len(nets)} nets identified. Loading now....')
    all_networks_discrepancy_data = []
    empirical_lH_dict = {}  # N -> lH vector
    N_list = []

    for i in range(len(nets)): 
        print(f'Loading net {i} @ {nets[i]}')
        model = torch.load(nets[i]).to(DEVICE)

        # W: d, N, ensembles
        W = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1)).to(DEVICE)

        d = W.shape[0] # Input dimension
        N = W.shape[1] # Width of W0
        q = W.shape[2] # Number of ensembles
        
        torch.set_printoptions(precision=3, sci_mode=False)

        X = get_data(d, 30, train_seed).to(DEVICE) # this is the train seed used in net.py
        X = X.squeeze()

        P = X.shape[0]

        Wm = torch.mean(W, axis=1)
        Wm2 = torch.einsum('ai,bi->abi', Wm,Wm)
        cov_W = torch.einsum('aji,bji->abi', W, W) / N - Wm2
        # Average over the ensemble dimension
        cov_W_m = torch.mean(cov_W, axis=2)
        cov_W = cov_W_m.cpu().detach()
        lH = cov_W_m.diagonal().squeeze().cpu().detach() if hasattr(cov_W_m.diagonal().squeeze(), 'cpu') else cov_W_m.diagonal().squeeze()
        empirical_lH_dict[N] = lH.cpu().detach().numpy()
        N_list.append(N)
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

        # Looking at the performance of this estimator 
        # relative to the NNGP and true GPR

        # I need to compute the empirical eigenvalues using a 
        # finite matrix "operator" projection 
        global X_inf
        if X_inf == None:
            X_inf = get_data(d, P_inf, test_seed).squeeze().to(DEVICE)

        f_inf = torch.einsum('ui,ijk->ujk', X_inf, W) # P * N * ensembles
        fm_inf = torch.mean(torch.mean(f_inf, dim=1), dim=1)

        # Kernel is averaged over ensemble and neuron indices
        hh_inf = torch.einsum('uim,vim->uv', f_inf, f_inf)/(N * q * P_inf) #- torch.einsum('u,v->uv', fm_inf,fm_inf)

        # Large matrix projection 
        Ls = torch.einsum('uj,uv,vj->j', X_inf, hh_inf, X_inf) / P_inf
        norm = torch.einsum('ij,ij->j',X_inf, X_inf) / P_inf
        lsT = Ls

        # breakpoint()


        # After obtaining the an approximation to the eigenvectors
        # through a projection, I am able to obtain an estimate 
        # of the gpr result. 

        # beta = 1.0/d
        true_gpr = gpr_dot_product_explicit(X, X[:,0], X, 1.0).cpu().detach().numpy()
        # X = X.to(torch.float64)
        # Xnp = X.cpu().detach().numpy()
        # K_yy_noisy_np = beta*np.dot(Xnp, Xnp.T) + 1.0 * np.eye(Xnp.shape[0])
        # K_yy_unnoisy_np = beta*np.dot(Xnp, Xnp.T) # This corresponds to K_xstar_x in your PyTorch
        # mu_pred_np = K_yy_unnoisy_np @ np.linalg.inv(K_yy_noisy_np) @ Xnp[:,0]

        # X = X.to(torch.float64)
        # K_xx = beta*X @ X.T + 1.0 * torch.eye(X.shape[0], device=X.device)
        # K_xstar_x = beta* X @ X.T
        # mu_pred_torch = K_xstar_x @ torch.linalg.inv(K_xx) @ X[:,0]
        # print("Mu Pred NP")
        # print(mu_pred_np)
        # print("Mu pred torch")
        # print(mu_pred_torch)
        # print("Actual")
        # print(X[:,0])

        # This is nngp with the eigenvalues obtained via operator projection onto kernel eigenfunctions
        projection_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lsT, 1.0).cpu().detach().numpy()
        
        # NNGP with eigenvalues obtained from the covariance matrix of the weights
        weight_cov_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lH, 1.0).cpu().detach().numpy()

        # NNGP with the exact theoretical NNGP limit eigenvalues (1/d) 
        nngp_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lsT * 0 + 1.0/d,  1.0).cpu().detach().numpy()
        
        # The true target function
        tru_y =  X[:,0].cpu().numpy()


        # Calculate the absolute discrepancy for each approximation against tru_y
        discrepancy_network_output = np.abs(model(X).mean(axis=1).cpu().detach().numpy() - tru_y)
        discrepancy_projection_gpr = np.abs(projection_gpr - true_gpr)
        discrepancy_weight_cov_gpr = np.abs(weight_cov_gpr - true_gpr)
        discrepancy_nngp_gpr = np.abs(nngp_gpr - true_gpr)

        # Group the discrepancies for easier processing
        all_discrepancies = [
            discrepancy_network_output,
            discrepancy_projection_gpr,
            discrepancy_weight_cov_gpr,
            discrepancy_nngp_gpr
        ]

        # Calculate the mean and standard deviation for each set of discrepancies
        means = [np.mean(d) for d in all_discrepancies]
        stds = [np.std(d) for d in all_discrepancies] # Using standard deviation for error bars

        # Define labels for the x-axis
        labels = [
            'Network Output',
            'Projection GPR',
            'Weight Cov. GPR',
            'NNGP Theoretical'
        ]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(labels)) # Positions for the bars

        # Create the bar plot with error bars
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)

        # Configure plot labels and title
        ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y_\mu^\text{GPR}|$')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right") # Rotate labels for better readability
        ax.set_title(f'Mean Discrepancy of GPR Approximations vs. $y(x)$ (N={N},d={d},P=30)')
        ax.yaxis.grid(True) # Add a grid for better readability

        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # To display the plot, uncomment the line below when you run your script:
        # plt.show()
        plt.savefig(os.path.join(Menagerie_dir, fname, f'gpr_discrepancy_P_{P}_N_{N}_d_{d}_chi_1.png'), dpi=300, bbox_inches='tight')


        dats.append({'width':N,
                    'ratio_mean': krat.mean(), 
                    'ratio_std': krat.std(),
                    'eig': lH,
                    'diff_mean': kdiff.mean(),
                    'diff_std': kdiff.std()})
          # Group the discrepancies for easier processing
        current_net_discrepancies = {
            'N': N, # Store N for sorting
            'Network Output': {'mean': np.mean(discrepancy_network_output), 'std': np.std(discrepancy_network_output)},
            'Projection GPR': {'mean': np.mean(discrepancy_projection_gpr), 'std': np.std(discrepancy_projection_gpr)},
            'Weight Cov. GPR': {'mean': np.mean(discrepancy_weight_cov_gpr), 'std': np.std(discrepancy_weight_cov_gpr)},
            'NNGP Theoretical': {'mean': np.mean(discrepancy_nngp_gpr), 'std': np.std(discrepancy_nngp_gpr)},
        }
        all_networks_discrepancy_data.append(current_net_discrepancies)

    # Sort the collected data by N (inner layer width)
    all_networks_discrepancy_data_sorted = sorted(all_networks_discrepancy_data, key=lambda x: x['N'])
    print(all_networks_discrepancy_data_sorted)
    # Prepare data for plotting
    Ns = [d['N'] for d in all_networks_discrepancy_data_sorted]

    # Define the discrepancy types and their corresponding titles for subplots
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
        'NNGP Theoretical': 'Theoretical $\lambda$ NNGP'
    }

    # Create a single figure and a single subplot
    fig, ax = plt.subplots(figsize=(15, 8)) # Adjust figure size as needed

    fig.suptitle('Mean Absolute Discrepancy with GPR $y^\\text{GPR}(x)$ (d=3, P=30)', fontsize=16)
    ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y^\\text{GPR}_\mu|$')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

    # Number of network widths (N values)
    num_Ns = len(Ns)
    # Number of discrepancy types
    num_disc_types = len(discrepancy_types)

    # Calculate bar width and spacing
    bar_width = 0.8 / num_Ns # Width for each individual N bar within a discrepancy type group
    group_spacing = 1.0      # Space between groups of bars (discrepancy types)

    # Positions for the main groups (discrepancy types)
    x_group_positions = np.arange(num_disc_types)

    # Colors for each N to differentiate networks across discrepancy types
    colors = plt.cm.viridis(np.linspace(0, 1, num_Ns))

    # List to store handles and labels for the legend
    legend_handles = []
    legend_labels = []

    nmax = 0.0
    # Iterate through each network (N value) to plot its bars across discrepancy types
    for j, N_val in enumerate(Ns):
        # Calculate the offset for the current N within each group
        # This centers the group of bars around the x_group_positions
        offset = (j - (num_Ns - 1) / 2) * bar_width

        # Collect means and stds for the current N across all discrepancy types
        current_N_means = []
        current_N_stds = []
        for disc_type in discrepancy_types:
            # Find the data for the current N_val in the sorted list
            net_data = next(item for item in all_networks_discrepancy_data_sorted if item["N"] == N_val)
            current_N_means.append(net_data[disc_type]['mean'])
            current_N_stds.append(net_data[disc_type]['std'])
            nmax = np.max([net_data[disc_type]['mean'], nmax]) # Update max for y-limit
        
        # Plot bars for the current N across all discrepancy types
        bars = ax.bar(x_group_positions + offset, current_N_means,
                    yerr=current_N_stds, align='center', alpha=0.7, capsize=5,
                    width=bar_width, color=colors[j], label=f'N={N_val}')
        
        # Add small circle markers for each bar
        for k, bar in enumerate(bars):
            x_val = bar.get_x() + bar.get_width() / 2
            y_val = current_N_means[k] # Use the mean value for the marker position
            
            # Plot a small circle marker at the top of the bar
            ax.plot(x_val, y_val, 'o', color='black', markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=3)


        # Add to legend handles and labels
        legend_handles.append(bars[0])
        legend_labels.append(f'N={N_val}')


    ax.set_ylim(top=nmax*2.0)
    # Set x-axis labels to be the discrepancy types
    ax.set_xticks(x_group_positions)
    ax.set_xticklabels([discrepancy_titles[dt] for dt in discrepancy_types], rotation=45, ha="right")

    # Add a legend for the N values
    ax.legend(handles=legend_handles, labels=legend_labels, title="Network Width (N)", loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # Assuming Menagerie_dir and fname are defined in the broader context
    final_plot_path = os.path.join(Menagerie_dir, fname, 'consolidated_gpr_discrepancy_plot.png')
    print(final_plot_path)
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    # print(f"\nConsolidated plot saved to: {final_plot_path}")
            
    # Only plot the lH comparison if chi != 1
    # Theoretical values from Mathematica output for N=[50, 200, 600, 2000], d=3
    theoretical_Ns = [50, 200, 600, 2000]
    theoretical_perp = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    theoretical_targ = [0.743841, 0.743841, 0.743841, 0.743841]

    colors = plt.cm.viridis(np.linspace(0, 1, len(theoretical_Ns)))
    plt.figure(figsize=(10,6))
    for idx, N in enumerate(theoretical_Ns):
        color = colors[idx]
        # Plot empirical lH for this N (dots)
        if N in empirical_lH_dict:
            lH_vec = empirical_lH_dict[N]
            x_vals = np.arange(1, len(lH_vec)+1)
            # Plot empirical lH as scatter
            plt.scatter(x_vals, lH_vec, color=color, label=f'Empirical lH (N={N})', marker='o')
            # Plot empirical lH as line
            plt.plot(x_vals, lH_vec, color=color, linestyle='-', alpha=0.7)
        # Plot theoretical perp (X) as scatter and line
        plt.scatter([2,3], [theoretical_perp[idx],theoretical_perp[idx]], color=color, marker='x', s=100, label=f'Theory perp (N={N})')
        plt.plot([2,3], [theoretical_perp[idx],theoretical_perp[idx]], color=color, linestyle='--', alpha=0.5)
        # Plot theoretical targ (X) as scatter and line
        plt.scatter([1], [theoretical_targ[idx]], color=color, marker='X', s=100, label=f'Theory targ (N={N})')
        plt.plot([1], [theoretical_targ[idx]], color=color, linestyle=':', alpha=0.5)
    plt.xlabel('Eigenvalue Index (1 to d)')
    plt.ylabel('lH Eigenvalue')
    plt.title('Empirical vs Theoretical lH Eigenvalues for Each N')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(Menagerie_dir, fname, 'lH_empirical_vs_theory_per_N.png'), dpi=300)
    # plt.show()

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

def plot_lH_comparison():
    # Provided data
    data = {"perps":[[3.0303030303030304e-2,0.3333333333333333],[3.0303030303030304e-2,0.3333333333333333],[3.0303030303030304e-2,0.3333333333333333],[3.0303030303030304e-2,0.3333333333333333]],"targ":[[0.8637623643058426,0.34924085250172204],[0.8586076472397874,0.3374244768618376],[0.8573772570623069,0.33470611792696187],[0.8569386080936743,0.33374613819478083]],"params":[[30,50,3],[30,200,3],[30,600,3],[30,2000,3]]}
    
    # Extract values
    params = data['params']
    theoretical_lH = [x[1] for x in data['perps']]
    actual_lH = [x[1] for x in data['targ']]
    labels = [f"P={p},N={n},d={d}" for p,n,d in params]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, theoretical_lH, width, label='Theoretical lH')
    plt.bar(x + width/2, actual_lH, width, label='Empirical lH')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('lH Eigenvalue')
    plt.title('Empirical vs Theoretical lH Eigenvalues')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lH_eigenvalue_comparison.png')
    plt.show()

if __name__=='__main__':
    print("HIO")
    Menagerie_dir = os.path.dirname(fname)
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

    # Optionally call the function here or from main
    # plot_lH_comparison()







            
