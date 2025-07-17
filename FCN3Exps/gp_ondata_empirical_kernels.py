"""

THIS FILE IS MEANT TO COMPARE AN FCN3 IN STANDARD SCALING
TO THE GPR

"""
import warnings
warnings.filterwarnings("ignore")
import re

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

parser = argparse.ArgumentParser(description="Post computation of FCN3 NN trains in MF and STD scaling")
parser.add_argument('filename', help='The name of the file to process.')
args = parser.parse_args()
fname = args.filename

DEVICE = 'cuda:1'


X_inf = None
P_inf = 5_000
test_seed = 10

class FCN3NetworkEnsembleLinear(nn.Module):

    def __init__(self, d, n1, n2,ensembles=1, weight_initialization_variance=(1.0, 1.0, 1.0)):
        super().__init__()

        self.arch = [d, n1, n2]
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.W0 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n1, d), weight_initialization_variance[0]**0.5)).to(DEVICE),
                                            requires_grad=True) # requires_grad moved here
        self.W1 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n2, n1), weight_initialization_variance[1]**0.5)).to(DEVICE),
                                            requires_grad=True).to(DEVICE) # requires_grad moved here
        self.A = nn.Parameter(torch.normal(mean=0.0, 
                                           std=torch.full((ensembles, 1, n2), weight_initialization_variance[2]**0.5)).to(DEVICE),
                                           requires_grad=True).to(DEVICE) # requires_grad moved here


    def h1_activation(self, X):
        
        return contract(
            'ijk,ikl,ul->uij',
            self.W1, self.W0, X,
            backend='torch'
        )

    def h0_activation(self, X):
        return contract(
            'ikl,unl->uik',
            self.W0, X,
            backend='torch'
        )


    def forward(self, X):
        """

        Efficiently computes the outputs of a three layer network
        using opt_einsum

        f : P*d -> P*e*1
        C1_ui = W1_ijk*x_uk
        C2_uij = W2_ijk*C1_uik
        C3_ui = A_ij*C2_uij
        """
        A = self.A
        W1 = self.W1
        W0 = self.W0

        return contract(
            'eij,ejk,ekl,ul->uie',
            A, W1, W0, X,
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
            if filename.startswith("netnum_") or filename.startswith("model"):
                full_path = os.path.join(root, filename)
                found_files.append(full_path)

    return found_files


def get_data(d,n,seed):
    np.random.seed(seed)
    X = torch.tensor(np.random.normal(loc=0,scale=1.0,size=(n,1,d))).to(dtype=torch.float32)
    return X.to(dtype=torch.float64)

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
    # print(f'{len(nets)} nets identified. Loading now....')
    all_networks_discrepancy_data = []
    empirical_lH_dict = {}  # N -> lH vector
    empirical_lH_list = []
    N_list = []
    d = None

    for i, ntname in enumerate(nets): 
        print(f'Loading net {i} @ {ntname}')
        # print(nets[i]) /
        try:
            model = torch.load(ntname).to(DEVICE)
        except AttributeError as e:
            print(f"Direct torch.load failed for {ntname}: {e}")
            # Try loading as a state_dict
            try:
                checkpoint = torch.load(ntname, map_location=DEVICE)
                # You must define the model class and instantiate it with correct args before loading state_dict
                # Here, we try to infer d, N, q from the checkpoint if possible, else you may need to set them manually
                # For this context, we try to infer from file name or elsewhere
                # Example: FCN3NetworkEnsembleLinear(d, N, N, ens=q)
                # We'll try to extract d, N, q from the checkpoint or filename
                # Fallback: d, N, q = 20, 1500, 5 (example defaults)
                d, N, q = None, None, None
                # Try to infer from checkpoint
                for k, v in checkpoint.items():
                    if k.endswith("W0"):
                        # v shape: (q, N, d)
                        q, N, d = v.shape
                        break
                if d is None or N is None or q is None:
                    # fallback: try to parse from filename or set manually
                    print("Could not infer d, N, q from checkpoint; using defaults d=20, N=1500, q=5")
                    d, N, q = 20, 1500, 5
                model = FCN3NetworkEnsembleLinear(d, N, N, ensembles=q)
                # Check if keys need adaptation (PyTorch 2.0+ compiled models may have _orig_mod. prefix)
                needs_adaptation = False
                model_keys = list(model.state_dict().keys())
                checkpoint_keys = list(checkpoint.keys())
                if len(model_keys) != len(checkpoint_keys):
                    needs_adaptation = True
                else:
                    for mk, ck in zip(model_keys, checkpoint_keys):
                        if mk != ck:
                            needs_adaptation = True
                            break
                if needs_adaptation:
                    print("Adapting state_dict keys for loaded model (assigning by order)...")
                    new_state_dict = {}
                    for i, mk in enumerate(model_keys):
                        if i < len(checkpoint_keys):
                            new_state_dict[mk] = checkpoint[checkpoint_keys[i]]
                        else:

                            print(f"Warning: checkpoint missing key for model param {mk}")
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(checkpoint)
                model = model.to(DEVICE)
            except Exception as e2:
                print(f"Failed to load model as state_dict for {ntname}: {e2}")
                raise e2

        # W: d, N, ensembles
        W = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1)).to(DEVICE)

        d = W.shape[0] # Input dimension
        N = W.shape[1] # Width of W0
        q = W.shape[2] # Number of ensembles
        ens = q
        # breakpoint()
        torch.set_printoptions(precision=6, sci_mode=False)
        P = 20
        X = get_data(d, P, train_seed).to(DEVICE) # this is the train seed used in net.py
        X = X.squeeze().to(dtype = torch.float64)

 

        # W: d, N, ensembles
        W0 = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1))
        W1 = model.W1.permute(*torch.arange(model.W1.ndim - 1, -1, -1))

        # print(f"Layer size: {N}")
        # print(f"Should have std W0: {1.0/np.sqrt(d)} and std W1: {1.0/np.sqrt(N)}")
        # print(f"But actually torch.std(W0): {torch.std(W0)} and torch.std(W1): {torch.std(W1)}")


        # print(f"Should have std A: {1.0/(np.sqrt(N))}")
        # print(f"But actually torch.std(model.A): {torch.std(model.A)}")


        covW0W1 = contract('kje,ije,nme,kme->ine', W1,W0,W0,W1, backend='torch') / N
        # covW0W1 = contract('abl,ial,nal,abe->ine', W1,W0,W0,W1, backend='torch') / N
        # Average over the ensemble dimension
        cov_W_m = torch.mean(covW0W1, axis=2)
        # print(cov_W_m.shape)
        lH = cov_W_m.diagonal().squeeze()
       

        N_list.append(N)


        # breakpoint()
        # print(f'-- d: {d} --- N: {N} ----- P: {P} ------')

        # print("Representative Preactivation Kernel Values")
        # print("X[:5,:]")
        # print(X[:3,:])
        # print("Empirical (ensemble averaged; q=5)")
        # print("K_Emp[:5,:5]")

        X = X.to(dtype = model.A.dtype)
        f = model.h1_activation(X)

        # Tensor product over the output dimensions
        # and average over the internal neurons
        # and the ensemble dimension
        hh = torch.einsum('uim,vim->uv', f, f)/(N * q)
        # print(hh[:5,:5])
        # print("Theoretical")
        # print("K_theory[:5,:5]")
        real_hh = torch.einsum('ui,vi->uv',X,X)/d
        # print(real_hh[:5,:5])

        a0 = hh.flatten()
        b0 = real_hh.flatten()
        krat = (a0/b0).cpu().detach().numpy()
        krat = krat
        kdiff = (a0-b0).cpu().detach().numpy()

        # Looking at the performance of this estimator 
        # relative to the NNGP and true GPR

        # I need to compute the empirical eigenvalues using a 
        # finite matrix "operator" projection 


        X_inf = get_data(d, P_inf, test_seed).squeeze().to(DEVICE, dtype=model.A.dtype) 

        f_inf = model.h1_activation(X_inf)# P * N * ensembles
        f_inf_A = model(X_inf)
        # breakpoint()

        # Kernel is averaged over ensemble and neuron indices
        hh_inf = torch.einsum('uim,vim->uv', f_inf, f_inf)/(N * q * P_inf) #- torch.einsum('u,v->uv', fm_inf,fm_inf)

        # Large matrix projection 
        Ls = torch.einsum('uj,uv,vj->j', X_inf, hh_inf, X_inf) / P_inf
        norm = torch.einsum('ij,ij->j',X_inf, X_inf) / P_inf
        lsT = Ls
        # print(norm)
        # print("Eigenvalues")   
        print(f"Projection Eigenvalues: {Ls/norm}")
        print(f"Weight Covariance Eigenvalues: {lH}")
        lH_proj = Ls/norm

        netname = os.path.basename(os.path.dirname(os.path.dirname(ntname)))

        empirical_lH_dict[N] = {'l': lH.cpu().detach().numpy(), 'd': d, 'N': N, 'P': P, 'netname':netname }
        empirical_lH_list.append({'l': lH.cpu().detach().numpy(), 'd': d, 'N': N, 'P': P, 'netname':netname})
        # print(f"Theoretical Eigenvalues: {lsT * 0 + 1.0/d}")
        # breakpoint()


        # After obtaining the an approximation to the eigenvectors
        # through a projection, I am able to obtain an estimate 
        # of the gpr result. 

        # beta = 1.0/d
        Y = X[:,0] 
        true_gpr = gpr_dot_product_explicit(X, Y, X, 1.0).cpu().detach().numpy()
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
        projection_gpr = compute_gpr_nngp_torch(X, Y, X, lsT, 1.0).cpu().detach().numpy()
        
        # NNGP with eigenvalues obtained from the covariance matrix of the weights
        weight_cov_gpr = compute_gpr_nngp_torch(X, Y, X, lH, 1.0).cpu().detach().numpy()

        # NNGP with the exact theoretical NNGP limit eigenvalues (1/d) 
        nngp_gpr = compute_gpr_nngp_torch(X, Y, X, lsT * 0 + 1.0/d,  1.0).cpu().detach().numpy()
        
        # The true target function
        tru_y =  Y.cpu().numpy() 


        # Calculate the absolute discrepancy for each approximation against tru_y
        discrepancy_network_output = np.abs(model(X ).squeeze().mean(axis=1).cpu().detach().numpy() - true_gpr)
        discrepancy_projection_gpr = np.abs(projection_gpr - true_gpr)
        discrepancy_weight_cov_gpr = np.abs(weight_cov_gpr - true_gpr)
        discrepancy_nngp_gpr = np.abs(nngp_gpr - true_gpr)
        discrepancy_tru_y = np.abs(tru_y - true_gpr)
        # Group the discrepancies for easier processing
        all_discrepancies = [
            discrepancy_network_output,
            discrepancy_projection_gpr,
            discrepancy_weight_cov_gpr,
            discrepancy_nngp_gpr,
            discrepancy_tru_y,
        ]

        # Plot per-x GPR vs. model output for comparison, and include tru_y as well
        # Original plot: Per-x GPR vs. Model Output vs. True y(x)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        x_indices = np.arange(len(tru_y))
        model_outputs = model(X).squeeze().mean(axis=1).cpu().detach().numpy()
        ax2.plot(x_indices[:30], true_gpr[:30], label='GPR', marker='o')
        ax2.plot(x_indices[:30], model_outputs[:30], label='Model Output', marker='x')
        ax2.plot(x_indices[:30], tru_y[:30], label='True $y(x)$', marker='s')
        ax2.set_xlabel('Data Index')
        ax2.set_ylabel('Output Value')
        ax2.set_title(f'Per-x GPR vs. Model Output vs. True $y(x)$ ({netname})')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        # To display the plot, uncomment the line below:
        # plt.show()
        path = os.path.join(Menagerie_dir, fname, os.path.basename(os.path.dirname(ntname)) + '_gpr_vs_model.png')

        print(f'Saving plot to {path}')
        plt.savefig(path, dpi=300, bbox_inches='tight')

        # New plot: Absolute deviation between tru_y and gpr, and model and gpr
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        gpr_model = np.abs(true_gpr - model_outputs)
        tru_y_model = np.abs(model_outputs - tru_y)
        ax3.plot(x_indices, tru_y_model, label=f'|True $y(x)$ - Model Output|: mean:{tru_y_model.mean():.3f} std:{tru_y_model.std():.3f}', marker='s')
        ax3.plot(x_indices, gpr_model, label=f'|Model Output - GPR|: mean:{gpr_model.mean():.3f} std:{gpr_model.std():.3f}', marker='x')
        ax3.set_xlabel('Data Index')
        ax3.set_ylabel('Absolute Deviation')
        ax3.set_title(f'Absolute Deviation: True $y(x)$ vs. GPR and Model Output vs. GPR ({netname})')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        absdev_path = os.path.join(Menagerie_dir, fname, os.path.basename(os.path.dirname(ntname)) + '_absdev_gpr_vs_model.png')
        print(f'Saving absolute deviation plot to {absdev_path}')
        plt.savefig(absdev_path, dpi=300, bbox_inches='tight')

        # Calculate the mean and standard deviation for each set of discrepancies
        means = [np.mean(d) for d in all_discrepancies]
        stds = [np.std(d) for d in all_discrepancies] # Using standard deviation for error bars

        # Define labels for the x-axis
        labels = [
            'Network Output',
            'Projection GPR',
            'Weight Cov. GPR',
            'NNGP Theoretical',
            'True Target'
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
        path = os.path.join(Menagerie_dir, fname, os.path.dirname(os.path.dirname(ntname)) + '.png')
        print(f'Saving MAD discrepancy plot to {path}')
        plt.savefig(path, dpi=300, bbox_inches='tight')


        dats.append({'width':N,
                    'ratio_mean': krat.mean(), 
                    'ratio_std': krat.std(),
                    'eig': lH,
                    'diff_mean': kdiff.mean(),
                    'diff_std': kdiff.std()})
          # Group the discrepancies for easier processing
        current_net_discrepancies = {
            'N': N, # Store N for sorting,
            'P': P,
            'netname': netname,
            'Network Output': {'mean': np.mean(discrepancy_network_output), 'std': np.std(discrepancy_network_output)},
            'Projection GPR': {'mean': np.mean(discrepancy_projection_gpr), 'std': np.std(discrepancy_projection_gpr)},
            'Weight Cov. GPR': {'mean': np.mean(discrepancy_weight_cov_gpr), 'std': np.std(discrepancy_weight_cov_gpr)},
            'NNGP Theoretical': {'mean': np.mean(discrepancy_nngp_gpr), 'std': np.std(discrepancy_nngp_gpr)},
            'True Target': {'mean': np.mean(discrepancy_tru_y), 'std': np.std(discrepancy_tru_y)},
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
        'NNGP Theoretical',
        'True Target'
    ]

    discrepancy_titles = {
        'Network Output': 'Network Output',
        'Projection GPR': 'Projection NNGP',
        'Weight Cov. GPR': 'Weight Covariance NNGP',
        'NNGP Theoretical': 'Theoretical $\lambda$ NNGP',
        'True Target': 'True Target'
    }

    # Create a single figure and a single subplot
    fig, ax = plt.subplots(figsize=(15, 8)) # Adjust figure size as needed

    fig.suptitle('Mean Absolute Discrepancy with GPR $y^\\text{GPR}(x)$ (d=3, P=30)', fontsize=16)
    ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y^\\text{GPR}_\mu|$')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

    # Number of network widths (N values)
    num_Ns = len(empirical_lH_list)
    # Number of discrepancy types
    num_disc_types = len(discrepancy_types)

    # Calculate bar width and spacing
    bar_width = 0.8 / num_Ns # Width for each individual N bar within a discrepancy type group
    group_spacing = 1.0      # Space between groups of bars (discrepancy types)

    # Positions for the main groups (discrepancy types)
    x_group_positions = np.arange(num_disc_types)

    # Colors for each N to differentiate networks across discrepancy types
    colors = plt.cm.viridis(np.linspace(0, 1, len(empirical_lH_list)))

    # List to store handles and labels for the legend
    legend_handles = []
    legend_labels = []

    nmax = 0.0
    # Iterate through each network (N value) to plot its bars across discrepancy types
    for j, lh in enumerate(empirical_lH_list):
        # Calculate the offset for the current N within each group
        # This centers the group of bars around the x_group_positions
        offset = (j - (num_Ns - 1) / 2) * bar_width

        # Collect means and stds for the current N across all discrepancy types
        current_N_means = []
        current_N_stds = []
        for disc_type in discrepancy_types:
            # Find the data for the current N_val in the sorted list
            net_data = next(item for item in all_networks_discrepancy_data_sorted if item["netname"] == lh['netname'])
            current_N_means.append(net_data[disc_type]['mean'])
            current_N_stds.append(net_data[disc_type]['std'])
            nmax = np.max([net_data[disc_type]['mean'], nmax]) # Update max for y-limit
        
        # Plot bars for the current N across all discrepancy types
        bars = ax.bar(x_group_positions + offset, current_N_means,
                    yerr=current_N_stds, align='center', alpha=0.7, capsize=5,
                    width=bar_width, color=colors[j], label=f'N={lh["netname"]}')


        # Add small circle markers for each bar and annotate with idx above
        for k, bar in enumerate(bars):
            x_val = bar.get_x() + bar.get_width() / 2
            y_val = current_N_means[k] # Use the mean value for the marker position
            
            # Plot a small circle marker at the top of the bar
            ax.plot(x_val, y_val, 'o', color='black', markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            
            # Annotate the marker with its index above the circle
            ax.annotate(
                str(j),
                (x_val, y_val),
                textcoords="offset points",
                xytext=(0, 8),  # 8 points above the marker
                ha='center',
                va='bottom',
                fontsize=9,
                color='black',
                fontweight='bold'
            )

        # Add to legend handles and labels
        legend_handles.append(bars[0])
        legend_labels.append(f'{j}: N={lh["netname"]}'.replace('_', ' '))


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
    theoretical_Ns = empirical_lH_dict.keys()
    
    # theoretical_perp = {50: 0.33333333333333354, 200: 0.33333333333333326, 600: 0.3333333333333333, 1000: 0.3333333333333333}# {50: 0.3333333333333333, 200: 0.3333333333333333, 600: 0.3333333333333333, 1000: 0.3333333333333333}
    # theoretical_targ = {50: 0.7314341803167995, 200: 0.7407831191433091, 600: 0.7428251116358064, 1000: 0.7432320110787786}#{50: 0.7314, 200: 0.7407, 600: 0.7428, 1000: 0.7435}
  # theoretical_targ = theoretical_perp
    theoretical_perp = {N: 1.0/d for N in theoretical_Ns}
    # theoretical_targ = {600: 0.363}
    theoretical_targ =  theoretical_perp
    colors = plt.cm.viridis(np.linspace(0, 1, len(empirical_lH_list)))
    plt.figure(figsize=(10,6))
    for idx, lh in enumerate(empirical_lH_list):
        color = colors[idx]
        # Plot empirical lH for this N (dots)

        lH_vec = lh['l']
        # print(f"Printing lH_vec: {lH_vec}")
        x_vals = np.arange(1, len(lH_vec)+1)
        # Plot empirical lH as scatter
        netname = os.path.basename(os.path.dirname(os.path.dirname(nets[idx])))
        plt.scatter(x_vals, lH_vec, color=color, label=f'idx: {idx} lH: {netname}'.replace('_', ' '), marker='o')
        # Plot empirical lH as line
        plt.plot(x_vals, lH_vec, color=color, linestyle='-', alpha=0.7)
    # Plot theoretical perp (X) as scatter and line
        perp_x = range(2,lh['d'])
        perp_y = theoretical_perp[lh['N']] * np.ones(len(perp_x))
        plt.scatter(perp_x, perp_y, color=color, marker='x', s=100, label=f'Theory perp (N={lh["N"]})')
        plt.plot(perp_x, perp_y, color=color, linestyle='--', alpha=0.5)
        # Plot theoretical targ (X) as scatter and
        # 
        target_x = [1]
        target_y = theoretical_targ[lh['N']]  
        plt.scatter(target_x, target_y, color=color, marker='X', s=100, label=f'Theory targ (N={lh["N"]})')
        plt.plot(target_x, target_y, color=color, linestyle=':', alpha=0.5)
        plt.annotate(f'l: {idx}', (x_vals[0], lH_vec[0]),  # Coordinates of the point to annotate
             textcoords="offset points", # How to position the text
             xytext=(0.1, 0.1),           # Distance from text to point (x,y)
             ha='left',                 # Horizontal alignment
             color='blue')
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
    plt.title( f'Ensemble of {len(dats)} FCN3s @ 30/N On Data | d:3 P:30 | '+ '$\hat{K}_{uv}-K^t_{uv}$')
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







            
