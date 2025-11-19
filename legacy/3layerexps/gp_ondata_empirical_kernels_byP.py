"""

THIS FILE IS MEANT TO COMPARE AN FCN3 IN STANDARD SCALING
TO THE GPR, GROUPING BY P INSTEAD OF N

"""
import warnings
warnings.filterwarnings("ignore")

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
                                            requires_grad=True)
        self.W1 = nn.Parameter(torch.normal(mean=0.0, 
                                            std=torch.full((ensembles, n2, n1), weight_initialization_variance[1]**0.5)).to(DEVICE),
                                            requires_grad=True)
        self.A = nn.Parameter(torch.normal(mean=0.0, 
                                           std=torch.full((ensembles, n2), weight_initialization_variance[2]**0.5)).to(DEVICE),
                                           requires_grad=True)
    def h1_activation(self, X):
        try:
            return contract(
                'ijk,ikl,ul->uij',
                self.W1, self.W0, X,
                backend='torch'
            )
        except Exception as e:
            import pdb; pdb.set_trace()
            raise e
    def h0_activation(self, X):
        return contract(
            'ikl,unl->uik',
            self.W0, X,
            backend='torch'
        )
    def forward(self, X):
        A = self.A
        W1 = self.W1
        W0 = self.W0
        return contract(
            'ij,ijk,ikl,ul->ui',
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
        p = d['P']
        km = d['ratio_mean'].tolist()
        ks = d['ratio_std'].tolist()
        Dm = d['diff_mean'].tolist()
        Ds = d['diff_std'].tolist()
        if not p in all_eigs:
            all_eigs[p] = d['eig'].tolist()
        else:
            all_eigs[p].extend(d['eig'])
        if not p in ratio_means:
            ratio_means[p] = [km]
            ratio_stds[p] = [ks]
        else: 
            ratio_means[p].append(km)
            ratio_stds[p].append(ks)
        if not p in diff_means:
            diff_means[p] = [Dm]
            diff_stds[p]= [Ds]
        else: 
            diff_means[p].append(Dm)
            diff_stds[p].append(Ds)
    def stdstd(xs):
        return np.sqrt(np.mean(np.array(xs)**2))
    Ps = np.array([i for i in ratio_means.keys()])
    kkm = [np.mean(i) for k,i in ratio_means.items()]
    kks = [stdstd(i) for k,i in ratio_stds.items()]
    Dkm = np.array([np.mean(i) for k,i in diff_means.items()])
    Dks = np.array([stdstd(i) for k,i in diff_stds.items()])
    return Ps, kkm, kks, Dkm, Dks
def find_net0_files_os_walk(base_dir):
    found_files = []
    target_subdir_prefix = fname
    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return []
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
    all_networks_discrepancy_data = []
    empirical_lH_dict = {}  # P -> lH vector
    P_list = []
    d = None
    I_list = [1, 4, 8, 16, 40]
    for i in range(len(nets)):
        print(f'Loading net {i} @ {nets[i]}')
        model = torch.load(nets[i]).to(DEVICE)
        W = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1)).to(DEVICE)
        d = W.shape[0]
        N = W.shape[1]
        q = W.shape[2]
        torch.set_printoptions(precision=6, sci_mode=False)
        P = 50 * I_list[i]
        X = get_data(d, P, train_seed).to(DEVICE)
        X = X.squeeze()
        W0 = model.W0.permute(*torch.arange(model.W0.ndim - 1, -1, -1))
        W1 = model.W1.permute(*torch.arange(model.W1.ndim - 1, -1, -1))
        covW0W1 = contract('kje,ije,nme,kme->ine', W1,W0,W0,W1, backend='torch') / N
        cov_W_m = torch.mean(covW0W1, axis=2)
        lH = cov_W_m.diagonal().squeeze()
        empirical_lH_dict[P] = lH.cpu().detach().numpy()
        P_list.append(P)
        f = model.h1_activation(X)
        hh = torch.einsum('uim,vim->uv', f, f)/(N * q)
        real_hh = torch.einsum('ui,vi->uv',X,X)/d
        a0 = hh.flatten()
        b0 = real_hh.flatten()
        krat = (a0/b0).cpu().detach().numpy()
        krat = krat
        kdiff = (a0-b0).cpu().detach().numpy()
        X_inf = get_data(d, P_inf, test_seed).squeeze().to(DEVICE)
        f_inf = model.h1_activation(X_inf)
        f_inf_A = model(X_inf)
        hh_inf = torch.einsum('uim,vim->uv', f_inf, f_inf)/(N * q * P_inf)
        Ls = torch.einsum('uj,uv,vj->j', X_inf, hh_inf, X_inf) / P_inf
        norm = torch.einsum('ij,ij->j',X_inf, X_inf) / P_inf
        lsT = Ls
        true_gpr = gpr_dot_product_explicit(X, X[:,0], X, 1.0).cpu().detach().numpy()
        projection_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lsT, 1.0).cpu().detach().numpy()
        weight_cov_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lH, 1.0).cpu().detach().numpy()
        nngp_gpr = compute_gpr_nngp_torch(X, X[:,0], X, lsT * 0 + 1.0/d,  1.0).cpu().detach().numpy()
        tru_y =  X[:,0].cpu().numpy()
        discrepancy_network_output = np.abs(model(X).mean(axis=1).cpu().detach().numpy() - tru_y)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
        ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y_\mu^\text{GPR}|$')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f'Mean Discrepancy of GPR Approximations vs. $y(x)$ (P={P},N={N},d={d})')
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Menagerie_dir, fname, f'gpr_discrepancy_P_{P}_N_{N}_d_{d}_chi_{1}.png'), dpi=300, bbox_inches='tight')
        dats.append({'P':P,
                    'ratio_mean': krat.mean(), 
                    'ratio_std': krat.std(),
                    'eig': lH,
                    'diff_mean': kdiff.mean(),
                    'diff_std': kdiff.std()})
        current_net_discrepancies = {
            'P': P,
            'Network Output': {'mean': np.mean(discrepancy_network_output), 'std': np.std(discrepancy_network_output)},
            'Projection GPR': {'mean': np.mean(discrepancy_projection_gpr), 'std': np.std(discrepancy_projection_gpr)},
            'Weight Cov. GPR': {'mean': np.mean(discrepancy_weight_cov_gpr), 'std': np.std(discrepancy_weight_cov_gpr)},
            'NNGP Theoretical': {'mean': np.mean(discrepancy_nngp_gpr), 'std': np.std(discrepancy_nngp_gpr)},
        }
        all_networks_discrepancy_data.append(current_net_discrepancies)
    all_networks_discrepancy_data_sorted = sorted(all_networks_discrepancy_data, key=lambda x: x['P'])
    print(all_networks_discrepancy_data_sorted)
    Ps = [d['P'] for d in all_networks_discrepancy_data_sorted]
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
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle('Mean Absolute Discrepancy with GPR $y^\\text{GPR}(x)$ (d=3, N varies, P varies)', fontsize=16)
    ax.set_ylabel('Mean Absolute Discrepancy $|\hat{y}_\mu-y^\\text{GPR}_\mu|$')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    num_Ps = len(Ps)
    num_disc_types = len(discrepancy_types)
    bar_width = 0.8 / num_Ps
    group_spacing = 1.0
    x_group_positions = np.arange(num_disc_types)
    colors = plt.cm.viridis(np.linspace(0, 1, num_Ps))
    legend_handles = []
    legend_labels = []
    pmax = 0.0
    for j, P_val in enumerate(Ps):
        offset = (j - (num_Ps - 1) / 2) * bar_width
        current_P_means = []
        current_P_stds = []
        for disc_type in discrepancy_types:
            net_data = next(item for item in all_networks_discrepancy_data_sorted if item["P"] == P_val)
            current_P_means.append(net_data[disc_type]['mean'])
            current_P_stds.append(net_data[disc_type]['std'])
            pmax = np.max([net_data[disc_type]['mean'], pmax])
        bars = ax.bar(x_group_positions + offset, current_P_means,
                    yerr=current_P_stds, align='center', alpha=0.7, capsize=5,
                    width=bar_width, color=colors[j], label=f'P={P_val}')
        for k, bar in enumerate(bars):
            x_val = bar.get_x() + bar.get_width() / 2
            y_val = current_P_means[k]
            ax.plot(x_val, y_val, 'o', color='black', markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        legend_handles.append(bars[0])
        legend_labels.append(f'P={P_val}')
    ax.set_ylim(top=pmax*2.0)
    ax.set_xticks(x_group_positions)
    ax.set_xticklabels([discrepancy_titles[dt] for dt in discrepancy_types], rotation=45, ha="right")
    ax.legend(handles=legend_handles, labels=legend_labels, title="Sample Size (P)", loc='best')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    final_plot_path = os.path.join(Menagerie_dir, fname, 'consolidated_gpr_discrepancy_plot_byP.png')
    print(final_plot_path)
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    theoretical_Ps = empirical_lH_dict.keys()
    theoretical_perp = {P: 1.0/d for P in theoretical_Ps}
    theoretical_targ =  theoretical_perp
    colors = plt.cm.viridis(np.linspace(0, 1, len(theoretical_Ps)))
    plt.figure(figsize=(10,6))
    for idx, P in enumerate(theoretical_Ps):
        color = colors[idx]
        if P in empirical_lH_dict:
            lH_vec = empirical_lH_dict[P]
            x_vals = np.arange(1, len(lH_vec)+1)
            plt.scatter(x_vals, lH_vec, color=color, label=f'Empirical lH (P={P})', marker='o')
            plt.plot(x_vals, lH_vec, color=color, linestyle='-', alpha=0.7)
        perp_x = range(2,d)
        perp_y = theoretical_perp[P] * np.ones(len(perp_x))
        plt.scatter(perp_x, perp_y, color=color, marker='x', s=100, label=f'Theory perp (P={P})')
        plt.plot(perp_x, perp_y, color=color, linestyle='--', alpha=0.5)
        target_x = [1]
        target_y = theoretical_targ[P]  
        plt.scatter(target_x, target_y, color=color, marker='X', s=100, label=f'Theory targ (P={P})')
        plt.plot(target_x, target_y, color=color, linestyle=':', alpha=0.5)
    plt.xlabel('Eigenvalue Index (1 to d)')
    plt.ylabel('lH Eigenvalue')
    plt.title('Empirical vs Theoretical lH Eigenvalues for Each P')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(Menagerie_dir, fname, 'lH_empirical_vs_theory_per_P.png'), dpi=300)
    return dats 
def kkrender(dats,ens):
    Ps, kkm, kks, Dkm, Dks = dats
    plt.scatter(np.log10(Ps), np.log10(np.array(kks)**2), label=f'{ens} ensemble')
    plt.xlabel('$\\log P$')
    plt.ylabel('$\\log\\sigma^2_{\\hat{K}/K}$')
    plt.title(f'd:3 N varies – $\\sigma_{\\hat{K}/K}$ loglog')
    plt.legend()
def dkrender(dats,ens):
    Ps, kkm, kks, Dkm, Dks = dats
    plt.scatter(np.log10(Ps), np.log10(np.array(Dks)**2), label=f'{ens} ensemble')
    plt.xlabel('$\\log P$')
    plt.ylabel('$\\log\\sigma^2_{\\hat{K} - K}$')
    plt.title("log $\\sigma^2_{\\hat{K} - K}$ against log P (samples) ")
    plt.legend()
def pl(dats,ens):
    Ps, kkm, kks, Dkm, Dks = dats
    plt.scatter(Ps, Dkm)    
    plt.errorbar(Ps, Dkm, yerr=Dks, fmt='o', capsize=4, label=f'{ens} ensemble')
    for i, txt in enumerate(kkm):
        plt.text(Ps[i], Dkm[i]+Dks[i], f'({Ps[i]}, {Dkm[i]:.2f})', color='black', ha='center', va='bottom', fontsize=8)
    plt.xlabel("P (samples)")
    plt.ylabel("Diff $K_{uv}$ Exp-Theory")
    plt.title( f'Ensemble of {len(dats)} FCN3s @ N On Data | d:3 P varies | '+'$\\hat{K}_{uv}-K^t_{uv}$')
    plt.legend()
    plt.tight_layout()
if __name__=='__main__':
    print("HIO")
    Menagerie_dir = os.path.dirname(fname)
    dats = run(fname)
    plt.figure()
    kkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_ratiok_byP.png'), dpi=300, bbox_inches='tight')
    plt.figure()
    dkrender(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'log_s2_diffk_byP.png'), dpi=300, bbox_inches='tight')
    plt.figure()
    pl(processdats(dats), 5)
    plt.savefig(os.path.join(Menagerie_dir, fname, 'kernel_discrepancy_scaling_byP')+'.png', dpi=300, bbox_inches='tight') 