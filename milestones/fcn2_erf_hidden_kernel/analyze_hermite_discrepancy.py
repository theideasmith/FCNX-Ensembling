import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/akiva/FCNX-Ensembling/lib')  # Adjust this path as needed
from FCN2Network import FCN2NetworkActivationGeneric

# Model directories (copied from user request)
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
    if not model_path.exists():
        model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        print(f"Model not found in {model_dir}")
        return None, None, None, None
    state_dict = torch.load(model_path, map_location=device)
    ens = state_dict['W0'].shape[0]
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=ens, activation="erf",
        weight_initialization_variance=(1/d, 1/(N*chi)), device=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, d, P, N

def compute_H_kernel(model, x):
    # x: (P, d), expects torch.Tensor
    return model.H_Kernel(x)

def get_hermite_basis(x):
    # x: (P, d), use only x[:,0], expects torch.Tensor
    x0 = x[:,0]
    He_1 = x0
    He_3_raw = x0**3 - 3*x0
    He_3 = He_3_raw / 6**0.5
    return He_1, He_3

def compute_Q(H, He_1, He_3):
    # Project H onto He_1, He_3 (all torch)
    P = He_1.shape[0]
    basis = torch.stack([He_1, He_3], dim=1)  # (P, 2)
    Q = torch.zeros((2,2), dtype=H.dtype, device=H.device)
    for i in range(2):
        for j in range(2):
            Q[i,j] = torch.dot(basis[:,i], torch.mv(H, basis[:,j])) / P
    return Q

def compute_discrepancy(Q, kappa=1e-6):
    # [Q + kappa I]^{-1} @ [1, 0]^T (all torch)
    Qk = Q + kappa * torch.eye(2, dtype=Q.dtype, device=Q.device)
    rhs = torch.tensor([1.0, 0.0], dtype=Q.dtype, device=Q.device)
    sol = torch.linalg.solve(Qk, rhs)
    return sol * kappa # [He3, He1] components

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discrepancies = []
    model_names = []
    for model_dir in model_dirs:

        model, d, P, N = load_model(model_dir, device)
        if model is None:
            print("Skipping model with data size P:{}, N:{}".format(P, N))
            discrepancies.append(torch.tensor([float('nan'), float('nan')], device=device))
            model_names.append(Path(model_dir).name)
            continue
        # Always generate x from seed extracted from model_dir
        import re
        m = re.search(r'seed_(\d+)', str(model_dir))
        if m:
            seed = int(m.group(1))
            torch.manual_seed(seed)
            x = torch.randn(1000, d, dtype=torch.float32, device=device)
        else:
            print(f"Could not extract seed from {model_dir}, skipping.")
            discrepancies.append(torch.tensor([float('nan'), float('nan')], device=device))
            model_names.append(Path(model_dir).name)
            continue
        H = compute_H_kernel(model, x)
        He_1, He_3 = get_hermite_basis(x)
        # breakpoint()
        Q = compute_Q(H, He_1, He_3)
        disc = compute_discrepancy(Q, kappa=P/600)
        print("Discrepancy for model P={}: He1 component = {:.6f}, He3 component = {:.6f}".format(
            P, disc[0].item(), disc[1].item()
        ))
        discrepancies.append(disc)
        model_names.append(Path(model_dir).name)

    discrepancies = torch.stack(discrepancies).cpu().numpy()  # (n_models, 2)
    # Group by P
    import collections
    P_to_discrepancies = collections.defaultdict(list)
    P_to_names = collections.defaultdict(list)
    for i, model_dir in enumerate(model_dirs):

        _, P, N, _ = parse_config_from_dirname(model_dir)

        P_to_discrepancies[P].append(discrepancies[i])
        P_to_names[P].append(model_names[i])


    Ps = sorted(P_to_discrepancies.keys())
    n_P = len(Ps)
    fig, axes = plt.subplots(1, n_P, figsize=(6*n_P, 5), squeeze=False)
    for idx, P in enumerate(Ps):
        ax = axes[0, idx]
        arr = np.array(P_to_discrepancies[P])  # (n_seeds, 2)
        means = np.nanmean(arr, axis=0)
        errs = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
        # Extract seed labels for this P
        seed_labels = []
        for name in P_to_names[P]:
            import re
            m = re.search(r'seed_(\d+)', name)
            seed_labels.append(m.group(1) if m else '?')
        bars = ax.bar(["He1", "He3"], means, yerr=errs, color=["royalblue", "orange"], capsize=8)
        # Annotate error bars with seed labels
        for j, bar in enumerate(bars):
            y = means[j] + errs[j]
            label = "Seeds: " + ", ".join(seed_labels)
            ax.text(bar.get_x() + bar.get_width()/2, y, label, ha='center', va='bottom', fontsize=10, rotation=0)
        ax.set_title(f"P={P}")
        ax.set_ylabel("Discrepancy Components")
        # ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    plt.suptitle("Hermite Discrepancy by P for FCN2 ERF Hidden Kernel Models\n kappa = P/600, MF scaling", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(MODELDIR, "discrepancy_barplots_by_P.png"), dpi=150)
    plt.close()
    print("Saved grouped discrepancy bar plots to discrepancy_barplots_by_P.png")

if __name__ == "__main__":
    main()
