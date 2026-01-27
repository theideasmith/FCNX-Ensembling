import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric

def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    seed = int(parts[10]) if len(parts) > 10 else None
    return d, P, N, chi, seed

def load_model(model_dir, device):
    d, P, N, chi, seed = parse_config_from_dirname(model_dir)
    model_path = Path(model_dir) / "model.pt"
    print(f"Loading model from {model_path}")
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
    return model, d, P, N, chi, seed
runs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_0_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_2_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_1_eps_0.03',
                '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P3000_N800_chi_10.0_lr_1e-05_T_16.0_seed_3_eps_0.03']
if __name__ == "__main__":
    # Use the first model in the provided list
    # model_dir = '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d100_P1500_N1600_chi_10.0_lr_3e-05_T_8.0_seed_0_eps_0.03'
    model_dir = runs[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, d, P, N, chi, seed = load_model(model_dir, device)
    if model is None:
        exit(1)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    sample_sizes = np.logspace(3, 9, num=10, dtype=int)
    variances = []
    for n in sample_sizes:
        batch_size =  10000
        n_batches = n // batch_size
        remainder = n % batch_size
        proj_accum = torch.zeros((model.ens, model.n1), dtype=torch.float32, device=device)
        for i in range(n_batches + (1 if remainder > 0 else 0)):
            bs = batch_size if i < n_batches else remainder
            if bs == 0:
                continue
            x = torch.randn(bs, d, dtype=torch.float32, device=device)
            with torch.no_grad():
                h0 = model.h0_activation(x)
            x0 = x[:, 0]
            h3 = x0**3 - 3 * x0
            proj_accum += torch.einsum('pqn,p->qn', h0, h3) / n

            del x, h0, x0, h3
            torch.cuda.empty_cache()
        proj_all = proj_accum.flatten()
        var = proj_all.var().item()
        variances.append(var)
        print(f"Samples: {n}, Variance: {var:.4e}")
        del proj_accum, proj_all
        torch.cuda.empty_cache()

    plt.figure(figsize=(7,5))
    plt.loglog(sample_sizes, variances, marker='o', label='Variance of projection')
    plt.xlabel('Number of samples')
    plt.ylabel('Variance')
    plt.title('Convergence of Hermite-3 projection variance vs sample size')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('projection_variance_convergence_vs_samples.png', dpi=150)
    print('Saved plot to projection_variance_convergence_vs_samples.png')
