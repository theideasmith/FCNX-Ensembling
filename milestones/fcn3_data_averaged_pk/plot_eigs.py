import torch
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract_params_from_dir(dirname):
    # Example: fcn3_d150_n1900_n2900_P2400_kappa4.0_chi900_seed1
    m = re.search(r'd(\d+)_n1(\d+)_n2(\d+)_P(\d+)_kappa([0-9.]+)_chi(\d+)_seed(\d+)', dirname)
    if not m:
        return None
    d, n1, n2, P, kappa, chi, seed = m.groups()
    if d == '20':
        return None  # Skip d=200 as 
    return dict(d=int(d), n1=int(n1), n2=int(n2), P=int(P), kappa=float(kappa), chi=int(chi), seed=int(seed))

def compute_largest_eigval(model, d, P, n2):
    # Use a large random X for eigenvalue computation
    X = torch.randn(P, d, device=DEVICE)
    with torch.no_grad():
        f = model.h1_preactivation(X)  # (P, ens, n2)
        q = model.ens
        # Compute kernel per ensemble: K_i[u,v] = sum_m f[u,i,m] * f[v,i,m] / n2
        Ks = torch.einsum('uqm,vqm->quv', f, f) / n2  # (ens, P, P)
        largest_eigs = []
        for i in range(Ks.shape[0]):
            eigvals = torch.linalg.eigvalsh(Ks[i]) / P
            largest_eigs.append(eigvals.max().item())
        largest_eig = np.max(model.H_eig(X, X).cpu().numpy())
        return largest_eig
        return np.mean(largest_eigs)

def main():
    models_dir = "./models"
    all_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    results = []
    for d in all_dirs:
        params = extract_params_from_dir(d)
        if params is None:
            continue
        model_path = os.path.join(d, f"model_seed{params['seed']}.pt")
        if not os.path.exists(model_path):
            continue
        model = FCN3NetworkActivationGeneric(
            d=params['d'], n1=params['n1'], n2=params['n2'], P=params['P'], ens=5, activation="erf",
            weight_initialization_variance=(1/params['d'], 1/params['n1'], 1/(params['n2']*params['chi'])),
            device=DEVICE
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        largest_eig = compute_largest_eigval(model, params['d'], 5000, params['n2'])
        results.append({
            "P": params['P'],
            "N": params['n2'],
            "largest_eig": largest_eig,
            "d": params['d'],
            "chi": params['chi']
        })
        print(f"Loaded {model_path}, P={params['P']}, N={params['n2']}, largest_eig={largest_eig:.6f}")

    # Sort by P for plotting
    results = sorted(results, key=lambda x: x['P'])
    Ps = [r['P'] for r in results]
    Ns = [r['N'] for r in results]
    eigs = [r['largest_eig'] for r in results]
    d = results[0]['d'] if results else 0
    chi = results[0]['chi'] if results else 0

    plt.figure(figsize=(8, 6))
    labels = [f"P:{P}, N:{N}" for P, N in zip(Ps, Ns)]
    Ps = np.array(Ps)
    eigs = np.array(eigs)
    plt.scatter(Ps, eigs, label="Largest eigenvalue")
    for i, txt in enumerate(labels):
        plt.annotate(txt, (Ps[i], eigs[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')
    plt.axhline(0.22138637010581488, color='red', linestyle='--', label='Theory')
    plt.xlabel("P")
    plt.ylabel("Largest eigenvalue of K (mean over ensemble)")
    plt.title(f"Largest eigenvalue vs P (d={d}, chi={chi})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fcn3_largest_eigvals_vs_P.png")
    plt.show()

if __name__ == "__main__":
    main()