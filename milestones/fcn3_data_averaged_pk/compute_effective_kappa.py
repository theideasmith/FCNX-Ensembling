import torch
import numpy as np
import os
import re
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Import your model class
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
JULIA_SOLVER = str(Path(__file__).parent.parent.parent / "julia_lib/self_consistent_kappa_solver.jl")

def extract_params_from_dir(dirname):
    m = re.search(r'd(\d+)_n1(\d+)_n2(\d+)_P(\d+)_kappa([0-9.]+)_chi(\d+)_seed(\d+)', dirname)
    if not m:
        return None
    d, n1, n2, P, kappa, chi, seed = m.groups()
    return dict(d=int(d), n1=int(n1), n2=int(n2), P=int(P), kappa=float(kappa), chi=int(chi), seed=int(seed))

def compute_kernel_eigvals(model, d, P, n2):
    X = torch.randn(P, d, device=DEVICE)
    with torch.no_grad():
        f = model.h1_preactivation(X)  # (P, ens, n2)
        q = model.ens
        Ks = torch.einsum('uqm,vqm->quv', f, f) / n2  # (ens, P, P)
        eigvals = []
        for i in range(Ks.shape[0]):
            eig = torch.linalg.eigvalsh(Ks[i]).cpu().numpy().tolist()
            
            eigvals.append(eig)
        return eigvals  # list of lists

def main():
    models_dir = "./models"
    all_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    results = []
    for d in tqdm(all_dirs):
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
        # Compute eigenvalues
        eigvals = compute_kernel_eigvals(model, params['d'],3000, params['n2'])
        # Average over ensembles
        eigvals_avg = np.mean(np.array(eigvals), axis=0).tolist()
        # Save to JSON for Julia
        eig_json = {
            "eigenvalues": eigvals_avg,
            "kappa_bare": params['kappa']
        }
        eig_json_path = os.path.join(d, "eigvals_for_julia.json")
        eig_json_path  = os.path.abspath(eig_json_path)
        with open(eig_json_path, "w") as f:
            json.dump(eig_json, f)
        # Call Julia solver
        try:
            CMD = ["julia", JULIA_SOLVER, eig_json_path, str(params['P'])]
            print("Running Julia command:", " ".join(CMD))
            result = subprocess.run(
                ["julia", JULIA_SOLVER, eig_json_path, str(params['P'])],
                capture_output=True, text=True, check=True
            )
            # Parse kappa_eff from Julia output
            for line in result.stdout.splitlines():
                if "kappa_eff" in line:
                    kappa_eff = float(line.split("=")[-1])
                    break
            else:
                kappa_eff = None
        except Exception as e:
            print(f"Julia solver failed for {d}: {e}")
            kappa_eff = None
        results.append({
            "P": params['P'],
            "N": params['n2'],
            "d": params['d'],
            "chi": params['chi'],
            "kappa_bare": params['kappa'],
            "kappa_eff": kappa_eff
        })
        print(f"Done {d}: kappa_bare={params['kappa']}, kappa_eff={kappa_eff}")

    # Save all results to a summary file
    with open("kappa_eff_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("All done. Results in kappa_eff_summary.json")

if __name__ == "__main__":
    main()