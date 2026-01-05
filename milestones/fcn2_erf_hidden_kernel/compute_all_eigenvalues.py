import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
import matplotlib.pyplot as plt
import subprocess
import tempfile
# List of model directories
model_dirs = [
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N1600_chi_1600.0_lr_0.003_T_2.0_seed_2',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P900_N1600_chi_1600.0_lr_0.003_T_3.0_seed_2',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1200_N1600_chi_1600.0_lr_0.003_T_4.0_seed_2',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_0',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_1',
'/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P1500_N1600_chi_1600.0_lr_0.003_T_5.0_seed_2']

# model_dirs = ['/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d20_P100_N200_chi_200.0_lr_0.001_T_1.0_seed_42',
# '/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d20_P100_N200_chi_200.0_lr_0.0001_T_1.0_seed_42']
def parse_config_from_dirname(dirname):
    # Example: d150_P600_N700_chi_700.0_lr_0.0003_T_2.0_seed_0
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
        return None, None
    model = FCN2NetworkActivationGeneric(
        d=d, n1=N, P=P, ens=10, activation="erf",
        weight_initialization_variance=(1/d, 1/(N*chi)), device=device
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, (d, P, N, chi)

def compute_eigenvalues(model, d, device, num_samples=600):
    X = torch.randn(num_samples, d, device=device)
    eigs = []
    with torch.no_grad():
        H = model.H_Kernel(X, avg_ens=False)
        for e in range(model.ens):
            eigs.append(torch.linalg.eigvalsh(H[e]) / num_samples)
        eigs = torch.stack(eigs, dim=0)  # (ens, d)
        eigs, stds = eigs.sort(descending=True).values.mean(dim=0), eigs.sort(descending=True).values.std(dim=0) / model.ens**0.5
        return eigs.cpu().numpy(), stds.cpu().numpy()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    results = {}
    xvals = []  # P/N
    yvals = []  # largest eigenvalue
    ystds = []
    labels = []
    theory_lJ = []
    theory_lJP = []
    theory_lJ_eff = []
    theory_lJP_eff = []
    for model_dir in model_dirs:
        model, config = load_model(model_dir, device)
        if model is None:
            continue
        d, P, N, chi = config
        eigs, stds = compute_eigenvalues(model, d, device, num_samples=1000)
        print(model_dir)
        results[model_dir] = {"eigenvalues": eigs, "stds": stds, "config": config}
        print(f"  P:{P} N:{N} max eig: {np.max(eigs):.6f}, max std: {stds[0]:.6f}, mean eig: {np.mean(eigs[1:(d-1)]):.6f}, std eig: {np.mean(stds[1:(d-1)]):.6f}")
        xvals.append( P / 600 )
        yvals.append(np.max(eigs))
        ystds.append(stds[0])
        labels.append(f"P: {P}, N:{N}")
        EK_kappa = P / 600 * 1.0
        # --- Run Julia theory solver for kappa=1.0 ---
        import time
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            theory_json = tf.name
        julia_cmd = [
            "julia", str(Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"),
            "--d", str(d), "--n1", str(N), "--P", str(P), "--chi", str(chi), "--kappa", str(EK_kappa), "--output", theory_json
        ]
        subprocess.run(julia_cmd, check=True)
        t1 = time.time()
        print(f"Theory solver (kappa=1.0) took {t1-t0:.2f} seconds.")
        with open(theory_json, "r") as f:
            theory_data = f.read()
        import json
        theory_result = json.loads(theory_data)
        lJ = theory_result["lJ"]
        lJP = theory_result["lJP"]
        theory_lJ.append(lJ)
        theory_lJP.append(lJP)

        # --- Run self-consistent solver with experimental eigenvalues ---
        # Save eigenvalues and kappa_bare=1.0 to a json file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf2:
            eig_json = tf2.name
        with open(eig_json, "w") as f:

            json.dump({"eigenvalues": eigs.tolist(), "kappa_bare": EK_kappa}, f)
        # Run self-consistent solver
        sc_cmd = [
            "julia", str(Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"),
            eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True)
        # Parse kappa_eff from output
        import re
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
            print(f"  Found kappa_eff: {kappa_eff:.6f}")
        else:
            print("Warning: could not parse kappa_eff from self-consistent solver output.")
            kappa_eff = 1.0

        # --- Rerun theory solver with kappa_eff ---
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf3:
            theory_json_eff = tf3.name
        julia_cmd_eff = [
            "julia", str(Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"),
            "--d", str(d), "--n1", str(N), "--P", str(P), "--chi", str(chi), "--kappa", str(kappa_eff), "--output", theory_json_eff
        ]
        subprocess.run(julia_cmd_eff, check=True)
        t1 = time.time()
        print(f"Theory solver (kappa_eff={kappa_eff:.6g}) took {t1-t0:.2f} seconds.")
        with open(theory_json_eff, "r") as f:
            theory_data_eff = f.read()
        theory_result_eff = json.loads(theory_data_eff)
        print(" Effective Ride Target Eigenvalues:", theory_result_eff["lJ"], theory_result_eff["lJP"])
        lJ_eff = theory_result_eff["lJ"]
        lJP_eff = theory_result_eff["lJP"]
        theory_lJ_eff.append(lJ_eff)
        theory_lJP_eff.append(lJP_eff)
    np.savez("eigenvalues_all_models.npz", **results)
    print("Saved eigenvalues to eigenvalues_all_models.npz")

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(xvals, yvals, c='b', label='Models $\lambda_* \\pm \\sigma^2 / \sqrt{5}$')
    for i, label in enumerate(labels):
        plt.annotate(label, (xvals[i], yvals[i]), fontsize=8, alpha=0.7)
        plt.plot(xvals, theory_lJ, 'ro-', label='Theory lJ (bare, kappa=P/600)')
        # plt.plot(xvals, theory_lJP, 'mo-', label='Theory lJP (bare, kappa=P/600)')
        plt.plot(xvals, theory_lJ_eff, 'gs-', label='Theory lJ (self-consistent)')
        # plt.plot(xvals, theory_lJP_eff, 'cs-', label='Theory lJP (self-consistent)')

   
    plt.errorbar(xvals, yvals, yerr=ystds, fmt='o', color='b', alpha=0.5, capsize=3)
    plt.axhline(y=0.0508, color='r', linestyle='--', label='Theoretical Target Eigenvalue')
    plt.xlabel(" $\\alpha = P/600$")
    plt.ylabel("Target $v^*$ eigenvalue")
    plt.title("Target $v^*$ eigenvalue vs $\\alpha = P/600, \\kappa = \\alpha$ for all models FCN2 Erf on Linear Task")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ylim((0.04, 0.053))
    # plt.legend()
     # Get all handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Use a dictionary to store unique labels and their corresponding handles
    # zip() pairs labels with handles, and dict() keeps only the first unique label
    unique_labels_handles = dict(zip(labels, handles))

    # Plot the legend using the unique handles and labels from the dictionary
    plt.legend(unique_labels_handles.values(), unique_labels_handles.keys())

    plt.savefig("scatter_PN_vs_maxeig.png", dpi=150)
    print("Saved scatter plot to scatter_PN_vs_maxeig.png")

if __name__ == "__main__":
    main()
