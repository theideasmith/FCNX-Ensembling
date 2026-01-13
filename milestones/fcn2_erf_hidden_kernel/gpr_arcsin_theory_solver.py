

import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess, tempfile, json
from pathlib import Path
import argparse
import sys

# Parameters
d = 150  # input dimension
n1 = 700  # hidden layer width
P = 600   # number of samples
chi = n1   # chi parameter
kappa = 1.0  # initial kappa

# Argument parsing for cache
parser = argparse.ArgumentParser()
parser.add_argument('--use-cache', action='store_true', help='Load results from cache if available')
args = parser.parse_args()

# Cache file path
cache_file = Path(__file__).parent / 'gpr_arcsin_theory_solver_cache.json'



if args.use_cache and cache_file.exists():
    print(f"Loading results from cache: {cache_file}")
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    gpr_eigs = np.array(cache['gpr_eigs'])
    gpr_kappa_eff = cache['gpr_kappa_eff']
    gpr_julia_lJ = cache['gpr_julia_lJ']
    net_eigs = np.array(cache['net_eigs'])
    net_julia_lJ = cache['net_julia_lJ']
else:
    # 1. Generate random data
    torch.manual_seed(42)
    P_inf = 3000
    X = torch.randn((P_inf, d), device=torch.device('cuda:1'))
    # 2. Compute the arcsin (erf) kernel (GPR kernel)
    def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
        """Compute arcsin kernel matrix for inputs X (P, d)."""
        XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
        diag = torch.sqrt((1 + 2 * XXT).diag())
        denom = diag[:, None] * diag[None, :]
        arg = 2 * XXT / denom
        arg = torch.clamp(arg, -1.0, 1.0)
        return (2 / torch.pi) * torch.arcsin(arg)

    print("Computing arcsin kernel (torch)...")
    K = arcsin_kernel(X)
    print("Kernel shape:", K.shape)

    # 3. Diagonalize the kernel (get eigenvalues)
    print("Diagonalizing kernel...")
    eigs = torch.linalg.eigvalsh(K).cpu().numpy() / X.shape[0]
    print("Eigenvalues computed.")
    # 4. Use Julia self_consistent_kappa_solver.jl to get kappa_eff
    print("Calling Julia self_consistent_kappa_solver.jl to compute kappa_eff...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        eig_json = tf.name
    with open(eig_json, "w") as f:
        json.dump({"eigenvalues": eigs.tolist(), "kappa_bare": kappa}, f)
    julia_sc_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
    sc_cmd = [
        "julia", str(julia_sc_script), eig_json, str(P)
    ]
    sc_out = subprocess.check_output(sc_cmd, text=True)
    import re
    match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
    if match:
        kappa_eff = float(match.group(1))
        print(f"  Found kappa_eff: {kappa_eff:.6f}")
    else:
        print("Warning: could not parse kappa_eff from self-consistent solver output.")
        kappa_eff = kappa

    # --- Modular pipeline for kernel analysis ---
    def analyze_kernel(kernel: torch.Tensor, label: str, d: int, n1: int, P: int, chi: float, kappa: float):
        eigs = torch.linalg.eigvalsh(kernel).cpu().numpy() / P
        print(f"Eigenvalues computed for {label}.")
        # 1. Julia self-consistent solver
        print(f"Calling Julia self_consistent_kappa_solver.jl for {label}...")
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            eig_json = tf.name
        with open(eig_json, "w") as f:
            json.dump({"eigenvalues": eigs.tolist(), "kappa_bare": kappa}, f)
        julia_sc_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
        sc_cmd = [
            "julia", str(julia_sc_script), eig_json, str(P)
        ]
        sc_out = subprocess.check_output(sc_cmd, text=True)
        match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
        if match:
            kappa_eff = float(match.group(1))
            print(f"  Found kappa_eff: {kappa_eff:.6f}")
        else:
            print("Warning: could not parse kappa_eff from self-consistent solver output.")
            kappa_eff = kappa
        # 2. Julia theory eigenvalues
        print(f"Calling Julia compute_fcn2_erf_eigs.jl for {label}...")
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            theory_json = tf.name
        julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"
        julia_cmd = [
            "julia", str(julia_script),
            "--d", str(d), "--n1", str(n1), "--P", str(P), "--chi", str(chi), "--kappa", str(kappa_eff), "--output", theory_json
        ]
        subprocess.run(julia_cmd, check=True)
        with open(theory_json, "r") as f:
            theory_data = f.read()
        theory_result = json.loads(theory_data)
        julia_lJ = theory_result["lJ"]
        julia_lJP = theory_result["lJP"]
        print(f"Julia theory eigenvalues (lJ) for {label}: {julia_lJ}")
        return eigs, kappa_eff, julia_lJ

    # --- Run for GPR kernel ---
    gpr_eigs, gpr_kappa_eff, gpr_julia_lJ = analyze_kernel(K, "GPR kernel", d, n1, P, chi, kappa)

    # --- Run for network kernel ---
    print("\nLoading network model and computing kernel...")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
    from FCN2Network import FCN2NetworkActivationGeneric
    model_dir = "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_0"
    def parse_config_from_dirname(dirname):
        parts = Path(dirname).name.split('_')
        d = int(parts[0][1:])
        P = int(parts[1][1:])
        N = int(parts[2][1:])
        chi = float(parts[4][:])
        return d, P, N, chi
    d_net, P_net, N_net, chi_net = parse_config_from_dirname(model_dir)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_dir) / "model.pt"
    if not model_path.exists():
        model_path = Path(model_dir) / "model_final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_dir}")
    model = FCN2NetworkActivationGeneric(
        d=d_net, n1=N_net, P=P_net, ens=5, activation="erf",
        weight_initialization_variance=(1/d_net, 1/(N_net*chi_net)), device=device
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X_net = X.to(device) 
        H = model.H_Kernel(X_net, avg_ens=True)
        H = H.cpu()

    net_eigs = torch.linalg.eigvalsh(H).cpu().numpy() / P_inf
    print("Network kernel eigenvalues computed.")
    print(np.sort(net_eigs)[-5:][::-1])
    # net_eigs, net_kappa_eff, net_julia_lJ = analyze_kernel(H, "Network kernel", d_net, N_net, P_net, chi_net, kappa)
    # For now, use theory for network kernel as in original code
    # Call Julia for network kernel theory
    print("Calling Julia compute_fcn2_erf_eigs.jl for Network kernel...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        net_theory_json = tf.name
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"
    julia_cmd = [
        "julia", str(julia_script),
        "--d", str(d_net), "--n1", str(N_net), "--P", str(P_net), "--chi", str(chi_net), "--kappa", str(gpr_kappa_eff), "--output", net_theory_json
    ]
    subprocess.run(julia_cmd, check=True)
    with open(net_theory_json, "r") as f:
        net_theory_data = f.read()
    net_theory_result = json.loads(net_theory_data)
    net_julia_lJ = net_theory_result["lJ"]
    print(f"Network Julia theory eigenvalues (lJ): {net_julia_lJ}")

    # Save all results to cache
    cache = {
        'gpr_eigs': gpr_eigs.tolist(),
        'gpr_kappa_eff': gpr_kappa_eff,
        'gpr_julia_lJ': gpr_julia_lJ,
        'net_eigs': net_eigs.tolist(),
        'net_julia_lJ': net_julia_lJ
    }
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    print(f"Saved results to cache: {cache_file}")

# --- Modular pipeline for kernel analysis ---
import subprocess, tempfile, json
from pathlib import Path
import re

def analyze_kernel(kernel: torch.Tensor, label: str, d: int, n1: int, P: int, chi: float, kappa: float):
    eigs = torch.linalg.eigvalsh(kernel).cpu().numpy() / P
    print(f"Eigenvalues computed for {label}.")
    # 1. Julia self-consistent solver
    print(f"Calling Julia self_consistent_kappa_solver.jl for {label}...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        eig_json = tf.name
    with open(eig_json, "w") as f:
        json.dump({"eigenvalues": eigs.tolist(), "kappa_bare": kappa}, f)
    julia_sc_script = Path(__file__).parent.parent.parent / "julia_lib" / "self_consistent_kappa_solver.jl"
    sc_cmd = [
        "julia", str(julia_sc_script), eig_json, str(P)
    ]
    sc_out = subprocess.check_output(sc_cmd, text=True)
    match = re.search(r"kappa_eff = ([0-9.eE+-]+)", sc_out)
    if match:
        kappa_eff = float(match.group(1))
        print(f"  Found kappa_eff: {kappa_eff:.6f}")
    else:
        print("Warning: could not parse kappa_eff from self-consistent solver output.")
        kappa_eff = kappa
    # 2. Julia theory eigenvalues
    print(f"Calling Julia compute_fcn2_erf_eigs.jl for {label}...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        theory_json = tf.name
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"
    julia_cmd = [
        "julia", str(julia_script),
        "--d", str(d), "--n1", str(n1), "--P", str(P), "--chi", str(chi), "--kappa", str(kappa_eff), "--output", theory_json
    ]
    subprocess.run(julia_cmd, check=True)
    with open(theory_json, "r") as f:
        theory_data = f.read()
    theory_result = json.loads(theory_data)
    julia_lJ = theory_result["lJ"]
    julia_lJP = theory_result["lJP"]
    print(f"Julia theory eigenvalues (lJ) for {label}: {julia_lJ}")
    return eigs, kappa_eff, julia_lJ

# --- Run for GPR kernel ---
gpr_eigs, gpr_kappa_eff, gpr_julia_lJ = analyze_kernel(K, "GPR kernel", d, n1, P, chi, kappa)

# --- Run for network kernel ---
print("\nLoading network model and computing kernel...")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric
model_dir = "/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d150_P600_N700_chi_700.0_lr_3e-06_T_2.0_seed_0"
def parse_config_from_dirname(dirname):
    parts = Path(dirname).name.split('_')
    d = int(parts[0][1:])
    P = int(parts[1][1:])
    N = int(parts[2][1:])
    chi = float(parts[4][:])
    return d, P, N, chi
d_net, P_net, N_net, chi_net = parse_config_from_dirname(model_dir)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_path = Path(model_dir) / "model.pt"
if not model_path.exists():
    model_path = Path(model_dir) / "model_final.pt"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found in {model_dir}")
model = FCN2NetworkActivationGeneric(
    d=d_net, n1=N_net, P=P_net, ens=5, activation="erf",
    weight_initialization_variance=(1/d_net, 1/(N_net*chi_net)), device=device
)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    X_net = X.to(device) 
    H = model.H_Kernel(X_net, avg_ens=True)
    H = H.cpu()

eigs = torch.linalg.eigvalsh(H).cpu().numpy() / P_inf
print("Network kernel eigenvalues computed.")
print(np.sort(eigs)[-5:][::-1])
# net_eigs, net_kappa_eff, net_julia_lJ = analyze_kernel(H, "Network kernel", d_net, N_net, P_net, chi_net, kappa)
print(f"Network eigenvalues top 5: {np.sort(net_eigs)[-5:][::-1]}")

# --- Plot both spectra ---
plt.figure(figsize=(8,5))
plt.plot(np.sort(gpr_eigs)[::-1], label='GPR Kernel Eigenvalues')
plt.axhline(y=gpr_julia_lJ, color='r', linestyle='--', label='GPR Julia Theory lJ')
plt.plot(np.sort(net_eigs)[::-1], label='Network Kernel Eigenvalues')
plt.axhline(y=net_julia_lJ, color='g', linestyle='--', label='Network Julia Theory lJ')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('GPR vs Network Kernel and Theory Eigenvalues')
plt.legend()
plt.tight_layout()
plt.savefig('gpr_vs_network_kernel_eigenvalues.png', dpi=150)
print('Saved plot to gpr_vs_network_kernel_eigenvalues.png')
plt.close()



# 5. Call Julia script to compute theory eigenvalues
print("Calling Julia script to compute theory eigenvalues...")
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
    theory_json = tf.name
julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"
julia_cmd = [
    "julia", str(julia_script),
    "--d", str(d), "--n1", str(n1), "--P", str(P), "--chi", str(chi), "--kappa", str(kappa_eff), "--output", theory_json
]
subprocess.run(julia_cmd, check=True)
with open(theory_json, "r") as f:
    theory_data = f.read()
theory_result = json.loads(theory_data)
julia_lJ = theory_result["lJ"]
julia_lJP = theory_result["lJP"]
print("Julia theory eigenvalues (lJ):", julia_lJ)


# 6. Plot spectrum
plt.figure(figsize=(8,5))
plt.plot(np.sort(eigs)[::-1], label='Empirical Kernel Eigenvalues')
plt.axhline(y=julia_lJ, color='r', linestyle='--', label='Julia Theory lJ')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('GPR Arcsin Kernel and Theory Eigenvalues')
plt.legend()
plt.tight_layout()
plt.savefig('gpr_arcsin_kernel_eigenvalues.png', dpi=150)
print('Saved plot to gpr_arcsin_kernel_eigenvalues.png')
plt.close()
