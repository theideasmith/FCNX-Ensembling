import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import subprocess
import tempfile

def run_julia_theory(P, d, N, chi, kappa, eps=0.03):
    julia_script = Path(__file__).parent.parent.parent / "julia_lib" / "eos_fcn3erf.jl"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)
    cmd = [
        "julia", str(julia_script), f"--d={d}", f"--P={P}", f"--n1={N}", f"--n2={N}",
        f"--chi={chi}", f"--kappa={kappa}", f"--epsilon={eps}", f"--to={to_path}", "--quiet"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        with open(to_path, "r") as f:
            data = json.load(f)
        lWT = data.get("target", {}).get("lWT", None)
        return lWT
    except Exception as e:
        print(f"Theory failed for P={P}: {e}")
        return None
    finally:
        if to_path.exists():
            to_path.unlink()

def main():
    # Directory containing all runs
    results_dir = Path(__file__).parent / "p_scan_erf_results"
    equilibrium_values = [0.0260, 0.026,  0.0240, 0.0215, 0.0200, 0.0175, 0.0145, 0.0130, 0.0095]
    # Find all d=150 runs
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("d150_")]
    print(run_dirs)
    # Parse config from directory names
    def parse_config_from_name(name):
        # Example: d150_P100_N1000_chi80_kappa0.1_nseeds10_ens20
        parts = name.split('_')
        cfg = {}
        for p in parts:
            if p.startswith('d') and p[1:].isdigit():
                cfg['d'] = int(p[1:])
            elif p.startswith('P') and p[1:].isdigit():
                cfg['P'] = int(p[1:])
            elif p.startswith('N') and p[1:].isdigit():
                cfg['N'] = int(p[1:])
            elif p.startswith('chi'):
                try:
                    cfg['chi'] = float(p[3:])
                except: pass
            elif p.startswith('kappa'):
                try:
                    cfg['kappa'] = float(p[5:])
                except: pass
        return cfg

    configs = [parse_config_from_name(run_dir.name) for run_dir in run_dirs]
    # Remove any incomplete configs
    configs = [c for c in configs if all(k in c for k in ('P','d','N','chi','kappa'))]
    # Sort configs by P ascending
    configs = sorted(configs, key=lambda c: c["P"])
    # Skip the largest P (last in sorted list)
    if configs:
        configs = configs[:-1]
    Ps = [cfg["P"] for cfg in configs]
    chis = [cfg["chi"] for cfg in configs]
    Ns = [cfg["N"] for cfg in configs]
    kappas = [cfg["kappa"] for cfg in configs]
    # Compute theory lWT for each config
    print(Ps)
    theory_lWTs = []
    for P, d, N, chi, kappa in zip(Ps, [150]*len(Ps), Ns, chis, kappas):
        lWT = run_julia_theory(P, 150, N, chi, kappa)
        theory_lWTs.append(lWT)
        print(lWT)
    # Sort empirical values by P ascending (assume user provided in descending order)
    empirical_lWTs = list(reversed(equilibrium_values))[:len(Ps)]
    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(Ps, empirical_lWTs, 'o-', label='Empirical lWT (estimated)')
    plt.plot(Ps, theory_lWTs, 's--', label='Theory lWT (Julia)')
    plt.xlabel('P')
    plt.ylabel('lWT')
    plt.title('lWT vs P for d=150')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(results_dir / "lWT_vs_P_d150.png", dpi=300)
    print("Full path to figure: ", results_dir / "lWT_vs_P_d150.png")

if __name__ == "__main__":
    main()
