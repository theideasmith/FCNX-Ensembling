import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Parameters
d = 150
chi = 50
N_list = [200, 1600]
P_list = np.logspace(np.log10(50), np.log10(3000), 10)  # 50 points from 50 to 3000

# Colors for different N
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

def compute_learnabilities(ags):
    N, P, d, chi = ags
    fcn3_mu1, fcn3_mu3, fcn2_mu1, fcn2_mu3 = np.nan, np.nan, np.nan, np.nan
    # Call FCN3 EOS solver
    try:
        temp_file = f"/tmp/fcn3_{N}_{int(P)}.json"
        cmd_fcn3 = f"cd /home/akiva/FCNX-Ensembling && julia julia_lib/eos_fcn3erf.jl --d {d} --n1 {N} --n2 {N} --chi {chi} --P {int(P)} --kappa 0.14 --quiet --to {temp_file}"
        result_fcn3 = subprocess.run(cmd_fcn3, shell=True, capture_output=True, text=True)
        if result_fcn3.returncode == 0 and os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                data_fcn3 = json.load(f)
            fcn3_mu1 = data_fcn3['target']['mu1']
            fcn3_mu3 = data_fcn3['target']['mu3']
            os.remove(temp_file)
    except Exception as e:
        pass
    # Call FCN2 solver
    try:
        temp_file = f"/tmp/fcn2_{N}_{int(P)}.json"
        cmd_fcn2 = f"cd /home/akiva/FCNX-Ensembling && julia julia_lib/compute_fcn2_erf_cubic_eigs.jl --d {d} --n1 {N} --P {int(P)} --chi {chi} --kappa 0.14 --quiet --to {temp_file}"
        result_fcn2 = subprocess.run(cmd_fcn2, shell=True, capture_output=True, text=True)
        if result_fcn2.returncode == 0 and os.path.exists(temp_file):

            with open(temp_file, 'r') as f:
                data_fcn2 = json.load(f)
            print("FCN2 output:", data_fcn2)  # Debug print to check the structure
            fcn2_mu1 = data_fcn2['target']['learnability1']
            fcn2_mu3 = data_fcn2['target']['learnability3']

            os.remove(temp_file)
    except Exception as e:
        pass
    print("Results for N={}, P={}: FCN3 μ₁={}, μ₃={}, FCN2 μ₁={}, μ₃={}".format(N, int(P), fcn3_mu1, fcn3_mu3, fcn2_mu1, fcn2_mu3))
    return N, P, fcn3_mu1, fcn3_mu3, fcn2_mu1, fcn2_mu3



# Data storage
fcn3_data = {}
fcn2_data = {}

print("Running solvers in parallel...")
all_tasks = [(N, P, d, chi) for N in N_list for P in P_list]
with mp.Pool(processes=8) as pool:
    results = list(tqdm(pool.imap_unordered(compute_learnabilities, all_tasks), total=len(all_tasks)))

# Organize results
for N in N_list:
    fcn3_mu1 = []
    fcn3_mu3 = []
    fcn2_mu1 = []
    fcn2_mu3 = []
    for res in results:
        if res[0] == N:
            fcn3_mu1.append(res[2] if res[2] is not None else np.nan)
            fcn3_mu3.append(res[3] if res[3] is not None else np.nan)
            fcn2_mu1.append(res[4] if res[4] is not None else np.nan)
            fcn2_mu3.append(res[5] if res[5] is not None else np.nan)
    fcn3_data[N] = {
        'P': P_list,
        'mu1': np.array(fcn3_mu1),
        'mu3': np.array(fcn3_mu3)
    }
    fcn2_data[N] = {
        'P': P_list,
        'mu1': np.array(fcn2_mu1),
        'mu3': np.array(fcn2_mu3)
    }

# Commit to file
output_path = Path("/home/akiva/FCNX-Ensembling/milestones/learnability_analysis/learnability_data.json")
output_data = {
    'fcn3': fcn3_data,
    'fcn2': fcn2_data
}
# Pickle the data using pickle
import pickle
with open(output_path.with_suffix('.pkl'), 'wb') as f:
    pickle.dump(output_data, f)
print(f"Data saved to {output_path.with_suffix('.pkl')}")


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

for i, N in enumerate(N_list):
    color = colors[i]
    # FCN3 mu1
    ax.plot(fcn3_data[N]['P'], fcn3_data[N]['mu1'], color=color, linestyle='-', linewidth=2, label=f'FCN3 N={N} μ₁')
    # FCN3 mu3
    ax.plot(fcn3_data[N]['P'], fcn3_data[N]['mu3'], color=color, linestyle='--', linewidth=2, label=f'FCN3 N={N} μ₃')
    # FCN2 mu1
    ax.plot(fcn2_data[N]['P'], fcn2_data[N]['mu1'], color=color, linestyle='-.', linewidth=2, label=f'FCN2 N={N} μ₁')
    # FCN2 mu3
    ax.plot(fcn2_data[N]['P'], fcn2_data[N]['mu3'], color=color, linestyle=':', linewidth=2, label=f'FCN2 N={N} μ₃')

ax.set_xscale('log')
ax.set_xlabel('P (Sample Size)', fontsize=14)
ax.set_ylabel('Learnability Ratio', fontsize=14)
ax.set_title('Learnability Ratios for FCN2 and FCN3 Networks\n(d=150, χ=50)', fontsize=16, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/akiva/FCNX-Ensembling/milestones/learnability_analysis/learnability_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved to /home/akiva/FCNX-Ensembling/milestones/learnability_analysis/learnability_plot.png")