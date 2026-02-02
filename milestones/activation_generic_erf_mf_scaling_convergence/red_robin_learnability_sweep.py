import numpy as np
import subprocess
import time
import os
from collections import deque

# Parameters
d = 50
P_values = np.logspace(np.log10(d/2), np.log10(5*d), num=5, dtype=int)
# Add to P_values powers of d from sqrt(d) to 2 * d^(3/2)
powers = np.linspace(0.5, 1.6, num=5)
for p in powers:
    val = int(d**p)
    if val not in P_values:
        P_values = np.append(P_values, val)
P_values = np.unique(np.sort(P_values))
seeds = 5
kappa = 0.1
N=800
lr=1e-3
device='cuda:0'
ens = 5
chi = int(N/10)
epochs = 10_000_000
max_parallel_jobs = 4  # Launch jobs at both ends
train_script = os.path.join(os.path.dirname(__file__), 'd_sweep_seeds.py')

# Command template (customize as needed)
def make_cmd(P, seed):
    return [
        'python3', train_script,
        '--d', str(d),
        '--P', str(P),
        '--chi', str(chi),
        '--kappa', str(kappa),
        '--N', str(N),
        '--lr', str(lr),
        '--device', device,
        '--epochs', str(epochs),
        '--base_seed', str(seed),
        '--ens', str(ens),
        '--to','p_scan_erf_results',
        '--eps','0.03'
    ]

# Job queue: deque for red robin
pending = deque(P_values)
running = []  # List of (process, P, end, seed)
completed = []

# Launch jobs at both ends
for end in ['left', 'right']:
    if pending:
        P = pending.popleft() if end == 'left' else pending.pop()
        for seed in range(seeds):
            cmd = make_cmd(P, seed)
            proc = subprocess.Popen(cmd)
            running.append({'proc': proc, 'P': P, 'end': end, 'seed': seed})
            time.sleep(0.5)  # Stagger launches

# Main loop
while pending or running:
    for job in running[:]:
        ret = job['proc'].poll()
        if ret is not None:
            running.remove(job)
            completed.append(job)
            # Launch next job at the same end
            if pending:
                P = pending.popleft() if job['end'] == 'left' else pending.pop()
                for seed in range(seeds):
                    cmd = make_cmd(P, seed)
                    proc = subprocess.Popen(cmd)
                    running.append({'proc': proc, 'P': P, 'end': job['end'], 'seed': seed})
                    time.sleep(0.5)
    time.sleep(2)

print(f"All jobs completed. {len(completed)} jobs run.")
