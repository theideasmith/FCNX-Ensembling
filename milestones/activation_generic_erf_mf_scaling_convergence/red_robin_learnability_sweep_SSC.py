import numpy as np
import subprocess
import time
import os
from collections import deque

# Parameters
d = 50
P_values = np.logspace(np.log10(d/2), np.log10(5*d), num=5, dtype=int)
powers = np.linspace(0.5, 1.6, num=5)
for p in powers:
    val = int(d**p)
    if val not in P_values:
        P_values = np.append(P_values, val)
P_values = np.unique(np.sort(P_values))

seeds = 5
kappa = 0.1
N = 800
lr = 1e-4
device = 'cuda:1'
ens = 5
chi = 1.0
epochs = 10_000_000
train_script = os.path.join(os.path.dirname(__file__), 'd_sweep_seeds.py')

# --- CRITICAL CHANGE: CONFIGURE TOTAL CONCURRENCY ---
MAX_TOTAL_CONCURRENT_JOBS = 4 # Change to 1 for strict one-by-one execution

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
        '--to', 'p_scan_erf_results',
        '--eps', '0.03'
    ]

# Create a flattened queue of every individual job (P + Seed)
job_queue = deque()
for P in P_values:
    for seed in range(seeds):
        job_queue.append((P, seed))

running_procs = []
completed_count = 0
total_jobs = len(job_queue)

print(f"Starting execution. Total jobs to run: {total_jobs}")

while job_queue or running_procs:
    # 1. Fill up the running slots
    while len(running_procs) < MAX_TOTAL_CONCURRENT_JOBS and job_queue:
        P, seed = job_queue.popleft()
        cmd = make_cmd(P, seed)
        
        print(f"[{time.strftime('%H:%M:%S')}] Launching P={P}, Seed={seed}")
        proc = subprocess.Popen(cmd)
        running_procs.append({'proc': proc, 'P': P, 'seed': seed})
        time.sleep(1) # Small stagger to prevent CPU/GPU spikes

    # 2. Check for completed jobs
    for job in running_procs[:]:
        ret = job['proc'].poll()
        if ret is not None:
            running_procs.remove(job)
            completed_count += 1
            print(f"[{time.strftime('%H:%M:%S')}] Finished P={job['P']}, Seed={job['seed']} (Total: {completed_count}/{total_jobs})")

    time.sleep(2) # Poll every 2 seconds

print(f"All {completed_count} jobs completed.")