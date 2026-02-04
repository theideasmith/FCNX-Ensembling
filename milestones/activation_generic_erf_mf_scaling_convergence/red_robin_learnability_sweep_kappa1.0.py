import numpy as np
import subprocess
import time
import os
from collections import deque

# --- Parameters ---
d = 150
P_values = np.logspace(np.log10(d/2), np.log10(5*d), num=5, dtype=int)
powers = np.linspace(0.5, 1.6, num=5)
for p in powers:
    val = int(d**p)
    if val not in P_values:
        P_values = np.append(P_values, val)
P_values = np.unique(np.sort(P_values))

seeds = 5
max_parallel_jobs = 10  # Adjust based on your GPU memory
train_script = os.path.join(os.path.dirname(__file__), 'd_sweep_seeds.py')

# --- Job Generation (The "Spread" Strategy) ---
job_queue = deque()
# By looping through seeds on the outer loop, we ensure P-spread priority.
# We will see Seed 0 for every P before we see Seed 1 for any P.
for s in range(seeds):
    for P in P_values:
        job_queue.append({'P': P, 'seed': s})

def make_cmd(P, seed):
    return [
        'python3', train_script,
        '--d', str(d),
        '--P', str(P),
        '--chi', str(80), # chi = N/10
        '--kappa', '1.0',
        '--N', '800',
        '--lr', '1e-3',
        '--device', 'cuda:0',
        '--epochs', '50000000',
        '--base_seed', str(seed),
        '--ens', '5',
        '--to', 'p_scan_erf_results',
        '--eps', '0.03'
    ]

running_procs = []

print(f"Total jobs to run: {len(job_queue)}")
print(f"P values being scanned: {P_values}")

# --- Main Execution Loop ---
while job_queue or running_procs:
    # 1. Check for finished processes
    for job in running_procs[:]:
        ret = job['proc'].poll()
        if ret is not None:
            print(f"[Done] P={job['P']}, Seed={job['seed']}")
            running_procs.remove(job)

    # 2. Fill the buffer up to max_parallel_jobs
    while len(running_procs) < max_parallel_jobs and job_queue:
        next_job = job_queue.popleft()
        cmd = make_cmd(next_job['P'], next_job['seed'])
        
        print(f"[Launching] P={next_job['P']}, Seed={next_job['seed']}")
        proc = subprocess.Popen(cmd)
        
        next_job['proc'] = proc
        running_procs.append(next_job)
        time.sleep(1.0) # Short stagger to prevent I/O collisions

    time.sleep(5) # Poll every 5 seconds

print("All tasks finished.")