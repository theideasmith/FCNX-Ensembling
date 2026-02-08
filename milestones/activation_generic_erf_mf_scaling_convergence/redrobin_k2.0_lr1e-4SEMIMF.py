import numpy as np
import subprocess
import time
import os
from collections import deque

# --- Parameters ---
d = 150
P_values = np.logspace(np.log10(d/2), np.log10(5*d), num=8, dtype=int)
powers = np.linspace(np.log10(5*d)/np.log10(d), 1.6, num=5)[1:] # Exclude the first one since it's already in P_values

for p in powers:
    val = int(d**p)
    if val not in P_values:
        P_values = np.append(P_values, val)

P_values = np.unique(np.sort(P_values))

# Eliminate
print(P_values)
seeds = 3
max_parallel_jobs = 4  # Adjust based on your GPU memory
train_script = os.path.join(os.path.dirname(__file__), 'd_sweep_seeds.py')

from collections import deque

# 1. Ensure P_values is sorted
P_sorted = sorted(P_values)
P_interleaved = []

# 2. Interleave Max and Min
# We pop from the end (max) and the start (min) until the list is empty
temp_p = list(P_sorted)
while temp_p:
    P_interleaved.append(temp_p.pop())  # Get Max
    if temp_p:
        P_interleaved.append(temp_p.pop(0)) # Get Min

# 3. Build the queue
job_queue = deque()
for s in range(seeds):
    for P in P_interleaved:
        job_queue.append({'P': P, 'seed': s})


def make_cmd(P, seed):
    return [
        'python3', train_script,
        '--d', str(d),
        '--P', str(P),
        '--chi', str(10), # chi = N/10
        '--kappa', '2.0',
        '--N', '1600',
        '--lr', '1e-4',
        '--device', 'cuda:1',
        '--epochs', '10000000',
        '--base_seed', str(seed),
        '--ens', '1',
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