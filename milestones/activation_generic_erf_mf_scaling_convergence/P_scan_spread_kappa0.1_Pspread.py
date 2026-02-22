import numpy as np
import subprocess
import time
import os
import heapq
from collections import deque

# 1. Generate all possible combinations
d_values = [150] #np.unique(np.sort(np.logspace(np.log10(50), np.log10(250), num=3, dtype=int)))
P_values = np.unique(np.sort(np.logspace(np.log10(min(d_values)), np.log10(20*max(d_values)), num=10, dtype=int)))
seeds = range(10)[1:]
N = 1600
chi = 10.0

all_jobs = []
for p_val in P_values:
    for d_val in d_values:
        for s in seeds:
            all_jobs.append({'P': p_val, 'd': d_val, 'seed': s})

# 2. Assign a "Spread Priority"
# We use a simple trick: assign a score based on the fractional binary representation 
# of the index. This mimics a Van der Corput sequence for high spread.
def get_priority(index, total):
    # This effectively interleaves the space
    return bin(index).replace('0b', '').zfill(10)[::-1]

# Sort jobs to maximize distance between consecutive picks
# We'll use a combination of normalized rank and seed staggering
n_p = len(P_values)
n_d = len(d_values)

scored_jobs = []
for i, p_val in enumerate(P_values):
    for j, d_val in enumerate(d_values):
        for s in seeds:
            # The priority key: (Seed first to get one of each seed ASAP, 
            # then a bit-reversal style spread for P and d)
            # We use a tuple for the priority to handle tie-breaking
            priority = (s, (i % 2), (j % 2), i, j) 
            scored_jobs.append((priority, {'P': p_val, 'd': d_val, 'seed': s}))

# Sort by our spread-based priority
scored_jobs.sort(key=lambda x: x[0])
job_queue = deque([job[1] for job in scored_jobs])

lr_base = (1e-1 / 3000) 
train_script = 'd_sweep.py'
def make_cmd(d, P, seed):
    return [
        'python3', train_script,
        '--d', str(d),
        '--P', str(P),
        '--chi', str(int(chi)), # chi = N/10
        '--kappa',str( 0.1 ),
        '--N', str(int(N)),
        '--lr', str(lr_base * P), # So that P-adjusted learning rate remain constant
        '--device', 'cuda:0',
        '--epochs', '700000',
        '--seed', str(seed),
        '--ens', '10',
        '--to', 'P_scan_KAPPA0.1_FINALIZED',
        '--eps', '0.03'
    ]

running_procs = []
max_parallel_jobs = 8
print(f"Total jobs to run: {len(job_queue)}")
print(f"P values being scanned: {P_values}")

# --- Main Execution Loop ---
while job_queue or running_procs:
    # 1. Check for finished processes
    for job in running_procs[:]:
        ret = job['proc'].poll()
        if ret is not None:
            print(f"[Done] P={job['P']}, d={job['d']}, Seed={job['seed']}")
            running_procs.remove(job)

    # 2. Fill the buffer up to max_parallel_jobs
    while len(running_procs) < max_parallel_jobs and job_queue:
        next_job = job_queue.popleft()
        cmd = make_cmd(next_job['d'], next_job['P'], next_job['seed'])
        
        print(f"[Launching] P={next_job['P']}, Seed={next_job['seed']}")
        proc = subprocess.Popen(cmd)
        
        next_job['proc'] = proc
        running_procs.append(next_job)
        time.sleep(1.0) # Short stagger to prevent I/O collisions

    time.sleep(5) # Poll every 5 seconds

print("All tasks finished.")