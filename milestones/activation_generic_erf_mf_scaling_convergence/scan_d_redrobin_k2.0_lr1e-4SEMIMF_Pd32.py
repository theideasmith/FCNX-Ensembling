import numpy as np
import subprocess
import time
import os
from collections import deque

# --- Parameters ---
# Log-spaced d values from 50 to 400 (inclusive), about as many as before
num_d_points = 10
d_values = np.unique(np.round(np.logspace(np.log10(50), np.log10(400), num=num_d_points)).astype(int))

seeds = 1
max_parallel_jobs = 10  # Adjust based on your GPU memory
train_script = os.path.join(os.path.dirname(__file__), 'd_sweep_seeds.py')

# 1. Interleave d values (max/min)
d_sorted = sorted(d_values)
d_interleaved = []
temp_d = list(d_sorted)
while temp_d:
    d_interleaved.append(temp_d.pop())  # Max
    if temp_d:
        d_interleaved.append(temp_d.pop(0))  # Min
kappas = []
# 2. Build the queue
job_queue = deque()
for s in range(seeds):
    for d in d_interleaved:
        P = int(0.5 * d ** 1.5)
        kappa = P / 600  # You may want to adjust denominator for your experiment
        kappas.append(kappa)
        job_queue.append({'d': d, 'P': P, 'kappa': kappa, 'seed': s})

def make_cmd(d, P, kappa, seed):
    return [
        'python3', train_script,
        '--d', str(d),
        '--P', str(P),
        '--chi', str(10), # chi = N/10
        '--kappa', str(kappa),
        '--N', '1600',
        '--lr', '1e-4',
        '--device', 'cuda:1',
        '--epochs', '5000000',
        '--base_seed', str(seed),
        '--ens', '1',
        '--to', 'd_scan_erf_results_Pd32',
        '--eps', '0.03'
    ]

running_procs = []

print(f"Total jobs to run: {len(job_queue)}")
print(f"d values being scanned: {d_values}")
print(f"kappa values being used: {kappas}")

# --- Main Execution Loop ---
while job_queue or running_procs:
    # 1. Check for finished processes
    for job in running_procs[:]:
        ret = job['proc'].poll()
        if ret is not None:
            print(f"[Done] d={job['d']}, P={job['P']}, Seed={job['seed']}")
            running_procs.remove(job)

    # 2. Fill the buffer up to max_parallel_jobs
    while len(running_procs) < max_parallel_jobs and job_queue:
        next_job = job_queue.popleft()
        cmd = make_cmd(next_job['d'], next_job['P'], next_job['kappa'], next_job['seed'])
        print(f"[Launching] d={next_job['d']}, P={next_job['P']}, Seed={next_job['seed']}")
        proc = subprocess.Popen(cmd)
        next_job['proc'] = proc
        running_procs.append(next_job)
        time.sleep(1.0) # Short stagger to prevent I/O collisions

    time.sleep(5) # Poll every 5 seconds

print("All tasks finished.")
