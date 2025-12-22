#!/usr/bin/env python3
"""
Polls all d_sweep runs and plots eigenvalue statistics (max, bulk mean) vs d for both linear and erf.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

run_dir = Path(__file__).parent / "runs"
dir_pat = re.compile(r"(erf|linear)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_D(?P<D>\d+)_Q(?P<Q>\d+)_([a-z0-9]+)_lr_.*")

results = {"linear": {}, "erf": {}}

for subdir in run_dir.iterdir():
    m = dir_pat.match(subdir.name)
    if not m:
        continue
    activation = m.group(1)
    d = int(m.group("d"))
    eig_path = subdir / "eigenvalues_over_time.json"
    if not eig_path.exists():
        continue
    with open(eig_path) as f:
        eigs_over_time = json.load(f)
    # Use last epoch
    last_epoch = max(map(int, eigs_over_time.keys()))
    eigs = np.array(eigs_over_time[str(last_epoch)])
    max_eig = float(np.max(eigs))
    bulk_mean = float(np.mean(eigs[1:])) if len(eigs) > 1 else float('nan')
    results[activation][d] = {"max": max_eig, "bulk_mean": bulk_mean}

# Plot
plt.figure(figsize=(8,5))
ds = sorted(set(results["linear"].keys()) | set(results["erf"].keys()))
for activation, marker, color in [("linear", "o-", "C0"), ("erf", "s-", "C1")]:
    maxs = [results[activation].get(d, {}).get("max", np.nan) for d in ds]
    bulks = [results[activation].get(d, {}).get("bulk_mean", np.nan) for d in ds]
    plt.plot(ds, maxs, marker, label=f"{activation} max eig", color=color)
    plt.plot(ds, bulks, marker+"--", label=f"{activation} bulk mean", color=color, alpha=0.7)
plt.xlabel("d")
plt.ylabel("Eigenvalue")
plt.title("Max and Bulk Mean Eigenvalue vs d (linearized erf vs linear)")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "plots/d_sweep_eigenvalues_vs_d.png", dpi=200)
plt.close()
print(f"Saved plot to plots/d_sweep_eigenvalues_vs_d.png")
