#!/usr/bin/env python3
"""
Plot eigenvalues of the readin layer weights covariance matrix for linear networks against theory.
- Loads all model.pt files in d*_P*_N*_chi* directories
- Reads d, P, N, chi from directory name using regex
- Computes empirical covariance eigenvalues for each ensemble member
- Aggregates by d: plots max eigenvalue and mean of bulk vs d with error bars
- Overlays theory lines for lWT and lWP
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import collections
from lib.Experiment import ExperimentLinear
import juliacall
from juliacall import Main as jl
from FCN3Network import FCN3NetworkEnsembleLinear
jl.include('/home/akiva/FCNX-Ensembling/julia_lib/FCSLinear.jl')

# Regex to extract parameters from directory name
dir_pat = re.compile(r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi(?P<chi>\d+)")

base_dir = Path(__file__).parent
run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and dir_pat.match(p.name)]

all_eigs = []
all_params = []

for run_dir in run_dirs:
    m = dir_pat.match(run_dir.name)
    d = int(m.group("d"))
    P = int(m.group("P"))
    N = int(m.group("N"))
    chi = int(m.group("chi"))
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        continue
    # Load model
    state = torch.load(model_path, map_location="cpu")
    # Try to infer ensemble dimension from checkpoint if possible
    ens = 1
    if "W0" in state:
        w0_shape = state["W0"].shape
        if len(w0_shape) == 3:
            ens = w0_shape[0]
    try:
        model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=ens, weight_initialization_variance=(1/d, 1/N, 1/(N*chi)))
        model.load_state_dict(state)
    except RuntimeError as e:
        # Try to reload with correct ensemble dimension if possible
        if "size mismatch" in str(e) and "W0" in state:
            w0_shape = state["W0"].shape
            if len(w0_shape) == 3:
                ens = w0_shape[0]
                model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=ens, weight_initialization_variance=(1/d, 1/N, 1/(N*chi)))
                model.load_state_dict(state)
            else:
                raise
        else:
            raise
    # Get W0 (readin weights): shape (ens, N, d)
    W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
    # Compute covariance for each ensemble member
    for i in range(ens):
        cov = np.cov(W0[i], rowvar=False)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)[::-1]
        all_eigs.append(eigs)
        all_params.append((d, P, N, chi))

# --- Aggregate by d: plot max and bulk mean eigenvalue vs d ---
by_d = collections.defaultdict(list)
for eigs, params in zip(all_eigs, all_params):
    d = params[0]
    eigs = np.asarray(eigs)
    if eigs.size < 2:
        continue
    max_eig = eigs[0]
    bulk_mean = eigs[1:].mean()
    by_d[d].append((max_eig, bulk_mean))

ds = sorted(by_d.keys())
max_eigs = []
bulk_means = []
max_eigs_err = []
bulk_means_err = []
for d in ds:
    vals = np.array(by_d[d])
    max_eigs.append(vals[:,0].mean())
    bulk_means.append(vals[:,1].mean())
    max_eigs_err.append(vals[:,0].std())
    bulk_means_err.append(vals[:,1].std())

# Compute theory using FCSLinear solvers
theory_by_d = {}
for d in ds:
    for params in all_params:
        if params[0] == d:
            P, N, chi = params[1], params[2], params[3]
            kappa = 1.0   # Assuming kappa = 1/d as in standard
            # Target regime (delta=1)
            i0 = [1.0, 1.0]
            i0 = juliacall.convert( jl.Vector[jl.Float64], i0)

            sol_T = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=chi, d=d, kappa=kappa, delta=1.0, n1=N, n2=N, P=P,
                verbose=False, anneal=True
            )
            if sol_T is not None:
                print(f"sol_T: {sol_T}")
                lJT, lHT = sol_T[0], sol_T[1]
                lHh_T = 1.0 / lJT - lHT / (lJT ** 2)
                lWT = 1.0 / (d + (N / N) * lHh_T)  # n2/n1 = 1
            else:
                print(f"  Warning: No solution found for d={d}, P={P}, N={N}, chi={chi} in target regime")
                lWT = np.nan
            # Population regime (delta=0)
            sol_P = jl.FCSLinear.nlsolve_solver(
                i0,
                chi=chi, d=d, kappa=kappa, delta=0.0, n1=N, n2=N, P=P,
                verbose=False, anneal=True
            )
            if sol_P is not None:
                lJP, lHP = sol_P[0], sol_P[1]
                lHh_P = 1.0 / lJP - lHP / (lJP ** 2)
                lWP = 1.0 / (d + (N / N) * lHh_P)
            else:
                lWP = np.nan
            theory_by_d[d] = (lWT, lWP)
            break

lWTs = [theory_by_d[d][0] for d in ds]
lWPs = [theory_by_d[d][1] for d in ds]

plt.figure(figsize=(8,5))
plt.errorbar(ds, max_eigs, yerr=max_eigs_err, fmt='o-', label='Max eigenvalue (empirical)', capsize=3)
plt.errorbar(ds, bulk_means, yerr=bulk_means_err, fmt='s-', label='Bulk mean (empirical)', capsize=3)
plt.plot(ds, lWTs, '--', color='C1', label='Theory lWT')
plt.plot(ds, lWPs, ':', color='C2', label='Theory lWP')
plt.yscale('log')
plt.xlabel('d')
plt.ylabel('Eigenvalue (weight covariance)')
plt.title('Max and Bulk Mean Eigenvalue vs d (Linear, all runs)')
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "plots/weight_cov_max_and_bulk_vs_d_linear.png", dpi=200)
plt.close()
print(f"Saved plot to {base_dir / 'plots/weight_cov_max_and_bulk_vs_d_linear.png'}")
