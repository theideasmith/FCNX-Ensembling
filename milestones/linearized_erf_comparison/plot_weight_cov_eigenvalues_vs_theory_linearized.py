#!/usr/bin/env python3
"""
Plot eigenvalues of the readin layer weights covariance matrix for both linear and erf networks (with rescaling for erf), vs theory.
- Loads all model.pt files in runs/ for both activations
- Aggregates by d and activation
- Plots max and bulk mean eigenvalue vs d, overlays theory
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import collections
from lib.FCN3Network import FCN3NetworkEnsembleLinear
from lib.FCN3Network import FCN3NetworkEnsembleErf
from lib.Experiment import ExperimentLinear

# Regex to extract parameters from directory name
dir_pat = re.compile(r"(erf|linear)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_D(?P<D>\d+)_Q(?P<Q>\d+)_([a-z0-9]+)_lr_.*")

base_dir = Path(__file__).parent
run_dirs = [p for p in (base_dir / "runs").iterdir() if p.is_dir() and dir_pat.match(p.name)]

all_eigs = {"linear": collections.defaultdict(list), "erf": collections.defaultdict(list)}
all_params = {"linear": {}, "erf": {}}

for run_dir in run_dirs:
    m = dir_pat.match(run_dir.name)
    activation = m.group(1)
    d = int(m.group("d"))
    P = int(m.group("P"))
    N = int(m.group("N"))
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        continue
    state = torch.load(model_path, map_location="cpu")
    ens = 1
    if "W0" in state:
        w0_shape = state["W0"].shape
        if len(w0_shape) == 3:
            ens = w0_shape[0]
    if activation == "linear":
        model = FCN3NetworkEnsembleLinear(d, N, N, P, ensembles=ens, weight_initialization_variance=(1/d, 1/N, 1/(N*N)))
    else:
        # Rescaled for erf
        model = FCN3NetworkEnsembleErf(d, N, N, P, ensembles=ens, weight_initialization_variance=(3*np.pi/(4*d), 1/N, 1/(N*N)))
    model.load_state_dict(state)
    W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
    for i in range(ens):
        cov = np.cov(W0[i], rowvar=False)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)[::-1]
        all_eigs[activation][d].append(eigs)
        all_params[activation][d] = (P, N)

# Aggregate and plot
plt.figure(figsize=(8,5))
for activation, color, marker in [("linear", "C0", "o"), ("erf", "C1", "s")]:
    ds = sorted(all_eigs[activation].keys())
    max_eigs = []
    bulk_means = []
    max_eigs_err = []
    bulk_means_err = []
    for d in ds:
        eigs_arr = np.array(all_eigs[activation][d])
        maxs = eigs_arr[:,0]
        bulks = eigs_arr[:,1:].mean(axis=1)
        max_eigs.append(maxs.mean())
        bulk_means.append(bulks.mean())
        max_eigs_err.append(maxs.std()/np.sqrt(len(maxs)))
        bulk_means_err.append(bulks.std()/np.sqrt(len(bulks)))
    # Theory
    lWTs = []
    lWPs = []
    for d in ds:
        P, N = all_params[activation][d]
        exp = ExperimentLinear(d=d, P=P, N=N, chi=N, ens=1)
        preds = exp.eig_predictions()
        lHT = preds.lHT
        lHh = 1.0 / preds.lJT - lHT / (preds.lJT ** 2)
        lWT = 1.0 / (d + lHh)
        lWTs.append(lWT)
    plt.errorbar(ds, max_eigs, yerr=max_eigs_err, fmt=marker+'-', color=color, label=f"{activation} max eig")
    plt.errorbar(ds, bulk_means, yerr=bulk_means_err, fmt=marker+'--', color=color, label=f"{activation} bulk mean")
    plt.plot(ds, lWTs, linestyle=':', color=color, label=f"{activation} theory lWT")
plt.xlabel("d")
plt.ylabel("Eigenvalue (weight covariance)")
plt.title("Max and Bulk Mean Eigenvalue vs d (linearized erf vs linear, with theory)")
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "plots/weight_cov_max_and_bulk_vs_d_linearized.png", dpi=200)
plt.close()
print(f"Saved plot to {base_dir / 'plots/weight_cov_max_and_bulk_vs_d_linearized.png'}")
