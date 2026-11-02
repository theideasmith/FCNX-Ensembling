"""
Per-chain HMC observables, for error bars from the chain-to-chain spread.

Each run of a group is one chain. A quantity is computed separately in every chain; the plotted value is the mean
over chains and the error bar the standard error across chains, std(ddof=1)/sqrt(n_chains) (for two chains, half
their difference). At d >= 300 the chains have not mixed in the aligned fraction, so this is a lower bound on the
true uncertainty there. Predictor coefficients are cached in plots/cache (they need the training data and a GPU).
"""

import glob
import json
from pathlib import Path

import numpy as np
import torch

from analyze_large_d import a_k
from collapsed_hmc_fcn2 import make_data
from runs import RUNS_ROOT

CACHE = Path(__file__).resolve().parent / "plots" / "cache"


def chain_runs(group):
    """[(label, run_dir, args)] for the runs of a group that have samples, in a fixed order. The label is
    init:replica (e.g. prior:r2), unique even when a group has several runs with the same initialization."""
    out = []
    for cp in sorted(RUNS_ROOT.glob("*/config.json")):
        a = json.loads(cp.read_text())["args"]
        if a.get("group") == group and glob.glob(str(cp.parent / "samples" / "iter_*.pt")):
            out.append((f"{a['init']}:{cp.parent.name.rsplit('_', 1)[-1]}", cp.parent, a))
    labels = [o[0] for o in out]                       # two file-initialized runs (e.g. theory and over-aligned
    for k, (lab, run, a) in enumerate(out):            # starts) share init:replica; add the run tag
        if labels.count(lab) > 1:
            out[k] = (f"{lab}:{a.get('tag', k)}", run, a)
    return out


def chain_weights(group):
    """{label: W of shape (snapshots, N, d)} with all post-burn samples of each chain."""
    out = {}
    for init, run, a in chain_runs(group):
        Ws = [torch.load(f)["W0"].double().numpy().reshape(a["N"], a["d"])
              for f in sorted(glob.glob(str(run / "samples" / "iter_*.pt")))]
        out[init] = np.stack(Ws)
    return out


def mean_se(values):
    """Mean over chains and its standard error (0 for a single chain)."""
    v = np.asarray(values, float)
    return v.mean(0), (v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros_like(v.mean(0)))


@torch.no_grad()
def chain_fbar(group, ks=(1, 3, 5, 7, 9, 11), n_snap=15, device="cuda:1"):
    """{label: fbar_k averaged over n_snap evenly spaced snapshots of that chain}: exact population He_k(x0)
    coefficients of the posterior-mean predictor (as analyze_large_d.hmc_exact, but per chain)."""
    key = CACHE / f"chain_fbar_{group}_{n_snap}_{'-'.join(map(str, ks))}.json"
    runs = chain_runs(group)
    if key.exists():
        cached = json.loads(key.read_text())
        if set(cached) == {i for i, _, _ in runs}:
            return {i: np.array(v) for i, v in cached.items()}
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    a = runs[0][2]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev)
    X, Y = X.double(), Y.double()
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    out = {}
    for init, run, _ in runs:
        files = sorted(glob.glob(str(run / "samples" / "iter_*.pt")))
        idx = np.unique(np.linspace(0, len(files) - 1, min(n_snap, len(files))).round().astype(int))
        fk = []
        for i in idx:
            W = torch.load(files[i])["W0"].to(dev).double().reshape(-1, a["d"])
            Phi = torch.erf(X @ W.T)
            M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
            A = torch.linalg.solve(M, Phi.T @ Y / hT)
            s = W.norm(dim=1); u0 = W[:, 0] / s
            fk.append([(A * a_k(s, k) * u0 ** k).sum().item() for k in ks])
            del Phi, M
        out[init] = np.mean(fk, 0)
        print(f"  {group} {init}: fbar {np.round(out[init], 4)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({i: v.tolist() for i, v in out.items()}))
    return out
