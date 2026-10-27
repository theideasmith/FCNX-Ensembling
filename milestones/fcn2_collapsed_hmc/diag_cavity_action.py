#!/usr/bin/env python3
"""
Exact single-neuron (cavity) action from collapsed-HMC samples, split into its pieces.

For neuron i given all other neurons (neuron_gibbs.NeuronGibbs._cond_energy), as a function of w0
with the neuron's own w_perp kept:
    U_i(w0) = w0²/(2σ_W²) + ½ log(1 + G_i) - ½ σ_A² r_i² / (1 + G_i),   G_i = σ_A² s_i - 1 ≥ 0,
    r_i(w0) = (2/T) φ(w0)ᵀ e_i,   e_i = y - Φ_{-i} z_{-i}  (cavity residual on the training set),
where G_i is the neuron's self-reaction. The mean-field action is β w0² - β (D·c(w0))², i.e. the
same prior with drive β (D·c)² and G = 0.

cavity_analysis() averages over neurons and snapshots:
    nlp_exact   -log of the average normalized conditional = the marginal p(w0), Rao-Blackwellized
    nlp_G0      the same with the self-reaction removed (prior - ½ σ_A² r²)
    drive_*     ⟨½ σ_A² r² / (1 + G)⟩ with e_i as is (exact), replaced by its least-squares projection
                on the training-set He_k(x0) (He13: k = 1,3; He1357: k = 1,3,5,7), or by the remainder
                after the He1357 projection (rest; the cross term exact - He1357 - rest is not shown)
Results are cached in plots/cache/ per (group, snapshot files, grid).

    python diag_cavity_action.py --group grok_k0p1 --snapshots 20
"""

import argparse
import glob
import hashlib
import json
from math import factorial, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.hermite_e import hermeval

from collapsed_hmc_fcn2 import make_data
from neuron_gibbs import NeuronGibbs
from runs import RUNS_ROOT

CACHE = Path(__file__).resolve().parent / "plots" / "cache"
DRIVES = ("exact", "He13", "He1357", "rest")


def group_args(group):
    """Config args of any run in the group (also before it has samples), or None."""
    for cfg_path in sorted(RUNS_ROOT.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg["args"].get("group") == group:
            return cfg["args"]
    return None


def sample_files(group, n):
    """Up to n post-burn sample files of the group, spread evenly over all its runs."""
    files = []
    for cfg_path in sorted(RUNS_ROOT.glob("*/config.json")):
        if json.loads(cfg_path.read_text())["args"].get("group") == group:
            files += sorted(glob.glob(str(cfg_path.parent / "samples" / "iter_*.pt")))
    if not files:
        return []
    idx = np.unique(np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int))
    return [files[i] for i in idx]


def hermite_basis(x0, ks):
    """(P, len(ks)) normalized He_k(x0) on the training inputs."""
    cols = []
    for k in ks:
        c = np.zeros(k + 1)
        c[k] = 1
        cols.append(hermeval(x0.cpu().numpy(), c) / sqrt(factorial(k)))
    return torch.tensor(np.stack(cols, 1), device=x0.device, dtype=torch.float64)


def projector(H):
    """Least-squares projector onto the columns of H (P, k), as a function on (P, N) matrices."""
    Q, _ = torch.linalg.qr(H)
    return lambda E: Q @ (Q.T @ E)


@torch.no_grad()
def cavity_pieces(ng, W, w0g, projs):
    """Per neuron and grid point: G = σ_A² s - 1 and ½ σ_A² r² / (1 + G) for each residual variant.

    projs maps a name to a function applied to the cavity residuals (P, N); 'exact' uses them as is.
    Returns G (N, K) and a dict of (N, K) drives.
    """
    Phi, Minv, _, zf = ng._rebuild(W)
    X0, Xp = ng.X[:, 0], ng.X[:, 1:]
    N, K = W.shape[0], len(w0g)
    zero = torch.zeros(K, device=W.device)
    G = torch.empty(N, K, dtype=torch.float64, device=W.device)
    R = {k: torch.empty_like(G) for k in projs}
    # cavity residuals e_i = y - Φ z_{-i} for all neurons at once: z_{-i} = z - M⁻¹[:, i] z_i / M⁻¹_ii, zeroed at i
    Minv_d = torch.diagonal(Minv)
    Zc = zf[:, None] - Minv * (zf / Minv_d)[None, :]
    Zc.fill_diagonal_(0.0)
    E = ng.Y.double()[:, None] - Phi.double() @ Zc                      # (P, N), column i = e_i
    Es = {k: (f(E) if f is not None else E) for k, f in projs.items()}
    for i in range(N):
        Fc = torch.erf(X0[:, None] * w0g[None, :] + (Xp @ W[i, 1:])[:, None]).double()
        _, aux = ng._cond_energy(i, Fc, zero, Phi, Minv, zf)
        G[i] = ng.sa2 * aux["s"] - 1.0
        for k, Ek in Es.items():
            r = ng.g * (Fc.T @ Ek[:, i])
            R[k][i] = 0.5 * ng.sa2 * r ** 2 / (1.0 + G[i])
    return G, R


def marginal_nlp(U, dw):
    """-log of the average of the normalized conditionals exp(-U_i)/Z_i (rows of U)."""
    U = U - U.min(1, keepdim=True).values
    p = torch.exp(-U)
    p = p / (p.sum(1, keepdim=True) * dw)
    return -torch.log(p.mean(0))


def cavity_analysis(group, snapshots=12, wmax=1.5, nw=301, device="cuda:0", verbose=True):
    """Neuron- and snapshot-averaged cavity quantities on the grid w0 ∈ [-wmax, wmax] (cached)."""
    a = group_args(group)
    files = sample_files(group, snapshots)
    if a is None or not files:
        return None
    key = hashlib.md5(json.dumps([files, wmax, nw]).encode()).hexdigest()[:12]
    path = CACHE / f"cavity_{group}_{key}.npz"
    if path.exists():
        return dict(np.load(path))
    dev = torch.device(device)
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    ng = NeuronGibbs(X, Y, a["N"], a["T"], a["s0"], a["sa0"], a["chi"])
    w0g = torch.linspace(-wmax, wmax, nw, device=dev, dtype=torch.float64)
    dw = (w0g[1] - w0g[0]).item()
    P1357 = projector(hermite_basis(X[:, 0], (1, 3, 5, 7)))
    projs = {"exact": None,
             "He13": projector(hermite_basis(X[:, 0], (1, 3))),
             "He1357": P1357,
             "rest": lambda E: E - P1357(E)}
    U_exact, U_G0, Gs, drives = [], [], [], {k: [] for k in DRIVES}
    for f in files:
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        G, R = cavity_pieces(ng, W, w0g, projs)
        prior = (0.5 * w0g ** 2 / ng.sw2)[None, :]
        U_exact.append(prior + 0.5 * torch.log1p(G) - R["exact"])
        U_G0.append(prior - R["exact"] * (1.0 + G))
        Gs.append(G.mean(0))
        for k in DRIVES:
            drives[k].append(R[k].mean(0))
    out = dict(w=w0g.cpu().numpy(), n_snap=len(files),
               nlp_exact=marginal_nlp(torch.cat(U_exact), dw).cpu().numpy(),
               nlp_G0=marginal_nlp(torch.cat(U_G0), dw).cpu().numpy(),
               G=torch.stack(Gs).mean(0).cpu().numpy(),
               **{f"drive_{k}": torch.stack(v).mean(0).cpu().numpy() for k, v in drives.items()})
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    if verbose:
        print(f"  cavity {group}: {len(files)} snapshots -> {path.name}")
    return out


def main():
    from plot_readin_action import load_group, theory_curve
    from plot_readin_action_sweep import hist_nlp

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--snapshots", type=int, default=20)
    ap.add_argument("--wmax", type=float, default=1.5)
    ap.add_argument("--nw", type=int, default=301)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a = group_args(args.group)
    cav = cavity_analysis(args.group, args.snapshots, args.wmax, args.nw, args.device)
    w, G = cav["w"], cav["G"]
    gamma = a["sa0"] * a["P"] / (a["N"] * a["T"] / 2)
    print(f"  s·P/(Nκ) = {gamma:.3f};  mean G(w0) at w0 = 0, 0.5, 1: "
          + ", ".join(f"{G[np.argmin(abs(w - x))]:.3f}" for x in (0, 0.5, 1.0)))

    W_all, _, _ = load_group(args.group, live=True)
    th_nlp, _ = theory_curve(a, 1 + 2 * a["s0"] * (a["d"] - 1) / a["d"], w)
    edges = np.linspace(-args.wmax, args.wmax, 101)
    ctr, h, err = hist_nlp(W_all[:, 0], edges)

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    ax.errorbar(ctr, h, yerr=err, fmt="o", ms=3.5, color="0.45", elinewidth=0.7, label="HMC histogram", zorder=1)
    ax.plot(w, cav["nlp_exact"], color="#2a6fdb", lw=2.5, label="exact cavity action (from samples)", zorder=3)
    ax.plot(w, cav["nlp_G0"], color="#e8a33d", lw=2.5, ls="--", label="cavity action, self-reaction G=0", zorder=2)
    ax.plot(w, th_nlp, color="#d1495b", lw=2, label="mean-field theory", zorder=2)
    top = max(np.nanmax(np.where(np.abs(w) < 0.8 * args.wmax, th_nlp, np.nan)), h.max()) + 1.5
    ax.set_ylim(min(th_nlp.min(), cav["nlp_exact"].min()) - 0.5, min(top, 25))
    ax.set_xlim(-args.wmax, args.wmax)
    ax.set_xlabel(r"$w_0$", fontsize=13)
    ax.set_ylabel(r"$-\log p(w_0)$ (normalized, no shift)", fontsize=13)
    ax.set_title(rf"$\kappa={a['T'] / 2:g}$, $d={a['d']}$, $P={a['P']}$, $N={a['N']}$, $\epsilon={a['eps']:.3g}$", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    axg.plot(w, G, color="#2a6fdb", lw=2.5)
    axg.set_xlabel(r"$w_0$", fontsize=13)
    axg.set_title(rf"self-reaction $\langle G(w_0)\rangle$   ($s P/N\kappa={gamma:.2f}$)", fontsize=13)
    axg.set_ylim(bottom=0)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / f"cavity_action_{args.group}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
