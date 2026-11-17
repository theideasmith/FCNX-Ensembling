#!/usr/bin/env python3
"""Start-swap test at fixed theory (d = 150, kappa/P fixed: P = 24000 at kappa = 0.015 and P = 48000 at kappa = 0.03).

Left (weak side wells): P = 24000, started from the prior and from the final P = 48000 network.
Right (strong side wells): P = 48000, started from the prior and from the final P = 24000 network.
Each panel: -log p of h|He3 over every hidden neuron of every network saved after burn-in, with the theory (the
same for both P). The legend gives the number of hidden neurons with |h3| > 0.25 (and |h1| <= 1.5) in the last
network, and test MSE / training MSE of the last network (readout at its posterior mean; 1 = no overfitting).

    python plot_init_swap.py [--device cuda:0]       -> plots/onset_checks/init_swap_he3_d150.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
from fcn3_matrix_solver import readin_features  # noqa: E402
from fcn3_sampler import make_data  # noqa: E402

RUNS = HERE / "runs"
R24 = "20261109-170459_onset_d150_P24000_N1500x1500_prior_r10"
R48 = "20261110-120502_ptheory_d150_P48000_N1500x1500_prior_r10"
R48_FROM24 = "20261115-143607_from24k_d150_P48000_N1500x1500_prior_r212"
R24_FROM48 = "20261115-105757_from48k_d150_P24000_N1500x1500_prior_r211"
THEORY = HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"     # P = 24000, kappa = 0.015 (same kappa/P)
OUT = HERE / "plots" / "onset_checks" / "init_swap_he3_d150.png"
COL = {"prior": "#2a78d6", "other": "#eb6834"}     # by starting point
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"


def hidden(run):
    """(h1, h3) of every hidden neuron in every saved network; and the group count in the last one."""
    H, last = [], None
    for f in sorted((RUNS / run / "nets").glob("net_*.pt")):
        nt = torch.load(f, map_location="cpu", weights_only=False)
        ZW, ZB = nt["ZW"].double(), nt["ZB"].double()
        w0, q = nt["sw"] * ZW[:, 0], nt["sw"] ** 2 * (ZW[:, 1:] ** 2).sum(1)
        last = ((nt["sb"] * ZB) @ readin_features(w0, q, (1, 3)).T).numpy()
        H.append(last)
    n = int(((np.abs(last[:, 1]) > 0.25) & (np.abs(last[:, 0]) <= 1.5)).sum())
    return np.concatenate(H), n


def overfit(run, dev):
    """test MSE / training MSE of the run's last network, readout at its posterior mean given the hidden layer."""
    a = json.loads((RUNS / run / "config.json").read_text())["args"]
    ck = torch.load(RUNS / run / "checkpoint.pt", map_location="cpu", weights_only=False)
    d, P = a["d"], a["P"]
    W = (ck["ZW"].double() * a["sigw"] / math.sqrt(d)).to(dev).float()
    B = (ck["ZB"].double() * a["sigB"] / math.sqrt(a["N1"])).to(dev).float()
    X, Y, _, _ = make_data(d, P, a["eps"], 1.0, a["seed"], dev)
    Xt, Yt, _, _ = make_data(d, 100000, a["eps"], 1.0, 12345, dev)
    feats = lambda X_: torch.cat([torch.erf(torch.erf(X_[i:i + 8192].float() @ W.T) @ B.T)
                                  for i in range(0, len(X_), 8192)]).double()
    v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
    Phi = feats(X)
    abar = torch.linalg.solve(Phi.T @ Phi / v + torch.eye(a["N2"], dtype=torch.float64, device=dev) / sa2,
                              Phi.T @ Y.double() / v)
    tr = float(((Y.double() - Phi @ abar) ** 2).mean())
    te = float(((Yt.double() - feats(Xt) @ abar) ** 2).mean())
    return te / tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    dev = torch.device(ap.parse_args().device)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    z = np.load(THEORY)
    g, yt = z["h3g_24000"], -z["lp3_24000"]
    panels = [("P = 24000, κ = 0.015: weak side wells", [(R24, 24000, "prior"), (R24_FROM48, 24000, "other")]),
              ("P = 48000, κ = 0.03: strong side wells", [(R48, 48000, "prior"), (R48_FROM24, 48000, "other")])]
    data = {}
    for _, rs in panels:
        for run, P, _ in rs:
            h, n = hidden(run)
            data[run] = (h[:, 1], n, overfit(run, dev))
            print(f"{run}: {n} neurons with |h3| > 0.25 in the last network, test/train MSE {data[run][2]:.2f}")
    lim = 1.08 * max(np.abs(v[0]).max() for v in data.values())
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    edges = np.linspace(-lim, lim, 81)
    mid, wid = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    for ax, (title, rs) in zip(axs, panels):
        k = np.abs(g) <= lim
        ax.plot(g[k], yt[k], color=INK, lw=1.6, zorder=2, label="theory (same in both panels)")
        lo = yt[k].min()
        for (run, P, how), mk in zip(rs, ("o", "s")):
            x, n, of = data[run]
            cnt, _ = np.histogram(x, edges)
            m = cnt > 0
            y = -np.log(cnt[m] / (cnt.sum() * wid[m]))
            src = "from the prior" if how == "prior" else f"from the final P = {72000 - P} network"
            ax.errorbar(mid[m], y, yerr=1 / np.sqrt(cnt[m]), fmt=mk, ms=4.5, color=COL[how], mec="white", mew=0.4,
                        elinewidth=0.7, capsize=0, zorder=3,
                        label=f"started {src}: {n} neurons with |h$_3$| > 0.25, test/train MSE {of:.2f}")
            lo = min(lo, y.min())
        for s in (-0.25, 0.25):
            ax.axvline(s, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lo - 0.5, lo + 13)
        ax.set_xlabel("h | He$_3$  (hidden neuron)")
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axs[0].set_ylabel("−log p")
    fig.suptitle("Hidden-layer He$_3$ distribution, d = 150: P = 24000 vs 48000 at the same theory (κ ∝ P)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
