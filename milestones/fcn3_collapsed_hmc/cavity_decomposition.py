#!/usr/bin/env python3
"""What does each hidden neuron gain and pay in the sampler's exact (collapsed) posterior, and which part of it does the
theory keep?

With the readout a integrated out, the sampler's energy of (W, B) contains, through M = Phi^T Phi / v + I / s_a2
(Phi: hidden-layer outputs erf(h_k(x_mu)) on the P training points, v = kappa/chi, s_a2 = sigma_A^2/(N2 chi)),
    U = ... + 1/2 (y^T y / v - b^T M^-1 b) + 1/2 log det M + ...,   b = Phi^T y / v.
Removing hidden neuron k changes U by exactly (rank-one formulas, s_k = [M^-1]_kk, z = M^-1 b)
    fit gain           F_k = 1/2 z_k^2 / s_k            (lowers U: the neuron helps fit y)
    fluctuation cost   C_k = -1/2 log(s_k / s_a2) >= 0  (raises U: the readout fluctuations it adds)
The theory's hidden-field action keeps a fit term, 1/2 g~ (t.e(h))^2, and drops the fluctuation term: its output-dual
covariance T = chi (sigma_A^2 K + kappa/P)^-1 - chi^2 t t^T is replaced by -chi^2 t t^T ("leading O(chi^2) part",
notes/fcn3_matrix_eos.tex), although chi = 1 here. If C_k is small and flat in h, dropping it is harmless; if it is
comparable to F_k and depends on h, the theory's hidden layer is missing a force.

For every hidden neuron of the last few saved networks of the dataset-0 onset run at each P: F_k, C_k, and the
neuron's He1 and He3 components; binned summaries are printed and plotted.

    python cavity_decomposition.py [--Ps 750 3000 12000 24000 48000] [--nets 3]
        ->  plots/onset_checks/cavity_decomposition_d150.png, plots/cache/cavity_decomposition_d150.npz
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_matrix_solver import readin_features
from fcn3_sampler import data_fingerprint, make_data

HERE = Path(__file__).resolve().parent


def decompose(X, Y, W, B, v, sa2, chunk=8192):
    P = X.shape[0]
    N2 = B.shape[0]
    G = torch.zeros(N2, N2, dtype=torch.float64, device=X.device)
    r = torch.zeros(N2, dtype=torch.float64, device=X.device)
    for i in range(0, P, chunk):
        F2 = torch.erf(torch.erf(X[i:i + chunk] @ W.T) @ B.T).double()
        G += F2.T @ F2
        r += F2.T @ Y[i:i + chunk].double()
    M = G / v + torch.eye(N2, dtype=torch.float64, device=X.device) / sa2
    Minv = torch.linalg.inv(M)
    z = Minv @ (r / v)
    s = torch.diagonal(Minv)
    F = 0.5 * z * z / s
    C = -0.5 * torch.log(s / sa2)                     # s / sa2: posterior over prior variance of a_k
    return F.cpu().numpy(), C.cpu().numpy(), (torch.diagonal(G) / P).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[750, 3000, 12000, 24000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    groups = po.by_P(po.sampler_runs())
    out = {}
    for P in a.Ps:
        r0 = [r for r in groups[P] if r["seed"] == 0 and (r["path"] / "nets").exists()][0]
        cfg = json.loads((r0["path"] / "config.json").read_text())
        args = cfg["args"]
        X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
        X, Y = X.float(), Y.float()
        if "data_fingerprint" in cfg:
            assert data_fingerprint(X) == cfg["data_fingerprint"], "training set does not match the run's"
        v = args["kappa"] / args["chi"]
        sa2 = args["sigA"] ** 2 / (args["N2"] * args["chi"])
        rows = []
        for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
            nt = torch.load(f, map_location="cpu", weights_only=False)
            W = (nt["sw"] * nt["ZW"]).to(dev).float()
            B = (nt["sb"] * nt["ZB"]).to(dev).float()
            F, C, phi2 = decompose(X, Y, W, B, v, sa2)
            Wd = W.double()
            c = readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), (1, 3))
            h = (B.double() @ c.T).cpu().numpy()
            rows.append(np.column_stack([h[:, 0], h[:, 1], F, C, phi2]))
        R = np.concatenate(rows)
        out[P] = R
        h1, h3, F, C, phi2 = R.T
        print(f"P={P}  (s_a2 P / v = sigma_A^2 P/(kappa N2) = {sa2 * P / v:.2f}; {len(R)} hidden neurons)")
        print(f"   fit gain F: mean {F.mean():.3f}  sd {F.std():.3f}  max {F.max():.2f}    "
              f"fluctuation cost C: mean {C.mean():.3f}  sd {C.std():.3f}  max {C.max():.2f}")
        for name, x, edges in (("|h1|", np.abs(h1), [0, 0.1, 0.2, 0.3, 0.45, 0.7, 1.2, 3.0]),
                               ("|h3|", np.abs(h3), [0, 0.03, 0.06, 0.09, 0.13, 0.2, 0.3, 0.6])):
            line = f"   by {name}:"
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (x >= lo) & (x < hi)
                if m.sum() >= 3:
                    line += f"  [{lo:.2f},{hi:.2f}) n={m.sum()}: F {F[m].mean():.2f} C {C[m].mean():.2f}"
            print(line)
    np.savez(HERE / "plots" / "cache" / "cavity_decomposition_d150.npz", **{f"P{P}": R for P, R in out.items()})

    plt = po.style()
    fig, axs = plt.subplots(2, len(out), figsize=(3.8 * len(out), 7.2), squeeze=False)
    for c, (P, R) in enumerate(out.items()):
        h1, h3, F, C, phi2 = R.T
        for i, (x, xl) in enumerate(((h1, "h | He$_1$"), (h3, "h | He$_3$"))):
            ax = axs[i, c]
            ax.scatter(x, F, s=3, color=po.C_THEORY, alpha=0.35, label="fit gain (kept by the theory)")
            ax.scatter(x, C, s=3, color=po.C_HMC, alpha=0.35, label="fluctuation cost (dropped by the theory)")
            ax.set_xlabel(xl)
            ax.set_yscale("symlog", linthresh=0.01)
            if c == 0:
                ax.set_ylabel("per-neuron change of the energy (nats)")
            if i == 0:
                ax.set_title(f"P = {P}", color=po.INK, loc="left", fontsize=12)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=10, markerscale=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Sampler's exact posterior, one point per hidden neuron (last saved networks of dataset 0): what each "
                 "hidden neuron gains in fit and pays in readout fluctuations", color=po.INK, x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "onset_checks" / "cavity_decomposition_d150.png", dpi=140, bbox_inches="tight")
    print("wrote", HERE / "plots" / "onset_checks" / "cavity_decomposition_d150.png")


if __name__ == "__main__":
    main()
