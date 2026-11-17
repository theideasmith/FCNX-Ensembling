#!/usr/bin/env python3
"""Does a higher x0 Hermite tower capture the pull of the 17 large-h|He3 neurons at P = 48000 (outer_he3_group.py)?
Splits each hidden neuron's exact pull into the part through x0 (tower to He11, 21, 31, 41) and the rest.

    PYTHONPATH=.:../fcn3_matrix_theory python outer_he3_tower.py
"""
import json, math, numpy as np, torch, plot_onset as po
from hidden_equation_test import e_of
from fcn3_sampler import make_data
from fcn3_matrix_solver import readin_features, hermite_norm
P = 48000
dev = torch.device("cuda:0")
for r in po.by_P(po.sampler_runs())[P]:
    fs = sorted((r["path"] / "nets").glob("net_*.pt"))
    if not fs: continue
    a = json.loads((r["path"] / "config.json").read_text())["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev); X, Y = X.float(), Y.float()
    v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
    nt = torch.load(fs[-1], map_location="cpu", weights_only=False)
    W = (nt["sw"] * nt["ZW"]).to(dev).double(); B = (nt["sb"] * nt["ZB"]).to(dev).double()
    Phi = torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.float().T) @ B.float().T) for i in range(0, X.shape[0], 8192)]).double()
    M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
    Minv = torch.linalg.inv(M)
    alpha = (Y.double() - Phi @ (Minv @ (Phi.T @ Y.double() / v))) / v
    lev = 1 - torch.diagonal(Minv) / sa2
    pull = Phi.T @ alpha
    nrm = 1 + 2 * (W * W).sum(1)
    Jfull = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
    h3 = (B @ readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), (3,)).T)[:, 0]
    o = (h3.abs() > 0.3).cpu().numpy(); inn = ~o
    gain = lambda p: (0.5 * sa2 * p ** 2 / (1 - lev)).cpu().numpy()
    F = gain(pull)
    line = f"{r['path'].name[-12:]}: total gain, 17 neurons {np.median(F[o]):.0f}, ordinary {np.median(F[inn]):.1f}"
    for K in (11, 21, 31, 41):
        ks = tuple(range(1, K + 1, 2))
        C = readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), ks)
        h = B @ C.T
        qh = ((B @ (Jfull - C.T @ C)) * B).sum(1)
        e = e_of(h, qh, ks, dev)
        p0 = e @ (hermite_norm(ks, X[:, 0].double()) @ alpha)
        frac = ((pull - p0).abs() / pull.abs()).cpu().numpy()
        line += (f"\n   tower to He{K}: through x0 {np.median(gain(p0)[o]):.0f}, other directions {np.median(gain(pull - p0)[o]):.0f}"
                 f" (share of the pull {np.median(frac[o]):.2f}); q_h {np.median(qh.cpu().numpy()[o]):.3f}; "
                 f"ordinary: x0 {np.median(gain(p0)[inn]):.1f}, other {np.median(gain(pull - p0)[inn]):.2f}")
    print(line, flush=True)
    del X, Y, Phi, M, Minv; torch.cuda.empty_cache()
