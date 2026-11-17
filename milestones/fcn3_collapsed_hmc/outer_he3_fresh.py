#!/usr/bin/env python3
"""Where the other-direction pull of the 17 large-h|He3 neurons at P = 48000 comes from, on the training set and on a
fresh set of the same size. The pull splits into: through x0 (tower to He21); the neuron's other-direction part times
the residual through x0; and its other-direction part times minus the other-direction part of the network's own
mean output. Training-only correlations (overfitting) appear in the second and vanish on fresh data.

    PYTHONPATH=.:../fcn3_matrix_theory python outer_he3_fresh.py
"""
import json, math, numpy as np, torch, plot_onset as po
from hidden_equation_test import e_of
from fcn3_sampler import make_data
from fcn3_matrix_solver import readin_features, hermite_norm
P = 48000; K = 21; ks = tuple(range(1, K + 1, 2))
dev = torch.device("cuda:0")
Xf, Yf, _, _ = make_data(150, P, -0.3 * math.sqrt(6), 1.0, 12345, dev); Xf, Yf = Xf.float(), Yf.float()
feat = lambda X, W, B: torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.float().T) @ B.float().T) for i in range(0, X.shape[0], 8192)]).double()
for r in po.by_P(po.sampler_runs())[P]:
    fs = sorted((r["path"] / "nets").glob("net_*.pt"))
    if not fs: continue
    a = json.loads((r["path"] / "config.json").read_text())["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev); X, Y = X.float(), Y.float()
    v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
    nt = torch.load(fs[-1], map_location="cpu", weights_only=False)
    W = (nt["sw"] * nt["ZW"]).to(dev).double(); B = (nt["sb"] * nt["ZB"]).to(dev).double()
    Phi = feat(X, W, B)
    M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
    Minv = torch.linalg.inv(M); ab = Minv @ (Phi.T @ Y.double() / v); lev = 1 - torch.diagonal(Minv) / sa2
    nrm = 1 + 2 * (W * W).sum(1)
    Jfull = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
    C = readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), ks); h = B @ C.T
    qh = ((B @ (Jfull - C.T @ C)) * B).sum(1); e = e_of(h, qh, ks, dev)             # (N2, S)
    fbn = ab @ e                                                                     # population He coefficients of f_bar
    o = (h[:, 1].abs() > 0.3).cpu().numpy(); inn = ~o
    gain = lambda p: (0.5 * sa2 * p ** 2 / (1 - lev)).cpu().numpy()
    md = lambda x, m: np.median(x[m])
    for lab, XX, YY, Ph in (("training", X, Y, Phi), ("fresh   ", Xf, Yf, None)):
        Ph = feat(XX, W, B) if Ph is None else Ph
        Hn = hermite_norm(ks, XX[:, 0].double())                                     # (S, P)
        fx0 = fbn @ Hn
        a_x0 = (YY.double() - fx0) / v                                               # residual through x0 only
        a_pp = -(Ph @ ab - fx0) / v                                                  # minus f_bar's other-direction part
        dphi = Ph - (e @ Hn).T                                                       # each neuron's other-direction part
        p_x0 = e @ (Hn @ (a_x0 + a_pp))
        p1, p2 = dphi.T @ a_x0, dphi.T @ a_pp
        print(f"{r['path'].name[-12:]} {lab}: 17 neurons: through x0 {md(gain(p_x0), o):5.0f}; other directions x x0-residual "
              f"{md(gain(p1), o):5.0f}, x other-direction residual {md(gain(p2), o):5.0f} (sign vs x0 part: "
              f"{md(np.sign((p1 * p_x0).cpu().numpy()), o):+.0f}, {md(np.sign((p2 * p_x0).cpu().numpy()), o):+.0f}) | ordinary: "
              f"{md(gain(p_x0), inn):.1f}; {md(gain(p1), inn):.2f}, {md(gain(p2), inn):.2f}", flush=True)
    del X, Y, Phi, M, Minv; torch.cuda.empty_cache()
