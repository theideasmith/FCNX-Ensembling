#!/usr/bin/env python3
"""Training vs test error of the onset runs (dataset 0, last saved network per P), He projections of the training and
population residuals, and each hidden neuron's exact pull (training residual) against the theory's form with the population
residual.   python overfit_check.py   (FINDINGS_onset.md)
"""
import json, math, numpy as np, torch
import plot_onset as po
from hidden_equation_test import e_of
from fcn3_matrix_solver import readin_features, hermite_norm
from fcn3_sampler import make_data
dev = torch.device("cuda:0")
groups = po.by_P(po.sampler_runs())
z = dict(np.load(po.HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
ks = (1, 3, 5, 7, 9, 11)
Xt, Yt, _, _ = make_data(150, 100000, -0.3 * math.sqrt(6), 1.0, 12345, dev)      # fresh test set
Xt, Yt = Xt.float(), Yt.float()
x0t = Xt[:, 0].double()
for P in (750, 3000, 12000, 24000, 48000):
    r0 = [r for r in groups[P] if r["seed"] == 0 and (r["path"] / "nets").exists()][0]
    a = json.loads((r0["path"] / "config.json").read_text())["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev); X, Y = X.float(), Y.float()
    v, sa2, lam = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"]), a["kappa"] / a["P"]
    nt = torch.load(sorted((r0["path"] / "nets").glob("net_*.pt"))[-1], map_location="cpu", weights_only=False)
    W = (nt["sw"] * nt["ZW"]).to(dev).float(); B = (nt["sb"] * nt["ZB"]).to(dev).float()
    Phi = torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.T) @ B.T) for i in range(0, P, 8192)]).double()
    G = Phi.T @ Phi; M = G / v + torch.eye(1500, dtype=torch.float64, device=dev) / sa2
    abar = torch.linalg.solve(M, Phi.T @ Y.double() / v)
    rtr = Y.double() - Phi @ abar                                                 # training residual
    ftest = torch.cat([torch.erf(torch.erf(Xt[i:i + 8192] @ W.T) @ B.T).double() @ abar for i in range(0, len(Xt), 8192)])
    rte = Yt.double() - ftest
    # He projections of the residuals: training-set sums vs population (fresh test set)
    phin_tr = hermite_norm(ks, X[:, 0].double()); phin_te = hermite_norm(ks, x0t)
    rn_tr = (phin_tr @ rtr) / P; rn_te = (phin_te @ rte) / len(Xt)
    # per hidden neuron: exact pull phi_k . alpha (training) vs the theory's t.e_k with the population residual
    alpha = rtr / v
    pull_exact = Phi.T @ alpha                                                   # (N2,)
    Wd, Bd = W.double(), B.double()
    C = readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), ks); h = Bd @ C.T
    nrm = 1 + 2 * (Wd * Wd).sum(1)
    Jp = (2 / math.pi) * torch.asin((2 * Wd @ Wd.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1)) - C.T @ C
    qh = ((Bd @ Jp) * Bd).sum(1); e = e_of(h, qh, ks, dev)
    t_pop = rn_te / lam                                                          # theory's t from the population residual
    pull_mf = e @ t_pop
    sl = float((pull_exact * pull_mf).sum() / (pull_mf * pull_mf).sum())
    print(f"P={P:6d}: training MSE {float((rtr**2).mean()):.5f}   test MSE {float((rte**2).mean()):.5f}   (ratio {float((rte**2).mean()/(rtr**2).mean()):.1f})")
    print(f"          He projections of the residual, training / population: n=1 {float(rn_tr[0]):+.4f}/{float(rn_te[0]):+.4f}  n=3 {float(rn_tr[1]):+.4f}/{float(rn_te[1]):+.4f}  n=5 {float(rn_tr[2]):+.4f}/{float(rn_te[2]):+.4f}")
    print(f"          per-neuron pull, exact (training) vs theory form with the population residual: slope {sl:.3f}, "
          f"correlation {float(torch.corrcoef(torch.stack([pull_exact, pull_mf]))[0,1]):.3f}; rms exact {float(pull_exact.std()):.0f}  rms theory-form {float(pull_mf.std()):.0f}"
          f"; theory's own t.e rms {float((e @ torch.tensor(z[f't_{P}'], dtype=torch.float64, device=dev)).std()):.0f}")
    del Phi, G, M
