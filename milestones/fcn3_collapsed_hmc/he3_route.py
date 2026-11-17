#!/usr/bin/env python3
"""Which hidden neurons produce the He3 part of the output, theory against sampler?

A hidden neuron can put He3 into the output in two ways: a large h|He3 (weights onto the outer read-in neurons), or a
large h|He1 that saturates erf (erf(h1 x0) has a He3 component of the sign the target wants here). The sampler shows a
large-h3 group at P = 48000 and few far-He1 neurons; the theory the reverse. This splits the output's He3 coefficient
f_bar_3 by region of the hidden preactivation:
    far He1    |h1| > 0.8
    large h3   |h3| > 0.25 and |h1| <= 0.8
    rest       everything else
Sampler: every saved network of the onset runs (d = 150, N1 = N2 = 1500, kappa = 0.015); neuron k contributes
a_bar_k e3_k, with a_bar the exact posterior mean of the readout on the training set and e3_k the neuron's population
He3 projection (hidden_equation_test.network_fbar). Theory: hidden-field particles of (E2) at the theory's own solution
(read-in kernels, M, t from the cache); region R contributes sigma_A^2 E[e3 (t.e) 1_R] (f_bar = sigma_A^2 K t).

    PYTHONPATH=.:../fcn3_matrix_theory python he3_route.py [--Ps 12000 24000 48000]
"""
import argparse
import json

import numpy as np
import torch

import plot_onset as po
import hidden_equation_test as het
from fcn3_matrix_solver import DT
from fcn3_sampler import make_data

HERE = po.HERE
H1_CUT, H3_CUT = 0.8, 0.25


def regions(h1, h3):
    far = np.abs(h1) > H1_CUT
    big3 = (np.abs(h3) > H3_CUT) & ~far
    return {"far He1": far, "large h3": big3, "rest": ~far & ~big3}


@torch.no_grad()
def sampler_side(P, rs, ks, dev):
    contrib = {k: [] for k in ("far He1", "large h3", "rest")}
    contrib1 = {k: [] for k in contrib}
    count = {k: [] for k in contrib}
    f3, f1 = [], []
    for r0 in rs:
        args = json.loads((r0["path"] / "config.json").read_text())["args"]
        X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
        X, Y = X.float(), Y.float()
        v, sa2 = args["kappa"] / args["chi"], args["sigA"] ** 2 / (args["N2"] * args["chi"])
        for f in sorted((r0["path"] / "nets").glob("net_*.pt")):
            nt = torch.load(f, map_location="cpu", weights_only=False)
            W = (nt["sw"] * nt["ZW"]).to(dev).float()
            B = (nt["sb"] * nt["ZB"]).to(dev).float()
            abar, e = network_parts(X, Y, W, B, v, sa2, ks)
            h = het_h(W, B, ks)
            c, c1 = abar * e[:, 1], abar * e[:, 0]
            f3.append(c.sum())
            f1.append(c1.sum())
            for k, m in regions(h[:, 0], h[:, 1]).items():
                contrib[k].append(c[m].sum())
                contrib1[k].append(c1[m].sum())
                count[k].append(m.sum())
        del X, Y
        torch.cuda.empty_cache()
    return {k: (np.mean(count[k]), np.mean(contrib[k]) / np.mean(f3), np.mean(contrib1[k]) / np.mean(f1))
            for k in contrib}, np.mean(f3)


def network_parts(X, Y, W, B, v, sa2, ks, chunk=8192):
    """(a_bar, e) of one network: exact posterior-mean readout and each hidden neuron's population projections."""
    import math
    from fcn3_matrix_solver import readin_features
    dev = X.device
    N2 = B.shape[0]
    G = torch.zeros(N2, N2, dtype=torch.float64, device=dev)
    r = torch.zeros(N2, dtype=torch.float64, device=dev)
    for i in range(0, X.shape[0], chunk):
        F2 = torch.erf(torch.erf(X[i:i + chunk] @ W.T) @ B.T).double()
        G += F2.T @ F2
        r += F2.T @ Y[i:i + chunk].double()
    abar = torch.linalg.solve(G / v + torch.eye(N2, dtype=torch.float64, device=dev) / sa2, r / v)
    Wd, Bd = W.double(), B.double()
    C = readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), ks)
    h = Bd @ C.T
    nrm = 1 + 2 * (Wd * Wd).sum(1)
    Jf = (2 / math.pi) * torch.asin((2 * Wd @ Wd.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
    qh = ((Bd @ (Jf - C.T @ C)) * Bd).sum(1)
    return abar.cpu().numpy(), het.e_of(h, qh, ks, dev).cpu().numpy()


def het_h(W, B, ks):
    from fcn3_matrix_solver import readin_features
    Wd, Bd = W.double(), B.double()
    return (Bd @ readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), ks).T).cpu().numpy()


@torch.no_grad()
def theory_side(P, z, dev):
    """From the solver's own hidden-field particles at its solution (2^14, resampled to equal weight, far basins
    included; cached by plot_onset.theory_all as hs_P, qs_P)."""
    p = po._params(P)
    ks = tuple(int(k) for k in z[f"ks_{P}"])
    hs = torch.tensor(z[f"hs_{P}"], dtype=DT, device=dev)
    qs = torch.tensor(z[f"qs_{P}"], dtype=DT, device=dev)
    t = torch.tensor(z[f"t_{P}"], dtype=DT, device=dev)
    e = het.e_of(hs, qs, ks, dev)
    c = (e[:, :2] * (e @ t)[:, None]).cpu().numpy() * p.sigA ** 2 / len(hs)      # (n, 2): He1, He3
    h = hs.cpu().numpy()
    f = c.sum(0)
    out = {k: (po.N * m.mean(), c[m, 1].sum() / f[1], c[m, 0].sum() / f[0]) for k, m in regions(h[:, 0], h[:, 1]).items()}
    return out, f[1] / float(p.y()[1]), float(z[f"ess_{P}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[12000, 24000, 48000])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        y3 = po._params(P).y()[1]
        th, L3_th, ess = theory_side(P, z, dev)
        sa, f3 = sampler_side(P, [r for r in groups[P] if (r["path"] / "nets").exists()], ks, dev)
        print(f"\nP = {P}: L3 theory {L3_th:.3f} (from its particles; ESS {ess:.0f}), sampler {f3 / y3:.3f}")
        print(f"   {'region':10s} {'neurons/network':>22s}   {'share of output He3':>24s}   {'share of output He1':>24s}")
        print(f"   {'':10s} {'theory':>10s} {'sampler':>10s}   {'theory':>11s} {'sampler':>11s}   {'theory':>11s} {'sampler':>11s}")
        for k in th:
            print(f"   {k:10s} {th[k][0]:10.2f} {sa[k][0]:10.2f}   {th[k][1]:11.3f} {sa[k][1]:11.3f}   "
                  f"{th[k][2]:11.3f} {sa[k][2]:11.3f}", flush=True)


if __name__ == "__main__":
    main()
