#!/usr/bin/env python3
"""The group of hidden neurons with large h|He3 in the sampler at P = 48000 on the onset line (d = 150, kappa = 0.015),
which the theory does not have (plot_hidden_wide.py: no theory well beyond |h3| ~ 0.2 out to |h3| = 3).

For the last saved network of every dataset: how many hidden neurons have |h3| > 0.3 and whether they are the same
neurons as in the first saved network; their h1; the share of their h3 that comes from the outer read-in neurons
(|w0| beyond the theory's barrier); and their fit gain against the theory's.

Fit gain: in the sampler's collapsed posterior hidden neuron k's exact fit gain is F_k = 1/2 s_a2 pull_k^2 / (1 - l_k),
pull_k = phi_k . alpha over the training set, alpha = training residual / v (cavity_decomposition.py; pull_k equals the
posterior-mean readout weight / s_a2). The pull splits exactly into the part through the neuron's dependence on x0,
e_k . (He projections of alpha), and the rest (its dependence on the other d - 1 input directions). The theory's
gain at the same neuron is 1/2 g~ (t . e_k)^2 at the theory's t; its prior cost is that of (h1, h3) under N(0, G).

    python outer_he3_group.py
"""
import json
import math

import numpy as np
import torch

import plot_onset as po
from hidden_equation_test import e_of
from fcn3_sampler import make_data
from fcn3_matrix_solver import readin_features, hermite_norm

P = 48000


def main():
    dev = torch.device("cuda:0")
    z = dict(np.load(po.HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    ks = tuple(int(k) for k in z[f"ks_{P}"])
    t = torch.tensor(z[f"t_{P}"], dtype=torch.float64, device=dev)
    gt = float(z[f"gt_{P}"])
    Gi = np.linalg.inv(z[f"G_{P}"][:2, :2])
    cut = po.barrier_cut(z, P)
    for r in po.by_P(po.sampler_runs())[P]:
        fs = sorted((r["path"] / "nets").glob("net_*.pt"))
        if not fs:
            continue
        a = json.loads((r["path"] / "config.json").read_text())["args"]
        X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
        X, Y = X.float(), Y.float()
        v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
        groups = []
        for f in (fs[0], fs[-1]):
            nt = torch.load(f, map_location="cpu", weights_only=False)
            W = (nt["sw"] * nt["ZW"]).to(dev).double()
            B = (nt["sb"] * nt["ZB"]).to(dev).double()
            C = readin_features(W[:, 0], (W[:, 1:] ** 2).sum(1), ks)                    # (S, N1)
            h = B @ C.T
            groups.append(set(torch.where(h[:, 1].abs() > 0.3)[0].tolist()))
        g = torch.tensor(sorted(groups[1]), device=dev)
        o = torch.zeros(B.shape[0], dtype=torch.bool, device=dev)
        o[g] = True
        outer_r = W[:, 0].abs() > cut
        share = ((B[o][:, outer_r] @ C[1, outer_r]) / h[o, 1]).median()
        zb2 = (nt["ZB"].double() ** 2).mean(1).to(dev)
        Phi = torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.float().T) @ B.float().T)
                         for i in range(0, X.shape[0], 8192)]).double()
        M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
        Minv = torch.linalg.inv(M)
        alpha = (Y.double() - Phi @ (Minv @ (Phi.T @ Y.double() / v))) / v
        lev = 1 - torch.diagonal(Minv) / sa2
        nrm = 1 + 2 * (W * W).sum(1)
        Jp = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1)) - C.T @ C
        qh = ((B @ Jp) * B).sum(1)
        e = e_of(h, qh, ks, dev)
        pull = Phi.T @ alpha
        p0 = e @ (hermite_norm(ks, X[:, 0].double()) @ alpha)
        gain = lambda p: (0.5 * sa2 * p ** 2 / (1 - lev)).cpu().numpy()
        F, F0, Fr = gain(pull), gain(p0), gain(pull - p0)
        Ft = (0.5 * gt * (e @ t) ** 2).cpu().numpy()
        hn = h[:, :2].cpu().numpy()
        cost = 0.5 * np.einsum("ni,ij,nj->n", hn, Gi, hn)
        on, inn = o.cpu().numpy(), ~o.cpu().numpy()
        md = lambda x, m: float(np.median(x[m]))
        print(f"{r['path'].name}: {len(groups[1])} neurons with |h3| > 0.3 ({len(groups[0] & groups[1])} the same as "
              f"in the first saved network); h3 {md(np.abs(hn[:, 1]), on):.2f}, |h1| {md(np.abs(hn[:, 0]), on):.2f}; "
              f"{float(share):.2f} of h3 from the {int(outer_r.sum())} outer read-in neurons; "
              f"|ZB|^2/N1 {md(zb2.cpu().numpy(), on):.2f} (others {md(zb2.cpu().numpy(), inn):.2f})")
        print(f"   exact fit gain {md(F, on):.0f} (through x0 {md(F0, on):.0f}, through the other directions "
              f"{md(Fr, on):.0f}); theory's gain at the same neurons {md(Ft, on):.0f}, theory's prior cost "
              f"{md(cost, on):.0f}.   Ordinary neurons: exact {md(F, inn):.1f} (through x0 {md(F0, inn):.1f}, other "
              f"directions {md(Fr, inn):.1f}); theory {md(Ft, inn):.1f}, prior cost {md(cost, inn):.1f}", flush=True)
        del X, Y, Phi, M, Minv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
