#!/usr/bin/env python3
"""Read-in cavity test: the exact energy of one read-in neuron moved through the barrier, everything else fixed.

Saved network of the onset run at P = 12000 (dataset 0, last saved network). For a few outer read-in neurons, w0_j is
moved from its value in the outer well to |w0| = 0.05 (sign and perpendicular weights kept) and the sampler's exact
collapsed energy U (readout integrated out, all priors included; fcn3_sampler.FCN3.U) is evaluated:
    fixed      the neuron's outgoing weights B[:, j] held at their values
    relaxed    B[:, j] re-optimised at every point (minimum of U over that column, warm-started along the path)
    relaxed+q  B[:, j] and the neuron's perpendicular norm q re-optimised (direction kept), minimising
               U - (d-3)/2 log q, i.e. including the volume factor of the (w0, q) density the theory uses
-log p(w0_j | rest) = U + const for the fixed version; the relaxed one is its profile over the column (the exact
marginal differs by the column's fluctuation entropy, small next to tens of nats). Compared with the theory's
-log p(w0) at P = 12000 (read-in marginal, plots/cache/theory_onset_d150_v6.npz), all relative to the outer well.

    PYTHONPATH=.:../fcn3_matrix_theory python readin_cavity.py [--n 3] [--steps 24]
"""
import argparse
import json
import math

import numpy as np
import torch

import plot_onset as po
from fcn3_sampler import FCN3, make_data

HERE = po.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=12000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    a_ = ap.parse_args()
    dev = torch.device(a_.device)
    P = a_.P
    run = [r for r in po.by_P(po.sampler_runs())[P] if r["seed"] == 0 and (r["path"] / "nets").exists()][0]["path"]
    a = json.loads((run / "config.json").read_text())["args"]
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], 1.0, a["seed"], dev)
    m = FCN3(X.double(), Y.double(), a["N1"], a["N2"], a["chi"], a["kappa"], a["sigA"], a["sigB"], a["sigw"])
    nt = torch.load(sorted((run / "nets").glob("net_*.pt"))[-1], map_location="cpu", weights_only=False)
    ZW, ZB = nt["ZW"].to(dev).double(), nt["ZB"].to(dev).double()
    sw = m.sw
    w0 = (ZW[:, 0] * sw).cpu().numpy()
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    wt, dn = z[f"w0_{P}"], z[f"dens_{P}"]
    kk = dn > 1e-300
    th = lambda x: np.interp(np.abs(x), wt[kk], -np.log(0.5 * dn[kk]))
    cand = np.where((np.abs(w0) > 0.68) & (np.abs(w0) < 0.80))[0]
    rng = np.random.default_rng(0)
    picks = rng.choice(cand, size=min(a_.n, len(cand)), replace=False)
    print(f"{run.name}: {len(cand)} neurons near the outer-well bottom; testing {list(picks)}")
    for j in picks:
        start = float(w0[j])
        path = np.sign(start) * np.linspace(abs(start), 0.05, a_.steps)
        col = ZB[:, j].clone()
        U0f = U0r = Fq0 = None
        wperp = ZW[j, 1:] * sw
        uhat = wperp / wperp.norm()
        lq = float(torch.log((wperp ** 2).sum()))
        qp = a["sigw"] ** 2 * (a["d"] - 1) / a["d"]
        rows = []
        for x in path:
            ZW1 = ZW.clone()
            ZW1[j, 0] = x / sw
            with torch.no_grad():
                uf, _, _ = m.U(ZW1, ZB, grad=False)
            c = col.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([c], lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                ZB1 = ZB.clone()
                ZB1[:, j] = c.detach()
                u, _, gB = m.U(ZW1, ZB1, grad=True)
                c.grad = gB[:, j].clone()
                return torch.tensor(u, dtype=torch.float64)

            opt.step(closure)
            col = c.detach()
            ZB1 = ZB.clone()
            ZB1[:, j] = col
            with torch.no_grad():
                ur, _, _ = m.U(ZW1, ZB1, grad=False)
            # column and log q together
            v = torch.cat([col, torch.tensor([lq], dtype=torch.float64, device=dev)]).requires_grad_(True)
            opt2 = torch.optim.LBFGS([v], lr=1.0, max_iter=30, history_size=10, line_search_fn="strong_wolfe")

            def closure2():
                opt2.zero_grad()
                ZW2 = ZW1.clone()
                ZW2[j, 1:] = uhat * torch.exp(0.5 * v.detach()[-1]) / sw
                ZB2 = ZB.clone()
                ZB2[:, j] = v.detach()[:-1]
                u, gW, gB = m.U(ZW2, ZB2, grad=True)
                g = torch.cat([gB[:, j], (gW[j, 1:] @ ZW2[j, 1:] * 0.5).reshape(1) - 0.5 * (a["d"] - 3)])
                v.grad = g.clone()
                return torch.tensor(u - 0.5 * (a["d"] - 3) * float(v.detach()[-1]), dtype=torch.float64)

            fq = float(opt2.step(closure2))
            fq = float(closure2())
            col2, lq = v.detach()[:-1], float(v.detach()[-1])
            U0f = uf if U0f is None else U0f
            U0r = ur if U0r is None else U0r
            Fq0 = fq if Fq0 is None else Fq0
            rows.append((abs(x), uf - U0f, ur - U0r, fq - Fq0, math.exp(lq) / qp, th(x) - th(start)))
        rows = np.array(rows)
        mid = (rows[:, 0] > 0.3) & (rows[:, 0] < 0.62)
        inn = rows[:, 0] < 0.2
        print(f"\nneuron {j}: w0 = {start:+.3f}  (q / q_prior {float((ZW[j, 1:] ** 2).sum()) * sw ** 2 / (a['sigw'] ** 2 * 149 / 150):.2f})")
        print(f"   {'|w0|':>6s} {'column fixed':>13s} {'column relaxed':>15s} {'column + q relaxed':>19s} {'q/q_prior':>9s} {'theory':>8s}   (nats, relative to the start)")
        for r in rows[::2]:
            print(f"   {r[0]:6.3f} {r[1]:13.2f} {r[2]:15.2f} {r[3]:19.2f} {r[4]:9.2f} {r[5]:8.2f}")
        print(f"   barrier (max between 0.3 and 0.62): column + q relaxed {rows[mid, 3].max():.1f}, theory {rows[mid, 5].max():.1f};  "
              f"inner well (min below 0.2): column + q relaxed {rows[inn, 3].min():.1f}, theory {rows[inn, 5].min():.1f}", flush=True)


if __name__ == "__main__":
    main()
