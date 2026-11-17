#!/usr/bin/env python3
"""Is the theory's average over the perpendicular directions right, neuron by neuron, in the sampler's networks?

The theory computes a hidden neuron's output projections as
    e_n = E_x0[ phi_n(x0) erf( a(x0) / sqrt(1 + 2 q_h) ) ],   a(x0) = sum_{m<=11} h_m phi_m(x0),  q_h = ||h_perp||^2,
i.e. at fixed x0 the part of h outside the x0 tower is Gaussian noise over x_perp with ONE variance q_h, the same at
every x0. Three versions of e_n for every hidden neuron of saved networks:
    exact     E_x[phi_n(x0) erf(h(x))]: Gauss-Hermite in x0 (64 nodes) x Monte Carlo over x_perp (fresh draws)
    profile   Gaussian at fixed x0 with the true mean m(x0) and variance s^2(x0) of h over x_perp:
              E_x0[phi_n(x0) erf(m(x0) / sqrt(1 + 2 s^2(x0)))]
    theory    the formula above (hidden_equation_test.e_of)
exact vs profile: non-Gaussianity of h at fixed x0. profile vs theory: the x0 dependence of the perpendicular variance
(and the tower truncation of the mean). The mean output f_bar_n = sum_k a_bar_k e_n,k is formed with each version
(a_bar: exact posterior-mean readout on the training set), in total and by region (he3_route.regions).

    PYTHONPATH=.:../fcn3_matrix_theory python perp_blur_test.py [--Ps 12000 48000] [--nets 3] [--nperp 4096]
"""
import argparse
import json

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import hermite_norm
from fcn3_sampler import make_data

HERE = po.HERE


@torch.no_grad()
def projections(W, B, ks, nperp, gen):
    """exact and profile e (N2, S), plus mean / variance profiles of h over x_perp at the x0 nodes."""
    dev = W.device
    x, wx = hermegauss(64)
    wx = torch.tensor(wx / wx.sum(), device=dev, dtype=torch.float64)
    x = torch.tensor(x, device=dev, dtype=torch.float64)
    Phi = hermite_norm(ks, x)                                   # (S, 64)
    d = W.shape[1]
    Xp = torch.randn(nperp, d - 1, device=dev, dtype=torch.float64, generator=gen)
    Up = Xp @ W[:, 1:].T                                        # (nperp, N1): w_perp . x_perp
    N2 = B.shape[0]
    Ex = torch.zeros(N2, 64, device=dev, dtype=torch.float64)  # E_perp erf(h) at each x0 node
    Pr = torch.zeros(N2, 64, device=dev, dtype=torch.float64)  # erf(m / sqrt(1 + 2 s^2))
    S2 = torch.zeros(N2, 64, device=dev, dtype=torch.float64)
    for i in range(64):
        h = torch.erf(x[i] * W[:, 0][None, :] + Up) @ B.T      # (nperp, N2)
        Ex[:, i] = torch.erf(h).mean(0)
        m, s2 = h.mean(0), h.var(0)
        Pr[:, i] = torch.erf(m / torch.sqrt(1 + 2 * s2))
        S2[:, i] = s2
    wP = (Phi * wx).T                                           # (64, S)
    return Ex @ wP, Pr @ wP, S2, wx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[12000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--nperp", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    gen = torch.Generator(device=dev).manual_seed(5)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        y = np.array(po._params(P).y())
        F = {v: [] for v in ("exact", "profile", "theory")}
        R = {}
        rms = {v: [] for v in ("profile", "theory")}
        prof = []
        for r0 in [r for r in groups[P] if (r["path"] / "nets").exists()]:
            args = json.loads((r0["path"] / "config.json").read_text())["args"]
            X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
            X, Y = X.float(), Y.float()
            v, sa2 = args["kappa"] / args["chi"], args["sigA"] ** 2 / (args["N2"] * args["chi"])
            for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(f, map_location="cpu", weights_only=False)
                W = (nt["sw"] * nt["ZW"]).to(dev)
                B = (nt["sb"] * nt["ZB"]).to(dev)
                abar, e_th = hr.network_parts(X, Y, W.float(), B.float(), v, sa2, ks)
                hS = hr.het_h(W, B, ks)
                e_ex, e_pr, S2, wx = projections(W.double(), B.double(), ks, a.nperp, gen)
                e_ex, e_pr = e_ex.cpu().numpy(), e_pr.cpu().numpy()
                regs = hr.regions(hS[:, 0], hS[:, 1])
                for name, e in (("exact", e_ex), ("profile", e_pr), ("theory", e_th)):
                    c = abar[:, None] * e
                    F[name].append(c.sum(0))
                    for k, m in regs.items():
                        R.setdefault((name, k), []).append(c[m].sum(0))
                for name, e in (("profile", e_pr), ("theory", e_th)):
                    rms[name].append(np.sqrt(((e - e_ex) ** 2).mean(0)))
                # x0 profile of the perpendicular variance: s^2(x0) / its x0-average, ordinary vs large-h3 neurons
                s2 = S2.cpu().numpy()
                avg = (s2 * wx.cpu().numpy()).sum(1, keepdims=True)
                xs = hermegauss(64)[0]
                at = lambda x0: np.interp(x0, xs, np.arange(64))
                idx = [int(round(at(x0))) for x0 in (0.0, 1.0, 2.0, 3.0)]
                prof.append({k: (s2[m][:, idx] / avg[m]).mean(0) for k, m in regs.items() if m.any()})
            del X, Y
            torch.cuda.empty_cache()
        print(f"\nP = {P}  (target He1 1, He3 {y[1]:.3f}; modes {ks})")
        print("   mean output f_bar_n with each version of the neurons' projections:")
        for name in F:
            print(f"     {name:8s} {np.round(np.mean(F[name], 0), 4)}   L3 {np.mean(F[name], 0)[1] / y[1]:.3f}")
        print("   by region (He1, He3, He5, He7):")
        for (name, k), vals in sorted(R.items(), key=lambda t: (t[0][1], t[0][0])):
            print(f"     {k:9s} {name:8s} {np.round(np.mean(vals, 0)[:4], 4)}")
        print("   rms per-neuron error against exact, per mode:")
        for name in rms:
            print(f"     {name:8s} {np.round(np.mean(rms[name], 0), 5)}")
        print("   perpendicular variance s^2(x0) / its x0-average, at x0 = 0, 1, 2, 3:")
        for k in prof[0]:
            vals = [p[k] for p in prof if k in p]
            print(f"     {k:9s} {np.round(np.mean(vals, 0), 3)}", flush=True)


if __name__ == "__main__":
    main()
