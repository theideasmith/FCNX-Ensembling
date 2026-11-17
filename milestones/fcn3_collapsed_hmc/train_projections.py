#!/usr/bin/env python3
"""The networks' projections evaluated on their own training set, against the population.

For the same 9 saved networks per P as measure_perp_pull.py (last 3 of each onset run):
    output        f_bar_n on the training set ((1/P) sum_mu phi_n(x0_mu) f_bar(x_mu)) and in the population
    residual      r = y - f_bar on the training set: its fit by He1..He11 of x0 (least squares on the training set)
                  and the rest (perpendicular part), rms of each; population: (y - f_bar)_n and rms of f_bar_perp
    per neuron    S_k = <erf o h_k | t>_train = sum_mu erf(h_k(x_mu)) r_mu / kappa (the overlap the exact readout
                  uses: a_bar = (sigma_A^2/N2) S exactly), split into the part from r's x0 fit (S_x0) and from
                  its perpendicular rest (S_perp); against the population s_k = t . e_k and c_k
The readout factor: slope of a_bar against (sigma_A^2/N2) S_k (must be 1) and against the population s + c.

    PYTHONPATH=.:../fcn3_matrix_theory python train_projections.py [--Ps 12000 24000 48000]
"""
import argparse
import json

import numpy as np
import torch

import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import hermite_norm
from fcn3_sampler import make_data

HERE = po.HERE


@torch.no_grad()
def one_net(X, Y, W, B, kappa, chi, sigA, ks, chunk=8192):
    dev = X.device
    N2 = B.shape[0]
    v, sa2 = kappa / chi, sigA ** 2 / (N2 * chi)
    Phi = torch.cat([torch.erf(torch.erf(X[i:i + chunk] @ W.T) @ B.T) for i in range(0, X.shape[0], chunk)])
    M = Phi.T @ Phi / v + torch.eye(N2, dtype=Phi.dtype, device=dev) / sa2
    abar = torch.linalg.solve(M, Phi.T @ Y / v)
    fb = Phi @ abar
    r = Y - fb
    Ph = hermite_norm(ks, X[:, 0]).T                                   # (P, S)
    P = X.shape[0]
    fn_tr = (Ph.T @ fb) / P
    yn_tr = (Ph.T @ Y) / P
    co = torch.linalg.lstsq(Ph, r[:, None]).solution[:, 0]            # r's fit by the x0 tower on the training set
    r_x0 = Ph @ co
    r_perp = r - r_x0
    S = Phi.T @ r / kappa
    S_x0 = Phi.T @ r_x0 / kappa
    S_perp = Phi.T @ r_perp / kappa
    return dict(abar=abar.cpu().numpy(), S=S.cpu().numpy(), S_x0=S_x0.cpu().numpy(), S_perp=S_perp.cpu().numpy(),
                fn_tr=fn_tr.cpu().numpy(), yn_tr=yn_tr.cpu().numpy(), rco=co.cpu().numpy(),
                r_x0_rms=float(r_x0.pow(2).mean().sqrt()), r_perp_rms=float(r_perp.pow(2).mean().sqrt()),
                r_rms=float(r.pow(2).mean().sqrt()), sa=sigA ** 2 / N2)


def slope(y, x):
    b = (x @ y) / (x @ x)
    return b, 1 - np.var(y - b * x) / np.var(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[12000, 24000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    for P in a.Ps:
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        p = po._params(P)
        f = HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"
        pp = dict(np.load(f)) if f.exists() else None
        out, n = [], 0
        for r0 in [r for r in groups[P] if (r["path"] / "nets").exists()]:
            args = json.loads((r0["path"] / "config.json").read_text())["args"]
            X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
            X, Y = X.double(), Y.double()
            for fn in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(fn, map_location="cpu", weights_only=False)
                W = (nt["sw"] * nt["ZW"]).to(dev).double()
                B = (nt["sb"] * nt["ZB"]).to(dev).double()
                o = one_net(X, Y, W, B, args["kappa"], args["chi"], args["sigA"], ks)
                o["h"] = hr.het_h(W, B, ks)
                o["net"] = n
                out.append(o)
                n += 1
            del X, Y
            torch.cuda.empty_cache()
        cat = lambda k: np.concatenate([o[k] for o in out])
        ab, S, Sx, Sp, h = cat("abar"), cat("S"), cat("S_x0"), cat("S_perp"), cat("h")
        sa = out[0]["sa"]
        y = np.array(p.y())
        print(f"\nP = {P}  ({n} networks)")
        fn_tr = np.mean([o["fn_tr"] for o in out], 0)
        print(f"   output f_bar_n, training set : {np.round(fn_tr, 4)}   L3 {fn_tr[1] / y[1]:.3f}")
        if pp is not None:
            print(f"   output f_bar_n, population   : {np.round(pp['fn'].mean(0), 4)}   L3 {pp['fn'].mean(0)[1] / y[1]:.3f}")
        print(f"   target y_n on the training set: {np.round(np.mean([o['yn_tr'] for o in out], 0), 4)}")
        print(f"   residual rms on the training set: total {np.mean([o['r_rms'] for o in out]):.4f} = x0 part "
              f"{np.mean([o['r_x0_rms'] for o in out]):.4f} (+) perpendicular part {np.mean([o['r_perp_rms'] for o in out]):.4f}")
        if pp is not None:
            res_pop = pp["t"].mean(0) * p.lam
            print(f"   residual in the population:      x0 part {np.sqrt((res_pop ** 2).sum()):.4f} "
                  f"(He1..He11 {np.round(res_pop, 4)}), perpendicular part {pp['fperp_rms'].mean():.4f}")
            print(f"   residual x0 coefficients, training set: {np.round(np.mean([o['rco'] for o in out], 0), 4)}")
        b, r2 = slope(ab, sa * S)
        print(f"   readout a_bar vs (sigma_A^2/N2) S_train: slope {b:.4f}, R^2 {r2:.5f}   (identity: 1)")
        if pp is not None and len(pp["s"]) == len(S):
            sp, cp = pp["s"], pp["c"]
            b2, r22 = slope(ab, sa * (sp + cp))
            print(f"   readout a_bar vs (sigma_A^2/N2)(s + c)_population: slope {b2:.3f}, R^2 {r22:.3f}")
            print(f"   per neuron, training vs population (rms over neurons):")
            print(f"      x0 part:   S_x0 {np.sqrt((Sx ** 2).mean()):9.0f}   s {np.sqrt((sp ** 2).mean()):9.0f}   "
                  f"slope S_x0 vs s {slope(Sx, sp)[0]:.3f} (R^2 {slope(Sx, sp)[1]:.3f})")
            print(f"      perp part: S_perp {np.sqrt((Sp ** 2).mean()):7.0f}   c {np.sqrt((cp ** 2).mean()):9.0f}   "
                  f"slope S_perp vs c {slope(Sp, cp)[0]:.3f} (R^2 {slope(Sp, cp)[1]:.3f})")
            reg = hr.regions(h[:, 0], h[:, 1])
            gt = p.gt
            for k, m in reg.items():
                if not m.any():
                    continue
                d = lambda x: x[m].mean() - x[reg["rest"]].mean()
                print(f"      {k:9s} (minus ordinary), -1/2 g~ (.)^2:  training S {d(-0.5 * gt * S ** 2):9.1f}   "
                      f"[x0 part only {d(-0.5 * gt * Sx ** 2):9.1f}]   population (s + c) {d(-0.5 * gt * (sp + cp) ** 2):9.1f}"
                      f"   [s only {d(-0.5 * gt * sp ** 2):9.1f}]")


if __name__ == "__main__":
    main()
