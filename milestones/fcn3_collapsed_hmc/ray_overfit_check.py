#!/usr/bin/env python3
"""Training vs test error along the kappa0 = 0.015 ray (final state of the dataset-0 prior-start runs, d = 150, 300, 600).
   python ray_overfit_check.py   (FINDINGS_onset.md)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fcn3_matrix_theory"))
import glob, json, math, numpy as np, torch
from fcn3_sampler import make_data
from fcn3_matrix_solver import hermite_norm
dev = torch.device("cuda:0")
for f in sorted(glob.glob("runs/*_ray_d*_N1500x1500_prior_r*/config.json")):
    a = json.load(open(f))["args"]
    if a.get("seed") != 0 or a.get("group", "").startswith(("fixedn", "ray_hist")) or a["d"] == 75:
        continue
    r = f.rsplit("/", 1)[0]
    ck = torch.load(r + "/checkpoint.pt", map_location="cpu", weights_only=False)
    d, P = a["d"], a["P"]
    W = (ck["ZW"].double() * math.sqrt(0.5 / d)).to(dev).float(); B = (ck["ZB"].double() * a["sigB"] / math.sqrt(a["N1"])).to(dev).float()
    X, Y, _, _ = make_data(d, P, a["eps"], 1.0, a["seed"], dev); X, Y = X.float(), Y.float()
    Xt, Yt, _, _ = make_data(d, 100000, a["eps"], 1.0, 12345, dev); Xt, Yt = Xt.float(), Yt.float()
    v, sa2 = a["kappa"] / a["chi"], a["sigA"] ** 2 / (a["N2"] * a["chi"])
    Phi = torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.T) @ B.T) for i in range(0, P, 8192)]).double()
    abar = torch.linalg.solve(Phi.T @ Phi / v + torch.eye(a["N2"], dtype=torch.float64, device=dev) / sa2, Phi.T @ Y.double() / v)
    rtr = Y.double() - Phi @ abar
    rte = Yt.double() - torch.cat([torch.erf(torch.erf(Xt[i:i + 8192] @ W.T) @ B.T).double() @ abar for i in range(0, len(Xt), 8192)])
    h3 = lambda X_, r_: float((hermite_norm((3,), X_[:, 0].double())[0] @ r_) / len(r_))
    print(f"ray d={d} P={P} kappa={a['kappa']:.4f} ({r.split('/')[-1][:15]}): train MSE {float((rtr**2).mean()):.5f}, test MSE {float((rte**2).mean()):.5f}, "
          f"ratio {float((rte**2).mean()/(rtr**2).mean()):.2f};  He3 residual train/test {h3(X, rtr):+.4f}/{h3(Xt, rte):+.4f} (ratio {h3(X, rtr)/h3(Xt, rte):.2f})")
    del Phi, X, Xt
