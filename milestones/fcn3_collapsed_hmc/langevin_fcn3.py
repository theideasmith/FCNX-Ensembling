#!/usr/bin/env python3
"""Langevin training of the three-layer erf network on exactly the posterior the collapsed HMC sampler draws from,
to check that the HMC runs equilibrate.

Same update as milestones/activation_generic_erf_mf_scaling_convergence/d_sweep_seeds.py, with the weight decay set by
sigma_w^2, sigma_B^2, sigma_A^2 (that script has all three equal to 1):

    f(x) = A . erf(W1 erf(W0 x)),   L = sum_mu (f(x_mu) - y_mu)^2,   T = 2 kappa / chi,
    theta <- theta - lr_theta (dL/dtheta + wd_theta theta) + sqrt(2 lr_theta T) xi,
    wd_W0 = T d / sigma_w^2,   wd_W1 = T N1 / sigma_B^2,   wd_A = T N2 chi / sigma_A^2.

Its stationary distribution is exp(-chi/(2 kappa) L) N(W0; 0, sigma_w^2/d) N(W1; 0, sigma_B^2/N1)
N(A; 0, sigma_A^2/(N2 chi)), which is the sampler's (fcn3_sampler.FCN3: noise variance kappa/chi, W = sw ZW,
B = sb ZB, readout variance sigA^2/(N2 chi)) before the readout is integrated out. The data are the sampler's
(make_data, same seed).

Step sizes: the readout is ~100x stiffer than the other layers (loss curvature at an equilibrium HMC network, d = 150,
P = 3000: 2.4e3 in W0, 3.2e3 in W1, 2.2e5 in A), so each layer gets its own step. A constant step per layer is a
constant diagonal mobility, which leaves the stationary distribution unchanged. The ensemble runs independent networks
on the same data; `--half` of them use half the steps, to check the discretization error.

Records: trace.csv every --trace-every steps (training MSE; L1, L3 = He1, He3 coefficients of f on a fixed set of
fresh inputs over the target's; number of read-in neurons with |w0| beyond --cut; mean w0^2 and q = |w_perp|^2), and
after burn-in every --save-every steps one file per network in nets/, in the sampler's format (ZW, ZB, sw, sb) plus A.

    python langevin_fcn3.py --P 3000 --steps 3000000 --burn 1000000 --device cuda:0
    python langevin_fcn3.py --P 3000 --burn 0 --init-from <HMC net files>      (start from HMC end states)
"""
import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import torch

from fcn3_sampler import make_data

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--P", type=int, default=3000)
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--chi", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.015)
    ap.add_argument("--eps", type=float, default=-0.3 * math.sqrt(6))
    ap.add_argument("--sigA", type=float, default=math.sqrt(8 / 1500))
    ap.add_argument("--sigB", type=float, default=1.25540121081682)
    ap.add_argument("--sigw", type=float, default=math.sqrt(0.5))
    ap.add_argument("--seed", type=int, default=0, help="data seed (same draws as the HMC runs)")
    ap.add_argument("--noise-seed", type=int, default=1)
    ap.add_argument("--ens", type=int, default=4)
    ap.add_argument("--half", type=int, default=2, help="how many of the networks use half the steps")
    ap.add_argument("--lr-w", type=float, default=4e-5, help="step for W0 and W1")
    ap.add_argument("--lr-a", type=float, default=4e-7, help="step for A")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--burn", type=int, default=1_000_000)
    ap.add_argument("--trace-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=100_000)
    ap.add_argument("--cut", type=float, default=0.6, help="|w0| beyond which a read-in neuron counts as outer")
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--init-from", nargs="+", default=None,
                    help="start network e from the e-th of these saved HMC states (nets/net_*.pt or checkpoint.pt, same "
                         "data seed); its readout A is drawn from its exact Gaussian conditional N(abar, M^-1)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    d, P, N, chi, kappa = a.d, a.P, a.N, a.chi, a.kappa
    T = 2 * kappa / chi
    sw, sb, sA = a.sigw / math.sqrt(d), a.sigB / math.sqrt(N), a.sigA / math.sqrt(N * chi)
    wd = dict(W0=T * d / a.sigw ** 2, W1=T * N / a.sigB ** 2, A=T * N * chi / a.sigA ** 2)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = HERE / "runs" / "langevin" / f"{stamp}_d{d}_P{P}_N{N}_k{kappa}_seed{a.seed}{'_' + a.tag if a.tag else ''}"
    (run / "nets").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(args=vars(a), T=T, wd=wd, sw=sw, sb=sb, sA=sA), indent=1))

    X, Y, _, _ = make_data(d, P, a.eps, 1.0, a.seed, dev)
    X, Y = X.float(), Y.float()
    g = torch.Generator(device=dev).manual_seed(a.noise_seed)
    Xt = torch.randn(a.n_test, d, device=dev, generator=g)
    x0 = Xt[:, 0]
    phi1, phi3 = x0, (x0 ** 3 - 3 * x0) / math.sqrt(6)

    E = a.ens
    W0 = torch.randn(E, N, d, device=dev, generator=g) * sw
    W1 = torch.randn(E, N, N, device=dev, generator=g) * sb
    A = torch.randn(E, N, device=dev, generator=g) * sA
    if a.init_from:
        v, s_a2 = kappa / chi, a.sigA ** 2 / (N * chi)
        for e in range(E):
            ck = torch.load(a.init_from[e % len(a.init_from)], map_location="cpu", weights_only=False)
            W0[e] = sw * ck["ZW"].to(dev).float()
            W1[e] = sb * ck["ZB"].to(dev).float()
            Phi = torch.erf(torch.erf(X @ W0[e].T) @ W1[e].T).double()
            L = torch.linalg.cholesky(Phi.T @ Phi / v + torch.eye(N, dtype=torch.float64, device=dev) / s_a2)
            abar = torch.cholesky_solve((Phi.T @ Y.double() / v)[:, None], L)[:, 0]
            z = torch.randn(N, dtype=torch.float64, device=dev, generator=torch.Generator(device=dev).manual_seed(100 + e))
            A[e] = (abar + torch.linalg.solve_triangular(L.T, z[:, None], upper=True)[:, 0]).float()
            del Phi, L
    params = dict(W0=W0, W1=W1, A=A)
    scale = torch.ones(E, device=dev)
    scale[E - a.half:] = 0.5
    lr = dict(W0=a.lr_w * scale, W1=a.lr_w * scale, A=a.lr_a * scale)
    shape = dict(W0=(E, 1, 1), W1=(E, 1, 1), A=(E, 1))

    def forward(Xb, W0, W1, A):
        return torch.einsum("en,epn->ep", A, torch.erf(torch.einsum("emn,epn->epm", W1,
                                                                     torch.erf(torch.einsum("end,pd->epn", W0, Xb)))))

    fh = open(run / "trace.csv", "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(["step", "net", "half_step", "train_mse", "L1", "L3", "outer", "w0sq_d", "q"])
    t0 = time.time()
    for step in range(a.steps + 1):
        for p in params.values():
            p.requires_grad_(True)
        f = forward(X, params["W0"], params["W1"], params["A"])
        loss = ((f - Y) ** 2).sum()
        grads = torch.autograd.grad(loss, list(params.values()))
        if step % a.trace_every == 0:
            with torch.no_grad():
                mse = ((f - Y) ** 2).mean(1)
                ft = torch.cat([forward(Xt[i:i + 5000], params["W0"], params["W1"], params["A"])
                                for i in range(0, a.n_test, 5000)], 1)
                L1 = (ft * phi1).mean(1) / 1.0
                L3 = (ft * phi3).mean(1) / a.eps
                w0 = params["W0"][:, :, 0]
                q = (params["W0"][:, :, 1:] ** 2).sum(2)
                outer = (w0.abs() > a.cut).sum(1)
                for e in range(E):
                    wr.writerow([step, e, int(e >= E - a.half), f"{mse[e]:.6g}", f"{L1[e]:.6g}", f"{L3[e]:.6g}",
                                 int(outer[e]), f"{float((w0[e] ** 2).mean()) * d:.6g}", f"{float(q[e].mean()):.6g}"])
                fh.flush()
                if not torch.isfinite(mse).all():
                    raise SystemExit(f"step {step}: non-finite loss, step sizes too large")
                if step % (a.trace_every * 50) == 0:
                    el = time.time() - t0
                    print(f"step {step}: train MSE {mse.mean():.4f}  L3 {L3.tolist()}  outer {outer.tolist()}  "
                          f"({el / max(step, 1) * 1e3:.2f} ms/step)", flush=True)
        if step >= a.burn and (step - a.burn) % a.save_every == 0:
            for e in range(E):
                torch.save(dict(ZW=(params["W0"][e] / sw).detach().cpu(), ZB=(params["W1"][e] / sb).detach().cpu(),
                                A=params["A"][e].detach().cpu(), sw=sw, sb=sb, iter=step, net=e,
                                half_step=bool(e >= E - a.half)),
                           run / "nets" / f"net_{step:09d}_m{e}.pt")
        if step == a.steps:
            break
        with torch.no_grad():
            for (name, p), gr in zip(params.items(), grads):
                lr_ = lr[name].view(shape[name])
                noise = torch.randn(p.shape, device=dev, generator=g) * torch.sqrt(2 * lr_ * T)
                p.add_(-lr_ * (gr + wd[name] * p) + noise)
    fh.close()
    print("done", run)


if __name__ == "__main__":
    main()
