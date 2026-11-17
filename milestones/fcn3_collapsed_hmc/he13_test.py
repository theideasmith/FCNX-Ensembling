#!/usr/bin/env python3
"""Does the read-in kernel's He1-He3 correlation decide whether the theory forms the large-h3 group?

The hidden-neuron action term 1/2 h^T G^-1 h charges large-h3 states (h1 and h3 of the same sign) ~95 nats more with
the theory's G (He1-He3 correlation −0.79) than with the networks' (−0.73) (action_terms.py). Here the hidden layer and
readout are solved with the read-in kernel held fixed (J, qbar, M; empirical_kernel_test.solve_hidden), with the full
correction measured in the networks (perp_surrogate.py --r): readout factor r = 0.66 (sigma_A^2 -> r sigma_A^2,
chi -> chi / r), c = gamma . e on all six modes (target y + lambda gamma). Read-in kernels:
    theory           the theory's own solution (check: with r = 1, c = 0 it must reproduce the theory)
    networks         each network's J_hat averaged over the saved networks (E6-E8 over their read-in neurons)
    theory, He13     the theory's, with the He1-He3 correlation set to the networks' value (if still positive definite)
    networks, He13   the networks', with the He1-He3 correlation set to the theory's value

    PYTHONPATH=.:../fcn3_matrix_theory python he13_test.py [--P 48000]
"""
import argparse
import math
from dataclasses import replace

import numpy as np
import torch

import empirical_kernel_test as ekt
import fcn3_fixed as F
import he3_route as hr
import plot_onset as po
from fcn3_matrix_solver import DT

HERE = po.HERE


def solve_hidden(p, J, T, M, t0, shift, dev, rounds=3):
    Fo = F.BlazingFreeEnergy(p, device=dev, verbose=False)
    Fo.y = Fo.y + torch.tensor(shift, dtype=DT, device=Fo.dev)
    Fo.last = dict(J=J, T=T)
    Fo.t = torch.tensor(t0, dtype=DT, device=Fo.dev)
    Fo.nprop = 0
    Fo.far_comps = []
    t_prev, moved = None, math.inf
    for r in range(rounds):
        Fo.set_samples(15, M, fresh=(r == 0))
        if t_prev is not None:
            moved = float((Fo.t - t_prev).abs().max() / Fo.t.abs().max())
            if moved < 1e-5:
                break
        t_prev = Fo.t.clone()
    L = F.scaled_cholesky(p.sigB ** 2 * J)
    t, info = Fo.solve_t(L, p.sigB ** 2 * T, M, Fo.t)
    fb = (Fo.y - p.lam * t).cpu().numpy()
    w, h = info["w"].cpu().numpy(), info["h"].cpu().numpy()
    reg = hr.regions(h[:, 0], h[:, 1])
    return dict(L3=fb[1] / p.eps, fb=fb, far=po.N * w[reg["far He1"]].sum(), big3=po.N * w[reg["large h3"]].sum(),
                sd1=math.sqrt(w @ h[:, 0] ** 2), sd3=math.sqrt(w @ h[:, 1] ** 2), ess=info["ess"], moved=moved)


def set_corr13(J, rho):
    D = torch.sqrt(torch.diagonal(J))
    R = J / torch.outer(D, D)
    R[0, 1] = R[1, 0] = rho
    Jn = R * torch.outer(D, D)
    ok = bool(torch.linalg.eigvalsh(R).min() > 0)
    return Jn, ok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=48000)
    ap.add_argument("--r", type=float, default=0.66)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    P, dev = a.P, torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    pp = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
    p0 = po._params(P)
    sb2 = p0.sigB ** 2
    ks = tuple(int(k) for k in z[f"ks_{P}"])
    J_t = torch.tensor(z[f"G_{P}"] / sb2, dtype=DT, device=dev)
    T_t, M_t = float(z[f"qbar_{P}"]) / sb2, float(z[f"M_{P}"])
    rs = [r for r in po.by_P(po.sampler_runs())[P] if (r["path"] / "nets").exists()]
    J_s, T_s, Ek2, _, _ = ekt.readin_from_nets(rs, ks, dev)
    M_s = (sb2 * T_s) ** 2 / (sb2 ** 2 * (Ek2 - float((J_s * J_s).sum())))
    corr = lambda J: float(J[0, 1] / torch.sqrt(J[0, 0] * J[1, 1]))
    rho_t, rho_s = corr(J_t), corr(J_s)
    gam = np.linalg.lstsq(pp["e"], pp["c"], rcond=None)[0]
    shift = p0.lam * gam
    kernels = [("theory", J_t, T_t, M_t), ("networks", J_s, T_s, M_s)]
    Jx, ok = set_corr13(J_t.clone(), rho_s)
    kernels.append(("theory, He13 of networks", Jx, T_t, M_t) if ok else ("theory, He13 of networks: not pos. def.", None, 0, 0))
    Jx, ok = set_corr13(J_s.clone(), rho_t)
    kernels.append(("networks, He13 of theory", Jx, T_s, M_s) if ok else ("networks, He13 of theory: not pos. def.", None, 0, 0))
    print(f"P = {P}: He1-He3 correlation of the read-in kernel: theory {rho_t:+.3f}, networks {rho_s:+.3f}")
    print(f"   {'read-in kernel':28s} {'correction':16s} {'L3':>6s} {'far He1':>8s} {'large h3':>9s} {'sd h1':>6s} "
          f"{'sd h3':>6s}  (t settled to)")
    jobs = [(k, (f"r {a.r}, c", a.r, shift)) for k in kernels[1:]] + [(kernels[1], ("none", 1.0, np.zeros(6)))]
    for (name, J, T, M), (lab, r, sh) in jobs:
        if J is None:
            print(f"   {name}")
            continue
        if True:
            p = replace(p0, sigA=p0.sigA * math.sqrt(r), chi=p0.chi / r)
            try:
                o = solve_hidden(p, J, T, M, z[f"t_{P}"], sh, dev)
                print(f"   {name:28s} {lab:16s} {o['L3']:6.3f} {o['far']:8.2f} {o['big3']:9.2f} {o['sd1']:6.3f} "
                      f"{o['sd3']:6.3f}  ({o['moved']:.0e}, ESS {o['ess']:.0f})", flush=True)
            except Exception as ex:
                print(f"   {name:28s} {lab:16s} failed: {ex!r}"[:200], flush=True)
    H, n = po.sampler_hidden(rs)
    reg = hr.regions(H[:, 0], H[:, 1])
    print(f"   {'networks (sampled)':28s} {'':16s} {np.mean([r['L3'][0] for r in rs]):6.3f} "
          f"{reg['far He1'].sum() / n:8.2f} {reg['large h3'].sum() / n:9.2f} {H[:, 0].std():6.3f} {H[:, 1].std():6.3f}")


if __name__ == "__main__":
    main()
