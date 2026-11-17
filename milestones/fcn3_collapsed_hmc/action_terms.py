#!/usr/bin/env python3
"""Term by term: which part of the hidden-neuron action (E2) sends the theory the wrong way?

(E2):  -log p(h, q_h) = 1/2 h^T G^-1 h  -  (M/2 - 1) log q_h + M q_h / (2 qbar)  -  1/2 g~ (t . e)^2
                        [prior on h, split into the He1-He3 block and the rest given them]  [prior on q_h]  [x0 gain]
plus, in the networks, the perpendicular gain  -1/2 g~ [(t.e + c)^2 - (t.e)^2],  c = <t_perp | erf o h>.

Each term is evaluated with two sets of ingredients:
    theory    G, qbar, M, t at the theory's own solution (plots/cache/theory_onset_d150_v6.npz)
    sampler   each network's own read-in kernel G = sigma_B^2 J_hat (E6 over its read-in neurons), its qbar (E7), M
              (E8, pooled), its population residual t, and the measured c (measure_perp_pull.py: same 9 networks)
on two populations:
    sampler neurons   every hidden neuron of those networks, its own h (He1..He11 tower), q_h and exact e
    theory particles  the theory's hidden-field particles at its solution (e from the theory's formula; for the
                      sampler's perpendicular gain c ~ gamma . e, gamma fitted on the networks, all six modes)
grouped as ordinary / far He1 / large h3 (he3_route.regions). Each group's mean is reported relative to the ordinary
group's mean: only differences between states decide where neurons sit. A term that differs between the two ingredient
sets at the same states is one pushing the theory differently from the networks.

    PYTHONPATH=.:../fcn3_matrix_theory python action_terms.py [--Ps 24000 48000]
"""
import argparse
import json
import math

import numpy as np
import torch

import empirical_kernel_test as ekt
import he3_route as hr
import hidden_equation_test as het
import plot_onset as po
from fcn3_matrix_solver import DT, jxx, readin_features

HERE = po.HERE
TERMS = ["prior h (He1,He3)", "prior h (rest | He1,He3)", "prior q_h", "x0 gain", "perp gain", "total"]


def quad(h, G):
    """1/2 h^T G^-1 h per row, with diagonal scaling (G is nearly singular); also its He1-He3 block part."""
    D = np.sqrt(np.diag(G))
    R = G / np.outer(D, D)
    x = h / D
    full = 0.5 * np.einsum("ni,ni->n", x, np.linalg.solve(R, x.T).T)
    x2 = x[:, :2]
    blk = 0.5 * np.einsum("ni,ni->n", x2, np.linalg.solve(R[:2, :2], x2.T).T)
    return blk, full - blk


def terms(h, q, e, G, qbar, M, t, gt, c=None):
    blk, rest = quad(h, G)
    pq = -(M / 2 - 1) * np.log(q) + M * q / (2 * qbar)
    s = e @ t
    x0 = -0.5 * gt * s * s
    perp = np.zeros_like(s) if c is None else -0.5 * gt * ((s + c) ** 2 - s * s)
    return np.stack([blk, rest, pq, x0, perp, blk + rest + pq + x0 + perp], 1)


def report(name, T, reg):
    base = T[reg["rest"]].mean(0)
    for k in ("far He1", "large h3"):
        m = reg[k]
        if m.sum() < 3:
            continue
        d = T[m].mean(0) - base
        print(f"      {k:9s} " + "  ".join(f"{v:10.1f}" for v in d))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ps", type=int, nargs="+", default=[24000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    sb2 = po.SIGB ** 2
    for P in a.Ps:
        p = po._params(P)
        gt = p.gt
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        th = dict(G=z[f"G_{P}"], qbar=float(z[f"qbar_{P}"]), M=float(z[f"M_{P}"]), t=z[f"t_{P}"])
        pp = dict(np.load(HERE / "plots" / "cache" / f"perp_pull_P{P}.npz"))
        rs = [r for r in groups[P] if (r["path"] / "nets").exists()]
        J_all, T_all, Ek2, _, _ = ekt.readin_from_nets(rs, ks, dev)
        M_s = (sb2 * T_all) ** 2 / (sb2 ** 2 * (Ek2 - float((J_all * J_all).sum())))
        # sampler neurons, network by network, in the order measure_perp_pull.py used
        rows_s, rows_t, regs, Gs, qbs = [], [], [], [], []
        n = 0
        for r0 in rs:
            for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(f, map_location="cpu", weights_only=False)
                W = (nt["sw"] * nt["ZW"]).to(dev).double()
                B = (nt["sb"] * nt["ZB"]).to(dev).double()
                w0, qw = W[:, 0], (W[:, 1:] ** 2).sum(1)
                C = readin_features(w0, qw, ks)                               # (S, N1)
                G = (sb2 * (C @ C.T) / C.shape[1]).cpu().numpy()
                qbar = float(sb2 * (jxx(w0 * w0 + qw) - (C * C).sum(0)).mean())
                h = (B @ C.T).cpu().numpy()
                nrm = 1 + 2 * (W * W).sum(1)
                Jf = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
                qh = ((B @ (Jf - C.T @ C)) * B).sum(1).cpu().numpy()
                m = pp["net"] == n
                assert np.allclose(pp["h"][m], h, rtol=1e-4, atol=1e-6), "network order differs from perp_pull"
                e, c, t_s = pp["e"][m], pp["c"][m], pp["t"][n]
                rows_s.append(terms(h, qh, e, G, qbar, M_s, t_s, gt, c))
                rows_t.append(terms(h, qh, e, th["G"], th["qbar"], th["M"], th["t"], gt))
                regs.append(hr.regions(h[:, 0], h[:, 1]))
                Gs.append(G)
                qbs.append(qbar)
                n += 1
        S_s, S_t = np.concatenate(rows_s), np.concatenate(rows_t)
        reg = {k: np.concatenate([r[k] for r in regs]) for k in regs[0]}
        # theory particles
        hs, qs = z[f"hs_{P}"], z[f"qs_{P}"]
        e_p = het.e_of(torch.tensor(hs, dtype=DT, device=dev), torch.tensor(qs, dtype=DT, device=dev), ks,
                       dev).cpu().numpy()
        gam, *_ = np.linalg.lstsq(pp["e"], pp["c"], rcond=None)
        P_t = terms(hs, qs, e_p, th["G"], th["qbar"], th["M"], th["t"], gt)
        P_s = terms(hs, qs, e_p, np.mean(Gs, 0), np.mean(qbs), M_s, pp["t"].mean(0), gt, e_p @ gam)
        reg_p = hr.regions(hs[:, 0], hs[:, 1])

        print(f"\nP = {P}   (nats; each group's mean minus the ordinary neurons' mean; negative favours the group)")
        print(f"   ingredients: theory qbar {th['qbar']:.4f} M {th['M']:.1f}; sampler qbar {np.mean(qbs):.4f} M {M_s:.1f}")
        print("      group     " + "  ".join(f"{s[:10]:>10s}" for s in ["prior He13", "prior rest", "prior q_h",
                                                                          "x0 gain", "perp gain", "total"]))
        cnt = lambda rg: ", ".join(f"{k} {rg[k].sum() / n:.1f}" for k in ("far He1", "large h3"))
        print(f"   SAMPLER NEURONS ({cnt(reg)} per network)")
        print("    with the sampler's ingredients")
        report("s", S_s, reg)
        print("    with the theory's ingredients")
        report("t", S_t, reg)
        frac = {k: reg_p[k].mean() * po.N for k in ("far He1", "large h3")}
        print(f"   THEORY PARTICLES (far He1 {frac['far He1']:.1f}, large h3 {frac['large h3']:.1f} per network)")
        print("    with the theory's ingredients")
        report("t", P_t, reg_p)
        print("    with the sampler's ingredients")
        report("s", P_s, reg_p)
        # where the h-prior difference comes from: the two G's
        Gt, Gm = th["G"], np.mean(Gs, 0)
        print("   prior covariance diagonal sqrt(G_nn), theory / sampler:",
              np.round(np.sqrt(np.diag(Gt)) / np.sqrt(np.diag(Gm)), 3))
        print("   eigenvalues of G (theory)  ", np.array2string(np.linalg.eigvalsh(Gt), precision=2))
        print("   eigenvalues of G (sampler) ", np.array2string(np.linalg.eigvalsh(Gm), precision=2))
        for k in ("far He1", "large h3"):
            if reg[k].sum():
                hh = pp["h"][reg[k]]
                print(f"   sampler {k} mean |h| per mode: {np.round(np.abs(hh).mean(0), 3)}")
            if reg_p[k].sum() > 3:
                print(f"   theory  {k} mean |h| per mode: {np.round(np.abs(hs[reg_p[k]]).mean(0), 3)}")


if __name__ == "__main__":
    main()
