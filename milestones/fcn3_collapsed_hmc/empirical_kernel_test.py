#!/usr/bin/env python3
"""Does the theory's hidden layer reproduce the sampler's when it is fed the sampler's own read-in layer?

The onset runs (queue_onset.sh, d = 150, kappa = 0.015) disagree with the theory in the second hidden layer at large P
(He3 splits into two wells in the sampler, the far He1 group sits at 2.2 instead of 1.5), but the sampler's read-in
layer also differs from the theory's there (about half the outer read-in neurons). This test separates the two.

From every saved network (nets/ of the onset runs: 45 per P), the read-in kernels that enter the hidden layer are
computed exactly as the theory defines them, but over the sampler's read-in neurons instead of the theory's read-in
distribution:
    J    = E_j[c(w0_j, q_j) c(w0_j, q_j)^T]                                     (E6)
    qbar = sigma_B^2 E_j[ ||erf(w_j.x)||^2 - |c(w0_j, q_j)|^2 ]                  (E7)
    M    = qbar^2 / sigma_B^4 (E_{j,j'} k(w_j, w_j')^2 - ||J||_F^2)             (E8)
(E8 with the theory's convention: pairs of independent neurons with a random relative perpendicular direction; the
value from the actual pairs of each network is printed alongside.) With J, qbar, M held fixed, the hidden layer and
the readout, (E2)-(E4), are solved with the same machinery as the full theory (fcn3_fixed.BlazingFreeEnergy:
importance-sampled hidden field, convex readout problem), and the hidden-layer marginals are computed as in
plot_onset.hidden_marginals.

    python empirical_kernel_test.py [--Ps 3000 12000 24000 48000]   ->  plots/onset_checks/empirical_kernel_test_d150.png
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
import fcn3_fixed as F
from fcn3_matrix_solver import DT, jxx, readin_features

HERE = Path(__file__).resolve().parent


def readin_from_nets(rs, ks, dev):
    w0s, qs, Mpair = [], [], []
    for r in rs:
        for f in sorted((r["path"] / "nets").glob("net_*.pt")):
            nt = torch.load(f, map_location="cpu", weights_only=False)
            W = nt["sw"] * nt["ZW"].double()
            w0s.append(W[:, 0])
            qs.append((W[:, 1:] ** 2).sum(1))
            if len(Mpair) < 3:                                   # actual pairs, a few networks
                W = W.to(dev)
                n2 = (W * W).sum(1)
                Tn = 1 + 2 * n2
                k = (2 / math.pi) * torch.asin((2 * W @ W.T / torch.sqrt(torch.outer(Tn, Tn))).clamp(-1, 1))
                off = ~torch.eye(len(W), dtype=torch.bool, device=dev)
                Mpair.append(float((k[off] ** 2).mean()))
    w0 = torch.cat(w0s).to(dev)
    q = torch.cat(qs).to(dev)
    C = readin_features(w0, q, ks)                               # (S, n)
    J = (C @ C.T) / C.shape[1]
    T = float((jxx(w0 * w0 + q) - (C * C).sum(0)).mean())
    # E k^2 over independent pairs with a random relative perpendicular direction (the theory's convention)
    from scipy.special import roots_jacobi
    d = po.D
    u, wu = roots_jacobi(64, (d - 4) / 2, (d - 4) / 2)
    u = torch.tensor(u, dtype=DT, device=dev)
    wu = torch.tensor(wu / wu.sum(), dtype=DT, device=dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(len(w0), generator=g)[:3000].to(dev)
    a0, aq = w0[idx], q[idx]
    Ta = 1 + 2 * (a0 * a0 + aq)
    z = (2 * torch.outer(a0, a0)[..., None] + 2 * torch.outer(aq.sqrt(), aq.sqrt())[..., None] * u) \
        / torch.sqrt(torch.outer(Ta, Ta))[..., None]
    Ek2 = float((((2 / math.pi) * torch.asin(z.clamp(-1, 1))) ** 2 @ wu).mean())
    return J, T, Ek2, float(np.mean(Mpair)), len(w0) // 1500


def solve_hidden(P, J, T, M, t0, dev, rounds=4):
    """(E2)-(E4) at fixed read-in kernels J, T (qbar = sigB^2 T) and M: hidden-field proposal, readout t."""
    p = po._params(P)
    Fo = F.BlazingFreeEnergy(p, device=dev)
    Fo.last = dict(J=J, T=T)
    Fo.t = torch.tensor(t0, dtype=DT, device=Fo.dev)
    Fo.nprop = 0
    Fo.far_comps = []
    t_prev = None
    for r in range(rounds):                                     # proposal and t until t stops moving
        Fo.set_samples(15, M, fresh=(r == 0))
        if t_prev is not None and float((Fo.t - t_prev).abs().max() / Fo.t.abs().max()) < 1e-6:
            break
        t_prev = Fo.t.clone()
    G = p.sigB ** 2 * J
    L = F.scaled_cholesky(G)
    qbar = p.sigB ** 2 * T
    t, info = Fo.solve_t(L, qbar, M, Fo.t)
    fb = (Fo.y - p.lam * t).cpu().numpy()
    w, h, q = info["w"], info["h"], info["q"]
    H = ((h * w[:, None]).T @ h).cpu().numpy()
    n = 2 ** 14
    u = (torch.rand(1, dtype=w.dtype, device=w.device, generator=torch.Generator(w.device).manual_seed(3))
         + torch.arange(n, dtype=w.dtype, device=w.device)) / n
    idx = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)
    r = dict(G=G.cpu().numpy(), qbar=float(qbar), M=float(M), t=t.cpu().numpy(), gt=float(p.gt), ks=np.array(p.ks),
             H=H, hs=h[idx].cpu().numpy(), qs=q[idx].cpu().numpy())
    return r, float(fb[0]), float(fb[1] / p.eps), float(info["ess"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[3000, 12000, 24000, 48000])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    po.set_outer([r for rs in groups.values() for r in rs], z)
    res = {}
    for P in a.Ps:
        rs = [r for r in groups[P] if (r["path"] / "nets").exists()]
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        J, T, Ek2, Ek2_pairs, nnet = readin_from_nets(rs, ks, dev)
        sb2 = po.SIGB ** 2
        qbar = sb2 * T
        Mth = qbar ** 2 / (sb2 ** 2 * (Ek2 - float((J * J).sum())))
        Mpr = qbar ** 2 / (sb2 ** 2 * (Ek2_pairs - float((J * J).sum())))
        Gth = z[f"G_{P}"] / sb2
        print(f"P={P}: {nnet} networks. Read-in kernels, sampler / theory: J11 {float(J[0, 0]):.5f} / {Gth[0, 0]:.5f}, "
              f"J33 {float(J[1, 1]):.6f} / {Gth[1, 1]:.6f}, J13 {float(J[0, 1]):.6f} / {Gth[0, 1]:.6f}, "
              f"qbar {qbar:.4f} / {float(z[f'qbar_{P}']):.4f}, M {Mth:.1f} (actual pairs {Mpr:.1f}) / "
              f"{float(z[f'M_{P}']):.1f}", flush=True)
        r, L1, L3, ess = solve_hidden(P, J, T, Mth, z[f"t_{P}"], dev)
        h1g, lp1, h3g, lp3 = po.hidden_marginals(r, a.device)
        sc = z[f"sc_{P}"]
        Ls = po.per_dataset(groups[P])
        print(f"   L3: sampler {np.mean([x['L3'][0] for x in Ls]):.4f}, theory {sc[1]:.4f}, theory with the "
              f"sampler's read-in kernels {L3:.4f}  (L1 {L1:.4f}; ESS {ess:.0f})", flush=True)
        res[P] = dict(h1g=h1g, lp1=lp1, h3g=h3g, lp3=lp3, L3=L3, L1=L1)
        H, _ = po.sampler_hidden(rs)
        res[P]["sampler"] = H
    np.savez(HERE / "plots" / "cache" / "empirical_kernel_test_d150.npz",
             **{f"{k}_{P}": v for P, d in res.items() for k, v in d.items() if k != "sampler"})

    plt = po.style()
    fig, axs = plt.subplots(2, len(a.Ps), figsize=(4.2 * len(a.Ps), 7.4), squeeze=False)
    C_EMP = "#1baf7a"
    for c, P in enumerate(a.Ps):
        for i, (j, xl) in enumerate(((1, "h | He$_1$"), (3, "h | He$_3$"))):
            ax = axs[i, c]
            x = res[P]["sampler"][:, i]
            g0, y0 = z[f"h{j}g_{P}"], -z[f"lp{j}_{P}"]
            g1, y1 = res[P][f"h{j}g"], -res[P][f"lp{j}"]
            lim = 1.08 * max(np.abs(x).max(), np.abs(g0[y0 <= y0.min() + 9]).max(), np.abs(g1[y1 <= y1.min() + 9]).max())
            edges = np.linspace(-lim, lim, 81)
            mid = 0.5 * (edges[1:] + edges[:-1])
            cnt, _ = np.histogram(x, edges)
            k = cnt > 0
            y = -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k]))
            ax.plot(g0[np.abs(g0) <= lim], y0[np.abs(g0) <= lim], color=po.C_THEORY, lw=2, label="theory", zorder=2)
            ax.plot(g1[np.abs(g1) <= lim], y1[np.abs(g1) <= lim], "--", color=C_EMP, lw=2,
                    label="theory, hidden layer fed the sampler's read-in kernels", zorder=3)
            ax.errorbar(mid[k], y, yerr=1 / np.sqrt(cnt[k]), xerr=0.5 * np.diff(edges)[k], fmt="o", ms=2.8,
                        color=po.C_HMC, mec="white", mew=0.3, elinewidth=0.7, capsize=0, zorder=4, label="sampler")
            lo = min(y.min(), y0.min(), y1.min())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(lo - 0.5, lo + 10)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"−log p({xl})")
            if i == 0:
                ax.set_title(f"P = {P}", color=po.INK, loc="left", fontsize=12)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Second hidden layer, d = {po.D}, κ = {po.KAPPA}: does the theory's hidden layer reproduce the "
                 f"sampler's when fed the sampler's read-in kernels?", color=po.INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "empirical_kernel_test_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
