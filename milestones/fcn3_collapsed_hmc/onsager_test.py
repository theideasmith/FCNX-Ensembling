#!/usr/bin/env python3
"""Does the readout reaction (Onsager) term, which mean-field decoupling drops, explain the hidden-layer mismatch at
large P?

Exact identity in the sampler's collapsed posterior (cavity_decomposition.py): hidden neuron k's fit gain is
    F_k = 1/2 s_a2 (phi_k^T t)^2 / (1 - l_k),        l_k = 1 - [M^-1]_kk / s_a2   (its readout leverage),
with t the network's own residual dual. The theory's hidden-field tilt is 1/2 g~ (t.e)^2, the same without the factor
1/(1 - l). The sampler's leverage is nearly the same for every hidden neuron (sd <= 0.02) and grows with P
(0.04, 0.08, 0.15, 0.21, 0.31 at P = 750 ... 48000), so the correction is a uniform rescaling g~ -> g~/(1 - l).
With the readout condition (sigma_A^2 K + kappa/P) t = y unchanged, that is exactly the hidden-layer + readout solve
with an effective hidden width N2 (1 - l) (g~ = sigma_A^2 chi / N2 and the readout potential's prefactor N2/chi are the
only places N2 enters there).

Here the read-in kernels (J, qbar, M) are held at the theory's own solution (plots/cache/theory_onset_d150_v6.npz) and only
the hidden layer and readout are re-solved, once with the plain tilt (a check: it must reproduce the theory) and once
with g~/(1 - l), l from the sampler at that P.

    python onsager_test.py [--Ps 3000 12000 24000 48000]   ->  plots/onset_checks/onsager_test_d150.png
"""
import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
import fcn3_fixed as F
from fcn3_matrix_solver import DT

HERE = Path(__file__).resolve().parent


def solve_hidden(p, J, T, M, t0, dev, rounds=4):
    Fo = F.BlazingFreeEnergy(p, device=dev)
    Fo.last = dict(J=J, T=T)
    Fo.t = torch.tensor(t0, dtype=DT, device=Fo.dev)
    Fo.nprop, Fo.far_comps = 0, []
    t_prev = None
    for r in range(rounds):
        Fo.set_samples(15, M, fresh=(r == 0))
        if t_prev is not None and float((Fo.t - t_prev).abs().max() / Fo.t.abs().max()) < 1e-6:
            break
        t_prev = Fo.t.clone()
    G = p.sigB ** 2 * J
    qbar = p.sigB ** 2 * T
    t, info = Fo.solve_t(F.scaled_cholesky(G), qbar, M, Fo.t)
    fb = (Fo.y - p.lam * t).cpu().numpy()
    w, h, q = info["w"], info["h"], info["q"]
    n = 2 ** 14
    u = (torch.rand(1, dtype=w.dtype, device=w.device, generator=torch.Generator(w.device).manual_seed(3))
         + torch.arange(n, dtype=w.dtype, device=w.device)) / n
    idx = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)
    H = ((h * w[:, None]).T @ h).cpu().numpy()
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
    cav = dict(np.load(HERE / "plots" / "cache" / "cavity_decomposition_d150.npz"))
    groups = po.by_P(po.sampler_runs())
    po.set_outer([r for rs in groups.values() for r in rs], z)
    res = {}
    for P in a.Ps:
        lev = float((1 - np.exp(-2 * cav[f"P{P}"][:, 3])).mean())
        sb2 = po.SIGB ** 2
        J = torch.tensor(z[f"G_{P}"] / sb2, dtype=DT, device=dev)
        T = float(z[f"qbar_{P}"]) / sb2
        M = float(z[f"M_{P}"])
        p0 = po._params(P)
        res[P] = {}
        for label, p in (("plain", p0), ("onsager", replace(p0, N2=p0.N2 * (1 - lev)))):
            r, L1, L3, ess = solve_hidden(p, J, T, M, z[f"t_{P}"], dev)
            h1g, lp1, h3g, lp3 = po.hidden_marginals(r, a.device)
            sd = [math.sqrt(r["H"][i][i]) for i in (0, 1)]
            res[P][label] = dict(h1g=h1g, lp1=lp1, h3g=h3g, lp3=lp3, L3=L3)
            print(f"P={P} {label:8s} (leverage {lev:.3f}, g~ x {1 / (1 - lev) if label == 'onsager' else 1:.3f}): "
                  f"L1 {L1:.4f} L3 {L3:.4f}; sd h1 {sd[0]:.4f} h3 {sd[1]:.4f}; ESS {ess:.0f}", flush=True)
        Ls = po.per_dataset(groups[P])
        H, _ = po.sampler_hidden([r for r in groups[P] if (r["path"] / "nets").exists()])
        res[P]["sampler"] = H
        print(f"P={P} sampler: L3 {np.mean([x['L3'][0] for x in Ls]):.4f}; sd h1 {H[:, 0].std():.4f} h3 {H[:, 1].std():.4f}"
              f"; theory (full solve) L3 {z[f'sc_{P}'][1]:.4f}", flush=True)

    plt = po.style()
    fig, axs = plt.subplots(2, len(a.Ps), figsize=(4.2 * len(a.Ps), 7.4), squeeze=False)
    for c, P in enumerate(a.Ps):
        for i, (j, xl) in enumerate(((1, "h | He$_1$"), (3, "h | He$_3$"))):
            ax = axs[i, c]
            x = res[P]["sampler"][:, i]
            curves = [(res[P]["plain"], po.C_THEORY, "-", "theory"),
                      (res[P]["onsager"], "#7b5ce0", "--", "theory with the readout reaction g~/(1 - l)")]
            m = [np.abs(x).max()]
            for cdat, *_ in curves:
                g, y = cdat[f"h{j}g"], -cdat[f"lp{j}"]
                m.append(np.abs(g[y <= y.min() + 9]).max())
            lim = 1.08 * max(m)
            lo = np.inf
            for cdat, col, ls, lab in curves:
                g, y = cdat[f"h{j}g"], -cdat[f"lp{j}"]
                k = np.abs(g) <= lim
                ax.plot(g[k], y[k], ls, color=col, lw=2, label=lab, zorder=2)
                lo = min(lo, y.min())
            edges = np.linspace(-lim, lim, 81)
            mid = 0.5 * (edges[1:] + edges[:-1])
            cnt, _ = np.histogram(x, edges)
            k = cnt > 0
            yy = -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k]))
            ax.errorbar(mid[k], yy, yerr=1 / np.sqrt(cnt[k]), xerr=0.5 * np.diff(edges)[k], fmt="o", ms=2.8,
                        color=po.C_HMC, mec="white", mew=0.3, elinewidth=0.7, capsize=0, zorder=4, label="sampler")
            lo = min(lo, yy.min())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(lo - 0.5, lo + 10)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(f"−log p({xl})")
            if i == 0:
                ax.set_title(f"P = {P}", color=po.INK, loc="left", fontsize=12)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Second hidden layer, d = {po.D}, κ = {po.KAPPA}: read-in layer held at the theory's; hidden layer and "
                 f"readout solved with and without the readout reaction", color=po.INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "onsager_test_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
