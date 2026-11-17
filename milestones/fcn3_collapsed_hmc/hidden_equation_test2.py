#!/usr/bin/env python3
"""(E2) driven by the force the hidden neurons actually feel.

In the sampler's exact posterior hidden neuron k's fit gain is 1/2 s_a2 (phi_k . alpha)^2 / (1 - l_k), with
alpha = (y - f_bar) / v the TRAINING-set residual (cavity_decomposition.py). The theory's tilt is 1/2 g~ (t . e_k)^2
with t the POPULATION residual / (kappa/P); the training residual is about half the population residual here
(networks fit their training sets ~3x better than they generalise), so (E2) fed the sampler's population t blows up
(hidden_equation_test.py). Here t_eff is the least-squares fit of the exact pulls phi_k . alpha over all hidden neurons,
pull_k ~ e_k . t_eff, the reaction factor 1/(1 - l) is included (t_eff / sqrt(1 - l)), and the read-in kernels are the
sampler's. If (E2) then reproduces the sampler's hidden layer, its functional form is right and the disagreement comes
from the training-set residual (and the read-in layer) that the population-average theory cannot represent.

    python hidden_equation_test2.py   ->  plots/onset_checks/hidden_equation_test2_d150.png
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
import empirical_kernel_test as ekt
from hidden_equation_test import e_of, sample_and_marginals
from fcn3_matrix_solver import DT, readin_features
from fcn3_sampler import make_data

HERE = Path(__file__).resolve().parent


def pulls(X, Y, W, B, v, sa2, ks):
    """Exact pull phi_k . alpha of every hidden neuron, its population output projections e_k, its leverage, the
    population projections of the network's mean output (f_bar_n = a_bar . e_n) and the He projections of the training
    residual, (1/P) sum_mu r_mu phi_n(x0_mu)."""
    from fcn3_matrix_solver import hermite_norm
    dev = X.device
    Phi = torch.cat([torch.erf(torch.erf(X[i:i + 8192] @ W.T) @ B.T) for i in range(0, X.shape[0], 8192)]).double()
    M = Phi.T @ Phi / v + torch.eye(B.shape[0], dtype=torch.float64, device=dev) / sa2
    Minv = torch.linalg.inv(M)
    abar = Minv @ (Phi.T @ Y.double() / v)
    alpha = (Y.double() - Phi @ abar) / v
    pull = Phi.T @ alpha
    lev = 1 - torch.diagonal(Minv) / sa2
    Wd, Bd = W.double(), B.double()
    C = readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), ks)
    h = Bd @ C.T
    nrm = 1 + 2 * (Wd * Wd).sum(1)
    Jp = (2 / math.pi) * torch.asin((2 * Wd @ Wd.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1)) - C.T @ C
    qh = ((Bd @ Jp) * Bd).sum(1)
    e = e_of(h, qh, ks, dev)
    r_tr = Y.double() - Phi @ abar
    rn_tr = (hermite_norm(ks, X[:, 0].double()) @ r_tr) / X.shape[0]
    return pull, e, lev, abar @ e, rn_tr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[3000, 12000, 24000, 48000])
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    res = {}
    for P in a.Ps:
        rs = [r for r in groups[P] if (r["path"] / "nets").exists()]
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        PL, E, LV, FB, RT = [], [], [], [], []
        p = po._params(P)
        y = torch.tensor(np.array(p.y()), dtype=DT, device=dev)
        for r0 in rs:
            args = json.loads((r0["path"] / "config.json").read_text())["args"]
            X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
            X, Y = X.float(), Y.float()
            v, sa2 = args["kappa"] / args["chi"], args["sigA"] ** 2 / (args["N2"] * args["chi"])
            for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(f, map_location="cpu", weights_only=False)
                pl, e, lv, fb, rt = pulls(X, Y, (nt["sw"] * nt["ZW"]).to(dev).float(),
                                          (nt["sb"] * nt["ZB"]).to(dev).float(), v, sa2, ks)
                PL.append(pl); E.append(e); LV.append(lv); FB.append(fb); RT.append(rt)
            del X, Y
        PL, E, LV = torch.cat(PL), torch.cat(E), torch.cat(LV)
        lam = p.lam
        t_pop = (y - torch.stack(FB).mean(0)) / lam                      # population residual / (kappa/P)
        t_trn = torch.stack(RT).mean(0) / lam                             # training residual's He projections
        q_pop = E @ t_pop
        slope = float((PL * q_pop).sum() / (q_pop * q_pop).sum())
        r2 = lambda pred: 1 - float(((PL - pred) ** 2).sum() / ((PL - PL.mean()) ** 2).sum())
        lev = float(LV.mean())
        print(f"P={P}: pull explained by slope x t_pop: slope {slope:.3f}, R^2 {r2(slope * q_pop):.3f};  by t_train: "
              f"R^2 {r2(E @ t_trn):.3f};  leverage {lev:.3f}", flush=True)
        t_use = (slope * t_pop / math.sqrt(1 - lev)).cpu().numpy()
        t_use2 = (t_trn / math.sqrt(1 - lev)).cpu().numpy()
        J_s, T_s, Ek2, _, _ = ekt.readin_from_nets(rs, ks, dev)
        sb2 = po.SIGB ** 2
        M_s = (sb2 * T_s) ** 2 / (sb2 ** 2 * (Ek2 - float((J_s * J_s).sum())))
        H, _ = po.sampler_hidden(rs)
        res[P] = dict(sampler=H)
        for lab, tu in (("scaled", t_use), ("train", t_use2)):
            rr, ess, _ = sample_and_marginals(P, J_s, T_s, M_s, tu, dev)
            h1g, lp1, h3g, lp3 = po.hidden_marginals(rr, a.device)
            res[P][lab] = dict(h1g=h1g, lp1=lp1, h3g=h3g, lp3=lp3)
            print(f"   (E2), force {lab}: sd h1 {math.sqrt(rr['H'][0][0]):.4f} h3 {math.sqrt(rr['H'][1][1]):.4f} "
                  f"(ESS {ess:.0f});  sampler sd h1 {H[:, 0].std():.4f} h3 {H[:, 1].std():.4f};  theory sd h1 "
                  f"{math.sqrt(z[f'H_{P}'][0][0]):.4f} h3 {math.sqrt(z[f'H_{P}'][1][1]):.4f}", flush=True)

    plt = po.style()
    fig, axs = plt.subplots(2, len(a.Ps), figsize=(4.2 * len(a.Ps), 7.4), squeeze=False)
    for c, P in enumerate(a.Ps):
        for i, (j, xl) in enumerate(((1, "h | He$_1$"), (3, "h | He$_3$"))):
            ax = axs[i, c]
            x = res[P]["sampler"][:, i]
            curves = [(z[f"h{j}g_{P}"], -z[f"lp{j}_{P}"], po.C_THEORY, "-", "theory (self-consistent)"),
                      (res[P]["scaled"][f"h{j}g"], -res[P]["scaled"][f"lp{j}"], "#1baf7a", "--",
                       "(E2), sampler's read-in kernels, force = population residual x measured pull ratio"),
                      (res[P]["train"][f"h{j}g"], -res[P]["train"][f"lp{j}"], "#7b5ce0", ":",
                       "(E2), sampler's read-in kernels, force = training residual")]
            lim = 1.08 * max([np.abs(x).max()] + [np.abs(g[y <= y.min() + 9]).max() for g, y, *_ in curves])
            lo = np.inf
            for g, y, col, ls, name in curves:
                k = np.abs(g) <= lim
                ax.plot(g[k], y[k], ls, color=col, lw=2, label=name, zorder=2)
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
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Hidden-layer equation (E2) driven by the force the sampler's hidden neurons feel, d = {po.D}, "
                 f"κ = {po.KAPPA}", color=po.INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "hidden_equation_test2_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
