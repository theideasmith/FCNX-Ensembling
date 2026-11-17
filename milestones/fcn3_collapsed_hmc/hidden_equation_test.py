#!/usr/bin/env python3
"""Test the theory's hidden-layer equation (E2) on its own: feed it every input from the sampler and compare the
distribution it predicts with the sampler's hidden layer.

(E2):  -log p(h, q_h) = 1/2 h^T G^-1 h - (M/2 - 1) log q_h + M q_h / (2 qbar) - g~/2 (t . e(h, q_h))^2.
Inputs measured from the saved networks (nets/, 45 per P) of the onset runs (d = 150, kappa = 0.015):
    G = sigma_B^2 J, qbar, M      from the read-in weights (E6-E8, as in empirical_kernel_test.py)
    t = (y - f_bar) / (kappa/P)    from the network's own mean output, f_bar_n = sum_k a_bar_k e_n(h_k, q_h,k), with
                                   a_bar the exact posterior mean of the readout on the training set and e_n the
                                   neuron's population output projections (checked against the sampler's own record
                                   of f_bar_1..7).
If (E2) with these inputs reproduces the sampler's h|He1 and h|He3, the hidden-layer equation is right and the
disagreement is inherited from the read-in layer and the readout residual (L3); if not, (E2) itself is missing
something.  Also shown: (E2) with the theory's own read-in kernels but the sampler's t.

    python hidden_equation_test.py [--Ps 3000 12000 24000 48000]   ->  plots/onset_checks/hidden_equation_test_d150.png
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
import empirical_kernel_test as ekt
import fcn3_fixed as F
from fcn3_matrix_solver import DT, Params, hermite_norm, jxx, readin_features
from fcn3_sampler import data_fingerprint, make_data

HERE = Path(__file__).resolve().parent


def e_of(h, qh, ks, dev):
    """e_n(h, q_h) = E_x0[phi_n(x0) erf(sum_m h_m phi_m(x0) / sqrt(1 + 2 q_h))]  for rows of h (n, S)."""
    from numpy.polynomial.hermite_e import hermegauss
    x, wx = hermegauss(64)
    Phi = hermite_norm(ks, torch.tensor(x, dtype=DT, device=dev))
    WPhi = (Phi * torch.tensor(wx / wx.sum(), dtype=DT, device=dev)).T
    return torch.erf((h @ Phi) / torch.sqrt(1 + 2 * qh)[:, None]) @ WPhi


def network_fbar(X, Y, W, B, v, sa2, ks, chunk=8192):
    """Population output projections of the network's mean predictor, and each hidden neuron's (h, q_h)."""
    dev = X.device
    N2 = B.shape[0]
    G = torch.zeros(N2, N2, dtype=torch.float64, device=dev)
    r = torch.zeros(N2, dtype=torch.float64, device=dev)
    for i in range(0, X.shape[0], chunk):
        F2 = torch.erf(torch.erf(X[i:i + chunk] @ W.T) @ B.T).double()
        G += F2.T @ F2
        r += F2.T @ Y[i:i + chunk].double()
    M = G / v + torch.eye(N2, dtype=torch.float64, device=dev) / sa2
    abar = torch.linalg.solve(M, r / v)
    Wd, Bd = W.double(), B.double()
    C = readin_features(Wd[:, 0], (Wd[:, 1:] ** 2).sum(1), ks)                 # (S, N1)
    h = Bd @ C.T                                                                 # (N2, S)
    nrm = 1 + 2 * (Wd * Wd).sum(1)
    Jf = (2 / math.pi) * torch.asin((2 * Wd @ Wd.T / torch.sqrt(torch.outer(nrm, nrm))).clamp(-1, 1))
    Jp = Jf - C.T @ C
    qh = ((Bd @ Jp) * Bd).sum(1)
    e = e_of(h, qh, ks, dev)
    return (abar @ e).cpu().numpy(), h, qh


def sample_and_marginals(P, J, T, M, t, dev):
    """Particles of (E2) at fixed (J, T, M, t) (no readout solve), then the marginals."""
    p = po._params(P)
    Fo = F.BlazingFreeEnergy(p, device=dev)
    Fo.last = dict(J=J, T=T)
    tt = torch.tensor(t, dtype=DT, device=Fo.dev)
    Fo.t = tt
    Fo.far_comps = []
    Fo.mc = F.HiddenMC(p, Fo.dev, n_log2=15, seed=0)
    Fo.mc_ready = False
    Fo.build_proposal(J, T, M, tt)
    Fo.build_proposal(J, T, M, tt)
    G = p.sigB ** 2 * J
    qbar = p.sigB ** 2 * T
    with torch.no_grad():
        e, h, q = Fo.mc.features(F.scaled_cholesky(G), qbar)
        w = torch.softmax(Fo.mc.logw(M) + 0.5 * p.gt * (e @ tt) ** 2, 0)
        n = 2 ** 14
        u = (torch.rand(1, dtype=w.dtype, device=w.device, generator=torch.Generator(w.device).manual_seed(3))
             + torch.arange(n, dtype=w.dtype, device=w.device)) / n
        idx = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)
        H = ((h * w[:, None]).T @ h).cpu().numpy()
        Kt = ((e * w[:, None]).T @ e) @ tt
    r = dict(G=G.cpu().numpy(), qbar=float(qbar), M=float(M), t=np.asarray(t), gt=float(p.gt), ks=np.array(p.ks),
             H=H, hs=h[idx].cpu().numpy(), qs=q[idx].cpu().numpy())
    L3_implied = float((p.sigA ** 2 * Kt)[1] / Fo.y[1])          # f_bar = sigma_A^2 K t with this distribution
    return r, float(1 / (w * w).sum()), L3_implied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ps", type=int, nargs="+", default=[3000, 12000, 24000, 48000])
    ap.add_argument("--nets", type=int, default=5, help="saved networks per run used for f_bar")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    res = {}
    for P in a.Ps:
        rs = [r for r in groups[P] if (r["path"] / "nets").exists()]
        ks = tuple(int(k) for k in z[f"ks_{P}"])
        p = po._params(P)
        y = np.array(p.y())
        fbars, checks = [], []
        for r0 in rs:
            cfg = json.loads((r0["path"] / "config.json").read_text())
            args = cfg["args"]
            X, Y, _, _ = make_data(args["d"], args["P"], args["eps"], 1.0, args["seed"], dev)
            X, Y = X.float(), Y.float()
            if "data_fingerprint" in cfg:
                assert data_fingerprint(X) == cfg["data_fingerprint"]
            v, sa2 = args["kappa"] / args["chi"], args["sigA"] ** 2 / (args["N2"] * args["chi"])
            fb_run = []
            for f in sorted((r0["path"] / "nets").glob("net_*.pt"))[-a.nets:]:
                nt = torch.load(f, map_location="cpu", weights_only=False)
                W = (nt["sw"] * nt["ZW"]).to(dev).float()
                B = (nt["sb"] * nt["ZB"]).to(dev).float()
                fb, _, _ = network_fbar(X, Y, W, B, v, sa2, ks)
                fb_run.append(fb)
            fb_run = np.mean(fb_run, 0)
            fbars.append(fb_run)
            rows = [x for x in csv.DictReader(open(r0["path"] / "obs.csv")) if x["burn"] == "0"]
            rec = [np.mean([float(x[f"fbar{n}"]) for x in rows]) for n in (1, 3, 5, 7)]
            checks.append((fb_run[:4], rec))
            del X, Y
        fbar = np.mean(fbars, 0)
        t_s = (y - fbar) / p.lam
        for mine, rec in checks:
            print(f"P={P}: f_bar_1..7 from saved networks {np.round(mine, 4)}  vs the sampler's record {np.round(rec, 4)}")
        print(f"P={P}: sampler L3 {fbar[1] / y[1]:.4f}; t sampler {np.round(t_s, 0)}  vs theory {np.round(z[f't_{P}'], 0)}",
              flush=True)
        J_s, T_s, Ek2, _, _ = ekt.readin_from_nets(rs, ks, dev)
        sb2 = po.SIGB ** 2
        M_s = (sb2 * T_s) ** 2 / (sb2 ** 2 * (Ek2 - float((J_s * J_s).sum())))
        J_t = torch.tensor(z[f"G_{P}"] / sb2, dtype=DT, device=dev)
        T_t, M_t = float(z[f"qbar_{P}"]) / sb2, float(z[f"M_{P}"])
        res[P] = {}
        for label, (J, T, M) in (("all sampler inputs", (J_s, T_s, M_s)), ("theory read-in, sampler t", (J_t, T_t, M_t))):
            r, ess, L3i = sample_and_marginals(P, J, T, M, t_s, dev)
            h1g, lp1, h3g, lp3 = po.hidden_marginals(r, a.device)
            res[P][label] = dict(h1g=h1g, lp1=lp1, h3g=h3g, lp3=lp3)
            print(f"   (E2) with {label}: sd h1 {math.sqrt(r['H'][0][0]):.4f} h3 {math.sqrt(r['H'][1][1]):.4f}; "
                  f"L3 implied by sigma_A^2 K t {L3i:.4f}; ESS {ess:.0f}", flush=True)
        H, _ = po.sampler_hidden(rs)
        res[P]["sampler"] = H
        print(f"   sampler hidden layer: sd h1 {H[:, 0].std():.4f} h3 {H[:, 1].std():.4f}", flush=True)
    np.savez(HERE / "plots" / "cache" / "hidden_equation_test_d150.npz",
             **{f"{lab}|{k}_{P}": v for P, d in res.items() for lab, dd in d.items() if lab != "sampler"
                for k, v in dd.items()})

    plt = po.style()
    fig, axs = plt.subplots(2, len(a.Ps), figsize=(4.2 * len(a.Ps), 7.4), squeeze=False)
    sty = {"all sampler inputs": ("#1baf7a", "-", "(E2) with every input from the sampler (read-in kernels and t)"),
           "theory read-in, sampler t": ("#7b5ce0", "--", "(E2) with the theory's read-in kernels and the sampler's t")}
    for c, P in enumerate(a.Ps):
        for i, (j, xl) in enumerate(((1, "h | He$_1$"), (3, "h | He$_3$"))):
            ax = axs[i, c]
            x = res[P]["sampler"][:, i]
            g0, y0 = z[f"h{j}g_{P}"], -z[f"lp{j}_{P}"]
            curves = [(g0, y0, po.C_THEORY, "-", "theory (self-consistent)")]
            for lab, (col, ls, name) in sty.items():
                curves.append((res[P][lab][f"h{j}g"], -res[P][lab][f"lp{j}"], col, ls, name))
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
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Hidden-layer equation (E2) fed the sampler's own inputs, d = {po.D}, κ = {po.KAPPA}", color=po.INK,
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "hidden_equation_test_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
