#!/usr/bin/env python3
"""Every input of the theory's read-in equation (E5), measured in the trained networks and compared with the theory.

(E5):  -log p(w0, q) = beta (w0^2 + q) - (d-3)/2 log q + (r_B/2) c^T G^-1 (G - H) G^-1 c
                       + (r_B/2) M (qbar - <q_h>)/qbar^2 P_perp(w)
with G = sigma_B^2 J. In a network (W, B):
  c_n(w_j)       closed form, n in S = {1, 3}            J = mean_j c c^T
  h_kn           = sum_j B_kj c_n(w_j)                     H = mean_k h_k h_k^T
  qbar           = sigma_B^2 (mean_j ||erf(w_j x)||^2 - tr J)
  <q_h>          = mean_k (||h_k||^2 - |h_k|_S^2),  ||h_k||^2 = B_k K_W B_k^T,  K_W the arcsine kernel of the rows
  M              = qbar^2 / sigma_B^4 (mean_{j != j'} k(w_j, w_j')^2 - ||J||_F^2)
Then (E5) is evaluated with the network's own values and its p(w0) and p(r^2) are compared with the network's snapshots.
n_max = 3 (S = {1, 3}): G is 2 x 2, so G^-1 (G - H) G^-1 is computed directly without loss of precision.

    python diag_readin_terms.py [--d 150]
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIGW2 = 0.5
SIGB = {75: 1.2574845533917596, 150: 1.25540121081682, 300: 1.254358125197598, 600: 1.2538362420680749}
S = (1, 3)
A = np.array([2 / math.sqrt(math.pi) * (-2.0) ** ((n - 1) // 2) * (1 if n == 1 else math.prod(range(n - 2, 0, -2)))
              / math.sqrt(math.factorial(n)) for n in S])


def feats(w0, q):
    """c_n (n in S), r^2, P_perp for arrays of (w0, q)."""
    D = 1 + 2 * (w0 ** 2 + q)
    c = np.stack([A[i] * w0 ** n / D ** (n / 2) for i, n in enumerate(S)], -1)
    kk = 2 / math.pi * np.arcsin(2 * (w0 ** 2 + q) / D)
    return c, w0 ** 2 / D, kk - (c ** 2).sum(-1)


def action(kern, d, sigB, r_B, w0, lq):
    """(E5) on a grid (w0 x log q), as -log density in (w0, log q) (includes the q of d log q), and its terms."""
    beta = d / (2 * SIGW2)
    W0, LQ = np.meshgrid(w0, lq, indexing="ij")
    Q = np.exp(LQ)
    c, r2, Pp = feats(W0, Q)
    G = sigB ** 2 * kern["J"]
    Gi = np.linalg.inv(G)
    V = Gi @ (G - kern["H"]) @ Gi
    qbar = sigB ** 2 * kern["T"]
    vperp = kern["M"] * (qbar - kern["Eq"]) / qbar ** 2
    terms = dict(prior=beta * (W0 ** 2 + Q) - (d - 1) / 2 * LQ,
                 V=0.5 * r_B * np.einsum("...i,ij,...j->...", c, V, c),
                 perp=0.5 * r_B * vperp * Pp)
    S_ = sum(terms.values())
    return S_ - S_.min(), terms, r2, V, vperp


def network_kernels(W, B, sigB):
    """J, H, T (= qbar / sigB^2), Eq (= <q_h>), M of a network, in S = {1, 3}."""
    w0 = W[:, 0]
    q = (W[:, 1:] ** 2).sum(1)
    c, _, _ = feats(w0, q)
    J = c.T @ c / len(w0)
    h = B @ c                                                         # (N2, |S|)
    H = h.T @ h / len(h)
    n2 = (W ** 2).sum(1)
    Dn = np.sqrt(1 + 2 * n2)
    KW = 2 / math.pi * np.arcsin(np.clip(2 * (W @ W.T) / np.outer(Dn, Dn), -1, 1))
    T = float(np.mean(np.diag(KW)) - np.trace(J))
    hn2 = np.einsum("kj,jl,kl->k", B, KW, B)                          # ||h_k||^2 over the input distribution
    Eq = float(np.mean(hn2 - (h ** 2).sum(1)))
    off = ~np.eye(len(w0), dtype=bool)
    Ek2 = float(np.mean(KW[off] ** 2))
    M = (sigB ** 2 * T) ** 2 / (sigB ** 4 * (Ek2 - (J ** 2).sum()))
    return dict(J=J, H=H, T=T, Eq=Eq, M=M)


def theory_kernels(d, device):
    sys.path.insert(0, str(HERE.parent / "fcn3_matrix_theory"))
    from fcn3_matrix_solver import Params
    import fcn3_fixed as F
    orig = F.BlazingFreeEnergy.solve

    def solve(self, *a, **k):
        o = orig(self, *a, **k)
        o["_pw"] = self.last["pw"].reshape(self.ri.W.shape).detach().cpu().numpy()
        o["_w"], o["_lq"] = self.ri.w.cpu().numpy(), self.ri.lq.cpu().numpy()
        return o
    F.BlazingFreeEnergy.solve = solve
    r = d / 150
    p = Params(d=d, P=round(3000 * r * r), N1=1500, N2=1500, chi=1.0, kappa=0.015 * r, eps=-0.3 * math.sqrt(6),
               sigA=math.sqrt((8 / 1500) / r), sigB=SIGB[d], sigw=math.sqrt(SIGW2), nmax=3)
    o = F.solve_fcn3(p, nmax=3, device=device)
    F.BlazingFreeEnergy.solve = orig
    kern = dict(J=np.array(o["J"]), H=np.array(o["H"]), T=float(o["T"]), Eq=float(o["Eq"]), M=float(o["M"]))
    return kern, o


def marg(nl, w0, lq, r2, bins_w, bins_r):
    p = np.exp(-nl) * np.gradient(w0)[:, None] * np.gradient(lq)[None, :]
    p /= p.sum()
    W0 = np.broadcast_to(w0[:, None], p.shape)
    pw, _ = np.histogram(W0.ravel(), bins_w, weights=p.ravel())
    pr, _ = np.histogram(r2.ravel(), bins_r, weights=p.ravel())
    return pw, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    d, sigB = a.d, SIGB[a.d]
    r_B = sigB ** 2                                                   # N1 = N2
    q0 = SIGW2 * (d - 1) / d
    w0g = np.linspace(0.0005, 1.4, 700)
    lqg = np.linspace(math.log(q0) - 2.5, math.log(q0) + 1.0, 700)
    bins_w, bins_r = np.linspace(0, 1.4, 71), np.linspace(0, 0.34, 69)
    mw, mr = 0.5 * (bins_w[1:] + bins_w[:-1]), 0.5 * (bins_r[1:] + bins_r[:-1])

    kt, ot = theory_kernels(d, a.device)
    rows = [("theory (n_max=3)", kt, None)]
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_ray_d{d}_*N1500x1500_*" / "checkpoint.pt"))):
        c = json.loads((Path(f).parent / "config.json").read_text())["args"]
        if "aborted" in f or c.get("tag") != "ray" or c["iters"] < 1000:
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        W = ck["ZW"].double().numpy() * math.sqrt(SIGW2 / d)
        B = ck["ZB"].double().numpy() * sigB / math.sqrt(c["N1"])
        sn = Path(f).parent / "snaps.npz"
        w0s = None
        if sn.exists():
            z = np.load(sn)
            post = z["iter"] >= c["burn"]
            w0s, qs = np.abs(z["w0"][post]).ravel(), z["q"][post].ravel()
        else:
            w0s, qs = np.abs(W[:, 0]), (W[:, 1:] ** 2).sum(1)
        rows.append((f"network, dataset {c.get('seed', 0)} {c['init']}", network_kernels(W, B, sigB), (w0s, qs)))

    print(f"d = {d}, n_max = 3 (S = {{1,3}}), r_B = {r_B:.3f}, beta = {d / (2 * SIGW2):.0f}\n")
    print(f"{'':32s} {'J11':>9s} {'J13':>10s} {'J33':>10s} {'H11':>9s} {'H13':>10s} {'H33':>10s} {'H11/G11':>8s} "
          f"{'H33/G33':>8s} {'V11':>9s} {'V13':>9s} {'V33':>9s} {'qbar':>7s} {'<q_h>':>7s} {'M':>7s} {'vperp':>8s}")
    res = []
    for name, k, samp in rows:
        nl, terms, r2, V, vperp = action(k, d, sigB, r_B, w0g, lqg)
        G = sigB ** 2 * k["J"]
        print(f"{name:32s} {k['J'][0,0]:9.5f} {k['J'][0,1]:10.6f} {k['J'][1,1]:10.7f} {k['H'][0,0]:9.5f} "
              f"{k['H'][0,1]:10.6f} {k['H'][1,1]:10.7f} {k['H'][0,0]/G[0,0]:8.3f} {k['H'][1,1]/G[1,1]:8.3f} "
              f"{V[0,0]:9.2f} {V[0,1]:9.1f} {V[1,1]:9.1f} {sigB**2*k['T']:7.4f} {k['Eq']:7.4f} {k['M']:7.1f} {vperp:8.2f}")
        res.append((name, nl, r2, samp))

    # (E5) evaluated with each network's own values vs the network's samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    cols = plt.cm.tab10(np.arange(10))
    print("\nbetween-well mass (0.35 < |w0| < 0.75) and outer mass (|w0| > 0.75):")
    for i, (name, nl, r2, samp) in enumerate(res):
        pw, pr = marg(nl, w0g, lqg, r2, bins_w, bins_r)
        lab = "(E5) with " + name
        sl = dict(color=cols[i], lw=2 if i == 0 else 1.2, ls="-" if i == 0 else "--")
        with np.errstate(divide="ignore"):
            axs[0].plot(mw, -np.log(pw / np.diff(bins_w)), label=lab, **sl)
            axs[1].plot(mr, -np.log(pr / np.diff(bins_r)), label=lab, **sl)
        mid, out = pw[(mw > 0.35) & (mw < 0.75)].sum(), pw[mw > 0.75].sum()
        line = f"  (E5) with {name:30s}: between {mid:.2e}, outer {out:.4f}"
        if samp is not None:
            w0s, qs = samp
            hw, _ = np.histogram(w0s, bins_w)
            hr, _ = np.histogram(w0s ** 2 / (1 + 2 * w0s ** 2 + 2 * qs), bins_r)
            okw, okr = hw > 0, hr > 0
            axs[0].plot(mw[okw], -np.log(hw[okw] / (hw.sum() * np.diff(bins_w)[okw])), "o", ms=3, color=cols[i])
            axs[1].plot(mr[okr], -np.log(hr[okr] / (hr.sum() * np.diff(bins_r)[okr])), "o", ms=3, color=cols[i])
            line += (f"   | network samples: between {np.mean((w0s > 0.35) & (w0s < 0.75)):.2e}, "
                     f"outer {np.mean(w0s > 0.75):.4f}")
        print(line)
    for ax, xl in zip(axs, ("|w$_0$|", "r²")):
        ax.set_xlabel(xl); ax.set_ylabel("−log p"); ax.set_ylim(-6, 30)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    axs[0].set_title(f"d = {d}: lines = (E5) with each set of measured values; points = that network's samples",
                     fontsize=9)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / f"readin_terms_d{d}.png"
    fig.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
