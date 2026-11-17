"""
FCN3 at N1 = inf vs the FCN2 theory: learnabilities, and the posterior distribution over neurons of the
per-neuron Hermite coefficients <phi_n | erf(.)> (n = 1, 3) of the layer that feeds the readout.

  FCN2      c_n(w0, q)   = <phi_n | erf(w.x)>,  weighted by the FCN2 single-neuron posterior p(w0, q)
            (readin_action_theory_q with the mapped prior: w0 ~ N(0, G11), q ~ Gamma(M/2, qbar))
  FCN3      B_n(h, q_h)  = <phi_n | erf(h(x))>, weighted by the FCN3 middle-layer posterior at N1 = inf
  FCN3 He1  the same, with the middle field G_S projected onto He1

All three are theory (no sampling). Distributions are mirrored under w -> -w (exact symmetry), so they are even.
Full distributions (no cut), S = {1,...,11}. Thesis point except d: P = 3000, N2 = 1600, chi = 10, kappa = 0.1, eps = 0.074.

    python plot_fcn2_limit.py                    # writes plots/fcn3_n1inf_vs_fcn2.png and .json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_fcn2_limit import KernelQMapped, fcn3_middle
from fcn3_matrix_solver import FCN3Solver, Numerics, Params
from readin_action_theory_q import solve_loop_q

COL = {"fcn2": "#2a78d6", "fcn3": "#eb6834", "fcn3_he1": "#1baf7a"}     # reference palette slots 1-3
LAB = {"fcn2": "FCN2 theory (mapped prior)", "fcn3": r"FCN3 theory, $N_1=\infty$",
       "fcn3_he1": r"FCN3 theory, $N_1=\infty$, field projected on He$_1$"}
INK, MUTED = "#1f1f1e", "#6b6a64"


@torch.no_grad()
def middle_coeffs(mid, t, gt, ds=0.02, smax=None, keep=2):
    """(B (M, keep), weight (M,)) over the middle-layer posterior at readout vector t, on a fine s lattice."""
    smax = smax or mid.nu.smax
    s_all = torch.arange(0, smax + ds / 2, ds, dtype=torch.float64, device=mid.dev)
    logw_qz = (mid.logwq[:, None] + mid.logwz[None, :]).reshape(-1)
    Bs, Ls = [], []
    per = max(1, (1 << 23) // (len(mid.qh) * len(mid.zeta) * mid.nu.nx))
    for i0 in range(0, len(s_all), per):
        s = s_all[i0:i0 + per]
        arg = (s[:, None, None] * mid.a + mid.A[None])[:, None] * mid.rq[None, :, None, None]
        B = (torch.erf(arg) @ mid.WPhi).reshape(len(s), -1, mid.p.nS)
        lw = logw_qz[None] + 0.5 * gt * (B @ t) ** 2 - 0.5 * (s * s)[:, None]
        lw = lw + torch.log(torch.where(s == 0, ds, 2 * ds))[:, None]
        Bs.append(B[..., :keep].reshape(-1, keep).cpu()); Ls.append(lw.reshape(-1).cpu())   # He1, He3 only
    B, L = torch.cat(Bs).numpy(), torch.cat(Ls).numpy()
    w = np.exp(L - L.max()); w = w / w.sum()
    return np.concatenate([B, -B]), np.concatenate([w, w]) / 2    # h -> -h


def fcn2_coeffs(p, G11, qbar, M):
    lam = p.kappa / (p.P * p.sigA ** 2)
    Chat = G11 * p.sigA ** 2 * p.chi * p.P ** 2 / (p.N2 * p.kappa ** 2)
    K = KernelQMapped(G11, M / 2, qbar, ks=p.ks)
    D, _ = solve_loop_q(Chat, lam, p.eps, K)
    pr, _, J, _ = K.stats(D)
    y = p.y()
    fbar = J @ np.linalg.solve(J + lam * np.eye(len(y)), y)
    C = K.C.reshape(p.nS, -1).T
    w = pr.ravel()
    return np.concatenate([C, -C]), np.concatenate([w, w]) / 2, fbar


def point(d, device):
    p = Params(d=d, N1=float("inf"))
    s = FCN3Solver(p, Numerics(), device=device, verbose=False)
    r = s.solve()
    G = s.last["L"] @ s.last["L"].T
    st = s.last["state"]
    out = dict(L={"fcn3": (r["L1"], r["L3"])}, dist={})
    out["dist"]["fcn3"] = middle_coeffs(s.mid, s.t, p.gt)
    Gp = torch.zeros_like(G); Gp[0, 0] = G[0, 0]
    for i in range(1, p.nS):
        Gp[i, i] = 1e-12 * G[0, 0]
    he1, _ = fcn3_middle(s, Gp)                                    # leaves s.mid set up on the projected field
    t_he1 = torch.tensor(p.y(), dtype=torch.float64, device=s.dev)
    t_he1 = (t_he1 - torch.tensor(he1["fbar"], dtype=torch.float64, device=s.dev)) / p.lam
    out["L"]["fcn3_he1"] = (he1["L1"], he1["L3"])
    out["dist"]["fcn3_he1"] = middle_coeffs(s.mid, t_he1, p.gt)
    C, w, fbar = fcn2_coeffs(p, float(G[0, 0]), st["qbar"], st["M"])
    out["L"]["fcn2"] = (fbar[0], fbar[1] / p.eps)
    out["dist"]["fcn2"] = (C, w)
    return out


def density(x, w, edges):
    h, _ = np.histogram(x, edges, weights=w)
    return h / np.diff(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="75,150,300,600,1200", help="d values for the learnability panels")
    ap.add_argument("--dist-d", default="150,1200", help="d values for the distribution rows")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "plots" / "fcn3_n1inf_vs_fcn2.png"))
    a = ap.parse_args()
    ds = [int(x) for x in a.ds.split(",")]
    dd = [int(x) for x in a.dist_d.split(",")]
    res = {}
    for d in sorted(set(ds) | set(dd)):
        res[d] = point(d, a.device)
        print(f"d={d}: " + "  ".join(f"{k} L1={v[0]:.5f} L3={v[1]:.5f}" for k, v in res[d]["L"].items()), flush=True)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1 + len(dd), 3, figsize=(16, 3.7 * (1 + len(dd))))
    order = ["fcn2", "fcn3", "fcn3_he1"]
    sty = {"fcn2": dict(ls="-", lw=2.4), "fcn3": dict(ls="--", lw=2), "fcn3_he1": dict(ls="none", marker="o", ms=8,
                                                                                     mfc="none", mew=2)}
    for j, name in enumerate(("L1", "L3")):
        A = ax[0, j]
        for k in order:
            A.plot(ds, [res[d]["L"][k][j] for d in ds], color=COL[k], label=LAB[k],
                   **({**sty[k], "marker": sty[k].get("marker", "o"), "ms": sty[k].get("ms", 5)}))
        A.set_xscale("log"); A.set_xticks(ds); A.set_xticklabels(ds); A.minorticks_off()
        A.set_xlabel(r"$d$"); A.set_ylabel(rf"$L_{name[1]}$")
        A.grid(True, color="#e6e5df", lw=0.6)
        if name == "L3":
            A.axhline(0, color=MUTED, lw=0.6)
        A.set_title(rf"learnability $L_{name[1]}$ vs $d$", color=INK)
    h, l = ax[0, 0].get_legend_handles_labels()
    ax[0, 2].axis("off")
    ax[0, 2].legend(h, l, frameon=False, fontsize=10, loc="center left")
    ax[0, 2].text(0.0, 0.12, "FCN2 and the He$_1$-projected FCN3 coincide (independent codes).\n"
                  "The full FCN3 middle field also carries the first layer's He$_3$:\n"
                  r"$\sqrt{G_{33}/G_{11}}\approx 1/d$, which moves $L_3$ and the outer wells.",
                  transform=ax[0, 2].transAxes, fontsize=9, color=MUTED, va="top")

    for r, d in enumerate(dd, start=1):
        B3, w3 = res[d]["dist"]["fcn3"][0][:, 1], res[d]["dist"]["fcn3"][1]
        m = np.abs(B3) < 0.05
        zoom = 6 * math.sqrt(np.sum(w3[m] * B3[m] ** 2) / np.sum(w3[m]))
        panels = [(0, 1.05, 421, "log", ""), (1, 0.33, 421, "log", ""), (1, zoom, 241, "linear", "  (centre, zoom)")]
        for j, (c, lim, nb, ysc, tag) in enumerate(panels):
            A = ax[r, j]
            n = 2 * c + 1
            edges = np.linspace(-lim, lim, nb)
            ctr = 0.5 * (edges[1:] + edges[:-1])
            ymax = 0
            for k in order:
                B, w = res[d]["dist"][k]
                y = density(B[:, c], w, edges)
                ymax = max(ymax, np.nanmax(y))
                y = np.where(y > 0, y, np.nan) if ysc == "log" else y
                if k == "fcn3_he1":
                    st = max(1, nb // 40)
                    A.plot(ctr[::st], y[::st], color=COL[k], ls="none", marker="o", ms=4, mfc="none", mew=1.2)
                else:
                    A.plot(ctr, y, color=COL[k], **sty[k])
            A.set_yscale(ysc)
            A.set_xlim(-lim, lim)
            if ysc == "log":
                A.set_ylim(1e-7, 5 * ymax)
            else:
                A.set_ylim(0, 1.08 * ymax)
                A.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            A.grid(True, color="#e6e5df", lw=0.6)
            A.set_xlabel(rf"per-neuron $\langle \mathrm{{He}}_{n} \,|\, \mathrm{{erf}}(\cdot)\rangle$")
            A.set_ylabel("posterior density over neurons" if j == 0 else "")
            A.set_title(rf"$d={d}$: He$_{n}$ coefficient{tag}", color=INK)
    fig.suptitle(r"FCN3 with an infinitely wide first layer vs FCN2 (theory)   —   "
                 r"$P=3000$, $N_2=1600$, $\chi=10$, $\kappa=0.1$, $\epsilon=0.074$, $S=\{1,\ldots,11\}$",
                 color=INK, fontsize=11)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    out.with_suffix(".json").write_text(json.dumps({d: v["L"] for d, v in res.items()}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
