#!/usr/bin/env python3
"""
HMC vs theory along an invariant ray, as a function of d: do the finite-d fluctuations drop out?

Per group (one ray point), from post-burn snapshots and a fresh 50k-point test set:
    L1, L3        of the posterior-mean predictor per snapshot (exact A|W mean), averaged
    sd_readout    sqrt(E_W Var(L3|W))       readout fluctuations (exact Gaussian A|W)
    sd_features   sd_W of E[L3|W]           read-in weight fluctuations (dropped by mean field)
    sd_theory     sqrt((κ/P) [J (J+λI)⁻¹]_33)/|ε|   the theory's readout-only spread
    w0² ratio     HMC / theory
    perp ratio    ⟨|w_perp|²⟩ / prior s0 (d-1)/d    (theory freezes it at 1)
    D3 ratio      |D3| fitted to the exact cavity drive / theory |D3|   (diag_cavity_action)
The theory point (λ, Ĉ) is the same along the ray, so these should go to the theory's values if
the finite-d corrections vanish.

    python plot_ray_trend.py --groups ray_k0p015_b1_float32 ray_k0p015_b1p41421356_float32 ...
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from collapsed_hmc_fcn2 import _he3
from diag_cavity_action import cavity_analysis, group_args, sample_files
from readin_action_theory import Kernel, features, reduced_params, solve_loop, analyse

CACHE = Path(__file__).resolve().parent / "plots" / "cache"


@torch.no_grad()
def hmc_stats(group, a, n_snap, n_test, device):
    from collapsed_hmc_fcn2 import make_data
    dev = torch.device(device)
    X, Y, _, _ = (t.double() for t in make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev))
    g = torch.Generator(device=dev).manual_seed(987654)
    Xt = torch.randn(n_test, a["d"], device=dev, dtype=torch.float64, generator=g)
    he1, he3 = Xt[:, 0], _he3(Xt[:, 0])
    he1 = he1 / he1.norm() * np.sqrt(n_test)          # exact unit norm on this test set
    he3 = he3 - (he3 @ he1) / n_test * he1
    he3 = he3 / he3.norm() * np.sqrt(n_test)
    sw2, sa2, hT = a["s0"] / a["d"], a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    L1s, L3s, v3, w0sq, perp = [], [], [], [], []
    for f in sample_files(group, n_snap):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        L = torch.linalg.cholesky(M)
        Abar = torch.cholesky_solve((Phi.T @ Y / hT)[:, None], L)[:, 0]
        Pt = torch.erf(Xt @ W.T)
        g1, g3 = Pt.T @ he1 / n_test, Pt.T @ he3 / n_test / a["eps"]       # L1 = g1·A, L3 = g3·A
        L1s.append((g1 @ Abar).item())
        L3s.append((g3 @ Abar).item())
        v3.append((g3 @ torch.cholesky_solve(g3[:, None], L)[:, 0]).item())
        w0sq.append((W[:, 0] ** 2).mean().item())
        perp.append((W[:, 1:] ** 2).sum(1).mean().item())
    return dict(L1=np.mean(L1s), L3=np.mean(L3s), sd_readout=np.sqrt(np.mean(v3)), sd_features=np.std(L3s),
                w0_sq=np.mean(w0sq), perp=np.mean(perp) / (a["s0"] * (a["d"] - 1) / a["d"]), n=len(L1s))


def theory_stats(a):
    T, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
    K = Kernel(T, beta)
    D, _ = solve_loop(Chat, lam, a["eps"], K)
    _, p, _, J, _ = K.stats(D)
    o = analyse(D, lam, a["eps"], K)
    Sig = (a["T"] / 2) / a["P"] * J @ np.linalg.inv(J + lam * np.eye(2))
    return dict(L1=o["L1"], L3=o["L3"], w0_sq=p @ K.w ** 2, sd=np.sqrt(Sig[1, 1]) / abs(a["eps"]),
                D=D, T=T, beta=beta, lam=lam, Chat=Chat)


def fit_D3(group, a, th, n_snap, device):
    cav = cavity_analysis(group, n_snap, wmax=1.6, nw=321, device=device, verbose=False)
    w = cav["w"]
    c1, c3 = features(np.abs(w), th["T"])
    pos = w > 0.02
    Dfit, *_ = np.linalg.lstsq(np.stack([c1[pos], c3[pos]], 1), np.sqrt(cav["drive_exact"][pos] / th["beta"]),
                               rcond=None)
    return Dfit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--snapshots", type=int, default=60)
    ap.add_argument("--cavity-snapshots", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=50000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for grp in args.groups:
        a = group_args(grp)
        if a is None or not sample_files(grp, 1):
            print(f"  {grp}: no samples yet")
            continue
        key = CACHE / f"raytrend_{grp}_{len(sample_files(grp, 10 ** 6))}_{args.snapshots}_{args.cavity_snapshots}.json"
        if key.exists():
            r = json.loads(key.read_text())
        else:
            th = theory_stats(a)
            h = hmc_stats(grp, a, args.snapshots, args.n_test, args.device)
            Dfit = fit_D3(grp, a, th, args.cavity_snapshots, args.device)
            r = dict(group=grp, d=a["d"], N=a["N"], P=a["P"], kappa=a["T"] / 2, hmc=h,
                     th={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in th.items()},
                     Dfit=Dfit.tolist())
            CACHE.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps(r, default=float))
        rows.append(r)
    rows.sort(key=lambda r: r["d"])

    hdr = (f"{'d':>4s} {'N':>5s} {'P':>6s} {'snaps':>5s} | {'L1':>6s} {'th':>6s} | {'L3':>6s} {'th':>6s} | "
           f"{'sd_ro':>6s} {'sd_ft':>6s} {'sd_th':>6s} | {'w0² h/th':>8s} {'perp/prior':>10s} {'|D3| fit/th':>11s}")
    print(hdr)
    for r in rows:
        h, th = r["hmc"], r["th"]
        print(f"{r['d']:>4d} {r['N']:>5d} {r['P']:>6d} {h['n']:>5d} | {h['L1']:>6.4f} {th['L1']:>6.4f} | "
              f"{h['L3']:>6.4f} {th['L3']:>6.4f} | {h['sd_readout']:>6.4f} {h['sd_features']:>6.4f} {th['sd']:>6.4f} | "
              f"{h['w0_sq'] / th['w0_sq']:>8.3f} {h['perp']:>10.4f} {abs(r['Dfit'][1] / th['D'][1]):>11.3f}")

    d = np.array([r["d"] for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    ax = axes[0]
    ax.plot(d, [r["hmc"]["L3"] for r in rows], "o", color="#2a6fdb", ms=7, label="HMC")
    ax.plot(d, [r["th"]["L3"] for r in rows], "-", color="#d1495b", lw=2, label="theory")
    ax.set_title(r"$L_3$ (50k fresh test points)")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(d, [r["hmc"]["sd_features"] for r in rows], "o-", color="#2a6fdb", ms=7, label="HMC: read-in weight part")
    ax.plot(d, [r["hmc"]["sd_readout"] for r in rows], "s-", color="#6a9f3a", ms=6, label="HMC: readout part")
    ax.plot(d, [r["th"]["sd"] for r in rows], "-", color="#d1495b", lw=2, label="theory (readout only)")
    ax.set_title(r"posterior spread of $L_3$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    ax.plot(d, [r["hmc"]["w0_sq"] / r["th"]["w0_sq"] for r in rows], "o-", color="#2a6fdb", ms=7,
            label=r"$\langle w_0^2\rangle$ HMC / theory")
    ax.plot(d, [abs(r["Dfit"][1] / r["th"]["D"][1]) for r in rows], "s-", color="#e8a33d", ms=6,
            label=r"$|D_3|$ fit to HMC / theory")
    ax.plot(d, [r["hmc"]["perp"] for r in rows], "^-", color="0.4", ms=6, label=r"$|w_\perp|^2$ / prior")
    ax.axhline(1, color="0.75", lw=1)
    ax.set_title("ratios to theory (1 = agreement)")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[3]
    ax.plot(d, [r["hmc"]["L1"] for r in rows], "o", color="#2a6fdb", ms=7, label="HMC")
    ax.plot(d, [r["th"]["L1"] for r in rows], "-", color="#d1495b", lw=2, label="theory")
    ax.set_title(r"$L_1$")
    for ax in axes:
        ax.set_xlabel(r"$d$")
        ax.set_xscale("log")
        ax.set_xticks(d)
        ax.set_xticklabels([str(x) for x in d])
        ax.minorticks_off()
    a0 = group_args(rows[0]["group"])
    fig.suptitle(rf"invariant ray $\kappa_0={rows[0]['kappa']:g}$: $\lambda={rows[0]['th']['lam']:.4g}$, "
                 rf"$\hat C={rows[0]['th']['Chat']:.1f}$, $\epsilon={a0['eps']:.3g}$, $\sigma_w^2={a0['s0']:g}$", fontsize=13)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / "ray_trend_k0p015.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
