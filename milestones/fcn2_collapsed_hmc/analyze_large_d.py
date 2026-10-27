#!/usr/bin/env python3
"""
Large-d limit at a fixed theory point (launch_large_d.sh): HMC vs the read-in theory as d grows.

Per d (finished or still running; post-burn samples so far):
  HMC   <w0²>, fraction |w0| > 2.5σ_W, outer-well position, aligned |w_perp|²/prior, and the exact
        He_k(x0) coefficients of the posterior-mean predictor (from the weights, no test set):
        L1, L3, residual Δ5 = -f5, Δ7 = -f7; per-chain <w0²> for convergence
  theory  original (frozen perp norm, He1-He3) and corrected (per-neuron perp norm + He1..He7 of x0)
Prints a table and the relative gaps, and saves plots/large_d.png (gaps against d).

    python analyze_large_d.py            # d = 150 (pscan_d150_P3000) and every larged_d* group present
"""

import json
from math import factorial, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from collapsed_hmc_fcn2 import make_data
from diag_cavity_action import group_args, sample_files
from plot_pscan import outer_well, per_chain_means
from plot_readin_action import load_group, theory_curve
from readin_action_theory import reduced_params
from readin_action_theory_q import KernelQ, analyse_q
from runs import RUNS_ROOT

HERE = Path(__file__).resolve().parent
CACHE = HERE / "plots" / "cache"


def _dfact(n):
    return 1 if n <= 0 else n * _dfact(n - 2)


def a_k(s, k):
    return (2 / sqrt(np.pi)) * (-1) ** ((k - 1) // 2) * _dfact(k - 1) / sqrt(factorial(k)) * s ** k / (1 + 2 * s * s) ** (k / 2)


def groups():
    gs = ["pscan_d150_P3000"]
    found = set()
    for cp in RUNS_ROOT.glob("*/config.json"):
        g = json.loads(cp.read_text())["args"].get("group", "")
        if g.startswith("larged_d"):
            found.add(g)
    return gs + sorted(found, key=lambda g: int(g.split("_d")[1]))


@torch.no_grad()
def hmc_exact(g, a, n_snap=30, device="cuda:0"):
    dev = torch.device(device)
    X, Y, _, _ = make_data(a["d"], a["P"], a["eps"], a["he1"], a["seed"], dev)
    X, Y = X.double(), Y.double()
    sa2, hT = a["sa0"] / (a["N"] * a["chi"]), a["T"] / (2 * a["chi"])
    fk = []
    for f in sample_files(g, n_snap):
        W = torch.load(f)["W0"].to(dev).double().reshape(-1, a["d"])
        Phi = torch.erf(X @ W.T)
        M = Phi.T @ Phi / hT + torch.eye(a["N"], device=dev, dtype=torch.float64) / sa2
        A = torch.linalg.solve(M, Phi.T @ Y / hT)
        s = W.norm(dim=1)
        u0 = W[:, 0] / s
        fk.append([(A * a_k(s, k) * u0 ** k).sum().item() for k in (1, 3, 5, 7)])
        del Phi, M
    return np.mean(fk, 0), len(fk)


def theory(a):
    key = CACHE / f"larged_theory_d{a['d']}_P{a['P']}.json"
    if key.exists():
        return json.loads(key.read_text())
    d = a["d"]
    T = 1 + 2 * a["s0"] * (d - 1) / d
    wgrid = np.linspace(0, 2.4, 4801)
    nlp0, th0 = theory_curve(a, T, wgrid)
    c = (a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2)
    q = analyse_q(*c, a["eps"], ks=(1, 3, 5, 7))
    _, _, lam, _ = reduced_params(*c)
    J = KernelQ(d, a["s0"], ks=(1, 3, 5, 7)).stats(q["D"])[2]
    fq = J @ np.linalg.solve(J + lam * np.eye(4), np.array([1, a["eps"], 0, 0]))
    out = dict(orig=dict(w0_sq=th0["w0_sq"], frac=th0["frac_aligned"], well=float(wgrid[400 + np.argmin(nlp0[400:])]),
                         L3=th0["L3"], q_al=1.0),
               corr=dict(w0_sq=q["w0_sq"], frac=q["frac_aligned"], well=float(q["outer_well"]), L3=q["L3"],
                         q_al=q["q_aligned_over_prior"], D5=-fq[2], D7=-fq[3]))
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(out, default=float))
    return out


def main():
    rows = []
    for g in groups():
        loaded = load_group(g, live=True)
        if loaded is None:
            print(f"  {g}: no post-burn samples yet")
            continue
        W, a, _ = loaded
        d = a["d"]
        w0 = W[:, 0]
        thr = 2.5 * np.sqrt(a["s0"] / d)
        al = np.abs(w0) > thr
        q0 = a["s0"] * (d - 1) / d
        fk, n = hmc_exact(g, a)
        h = dict(w0_sq=float(np.mean(w0 ** 2)), frac=float(al.mean()), well=outer_well(w0),
                 q_al=float((W[al, 1:] ** 2).sum(1).mean() / q0), L1=fk[0], L3=fk[1] / a["eps"], D5=-fk[2], D7=-fk[3],
                 chains=per_chain_means(g))
        rows.append(dict(d=d, P=a["P"], n=n, h=h, t=theory(a)))
    print(f"\n{'d':>5s} {'P':>7s} {'snaps':>5s} | {'':>9s} {'<w0²>':>8s} {'frac':>6s} {'well':>5s} {'q_al':>5s} {'L3':>6s} {'Δ5':>7s} {'Δ7':>7s}")
    for r in rows:
        h, t = r["h"], r["t"]
        print(f"{r['d']:>5d} {r['P']:>7d} {r['n']:>5d} | {'HMC':>9s} {h['w0_sq']:>8.5f} {h['frac']:>6.3f} {h['well']:>5.2f} {h['q_al']:>5.2f} {h['L3']:>6.3f} {h['D5']:>+7.3f} {h['D7']:>+7.3f}"
              f"   chains: " + ", ".join(f"{k} {v['w0_sq']:.5f}" for k, v in h["chains"].items()))
        for name in ("orig", "corr"):
            x = t[name]
            print(f"{'':>19s} | {name:>9s} {x['w0_sq']:>8.5f} {x['frac']:>6.3f} {x['well']:>5.2f} {x['q_al']:>5.2f} {x['L3']:>6.3f}"
                  + (f" {x['D5']:>+7.3f} {x['D7']:>+7.3f}" if name == "corr" else ""))
    ds = np.array([r["d"] for r in rows])
    gaps = {f"{name} <w0²>": [r["t"][name]["w0_sq"] / r["h"]["w0_sq"] - 1 for r in rows] for name in ("orig", "corr")}
    gaps.update({f"{name} L3": [r["t"][name]["L3"] / r["h"]["L3"] - 1 for r in rows] for name in ("orig", "corr")})
    gaps.update({f"{name} well": [r["t"][name]["well"] - r["h"]["well"] for r in rows] for name in ("orig", "corr")})
    print("\nrelative gaps (theory/HMC - 1; well: theory - HMC):")
    for k, v in gaps.items():
        print(f"  {k:>12s}: " + "  ".join(f"d={d}: {x:+.3f}" for d, x in zip(ds, v)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, key, lab in zip(axes, ("<w0²>", "L3", "well"), (r"$\langle w_0^2\rangle$: theory/HMC $-1$",
                                                            r"$L_3$: theory/HMC $-1$", "outer well: theory $-$ HMC")):
        ax.plot(ds, gaps[f"orig {key}"], "o--", color="#d1495b", label="original theory (frozen perp, He1-He3)")
        ax.plot(ds, gaps[f"corr {key}"], "s-", color="#2a6fdb", label="corrected (relaxed perp + He1..He7)")
        ax.axhline(0, color="0.6", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(ds)
        ax.set_xticklabels([str(x) for x in ds])
        ax.minorticks_off()
        ax.set_xlabel("d  (P ∝ d², N = 1500, λ and Ĉ fixed)")
        ax.set_title(lab)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = HERE / "plots" / "large_d.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
