#!/usr/bin/env python3
"""
-log p(w0) of the read-in weights along the teacher direction: collapsed-HMC samples
against the self-consistent mean-field action (readin_action_theory.py).

Both sides are normalized densities on the full line, so there is no fitted offset;
the plot shifts both by the same constant (the theory minimum).

Mapping to the theory: σ_w² = s0, σ_A² = sa0, κ = T/2 (T the Langevin temperature).
The theory freezes the perp norm at its prior value, T_th = 1 + 2 s0 (d-1)/d; the dashed
curve uses the perp norm measured in the samples instead.

    python plot_readin_action.py --group b100_gibbs
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bookkeeping import RunRecorder
from readin_action_theory import Kernel, analyse, features, reduced_params, solve_loop
from runs import RUNS_ROOT


def load_group(group, live=False):
    """Post-burn W0 samples of the finished runs in a group, and their (shared) config args.

    live=True also takes runs still in progress (samples saved so far, L1/L3 from the
    post-burn rows of obs.csv); returns None if the group has no post-burn samples yet.
    """
    W, args, post = [], None, []
    for cfg_path in sorted(RUNS_ROOT.glob("*/config.json")):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        if cfg["args"].get("group") != group or not (live or (run / "summary.json").exists()):
            continue
        n = 0
        for f in sorted(glob.glob(str(run / "samples" / "iter_*.pt"))):
            try:
                W.append(torch.load(f)["W0"].double().cpu().numpy().reshape(-1, cfg["args"]["d"]))
                n += 1
            except Exception:                      # sample still being written
                pass
        if n == 0:
            continue
        args = args or cfg["args"]
        post.append(RunRecorder.load(run).obs_summary()[0])
        print(f"  {run.name}: {n} samples{'' if (run / 'summary.json').exists() else ' (running)'}")
    if not W:
        return None
    post = {k: np.nanmean([p[k][0] for p in post]) for k in ("L3", "L1_he1")}
    return np.concatenate(W), args, post


def theory_curve(a, T_th, w):
    """Theory -log p(w) (normalized on the full line), the solution D, and its analysis."""
    _, beta, lam, Chat = reduced_params(a["N"], a["d"], a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
    K = Kernel(T_th, beta)
    D, res = solve_loop(Chat, lam, a["eps"], K)
    _, p, logZ, _, _ = K.stats(D)
    c1, c3 = features(np.abs(w), T_th)
    nlp = beta * (w * w - (D[0] * c1 + D[1] * c3) ** 2) + logZ + np.log(2.0)
    thr = 2.5 * np.sqrt(a["s0"] / a["d"])
    out = analyse(D, lam, a["eps"], K)
    out.update(res=res, beta=beta, lam=lam, Chat=Chat, frac_aligned=p[K.w > thr].sum(),
               w0_sq=p @ K.w ** 2)
    return nlp, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="b100_gibbs")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live", action="store_true", help="include runs still in progress (samples so far)")
    args = ap.parse_args()

    loaded = load_group(args.group, live=args.live)
    if loaded is None:
        raise SystemExit(f"{args.group}: no post-burn samples yet")
    W, a, post = loaded
    w0 = W[:, 0]
    perp = (W[:, 1:] ** 2).sum(1).mean()
    T_prior = 1 + 2 * a["s0"] * (a["d"] - 1) / a["d"]
    T_meas = 1 + 2 * perp
    thr = 2.5 * np.sqrt(a["s0"] / a["d"])

    wmax = np.abs(w0).max() * 1.05
    edges = np.linspace(-wmax, wmax, args.bins + 1)
    cnt, _ = np.histogram(w0, edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dens = cnt / (cnt.sum() * np.diff(edges))
    ok = cnt > 0
    nlp_hist = -np.log(dens[ok])
    err = 1 / np.sqrt(cnt[ok])                         # Poisson error on -log p (ignores autocorrelation)

    w = np.linspace(-wmax, wmax, 2001)
    nlp_prior, th_prior = theory_curve(a, T_prior, w)
    nlp_meas, th_meas = theory_curve(a, T_meas, w)
    shift = nlp_prior.min()

    print(f"\n{len(w0)} w0 values from group {args.group}; |w_perp|² = {perp:.4f} (prior {a['s0'] * (a['d'] - 1) / a['d']:.4f})")
    print(f"{'':>24s} {'HMC':>9s} {'theory T_prior':>15s} {'theory T_meas':>14s}")
    print(f"{'T':>24s} {'':>9s} {T_prior:>15.4f} {T_meas:>14.4f}")
    print(f"{'D':>24s} {'':>9s} {np.array2string(th_prior['D'], precision=3):>15s} "
          f"{np.array2string(th_meas['D'], precision=3):>14s}")
    print(f"{'w0²':>24s} {np.mean(w0 ** 2):>9.4f} {th_prior['w0_sq']:>15.4f} {th_meas['w0_sq']:>14.4f}")
    print(f"{'frac |w0| > 2.5 σ_W':>24s} {np.mean(np.abs(w0) > thr):>9.4f} "
          f"{th_prior['frac_aligned']:>15.4f} {th_meas['frac_aligned']:>14.4f}")
    print(f"{'L3':>24s} {post['L3']:>9.4f} {th_prior['L3']:>15.4f} {th_meas['L3']:>14.4f}")
    print(f"{'L1':>24s} {post['L1_he1']:>9.4f} {th_prior['L1']:>15.4f} {th_meas['L1']:>14.4f}")
    for name, th in (("T_prior", th_prior), ("T_meas", th_meas)):
        print(f"  {name} basins: " + ", ".join(f"{b['basin']} weight={b['weight']:.3f}" for b in th["basins"])
              + f"   (Newton residual {th['res']:.1e})")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(ctr[ok], nlp_hist - shift, yerr=err, fmt="o", ms=4, color="#2a6fdb", ecolor="#2a6fdb",
                elinewidth=1, label=f"collapsed HMC + Gibbs ({len(w0) // a['N']} snapshots × N={a['N']})")
    ax.plot(w, nlp_prior - shift, color="#d1495b", lw=2.5,
            label=rf"theory, prior perp norm ($T={T_prior:.2f}$)")
    ax.plot(w, nlp_meas - shift, color="#d1495b", lw=2, ls="--",
            label=rf"theory, measured perp norm ($T={T_meas:.2f}$)")
    ax.set_xlim(-wmax, wmax)
    ax.set_ylim(-0.5, max(nlp_hist.max() - shift + 2, 6))
    ax.set_xlabel(r"$w_0$ (read-in weight along the teacher)", fontsize=14)
    ax.set_ylabel(r"$-\log p(w_0)$ + const", fontsize=14)
    ax.set_title(rf"$d={a['d']}$, $P={a['P']}$, $N={a['N']}$, $\kappa={a['T'] / 2:.4g}$, "
                 rf"$\epsilon={a['eps']:.3g}$, $\sigma_w^2={a['s0']}$", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, frameon=False, loc="upper center")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "plots" / f"readin_action_{args.group}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
