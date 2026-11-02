#!/usr/bin/env python3
"""
Scaling with d of the gap between the outer wells and the centre of the read-in action along the κ0 = 0.015 ray
(N = 1500, P = 3000 (d/150)², κ = 0.015 (d/150), σ_A² = (8/1500)(150/d), so κ/(Pσ_A²) and Ĉ are fixed).

    gap = -log p(w_out) + log p(0)        (raw, in nats; p the normalized density of w0, w_out the outer minimum)

and gap / β with β = d/(2σ_w²), the normalization of the action plot. If the raw gap tends to a constant,
gap/β ∝ 1/d; corrections are fitted as gap = A + B/d (no log d expected: the w0 and q_perp widths at both minima
scale alike, as 1/√d).
  HMC: minima from local quadratic fits to the histogram of |w0| (centre: even fit a + b w², |w0| < 0.05;
       outer: full quadratic within ±0.1 of the outer minimum), errors by bootstrap over snapshots.
  theory: per-neuron perp norm, He1..He11, solved densely along the ray up to d = 4800.

    python plot_gap_scaling.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_readin_action import load_group
from chains import chain_weights, mean_se
from readin_action_theory_q import analyse_q

GROUPS = ["larged_d75", "pscan_d150_P3000", "mix_d300", "w0q_d600", "w0q_d800", "w0q_d1200"]
D_THEORY = [75, 106, 150, 212, 300, 424, 600, 800, 1200, 1697, 2400, 3394, 4800]
KS = (1, 3, 5, 7, 9, 11)
S0, EPS = 0.5, -0.3 * np.sqrt(6)
EDGES = np.linspace(0, 1.9, 191)
NORM = plt.Normalize(np.log2(75), np.log2(1200))
CURVES, W0S, W0C = {}, {}, {}                 # theory (w, -log p) and HMC w0 samples per d, for the line-up figure


def ray(d):
    """Raw parameters of the κ0 = 0.015 ray at dimension d."""
    r = d / 150
    return dict(N=1500, d=d, s0=S0, sa0=(8 / 1500) / r, P=round(3000 * r * r), kappa=0.015 * r)


def theory_gap(d):
    p = ray(d)
    q = analyse_q(p["N"], d, S0, p["sa0"], p["P"], p["kappa"], EPS, ks=KS, nq=int(500 * max(1, np.sqrt(d / 600))))
    w, nlp = q["w"], q["nlp"]
    CURVES[d] = (np.concatenate([-w[::-1], w[1:]]), np.concatenate([nlp[::-1], nlp[1:]]))
    i = np.where(w > 0.4)[0]
    j = i[np.argmin(nlp[i])]
    a, b, c = np.polyfit(w[j - 3:j + 4], nlp[j - 3:j + 4], 2)    # refine the minimum between grid points
    return c - b * b / (4 * a) - nlp[0]


def hist_min(x):
    """Centre and outer minima of -log density of |w0| (the factor 1/2 to the full line cancels in the gap)."""
    cnt, _ = np.histogram(x, EDGES)
    ctr = 0.5 * (EDGES[1:] + EDGES[:-1])
    ok = cnt > 0
    h = -np.log(cnt[ok] / (cnt.sum() * np.diff(EDGES)[ok])); c = ctr[ok]; wt = np.sqrt(cnt[ok])
    m0 = c < 0.05
    A = np.stack([np.ones(m0.sum()), c[m0] ** 2], 1)
    centre = np.linalg.lstsq(A * wt[m0, None], h[m0] * wt[m0], rcond=None)[0][0]
    o = c > 0.5
    k = np.where(o)[0][np.argmin(h[o])]
    m1 = np.abs(c - c[k]) <= 0.1
    a, b, cc = np.polyfit(c[m1], h[m1], 2, w=wt[m1])
    return centre, cc - b * b / (4 * a)


def hmc_gap(group, n_boot=200, seed=0):
    """Gap per chain; value = mean over chains, error = standard error across chains combined in quadrature with
    the bootstrap error (over snapshots) of the pooled estimate."""
    W, a, _ = load_group(group, live=True)
    W0S[a["d"]] = W[:, 0]
    per_chain = []
    W0C[a["d"]] = [Wc[:, :, 0].ravel() for Wc in chain_weights(group).values()]
    for Wc in chain_weights(group).values():
        c0, c1 = hist_min(np.abs(Wc[:, :, 0]).ravel())
        per_chain.append(c1 - c0)
    gap, se = mean_se(per_chain)
    w0 = np.abs(W[:, 0].reshape(-1, a["N"]))                    # (snapshots, neurons)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, len(w0), len(w0))
        b0, b1 = hist_min(w0[s].ravel())
        boot.append(b1 - b0)
    print(f"  {group}: gap per chain {np.round(per_chain, 3)}")
    return a["d"], gap, np.hypot(se, np.std(boot)), len(w0)


def main():
    hm = [hmc_gap(g) for g in GROUPS]
    for d, g, e, n in hm:
        print(f"HMC    d={d:5d}: gap {g:.3f} ± {e:.3f} nats ({n} snapshots),  gap/β = {g / (d / (2 * S0)):.5f}")
    th = []
    for d in D_THEORY:
        th.append((d, theory_gap(d)))
        print(f"theory d={d:5d}: gap {th[-1][1]:.4f} nats,  gap/β = {th[-1][1] / (d / (2 * S0)):.6f}", flush=True)
    dt, gt = map(np.array, zip(*th))
    dh, gh, eh = (np.array([r[i] for r in hm]) for i in range(3))
    beta = lambda d: d / (2 * S0)

    # fits of the raw gap: A + B/d (weighted for HMC), and a free power law to gap/β for comparison
    Bt, At = np.polyfit(1 / dt, gt, 1)
    slope_t = np.polyfit(np.log(dt), np.log(gt / beta(dt)), 1)[0]
    print(f"\ntheory: gap = {At:.4f} + {Bt:.2f}/d;  free power law gap/β ∝ d^{slope_t:.4f}")
    # no A + B/d fit for HMC: its raw gap is not monotonic in d, and at d >= 300 the chains have not mixed

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    dd = np.geomspace(60, 6000, 200)
    ax1.loglog(dt, gt / beta(dt), "o-", color="0.35", mfc="white", lw=1.4, label="theory")
    ax1.loglog(dd, (At + Bt / dd) / beta(dd), "-", color="#2a6fdb", lw=1, alpha=0.7,
               label=rf"theory fit $({At:.2f} {Bt:+.1f}/d)\,/\,\beta$ (free fit: $d^{{{slope_t:.3f}}}$)")
    ax1.loglog(dd, At / beta(dd), ":", color="0.5", lw=1, label=r"pure $d^{-1}$")
    for d, g, e, _ in hm:
        ax1.errorbar(d, g / beta(d), yerr=e / beta(d), fmt="o", ms=8, color=plt.cm.viridis(0.9 * NORM(np.log2(d))),
                     mec="k", mew=0.5, zorder=3)
    ax1.plot([], [], "o", color="0.3", ms=8, label="HMC (error: spread across chains)")
    ax1.set_xlim(60, 6000)
    ax1.set_xlabel(r"$d$")
    ax1.set_ylabel(r"gap $/\, \beta$,  $\beta = d/2\sigma_w^2$")
    ax1.set_title(r"outer-well gap in the normalized action (log-log)")
    ax1.legend(frameon=False, fontsize=9)

    x = np.linspace(0, 1 / 60, 50)
    ax2.plot(1 / dt, gt, "o", color="0.35", mfc="white", label="theory")
    ax2.plot(x, At + Bt * x, "-", color="#2a6fdb", lw=1, alpha=0.7, label=rf"theory fit $A + B/d$: $A = {At:.3f}$, $B = {Bt:.1f}$")
    for d, g, e, _ in hm:
        ax2.errorbar(1 / d, g, yerr=e, fmt="o", ms=8, color=plt.cm.viridis(0.9 * NORM(np.log2(d))), mec="k", mew=0.5)
    ax2.plot([], [], "o", color="0.3", ms=8, label="HMC (error: spread across chains)")
    ax2.set_xlim(0, 1 / 60)
    ax2.set_xlabel(r"$1/d$")
    ax2.set_ylabel(r"raw gap $-\log p(w_{\rm out}) + \log p(0)$  (nats)")
    ax2.set_title(r"raw gap vs $1/d$ (extrapolates to $d\to\infty$ at $1/d = 0$)")
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle(r"$\kappa_0 = 0.015$ ray: $\kappa/(P\sigma_A^2) = 9.375\times10^{-4}$, $\hat C = 474.1$, $N = 1500$, $P \propto d^2$",
                 fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "gap_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    lineup(dt, gt, dh, gh, eh)


def lineup(dt, gt, dh, gh, eh):
    """The minima line up as d -> ∞: unshifted -log p(w0)/β for all d, and gap/β vs 1/d with a free intercept."""
    from plot_readin_action_sweep import hist_nlp
    from plot_large_d_actions import chain_bin_se
    beta = lambda d: d / (2 * S0)
    norm = plt.Normalize(np.log2(D_THEORY[0]), np.log2(D_THEORY[-1]))
    col = lambda d: plt.cm.viridis(0.9 * norm(np.log2(d)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

    for d in D_THEORY:
        w, nlp = CURVES[d]
        ax1.plot(w, nlp / beta(d), "-", color=col(d), lw=1.6, label=rf"$d={d}$")
    for d, w0 in W0S.items():
        ctr, h, err = hist_nlp(w0, np.linspace(-1.9, 1.9, 161))
        err = np.hypot(err, chain_bin_se(W0C[d], ctr))
        ax1.errorbar(ctr, h / beta(d), yerr=err / beta(d), fmt="o", ms=4, color=col(d), mec="k", mew=0.4,
                     elinewidth=0.6, zorder=3)
    ax1.axhline(0, color="0.4", lw=1, ls=":")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.016, 0.04)
    ax1.set_xlabel(r"$w_0$", fontsize=12)
    ax1.set_ylabel(r"$-\log p(w_0) \,/\, (d/2\sigma_w^2)$", fontsize=12)
    ax1.set_title(r"normalized action, no shift: both minima $\to 0$ as $d \to \infty$" + "\n"
                  r"(lines: theory, $d$ = 75 ... 4800; points: HMC, $d$ = 75 ... 1200, error = spread across chains)", fontsize=11)
    leg = ax1.legend(frameon=False, fontsize=9, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                     title=r"theory lines and HMC points coloured by $d$", title_fontsize=9)

    # gap/β vs 1/d: theory c + a x + b x²; c is the leading-order gap. HMC as points (not fitted: see main())
    xt, yt = 1 / dt, gt / beta(dt)
    pt, Ct = np.polyfit(xt, yt, 2, cov=True)
    xh, yh, sh = 1 / dh, gh / beta(dh), eh / beta(dh)
    ct, ect = pt[2], np.sqrt(Ct[2, 2])
    print(f"leading-order gap/β (intercept at 1/d = 0): theory {ct:.2e} ± {ect:.1e}")
    x = np.linspace(0, 1 / 65, 100)
    ax2.plot(xt, yt, "o", color="0.35", mfc="white", label="theory")
    ax2.plot(x, np.polyval(pt, x), "-", color="#2a6fdb", lw=1.2,
             label=rf"theory fit $c + a/d + b/d^2$: $c = ({ct * 1e6:.1f} \pm {ect * 1e6:.1f})\times10^{{-6}}$")
    for d, y, e in zip(dh, yh, sh):
        ax2.errorbar(1 / d, y, yerr=e, fmt="o", ms=8, color=col(d), mec="k", mew=0.5, zorder=3)
    ax2.plot([], [], "o", color="0.3", ms=8, label="HMC (error: spread across chains)")
    ax2.axhline(0, color="0.4", lw=1, ls=":")
    ax2.set_xlim(0, 1 / 65)
    ax2.set_ylim(-0.002, 0.058)
    ax2.set_xlabel(r"$1/d$", fontsize=12)
    ax2.set_ylabel(r"gap $/\, \beta$ = $[-\log p(w_{\rm out}) + \log p(0)]\,/\,\beta$", fontsize=11)
    ax2.set_title(r"gap $/\,\beta$ vs $1/d$: intercept = gap at $d = \infty$", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(r"$\kappa_0 = 0.015$ ray: $\kappa/(P\sigma_A^2) = 9.375\times10^{-4}$, $\hat C = 474.1$, $N = 1500$, $P \propto d^2$",
                 fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "gap_lineup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
