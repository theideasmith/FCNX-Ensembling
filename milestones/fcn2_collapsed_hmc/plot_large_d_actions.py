#!/usr/bin/env python3
"""
Read-in action along two large-d rays, one panel each, sharing the y axis:
  left:  the κ0 = 0.015 ray (launch_large_d.sh; d=150 is pscan_d150_P3000), N = 1500, P ∝ d², κ/(Pσ_A²) and Ĉ fixed;
  right: the κ0 = 0.1 ray of the grokking paper (launch_ray_paper.sh; d=150 is grok_k0p1), the paper's α/β scaling.
Colour is d (viridis, one log-d scale shared by both panels). HMC histogram (points) and the theory (per-neuron perp
norm, He1..He11 of x0, where the tower has converged; solid). Every curve is -log p(w0) / beta, beta = d / (2 sigma_w²),
with p the normalized density on the full line (histogram: counts / (total * bin width)); no shift is added. Each panel has a
learnability inset (HMC filled, theory hatched) and a legend below with the values: HMC learnabilities are the exact
population He_k coefficients of the posterior-mean predictor (chains.chain_fbar).
HMC error bars are the standard error across chains (chains.py): per histogram bin combined in quadrature with the
Poisson error, and on L1, L3. At d >= 300 the chains have not mixed, so these are lower bounds there.

    python plot_large_d_actions.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chains import chain_fbar, chain_weights, mean_se
from plot_readin_action import load_group
from plot_readin_action_sweep import hist_nlp
from readin_action_theory import reduced_params
from readin_action_theory_q import analyse_q

RAYS = [(r"$\kappa_0 = 0.015$ ray", ["larged_d75", "pscan_d150_P3000", "mix_d300", "w0q_d600", "w0q_d800",
                                     "w0q_d1200"]),
        (r"$\kappa_0 = 0.1$ ray (grokking paper)", ["grok_k0p1", "raypaper_b2", "raypaper_b8"])]
KS = (1, 3, 5, 7, 9, 11)
NORM = plt.Normalize(np.log2(75), np.log2(1200))
EDGES = np.linspace(-1.9, 1.9, 161)


def colour(d):
    return plt.cm.viridis(0.9 * NORM(np.log2(d)))


def load_ray(groups):
    rows = []
    for g in groups:
        loaded = load_group(g, live=True)
        if loaded is None:
            print(f"  {g}: no samples, skipped")
            continue
        W, a, _ = loaded
        d = a["d"]
        q = analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], a["chi"], ks=KS,
                      nq=int(500 * max(1, np.sqrt(d / 600))))
        wq = np.concatenate([-q["w"][::-1], q["w"][1:]])
        nq = np.concatenate([q["nlp"][::-1], q["nlp"][1:]])
        m = np.abs(wq) <= 1.9
        beta = d / (2 * a["s0"])
        fb = chain_fbar(g)
        L1, eL1 = mean_se([f[0] for f in fb.values()])
        L3, eL3 = mean_se([f[1] / a["eps"] for f in fb.values()])
        w0c = [Wc[:, :, 0].ravel() for Wc in chain_weights(g).values()]
        _, _, lam, Chat = reduced_params(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["chi"])
        rows.append(dict(d=d, P=a["P"], N=a["N"], beta=beta, w0=W[:, 0], w0c=w0c, wq=wq[m], th=nq[m] / beta,
                         L1=L1, eL1=eL1, L3=L3, eL3=eL3, L1th=q["L1"], L3th=q["L3"], lam=lam, Chat=Chat))
        print(f"{g}: d={d}, {len(W) // a['N']} snapshots, {len(fb)} chains; L1 HMC {L1:.3f}±{eL1:.3f} theory "
              f"{q['L1']:.3f}; L3 HMC {L3:.3f}±{eL3:.3f} theory {q['L3']:.3f}")
    return rows


def chain_bin_se(w0c, ctr):
    """Standard error across chains of -log density in the bins centred at ctr (0 where a chain has no counts)."""
    if len(w0c) < 2:
        return np.zeros_like(ctr)
    hs = []
    for w in w0c:
        cnt, _ = np.histogram(w, EDGES)
        c = 0.5 * (EDGES[1:] + EDGES[:-1])
        with np.errstate(divide="ignore"):
            hs.append(np.interp(ctr, c, -np.log(cnt / (cnt.sum() * np.diff(EDGES)))))
    hs = np.array(hs)
    se = hs.std(0, ddof=1) / np.sqrt(len(hs))
    return np.where(np.isfinite(se), se, 0.0)


def panel(ax, rows, name):
    for r in rows:
        col = colour(r["d"])
        ctr, h, err = hist_nlp(r["w0"], EDGES)
        err = np.hypot(err, chain_bin_se(r["w0c"], ctr))
        ax.errorbar(ctr, h / r["beta"], yerr=err / r["beta"], fmt="o", ms=3, color=col, elinewidth=0.6,
                    mec="white", mew=0.4, zorder=3)
        ax.plot(r["wq"], r["th"], "-", color=col, lw=2.2, zorder=2)
    hs = [ax.plot([], [], "s", color=colour(r["d"]),
                  label=rf"$d={r['d']}$, $P={r['P']}$, $N={r['N']}$:   $L_1$ {r['L1']:.3f}$\pm${r['eL1']:.3f} / {r['L1th']:.3f},   "
                        rf"$L_3$ {r['L3']:.3f}$\pm${r['eL3']:.3f} / {r['L3th']:.3f}")[0] for r in rows]
    leg = ax.legend(handles=hs, title=r"learnability, HMC / theory", frameon=False, fontsize=9.5, title_fontsize=9.5,
                    loc="upper left", bbox_to_anchor=(0.0, -0.11), alignment="left")
    # learnability inset (bottom right, beyond the right well): HMC filled, theory hatched, one colour per d
    ins = ax.inset_axes([0.725, 0.06, 0.26, 0.42])
    bw = 0.8 / (2 * len(rows))
    for j, (key, keyth) in enumerate((("L1", "L1th"), ("L3", "L3th"))):
        for i, r in enumerate(rows):
            col = colour(r["d"])
            x0 = j + (2 * i - len(rows)) * bw + bw / 2
            ins.bar(x0, r[key], bw, color=col, edgecolor=col, yerr=r["e" + key], ecolor="k",
                    error_kw=dict(lw=0.8, capsize=1.5))
            ins.bar(x0 + bw, r[keyth], bw, facecolor="white", edgecolor=col, hatch="////", lw=0.8)
    ins.set_xticks([0, 1])
    ins.set_xticklabels([r"$L_1$", r"$L_3$"], fontsize=10)
    ins.set_ylim(0, 1.15)
    ins.tick_params(labelsize=8)
    ins.set_title("learnability", fontsize=9)
    ins.bar(np.nan, 0, color="0.4", label="HMC")
    ins.bar(np.nan, 0, facecolor="white", edgecolor="0.4", hatch="////", label="theory")
    ins.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=1.2)
    ax.set_xlim(-1.5, 2.7)                      # room on the right for the inset
    ax.set_xlabel(r"$w_0$", fontsize=13)
    r0 = rows[0]
    ax.set_title(f"{name}:  " + rf"$\kappa/(P\sigma_A^2)={r0['lam']:.3g}$, $\hat C={r0['Chat']:.1f}$", fontsize=12)
    return leg


def main():
    loaded = [(name, load_ray(groups)) for name, groups in RAYS]
    fig, axes = plt.subplots(1, len(loaded), figsize=(10 * len(loaded), 8), sharey=True)
    legs = [panel(ax, rows, name) for ax, (name, rows) in zip(axes, loaded)]
    lo = min(r["th"].min() for _, rows in loaded for r in rows)
    axes[0].set_ylim(lo - 0.01, lo + 0.1)
    axes[0].set_ylabel(r"$-\log p(w_0) \,/\, (d/2\sigma_w^2)$", fontsize=13)
    hk = [axes[-1].plot([], [], "o", color="0.3", label="HMC")[0],
          axes[-1].plot([], [], "-", color="0.3", lw=2.2, label=f"theory (per-neuron perp norm, He1..He{KS[-1]})")[0]]
    keyleg = fig.legend(handles=hk, frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(0.99, 0.08))
    fig.suptitle(r"large-$d$ limit along two rays ($\epsilon=-0.3\sqrt{6}$, $\sigma_w^2=0.5$; theory point fixed along each ray)",
                 fontsize=13)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "large_d_actions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=legs + [keyleg])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
