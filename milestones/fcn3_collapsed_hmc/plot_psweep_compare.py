#!/usr/bin/env python3
"""Theory against the sampler on the P-sweep (d = 150, kappa = 0.015 fixed, P = 750 ... 48000), with the
infinite-width (NNGP) learnabilities. Same layout as plot_perp_compare.py, but here the theory changes with P.

Sampler: the onset_d150 runs (datasets 0-2, started from the prior; plot_onset.sampler_runs, which at P = 3000 also
takes the ray runs with the same parameters). Top: -log p of the read-in w0, of h|He1 and of h|He3, every neuron of
every network saved after burn-in, all datasets pooled; theory at the same P as a line of the same colour.
Bottom: L3 and L1 against alpha = ln P / ln d (y axis from 0); sampler = mean over datasets with its standard error.
The L3 panel runs to alpha = 4.6, where the NNGP L3 has saturated; beyond the sampled P the NNGP is eigenlearning on
the sphere (nngp_fcn3.eigen), since exact regression is out of reach there.
Theory: plots/cache/theory_onset_d150_v6.npz (plot_onset.py; not recomputed).
NNGP: exact GP regression with the twice-composed arcsin kernel on dataset 0's training set at each P
(nngp_fcn3.gpr), cached in plots/cache/nngp_psweep_d150.json.

    python plot_psweep_compare.py          -> plots/onset_checks/psweep_compare_d150.png
"""
import json
import math

import numpy as np

import plot_onset as po
from plot_perp_compare import C_NNGP, GRID, HERE, INK, MUTED, NNGP_CACHE as FIXED_CACHE

D, KAPPA = 150, 0.015
OUT = HERE / "plots" / "onset_checks" / "psweep_compare_d150.png"
CACHE = HERE / "plots" / "cache" / "nngp_psweep_d150.json"
A_MAX = 4.6          # L3 panel runs to here: the NNGP L3 has saturated near 1


def nngp(groups):
    """{P: (L1, L3)} of exact NNGP regression on dataset 0 of the onset run at each P (cached by run name)."""
    import nngp_fcn3
    c = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    old = json.loads(FIXED_CACHE.read_text()) if FIXED_CACHE.exists() else {}
    out = {}
    for P, rs in groups.items():
        r = next(r for r in rs if r["seed"] == 0 and r["cfg"].get("group") == "onset_d150")
        if r["run"] not in c:
            if r["run"] in old:
                c[r["run"]] = old[r["run"]]
            else:
                g = nngp_fcn3.gpr(r["path"])
                c[r["run"]] = [g["L1"], g["L3"]]
                print(f"NNGP P={P}: L1 {g['L1']:.4f} L3 {g['L3']:.4f}", flush=True)
            CACHE.write_text(json.dumps(c, indent=1))
        out[P] = c[r["run"]]
    return out


def main():
    plt = po.style()
    plt.rcParams.update({"font.size": 9.5})
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    groups = po.by_P(po.sampler_runs())
    po.set_outer([r for rs in groups.values() for r in rs], z)
    Ps = list(groups)
    S = {}
    for P, rs in groups.items():
        h, _ = po.sampler_hidden(rs)
        ds = po.per_dataset(rs)
        S[P] = dict(w0=np.concatenate([r["w0"] for r in rs]), h=h,
                    L1=po.combine([e["L1"] for e in ds]), L3=po.combine([e["L3"] for e in ds]))
    G = nngp(groups)
    TP = sorted(int(k[2:]) for k in z if k.startswith("W_"))
    for P in Ps:
        print(f"P={P:6d}: sampler L1 {S[P]['L1'][0]:.4f} L3 {S[P]['L3'][0]:.4f};  theory L1 {z[f'sc_{P}'][0]:.4f} "
              f"L3 {z[f'sc_{P}'][1]:.4f};  NNGP L1 {G[P][0]:.4f} L3 {G[P][1]:.4f}")
    lP = np.log(Ps)
    col = {P: plt.cm.viridis(0.9 * (math.log(P) - lP.min()) / (lP.max() - lP.min())) for P in Ps}

    def th_curve(i, P):
        if i == 0:
            w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
            k = dn > 1e-300
            w, y = w[k], -np.log(0.5 * dn[k])
            return np.concatenate([-w[::-1], w]), np.concatenate([y[::-1], y])
        n = (1, 3)[i - 1]
        return z[f"h{n}g_{P}"], -z[f"lp{n}_{P}"]

    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    lim_w = max(1.6, 1.08 * max(np.abs(s["w0"]).max() for s in S.values()))
    lim_h = [1.08 * max(np.abs(s["h"][:, j]).max() for s in S.values()) for j in (0, 1)]
    rows = [("w$_0$  (read-in neuron)", "read-in layer", lim_w, 81, 14),
            ("h | He$_1$  (hidden neuron)", "hidden layer, He$_1$", lim_h[0], 81, 10),
            ("h | He$_3$  (hidden neuron)", "hidden layer, He$_3$", lim_h[1], 81, 10)]
    for i, (xl, name, lim, nb, top) in enumerate(rows):
        ax = axs[0, i]
        lo = np.inf
        edges = np.linspace(-lim, lim, nb)
        for P in Ps:
            g, y = th_curve(i, P)
            k = np.abs(g) <= lim
            ax.plot(g[k], y[k], "-", color=col[P], lw=1.8, zorder=4)
            x = S[P]["w0"] if i == 0 else S[P]["h"][:, i - 1]
            c, _ = np.histogram(x, edges)
            ym = np.where(c > 0, -np.log(np.maximum(c, 1) / (c.sum() * np.diff(edges))), np.nan)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), ym, "o", ms=3.2, color=col[P], mec="white", mew=0.3, zorder=3)
            lo = min(lo, np.nanmin(ym), y[k].min())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lo - 0.5, lo + top)
        ax.set_xlabel(xl)
        ax.set_ylabel("−log p")
        ax.set_title(name, color=INK, loc="left", fontsize=11.5)
    al = np.array([math.log(P) / math.log(D) for P in Ps])
    import nngp_fcn3
    aE = np.linspace(1.25, A_MAX, 200)
    LE = [nngp_fcn3.eigen(D, D ** a, KAPPA)["L3"] for a in aE]
    print("NNGP eigenlearning L3:", ", ".join(f"α={a:.2f}: {L:.3f}" for a, L in zip(aE[::20], LE[::20])))
    alT = np.array([math.log(P) / math.log(D) for P in TP])
    for j, (name, key, idx) in enumerate((("L$_3$", "L3", 1), ("L$_1$", "L1", 0))):
        ax = axs[1, j]
        ax.plot(alT, [z[f"sc_{P}"][idx] for P in TP], "-", color=INK, lw=2, zorder=2)
        if key == "L3":       # NNGP continued to large alpha by eigenlearning on the sphere, until it saturates
            ax.plot(aE, LE, "-", color=C_NNGP, lw=1.6, zorder=2)
            ax.plot(al, [G[P][idx] for P in Ps], "s", color=C_NNGP, ms=6, mfc="white", mew=1.4, zorder=2)
        else:
            ax.plot(al, [G[P][idx] for P in Ps], "s--", color=C_NNGP, lw=1.6, ms=6, mfc="white", mew=1.4, zorder=2)
        for a, P in zip(al, Ps):
            ax.errorbar([a], [S[P][key][0]], yerr=[S[P][key][1]], fmt="o", ms=8, color=col[P], mec="white",
                        mew=0.5, capsize=0, zorder=3)
        if key == "L3":
            tk = np.arange(1.5, A_MAX + 1e-9, 0.5)
            ax.set_xticks(tk)
            ax.set_xticklabels([f"{a:.1f}\nP={D ** a:.1e}".replace("e+0", "e") for a in tk], fontsize=8)
            ax.set_xlim(1.2, A_MAX + 0.05)
        else:
            ax.set_xticks(al)
            ax.set_xticklabels([f"{a:.2f}\nP={P}" for a, P in zip(al, Ps)], fontsize=8)
        ax.set_xlabel("α = ln P / ln d")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name} against α", color=INK, loc="left", fontsize=11.5)
    hT = [plt.Line2D([], [], color=INK, lw=2),
          plt.Line2D([], [], color=C_NNGP, ls="", marker="s", mfc="white", mew=1.4, ms=6),
          plt.Line2D([], [], color=C_NNGP, lw=1.6)]
    axs[1, 0].legend(hT, ["theory", "NNGP, exact regression (dataset 0)", "NNGP, eigenlearning (sphere)"],
                     fontsize=9, loc="upper left", bbox_to_anchor=(0.14, 0.42))
    axs[1, 1].legend(hT[:2], ["theory", "NNGP, exact GP regression on dataset 0"], fontsize=9, loc="lower right")
    fig.suptitle("Theory against the sampler with d and κ fixed and P growing; colour = P "
                 "(line: theory at that P, points: sampler)", color=INK, x=0.01, ha="left", fontsize=12.5)
    ax = axs[1, 2]
    ax.axis("off")
    ax.text(0.02, 1.0,
            "Parameters (FCN3)\n"
            f"  P = {', '.join(str(P) for P in Ps)}\n"
            f"  κ = {KAPPA},  d = {D},  N$_1$ = N$_2$ = 1500,  χ = 1\n"
            "  ε = −0.3√6,  σ$_A^2$ = 8/1500,  σ$_B$ = 1.2554,  σ$_w^2$ = 0.5\n"
            "Sampler: datasets 0–2 (0–3 at P = 3000), started from\n"
            "  the prior, 1000 iterations, first 300 discarded;\n"
            "  top row: every neuron of every network saved after\n"
            "  burn-in, datasets pooled; bottom row: mean over\n"
            "  datasets with its standard error\n"
            "NNGP: GP regression with the arcsin kernel composed\n"
            "  twice (N$_1$, N$_2$ → ∞), on dataset 0's training set",
            transform=ax.transAxes, fontsize=10, color=INK, va="top", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRID))
    hP = [plt.Line2D([], [], color=col[P], marker="o", ms=5, lw=1.8) for P in Ps]
    ax.legend(hP, [f"P = {P}  (α = {math.log(P) / math.log(D):.2f})" for P in Ps], fontsize=9, loc="lower left",
              bbox_to_anchor=(0.0, 0.0), ncol=2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
