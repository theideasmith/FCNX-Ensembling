#!/usr/bin/env python3
"""Coupling between the read-in and the hidden layer as the width grows, at P = 750, kappa = 0.015, d = 150
(queue_wide.sh: N1 = N2 = 3000 and 6000 with chi = N/1500, same theory; onset runs r10-r12 are N = 1500).

For every saved network after burn-in, with s = sign(w0) and "outer" = read-in neurons with |w0| beyond the theory's
barrier top (plot_onset.barrier_cut):
  rms_out_i   rms of hidden neuron i's weights onto the outer read-in neurons (prior units, ZB ~ N(0, 1) in the prior)
  corr        correlation over hidden neurons of |h_i . He3| with rms_out_i
  cos_out     cosine of an outer neuron's sign-aligned outgoing column with the summed columns of the other outer ones
  cos_in      the same for inner read-in neurons against the summed outer columns
  cos_pair    mean cosine between the sign-aligned outgoing columns of two outer neurons
Comparison case (what the theory assumes): hidden neurons independent, each one's weights drawn from the prior given
its own projections h_k on He_k(x0), k = 1, 3, ..., 11 (ZB_i = h_i (C C^T)^-1 C / sb + xi_i (1 - Pi), Pi the projector
on the rows of C). "Leftover": the same statistics on ZB (1 - Pi), the part of each neuron's weights its own h does not
fix (compare its cos_pair with the comparison case's: removing the h part leaves the outer columns slightly
anti-aligned in both, since their sign pattern lies almost in the rows of C).

    python coupling_width.py            ->  plots/cache/coupling_width_P750.json, plots/width/coupling_width_P750.png
    python coupling_width.py --P 48000  ->  plots/cache/coupling_width_P48000.json (N = 1500 only; read by
                                            plot_ray_readin_fcn2_vs_fcn3.panel_coupling)
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_matrix_solver import readin_features

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--P", type=int, default=750)
P = ap.parse_args().P
KS = (1, 3, 5, 7, 9, 11)
CACHE = HERE / "plots" / "cache" / f"coupling_width_P{P}.json"


def runs():
    out = {1500: sorted(glob.glob(str(HERE / "runs" / f"*_onset_d150_P{P}_N1500x1500_prior_r1*")))}
    for N in (3000, 6000):
        out[N] = sorted(glob.glob(str(HERE / "runs" / f"*_wide_d150_P{P}_N{N}x{N}_prior_r1*")))
    return {N: [r for r in rs if (Path(r) / "summary.json").exists()] for N, rs in out.items()}


def stats(Z, s, out, h3):
    """Z (N2, N1) outgoing weights in prior units; s sign(w0); out mask of outer read-in neurons; h3 (N2,)."""
    rms = Z[:, out].pow(2).mean(1).sqrt()
    cols = Z * s[None, :]
    pat = cols[:, out].sum(1)
    cs = torch.nn.functional.cosine_similarity
    io = torch.where(out)[0]
    ii = torch.where(~out)[0][:600]
    cos_o = torch.stack([cs(cols[:, j], pat - cols[:, j], dim=0) for j in io])
    cos_i = torch.stack([cs(cols[:, j], pat, dim=0) for j in ii])
    U = torch.nn.functional.normalize(cols[:, out], dim=0)
    M = U.T @ U
    n = M.shape[0]
    pair = float((M.sum() - M.diagonal().sum()) / (n * (n - 1)))
    return dict(rms_out=float(rms.pow(2).mean().sqrt()), corr=float(np.corrcoef(h3.abs(), rms)[0, 1]),
                cos_out=float(cos_o.mean()), cos_in=float(cos_i.mean()), cos_pair=pair), rms


@torch.no_grad()
def one_net(f, cut, gen):
    nt = torch.load(f, map_location="cpu", weights_only=False)
    ZW, Z = nt["ZW"].double(), nt["ZB"].double()
    w0 = nt["sw"] * ZW[:, 0]
    q = nt["sw"] ** 2 * (ZW[:, 1:] ** 2).sum(1)
    C = readin_features(w0, q, KS)                              # (len(KS), N1)
    H = nt["sb"] * Z @ C.T                                      # (N2, len(KS)): h_i on He_k(x0)
    G = torch.linalg.pinv(C @ C.T, rtol=1e-12)
    Pi = C.T @ G @ C                                            # projector on the rows of C
    s, out = torch.sign(w0), w0.abs() > cut
    Xi = torch.randn(Z.shape, generator=gen, dtype=torch.float64)
    Znull = (H / nt["sb"]) @ G @ C + Xi - Xi @ Pi
    R = Z - Z @ Pi
    h3 = H[:, 1]
    res = {"n_out": int(out.sum())}
    for name, M in (("sampler", Z), ("comparison", Znull), ("leftover", R), ("leftover_comparison", Znull - Znull @ Pi)):
        st, rms = stats(M, s, out, h3)
        res[name] = st
        if name in ("sampler", "comparison"):
            res[f"scatter_{name}"] = (h3.abs().numpy().tolist(), rms.numpy().tolist())
    res["h3"] = h3.numpy().tolist()
    res["h1"] = H[:, 0].numpy().tolist()
    res["w0"] = w0.numpy().tolist()
    return res


def collect():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    cut = po.barrier_cut(z, P)
    gen = torch.Generator().manual_seed(0)
    for N, rs in runs().items():
        for r in rs:
            key = Path(r).name
            if key in cache:
                continue
            nets = sorted((Path(r) / "nets").glob("net_*.pt"))
            per = [one_net(f, cut, gen) for f in nets]
            last = per[-1]
            agg = {k: {m: float(np.mean([p[k][m] for p in per])) for m in per[0][k]}
                   for k in ("sampler", "comparison", "leftover", "leftover_comparison")}
            agg.update(N=N, n_nets=len(per), n_out=float(np.mean([p["n_out"] for p in per])),
                       h3=sum((p["h3"] for p in per), []), h1=sum((p["h1"] for p in per), []), w0=sum((p["w0"] for p in per), []),
                       scatter_sampler=last["scatter_sampler"], scatter_comparison=last["scatter_comparison"],
                       L3=json.loads((Path(r) / "summary.json").read_text())["posterior"]["L3"][0],
                       L1=json.loads((Path(r) / "summary.json").read_text())["posterior"]["L1"][0])
            cache[key] = agg
            print(f"N={N} {key[-3:]}: {len(per)} nets, n_out {agg['n_out']:.1f}, L3 {agg['L3']:.3f}  "
                  + "  ".join(f"{k}: " + ", ".join(f"{m} {v:.3f}" for m, v in agg[k].items())
                              for k in ("sampler", "comparison", "leftover")), flush=True)
            CACHE.write_text(json.dumps(cache))
    return cache


def hist_action(ax, x, edges, col, label):
    mid = 0.5 * (edges[1:] + edges[:-1])
    c, _ = np.histogram(x, edges)
    k = c > 0
    ax.plot(mid[k], -np.log(c[k] / (c.sum() * np.diff(edges)[k])), "o", ms=3.2, color=col, mec="white", mew=0.3,
            zorder=3, label=label)


def fig(cache):
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    cut = po.barrier_cut(z, P)
    Ns = sorted({a["N"] for a in cache.values()})
    by = {N: [a for k, a in sorted(cache.items()) if a["N"] == N] for N in Ns}
    plt = po.style()
    cols = {N: plt.get_cmap("Oranges")(0.45 + 0.5 * i / max(1, len(Ns) - 1)) for i, N in enumerate(Ns)}
    f, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    # row 1: read-in w0, h|He1, h|He3 at each N against the theory (the same at every N)
    w, dn = z[f"w0_{P}"], z[f"dens_{P}"]
    k = (dn > 1e-300) & (w <= 1.6)
    axs[0, 0].plot(np.r_[-w[k][::-1], w[k]], -np.log(0.5 * np.r_[dn[k][::-1], dn[k]]), color=po.INK, lw=1.8,
                   label="theory (the same at every N)")
    for j, (g, lp) in enumerate(((f"h1g_{P}", f"lp1_{P}"), (f"h3g_{P}", f"lp3_{P}"))):
        yt = -z[lp]
        kk = yt <= yt.min() + 12
        axs[0, 1 + j].plot(z[g][kk], yt[kk], color=po.INK, lw=1.8, label="theory")
    for N in Ns:
        rs = by[N]
        lab = (f"N = {N}: L$_3$ {np.mean([a['L3'] for a in rs]):.3f}, outer read-in neurons "
               f"{np.mean([a['n_out'] for a in rs]) / N * 100:.2f} %  ({len(rs)} run{'s' if len(rs) > 1 else ''})")
        hist_action(axs[0, 0], np.concatenate([a["w0"] for a in rs]), np.linspace(-1.6, 1.6, 81), cols[N], lab)
        hist_action(axs[0, 1], np.concatenate([a["h1"] for a in rs]), np.linspace(-1.1, 1.1, 61), cols[N], f"N = {N}")
        hist_action(axs[0, 2], np.concatenate([a["h3"] for a in rs]), np.linspace(-0.16, 0.16, 61), cols[N],
                    f"N = {N}")
    th_out = po.outer_theory(z, P, cut) * 100
    axs[0, 0].axvline(cut, color=po.MUTED, lw=0.8, ls=":")
    axs[0, 0].axvline(-cut, color=po.MUTED, lw=0.8, ls=":")
    axs[0, 0].set_title(f"read-in layer (dotted: theory's barrier top; theory outer share {th_out:.2f} %)",
                        color=po.INK, loc="left", fontsize=11)
    axs[0, 0].set_xlabel("w$_0$")
    axs[0, 0].set_ylabel("−log p")
    axs[0, 0].set_ylim(None, 12)
    axs[0, 0].legend(fontsize=8.5, loc="upper center")
    for j, nm in enumerate(("He$_1$", "He$_3$")):
        axs[0, 1 + j].set_title(f"hidden layer, h | {nm}", color=po.INK, loc="left", fontsize=11)
        axs[0, 1 + j].set_xlabel(f"h | {nm}")
        axs[0, 1 + j].legend(fontsize=8.5, loc="upper center")
    # row 2: coupling scatter, smallest and largest N, and the summary against N
    for ax, N in ((axs[1, 0], Ns[0]), (axs[1, 1], Ns[-1])):
        a = by[N][0]
        hc, rc = a["scatter_comparison"]
        hs, rsm = a["scatter_sampler"]
        ax.scatter(hc, rc, s=8, facecolor="none", edgecolor=po.C_THEORY, lw=0.5, alpha=0.5, zorder=2,
                   label="independent neurons with the same h (what the theory assumes)")
        ax.scatter(hs, rsm, s=4, color=cols[N], alpha=0.45, zorder=3, label="sampler")
        ax.axhline(1, color=po.MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 2.6)
        ax.set_xlim(0, 0.115)
        ax.set_xlabel("|h | He$_3$| of the hidden neuron")
        ax.set_ylabel("rms weight onto outer read-in neurons (prior units)")
        ax.set_title(f"N = {N}, one network ({a['n_out']:.0f} outer read-in neurons)", color=po.INK, loc="left",
                     fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
    ax = axs[1, 2]
    for m, mk, nm in (("corr", "o", "correlation of |h$_3$| with rms weight"),
                      ("cos_out", "s", "cosine of an outer neuron's outgoing column\nwith the other outer ones' sum")):
        for src, fc in (("sampler", None), ("comparison", "none")):
            x = [N for N in Ns for _ in by[N]]
            y = [a[src][m] for N in Ns for a in by[N]]
            ax.plot(x, y, mk, ms=7, color=po.C_HMC if src == "sampler" else po.C_THEORY,
                    mfc=fc if fc else (po.C_HMC), mew=1.3, zorder=3 if src == "sampler" else 2,
                    label=f"{nm}: {'sampler' if src == 'sampler' else 'independent neurons'}")
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_ylim(0, 1)
    ax.set_xlabel("N = N$_1$ = N$_2$   (χ = N / 1500, same theory)")
    ax.set_title("coupling statistics against width (one point per run)", color=po.INK, loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    f.suptitle(f"Width test at P = {P}, κ = 0.015, d = 150 (theory L$_3$ {z[f'sc_{P}'][1]:.3f})", x=0.01,
               ha="left", color=po.INK, fontsize=13)
    f.tight_layout()
    out = HERE / "plots" / "width" / f"coupling_width_P{P}.png"
    out.parent.mkdir(exist_ok=True)
    f.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


def main():
    cache = collect()
    for key, a in sorted(cache.items(), key=lambda kv: (kv[1]["N"], kv[0])):
        print(f"N={a['N']:5d} {key[-3:]} n_out {a['n_out']:6.1f} ({a['n_out'] / a['N']:.4f})  L3 {a['L3']:.3f}  "
              f"corr {a['sampler']['corr']:.2f} / {a['comparison']['corr']:.2f}  "
              f"cos_out {a['sampler']['cos_out']:.3f} / {a['comparison']['cos_out']:.3f}  "
              f"cos_pair {a['sampler']['cos_pair']:.3f} / {a['comparison']['cos_pair']:.3f}  "
              f"leftover cos_pair {a['leftover']['cos_pair']:.4f} / {a['leftover_comparison']['cos_pair']:.4f}  "
              f"leftover rms_out {a['leftover']['rms_out']:.3f} / {a['leftover_comparison']['rms_out']:.3f}")
    if P == 750:
        fig(cache)


if __name__ == "__main__":
    main()
