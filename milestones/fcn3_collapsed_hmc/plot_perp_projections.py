#!/usr/bin/env python3
"""How much of the second hidden layer's learning shows up in the perpendicular input directions? Onset scan
(queue_onset.sh: d = 150, N1 = N2 = 1500, kappa = 0.015, P = 750 ... 48000; 15 saved networks per run, 3 datasets).

For hidden neuron k with pre-activation h_k(x) = sum_j B_kj erf(w_j . x), x ~ N(0, I_d), the projection onto
He_n(u . x)/sqrt(n!) for a unit vector u is exact:  sum_j B_kj c_n(w_j . u, |w_j|^2 - (w_j . u)^2)  (readin_features).
h is odd in x, so even n vanish. Directions u:
  x0                      the target direction (He1 and He3 here are the h|He1, h|He3 of plot_onset.py),
  x_i, i = 1 ... d-1      every perpendicular coordinate, pooled,
  strongest perpendicular the top right singular vector of the (N2 x (d-1)) matrix of perpendicular He1 projections of
                          that network (the perpendicular direction its hidden layer projects onto most).
Perpendicular projections are measured in units of their standard deviation in untrained networks (ZW, ZB standard
normal, same sizes; the same for every direction by symmetry), and the same statistics of 15 untrained networks are
shown as "prior" (for the strongest direction this includes the effect of picking the largest of 149). The target
projections, thousands of those units across, are shown in their own units (He3 and He5 scaled up to the He1 range).

    python plot_perp_projections.py [--recompute]
        ->  plots/onset/onset_d150_perp_heatmap.png    -log p of each projection, one panel per P
            plots/onset/onset_d150_perp_magnitude.png  mean squared projection / its prior value, against P
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po
from fcn3_matrix_solver import readin_features

HERE = Path(__file__).resolve().parent
CACHE = HERE / "plots" / "cache" / "perp_projections_d150.npz"
KS = (1, 3, 5)
ROWS = ["x0", "perp", "top"]                        # direction families
XP, WB = 30.0, 0.02                                 # perpendicular histograms: half-range and bin (prior units)
XT = (4.0, 1.0, 0.5)                                # target histograms: half-ranges (raw units), 4000 bins each


def projections(ZW, ZB, sw, sb):
    """(target (N2, 3), perp (3, N2, d-1), top (N2, 3)) projections onto He1, He3, He5, in the network's units."""
    W, B = sw * ZW.double(), sb * ZB.double()
    n2 = (W * W).sum(1, keepdim=True)
    C = readin_features(W, n2 - W * W, KS)                                   # (3, N1, d): every coordinate direction
    Pc = torch.einsum("kj,sjd->skd", B, C)                                   # (3, N2, d)
    u = torch.linalg.svd(Pc[0, :, 1:], full_matrices=False)[2][0]            # strongest perpendicular direction
    wu = W[:, 1:] @ u
    Pu = (B @ readin_features(wu, n2[:, 0] - wu * wu, KS).T)                 # (N2, 3)
    return Pc[:, :, 0].T, Pc[:, :, 1:], Pu


def compute():
    groups = po.by_P(po.sampler_runs())
    nets = {P: [f for r in groups[P] for f in sorted((r["path"] / "nets").glob("net_*.pt"))] for P in groups}
    runs = {P: [sorted((r["path"] / "nets").glob("net_*.pt")) for r in groups[P] if (r["path"] / "nets").exists()]
            for P in groups}
    nt0 = torch.load(nets[max(nets)][0], map_location="cpu", weights_only=False)
    sw, sb = float(nt0["sw"]), float(nt0["sb"])
    N1, d = nt0["ZW"].shape
    N2 = nt0["ZB"].shape[0]
    g = torch.Generator().manual_seed(0)
    prior = [(torch.randn(N1, d, generator=g), torch.randn(N2, N1, generator=g)) for _ in range(15)]
    # prior standard deviation of each He_n projection, pooled over all d coordinates (isotropic)
    acc = torch.zeros(3, dtype=torch.float64)
    for ZW, ZB in prior:
        t, p, _ = projections(ZW, ZB, sw, sb)
        acc += (t ** 2).sum(0) + (p ** 2).sum((1, 2))
    s = torch.sqrt(acc / (len(prior) * N2 * d))
    print("prior standard deviation of the He1, He3, He5 projections:", s.numpy())
    eT = [np.linspace(-x, x, 4001) for x in XT]
    eP = np.arange(-XP, XP + WB / 2, WB)
    out = dict(s=s.numpy(), eP=eP, **{f"eT{n}": e for n, e in enumerate(eT)})
    sets = {0: [[(ZW, ZB) for ZW, ZB in prior]]}
    sets.update({P: [[None] * len(fs) for fs in runs[P]] for P in runs})
    for P in sorted(sets):
        H = {(row, n): np.zeros(len((eT[n] if row == "x0" else eP)) - 1) for row in ROWS for n in range(3)}
        mags = []                                                            # per dataset: (3 rows, 3 n)
        for di, fs in enumerate(sets[P] if P == 0 else runs[P]):
            m = np.zeros((3, 3))
            for item in fs:
                if P == 0:
                    ZW, ZB = item
                else:
                    nt = torch.load(item, map_location="cpu", weights_only=False)
                    ZW, ZB = nt["ZW"], nt["ZB"]
                t, p, u = projections(ZW, ZB, sw, sb)
                for n in range(3):
                    vals = {"x0": t[:, n], "perp": p[n].reshape(-1) / s[n], "top": u[:, n] / s[n]}
                    for ri, row in enumerate(ROWS):
                        v = vals[row].numpy()
                        H[row, n] += np.histogram(v, eT[n] if row == "x0" else eP)[0]
                        m[ri, n] += float((v ** 2).mean()) / (float(s[n]) ** 2 if row == "x0" else 1)
            mags.append(m / len(fs))
        for (row, n), h in H.items():
            out[f"H_{P}_{row}_{n}"] = h
        out[f"mag_{P}"] = np.array(mags)
        print(f"P={P or 'prior'}: mean squared projection / prior, He1 He3 He5 (datasets averaged):\n"
              + "\n".join(f"   {row:5s} " + "  ".join(f"{x:7.3f}" for x in np.mean(mags, 0)[ri])
                          for ri, row in enumerate(ROWS)), flush=True)
    np.savez(CACHE, **out)
    return out


def rebin(h, k):
    n = len(h) // k * k
    return h[:n].reshape(-1, k).sum(1)


def action_row(h, edges, k, grid):
    """-log p of a histogram on fine bins, merged k at a time, above its minimum, sampled on grid (nan: no neurons)."""
    h = rebin(h, k)
    e = edges[: len(h) * k + 1: k]
    with np.errstate(divide="ignore"):
        A = -np.log(h / (h.sum() * np.diff(e)))
    A = A - A[np.isfinite(A)].min()
    A[~np.isfinite(A)] = np.nan
    mid = 0.5 * (e[1:] + e[:-1])
    j = np.clip(np.searchsorted(mid, grid), 1, len(mid) - 1)
    j = np.where(np.abs(grid - mid[j - 1]) < np.abs(grid - mid[j]), j - 1, j)    # nearest bin, no smoothing
    out = A[j]
    out[(grid < e[0]) | (grid > e[-1])] = np.nan
    return out


def extent(z, Ps, row, n, edges):
    m = 0.0
    for P in Ps:
        nz = np.nonzero(z[f"H_{P}_{row}_{n}"])[0]
        m = max(m, abs(edges[nz[0]]), abs(edges[nz[-1] + 1]))
    return m


def nice(x):
    """Largest of 1, 2, 5, 10, 20, ... not above x."""
    c = [m * 10 ** e for e in range(4) for m in (1, 2, 5)]
    return max([v for v in c if v <= x] or [1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    z = dict(np.load(CACHE)) if CACHE.exists() and not a.recompute else compute()
    Ps = sorted(int(k.split("_")[1]) for k in z if k.startswith("mag_"))
    plt = po.style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("action", ["#0b3d91", "#2a78d6", "#9cc3ee", "#f4f8fd"])
    cmap.set_bad("white")
    top = 12.0
    names = {"perp": "x$_i$, every i ≥ 1", "top": "strongest perp. direction"}

    # target rows in raw units, He3 and He5 multiplied by a factor that brings them to the He1 range
    extT = [extent(z, Ps, "x0", n, z[f"eT{n}"]) for n in range(3)]
    fac = [1] + [nice(extT[0] / extT[n]) for n in (1, 2)]
    ext0 = 1.05 * max(extT[n] * fac[n] for n in range(3))
    extP = 1.05 * max(extent(z, Ps, r, n, z["eP"]) for r in ("perp", "top") for n in range(3))
    gT, gP = np.linspace(-ext0, ext0, 601), np.linspace(-extP, extP, 601)

    fig, axs = plt.subplots(2, len(Ps), figsize=(2.75 * len(Ps), 7.8), squeeze=False,
                            gridspec_kw=dict(height_ratios=[3, 6], hspace=0.42, wspace=0.08))
    for c, P in enumerate(Ps):
        img, lab = [], []
        for n in range(3):
            e = z[f"eT{n}"] * fac[n]
            k = max(1, int(round(2 * ext0 / (e[1] - e[0]) / 150)))
            img.append(action_row(z[f"H_{P}_x0_{n}"], e, k, gT))
            lab.append(f"He$_{2 * n + 1}$(x$_0$)" + (f" × {fac[n]}" if fac[n] > 1 else ""))
        ax = axs[0, c]
        im = ax.imshow(np.array(img), aspect="auto", cmap=cmap, vmin=0, vmax=top, interpolation="nearest",
                       extent=(-ext0, ext0, 2.5, -0.5))
        ax.set_yticks(range(3))
        ax.set_yticklabels(lab if c == 0 else [], fontsize=9)
        ax.set_xlabel("projection", fontsize=9)
        ax.set_title("prior (untrained)" if P == 0 else f"P = {P}", color=po.INK, loc="left", fontsize=11, pad=4)
        img, lab = [], []
        for row in ("perp", "top"):
            for n in range(3):
                k = max(1, int(round(2 * extP / WB / 150)))
                img.append(action_row(z[f"H_{P}_{row}_{n}"], z["eP"], k, gP))
                lab.append(f"He$_{2 * n + 1}$ ({names[row]})")
        ax = axs[1, c]
        ax.imshow(np.array(img), aspect="auto", cmap=cmap, vmin=0, vmax=top, interpolation="nearest",
                  extent=(-extP, extP, 5.5, -0.5))
        ax.axhline(2.5, color="white", lw=2)
        ax.set_yticks(range(6))
        ax.set_yticklabels(lab if c == 0 else [], fontsize=9)
        ax.set_xlabel("projection / untrained sd", fontsize=9)
        for ax in axs[:, c]:
            ax.grid(False)
    axs[0, 0].text(-0.02, 1.12, "target direction", transform=axs[0, 0].transAxes, ha="right", fontsize=10,
                   color=po.MUTED)
    axs[1, 0].text(-0.02, 1.06, "perpendicular directions", transform=axs[1, 0].transAxes, ha="right",
                   fontsize=10, color=po.MUTED)
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
    cb.set_label("−log p, nats above each row's minimum (white: no neurons)")
    fig.suptitle(f"Second-hidden-layer pre-activations projected on Hermite functions of single input directions, "
                 f"d = {po.D}, N = {po.N}, κ = {po.KAPPA}: every hidden neuron of the saved networks",
                 color=po.INK, x=0.01, ha="left", fontsize=12)
    out = HERE / "plots" / "onset" / "onset_d150_perp_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)

    # sizes against P: target in raw units (root mean square), perpendicular relative to untrained networks
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    Pn = [P for P in Ps if P > 0]
    panels = (("x0", "target direction x$_0$", "root mean square projection"),
              ("perp", "each perpendicular coordinate x$_i$ (averaged over i)",
               "mean squared projection / untrained value"),
              ("top", "strongest perpendicular direction", "mean squared projection / untrained value (single direction)"))
    for ri, (row, ttl, yl) in enumerate(panels):
        ax = axs[ri]
        for n in range(3):
            m = np.array([z[f"mag_{P}"][:, ri, n] for P in Pn])               # (P, datasets), prior units
            m0 = z["mag_0"][:, ri, n].mean()
            if row == "x0":
                m, m0 = np.sqrt(m) * z["s"][n], math.sqrt(m0) * z["s"][n]
            mu, se = m.mean(1), m.std(1, ddof=1) / math.sqrt(m.shape[1])
            ax.errorbar(Pn, mu, yerr=se, color=cols[n], marker="o", ms=5, lw=2, capsize=0,
                        label=f"He$_{2 * n + 1}$")
            ax.axhline(m0, color=cols[n], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("P")
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(ttl, color=po.INK, loc="left", fontsize=11)
        if ri == 0:
            ax.legend(loc="upper left")
    fig.suptitle(f"Size of the hidden layer's projections, d = {po.D}, N = {po.N}, κ = {po.KAPPA} "
                 f"(dashed: untrained networks; bars: spread over the 3 datasets)", color=po.INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset" / "onset_d150_perp_magnitude.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
