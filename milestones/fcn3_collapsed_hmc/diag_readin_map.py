#!/usr/bin/env python3
"""Where do read-in neurons sit in (|w0|, log q)?  Sampler snapshots (pooled over the snapshot chains at one d) against
the theory's p(w0, q) from fcn3_fixed.solve_fcn3, with curves of constant r^2 = w0^2/(1 + 2 w0^2 + 2 q): the read-in
features c_n = A_n r^n depend on (w0, q) only through r, so the data term is constant along these curves and only
the prior beta (w0^2 + q) - (d-3)/2 log q and P_perp change.

    python diag_readin_map.py [--d 150]      ->  plots/ray/readin_map_d<d>.png
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIGW2 = 0.5
SIGB = {75: 1.2574845533917596, 150: 1.25540121081682, 300: 1.254358125197598, 600: 1.2538362420680749}


def theory_grid(d, device):
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
               sigA=math.sqrt((8 / 1500) / r), sigB=SIGB[d], sigw=math.sqrt(SIGW2), nmax=11)
    o = F.solve_fcn3(p, nmax=11, device=device)
    F.BlazingFreeEnergy.solve = orig
    return o["_w"], o["_lq"], o["_pw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    d = a.d
    q0 = SIGW2 * (d - 1) / d
    cache = HERE / "plots" / "cache" / f"theory_readin2d_d{d}.npz"
    if cache.exists():
        z = np.load(cache)
        w, lq, pw = z["w"], z["lq"], z["pw"]
    else:
        w, lq, pw = theory_grid(d, a.device)
        np.savez(cache, w=w, lq=lq, pw=pw)
    # theory: -log density in (|w0|, log q/q0)
    dens = pw / (np.gradient(w)[:, None] * np.gradient(lq)[None, :])
    with np.errstate(divide="ignore"):
        nlt = -np.log(dens)
    # sampler
    W0, Q, seeds = [], [], []
    for f in sorted(glob.glob(str(HERE / "runs" / f"*_ray_d{d}_*N1500x1500_*" / "snaps.npz"))):
        c = json.loads((Path(f).parent / "config.json").read_text())["args"]
        if not str(c.get("group", "")).startswith("ray_hist"):
            continue
        s = np.load(f)
        post = s["iter"] >= c["burn"]
        W0.append(np.abs(s["w0"][post]).ravel()); Q.append(s["q"][post].ravel()); seeds.append(c["seed"])
    w0s, qs = np.concatenate(W0), np.concatenate(Q)
    x, y = w0s, np.log(qs / q0)
    r2 = w0s ** 2 / (1 + 2 * w0s ** 2 + 2 * qs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axs[0]
    xe, ye = np.linspace(0, 1.3, 131), np.linspace(-1.6, 0.5, 106)
    H, _, _ = np.histogram2d(x, y, [xe, ye])
    with np.errstate(divide="ignore"):
        Hn = -np.log(H / (H.sum() * np.diff(xe)[0] * np.diff(ye)[0]))
    im = ax.pcolormesh(xe, ye, np.where(np.isfinite(Hn), Hn, np.nan).T, cmap="viridis_r", vmin=Hn[np.isfinite(Hn)].min(),
                       vmax=Hn[np.isfinite(Hn)].min() + 14, shading="flat")
    fig.colorbar(im, ax=ax, label="sampler −log p(|w$_0$|, log q)")
    W, LQ = np.meshgrid(w, lq - math.log(q0), indexing="ij")
    lev = np.nanmin(nlt[np.isfinite(nlt)]) + np.array([1, 3, 6, 10, 14])
    cs = ax.contour(W, LQ, np.where(np.isfinite(nlt), nlt, 1e9), levels=lev, colors="#d62728", linewidths=1.1)
    ax.clabel(cs, fmt=lambda v: f"{v - lev[0] + 1:.0f}", fontsize=7)
    for rr in (0.05, 0.1, 0.15, 0.2, 0.24):
        ww = np.linspace(0.01, 1.3, 300)
        qq = (ww ** 2 * (1 - 2 * rr) / rr - 1) / 2        # r^2 = w0^2/(1+2w0^2+2q)  ->  q
        ok = qq > 0
        ax.plot(ww[ok], np.log(qq[ok] / q0), ":", color="white", lw=1)
        j = np.argmin(np.abs(np.log(np.maximum(qq, 1e-9) / q0) - 0.35))
        ax.text(ww[j], 0.38, f"r²={rr}", color="white", fontsize=7, ha="center")
    ax.set_xlim(0, 1.3); ax.set_ylim(-1.6, 0.5)
    ax.set_xlabel("|w$_0$|"); ax.set_ylabel("log (q / q$_{prior}$)")
    ax.set_title(f"d = {d}: sampler (colour, datasets {seeds}) vs theory (red contours, nats above min)", fontsize=10)

    ax = axs[1]
    re = np.linspace(0, 0.32, 65)
    cnt, _ = np.histogram(r2, re)
    ok = cnt > 0
    ax.plot(0.5 * (re[1:] + re[:-1])[ok], -np.log(cnt[ok] / (cnt.sum() * np.diff(re)[0])), "o", color="#eb6834",
            ms=4, label="sampler")
    R2 = W ** 2 / (1 + 2 * W ** 2 + 2 * np.exp(LQ) * q0)
    ct, _ = np.histogram(R2.ravel(), re, weights=pw.ravel())
    okt = ct > 0
    ax.plot(0.5 * (re[1:] + re[:-1])[okt], -np.log(ct[okt] / (ct.sum() * np.diff(re)[0])), "-", color="#2a78d6",
            lw=2, label="theory")
    ax.set_xlabel("r² = w$_0^2$/(1 + 2w$_0^2$ + 2q)"); ax.set_ylabel("−log p(r²)")
    ax.set_title("the variable the features depend on", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = HERE / "plots" / "ray" / f"readin_map_d{d}.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    # numbers: where are the shoulder neurons
    sh = (w0s > 0.35) & (w0s < 0.75)
    ou = w0s >= 0.75
    print(f"shoulder (0.35<|w0|<0.75): {sh.mean():.2e} of samples; median log q/q0 {np.median(y[sh]):.3f}, median r2 "
          f"{np.median(r2[sh]):.3f}")
    print(f"outer (|w0|>=0.75): median log q/q0 {np.median(y[ou]):.3f}, median r2 {np.median(r2[ou]):.3f}, r2 sd "
          f"{r2[ou].std():.3f}")


if __name__ == "__main__":
    main()
