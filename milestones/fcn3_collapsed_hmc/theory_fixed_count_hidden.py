#!/usr/bin/env python3
"""Second hidden layer of the theory held at the sampler runs' outer read-in count (theory_fixed_count.py), against
the sampler. If the held theory develops the sampler's structure (the far h|He1 group, the large-|h3| group at
P = 48000), the hidden-layer mismatch comes from the frozen count, not from the theory.

For each P the read-in prior is tilted by exp(mu 1[|w0| > barrier]) and mu is lowered in steps, each solve warm-started
from the previous one, until the outer count reaches the runs' mean. The hidden-layer marginals at that solution are
computed as in plot_onset.hidden_marginals (|h1| to 8, |h3| to 3).

    python theory_fixed_count_hidden.py   ->  plots/onset_checks/theory_fixed_count_hidden_d150.png
"""
import gc
from pathlib import Path

import numpy as np
import torch

import plot_onset as po

HERE = Path(__file__).resolve().parent
TARGET = {24000: 51.0, 48000: 57.0}            # mean outer count of the onset runs (datasets 0-2)


def solve(Pt, mu, cut, start, device):
    import fcn3_fixed as F
    from fcn3_matrix_solver import DT
    Fn = F.BlazingFreeEnergy(po._params(Pt), device=device, verbose=False)
    Fn.lp0 = Fn.lp0 + mu * (Fn.ri.W >= cut).to(DT)
    o = Fn.solve(theta0=torch.tensor(start["theta"], dtype=DT, device=Fn.dev), M0=float(start["M"]))
    p = Fn.p
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape)
    r = dict(theta=np.array(o["theta"]), M=float(o["M"]), L3=float(o["L3"]), conv=bool(o["converged"]),
             n=float(pw[Fn.ri.W >= cut].sum()) * po.N,
             G=(p.sigB ** 2 * Fn.last["J"]).detach().cpu().numpy(), t=Fn.t.detach().cpu().numpy(), gt=float(p.gt),
             ks=np.array(p.ks), H=np.array(o["H"]))
    with torch.no_grad():
        e, h, q = Fn.mc.features(F.scaled_cholesky(p.sigB ** 2 * Fn.last["J"]), p.sigB ** 2 * Fn.last["T"])
        w = torch.softmax(Fn.mc.logw(o["M"]) + 0.5 * p.gt * (e @ Fn.t) ** 2, 0)
        n = 2 ** 14
        u = (torch.rand(1, dtype=w.dtype, device=w.device, generator=torch.Generator(w.device).manual_seed(3))
             + torch.arange(n, dtype=w.dtype, device=w.device)) / n
        idx = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=len(w) - 1)
        r["hs"], r["qs"] = h[idx].cpu().numpy(), q[idx].cpu().numpy()
    del Fn
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main():
    dev = "cuda:0"
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    wide = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_hidden_wide.npz"))
    groups = po.by_P(po.sampler_runs())
    res = {}
    for Pt, target in TARGET.items():
        cut = po.barrier_cut(z, Pt)
        cur = dict(theta=z[f"theta_{Pt}"], M=float(z[f"M_{Pt}"]))
        best, mu = None, 0.0
        while mu > -400:
            mu -= 10.0 if mu > -100 else 20.0
            try:
                r = solve(Pt, mu, cut, cur, dev)
            except Exception as ex:
                print(f"P={Pt} mu={mu}: failed {ex!r}"[:160], flush=True)
                continue
            print(f"P={Pt} mu={mu:+.0f}: outer {r['n']:.1f}  L3 {r['L3']:.4f}  {'' if r['conv'] else '(not converged)'}",
                  flush=True)
            if r["n"] > 1000 or r["n"] < 1:                      # jumped to another branch: keep the last good one
                break
            cur, best = r, r
            if r["n"] <= target:
                break
        r = best
        h1g, lp1, h3g, lp3 = po.hidden_marginals(r, dev, h1max=8.0, n1=4001, n3=6001, h3max=3.0)
        H, _ = po.sampler_hidden([q for q in groups[Pt] if (q["path"] / "nets").exists()])
        res[Pt] = dict(held=(h1g, lp1, h3g, lp3), n=r["n"], L3=r["L3"], sampler=H)
        for nm, g, lp, x in (("h1", h1g, lp1, H[:, 0]), ("h3", h3g, lp3, H[:, 1])):
            a = -lp - (-lp).min()
            mins = [(g[j], a[j]) for j in range(1, len(g) - 1) if g[j] >= 0 and a[j] < a[j - 1] and a[j] <= a[j + 1]]
            print(f"   held theory (n = {r['n']:.0f}) {nm}: wells " + ", ".join(f"{m:.3f} ({v:.1f})" for m, v in mins)
                  + f"; action at |{nm}| = 0.4: {np.interp(0.4, g, a):.1f}, at 2.0: {np.interp(2.0, g, a):.1f}", flush=True)

    plt = po.style()
    Ps = list(TARGET)
    fig, axs = plt.subplots(2, len(Ps), figsize=(6.2 * len(Ps), 8), squeeze=False)
    for c, Pt in enumerate(Ps):
        for i, (j, lim, xl) in enumerate(((1, 3.0, "h | He$_1$"), (3, 0.7, "h | He$_3$"))):
            ax = axs[i, c]
            curves = [(wide[f"h{j}g_{Pt}"], -wide[f"lp{j}_{Pt}"], po.C_THEORY, "-",
                       f"theory, count free (n = {po.outer_theory(z, Pt, po.barrier_cut(z, Pt)) * po.N:.0f})"),
                      (res[Pt]["held"][2 * (i)], -res[Pt]["held"][2 * i + 1], "#7b5ce0", "--",
                       f"theory held at the runs' count (n = {res[Pt]['n']:.0f})")]
            lo = np.inf
            for g, y, col, ls, lab in curves:
                k = np.abs(g) <= lim
                ax.plot(g[k], y[k], ls, color=col, lw=2, label=lab, zorder=2)
                lo = min(lo, y.min())
            x = res[Pt]["sampler"][:, i]
            edges = np.linspace(-lim, lim, 121)
            mid = 0.5 * (edges[1:] + edges[:-1])
            cnt, _ = np.histogram(x, edges)
            k = cnt > 0
            yy = -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k]))
            ax.errorbar(mid[k], yy, yerr=1 / np.sqrt(cnt[k]), fmt="o", ms=2.8, color=po.C_HMC, mec="white", mew=0.3,
                        elinewidth=0.7, capsize=0, zorder=3, label="sampler (outer count frozen at ~"
                        f"{TARGET[Pt]:.0f})")
            lo = min(lo, yy.min())
            ax.set_xlim(-lim, lim)
            ax.set_ylim(lo - 0.5, lo + 25)
            ax.set_xlabel(xl)
            ax.set_ylabel(f"−log p({xl})")
            if i == 0:
                ax.set_title(f"P = {Pt}", color=po.INK, loc="left", fontsize=12)
            ax.legend(fontsize=8.5, loc="upper center")
    fig.suptitle(f"Second hidden layer: theory with the outer read-in count free and held at the sampler's, d = {po.D}, "
                 f"N = {po.N}, κ = {po.KAPPA}", color=po.INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "theory_fixed_count_hidden_d150.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
