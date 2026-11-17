#!/usr/bin/env python3
"""The theory at a given number of outer read-in neurons, against the sampler runs, whose count is fixed from burn-in on.

In every sampler run the number n of read-in neurons beyond the theory's barrier is set during burn-in and never
changes, and it differs from run to run and from the theory's. Does the theory, held at a run's n, give that run's L3?

Held count. The theory's read-in distribution is p(w) ∝ p0(w) exp(-S(w)). Holding the outer fraction at f is the
same as tilting the read-in prior, p0(w) -> p0(w) exp(mu 1[|w0| > cut]), with mu chosen so that the solved
distribution has outer fraction f (the tilt is the Lagrange multiplier of the constraint; it changes W only by a
constant at fixed mu). cut is the theory's barrier at that P (plot_onset.barrier_cut), the cut the sampler counts use.
For each theory point the theory is solved along a ladder of mu, each solve warm-started from the previous one, which
traces L3 against n; every run is a point (n, L3) on the same axes.

Theory points and runs:
    P = 6000, 12000, 24000, 48000 at kappa = 0.015 (onset scan, three datasets each);
    theory points P = 12000 and 24000 reached with kappa ∝ P (queue_fixedtheory.sh: P x 2 and x 4).

    python theory_fixed_count.py [--recompute]   ->  plots/onset_checks/theory_fixed_count_d150.png
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

import plot_onset as po

HERE = Path(__file__).resolve().parent
CACHE = HERE / "plots" / "cache" / "theory_fixed_count_d150.npz"
PTS = [6000, 12000, 24000, 48000]
MUS = [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, -2.0, -5.0, -10.0, -20.0, -35.0, -50.0, -75.0, -100.0, -150.0]


def solve_tilted(Pt, mu, cut, start, device):
    import fcn3_fixed as F
    from fcn3_matrix_solver import DT
    Fn = F.BlazingFreeEnergy(po._params(Pt), device=device, verbose=False)
    Fn.lp0 = Fn.lp0 + mu * (Fn.ri.W >= cut).to(DT)
    o = Fn.solve(theta0=torch.tensor(start["theta"], dtype=DT, device=Fn.dev), M0=float(start["M"]))
    pw = Fn.last["pw"].reshape(Fn.ri.W.shape)
    o_ = Fn.ri.W >= cut
    out = dict(mu=mu, f=float(pw[o_].sum()), w_out=float((pw * Fn.ri.W)[o_].sum() / pw[o_].sum().clamp_min(1e-300)), L1=float(o["L1"]), L3=float(o["L3"]), M=float(o["M"]),
               theta=np.array(o["theta"]), W=float(o["W"]), H=np.array(o["H"]), conv=bool(o["converged"]))
    del Fn
    return out


def theory_curves(z, device):
    import gc
    res = {}
    for Pt in PTS:
        cut = po.barrier_cut(z, Pt)
        base = dict(theta=z[f"theta_{Pt}"], M=float(z[f"M_{Pt}"]))
        rows, prev = [], None
        for mu in MUS:
            if mu == MUS[0] or (mu < 0 and prev is not None and prev["mu"] > 0):
                prev = None                                                     # each branch starts from mu = 0
            start = base if prev is None else prev
            try:
                r = solve_tilted(Pt, mu, cut, start, device)
            except Exception as e:
                print(f"  P={Pt} mu={mu}: failed {e!r}"[:200], flush=True)
                continue
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(r)
            prev = r
            print(f"P={Pt} mu={mu:+.2f}: outer read-in neurons {r['f'] * po.N:6.1f}  L1 {r['L1']:.4f}  L3 {r['L3']:.4f}"
                  f"  outer well at |w0| {r['w_out']:.3f}  {'' if r['conv'] else '(not converged)'}", flush=True)
        rows.sort(key=lambda r: r["f"])
        res[Pt] = dict(cut=cut, n=np.array([r["f"] * po.N for r in rows]), L3=np.array([r["L3"] for r in rows]),
                       L1=np.array([r["L1"] for r in rows]), mu=np.array([r["mu"] for r in rows]))
    return res


def sampler_points(z):
    """(theory point, n, L3, L3 error, kind) for every onset run at PTS and every kappa ∝ P run."""
    pts = []
    runs = [r for r in po.sampler_runs() if r["P"] in PTS and r["w0"] is not None]
    for r in runs:
        cut = po.barrier_cut(z, r["P"])
        n = float(np.mean(np.abs(r["w0"]) >= cut)) * po.N
        pts.append((r["P"], n, r["L3"][0], r["L3"][1], "P scan, κ = 0.015", r["seed"]))
    for c in sorted(glob.glob(str(HERE / "runs" / "*ptheory*" / "config.json"))):
        run = Path(c).parent
        if not (run / "summary.json").exists():
            continue
        a = json.loads(Path(c).read_text())["args"]
        Pt = int(round(a["P"] * po.KAPPA / a["kappa"]))
        sn = np.load(run / "snaps.npz")
        w0 = sn["w0"][sn["iter"] >= a["burn"]].ravel()
        n = float(np.mean(np.abs(w0) >= po.barrier_cut(z, Pt))) * po.N
        L3 = json.loads((run / "summary.json").read_text())["posterior"]["L3"]
        pts.append((Pt, n, L3[0], L3[1], f"κ ∝ P, P × {a['P'] // Pt}", a["seed"]))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    if CACHE.exists() and not a.recompute:
        c = np.load(CACHE)
        res = {Pt: {k: c[f"{k}_{Pt}"] for k in ("n", "L3", "L1", "mu")} for Pt in PTS}
    else:
        res = theory_curves(z, a.device)
        np.savez(CACHE, **{f"{k}_{Pt}": v for Pt, d in res.items() for k, v in d.items() if k != "cut"})
    pts = sampler_points(z)

    print("\nrun-by-run: sampler L3 against the theory held at the run's count (and the free theory)")
    for Pt, n, L3, e, kind, seed in sorted(pts):
        th = float(np.interp(n, res[Pt]["n"], res[Pt]["L3"]))
        inside = res[Pt]["n"].min() <= n <= res[Pt]["n"].max()
        print(f"  theory point P={Pt:5d} {kind:22s} dataset {seed}: n = {n:6.1f}; L3 sampler {L3:.4f}, theory at n "
              f"{th:.4f}{'' if inside else ' (outside the solved range)'}, gap {th - L3:+.4f}; free theory "
              f"{float(z[f'sc_{Pt}'][1]):.4f} at n = {po.outer_theory(z, Pt, po.barrier_cut(z, Pt)) * po.N:.0f}")

    plt = po.style()
    fig, axs = plt.subplots(1, len(PTS), figsize=(4.3 * len(PTS), 4.6), squeeze=False)
    marks = {"P scan, κ = 0.015": ("o", po.C_HMC), "κ ∝ P, P × 2": ("s", "#1baf7a"), "κ ∝ P, P × 4": ("D", "#7b5ce0")}
    for c, Pt in enumerate(PTS):
        ax = axs[0, c]
        nr = [n for P_, n, *_ in pts if P_ == Pt] + [po.outer_theory(z, Pt, po.barrier_cut(z, Pt)) * po.N]
        lo, hi = 0.8 * min(nr), 1.25 * max(nr)
        k = (res[Pt]["n"] > 5) & (res[Pt]["n"] < 1400)              # drop the extreme tilts, where all or none are outer
        n_, L_ = res[Pt]["n"][k], res[Pt]["L3"][k]
        k = (n_ >= lo - 10) & (n_ <= hi + 10)
        ax.plot(n_[k], L_[k], color=po.C_THEORY, lw=2, marker=".", ms=5, label="theory held at n outer read-in neurons")
        ax.set_xlim(lo, hi)
        ax.plot([po.outer_theory(z, Pt, po.barrier_cut(z, Pt)) * po.N], [float(z[f"sc_{Pt}"][1])], "*", color=po.C_THEORY, ms=13,
                mec="white", label="theory, count free")
        for kind, (m, col) in marks.items():
            q = [(n, L3, e) for P_, n, L3, e, k, _ in pts if P_ == Pt and k == kind]
            if q:
                n, L3, e = map(np.array, zip(*q))
                ax.errorbar(n, L3, yerr=e, fmt=m, color=col, ms=6, mec="white", capsize=0, label=f"sampler, {kind}")
        ax.set_xlabel("outer read-in neurons n")
        if c == 0:
            ax.set_ylabel("L$_3$")
        ax.set_title(f"theory point P = {Pt}", color=po.INK, loc="left", fontsize=11)
    hs, ls = [], []
    for ax in axs[0]:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_)
                ls.append(l_)
    fig.legend(hs, ls, loc="lower center", ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"L$_3$ against the number of outer read-in neurons, d = {po.D}, N = {po.N}: theory held at each count, "
                 f"and every sampler run at its own (fixed) count", color=po.INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out = HERE / "plots" / "onset_checks" / "theory_fixed_count_d150.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
