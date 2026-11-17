#!/usr/bin/env python3
"""Read-in action along the kappa0 = 0.015 ray, two-layer (left) against three-layer network (right), colour = d (magma,
one log-d scale for both panels). Lines: theory; points: sampler. Every curve is -log p(w0) / beta with
beta = d / (2 sigma_w^2) (the convention of fcn2_collapsed_hmc/plot_large_d_actions.py: every term of the action grows
like d, so this puts all d on one scale), p the density on the full line.

FCN2: every saved sample of the chains of groups larged_d75, pscan_d150_P3000, mix_d300, w0q_d600, w0q_d800, w0q_d1200;
theory with a per-neuron perpendicular norm and the x0 tower to He11 (readin_action_theory_q.analyse_q).
FCN3: N1 = N2 = 1500 ray runs (and the onset runs at d = 150, P = 3000, same parameters), every post-burn snapshot where
saved, otherwise the final network (d = 75 and 600 have final networks only, two runs each); runs started with a fixed
outer count left out. Theory: the read-in action at each solution on the full grid (fcn3_theory).

Third row: the outer read-in well's mean |w0| and width against d (sampler: |w0| beyond the theory's barrier),
and sampler minus theory.

Second row: L1 and L3 against d (both from 0), same colours; squares / dashed line FCN2, circles / solid line FCN3. Sampler
learnabilities are exact population coefficients (FCN2: fcn2_ray_L3_exact.py; FCN3: each run's summary).

Fourth row: FCN3 read-in action at d = 150 with the theory held fixed and P doubled twice (kappa ∝ P): theory points
P = 3000 (dataset 2: P = 3000, 6000, 12000), 12000 and 24000 (dataset 0: up to P = 96000); raw -log p(w0), every
post-burn snapshot; the sampler's L3 in the legend.

Seventh row: the per-P actions of plot_onset.py (plots/onset/onset_d150_actions.png) overlaid, colour = P: read-in at
full height, h|He1 and h|He3 of the second hidden layer (sampler part cached in plots/cache/onset_overlay_d150.npz).

    python plot_ray_readin_fcn2_vs_fcn3.py   ->  plots/ray/readin_ray_fcn2_vs_fcn3.png
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import plot_onset as po

HERE = Path(__file__).resolve().parent
FCN2 = HERE.parent / "fcn2_collapsed_hmc"
GROUPS2 = {75: "larged_d75", 150: "pscan_d150_P3000", 300: "mix_d300", 600: "w0q_d600", 800: "w0q_d800",
           1200: "w0q_d1200"}
SW2 = 0.5
EDGES = np.linspace(-1.6, 1.6, 81)
CACHE2 = HERE / "plots" / "cache" / "fcn2_ray_readin.npz"


def colour(d, plt):
    t = (math.log2(d) - math.log2(75)) / (math.log2(1200) - math.log2(75))
    return plt.cm.magma(0.08 + 0.72 * t)


def fcn2_data():
    if CACHE2.exists():
        z = dict(np.load(CACHE2))
        return {d: dict(w0=z[f"w0_{d}"], wq=z[f"wq_{d}"], th=z[f"th_{d}"]) for d in GROUPS2}
    sys.path.insert(0, str(FCN2))
    from chains import chain_runs
    from readin_action_theory_q import analyse_q
    out, save = {}, {}
    for d, g in GROUPS2.items():
        w0 = []
        for _, run, a in chain_runs(g):
            for f in sorted(glob.glob(str(run / "samples" / "iter_*.pt"))):
                w0.append(torch.load(f)["W0"].double().reshape(a["N"], a["d"])[:, 0].numpy())
        q = analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], a["chi"], ks=(1, 3, 5, 7, 9, 11),
                      nq=int(500 * max(1, np.sqrt(d / 600))))
        wq = np.concatenate([-q["w"][::-1], q["w"][1:]])
        nq = np.concatenate([q["nlp"][::-1], q["nlp"][1:]])
        m = np.abs(wq) <= 1.6
        out[d] = dict(w0=np.concatenate(w0), wq=wq[m], th=nq[m])
        save.update({f"w0_{d}": out[d]["w0"], f"wq_{d}": wq[m], f"th_{d}": nq[m]})
        print(f"FCN2 d={d}: {len(w0)} samples; theory L3 {q['L3']:.3f}", flush=True)
    np.savez(CACHE2, **save)
    return out


def fcn2_learn():
    """{d: (L1 theory, L3 theory, [(L1, L3) per chain])}: theory from analyse_q (cached), sampler from
    fcn2_ray_L3_exact.py (exact population coefficients from every saved sample)."""
    cache = HERE / "plots" / "cache" / "fcn2_theory_ray_learn.json"
    if cache.exists():
        th = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    else:
        sys.path.insert(0, str(FCN2))
        from chains import chain_runs
        from readin_action_theory_q import analyse_q
        th = {}
        for d, g in GROUPS2.items():
            a = chain_runs(g)[0][2]
            q = analyse_q(a["N"], d, a["s0"], a["sa0"], a["P"], a["T"] / 2, a["eps"], a["chi"],
                          ks=(1, 3, 5, 7, 9, 11), nq=int(500 * max(1, np.sqrt(d / 600))))
            th[d] = [float(q["L1"]), float(q["L3"])]
        cache.write_text(json.dumps(th, indent=1))
    ex = json.loads((HERE / "plots" / "cache" / "fcn2_ray_L3_exact.json").read_text())
    return {d: (th[d][0], th[d][1], [(c["L1"], c["L3"]) for c in ex[str(d)].values()]) for d in GROUPS2}


def full_readin_theory(p, device="cuda:0"):
    """(w0 grid, -log p(w0) on the full line, L1, L3) of the theory at params p, from the read-in action on the full
    (w0, log q) grid (the solver itself prunes points more than 60 nats below the peak)."""
    import gc
    from dataclasses import replace
    import fcn3_fixed as F
    from fcn3_matrix_solver import DT, trapz_weights
    F3 = F.BlazingFreeEnergy(replace(p, nmax=3), device=device, verbose=False)
    o3 = F3.solve()
    Fn = F.BlazingFreeEnergy(p, device=device, verbose=False)
    th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
    th[:3] = F3.theta_ref[:3].to(Fn.dev)
    th[-1] = F3.theta_ref[-1].to(Fn.dev)
    o = Fn.solve(theta0=th, M0=o3["M"])
    with torch.no_grad():
        lw = (Fn.lp0.reshape(-1) - 0.5 * (Fn.theta_ref @ Fn.Tst)).reshape(Fn.ri.W.shape)
        lw = lw - torch.logsumexp(lw.reshape(-1), 0)
        lm = torch.logsumexp(lw, 1)
        nlp = -(lm - torch.log(trapz_weights(Fn.ri.w)) + math.log(0.5))
    r = (Fn.ri.w.cpu().numpy(), nlp.cpu().numpy(), float(o["L1"]), float(o["L3"]))
    del F3, Fn
    gc.collect()
    torch.cuda.empty_cache()
    return r


CACHE3 = HERE / "plots" / "cache" / "theory_readin_ray_full.npz"
CACHE_EK = HERE / "plots" / "cache" / "theory_readin_ek_full.npz"
# fixed theory, P doubled twice with kappa ∝ P: theory point -> (dataset, [(multiple, run-dir pattern)])
EK = {3000: (2, [(1, "*_onset_d150_P3000_N1500x1500_prior_r12"), (1, "*_ray_d150_P3000_N1500x1500_prior_r12"),
                 (2, "*_pscan_d150_P6000_N1500x1500_prior_r22"), (4, "*_pscan_d150_P12000_N1500x1500_prior_r22")]),
      12000: (0, [(1, "*_onset_d150_P12000_N1500x1500_prior_r10"), (2, "*_ptheory_d150_P24000_N1500x1500_*"),
                  (4, "*_ptheory_d150_P48000_N1500x1500_*")]),
      24000: (0, [(1, "*_onset_d150_P24000_N1500x1500_prior_r10"), (2, "*_ptheory_d150_P48000_N1500x1500_*"),
                  (4, "*_ptheory_d150_P96000_N1500x1500_*")])}
EK_COLS = {1: "#f4a582", 2: "#d6604d", 4: "#8a1c1c"}


CACHE_EKR = HERE / "plots" / "cache" / "theory_readin_ek_runs.npz"


def run_theory(Pt, runs):
    """The theory solved at each run's own (P, kappa) (kappa ∝ P leaves only kappa/P in the equations, but this is
    checked rather than assumed). Adds r["theory"] = (w, nlp, L1, L3) to every multiple; cached."""
    from dataclasses import replace
    z = dict(np.load(CACHE_EKR)) if CACHE_EKR.exists() else {}
    new = False
    for mult, r in runs.items():
        key = f"{Pt}_{mult}"
        if f"w_{key}" not in z:
            p = replace(po._params(Pt), P=float(r["P"]), kappa=float(r["kappa"]))
            w, nlp, L1, L3 = full_readin_theory(p)
            z.update({f"w_{key}": w, f"nlp_{key}": nlp, f"L1_{key}": L1, f"L3_{key}": L3})
            print(f"theory at P = {r['P']}, kappa = {r['kappa']:g} (theory point {Pt} x{mult}): L1 {L1:.4f} L3 {L3:.4f}",
                  flush=True)
            new = True
        r["theory"] = (z[f"w_{key}"], z[f"nlp_{key}"], float(z[f"L1_{key}"]), float(z[f"L3_{key}"]))
    if new:
        np.savez(CACHE_EKR, **z)


def ek_data():
    """{theory point: dict(theory=(w, nlp, L1, L3), runs={multiple: dict(P, kappa, w0, L1, L3)})}."""
    if CACHE_EK.exists():
        z = dict(np.load(CACHE_EK))
        th = {Pt: (z[f"w_{Pt}"], z[f"nlp_{Pt}"], float(z[f"L1_{Pt}"]), float(z[f"L3_{Pt}"])) for Pt in EK}
    else:
        th, save = {}, {}
        for Pt in EK:
            th[Pt] = full_readin_theory(po._params(Pt))
            save.update({f"w_{Pt}": th[Pt][0], f"nlp_{Pt}": th[Pt][1], f"L1_{Pt}": th[Pt][2], f"L3_{Pt}": th[Pt][3]})
            print(f"theory point P={Pt}: L1 {th[Pt][2]:.4f} L3 {th[Pt][3]:.4f}", flush=True)
        np.savez(CACHE_EK, **save)
    out = {}
    for Pt, (seed, pats) in EK.items():
        runs = {}
        for mult, pat in pats:
            for f in sorted(glob.glob(str(HERE / "runs" / pat / "config.json"))):
                a = json.loads(Path(f).read_text())["args"]
                if a["seed"] != seed or int(round(a["P"] * po.KAPPA / a["kappa"])) != Pt:
                    continue
                run = Path(f).parent
                sn = np.load(run / "snaps.npz")
                w0 = sn["w0"][sn["iter"] >= a["burn"]].ravel()
                L = json.loads((run / "summary.json").read_text())["posterior"]
                r = runs.setdefault(mult, dict(P=a["P"], kappa=a["kappa"], w0=[], L1=[], L3=[], paths=[]))
                r["paths"].append(run)
                r["w0"].append(w0)
                r["L1"].append(L["L1"][0])
                r["L3"].append(L["L3"][0])
        for r in runs.values():
            r["w0"] = np.concatenate(r["w0"])
        run_theory(Pt, runs)
        out[Pt] = dict(theory=th[Pt], runs=runs, seed=seed)
    return out


def fcn3_theory(device="cuda:0"):
    """-log p(w0) of the theory's read-in distribution on the full (w0, log q) grid at each ray point. The solver
    prunes grid points more than 60 nats below the peak, so its stored density stops there; here the read-in action
    lp0 - theta . T / 2 at the solution is evaluated on every grid point and marginalised over q."""
    if CACHE3.exists() and "L3_75" in np.load(CACHE3).files:
        z = dict(np.load(CACHE3))
        return {d: (z[f"w_{d}"], z[f"nlp_{d}"], float(z[f"L1_{d}"]), float(z[f"L3_{d}"])) for d in (75, 150, 300, 600)}
    import gc
    from dataclasses import replace
    import fcn3_fixed as F
    from fcn3_matrix_solver import DT, Params, trapz_weights
    out, save = {}, {}
    for d in (75, 150, 300, 600):
        a = [json.loads(Path(f).read_text())["args"]
             for f in glob.glob(str(HERE / "runs" / f"*_ray_d{d}_P*_N1500x1500_prior_r1" / "config.json"))][0]
        p = Params(d=d, P=float(a["P"]), N1=1500, N2=1500, chi=1.0, kappa=a["kappa"], eps=a["eps"], sigA=a["sigA"],
                   sigB=a["sigB"], sigw=a["sigw"], nmax=11)
        F3 = F.BlazingFreeEnergy(replace(p, nmax=3), device=device, verbose=False)
        o3 = F3.solve()
        Fn = F.BlazingFreeEnergy(p, device=device, verbose=False)
        th = torch.zeros(Fn.nth, dtype=DT, device=Fn.dev)
        th[:3] = F3.theta_ref[:3].to(Fn.dev)
        th[-1] = F3.theta_ref[-1].to(Fn.dev)
        o = Fn.solve(theta0=th, M0=o3["M"])
        with torch.no_grad():
            lw = (Fn.lp0.reshape(-1) - 0.5 * (Fn.theta_ref @ Fn.Tst)).reshape(Fn.ri.W.shape)
            lw = lw - torch.logsumexp(lw.reshape(-1), 0)
            lm = torch.logsumexp(lw, 1)                                       # log mass per w0 cell (w0 >= 0)
            nlp = -(lm - torch.log(trapz_weights(Fn.ri.w)) + math.log(0.5))   # -log density on the full line
        w = Fn.ri.w.cpu().numpy()
        out[d] = (w, nlp.cpu().numpy(), float(o["L1"]), float(o["L3"]))
        save.update({f"w_{d}": w, f"nlp_{d}": out[d][1], f"L1_{d}": out[d][2], f"L3_{d}": out[d][3]})
        print(f"FCN3 theory d={d}: L3 {o['L3']:.4f}", flush=True)
        del F3, Fn
        gc.collect()
        torch.cuda.empty_cache()
    np.savez(CACHE3, **save)
    return out


def fcn3_data():
    th3 = fcn3_theory()
    out = {}
    for d in (75, 150, 300, 600):
        w0, L = [], []
        pats = [f"*_ray_d{d}_P*_N1500x1500_*"] + ([f"*_onset_d150_P3000_N1500x1500_*"] if d == 150 else [])
        for pat in pats:
            for f in sorted(glob.glob(str(HERE / "runs" / pat / "config.json"))):
                run = Path(f).parent
                if "aborted" in run.name or run.name.endswith(("r71", "r72")):
                    continue
                a = json.loads(Path(f).read_text())["args"]
                if (run / "summary.json").exists():
                    ps = json.loads((run / "summary.json").read_text())["posterior"]
                    L.append((ps["L1"][0], ps["L3"][0]))
                if (run / "snaps.npz").exists():
                    s = np.load(run / "snaps.npz")
                    w0.append(s["w0"][s["iter"] >= a["burn"]].ravel())
                elif (run / "checkpoint.pt").exists():
                    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
                    w0.append((ck["ZW"][:, 0].double() * a["sigw"] / math.sqrt(d)).numpy())
        w, nlp = th3[d][:2]
        k = w <= 1.6
        out[d] = dict(w0=np.concatenate(w0), wq=np.concatenate([-w[k][::-1], w[k][1:]]),
                      th=np.concatenate([nlp[k][::-1], nlp[k][1:]]), L1th=th3[d][2], L3th=th3[d][3], L=L)
        print(f"FCN3 d={d}: {sum(len(x) for x in w0)} read-in neurons pooled", flush=True)
    return out


# along the ray: test MSE / training MSE and He3 residual on the training set / in the population, at d = 150, 300, 600
# (ray_overfit_check.py; FINDINGS_onset.md)
OVERFIT = {150: (2.75, 0.60), 300: (1.67, 0.76), 600: (1.39, 0.81)}


def fcn3_wide_runs():
    """(d, L1, L3) of the FCN3 ray runs at N1 = N2 = 3000 (chi = 2, theory unchanged) with a summary."""
    out = []
    for f in sorted(glob.glob(str(HERE / "runs" / "*_ray_d*_N3000x3000_*" / "config.json"))):
        sm = Path(f).with_name("summary.json")
        if sm.exists():
            a = json.loads(Path(f).read_text())["args"]
            ps = json.loads(sm.read_text())["posterior"]
            out.append((a["d"], ps["L1"][0], ps["L3"][0]))
    return out


def well_stats_theory(wq, th):
    """Theory: barrier |w0| between the wells, outer-well mean |w0| and sd (density beyond the barrier, |w0| < 1.5)."""
    w, y = np.abs(wq), th
    o = np.argsort(w)
    w, y = w[o], y[o]
    ko = (w > 0.3) & (w < 1.5)
    wo = w[ko][np.argmin(y[ko])]
    kb = (w > 0.1) & (w < wo)
    wb = w[kb][np.argmax(y[kb])]
    p = np.exp(-(y - y.min())) * np.gradient(w)
    oo = (w > wb) & (w < 1.5)
    mo = (p[oo] * w[oo]).sum() / p[oo].sum()
    return wb, mo, np.sqrt((p[oo] * (w[oo] - mo) ** 2).sum() / p[oo].sum())


def panels_wells(axs, d2, d3, plt):
    """Outer read-in well along the ray: position, sampler - theory, width; FCN2 and FCN3. Sampler outer well = |w0|
    beyond the theory's barrier. FCN3 d = 75 and 600: final networks of two runs only (open markers)."""
    rows = {}
    for name, dd in (("FCN2", d2), ("FCN3", d3)):
        for d, r in sorted(dd.items()):
            wb, mo, so = well_stats_theory(r["wq"], r["th"])
            x = np.abs(r["w0"])
            o = x[(x > wb) & (x < 1.5)]
            rows[(name, d)] = (mo, so, o.mean(), o.std(), len(o))
    for name, ls, mk, col in (("FCN2", "--", "s", po.MUTED), ("FCN3", "-", "o", po.INK)):
        ds = sorted(d for (n, d) in rows if n == name)
        th_m = [rows[(name, d)][0] for d in ds]
        th_s = [rows[(name, d)][1] for d in ds]
        axs[0].plot(ds, th_m, ls, color=col, lw=1.8, zorder=1, label=f"{name}, theory")
        axs[2].plot(ds, th_s, ls, color=col, lw=1.8, zorder=1, label=f"{name}, theory")
        gaps = []
        for d in ds:
            tm, ts, sm, ss, n = rows[(name, d)]
            few = name == "FCN3" and n < 1000
            kw = dict(color=colour(d, plt), ms=7, mew=1.2, zorder=3)
            kw.update(mfc="white", mec=colour(d, plt)) if few else kw.update(mec="white")
            axs[0].plot([d], [sm], mk, **kw)
            axs[2].plot([d], [ss], mk, **kw)
            axs[1].plot([d], [sm - tm], mk, **kw)
            gaps.append(sm - tm)
        axs[1].plot(ds, gaps, ls, color=col, lw=1.2, zorder=2, label=name)
    for ax in axs:
        ax.plot([], [], "s", color=po.MUTED, ms=7, label="FCN2, sampler")
        ax.plot([], [], "o", color=po.MUTED, ms=7, label="FCN3, sampler")
        ax.plot([], [], "o", mfc="white", mec=po.MUTED, ms=7, label="FCN3, two final networks only")
        ax.set_xscale("log")
        ax.set_xticks([75, 150, 300, 600, 1200])
        ax.set_xticklabels(["75", "150", "300", "600", "1200"])
        ax.minorticks_off()
        ax.set_xlabel("d")
    axs[1].axhline(0, color=po.MUTED, lw=0.8, ls=":")
    axs[0].set_title("outer read-in well: mean |w$_0$|", color=po.INK, loc="left", fontsize=12)
    axs[1].set_title("outer well position: sampler − theory", color=po.INK, loc="left", fontsize=12)
    axs[2].set_title("outer read-in well: width (sd of |w$_0$|)", color=po.INK, loc="left", fontsize=12)
    axs[2].set_ylim(0, None)
    axs[0].legend(fontsize=8, loc="lower left")
    axs[1].legend(fontsize=8, loc="lower right")


def panel_residents(ax, plt, lo=0.3, hi=0.85):
    """d = 150, P = 3000: |w0| against iteration for every read-in neuron that is in the barrier region
    (lo < |w0| < hi) in more than 80 % of the post-burn snapshots, coloured by its perpendicular norm q / q_prior.
    Runs with snapshots: ray d150 and onset P3000 (identical trajectories counted once)."""
    import matplotlib as mpl
    runs = [Path(f).parent for f in sorted(glob.glob(str(HERE / "runs" / "*_ray_d150_P*_N1500x1500_*" / "config.json")))
            + sorted(glob.glob(str(HERE / "runs" / "*_onset_d150_P3000_N1500x1500_*" / "config.json")))]
    runs = [r for r in runs if (r / "snaps.npz").exists() and "aborted" not in r.name and not r.name.endswith(("r71", "r72"))]
    q0 = SW2 * 149 / 150
    norm = mpl.colors.Normalize(0.4, 1.6)
    cmap = plt.cm.viridis
    seen, n = [], 0
    for r in runs:
        a = json.loads((r / "config.json").read_text())["args"]
        s = np.load(r / "snaps.npz")
        post = s["iter"] >= a["burn"]
        W, Q, it = np.abs(s["w0"][post]), s["q"][post] / q0, s["iter"][post]
        for j in np.where(((W > lo) & (W < hi)).mean(0) > 0.8)[0]:
            if any(np.allclose(W[:, j], x) for x in seen):
                continue
            seen.append(W[:, j])
            n += 1
            pts = np.stack([it, W[:, j]], 1)
            seg = np.stack([pts[:-1], pts[1:]], 1)
            lc = mpl.collections.LineCollection(seg, cmap=cmap, norm=norm, lw=1.6)
            lc.set_array(0.5 * (Q[:-1, j] + Q[1:, j]))
            ax.add_collection(lc)
    w, nlp = fcn3_theory()[150][:2]
    ko = (w > 0.4) & (w < 1.2)
    wo = w[ko][np.argmin(nlp[ko])]
    kb = (w > 0.1) & (w < wo)
    wb = w[kb][np.argmax(nlp[kb])]
    for y, lab in ((wo, "theory: outer-well bottom"), (wb, "theory: barrier top")):
        ax.axhline(y, color=po.MUTED, lw=0.9, ls="--")
        ax.text(305, y + 0.01, lab, color=po.MUTED, fontsize=7.5, ha="left", va="bottom",
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    ax.axhspan(lo, hi, color=po.GRID, alpha=0.5, zorder=0)
    ax.set_xlim(300, 1000)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("HMC iteration (after burn-in)")
    ax.set_ylabel("|w$_0$| of the neuron")
    ax.set_title(f"{n} neurons staying between the wells (d = 150)", color=po.INK, loc="left", fontsize=11)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
    cb.set_label("q / q$_{prior}$  (inner well 1.0, outer well 0.46)", fontsize=8)


def panel_overfit(ax):
    ds = sorted(OVERFIT)
    ax.axhline(1, color=po.MUTED, lw=1, ls=":")
    ax.plot(ds, [OVERFIT[d][0] for d in ds], "o-", color=po.INK, lw=1.8, ms=6, label="test MSE / training MSE")
    ax.plot(ds, [OVERFIT[d][1] for d in ds], "s--", color=po.MUTED, lw=1.8, ms=6,
            label="He$_3$ residual: training set / population")
    ax.set_xscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.minorticks_off()
    ax.set_ylim(0, 3)
    ax.set_xlabel("d")
    ax.set_title("FCN3 overfitting along the ray (→ 1)", color=po.INK, loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")


def panel_ek_gap(ax, ek):
    cols = {3000: "#9cc3ee", 12000: "#2a78d6", 24000: "#0b3d91"}
    for Pt, e in ek.items():
        L3t = e["theory"][3]
        m = sorted(e["runs"])
        ax.plot(m, [e["runs"][k]["theory"][3] - np.mean(e["runs"][k]["L3"]) for k in m], "o-", color=cols[Pt], lw=2, ms=6,
                label=f"theory point P = {Pt} (dataset {e['seed']})")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["P × 1", "× 2", "× 4"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.1)
    ax.set_ylabel("L$_3$ theory − sampler")
    ax.set_title("L$_3$ gap at fixed theory, κ ∝ P", color=po.INK, loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")


def panel_ek_h3(ax, ek, Pt=24000):
    wide = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_hidden_wide.npz"))
    g, y = wide[f"h3g_{Pt}"], -wide[f"lp3_{Pt}"]
    lim = 0.45
    k = np.abs(g) <= lim
    ax.plot(g[k], y[k], "-", color=po.C_THEORY, lw=2, zorder=2, label="theory")
    lo = y[k].min()
    edges = np.linspace(-lim, lim, 61)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for mult, r in sorted(ek[Pt]["runs"].items()):
        H, n = po.sampler_hidden([dict(path=q) for q in r["paths"]])
        far = np.abs(H[:, 0]) > 1.5
        big = (np.abs(H[:, 1]) > 0.25) & ~far
        c, _ = np.histogram(H[:, 1], edges)
        kk = c > 0
        yy = -np.log(c[kk] / (c.sum() * np.diff(edges)[kk]))
        ax.plot(mid[kk], yy, "o", color=EK_COLS[mult], ms=3.6, zorder=3,
                label=f"P × {mult}: {big.sum() / n:.0f} neurons per network with |h$_3$| > 0.25")
        lo = min(lo, yy.min())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lo - 0.5, lo + 12)
    ax.set_xlabel("h | He$_3$")
    ax.set_ylabel("−log p")
    ax.set_title(f"hidden layer at fixed theory P = {Pt}", color=po.INK, loc="left", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper center")


def panel_langevin(ax, th150):
    import torch as _t
    hmc = HERE / "runs" / "20261109-173149_onset_d150_P3000_N1500x1500_prior_r10"
    a = json.loads((hmc / "config.json").read_text())["args"]
    sn = np.load(hmc / "snaps.npz")
    wh = sn["w0"][sn["iter"] >= a["burn"]].ravel()
    lr = sorted(q for q in (HERE / "runs" / "langevin").glob("*from_hmc") if q.is_dir())[-1]
    fs = [f for f in sorted(lr.glob("nets/net_*.pt")) if int(f.name.split("_")[1]) >= 500000]
    wl = np.concatenate([(_t.load(f, weights_only=False)["ZW"][:, 0] * _t.load(f, weights_only=False)["sw"]).numpy()
                         for f in fs])
    w, nlp = th150[0], th150[1]
    k = w <= 1.3
    ax.plot(np.concatenate([-w[k][::-1], w[k][1:]]), np.concatenate([nlp[k][::-1], nlp[k][1:]]), "-",
            color=po.C_THEORY, lw=2, zorder=1, label="theory")
    edges = np.linspace(-1.3, 1.3, 61)
    mid = 0.5 * (edges[1:] + edges[:-1])
    lo = np.inf
    for x, lab, col, mk in ((wh, "HMC, every post-burn snapshot", po.C_HMC, "o"),
                            (wl, "Langevin from HMC states, steps 5·10$^5$–10$^6$", "#1baf7a", "s")):
        c, _ = np.histogram(x, edges)
        kk = c > 0
        yy = -np.log(c[kk] / (c.sum() * np.diff(edges)[kk]))
        ax.plot(mid[kk], yy, mk, color=col, ms=4, mec="white", mew=0.3, zorder=3, label=lab)
        lo = min(lo, yy.min())
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(lo - 1, lo + 42)
    ax.set_xlabel("w$_0$")
    ax.set_title("HMC against Langevin, d = 150, P = 3000", color=po.INK, loc="left", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper center")


def panel_counts(ax, plt):
    z = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    Ps = [3000, 6000, 12000, 24000, 48000]
    cm = plt.cm.Blues
    groups = po.by_P(po.sampler_runs())
    for i, P in enumerate(Ps):
        cut = po.barrier_cut(z, P)
        nth = po.outer_theory(z, P, cut) * po.N
        col = cm(0.35 + 0.6 * i / (len(Ps) - 1))
        for j, r in enumerate([q for q in groups[P] if "onset" in q["path"].name]):
            sn = np.load(r["path"] / "snaps.npz")
            n = (np.abs(sn["w0"]) > cut).sum(1)
            ax.plot(sn["iter"], n / nth, "-", color=col, lw=1.3, label=f"P = {P} (theory {nth:.0f})" if j == 0 else None)
    ax.axhline(1, color=po.MUTED, lw=1, ls=":")
    ax.axvline(300, color=po.MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, 2.3)
    ax.set_xlabel("HMC iteration (burn-in ends at 300)")
    ax.set_ylabel("outer read-in count / theory's")
    ax.set_title("outer read-in count, every run (d = 150)", color=po.INK, loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)


def panel_coupling(ax):
    """One P = 48000 network (dataset 0, last saved network): each hidden neuron's |h3| against the rms of its weights
    onto the outer read-in neurons, sampler against the comparison case of coupling_width.py (hidden neurons
    independent, each one's weights drawn from the prior given its own h on He1..He11 of x0: what the theory assumes).
    Run `python coupling_width.py --P 48000` first."""
    cache = json.loads((HERE / "plots" / "cache" / "coupling_width_P48000.json").read_text())
    a = [v for k, v in cache.items() if k.endswith("r10")][0]
    h3, rms = (np.array(x) for x in a["scatter_sampler"])
    h3c, rmsc = (np.array(x) for x in a["scatter_comparison"])
    big = h3 > 0.3
    ax.scatter(h3c, rmsc, s=9, facecolor="none", edgecolor=po.C_THEORY, lw=0.6, alpha=0.6, zorder=2,
               label="independent neurons with the same h (what the theory assumes)")
    ax.scatter(h3[~big], rms[~big], s=5, color=po.MUTED, alpha=0.5, zorder=3, label="sampler, hidden neurons")
    ax.scatter(h3[big], rms[big], s=22, color=po.C_HMC, edgecolor="white", lw=0.4, zorder=4,
               label=f"sampler, the {big.sum()} with |h$_3$| > 0.3")
    s, c = a["sampler"], a["comparison"]
    ax.axhline(1, color=po.MUTED, lw=0.8, ls=":")
    ax.set_xlabel("|h | He$_3$| of the hidden neuron")
    ax.set_ylabel("rms weight onto outer read-in neurons (prior units)")
    ax.set_ylim(0, None)
    ax.set_title("weights onto outer read-in neurons, P = 48000", color=po.INK, loc="left", fontsize=11)
    ax.legend(fontsize=7, loc="upper left",
              title=f"sampler / independent neurons, mean over {a['n_nets']} saved networks:\n"
                    f"correlation of |h$_3$| with rms weight {s['corr']:.2f} / {c['corr']:.2f};  outer neurons' "
                    f"outgoing columns\nagainst the other outer ones' sum: cos {s['cos_out']:.3f} / "
                    f"{c['cos_out']:.3f};  the same seen\nfor inner neurons: {s['cos_in']:.3f} / {c['cos_in']:.3f}",
              title_fontsize=6.5)


ONSET_CACHE = HERE / "plots" / "cache" / "onset_overlay_d150.npz"


def onset_data():
    """The data of plot_onset.fig_per_P (onset group, d = 150, kappa = 0.015, N = 1500): per P the sampler's read-in w0
    (every post-burn snapshot) and h|He1, h|He3 of every hidden neuron of every saved network, all datasets pooled;
    the theory from plot_onset's cache. Sampler part cached in ONSET_CACHE, redone when the set of runs changes."""
    groups = po.by_P(po.sampler_runs())
    key = ";".join(f"{r['run']}:{r['last']}" for rs in groups.values() for r in rs)
    z = dict(np.load(ONSET_CACHE)) if ONSET_CACHE.exists() else {}
    if str(z.get("key", "")) != key:
        z = {"key": np.array(key)}
        for P, rs in groups.items():
            z[f"w0_{P}"] = np.concatenate([r["w0"] for r in rs if r["w0"] is not None])
            z[f"h_{P}"] = po.sampler_hidden([r for r in rs if (r["path"] / "nets").exists()])[0]
        np.savez(ONSET_CACHE, **z)
    th = dict(np.load(HERE / "plots" / "cache" / "theory_onset_d150_v6.npz"))
    return list(groups), z, th


def panels_onset(axs, plt):
    """Last row: the per-P actions of plots/onset/onset_d150_actions.png overlaid, colour = P. Read-in at full height
    (up to the tallest theory barrier), h|He1 and h|He3 of the second hidden layer. Lines: theory; points: sampler
    (histogram, no error bars)."""
    Ps, z, th = onset_data()
    cm = plt.cm.viridis
    col = {P: cm(0.9 * (math.log2(P) - math.log2(Ps[0])) / (math.log2(Ps[-1]) - math.log2(Ps[0]))) for P in Ps}
    tops = {0: [], 1: [], 2: []}
    lims = {0: 1.5, 1: 0.0, 2: 0.0}
    for P in Ps:
        c = col[P]
        # read-in, full grid
        w, dn = th[f"w0_{P}"], th[f"dens_{P}"]
        k = (dn > 1e-300) & (w > 0)
        xs = np.concatenate([-w[k][::-1], w[k]])
        ys = -np.log(0.5 * np.concatenate([dn[k][::-1], dn[k]]))
        mins = [xs[j] for j in range(1, len(ys) - 1) if ys[j] < ys[j - 1] and ys[j] < ys[j + 1] and ys[j] < ys.min() + 12]
        lim = max([1.5, 1.1 * np.abs(z[f"w0_{P}"]).max()] + [1.15 * abs(m) for m in mins])
        kk = np.abs(xs) <= lim
        axs[0].plot(xs, ys, color=c, lw=1.8, zorder=2, label=f"P = {P}")
        tops[0] += [ys[kk][m] for m in range(1, kk.sum() - 1) if ys[kk][m] > ys[kk][m - 1] and ys[kk][m] > ys[kk][m + 1]]
        lims[0] = max(lims[0], lim)
        series = [(axs[0], z[f"w0_{P}"], lim, 61, None)]
        # hidden layer, He1 and He3
        for i, j in ((1, 1), (2, 3)):
            x = z[f"h_{P}"][:, i - 1]
            g, yt = th[f"h{j}g_{P}"], -th[f"lp{j}_{P}"]
            lim = 1.08 * max(np.abs(x).max(), np.abs(g[yt <= yt.min() + 9]).max())
            axs[i].plot(g, yt, color=c, lw=1.8, zorder=2)
            lims[i] = max(lims[i], lim)
            tops[i].append(max(np.interp(np.abs(x).max(), g, yt), np.interp(-np.abs(x).max(), g, yt)) - yt.min())
            series.append((axs[i], x, lim, 81, None))
        for ax, x, lim, nb, _ in series:
            edges = np.linspace(-lim, lim, nb)
            mid = 0.5 * (edges[1:] + edges[:-1])
            cnt, _ = np.histogram(x, edges)
            k = cnt > 0
            ax.plot(mid[k], -np.log(cnt[k] / (cnt.sum() * np.diff(edges)[k])), "o", color=c, ms=3.2, mec="white",
                    mew=0.3, zorder=3)
    axs[0].set_xlim(-lims[0], lims[0])
    axs[0].set_ylim(-3, max(tops[0]) + 3)
    axs[0].set_xlabel("w$_0$")
    axs[0].set_ylabel("−log p")
    axs[0].set_title("read-in layer, full height", color=po.INK, loc="left", fontsize=11)
    axs[1].legend(*axs[0].get_legend_handles_labels(), fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=7, title="colour = P (d = 150, κ = 0.015, N = 1500);  lines: theory, points: sampler (all "
                  "datasets pooled)", title_fontsize=9)
    for i, xl, name in ((1, "h | He$_1$", "hidden layer, He$_1$"), (2, "h | He$_3$", "hidden layer, He$_3$")):
        axs[i].set_xlim(-lims[i], lims[i])
        axs[i].set_ylim(-3, max(10, min(max(tops[i]), 25)) + 1)
        axs[i].set_xlabel(xl)
        axs[i].set_title(name, color=po.INK, loc="left", fontsize=11)


def main():
    d2, d3 = fcn2_data(), fcn3_data()
    l2 = fcn2_learn()
    plt = po.style()
    ek = ek_data()
    # the P = 3000 point is the ray point d = 150: use one solve for both. (Two solves of the same theory agree in L3,
    # the wells and the flanks to < 0.4 nats, but their barrier tops differ by ~6 nats, 38.9 against 32.6: the
    # barrier carries no mass and is set by near-singular directions of the hidden prior; treat it as ±15 %.)
    ek[3000]["theory"] = fcn3_theory()[150]
    fig = plt.figure(figsize=(17, 40))
    gs = fig.add_gridspec(7, 6, height_ratios=[1.25, 0.9, 0.9, 1.1, 1.0, 1.0, 1.1], hspace=0.42, wspace=0.55)
    axa = [fig.add_subplot(gs[0, :3])]
    axa.append(fig.add_subplot(gs[0, 3:], sharey=axa[0]))
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    lo, hi = np.inf, -np.inf
    for ax, (name, dd) in zip(axa, (("two-layer network (FCN2)", d2), ("three-layer network (FCN3)", d3))):
        for d, r in sorted(dd.items()):
            beta = d / (2 * SW2)
            col = colour(d, plt)
            ax.plot(r["wq"], r["th"] / beta, "-", color=col, lw=2, zorder=2, label=f"d = {d}")
            c, _ = np.histogram(r["w0"], EDGES)
            k = c > 0
            y = -np.log(c[k] / (c.sum() * np.diff(EDGES)[k])) / beta
            ax.plot(mid[k], y, "o", color=col, ms=3.5, mec="white", mew=0.4, zorder=3)
            lo = min(lo, y.min(), (r["th"] / beta).min())
            hi = max(hi, (r["th"][np.abs(r["wq"]) < 1.0] / beta).max())
        ax.set_title(name, color=po.INK, loc="left", fontsize=12)
        ax.set_xlabel("w$_0$")
        ax.set_xlim(-1.6, 1.6)
        ax.legend(title="lines: theory, points: sampler", fontsize=9, title_fontsize=9, loc="upper center", ncol=3)
    axa[1].tick_params(labelleft=False)
    axa[0].set_ylabel("−log p(w$_0$) / β,   β = d / (2σ$_w^2$)")
    axa[0].set_ylim(lo - 0.02, 1.15 * hi)

    # learnabilities against d: colour = d as above; marker / line style = network
    for j, (key, lab) in enumerate(((0, "L$_1$"), (1, "L$_3$"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        ds2, ds3 = sorted(l2), sorted(d3)
        ax.plot(ds2, [l2[d][key] for d in ds2], "--", color=po.MUTED, lw=1.8, zorder=1, label="FCN2, theory")
        ax.plot(ds3, [d3[d]["L1th" if key == 0 else "L3th"] for d in ds3], "-", color=po.INK, lw=1.8, zorder=1,
                label="FCN3, theory")
        for d in ds2:
            v = [x[key] for x in l2[d][2]]
            ax.plot([d] * len(v), v, "s", color=colour(d, plt), ms=7, mec="white", mew=0.5, zorder=3)
        for d in ds3:
            v = [x[key] for x in d3[d]["L"]]
            ax.plot([d] * len(v), v, "o", color=colour(d, plt), ms=7, mec="white", mew=0.5, zorder=3)
        wide = fcn3_wide_runs()
        if wide:
            ax.plot([x[0] for x in wide], [x[1 + key] for x in wide], "D", color="#0b3d91", ms=6, mec="white",
                    zorder=4, label="FCN3, sampler at N = 3000 (χ = 2)")
        ax.plot([], [], "s", color=po.MUTED, ms=7, label="FCN2, sampler (one point per chain)")
        ax.plot([], [], "o", color=po.MUTED, ms=7, label="FCN3, sampler (one point per run)")
        ax.set_xscale("log")
        ax.set_xticks([75, 150, 300, 600, 1200])
        ax.set_xticklabels(["75", "150", "300", "600", "1200"])
        ax.minorticks_off()
        ax.set_xlabel("d")
        ax.set_title(f"{lab} along the ray", color=po.INK, loc="left", fontsize=12)
        ax.set_ylim(0, 1.15 if key == 0 else 1)
        if key == 1:
            ax.legend(fontsize=8.5, loc="upper left")
    panel_overfit(fig.add_subplot(gs[1, 4:]))
    # third row: the outer read-in well's position and width against d
    panels_wells([fig.add_subplot(gs[2, 2 * i:2 * i + 2]) for i in range(3)], d2, d3, plt)
    # third row: the theory held fixed at d = 150 while P is doubled twice with kappa ∝ P
    edges = np.linspace(-1.3, 1.3, 61)
    mid3 = 0.5 * (edges[1:] + edges[:-1])
    axe = [fig.add_subplot(gs[3, 2 * i:2 * i + 2]) for i in range(3)]
    for ax, (Pt, e) in zip(axe, ek.items()):
        w, nlp, L1t, L3t = e["theory"]
        k = w <= 1.3
        ax.plot(np.concatenate([-w[k][::-1], w[k][1:]]), np.concatenate([nlp[k][::-1], nlp[k][1:]]), "-",
                color=po.C_THEORY, lw=2, zorder=2, label=f"theory at P = {Pt}, κ = 0.015: L$_3$ {L3t:.3f}")
        lo3 = nlp[k].min()
        for mult, r in sorted(e["runs"].items()):
            if mult > 1:                                   # the theory at this run's own (P, kappa)
                wr, nr, _, L3r = r["theory"]
                kr = wr <= 1.3
                ax.plot(np.concatenate([-wr[kr][::-1], wr[kr][1:]]), np.concatenate([nr[kr][::-1], nr[kr][1:]]), "--",
                        color=EK_COLS[mult], lw=1.4, zorder=2,
                        label=f"theory at P = {r['P']}, κ = {r['kappa']:g}: L$_3$ {L3r:.3f}")
            c, _ = np.histogram(r["w0"], edges)
            kk = c > 0
            y = -np.log(c[kk] / (c.sum() * np.diff(edges)[kk]))
            ax.plot(mid3[kk], y, "o", color=EK_COLS[mult], ms=3.6, zorder=3,
                    label=f"sampler P = {r['P']} (×{mult}): L$_3$ " + ", ".join(f"{x:.3f}" for x in r["L3"]))
            lo3 = min(lo3, y.min())
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(lo3 - 1, lo3 + 42)
        ax.set_xlabel("w$_0$")
        if Pt == 3000:
            ax.set_ylabel("−log p(w$_0$)   (d = 150)")
        ax.set_title(f"theory held at P = {Pt} (dataset {e['seed']})", color=po.INK, loc="left", fontsize=11)
        ax.legend(fontsize=7, loc="upper center", title="points near 9 nats are single counts", title_fontsize=7)
    # fourth row: more data at fixed theory, summarised; and the sampler check
    panel_ek_gap(fig.add_subplot(gs[4, :2]), ek)
    panel_ek_h3(fig.add_subplot(gs[4, 2:4]), ek)
    panel_langevin(fig.add_subplot(gs[4, 4:]), fcn3_theory()[150])
    # fifth row: the outer read-in count is not equilibrated; the two layers are coupled
    panel_counts(fig.add_subplot(gs[5, :2]), plt)
    panel_coupling(fig.add_subplot(gs[5, 2:4]))
    panel_residents(fig.add_subplot(gs[5, 4:]), plt)
    # seventh row: the onset actions (plots/onset/onset_d150_actions.png) overlaid over P
    panels_onset([fig.add_subplot(gs[6, 2 * i:2 * i + 2]) for i in range(3)], plt)
    fig.suptitle("Evidence on the convergence of the FCN3 theory. Rows 1–3: along the κ$_0$ = 0.015 ray (P = 3000 (d/150)$^2$, "
                 "κ = 0.015 d/150, N = 1500).\nRows 4–5: theory held fixed at d = 150 while P doubles with κ ∝ P, and the "
                 "HMC–Langevin check. Row 6: the outer read-in count, the coupling between layers, and the neurons between the wells.\n"
                 "Row 7: the actions as P grows at d = 150, κ = 0.015 fixed (onset runs), all P overlaid.",
                 color=po.INK, x=0.01, ha="left", fontsize=13, y=0.91)
    out = HERE / "plots" / "ray" / "readin_ray_fcn2_vs_fcn3.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    for d in sorted(set(l2) | set(d3)):
        f2 = l2.get(d); f3 = d3.get(d)
        print(f"d={d}: FCN2 L1/L3 theory {f2[0]:.3f}/{f2[1]:.3f} sampler {np.round(np.mean(f2[2], 0), 3).tolist()}" if f2 else f"d={d}:",
              f";  FCN3 theory {f3['L1th']:.3f}/{f3['L3th']:.3f} sampler {np.round(np.mean(f3['L'], 0), 3).tolist()}" if f3 else "")


if __name__ == "__main__":
    main()
