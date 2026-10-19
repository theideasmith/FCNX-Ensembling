#!/usr/bin/env python3
"""On-data particle theory (wall-free cavity continuum) for PureCubicChiNRhoRay.

Same dynamics as wall_free_cavity_continuum.py / Section 9.6:
  one continuum of M particles for all N neurons (weight N/M), own term removed,
  kernel rebuilt every step, no wall between wells. Samples e^{-U} on the actual
  training set with χ-aware σ_A² = sa0/(Nχ) and κ = T_eff/2 = T/(2χ).

Usage:
  python -u wall_free_cavity_pure_cubic_rho.py --rho 0.015
  python -u wall_free_cavity_pure_cubic_rho.py --all --steps 40000 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import plot_action_h0_activation as PA
from precond_langevin_warmstart import recover_training_set, he3

HERE = Path(__file__).resolve().parent
GROUP = "PureCubicChiNRhoRay"
OUT_DIR = HERE / "action_h0_activation_plots" / GROUP / "particle_theory"
_RHO_RE = re.compile(r"purecubic_rho(?P<rho>[\d.eE+-]+)_")
W_CUT_DEFAULT = 1.0  # specialists sit at |w|~2–3 on this ray; 0.6 mixes bulk shoulder


def he_proj(X, f):
    z = X[:, 0]
    H = torch.stack([z, (z ** 3 - 3 * z) / np.sqrt(6.0)], 1)
    return (H.T @ f / X.shape[0]).cpu().numpy()


def peak(a, hi):
    h, e = np.histogram(a, bins=60, range=(0, hi))
    if h.sum() == 0:
        return float("nan")
    return 0.5 * (e[h.argmax()] + e[h.argmax() + 1])


def summarize(W, X, y, kappa, sig_a2, wt, w_cut, DEV):
    o = W[:, 0].abs() > w_cut
    Phi = torch.erf(X @ W.T)
    K_in = sig_a2 * wt * Phi[:, ~o] @ Phi[:, ~o].T if (~o).any() else 0.0
    K_out = sig_a2 * wt * Phi[:, o] @ Phi[:, o].T if o.any() else 0.0
    R = torch.cholesky_inverse(
        torch.linalg.cholesky(K_in + K_out + kappa * torch.eye(X.shape[0], device=DEV, dtype=torch.float64))
    )
    t = R @ y
    r = he_proj(X, kappa * t)
    fb = he_proj(X, K_in @ t) if (~o).any() else np.zeros(2)
    fs = he_proj(X, K_out @ t) if o.any() else np.zeros(2)
    wi, ws = W[~o].cpu().numpy(), W[o].cpu().numpy()
    return dict(
        f_out=float(o.double().mean()),
        in_mean=float(np.abs(wi[:, 0]).mean()) if len(wi) else float("nan"),
        in_peak=peak(np.abs(wi[:, 0]), w_cut) if len(wi) else float("nan"),
        in_p2=float((wi[:, 1:] ** 2).sum(1).mean()) if len(wi) else float("nan"),
        sp_med=float(np.median(np.abs(ws[:, 0]))) if len(ws) else float("nan"),
        sp_p2=float((ws[:, 1:] ** 2).sum(1).mean()) if len(ws) else float("nan"),
        r=r, fb=fb, fs=fs,
    )


def fmt(tag, o, N):
    return (
        f"{tag}: outer fraction {o['f_out']:.3f} ({o['f_out'] * N:.0f}); "
        f"bulk mean |w0| {o['in_mean']:.3f}, peak {o['in_peak']:.3f}, |w_perp|^2 {o['in_p2']:.3f}; "
        f"specialists median |w0| {o['sp_med']:.2f}, |w_perp|^2 {o['sp_p2']:.2f}; "
        f"discrepancy (He1, He3) ({o['r'][0]:+.4f}, {o['r'][1]:+.4f}); "
        f"output bulk ({o['fb'][0]:+.3f}, {o['fb'][1]:+.3f}), specialists ({o['fs'][0]:+.3f}, {o['fs'][1]:+.3f})"
    )


def select_run(rho: float) -> Path:
    dirs = [Path(p) for p in PA.EXPERIMENT_GROUP_BY_NAME[GROUP].model_dirs]
    cands = []
    for p in dirs:
        m = _RHO_RE.match(p.name)
        if m:
            cands.append((abs(float(m.group("rho")) - rho), float(m.group("rho")), p))
    if not cands:
        raise SystemExit(f"no runs in {GROUP}")
    cands.sort()
    return cands[0][2]


def run_particles(run: Path, *, steps, M, starts, f_start, w_start, w_cut, device, every):
    c = json.load(open(run / "config.json"))
    d, P, N = int(c["d"]), int(c["P"]), int(c["N"])
    eps, he1 = float(c["eps"]), float(c.get("he1", 0.0))
    sa0, s0 = float(c["sa0"]), float(c["s0"])
    chi = float(c.get("chi", 1.0))
    T_bare = float(c["temperature"])
    T_eff = T_bare / chi
    sig_a2 = sa0 / (N * chi)
    kappa = T_eff / 2.0
    rho = (T_bare / 2.0) / (sa0 * P)

    rec = json.load(open(run / "pred_vs_true.json"))
    y_np = np.asarray(rec[max(rec, key=int)]["y_true"], dtype=float)
    X_np, seed, data_dev = recover_training_set(P, d, y_np, eps, he1=he1)
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    I = torch.eye(P, device=device, dtype=torch.float64)

    # network reference (frozen A_snapshots — not equilibrated, for comparison only)
    snaps = sorted((run / "A_snapshots").glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    use = snaps[len(snaps) // 2 :]
    w_net = []
    for sp in use:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        w_net.append(sd["W0"].double().reshape(-1, d).numpy()[:, 0])
    w_net = np.concatenate(w_net)
    f_net = float(np.mean(np.abs(w_net) > w_cut))

    h = 0.01 / d
    wt = N / M
    dt = torch.float32 if P > 500 else torch.float64
    Xc, yc = X.to(dt), y.to(dt)

    print(
        f"\n=== ρ={rho:g}  {run.name} ===\n"
        f"d={d} P={P} N={N} χ={chi:g} σ_A²={sig_a2:.3e} κ={kappa:.4g} T_eff={T_eff:.4g} "
        f"M={M} wt={wt:.3f} steps={steps} W_CUT={w_cut} device={device}\n"
        f"train set seed={seed} on {data_dev}; network fo(|w0|>{w_cut})={f_net:.3f} "
        f"(A_snapshots 2nd half, not equilibrated)",
        flush=True,
    )

    results = {}
    for start in starts:
        torch.manual_seed(0)
        W = torch.randn(M, d, device=device, dtype=dt) * np.sqrt(s0 / d)
        if start == "outer":
            k = int(f_start * M)
            W[:k, 0] = torch.sign(torch.randn(k, device=device, dtype=dt)) * w_start
        traj = []
        for it in range(steps):
            with torch.no_grad():
                Phi_ref = torch.erf(Xc @ W.T)
                R = torch.cholesky_inverse(
                    torch.linalg.cholesky(sig_a2 * wt * (Phi_ref @ Phi_ref.T).double() + kappa * I)
                ).to(dt)
                t = R @ yc
                u = R @ Phi_ref
                den = 1 - sig_a2 * (Phi_ref * u).sum(0)
                at = Phi_ref.T @ t
            W.requires_grad_(True)
            Phi = torch.erf(Xc @ W.T)
            pu = (Phi * u).sum(0)
            s = (Phi * (R @ Phi)).sum(0) + sig_a2 * pu ** 2 / den
            b = t @ Phi + sig_a2 * pu * at / den
            x = sig_a2 * s
            U = (d / (2 * s0)) * (W ** 2).sum(1) + 0.5 * torch.log1p(x) - 0.5 * sig_a2 * b ** 2 / (1 + x)
            g, = torch.autograd.grad(U.sum(), W)
            with torch.no_grad():
                W = W - h * g + np.sqrt(2 * h) * torch.randn_like(W)
            if (it + 1) % every == 0:
                fo = float((W[:, 0].abs() > w_cut).double().mean())
                traj.append((it + 1, fo))
                med = float(W[W[:, 0].abs() > w_cut, 0].abs().median()) if fo > 0 else float("nan")
                print(f"  [{start}] step {it + 1}: outer frac {fo:.3f}, outer median |w0| {med:.2f}", flush=True)

        Wf = W.detach().double()
        o = summarize(Wf, X, y, kappa, sig_a2, wt, w_cut, device)
        print(fmt(f"particle theory, start {start}", o, N), flush=True)
        w0 = Wf[:, 0].cpu().numpy()
        results[start] = dict(w0=w0, W=Wf.cpu().numpy(), traj=np.array(traj), summary=o)

    tag = f"rho{rho:g}".replace(".", "p")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"{tag}_particle.npz"
    save = dict(rho=rho, d=d, P=P, N=N, chi=chi, sig_a2=sig_a2, kappa=kappa, w_cut=w_cut,
                w_net=w_net, f_net=f_net, steps=steps, M=M)
    for start, res in results.items():
        save[f"w0_{start}"] = res["w0"]
        save[f"W_{start}"] = res["W"]
        save[f"traj_{start}"] = res["traj"]
        save[f"f_out_{start}"] = res["summary"]["f_out"]
        save[f"r_{start}"] = res["summary"]["r"]
        save[f"sp_med_{start}"] = res["summary"]["sp_med"]
        save[f"in_mean_{start}"] = res["summary"]["in_mean"]
    np.savez(npz_path, **save)
    print(f"wrote {npz_path}", flush=True)

    # action / density plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wmax = max(5.0, float(np.percentile(np.abs(np.concatenate([r["w0"] for r in results.values()])), 99)) * 1.2)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for k, logy in enumerate((False, True)):
        # emp action from particles (density-normalized -log p)
        for start, col in zip(results, ("C3", "C0")):
            h, e = np.histogram(results[start]["w0"], bins=120, range=(-wmax, wmax), density=True)
            xc = 0.5 * (e[1:] + e[:-1])
            ok = h > 0
            ax[k].plot(xc[ok], -np.log(h[ok]), color=col, lw=1.2, label=f"particle −log p ({start})")
        hn, en = np.histogram(w_net, bins=80, range=(-wmax, wmax), density=True)
        xcn = 0.5 * (en[1:] + en[:-1])
        okn = hn > 0
        ax[k].plot(xcn[okn], -np.log(hn[okn]), "o", ms=2.5, color="0.4", alpha=0.7,
                   label="network A_snaps (frozen)")
        if logy:
            # density panel instead
            ax[k].cla()
            ax[k].hist(w_net, bins=120, range=(-wmax, wmax), density=True, color="0.75", label="network")
            for start, col in zip(results, ("C3", "C0")):
                ax[k].hist(results[start]["w0"], bins=120, range=(-wmax, wmax), density=True,
                           histtype="step", color=col, lw=1.1, label=f"particle ({start})")
            ax[k].set_yscale("log")
            ax[k].set_ylim(1e-4, 20)
        ax[k].set_xlabel(r"$w_0$")
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel(r"$-\log p(w_0)$")
    ax[0].set_title(rf"ρ={rho:g}: particle theory on data (χ=N)", fontsize=9)
    ax[1].set_title("density (log)", fontsize=9)
    for start, col in zip(results, ("C3", "C0")):
        tr = results[start]["traj"]
        if len(tr):
            ax[2].plot(tr[:, 0], tr[:, 1], color=col, label=start)
    ax[2].axhline(f_net, color="k", ls=":", label="network (frozen)")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel(rf"fraction $|w_0|>{w_cut}$")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig_path = OUT_DIR / f"{tag}_particle.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved {fig_path}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--all", action="store_true", help="all rhos in PureCubicChiNRhoRay")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--starts", default="prior,outer")
    ap.add_argument("--f-start", type=float, default=0.5)
    ap.add_argument("--w-start", type=float, default=2.5)
    ap.add_argument("--w-cut", type=float, default=W_CUT_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=1000)
    args = ap.parse_args()

    if args.all:
        rhos = sorted({
            float(_RHO_RE.match(Path(p).name).group("rho"))
            for p in PA.EXPERIMENT_GROUP_BY_NAME[GROUP].model_dirs
            if _RHO_RE.match(Path(p).name)
        }, reverse=True)
    elif args.rho is not None:
        rhos = [args.rho]
    else:
        raise SystemExit("pass --rho or --all")

    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    for rho in rhos:
        run = select_run(rho)
        run_particles(
            run, steps=args.steps, M=args.M, starts=starts,
            f_start=args.f_start, w_start=args.w_start, w_cut=args.w_cut,
            device=args.device, every=args.every,
        )


if __name__ == "__main__":
    main()
