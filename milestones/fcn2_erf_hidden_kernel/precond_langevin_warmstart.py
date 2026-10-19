"""Warm-started Langevin with a larger read-in step (equilibration of w0).

Same loss / priors / temperature as train_fcn2_erf_sigma_a.py (ens = 1, full batch):
    L = sum_p (f_p - y_p)^2,  f = erf(X W^T) A
    T_eff = T / χ,  wd_W0 = (d/s0) T_eff,  wd_A = (N χ / sa0) T_eff
    theta <- theta - lr_g (grad L + wd_g theta) + sqrt(2 lr_g T_eff) xi
    lr_A = lr (run config),  lr_W0 = M_W * lr
A per-group constant step leaves the stationary distribution unchanged; the read-in
block is ~10^3 softer than the readout, so M_W ~ 200.

Selects a run from an experiment group in plot_action_h0_activation.py:
  --group SteepwellInvariantEps0p5Asnap --d 16     (original steepwell ray)
  --group PureCubicChiNRhoRay --rho 0.015           (pure-cubic χ=N ρ-ray)

Legacy (unchanged):  precond_langevin_warmstart.py D [M_W] [STEPS] [DEVICE] [resume]
  → SteepwellInvariantEps0p5Asnap, --d D.

Snapshots: action_h0_activation_plots/<group>/precond_langevin/<tag>_M{M_W}/snapshots/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import plot_action_h0_activation as PA
from train_discrepancy_two_pop import he3

HERE = Path(__file__).resolve().parent
EVERY, SUMMARY_EVERY, W_CUT = 20_000, 500_000, 0.6
_RHO_RE = re.compile(r"purecubic_rho(?P<rho>[\d.eE+-]+)_")


def _legacy_argv(argv: list[str]) -> list[str]:
    """Map old positional CLI onto --group/--d/... flags."""
    if not argv or argv[0].startswith("-") or not re.fullmatch(r"\d+", argv[0]):
        return argv
    d, rest = argv[0], argv[1:]
    resume = []
    if rest and rest[-1] == "resume":
        resume = ["--resume"]
        rest = rest[:-1]
    out = ["--group", "SteepwellInvariantEps0p5Asnap", "--d", d]
    for key, val in zip(("--M-W", "--steps", "--device"), rest):
        out += [key, val]
    return out + resume


def parse_args(argv=None):
    argv = _legacy_argv(list(argv if argv is not None else sys.argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="SteepwellInvariantEps0p5Asnap",
                    help="Experiment group name from plot_action_h0_activation.py")
    ap.add_argument("--d", type=int, default=None, help="Select run by input dimension (steepwell ray)")
    ap.add_argument("--rho", type=float, default=None, help="Select run by ρ=κ/(sa0 P) (pure-cubic ρ-ray)")
    ap.add_argument("--M-W", dest="M_W", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-name", default=None, help="Override output dir name under precond_langevin/")
    return ap.parse_args(argv)


def recover_training_set(P, d, y_true, eps, he1=1.0, seeds=(0, 42)):
    """Reproduce X from y = he1·z + eps·He3(z). CUDA RNG (matches train_fcn2_erf_sigma_a)."""
    y_true = np.asarray(y_true, float)
    for seed in seeds:
        for dev in ("cuda:0", "cuda:1", "cpu"):
            try:
                torch.manual_seed(seed)
                X = torch.randn(P, d, device=dev).cpu().double().numpy()
            except Exception:
                continue
            z = X[:, 0]
            y_hat = he1 * z + eps * he3(z)
            if np.max(np.abs(y_true - y_hat) / (1 + np.abs(y_true))) < 1e-5:
                return X, seed, dev
    raise RuntimeError(f"training set not reproduced (he1={he1}, eps={eps}, P={P}, d={d})")


def crossings(a):
    state = np.where(a[0] < 0.5, 0, np.where(a[0] > 1.0, 1, -1))
    i2o = o2i = 0
    for k in range(1, len(a)):
        s = np.where(a[k] < 0.5, 0, np.where(a[k] > 1.0, 1, -1))
        i2o += int(((state == 0) & (s == 1)).sum())
        o2i += int(((state == 1) & (s == 0)).sum())
        state = np.where(s >= 0, s, state)
    return i2o, o2i


def summary_line(Wl, Al, X_np, y_np, tag):
    Wl, Al = np.stack(Wl), np.stack(Al)
    a = np.abs(Wl[:, :, 0])
    inner = a <= W_CUT
    wp2 = (Wl[:, :, 1:] ** 2).sum(2)
    z = X_np[:, 0]
    pred = np.mean([torch.erf(torch.from_numpy(X_np @ Wk.T)).numpy() @ Ak for Wk, Ak in zip(Wl, Al)], 0)
    r = y_np - pred
    h, e = np.histogram(a[inner], bins=60, range=(0, W_CUT)) if inner.any() else (np.array([0]), np.array([0.0, W_CUT]))
    i2o, o2i = crossings(a)
    peak = 0.5 * (e[h.argmax()] + e[h.argmax() + 1]) if inner.any() and h.sum() else float("nan")
    return (
        f"{tag:22s}: outer frac {np.mean(~inner):.4f} | inner mean |w0| "
        f"{a[inner].mean() if inner.any() else float('nan'):.3f}, peak {peak:.3f} | "
        f"inner |w_perp|^2 {wp2[inner].mean() if inner.any() else float('nan'):.3f} | "
        f"outer median |w0| {np.median(a[~inner]) if (~inner).any() else float('nan'):.2f} | "
        f"train discrepancy He1/He3 ({np.mean(r * z):+.3f}, {np.mean(r * he3(z)):+.3f}) | "
        f"crossings in->out/out->in {i2o}/{o2i} over {len(a)} snaps"
    )


def select_run(group_name: str, d: int | None, rho: float | None) -> Path:
    if group_name not in PA.EXPERIMENT_GROUP_BY_NAME:
        known = ", ".join(sorted(PA.EXPERIMENT_GROUP_BY_NAME))
        raise SystemExit(f"unknown group {group_name!r}; known: {known}")
    dirs = [Path(p) for p in PA.EXPERIMENT_GROUP_BY_NAME[group_name].model_dirs]
    if not dirs:
        raise SystemExit(f"group {group_name} has no model_dirs")

    if rho is not None:
        cands = []
        for p in dirs:
            m = _RHO_RE.match(p.name)
            if not m:
                continue
            cands.append((abs(float(m.group("rho")) - rho), float(m.group("rho")), p))
        if not cands:
            raise SystemExit(f"no purecubic_rho* runs in group {group_name}")
        cands.sort()
        best = cands[0]
        if best[0] / max(rho, 1e-12) > 0.05 and best[0] > 1e-6:
            raise SystemExit(f"no run near ρ={rho:g} in {group_name}; nearest {best[1]:g}")
        return best[2]

    if d is not None:
        for p in dirs:
            cfg = p / "config.json"
            if cfg.is_file() and json.load(open(cfg))["d"] == d:
                return p
        raise SystemExit(f"no run with d={d} in group {group_name}")

    raise SystemExit("select a run with --d or --rho")


def default_tag(run: Path, d: int, rho: float | None) -> str:
    if rho is not None:
        return f"rho{rho:g}".replace(".", "p")
    m = _RHO_RE.match(run.name)
    if m:
        return f"rho{float(m.group('rho')):g}".replace(".", "p")
    return f"d{d}"


def main():
    args = parse_args()
    group = args.group
    M_W, STEPS, DEV, RESUME = args.M_W, args.steps, args.device, args.resume
    torch.manual_seed(int(os.environ.get("SEED", 1 if RESUME else 0)))

    run = select_run(group, args.d, args.rho)
    c = json.load(open(run / "config.json"))
    d, P, N = int(c["d"]), int(c["P"]), int(c["N"])
    eps = float(c["eps"])
    he1 = float(c.get("he1", 1.0))
    T_bare = float(c["temperature"])
    chi = float(c.get("chi", 1.0))
    T_eff = T_bare / chi
    a0, s0 = float(c["sa0"]), float(c["s0"])
    lr = float(c["lr"])
    kappa = T_bare / 2.0
    rho_run = kappa / (a0 * P)

    rec = json.load(open(run / "pred_vs_true.json"))
    y_np = np.asarray(rec[max(rec, key=int)]["y_true"], dtype=float)
    X_np, data_seed, data_dev = recover_training_set(P, d, y_np, eps, he1=he1)
    X = torch.from_numpy(X_np).float().to(DEV)
    y = torch.from_numpy(y_np).float().to(DEV)

    snaps = sorted(glob.glob(str(run / "A_snapshots" / "*.pt")),
                   key=lambda s: int(s.split("_")[-1][:-3]))
    if not snaps:
        raise SystemExit(f"no A_snapshots in {run}")
    start, offset = snaps[-1], 0

    out_root = HERE / "action_h0_activation_plots" / group / "precond_langevin"
    tag = args.out_name or default_tag(run, d, args.rho if args.rho is not None else rho_run)
    base_name = f"{tag}_M{M_W:g}"
    if RESUME:
        prev_name = os.environ.get("RESUME_FROM", base_name)
        prev = sorted(glob.glob(str(out_root / prev_name / "snapshots" / "*.pt")),
                      key=lambda s: int(s.split("_")[-1][:-3]))
        if not prev:
            raise SystemExit(f"no snapshots to resume under {out_root / prev_name}")
        start, offset = prev[-1], int(prev[-1].split("_")[-1][:-3])
        out_name = os.environ.get("OUT_NAME", base_name + "_cont")
    else:
        out_name = os.environ.get("OUT_NAME", base_name)
    out = out_root / out_name
    (out / "snapshots").mkdir(parents=True, exist_ok=True)

    sd = torch.load(start, map_location="cpu", weights_only=False)
    W = sd["W0"].float().reshape(-1, d).to(DEV)
    A = sd["A"].float().reshape(-1).to(DEV)
    wdW = (d / s0) * T_eff
    wdA = (N * chi / a0) * T_eff
    lrW, lrA = M_W * lr, lr
    nW, nA = np.sqrt(2 * lrW * T_eff), np.sqrt(2 * lrA * T_eff)

    meta = dict(
        group=group, source_run=str(run), source_snapshot=str(start), step_offset=offset,
        d=d, P=P, N=N, T=T_bare, T_eff=T_eff, chi=chi, sa0=a0, s0=s0, eps=eps, he1=he1,
        rho=rho_run, lr_A=lrA, lr_W0=lrW, M_W=M_W, steps=STEPS, every=EVERY, device=DEV,
        data_seed=data_seed, data_dev=data_dev, wd_W0=wdW, wd_A=wdA,
    )
    json.dump(meta, open(out / "config.json", "w"), indent=1)

    orig_W = [torch.load(s, map_location="cpu", weights_only=False)["W0"].double().reshape(-1, d).numpy() for s in snaps]
    orig_A = [torch.load(s, map_location="cpu", weights_only=False)["A"].double().reshape(-1).numpy() for s in snaps]
    orig_line = summary_line(orig_W, orig_A, X_np, y_np, f"original ({len(snaps)} snaps)")
    print(
        f"group={group} run={run.name}\n"
        f"d={d} P={P} N={N} χ={chi:g} T={T_bare:.4g} T_eff={T_eff:.4g} ρ={rho_run:.4g} "
        f"he1={he1:g} eps={eps:g} lr_A={lrA:.3g} lr_W0={lrW:.3g} (x{M_W:g}) steps={STEPS} -> {out}",
        flush=True,
    )
    print(f"train set recovered seed={data_seed} on {data_dev}", flush=True)
    print(orig_line, flush=True)

    Ws, As = [], []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        W.requires_grad_(True)
        A.requires_grad_(True)
        L = ((torch.erf(X @ W.T) @ A - y) ** 2).sum()
        gW, gA = torch.autograd.grad(L, (W, A))
        with torch.no_grad():
            W = W - lrW * (gW + wdW * W) + nW * torch.randn_like(W)
            A = A - lrA * (gA + wdA * A) + nA * torch.randn_like(A)
        if step % EVERY == 0:
            Wc, Ac = W.detach().cpu(), A.detach().cpu()
            torch.save(
                {"step": offset + step, "W0": Wc.reshape(1, N, d), "A": Ac.reshape(1, N)},
                out / "snapshots" / f"step_{offset + step}.pt",
            )
            Ws.append(Wc.double().numpy())
            As.append(Ac.double().numpy())
            a = np.abs(Ws[-1][:, 0])
            print(
                f"  step {offset + step}: outer frac {np.mean(a > W_CUT):.4f}, "
                f"inner mean |w0| {a[a <= W_CUT].mean() if (a <= W_CUT).any() else float('nan'):.3f}, "
                f"loss {float(L.detach()):.4f}, {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % SUMMARY_EVERY == 0 or step == STEPS:
            lines = [orig_line]
            k = len(Ws)
            for lo, hi, tag_ in ((0, k // 2, "new, first half"), (k // 2, k, "new, second half")):
                if hi - lo >= 2:
                    lines.append(summary_line(Ws[lo:hi], As[lo:hi], X_np, y_np, tag_))
            with open(out / "summary.txt", "w") as fh:
                fh.write(f"step {offset + step} of {STEPS} (+offset {offset}), {time.time() - t0:.0f}s\n"
                         + "\n".join(lines) + "\n")
            print("\n".join(lines[1:]), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    w_new = np.stack(Ws[len(Ws) // 2:])[:, :, 0].ravel()
    wmax = max(3.0, float(np.percentile(np.abs(w_new), 99)) * 1.1)
    for k, logy in enumerate((False, True)):
        ax[k].hist(np.stack(orig_W)[:, :, 0].ravel(), bins=240, range=(-wmax, wmax), density=True,
                   color="0.75", label="original")
        ax[k].hist(w_new, bins=240, range=(-wmax, wmax), density=True, histtype="step", color="C3",
                   label=f"read-in lr x{M_W:g}, second half")
        if logy:
            ax[k].set_yscale("log")
            ax[k].set_ylim(1e-4, 20)
        ax[k].set_xlabel("w0")
        ax[k].legend(fontsize=7)
    ax[0].set_title(f"{group}: d={d}, ρ={rho_run:.3g}, P={P}", fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "w0_hist.png", dpi=120)
    print(out / "w0_hist.png", flush=True)


if __name__ == "__main__":
    main()
