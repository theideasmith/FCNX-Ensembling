#!/usr/bin/env python3
"""Cubic learnability L3 vs ρ: network, particle, mode-Laplace, arcsin NNGP.

L3 = E[f_res He3] / E[y He3] (He1 residualized), same as train_fcn2_erf_sigma_a.cubic_learnability.

  network  — mean L3 over second-half precond Langevin snaps (M2000 > M200) when present;
             else final cubic_learnability from each run's losses.json
  particle — cavity GP predictor from cached W (prior / outer starts) on a fresh test set
  Laplace  — (ε − pop_he3) / ε from mode_laplace_scfrac/summary (learned He3 mode coeff)
  NNGP     — infinite-width erf arcsin kernel ridge on the same recovered training set
             (plot_model_vs_preact_krr.arcsin_krr_predict_from_train; σ² = κ = T/2)

Usage:
  python -u plot_learnability_three_way.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_model_vs_preact_krr import arcsin_krr_predict_from_train

HERE = Path(__file__).resolve().parent
SWEEP = HERE / (
    "red_robin_pure_cubic_rho_ray"
    "_d16_N200_P1000_chiN_sa00.03_eps1_he10"
    "_lr0.01_ep60M_sched2_3_5"
)
PARTICLE = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory"
PRECOND = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/precond_langevin"
LAPLACE = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
RHOS = (0.15, 0.08, 0.03, 0.015)
_RHO_RE = re.compile(r"purecubic_rho(?P<rho>[\d.eE+-]+)_")


def tag(rho: float) -> str:
    return f"rho{rho:g}".replace(".", "p")


def he3(z):
    return (z ** 3 - 3.0 * z) / np.sqrt(6.0)


def cubic_L3_numpy(f, x, y):
    """f,y shape (P,), x shape (P,d)."""
    z = x[:, 0]
    h1, h3 = z, he3(z)
    inner_f_he1 = float(np.mean(f * h1))
    inner_y_he3 = float(np.mean(y * h3))
    f_res = f - inner_f_he1 * h1
    if abs(inner_y_he3) < 1e-12:
        return float("nan")
    return float(np.mean(f_res * h3) / inner_y_he3)


def select_run(rho: float) -> Path:
    models = SWEEP / "models"
    cands = []
    for p in models.iterdir():
        m = _RHO_RE.match(p.name)
        if m:
            cands.append((abs(float(m.group("rho")) - rho), p))
    cands.sort()
    return cands[0][1]


def precond_snap_dir(rho: float) -> Path | None:
    t = tag(rho)
    for mw in ("M2000", "M200"):
        snap_dir = PRECOND / f"{t}_{mw}" / "snapshots"
        if snap_dir.is_dir() and any(snap_dir.glob("step_*.pt")):
            return snap_dir
    return None


def network_L3(rho: float, *, n_test: int, device: str = "cpu", seed: int = 1) -> tuple[float, str]:
    """Return (L3, source tag). Prefer precond second-half mean L3 (CPU)."""
    del device  # precond L3 is evaluated on CPU; arg kept for call-site symmetry
    snap_dir = precond_snap_dir(rho)
    if snap_dir is not None:
        run = select_run(rho)
        c = json.loads((run / "config.json").read_text())
        d = int(c["d"])
        eps, he1 = float(c["eps"]), float(c.get("he1", 0.0))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        Xt = torch.randn(n_test, d, generator=gen, dtype=torch.float64)
        yt = he1 * Xt[:, 0] + eps * ((Xt[:, 0] ** 3 - 3 * Xt[:, 0]) / np.sqrt(6.0))
        Xt_np, yt_np = Xt.numpy(), yt.numpy()
        snaps = sorted(snap_dir.glob("step_*.pt"), key=lambda s: int(s.stem.split("_")[1]))
        use = snaps[len(snaps) // 2 :]
        # subsample for speed (still posterior mean over ~50 states)
        stride = max(1, len(use) // 50)
        use = use[::stride]
        vals = []
        for s in use:
            sd = torch.load(s, map_location="cpu", weights_only=False)
            W = sd["W0"].double().reshape(-1, d)
            A = sd["A"].double().reshape(-1)
            f = (torch.erf(Xt @ W.T) @ A).numpy()
            vals.append(cubic_L3_numpy(f, Xt_np, yt_np))
        mw = snap_dir.parent.name.split("_")[-1]
        return float(np.mean(vals)), f"precond {mw} (n={len(vals)})"
    run = select_run(rho)
    data = json.loads((run / "losses.json").read_text())
    cl = data["cubic_learnability"]
    k = max(int(x) for x in cl)
    return float(cl[str(k)]), "losses.json"


def laplace_L3(summary_path: Path) -> dict[float, float]:
    import csv

    out = {}
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            rho = float(row["rho"])
            eps = 1.0  # pure-cubic teacher ym He3
            out[rho] = (eps - float(row["pop_he3"])) / eps
    return out


def _train_and_test(rho: float, *, n_test: int, device: str, seed: int = 0):
    """Recover the run's training set; draw a fresh test set (same seed as particle path)."""
    from precond_langevin_warmstart import recover_training_set

    run = select_run(rho)
    c = json.loads((run / "config.json").read_text())
    d, P = int(c["d"]), int(c["P"])
    eps, he1 = float(c["eps"]), float(c.get("he1", 0.0))
    T_bare = float(c["temperature"])
    kappa = T_bare / 2.0  # bare κ in ρ=κ/(sa0 P); arcsin ridge σ²=κ (project convention)

    rec = json.loads((run / "pred_vs_true.json").read_text())
    y_np = np.asarray(rec[max(rec, key=int)]["y_true"], dtype=float)
    X_np, _, _ = recover_training_set(P, d, y_np, eps, he1=he1)
    DEV = torch.device(device if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(DEV, dtype=torch.float64)
    y = torch.from_numpy(y_np).to(DEV, dtype=torch.float64)

    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    Xt = torch.randn(n_test, d, generator=gen, device=DEV, dtype=torch.float64)
    yt = he1 * Xt[:, 0] + eps * ((Xt[:, 0] ** 3 - 3 * Xt[:, 0]) / np.sqrt(6.0))
    return run, c, kappa, X, y, Xt, yt


def particle_L3(rho: float, *, n_test: int, device: str, seed: int = 0) -> dict[str, float]:
    z = np.load(PARTICLE / f"{tag(rho)}_particle.npz")
    _, c, _, X, y, Xt, yt = _train_and_test(rho, n_test=n_test, device=device, seed=seed)
    d, N = int(c["d"]), int(c["N"])
    sa0, chi = float(c["sa0"]), float(c.get("chi", N))
    T_bare = float(c["temperature"])
    # cavity ridge is T_eff/2 on K = σ_A² ΦΦᵀ (χ=N amplitude)
    kappa_cav = (T_bare / chi) / 2.0
    sig_a2 = sa0 / (N * chi)
    M = int(z["M"])
    wt = N / M
    DEV = X.device
    I = torch.eye(X.shape[0], device=DEV, dtype=torch.float64)

    out = {}
    for start in ("prior", "outer"):
        W = torch.from_numpy(np.asarray(z[f"W_{start}"], dtype=np.float64)).to(DEV)
        Phi = torch.erf(X @ W.T)
        K = sig_a2 * wt * (Phi @ Phi.T)
        t = torch.cholesky_inverse(torch.linalg.cholesky(K + kappa_cav * I)) @ y
        Phi_t = torch.erf(Xt @ W.T)
        f = (sig_a2 * wt * (Phi_t @ (Phi.T @ t))).detach().cpu().numpy()
        out[start] = cubic_L3_numpy(f, Xt.cpu().numpy(), yt.cpu().numpy())
    return out


def nngp_arcsin_L3(rho: float, *, n_test: int, device: str, seed: int = 0) -> float:
    """Arcsin (erf NNGP) KRR on the same training set; σ² = κ = T/2."""
    _, _, kappa, X, y, Xt, yt = _train_and_test(rho, n_test=n_test, device=device, seed=seed)
    with torch.no_grad():
        f = arcsin_krr_predict_from_train(X, y, Xt, float(kappa))
    return cubic_L3_numpy(f.detach().cpu().numpy(), Xt.cpu().numpy(), yt.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = args.out or (PARTICLE / "learnability_three_way.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    lap = laplace_L3(LAPLACE / "summary_scfrac.csv")
    rows = []
    print(
        f"{'rho':>7} {'net':>8} {'net_src':>22} {'part_pri':>9} {'part_out':>9} "
        f"{'nngp':>9} {'laplace':>9}",
        flush=True,
    )
    for rho in RHOS:
        net, net_src = network_L3(rho, n_test=args.n_test, device=args.device)
        part = particle_L3(rho, n_test=args.n_test, device=args.device)
        nngp = nngp_arcsin_L3(rho, n_test=args.n_test, device=args.device)
        # nearest laplace rho key
        rkey = min(lap, key=lambda r: abs(r - rho))
        L = lap[rkey]
        rows.append(
            dict(
                rho=rho,
                network=net,
                network_src=net_src,
                particle_prior=part["prior"],
                particle_outer=part["outer"],
                nngp=nngp,
                laplace=L,
            )
        )
        print(
            f"{rho:7g} {net:8.4f} {net_src:>22} {part['prior']:9.4f} {part['outer']:9.4f} "
            f"{nngp:9.4f} {L:9.4f}",
            flush=True,
        )

    # CSV
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w") as f:
        f.write("rho,network,network_src,particle_prior,particle_outer,nngp_arcsin,laplace\n")
        for r in rows:
            f.write(
                f"{r['rho']},{r['network']},{r['network_src']},"
                f"{r['particle_prior']},{r['particle_outer']},{r['nngp']},{r['laplace']}\n"
            )
    print(f"wrote {csv_path}", flush=True)

    rows = sorted(rows, key=lambda r: r["rho"])  # increasing ρ = more noise →
    rhos = np.array([r["rho"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        rhos,
        [r["network"] for r in rows],
        "o-",
        color="0.35",
        lw=1.6,
        ms=7,
        label="network (precond when available)",
    )
    ax.plot(
        rhos,
        [r["particle_prior"] for r in rows],
        "s--",
        color="C3",
        lw=1.4,
        ms=6,
        label="particle (prior start)",
    )
    ax.plot(
        rhos,
        [r["particle_outer"] for r in rows],
        "^--",
        color="C0",
        lw=1.4,
        ms=6,
        label="particle (outer start)",
    )
    ax.plot(
        rhos,
        [r["nngp"] for r in rows],
        "v-.",
        color="C2",
        lw=1.5,
        ms=6,
        label=r"NNGP arcsin KRR ($\sigma^2=\kappa=T/2$)",
    )
    ax.plot(
        rhos,
        [r["laplace"] for r in rows],
        "D-",
        color="k",
        lw=1.5,
        ms=6,
        label=r"mode-Laplace ($S=I$, SELF=full): $1-\mathrm{pop}_{He3}$",
    )
    ax.set_xlabel(r"$\rho=\kappa/(\mathrm{sa0}\,P)$  (larger $\rho$ = more noise $\rightarrow$)")
    ax.set_ylabel(r"cubic learnability $L_3$")
    ax.set_title(
        r"Pure cubic $\chi=N$: $L_3$ vs $\rho$"
        "\n"
        r"$d=16$, $N=200$, $P=10^3$, $\mathrm{sa0}=0.03$, $\varepsilon=1$"
    )
    ax.set_xticks(list(rhos))
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.0)
    ax.set_ylim(bottom=min(0.0, min(r["network"] for r in rows) - 0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
