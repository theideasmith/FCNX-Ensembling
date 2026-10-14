#!/usr/bin/env python3
"""Rank mode-Laplace g(f)=0 roots by mixture free energy Φ; overlay experiment.

Uses existing mode_laplace_jump/gscan_rho*.csv to find roots, then re-solves at
each root f* to get log Z_b, log Z_s and

  Φ(f) = -(1-f) log Z_b - f log Z_s + f log f + (1-f) log(1-f)

Picks argmin_Φ among roots (vs naive bisect pick). Overlays precond network /
particle L3 and continuum multiwell.

Usage:
  ITERS=30 LAPLACE_DEV=cuda:1 python -u plot_mode_laplace_jump_phimin.py
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
JUMP = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_jump"
CONT = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac_fine/summary_multiwell_fine.csv"
THREE = HERE / "action_h0_activation_plots/PureCubicChiNRhoRay/particle_theory/learnability_three_way.csv"

os.environ.setdefault("D", "16")
os.environ.setdefault("SELF", "full")
os.environ.setdefault("M_DAMP", "0.1")
os.environ.setdefault("FIX_TRACE", "1")
os.environ.setdefault("BULK_LO", "0.0")
os.environ.setdefault("BULK_HI", "1.0")
os.environ.setdefault("SPEC_LO", "1.0")
os.environ.setdefault("SPEC_HI", "5.5")
os.environ.setdefault("MU_B0", "0.45")
os.environ.setdefault("SIG_B0", "0.10")
os.environ.setdefault("MU_S0", "3.2")
os.environ.setdefault("SIG_S0", "0.25")
os.environ.setdefault("LAPLACE_DEV", os.environ.get("LAPLACE_DEV", "cuda:1"))

ITERS = int(os.environ.get("ITERS", "30"))
sys.argv = [sys.argv[0], "none", str(ITERS)]

import tmp_mode_laplace as T  # noqa: E402
import tmp_mode_laplace_pure_cubic_rho as PC  # noqa: E402


def load_gscan(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: float(v) if k != "rho" else float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def find_roots(rows: list[dict]) -> list[dict]:
    roots = []
    for i in range(len(rows) - 1):
        g0, g1 = rows[i]["g"], rows[i + 1]["g"]
        if g0 * g1 >= 0:
            continue
        f0, f1 = rows[i]["f"], rows[i + 1]["f"]
        t = g0 / (g0 - g1)
        f_star = math.exp(math.log(f0) + t * (math.log(f1) - math.log(f0)))
        blend = lambda k: (1 - t) * rows[i][k] + t * rows[i + 1][k]
        roots.append(
            dict(
                f_star=f_star,
                L3_interp=blend("L3"),
                mu_b_interp=blend("mu_b"),
                mu_s_interp=blend("mu_s"),
                f_lo=f0,
                f_hi=f1,
            )
        )
    return roots


def solve_at_f(rho: float, f: float) -> dict:
    if rho <= 0.03:
        os.environ["SPEC_HI"] = "5.5"
        os.environ["MU_S0"] = "3.2"
    PC.configure(rho, d=16, P=1000, N=200, chi=200, sa0=0.03, eps=1.0, he1=0.0, s0=1.0)
    S = torch.eye(T.NM, device=T.DEV, dtype=T.DT)
    # capture logZ by wrapping the inner loop result: run solve_sc then recompute Z
    f_out, tr, pop = T.solve_sc(S, f"Φ ρ={rho:g} f={f:.4f}", float(f), f_damp=0.0)
    st, th = T.LAST["st"], T.LAST["th"]
    dlogZ = float(T.LAST["dlogZ"])
    # recompute absolute logZ_b, logZ_s (same formula as solve_sc)
    N_PERP, N_GRID = T.N_PERP, T.N_GRID
    g = torch.Generator(device="cpu").manual_seed(0)
    Zp = torch.randn(N_PERP, T.d - 1, generator=g, dtype=T.DT).to(T.DEV)
    rng = T._well_ranges()
    logZ = {}
    for k in ("b", "s"):
        w = st[k]
        L = torch.linalg.cholesky(w["Sig"])
        lo, hi = rng[k]
        grid = torch.linspace(lo, hi, N_GRID, device=T.DEV, dtype=T.DT)
        Wg = torch.cat(
            [grid.repeat_interleave(N_PERP)[:, None], (w["m"] + Zp @ L.T).repeat(N_GRID, 1)],
            1,
        )
        with torch.no_grad():
            V = th.U(Wg).reshape(N_GRID, N_PERP).mean(1).cpu().numpy()
        gr = grid.cpu().numpy()
        logZ[k] = float(
            np.log(np.trapz(np.exp(-(V - V.min())), gr))
            - V.min()
            + 0.5 * float(torch.logdet(w["Sig"]))
        )
    # mixture free energy
    Phi = (
        -(1.0 - f) * logZ["b"]
        - f * logZ["s"]
        + f * math.log(f)
        + (1.0 - f) * math.log(1.0 - f)
    )
    return dict(
        rho=rho,
        f_star=float(f),
        f_solved=float(f_out),
        L3=1.0 - float(pop[1]),
        pop_he1=float(pop[0]),
        pop_he3=float(pop[1]),
        mu_b=float(st["b"]["mu"]),
        mu_s=float(st["s"]["mu"]),
        sig_b=float(st["b"]["sig"]),
        sig_s=float(st["s"]["sig"]),
        logZ_b=logZ["b"],
        logZ_s=logZ["s"],
        dlogZ=dlogZ,
        Phi=Phi,
        g=dlogZ - math.log(f / (1.0 - f)),
    )


def bisect_pick_from_gscan(rows: list[dict], f_lo=0.01, f_hi=0.85) -> float | None:
    """Emulate solve_sc_root: first sign change from left in log-f, then mid."""
    # use dense rows; find bracket containing a root that bisect from [f_lo,f_hi] would hit
    # Approximate: evaluate sign at endpoints via nearest row, then walk
    fs = np.array([r["f"] for r in rows])
    gs = np.array([r["g"] for r in rows])
    # extrapolate endpoints roughly from nearest
    g_lo = gs[0] if fs[0] <= f_lo * 1.5 else None
    g_hi = gs[-1] if fs[-1] >= f_hi * 0.9 else None
    if g_lo is None or g_hi is None or g_lo * g_hi > 0:
        # fall back: first sign change
        for i in range(len(fs) - 1):
            if gs[i] * gs[i + 1] < 0:
                t = gs[i] / (gs[i] - gs[i + 1])
                return float(math.exp(math.log(fs[i]) + t * (math.log(fs[i + 1]) - math.log(fs[i]))))
        return None
    # bisect in log space using interpolated g from table
    def g_interp(f):
        if f <= fs[0]:
            return gs[0]
        if f >= fs[-1]:
            return gs[-1]
        j = int(np.searchsorted(fs, f) - 1)
        j = max(0, min(j, len(fs) - 2))
        # log-linear in f for root location; linear in g for sign
        t = (f - fs[j]) / (fs[j + 1] - fs[j])
        return (1 - t) * gs[j] + t * gs[j + 1]

    lo, hi = f_lo, f_hi
    glo, ghi = g_interp(lo), g_interp(hi)
    if glo * ghi > 0:
        return bisect_pick_from_gscan(rows)  # fallback first root
    for _ in range(12):
        mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        gm = g_interp(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi, ghi = mid, gm
    return float(math.exp(0.5 * (math.log(lo) + math.log(hi))))


def load_experiment() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(THREE.open()))
    return dict(
        rho=np.array([float(r["rho"]) for r in rows]),
        network=np.array([float(r["network"]) for r in rows]),
        particle=np.array([float(r["particle_prior"]) for r in rows]),
        continuum_old=np.array([float(r["laplace"]) for r in rows]),  # old mode 4-pt
    )


def load_continuum() -> np.ndarray | None:
    if not CONT.is_file():
        return None
    rows = list(csv.DictReader(CONT.open()))
    rows.sort(key=lambda r: float(r["rho"]))
    return np.array(
        [(float(r["rho"]), float(r["L3"]), float(r["pi_out"])) for r in rows],
        dtype=[("rho", float), ("L3", float), ("pi", float)],
    )


def main() -> None:
    JUMP.mkdir(parents=True, exist_ok=True)
    gfiles = sorted(JUMP.glob("gscan_rho*.csv"))
    if not gfiles:
        raise SystemExit(f"no gscan_rho*.csv in {JUMP}")

    all_root_evals: list[dict] = []
    summary: list[dict] = []

    for gf in gfiles:
        rows = load_gscan(gf)
        rho = float(rows[0]["rho"])
        roots = find_roots(rows)
        print(f"\nρ={rho:g}: {len(roots)} root(s) from {gf.name}", flush=True)
        if not roots:
            continue
        evals = []
        for k, rt in enumerate(roots):
            print(f"  resolve root{k} f*≈{rt['f_star']:.4f} …", flush=True)
            ev = solve_at_f(rho, rt["f_star"])
            ev["branch"] = k
            ev["n_roots"] = len(roots)
            evals.append(ev)
            all_root_evals.append(ev)
            print(
                f"    Φ={ev['Phi']:+.4f}  L3={ev['L3']:.4f}  μs={ev['mu_s']:.3f}  g={ev['g']:+.3f}",
                flush=True,
            )
        # min-Φ pick
        best = min(evals, key=lambda e: e["Phi"])
        # bisect pick (from table)
        f_bis = bisect_pick_from_gscan(rows)
        bis = min(evals, key=lambda e: abs(e["f_star"] - (f_bis or e["f_star"])))
        summary.append(
            dict(
                rho=rho,
                n_roots=len(roots),
                f_phimin=best["f_star"],
                L3_phimin=best["L3"],
                Phi_phimin=best["Phi"],
                mu_s_phimin=best["mu_s"],
                f_bisect=bis["f_star"],
                L3_bisect=bis["L3"],
                Phi_bisect=bis["Phi"],
                mu_s_bisect=bis["mu_s"],
                phimin_is_bisect=int(abs(best["f_star"] - bis["f_star"]) < 0.02),
            )
        )
        print(
            f"  → Φ-min: f*={best['f_star']:.4f} L3={best['L3']:.4f} Φ={best['Phi']:+.3f} | "
            f"bisect: f*={bis['f_star']:.4f} L3={bis['L3']:.4f} Φ={bis['Phi']:+.3f}",
            flush=True,
        )

    # write CSVs
    with (JUMP / "roots_Phi_vs_rho.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_root_evals[0].keys()))
        w.writeheader()
        w.writerows(all_root_evals)
    with (JUMP / "phimin_vs_bisect.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(sorted(summary, key=lambda r: r["rho"]))

    summary = sorted(summary, key=lambda r: r["rho"])
    emp = load_experiment()
    cont = load_continuum()

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))

    # L3
    ax = axes[0, 0]
    ax.plot(
        [r["rho"] for r in summary],
        [r["L3_bisect"] for r in summary],
        "x--",
        color="0.35",
        ms=7,
        lw=1.3,
        label="mode bisect pick",
    )
    ax.plot(
        [r["rho"] for r in summary],
        [r["L3_phimin"] for r in summary],
        "o-",
        color="C0",
        ms=6,
        lw=2.2,
        label=r"mode $\Phi$-min root",
    )
    if cont is not None:
        m = (cont["rho"] >= 0.012) & (cont["rho"] <= 0.08)
        ax.plot(cont["rho"][m], cont["L3"][m], "D-", color="C3", ms=4, lw=1.6, label="matrix continuum")
    ax.plot(emp["rho"], emp["network"], "s", color="C2", ms=8, label="experiment (precond)")
    ax.plot(emp["rho"], emp["particle"], "^", color="C4", ms=7, label="particle prior")
    ax.axvline(0.0185, color="0.7", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$L_3$")
    ax.set_title(r"$L_3$: $\Phi$-min root vs bisect jump")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.012, 0.16)

    # f*
    ax = axes[0, 1]
    # all roots as faint points
    for ev in all_root_evals:
        ax.plot(ev["rho"], ev["f_star"], "o", color="0.75", ms=4, zorder=1)
    ax.plot(
        [r["rho"] for r in summary],
        [r["f_bisect"] for r in summary],
        "x--",
        color="0.35",
        ms=7,
        lw=1.3,
        label="bisect",
    )
    ax.plot(
        [r["rho"] for r in summary],
        [r["f_phimin"] for r in summary],
        "o-",
        color="C0",
        ms=6,
        lw=2.2,
        label=r"$\Phi$-min",
    )
    ax.axvline(0.0185, color="0.7", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$f^\star$")
    ax.set_title("specialist fraction (grey = all roots)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Φ of each root
    ax = axes[1, 0]
    rhos_u = sorted({e["rho"] for e in all_root_evals})
    for rho in rhos_u:
        sub = [e for e in all_root_evals if e["rho"] == rho]
        ax.plot(
            [e["f_star"] for e in sub],
            [e["Phi"] for e in sub],
            "o-",
            ms=4,
            lw=1.2,
            label=f"{rho:g}",
        )
    ax.set_xlabel(r"$f^\star$")
    ax.set_ylabel(r"$\Phi(f)$")
    ax.set_title("mixture free energy at each root")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # mu_s
    ax = axes[1, 1]
    ax.plot(
        [r["rho"] for r in summary],
        [r["mu_s_bisect"] for r in summary],
        "x--",
        color="0.35",
        ms=7,
        lw=1.3,
        label="bisect μs",
    )
    ax.plot(
        [r["rho"] for r in summary],
        [r["mu_s_phimin"] for r in summary],
        "o-",
        color="C1",
        ms=6,
        lw=2.0,
        label=r"$\Phi$-min μs",
    )
    ax.axvline(0.0185, color="0.7", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\mu_s$")
    ax.set_title("specialist well location")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Mode-Laplace jump: choose $f^\star=\mathrm{argmin}\,\Phi$ among $g(f)=0$ roots"
        "\n"
        r"(experiment = precond Langevin $L_3$ from three-way table)",
        fontsize=11,
    )
    fig.tight_layout()
    out = JUMP / "L3_phimin_vs_experiment.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nsaved {out}", flush=True)

    (JUMP / "README.md").write_text(
        "# Mode-Laplace f* jump: Φ-min root selection\n\n"
        "Among multiple roots of `g(f)=0`, pick the mixture free-energy minimum\n\n"
        "```\n"
        "Φ(f) = -(1-f) log Z_b - f log Z_s + f log f + (1-f) log(1-f)\n"
        "```\n\n"
        "Artifacts:\n"
        "- `gscan_rho*.csv` — fixed-f `g(f)` tables\n"
        "- `roots_Phi_vs_rho.csv` — each root re-solved with Φ\n"
        "- `phimin_vs_bisect.csv` — Φ-min vs naive bisect pick\n"
        "- `L3_phimin_vs_experiment.png` — L3 with experiment overlay\n\n"
        "Reproduce:\n"
        "```bash\n"
        "ITERS=30 LAPLACE_DEV=cuda:1 python -u plot_mode_laplace_jump_phimin.py\n"
        "```\n"
    )


if __name__ == "__main__":
    main()
