#!/usr/bin/env python3
"""Mode-space Laplace (all perp modes) on the pure-cubic χ=N ρ-ray, with self-consistent f*.

Patches tmp_mode_laplace for y=(he1, ε)=(0, ε), σ_A²=sa0/(N χ), κ=ρ sa0 P, then runs
solve_sc_root (S=I) per ρ. Dumps well parameters + density-normalized −log p(w0) for plotting.

Usage:
  python -u tmp_mode_laplace_pure_cubic_rho.py
  RHOS=0.15,0.08 ITERS=40 N_BISECT=6 python -u tmp_mode_laplace_pure_cubic_rho.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE / (
    "red_robin_pure_cubic_rho_ray"
    "_d16_N200_P1000_chiN_sa00.03_eps1_he10"
    "_lr0.01_ep60M_sched2_3_5"
)
OUT_DIR = HERE / "action_h0_activation_plots/PureCubicChiN/mode_laplace_scfrac"
XMAX = float(os.environ.get("XMAX", "6.0"))
NG = int(os.environ.get("NG", "601"))

# Steepwell d=16 loader (tmp_mode_laplace reads D at import).
os.environ.setdefault("D", "16")
os.environ.setdefault("SELF", "span")
os.environ.setdefault("M_DAMP", "0.1")
os.environ.setdefault("FIX_TRACE", "1")
# Pure-cubic outer wells sit farther out than steepwell (emp ~1.5–3.5).
os.environ.setdefault("BULK_LO", "0.0")
os.environ.setdefault("BULK_HI", "1.0")
os.environ.setdefault("SPEC_LO", "1.0")
os.environ.setdefault("SPEC_HI", "5.5")
os.environ.setdefault("MU_B0", "0.35")
os.environ.setdefault("SIG_B0", "0.18")
os.environ.setdefault("MU_S0", "2.5")
os.environ.setdefault("SIG_S0", "0.35")

ITERS = int(os.environ.get("ITERS", "60"))
sys.argv = [sys.argv[0], "none", str(ITERS)]

import tmp_mode_laplace as T  # noqa: E402


def configure(rho: float, *, d: int, P: int, N: int, chi: float, sa0: float, eps: float, he1: float, s0: float):
    """Retarget module globals to one pure-cubic χ=N point."""
    T.d, T.P, T.N = int(d), int(P), int(N)
    T.eps, T.s0 = float(eps), float(s0)
    T.sig_a2 = float(sa0) / (float(N) * float(chi))
    T.kappa = float(rho) * float(sa0) * float(P)
    T.chi = float(chi)
    T.ITERS = ITERS
    T.ym = torch.zeros(T.NM, device=T.DEV, dtype=T.DT)
    T.ym[0] = float(he1)
    T.ym[1] = float(eps)
    print(
        f"configure ρ={rho:g}: d={T.d} P={T.P} N={T.N} χ={chi:g} sa0={sa0:g} "
        f"σ_A²={T.sig_a2:.3e} κ={T.kappa:g} χ={T.chi:g} ρ'=κ/(χP)={T.kappa/(T.chi*T.P):.3e} y_m=({he1:g},{eps:g}) NM={T.NM} SELF={T.SELF}",
        flush=True,
    )


def theory_density(x, mu_b, sig_b, mu_s, sig_s, f_s):
    dens = np.zeros_like(x)
    for mu, sd, f in ((mu_b, sig_b, 1.0 - f_s), (mu_s, sig_s, f_s)):
        dens += (
            f
            * 0.5
            * (
                np.exp(-(x - mu) ** 2 / (2 * sd * sd))
                + np.exp(-(x + mu) ** 2 / (2 * sd * sd))
            )
            / (np.sqrt(2 * np.pi) * sd)
        )
    return dens


def dump_action(path: Path, *, rho, meta, st, f_s, tr, pop):
    x = np.linspace(-XMAX, XMAX, NG)
    dens = theory_density(
        x,
        float(st["b"]["mu"]),
        float(st["b"]["sig"]),
        float(st["s"]["mu"]),
        float(st["s"]["sig"]),
        float(f_s),
    )
    nlogp = -np.log(dens + 1e-300)
    np.savez(
        path,
        x=x,
        dens=dens,
        nlogp=nlogp,
        f_s=float(f_s),
        mu_b=float(st["b"]["mu"]),
        sig_b=float(st["b"]["sig"]),
        mu_s=float(st["s"]["mu"]),
        sig_s=float(st["s"]["sig"]),
        train=np.asarray(tr),
        pop=np.asarray(pop),
        rho=float(rho),
        **meta,
    )
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "w", "nlogp", "dens", "f_s", "mu_b", "sig_b", "mu_s", "sig_s"])
        for xi, yi, di in zip(x, nlogp, dens):
            w.writerow(
                [
                    rho,
                    xi,
                    yi,
                    di,
                    f_s,
                    st["b"]["mu"],
                    st["b"]["sig"],
                    st["s"]["mu"],
                    st["s"]["sig"],
                ]
            )
    print(f"wrote {path} and {csv_path}", flush=True)


def discover_rhos():
    env = os.environ.get("RHOS")
    if env:
        return [float(x) for x in env.split(",") if x.strip()]
    models = SWEEP / "models"
    rhos = []
    for d in sorted(models.iterdir()) if models.is_dir() else []:
        cfg = d / "config.json"
        if not cfg.is_file():
            continue
        c = json.loads(cfg.read_text())
        kappa = float(c["temperature"]) / 2.0
        rho = kappa / (float(c["sa0"]) * float(c["P"]))
        rhos.append(rho)
    return sorted(set(rhos), reverse=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rhos = discover_rhos()
    d = int(os.environ.get("PC_D", "16"))
    P = int(os.environ.get("PC_P", "1000"))
    N = int(os.environ.get("PC_N", "200"))
    chi = float(os.environ.get("PC_CHI", str(N)))
    sa0 = float(os.environ.get("PC_SA0", "0.03"))
    eps = float(os.environ.get("PC_EPS", "1.0"))
    he1 = float(os.environ.get("PC_HE1", "0.0"))
    s0 = float(os.environ.get("PC_S0", "1.0"))
    n_bisect = int(os.environ.get("N_BISECT", "7"))
    f_lo = float(os.environ.get("F_LO", "0.01"))
    f_hi = float(os.environ.get("F_HI", "0.35"))

    summary = []
    for rho in rhos:
        tag = f"rho{rho:g}".replace(".", "p")
        out = OUT_DIR / f"{tag}_scfrac.npz"
        if os.environ.get("SKIP_EXISTING", "0") == "1" and out.is_file():
            print(f"skip existing {out}", flush=True)
            continue
        configure(rho, d=d, P=P, N=N, chi=chi, sa0=sa0, eps=eps, he1=he1, s0=s0)
        if rho <= 0.03:
            os.environ["SPEC_HI"] = "5.5"
            os.environ["MU_S0"] = "3.0"
        elif rho <= 0.08:
            os.environ["SPEC_HI"] = "4.5"
            os.environ["MU_S0"] = "2.2"
        else:
            os.environ["SPEC_HI"] = "3.5"
            os.environ["MU_S0"] = "1.7"

        S = torch.eye(T.NM, device=T.DEV, dtype=T.DT)
        out_sc = T.solve_sc_root(
            S,
            f"pure-cubic χ=N ρ={rho:g} off data",
            lo=f_lo,
            hi=f_hi,
            n_bisect=n_bisect,
        )
        if out_sc is None:
            print(f"NO ROOT for ρ={rho:g}; trying iterative solve_sc from f0=0.08", flush=True)
            f_star, tr, pop = T.solve_sc(S, f"pure-cubic χ=N ρ={rho:g} off data", 0.08)
        else:
            f_star, tr, pop = out_sc
        st = T.LAST["st"]
        meta = dict(
            d=d,
            P=P,
            N=N,
            chi=chi,
            sa0=sa0,
            eps=eps,
            he1=he1,
            s0=s0,
            kappa=float(T.kappa),
            sig_a2=float(T.sig_a2),
            NM=int(T.NM),
            SELF=str(T.SELF),
            iters=ITERS,
        )
        dump_action(out, rho=rho, meta=meta, st=st, f_s=f_star, tr=tr, pop=pop)
        row = {
            "rho": rho,
            "f_star": float(f_star),
            "mu_b": float(st["b"]["mu"]),
            "sig_b": float(st["b"]["sig"]),
            "mu_s": float(st["s"]["mu"]),
            "sig_s": float(st["s"]["sig"]),
            "train_he1": float(tr[0]),
            "train_he3": float(tr[1]),
            "pop_he1": float(pop[0]),
            "pop_he3": float(pop[1]),
            "kappa": float(T.kappa),
        }
        summary.append(row)
        print(
            f"ρ={rho:g}: f*={f_star:.4f} bulk μ={st['b']['mu']:.3f}±{st['b']['sig']:.3f} "
            f"spec μ={st['s']['mu']:.3f}±{st['s']['sig']:.3f} "
            f"pop ({pop[0]:+.4f},{pop[1]:+.4f})",
            flush=True,
        )

    sum_path = OUT_DIR / "summary_scfrac.csv"
    if summary:
        with sum_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"wrote {sum_path}", flush=True)
    meta_path = OUT_DIR / "README.md"
    meta_path.write_text(
        "# Pure-cubic χ=N mode-space Laplace (self-consistent f*)\n\n"
        "Solver: `tmp_mode_laplace.py` `solve_sc_root` with all 154 modes (He1/3/5/7 + perps), "
        "S=I, SELF=span, y=(0,ε), σ_A²=sa0/(Nχ), κ=ρ sa0 P.\n\n"
        f"Params: d={d} N={N} P={P} χ={chi} sa0={sa0} ε={eps} he1={he1}.\n"
        "Action curves are density-normalized −log p of the two-well Laplace mixture.\n"
    )
    print(f"done; caches in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
