#!/usr/bin/env python3
"""Beta-indexed companion to plot_action_h0_activation.py.

For fixed-P invariant beta sweeps, the P-indexed summaries collapse all runs
onto one abscissa. This script regenerates the key summary + empirical-action
figures versus beta.

Snapshot / ensemble averaging
-----------------------------
For Asnap groups, every model dir is required to contain ``A_snapshots/``.
Empirical quantities are recomputed via ``plot_action_h0_activation`` by
averaging over Langevin snapshots:
  - learnability of the predictive mean f_bar = mean_t f_t(x)
  - h0 projection histograms / variances pooled over snapshot W0_t
  - H_eig averaged over snapshots
  - weight / A–W plots pooled over snapshots
This script refuses a stale cache that lacks those snapshot markers.
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

import torch

from plot_action_h0_activation import (
    EXPERIMENT_GROUP_BY_NAME,
    OUTPUT_BASE_DIR,
    annotate_scaling_exponents,
    annotate_theory_empirical_gaps,
    curve_color_scale,
    learnability_from_eigenvalue,
    load_a_snapshots,
    make_training_dataset,
    parse_config_from_dirname,
    plot_experiment_group,
    population_teacher_hermite_coeffs,
    slugify,
)
from plot_model_vs_preact_krr import (
    arcsin_krr_predict_from_train,
    make_test_dataset,
)

# Groups whose natural control parameter is beta (typically fixed P / invariant ray).
BETA_INDEXED_GROUPS = (
    "InvariantPfixed1500Asnap",
    "SteepwellInvariantEps0p5Asnap",
    "SteepwellInvariantEps0p03N30dAsnap",
)

# Must match plot_action_h0_activation.plot_experiment_group cache_version when
# A_snapshots are stacked as the ensemble.
REQUIRED_CACHE_VERSION = 9

_BETA_RE = re.compile(r"beta([\d.]+)", re.IGNORECASE)


def beta_from_entry(entry: dict) -> float | None:
    for key in ("beta", "model_name", "model_dir"):
        val = entry.get(key)
        if val is None:
            continue
        if key == "beta":
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        m = _BETA_RE.search(str(val))
        if m:
            return float(m.group(1))
    return None


def require_a_snapshots(model_dirs: list[str]) -> dict[str, int]:
    """Return model_dir -> T. Mid-train runs without A_snapshots get T=0."""
    counts: dict[str, int] = {}
    missing = []
    for model_dir in model_dirs:
        pack = load_a_snapshots(model_dir, device=None)
        if pack is None:
            missing.append(model_dir)
            counts[model_dir] = 0
            continue
        A_snaps, _W0, _epochs = pack
        counts[model_dir] = int(A_snaps.shape[0])
        del A_snaps, _W0
    if missing:
        preview = "\n  ".join(Path(m).name for m in missing[:5])
        print(
            f"Note: {len(missing)} run(s) lack A_snapshots/; "
            f"plotting from latest checkpoint, e.g.:\n  {preview}"
        )
    return counts


def cache_uses_a_snapshots(cached: dict | None, snap_counts: dict[str, int]) -> bool:
    """True iff cache was built with snapshot-stacked ensemble averaging."""
    if cached is None:
        return False
    if int(cached.get("version", -1)) != REQUIRED_CACHE_VERSION:
        return False
    entries = cached.get("p_summary_entries") or []
    if not entries:
        return False
    for entry in entries:
        model_dir = entry.get("model_dir")
        if model_dir not in snap_counts:
            return False
        expected = snap_counts[model_dir]
        # Mid-train: no A_snapshots yet; a checkpoint-based cache is valid.
        if expected == 0:
            continue
        # Number of Langevin snapshots averaged into empirical quantities.
        ens = entry.get("n_snapshots", entry.get("ensemble_size"))
        src = entry.get("learnability_source", "")
        if ens is None or int(ens) < expected:
            return False
        if "asnap" not in str(src):
            return False
    return True


def load_group_cache(group_name: str) -> dict | None:
    cache_path = OUTPUT_BASE_DIR / slugify(group_name) / "computation_cache.pkl"
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def hermite_learnability_from_preds(
    f: torch.Tensor,
    he1: torch.Tensor,
    he3: torch.Tensor,
    eps: float,
) -> tuple[float, float]:
    """Population-denom linear / residualized-cubic learnability of predictor f."""
    y_He1, y_He3 = population_teacher_hermite_coeffs(eps)
    mean_fx0 = float((f * he1).mean().item())
    mean_fh3 = float((f * he3).mean().item())
    mean_x0h3 = float((he1 * he3).mean().item())
    linear = mean_fx0 / y_He1 if abs(y_He1) > 1e-12 else float("nan")
    if abs(y_He3) < 1e-12:
        cubic = float("nan")
    else:
        cubic = (mean_fh3 - mean_fx0 * mean_x0h3) / y_He3
    return linear, cubic


def arcsin_gpr_learnability_for_entry(
    entry: dict,
    device: torch.device,
    P_test: int | None = None,
) -> tuple[float, float]:
    """Test-set He1/He3 learnability of arcsin (erf NNGP) GPR-KRR with entry kappa_eff."""
    model_dir = entry.get("model_dir")
    kappa = entry.get("kappa_eff", entry.get("kappa"))
    if model_dir is None or not np.isfinite(kappa):
        return float("nan"), float("nan")

    d, P, _N, _chi, seed, _T, epsilon, _s0 = parse_config_from_dirname(model_dir)
    eps = float(epsilon) if epsilon is not None else 0.0
    X_train, y_train, _he1_tr, _he3_tr = make_training_dataset(P, d, seed, eps, device)
    X_test, _y_test, he1_te, he3_te = make_test_dataset(
        P, d, seed, eps, device, P_test=P_test
    )
    with torch.no_grad():
        f = arcsin_krr_predict_from_train(
            X_train, y_train, X_test, float(kappa)
        )
    return hermite_learnability_from_preds(f, he1_te, he3_te, eps)


def load_or_compute_gpr_cache(
    group_name: str,
    p_summary_entries: list[dict],
    device: torch.device,
    P_test: int | None = None,
    recompute: bool = False,
) -> dict[str, dict]:
    """Per-model_dir arcsin-GPR learnability, cached next to the group plots."""
    output_dir = OUTPUT_BASE_DIR / slugify(group_name)
    cache_path = output_dir / "arcsin_gpr_learnability_cache.pkl"
    cached: dict[str, dict] = {}
    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("P_test") == P_test:
            cached = dict(payload.get("by_model_dir", {}))

    dirty = False
    for entry in p_summary_entries:
        model_dir = entry.get("model_dir")
        if not model_dir:
            continue
        kappa = float(entry.get("kappa_eff", entry.get("kappa", float("nan"))))
        hit = cached.get(model_dir)
        if (
            hit is not None
            and abs(float(hit.get("kappa_eff", float("nan"))) - kappa) < 1e-12
            and hit.get("P_test") == P_test
        ):
            continue
        print(
            f"  arcsin GPR-KRR learnability for {Path(model_dir).name} "
            f"(kappa_eff={kappa:.4g}, P_test={P_test})"
        )
        lin, cub = arcsin_gpr_learnability_for_entry(entry, device, P_test=P_test)
        print(f"    He1={lin:.4f}  He3={cub:.4f}")
        cached[model_dir] = {
            "kappa_eff": kappa,
            "P_test": P_test,
            "linear": lin,
            "cubic": cub,
        }
        dirty = True

    if dirty:
        with open(cache_path, "wb") as f:
            pickle.dump({"P_test": P_test, "by_model_dir": cached}, f)
        print(f"Saved GPR learnability cache to {cache_path}")
    return cached


def build_beta_series(
    p_summary_entries: list[dict],
    gpr_by_dir: dict[str, dict] | None = None,
) -> dict:
    """One series point per distinct beta (mean over seeds if several)."""
    grouped = defaultdict(list)
    for entry in p_summary_entries:
        beta = beta_from_entry(entry)
        if beta is None:
            continue
        grouped[beta].append(entry)

    betas = sorted(grouped.keys())
    series = {
        "beta": betas,
        "linear_empirical": [],
        "linear_theory": [],
        "cubic_empirical": [],
        "cubic_theory": [],
        "linear_learnability_empirical": [],
        "linear_learnability_theory": [],
        "cubic_learnability_empirical": [],
        "cubic_learnability_theory": [],
        "linear_learnability_arcsin_gpr": [],
        "cubic_learnability_arcsin_gpr": [],
        "P": [],
        "kappa_eff": [],
        "d": [],
        "N": [],
    }

    for beta in betas:
        entries = grouped[beta]
        series["linear_empirical"].append(
            _finite_mean([e.get("linear_empirical") for e in entries])
        )
        series["linear_theory"].append(
            _finite_mean([e.get("linear_theory") for e in entries])
        )
        series["cubic_empirical"].append(
            _finite_mean([e.get("cubic_empirical") for e in entries])
        )
        series["cubic_theory"].append(
            _finite_mean([e.get("cubic_theory") for e in entries])
        )
        series["linear_learnability_empirical"].append(
            _finite_mean([e.get("linear_learnability_empirical") for e in entries])
        )
        series["cubic_learnability_empirical"].append(
            _finite_mean([e.get("cubic_learnability_empirical") for e in entries])
        )

        lin_th, cub_th = [], []
        lin_gpr, cub_gpr = [], []
        for e in entries:
            kappa = e.get("kappa_eff", e.get("kappa"))
            p_val = e.get("P")
            if not (np.isfinite(kappa) and np.isfinite(p_val)):
                continue
            lin_th.append(
                learnability_from_eigenvalue(e.get("linear_theory"), kappa, p_val)
            )
            cub_th.append(
                learnability_from_eigenvalue(e.get("cubic_theory"), kappa, p_val)
            )
            if gpr_by_dir is not None:
                hit = gpr_by_dir.get(e.get("model_dir", ""))
                if hit is not None:
                    lin_gpr.append(hit.get("linear"))
                    cub_gpr.append(hit.get("cubic"))
        series["linear_learnability_theory"].append(_finite_mean(lin_th))
        series["cubic_learnability_theory"].append(_finite_mean(cub_th))
        series["linear_learnability_arcsin_gpr"].append(_finite_mean(lin_gpr))
        series["cubic_learnability_arcsin_gpr"].append(_finite_mean(cub_gpr))
        series["P"].append(_finite_mean([e.get("P") for e in entries]))
        series["kappa_eff"].append(
            _finite_mean([e.get("kappa_eff", e.get("kappa")) for e in entries])
        )

        # d, N from first entry's name when available
        name = entries[0].get("model_name", "")
        dm = re.search(r"_d(\d+)_", name)
        nm = re.search(r"_N(\d+)_", name)
        series["d"].append(int(dm.group(1)) if dm else float("nan"))
        series["N"].append(int(nm.group(1)) if nm else float("nan"))

    return series


def plot_eigenvalue_variance_vs_beta(series: dict, output_dir: Path, group_name: str):
    betas = series["beta"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    specs = [
        (
            axes[0],
            "Linear eigenvalue vs beta",
            series["linear_empirical"],
            series["linear_theory"],
            "royalblue",
        ),
        (
            axes[1],
            "Cubic eigenvalue vs beta",
            series["cubic_empirical"],
            series["cubic_theory"],
            "forestgreen",
        ),
    ]
    for ax, title, emp, theory, color in specs:
        emp_arr = np.asarray(emp, dtype=np.float64)
        th_arr = np.asarray(theory, dtype=np.float64)
        ax.plot(betas, emp_arr, "o-", color=color, markersize=6, label="Empirical")
        ax.plot(betas, th_arr, "s--", color="black", markersize=5, linewidth=1.6, label="Theory")
        ax.set_title(title)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Eigenvalue / variance")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        annotate_theory_empirical_gaps(ax, betas, emp_arr, th_arr, None)

    annotate_scaling_exponents(fig, group_name)
    out = output_dir / "eigenvalue_variance_vs_beta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_learnability_vs_beta(series: dict, output_dir: Path, group_name: str):
    betas = series["beta"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharex=True)
    specs = [
        (
            axes[0],
            "Linear test learnability vs beta",
            series["linear_learnability_empirical"],
            series["linear_learnability_theory"],
            series.get("linear_learnability_arcsin_gpr"),
            "royalblue",
            0.0,
        ),
        (
            axes[1],
            "Cubic test learnability vs beta",
            series["cubic_learnability_empirical"],
            series["cubic_learnability_theory"],
            series.get("cubic_learnability_arcsin_gpr"),
            "forestgreen",
            None,
        ),
    ]
    for ax, title, emp, theory, gpr, color, ylim_bottom in specs:
        emp_arr = np.asarray(emp, dtype=np.float64)
        th_arr = np.asarray(theory, dtype=np.float64)
        ax.plot(betas, emp_arr, "o-", color=color, markersize=6, label="Empirical (test)")
        ax.plot(betas, th_arr, "s--", color="black", markersize=5, linewidth=1.6, label="Theory")
        if gpr is not None:
            gpr_arr = np.asarray(gpr, dtype=np.float64)
            if np.any(np.isfinite(gpr_arr)):
                ax.plot(
                    betas,
                    gpr_arr,
                    "^-",
                    color="darkorange",
                    markersize=6,
                    linewidth=1.6,
                    label=r"arcsin GPR-KRR ($\sigma^2=\kappa_{\mathrm{eff}}$)",
                )
        ax.set_title(title)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Learnability")
        ax.set_xscale("log")
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
        ax.grid(True, alpha=0.3)
        ax.legend()

    annotate_scaling_exponents(fig, group_name)
    out = output_dir / "learnability_vs_beta.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_empirical_actions_vs_beta(empirical_curves: list[dict], output_dir: Path, group_name: str):
    curves = []
    for entry in empirical_curves:
        beta = beta_from_entry(entry)
        if beta is None:
            continue
        curves.append((beta, entry))
    if not curves:
        print("No empirical curves with parseable beta; skipping action overlays.")
        return

    curves.sort(key=lambda t: t[0])
    # Attach beta onto entries so curve_color_scale can key continuously by beta.
    for beta, entry in curves:
        entry["beta"] = beta
    color_for, cmap, norm, color_label = curve_color_scale([e for _, e in curves])

    fig = plt.figure(figsize=(11, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1], sharey=ax_h)
    cax = fig.add_subplot(gs[0, 2])

    for beta, entry in curves:
        color = color_for(entry)
        for ax, key in ((ax_h, "hermite3"), (ax_l, "linear")):
            centers, action = entry[key]
            mask = np.isfinite(action)
            ax.plot(centers[mask], action[mask], color=color, linewidth=1.8, alpha=0.9)

    for ax, title in ((ax_h, "Hermite3"), (ax_l, "Linear")):
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_xlabel("Projection value", fontsize=14)
        ax.grid(True, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_h.set_ylabel(r"Action: $-\log \mathcal{P}$", fontsize=14)
    ax_l.tick_params(labelleft=False)
    fig.suptitle(f"Empirical first-layer activation actions vs beta | {group_name}", fontsize=15, y=0.98)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(color_label, fontsize=13)

    out = output_dir / "empirical_h0_activation_actions_vs_beta.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Theory overlay variant
    fig_t = plt.figure(figsize=(11, 6), dpi=300)
    gs_t = fig_t.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)
    ax_ht = fig_t.add_subplot(gs_t[0, 0])
    ax_lt = fig_t.add_subplot(gs_t[0, 1], sharey=ax_ht)
    cax_t = fig_t.add_subplot(gs_t[0, 2])

    for beta, entry in curves:
        color = color_for(entry)
        for ax, key, lkey in (
            (ax_ht, "hermite3", "lJ3T"),
            (ax_lt, "linear", "lJ1T"),
        ):
            centers, action = entry[key]
            mask = np.isfinite(action)
            ax.plot(centers[mask], action[mask], color=color, linewidth=1.8, alpha=0.9)
            lJ = entry.get(lkey)
            if lJ is not None and lJ > 0 and np.any(mask):
                x = np.linspace(float(centers[mask].min()), float(centers[mask].max()), 1000)
                gauss = 0.5 * x**2 / float(lJ) + 0.5 * np.log(2.0 * np.pi * float(lJ))
                ax.plot(x, gauss, "--", color=color, linewidth=1.3, alpha=0.95)

    for ax, title in ((ax_ht, "Hermite3"), (ax_lt, "Linear")):
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_xlabel("Projection value", fontsize=14)
        ax.grid(True, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_ht.set_ylabel(r"Action: $-\log \mathcal{P}$", fontsize=14)
    ax_lt.tick_params(labelleft=False)
    fig_t.suptitle(
        f"Empirical + VGA Gaussian theory vs beta | {group_name}", fontsize=15, y=0.98
    )
    sm_t = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_t.set_array([])
    cb_t = fig_t.colorbar(sm_t, cax=cax_t)
    cb_t.set_label(color_label, fontsize=13)
    out_t = output_dir / "empirical_h0_activation_actions_vs_beta_theory_overlay.pdf"
    fig_t.savefig(out_t, bbox_inches="tight")
    plt.close(fig_t)
    print(f"Saved {out_t}")


def plot_group_vs_beta(
    group_name: str,
    ensure_cache: bool = False,
    recompute: bool = False,
    recompute_gpr: bool = False,
    P_test: int | None = None,
):
    if group_name not in EXPERIMENT_GROUP_BY_NAME:
        raise KeyError(f"Unknown group {group_name}")
    group = EXPERIMENT_GROUP_BY_NAME[group_name]
    output_dir = OUTPUT_BASE_DIR / slugify(group_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{group_name}: requiring A_snapshots/ and averaging empirical "
        f"quantities over Langevin snapshots (predictive / pooled posterior)"
    )
    snap_counts = require_a_snapshots(list(group.model_dirs))
    for model_dir, t_snaps in snap_counts.items():
        print(f"  {Path(model_dir).name}: T={t_snaps} snapshots")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cached = load_group_cache(group_name)
    need_recompute = (
        recompute
        or ensure_cache
        or cached is None
        or not cache_uses_a_snapshots(cached, snap_counts)
    )
    if need_recompute:
        reason = (
            "explicit --recompute/--ensure-cache"
            if (recompute or ensure_cache)
            else "missing or non-snapshot cache"
        )
        print(
            f"Recomputing plot_action_h0_activation cache with A_snapshots "
            f"averaging ({reason})"
        )
        plot_experiment_group(group, device, recompute=True, vga_advanced=False)
        cached = load_group_cache(group_name)
        if not cache_uses_a_snapshots(cached, snap_counts):
            raise RuntimeError(
                "Cache still missing snapshot-ensemble markers after recompute. "
                "Check plot_action_h0_activation.load_model(use_a_snapshots=True)."
            )
    else:
        print(
            f"Using snapshot-averaged cache v{REQUIRED_CACHE_VERSION} at "
            f"{output_dir / 'computation_cache.pkl'}"
        )

    entries = cached.get("p_summary_entries", [])
    print(f"Computing arcsin GPR-KRR learnability (sigma^2=kappa_eff) on {device}")
    gpr_by_dir = load_or_compute_gpr_cache(
        group_name,
        entries,
        device,
        P_test=P_test,
        recompute=recompute_gpr or recompute,
    )
    series = build_beta_series(entries, gpr_by_dir=gpr_by_dir)
    if not series["beta"]:
        raise RuntimeError(f"No beta-parseable summary entries for {group_name}")

    print(f"{group_name}: plotting {len(series['beta'])} beta points")
    for i, b in enumerate(series["beta"]):
        print(
            f"  beta={b:.4g}  d={series['d'][i]}  N={series['N'][i]}  "
            f"P={series['P'][i]:.0f}  kappa_eff={series['kappa_eff'][i]:.4g}  "
            f"He1={series['linear_learnability_empirical'][i]:.4f}  "
            f"He3={series['cubic_learnability_empirical'][i]:.4f}  "
            f"GPR_He1={series['linear_learnability_arcsin_gpr'][i]:.4f}  "
            f"GPR_He3={series['cubic_learnability_arcsin_gpr'][i]:.4f}"
        )

    plot_eigenvalue_variance_vs_beta(series, output_dir, group_name)
    plot_learnability_vs_beta(series, output_dir, group_name)
    plot_empirical_actions_vs_beta(cached.get("empirical_curves", []), output_dir, group_name)

    # compact text summary
    summary_path = output_dir / "beta_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"group={group_name}\n")
        f.write("empirical_source=A_snapshots_stacked_as_ensemble\n")
        f.write(f"arcsin_gpr_sigma2=kappa_eff  P_test={P_test}\n")
        f.write(
            f"{'beta':>8} {'d':>5} {'N':>6} {'P':>6} {'kappa_eff':>10} "
            f"{'He1_emp':>8} {'He1_th':>8} {'He1_gpr':>8} "
            f"{'He3_emp':>8} {'He3_th':>8} {'He3_gpr':>8}\n"
        )
        for i, beta in enumerate(series["beta"]):
            f.write(
                f"{beta:8.4g} {series['d'][i]:5.0f} {series['N'][i]:6.0f} "
                f"{series['P'][i]:6.0f} {series['kappa_eff'][i]:10.4g} "
                f"{series['linear_learnability_empirical'][i]:8.4f} "
                f"{series['linear_learnability_theory'][i]:8.4f} "
                f"{series['linear_learnability_arcsin_gpr'][i]:8.4f} "
                f"{series['cubic_learnability_empirical'][i]:8.4f} "
                f"{series['cubic_learnability_theory'][i]:8.4f} "
                f"{series['cubic_learnability_arcsin_gpr'][i]:8.4f}\n"
            )
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot action/h0 summaries vs beta. For Asnap groups, requires "
            "A_snapshots/ and averages empirical quantities over those snapshots "
            "(stacked as the ensemble)."
        )
    )
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        choices=list(EXPERIMENT_GROUP_BY_NAME),
        help="Experiment group (repeatable). Default: all BETA_INDEXED_GROUPS.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List default beta-indexed groups and exit",
    )
    parser.add_argument(
        "--ensure-cache",
        action="store_true",
        help="Run plot_action_h0_activation first if cache is missing",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force full recompute via plot_action_h0_activation before beta plots",
    )
    parser.add_argument(
        "--recompute-gpr",
        action="store_true",
        help="Recompute arcsin GPR-KRR learnability even if its cache exists",
    )
    parser.add_argument(
        "--P-test",
        type=int,
        default=None,
        help="Held-out size for GPR learnability (default: same as train P)",
    )
    args = parser.parse_args()

    if args.list_groups:
        for name in BETA_INDEXED_GROUPS:
            g = EXPERIMENT_GROUP_BY_NAME.get(name)
            n = len(g.model_dirs) if g is not None else 0
            print(f"{name}: {n} model dirs")
        return

    selected = args.groups if args.groups else list(BETA_INDEXED_GROUPS)
    for name in selected:
        plot_group_vs_beta(
            name,
            ensure_cache=args.ensure_cache or args.recompute,
            recompute=args.recompute,
            recompute_gpr=args.recompute_gpr,
            P_test=args.P_test,
        )


if __name__ == "__main__":
    main()
