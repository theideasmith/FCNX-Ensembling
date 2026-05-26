#!/usr/bin/env python3
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "lib").exists() and (candidate / "milestones").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {current}")


REPO_ROOT = find_repo_root()
LIB_PATH = REPO_ROOT / "lib"
MILESTONE_DIR = REPO_ROOT / "milestones" / "activation_generic_erf_mf_scaling_convergence"
DEFAULT_SCAN_DIR = MILESTONE_DIR / "p_scan_erf_results"
sys.path.insert(0, str(LIB_PATH))

from FCN3Network import FCN3NetworkActivationGeneric
from kappa_eff_solver import compute_kappa_eff

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

DEVICE = torch.device("cuda")
DTYPE = torch.float64

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "figure.dpi": 140,
})

MODEL_COLOR = "#FF7F0E"
GPR_COLOR = "#2CA02C"
THEORY_COLOR = "#1F77B4"
ERRORBAR_COLOR = "black"


def target_fn(X: torch.Tensor) -> torch.Tensor:
    x0 = X[:, 0]
    return x0 + 0.03 * (x0**3 - 3.0 * x0)


def hermite_h3(x: torch.Tensor) -> torch.Tensor:
    return (x**3 - 3.0 * x) / math.sqrt(6.0)


def derive_seed(cfg: dict, default_seed: int = 4324) -> int:
    for key in ("seed", "base_seed", "torch_seed", "rng_seed"):
        if key in cfg and cfg[key] is not None:
            try:
                return int(cfg[key])
            except Exception:
                continue
    return default_seed


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)

    match = re.search(
        r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi(?P<chi>[-+]?\d*\.?\d+)_kappa(?P<kappa>[-+]?\d*\.?\d+)",
        run_dir.name,
    )
    if not match:
        raise FileNotFoundError(f"No config.json and name did not match pattern: {run_dir}")
    return {
        "d": int(match.group("d")),
        "P": int(match.group("P")),
        "N": int(match.group("N")),
        "chi": float(match.group("chi")),
        "kappa": float(match.group("kappa")),
        "activation": "erf",
    }


def normalize_state_dict(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if "W0" in state and state["W0"].ndim == 4:
        state["W0"] = state["W0"].squeeze(0)
        state["W1"] = state["W1"].squeeze(0)
        state["A"] = state["A"].squeeze(0)
    return state


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device = DEVICE):
    run_dir = checkpoint_path.parent
    cfg = load_run_config(run_dir)
    d = int(cfg["d"])
    n1 = int(cfg.get("N") or cfg.get("n1") or cfg.get("n"))
    n2 = int(cfg.get("n2", n1))
    P = int(cfg["P"])
    ens = int(cfg.get("ens", 1))
    activation = str(cfg.get("activation", "erf")).lower()

    model = FCN3NetworkActivationGeneric(
        d, n1, n2, P, ens=ens, activation=activation, device=device,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    state = normalize_state_dict(state)
    model.load_state_dict(state)
    model = model.double().eval()
    return cfg, model


def make_gaussian_dataset(d: int, num_samples: int, seed: int, device: torch.device = DEVICE) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    return torch.randn(num_samples, d, generator=generator, dtype=DTYPE, device=device)


def compute_theory_eigenvalues(d: int, P: int, N: int, chi: float, kappa: float, eps: float = 0.03) -> dict:
    julia_script = REPO_ROOT / "julia_lib" / "eos_fcn3erf.jl"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia", str(julia_script),
        f"--d={d}", f"--P={P}", f"--n1={N}", f"--n2={N}",
        f"--chi={chi}", f"--kappa={kappa}", f"--epsilon={eps}",
        f"--to={to_path}", "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(to_path, "r") as f:
            data = json.load(f)
        target = data.get("target", {})
        return {
            "lH1T": float(target.get("lH1T")) if target.get("lH1T") is not None else None,
            "lH3T": float(target.get("lH3T")) if target.get("lH3T") is not None else None,
        }
    except Exception as exc:
        print(f"Warning: Julia theory solver failed: {exc}")
        return {"lH1T": None, "lH3T": None}
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    XXT = torch.einsum("ui,vi->uv", X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def gpr_from_kernel_matrix(K_all: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor, ridge: float):
    n_train = X_train.shape[0]
    K_train = K_all[:n_train, :n_train]
    K_cross = K_all[n_train:, :n_train]
    eye = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    chol = torch.linalg.cholesky(K_train + ridge * eye)
    alpha = torch.cholesky_solve(y_train[:, None], chol).squeeze(-1)
    y_pred = K_cross @ alpha
    return y_pred, K_train, K_all


def h3_learnability_from_predictions(y_pred: torch.Tensor, X: torch.Tensor, X_train: Optional[torch.Tensor] = None) -> dict:
    x0_test = X[:, 0]
    h3_comp = hermite_h3(x0_test)
    y_true = target_fn(X)

    true_he1 = (y_true * x0_test).mean()
    true_rem = y_true - true_he1 * x0_test
    true_proj3 = (true_rem * h3_comp).mean()

    pred_he1 = (y_pred * x0_test).mean()
    pred_rem = y_pred - pred_he1 * x0_test
    pred_proj3 = (pred_rem * h3_comp).mean()

    h1_sum = float(pred_he1 / true_he1) if abs(float(true_he1)) > 1e-10 else 0.0
    h3_sum = float(pred_proj3 / true_proj3) if abs(float(true_proj3)) > 1e-10 else 0.0

    return {
        "h1_sum": h1_sum,
        "h3_sum": h3_sum,
        "proj3_target_sum": float(pred_proj3),
    }


def learnability_from_eigenvalue(eigenvalue: Optional[float], ridge: float, P: int) -> Optional[float]:
    if eigenvalue is None or not math.isfinite(eigenvalue):
        return None
    return float(eigenvalue / (eigenvalue + ridge / P))


def collapse_model_prediction(y_pred_raw: torch.Tensor) -> torch.Tensor:
    if y_pred_raw.ndim == 1:
        return y_pred_raw
    if y_pred_raw.ndim == 2:
        return y_pred_raw.mean(dim=1)
    return y_pred_raw.reshape(y_pred_raw.shape[0], -1).mean(dim=1)


def find_model_files(scan_dir: Path):
    patterns = [
        "**/*seed*/model_final.pt",
        "**/*seed*/model.pt",
        "**/base_seed*/model_final.pt",
        "**/base_seed*/model.pt",
        "**/model_final.pt",
        "**/model.pt",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(list(scan_dir.glob(pat)))

    best_by_run_dir = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        run_dir = candidate.parent.resolve()
        score = 0 if candidate.name == "model_final.pt" else 1
        current = best_by_run_dir.get(run_dir)
        if current is None or score < current[0] or (score == current[0] and str(candidate) < str(current[1])):
            best_by_run_dir[run_dir] = (score, candidate.resolve())
    return sorted((entry[1] for entry in best_by_run_dir.values()), key=str)


def evaluate_runs(scan_dir: Path, test_size: int = 5000, limit: Optional[int] = None):
    model_files = find_model_files(scan_dir)
    if limit is not None:
        model_files = model_files[:int(limit)]

    print(f"Found {len(model_files)} checkpoint files in {scan_dir}")
    results = []
    kappa_eff_cache: dict = {}

    for checkpoint_path in model_files:
        cfg, model = load_model_from_checkpoint(checkpoint_path, device=DEVICE)
        run_dir = checkpoint_path.parent

        d = int(cfg["d"])
        P = int(cfg["P"])
        N = int(cfg.get("N") or cfg.get("n1") or cfg.get("n") or 0)
        ridge = float(cfg.get("kappa", 0.1))
        chi = float(cfg.get("chi", 20.0))
        eps = float(cfg.get("eps", 0.03))

        train_seed = derive_seed(cfg)
        test_seed = train_seed + 1_000_000

        key = (d, P, chi, ridge)
        if key not in kappa_eff_cache:
            try:
                kappa_eff_cache[key] = compute_kappa_eff(
                    d=d, P=P, kappa_bare=ridge, n1=N, n2=N,
                    chi=chi, num_samples=P, device=DEVICE, verbose=False
                )
            except Exception:
                kappa_eff_cache[key] = ridge
        kappa_eff = kappa_eff_cache[key]

        X_train = make_gaussian_dataset(d, P, train_seed, device=DEVICE)
        y_train = target_fn(X_train).to(dtype=DTYPE)
        X_test = make_gaussian_dataset(d, test_size, test_seed, device=DEVICE)
        y_test = target_fn(X_test).to(dtype=DTYPE)

        # Plain NNGP GPR
        X_all = torch.cat([X_train, X_test], dim=0)
        K_all = arcsin_kernel(X_all)
        y_pred_gpr, _, _ = gpr_from_kernel_matrix(K_all, X_train, y_train, ridge=kappa_eff)

        mse_gpr = torch.mean((y_pred_gpr - y_test) ** 2).item()
        learn_gpr = h3_learnability_from_predictions(y_pred_gpr, X_test, X_train)

        # Trained Model
        with torch.no_grad():
            y_pred_model_raw = model(X_test)
        y_pred_model = collapse_model_prediction(y_pred_model_raw)
        mse_model = torch.mean((y_pred_model - y_test) ** 2).item()
        learn_model = h3_learnability_from_predictions(y_pred_model, X_test, X_train)

        # Raw Theory
        theory_eigs = compute_theory_eigenvalues(d, P, N, chi, kappa_eff, eps)
        theory_h1 = learnability_from_eigenvalue(theory_eigs.get("lH1T"), kappa_eff, P)
        theory_h3 = learnability_from_eigenvalue(theory_eigs.get("lH3T"), kappa_eff, P)

        results.append({
            "run_dir": str(run_dir),
            "d": d, "P": P,
            "chi": chi,
            "kappa": ridge,
            "kappa_eff": float(kappa_eff),

            "test_mse": float(mse_gpr),
            "h1_sum": float(learn_gpr["h1_sum"]),
            "h3_sum": float(learn_gpr["h3_sum"]),

            "test_mse_model": float(mse_model),
            "h1_sum_model": float(learn_model["h1_sum"]),
            "h3_sum_model": float(learn_model["h3_sum"]),

            "lH1T_theory": theory_eigs.get("lH1T"),
            "lH3T_theory": theory_eigs.get("lH3T"),
            "h1_sum_theory": float(theory_h1) if theory_h1 is not None else None,
            "h3_sum_theory": float(theory_h3) if theory_h3 is not None else None,
        })

        print(f"P={P:4d} | d={d} | model={mse_model:.2e} | gpr={mse_gpr:.2e}")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results


def plot_all_d(results, out_dir: Path, separate_plots: bool = False, filename_suffix: str = ""):
    if not results:
        raise RuntimeError("No results to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)

    d_values = sorted({int(r["d"]) for r in results})
    print(f"Plotting all d values: {d_values}")

    # Color mapping
    colors = ["#7B2CBF", "#FF7F0E", "#2CA02C", "#1F77B4", "#E15759", "#76B7B2", "#F1CE63", "#B07AA1"]
    d_color = {d: colors[i % len(colors)] for i, d in enumerate(d_values)}
    d_color[10] = "#7B2CBF"  # purple
    d_color[15] = "#FF7F0E"  # orange

    def _mean_std(arr, p):
        mask = np.array([r["P"] for r in arr]) == p
        vals = np.array([r[key] for r in arr if mask[i]], dtype=float)  # wait, better implementation below
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.0, 0.0
        return float(vals.mean()), float(vals.std(ddof=0) / math.sqrt(len(vals)))

    def plot_mode(ax, key_model, key_gpr, key_theory, title):
        for d_val in d_values:
            d_res = [r for r in results if int(r["d"]) == d_val]
            color = d_color[d_val]
            p_values = sorted({int(r["P"]) for r in d_res})

            # Model
            means = [np.mean([r[key_model] for r in d_res if r["P"] == p]) for p in p_values]
            stds = [np.std([r[key_model] for r in d_res if r["P"] == p], ddof=0) / np.sqrt(len([r for r in d_res if r["P"] == p])) for p in p_values]
            ax.errorbar(p_values, means, yerr=stds, fmt="s--", lw=2, color=color, label=f"Model d={d_val}")

            # NNGP GPR
            means = [np.mean([r[key_gpr] for r in d_res if r["P"] == p]) for p in p_values]
            stds = [np.std([r[key_gpr] for r in d_res if r["P"] == p], ddof=0) / np.sqrt(len([r for r in d_res if r["P"] == p])) for p in p_values]
            ax.errorbar(p_values, means, yerr=stds, fmt="o-", lw=2, color=color, label=f"NNGP d={d_val}")

            # Theory
            means = [np.mean([r[key_theory] for r in d_res if r["P"] == p and r[key_theory] is not None]) for p in p_values]
            stds = [np.std([r[key_theory] for r in d_res if r["P"] == p and r[key_theory] is not None], ddof=0) / np.sqrt(len([r for r in d_res if r["P"] == p and r[key_theory] is not None])) for p in p_values]
            ax.errorbar(p_values, means, yerr=stds, fmt="d-.", lw=2, color=color, label=f"Theory d={d_val}")

    if separate_plots:
        for mode, ylabel in [("h1", "linear learnability"), ("h3", "He3 learnability")]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            plot_mode(ax, f"{mode}_sum_model", f"{mode}_sum", f"{mode}_sum_theory", f"He{mode[1]} Learnability")
            ax.set_xscale("log")
            ax.set_xlabel("P")
            ax.set_ylabel(ylabel)
            ax.set_title(f"All d — He{mode[1]} Learnability")
            ax.grid(alpha=0.3)
            ax.legend(ncol=2, fontsize=10)
            fig.savefig(out_dir / f"all_d_he{mode[1]}{filename_suffix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        plot_mode(ax1, "h1_sum_model", "h1_sum", "h1_sum_theory", "He1")
        plot_mode(ax2, "h3_sum_model", "h3_sum", "h3_sum_theory", "He3")

        for ax, title in [(ax1, "He1 Learnability — All d"), (ax2, "He3 Learnability — All d")]:
            ax.set_xscale("log")
            ax.set_xlabel("P")
            ax.set_ylabel("Learnability")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(ncol=2, fontsize=10)

        plt.tight_layout()
        fig.savefig(out_dir / f"all_d_model_nngp_theory{filename_suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Save results
    with open(out_dir / f"all_d_results{filename_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model vs NNGP vs Theory — All d in one plot")
    parser.add_argument("--scan-dir", type=str, default=str(DEFAULT_SCAN_DIR))
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--separate-plots", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args(argv)

    scan_dir = Path(args.scan_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else scan_dir / "model_vs_nngp_vs_theory_all_d"

    results = None
    if args.use_cache:
        cache_files = sorted(output_dir.glob("**/*results*.json"))
        if cache_files:
            print(f"Loading cache: {cache_files[0]}")
            with open(cache_files[0]) as f:
                results = json.load(f)

    if results is None:
        results = evaluate_runs(scan_dir, test_size=args.test_size, limit=args.limit)

    if not results:
        print("No results found.")
        return

    plot_all_d(results, output_dir, separate_plots=args.separate_plots)


if __name__ == "__main__":
    main()