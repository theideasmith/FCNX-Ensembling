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


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "lib").exists() and (candidate / "milestones").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {current}")


REPO_ROOT = find_repo_root()
LIB_PATH = REPO_ROOT / "lib"
MILESTONE_DIR = REPO_ROOT / "milestones" / "activation_generic_erf_mf_scaling_convergence"
DEFAULT_SCAN_DIR = MILESTONE_DIR / "d10_P_scan_kappa_0.1"
sys.path.insert(0, str(LIB_PATH))

from FCN3Network import FCN3NetworkActivationGeneric

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

DEVICE = torch.device("cuda")
DTYPE = torch.float64

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "figure.dpi": 140,
    }
)


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
            except (TypeError, ValueError):
                continue
    return int(default_seed)


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
        raise FileNotFoundError(
            f"No config.json found and run name did not match expected pattern: {run_dir}"
        )
    return {
        "d": int(match.group("d")),
        "P": int(match.group("P")),
        "N": int(match.group("N")),
        "chi": float(match.group("chi")),
        "kappa": float(match.group("kappa")),
        "activation": "erf",
    }


def find_checkpoint_file(run_dir: Path) -> Path:
    candidates = [run_dir / "model_final.pt", run_dir / "model.pt", run_dir / "checkpoint.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def normalize_state_dict(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state and "W0" not in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")
    if "W0" in state and state["W0"].ndim == 4:
        state["W0"] = state["W0"].squeeze(0)
        state["W1"] = state["W1"].squeeze(0)
        state["A"] = state["A"].squeeze(0)
    return state


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device = DEVICE):
    run_dir = checkpoint_path.parent
    cfg = load_run_config(run_dir)
    d = int(cfg["d"])
    n1_value = cfg.get("N") or cfg.get("n1") or cfg.get("n")
    if n1_value is None:
        raise KeyError(f"Could not infer hidden width from config in {run_dir}")
    n1 = int(n1_value)
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


def make_gaussian_dataset(
    d: int, num_samples: int, seed: int, device: torch.device = DEVICE
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    return torch.randn(num_samples, d, generator=generator, dtype=DTYPE, device=device)


def compute_theory_eigenvalues(
    d: int, P: int, N: int, chi: float, kappa: float, eps: float
) -> dict:
    """Call Julia solver and extract He1/He3 target/perp theoretical eigenvalues."""
    julia_script = REPO_ROOT / "julia_lib" / "eos_fcn3erf.jl"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        to_path = Path(tf.name)

    cmd = [
        "julia", str(julia_script),
        f"--d={d}", f"--P={P}", f"--n1={N}", f"--n2={N}",
        f"--chi={chi}", f"--kappa={kappa}", f"--epsilon={eps}",
        f"--to={to_path}", "--quiet",
    ]

    data = {}
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(to_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: Julia theory solver failed for d={d}, P={P}, N={N}: {exc}")
    finally:
        try:
            to_path.unlink(missing_ok=True)
        except Exception:
            pass

    target = data.get("target", {}) if isinstance(data, dict) else {}
    perp   = data.get("perpendicular", {}) if isinstance(data, dict) else {}
    return {
        "lH1T": float(target["lH1T"]) if target.get("lH1T") is not None else None,
        "lH1P": float(perp["lH1P"])   if perp.get("lH1P")   is not None else None,
        "lH3T": float(target["lH3T"]) if target.get("lH3T") is not None else None,
        "lH3P": float(perp["lH3P"])   if perp.get("lH3P")   is not None else None,
    }


def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute the infinite-width erf-network (arcsin) kernel matrix for inputs X (n, d)."""
    XXT = torch.einsum("ui,vi->uv", X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = 2 * XXT / denom
    return (2 / torch.pi) * torch.arcsin(arg)


def _hermite_vecs(X: torch.Tensor) -> dict:
    """
    Evaluate the He1 and He3 target eigenfunctions on the rows of X,
    normalized by 1/sqrt(P) to match field-theory eigenvalue conventions.

        phi_1(x) = x_0                         (He1, target direction)
        phi_3(x) = (x_0^3 - 3*x_0) / sqrt(6)  (He3, target direction)

    Dividing by sqrt(n) means the outer product phi phi^T has entries ~ O(1),
    consistent with the Gram matrix K_train ~ O(1), so that delta_lambda
    (in Gram-matrix units = field-theory units * P) directly gives the
    correct spectral shift in the same units as K_train.
    """
    x0 = X[:, 0]
    n  = x0.shape[0]

    phi1 = x0                                    
    phi3 = (x0 ** 3 - 3.0 * x0) / math.sqrt(6.0) 

    return {"h1t": phi1, "h3t": phi3}


def gpr_rank1_corrected(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    ridge:   float,
    theory_eigs: dict,
    P: int,
) -> tuple:
    """
    GPR with the bare arcsin kernel plus rank-1 eigenvalue corrections for the
    He1 and He3 target modes.

    Kernel correction
    -----------------
    The corrected kernel is simply:

        K_corr(x, x') = K_arcsin(x, x')
                      + delta_lambda_1 * phi_1(x) phi_1(x')
                      + delta_lambda_3 * phi_3(x) phi_3(x')

    where:
        phi_1(x) = x_0 / sqrt(P)
        phi_3(x) = (x_0^3 - 3*x_0) / (sqrt(6) * sqrt(P))
        delta_lambda_k = (lH{1,3}T - lH{1,3}P) * P

    lH{1,3}T and lH{1,3}P are the theory eigenvalues in field-theory units
    (i.e. divided by P); multiplying by P converts to Gram-matrix units where
    K_train ~ O(1).  lH{1,3}P is the bare arcsin kernel eigenvalue (isotropic,
    same in every direction); lH{1,3}T is the DNN-corrected target eigenvalue.

    In sample space the correction for each mode k is the rank-1 matrix:
        delta_lambda_k * phi_k(X_train) phi_k(X_train)^T   added to K_train
        delta_lambda_k * phi_k(X_test)  phi_k(X_train)^T   added to K_cross

    The corrected matrices are then passed to the standard Cholesky-based
    GPR solver — no Sherman-Morrison or custom inverse update needed.
    """
    n_train = X_train.shape[0]
    X_all   = torch.cat([X_train, X_test], dim=0)
    print(X_all.shape, X_train.shape, X_test.shape)
    # Bare arcsin kernel
    K_all   = arcsin_kernel(X_all)
    K_train = K_all[:n_train, :n_train].clone()
    K_cross = K_all[n_train:, :n_train].clone()

    # Hermite feature vectors, normalized by 1/sqrt(n) so that the outer
    # product is in the same units as K_train.
    phi_tr = _hermite_vecs(X_train)
    phi_te = _hermite_vecs(X_test)

    for vec_key, eig_key_adapted, eig_key_gpr in [
        ("h1t", "lH1T", "lH1P"),
        ("h3t", "lH3T", "lH3P"),
    ]:
        lam_adapted_ft = theory_eigs.get(eig_key_adapted)
        lam_gpr_ft     = theory_eigs.get(eig_key_gpr)
        if lam_adapted_ft is None or lam_gpr_ft is None:
            print(f"Warning: missing theory eigenvalue for {vec_key}, skipping correction.")
            continue

        # Convert from field-theory units (÷P) to Gram-matrix units (×P).
        delta_lambda = (float(lam_adapted_ft) - float(lam_gpr_ft)) * X_all.shape[0]

        phi_train = phi_tr[vec_key]   # shape (n_train,)
        phi_test  = phi_te[vec_key]   # shape (n_test,)

        # Rank-1 update: add delta_lambda * phi phi^T to the relevant blocks.
        K_train = K_train + delta_lambda * torch.outer(phi_train, phi_train)
        K_cross = K_cross + delta_lambda * torch.outer(phi_test,  phi_train)

    # Standard Cholesky-based GPR solve on the corrected kernel.
    eye   = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    chol  = torch.linalg.cholesky(K_train + ridge * eye)
    alpha = torch.cholesky_solve(y_train[:, None], chol).squeeze(-1)
    y_pred = K_cross @ alpha

    return y_pred, K_train, K_all


# ---------------------------------------------------------------------------
# Baseline GPR helpers
# ---------------------------------------------------------------------------

def gpr_from_empirical_kernel(
    model, X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, ridge: float,
):
    X_all   = torch.cat([X_train, X_test], dim=0)
    K_all   = arcsin_kernel(X_all)
    n_train = X_train.shape[0]
    K_train = K_all[:n_train, :n_train]
    K_cross = K_all[n_train:, :n_train]
    eye  = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    chol = torch.linalg.cholesky(K_train + ridge * eye)
    alpha  = torch.cholesky_solve(y_train[:, None], chol).squeeze(-1)
    y_pred = K_cross @ alpha
    return y_pred, K_train, K_all


def gpr_from_kernel_matrix(
    K_all: torch.Tensor, X_train: torch.Tensor,
    y_train: torch.Tensor, ridge: float,
):
    """Run kernel ridge regression given a precomputed full kernel matrix."""
    n_train = X_train.shape[0]
    K_train = K_all[:n_train, :n_train]
    K_cross = K_all[n_train:, :n_train]
    eye  = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    chol = torch.linalg.cholesky(K_train + ridge * eye)
    alpha  = torch.cholesky_solve(y_train[:, None], chol).squeeze(-1)
    y_pred = K_cross @ alpha
    return y_pred, K_train, K_all


def h3_learnability_from_predictions(y_pred: torch.Tensor, X: torch.Tensor) -> dict:
    x0 = X[:, 0]
    h3_comp      = hermite_h3(x0)
    linear_coeff = (y_pred * x0).mean()
    remainder    = y_pred - linear_coeff * x0
    proj3        = (remainder * h3_comp).mean()
    return {
        "h1_sum":           float(linear_coeff.item()),
        "h3_sum":           float((proj3 / 0.03).item()),
        "proj3_target_sum": float(proj3.item()),
    }


def learnability_from_eigenvalue(
    eigenvalue: Optional[float], ridge: float, P: int
) -> Optional[float]:
    if eigenvalue is None:
        return None
    return float(eigenvalue / (eigenvalue + ridge / P))


def collapse_model_prediction(y_pred_raw: torch.Tensor) -> torch.Tensor:
    if y_pred_raw.ndim == 1:
        return y_pred_raw
    if y_pred_raw.ndim == 2:
        return y_pred_raw.mean(dim=1)
    return y_pred_raw.reshape(y_pred_raw.shape[0], -1).mean(dim=1)


def find_model_files(scan_dir: Path):
    candidates = (
        list(scan_dir.glob("**/seed*/model.pt")) +
        list(scan_dir.glob("**/seed*/model_final.pt"))
    )
    return sorted({p.resolve() for p in candidates if p.is_file()}, key=str)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_runs(scan_dir: Path, test_size: int = 5000, limit: Optional[int] = None):
    model_files = find_model_files(scan_dir)
    if limit is not None:
        model_files = model_files[:int(limit)]

    print(f"Found {len(model_files)} checkpoint files in {scan_dir}")
    results = []

    for checkpoint_path in model_files:
        cfg, model = load_model_from_checkpoint(checkpoint_path, device=DEVICE)
        run_dir = checkpoint_path.parent

        d   = int(cfg["d"])
        P   = int(cfg["P"])
        N_for_theory = int(cfg.get("N") or cfg.get("n1") or cfg.get("n") or 0)
        if N_for_theory == 0:
            raise KeyError(f"Could not infer hidden width for theory solver in {run_dir}")
        ridge = float(cfg.get("kappa", 0.1))
        chi   = float(cfg.get("chi", 20.0))  if cfg.get("chi")  is not None else 20.0
        eps   = float(cfg.get("eps", 0.03))  if cfg.get("eps")  is not None else 0.03
        train_seed = derive_seed(cfg)
        test_seed  = train_seed + 1_000_000

        X_train = make_gaussian_dataset(d, P,         train_seed, device=DEVICE)
        y_train = target_fn(X_train).to(dtype=DTYPE)
        X_test  = make_gaussian_dataset(d, test_size, test_seed,  device=DEVICE)
        y_test  = target_fn(X_test).to(dtype=DTYPE)

        # Bare empirical-kernel GPR
        X_all  = torch.cat([X_train, X_test], dim=0)
        K_all  = arcsin_kernel(X_all)
        y_pred, K_train, K_all = gpr_from_kernel_matrix(K_all, X_train, y_train, ridge=ridge)
        mse          = torch.mean((y_pred - y_test) ** 2).item()
        learnability = h3_learnability_from_predictions(y_pred, X_test)

        # Theory eigenvalues from Julia
        theory_eigs = compute_theory_eigenvalues(
            d=d, P=P, N=N_for_theory, chi=chi, kappa=ridge, eps=eps
        )
        theory_h1_learnability = learnability_from_eigenvalue(theory_eigs.get("lH1T"), ridge, P)
        theory_h3_learnability = learnability_from_eigenvalue(theory_eigs.get("lH3T"), ridge, P)

        # Rank-1 corrected GPR (He1-target + He3-target only)
        y_pred_theory, K_train_theory, _ = gpr_rank1_corrected(
            X_train=X_train, y_train=y_train, X_test=X_test,
            ridge=ridge, theory_eigs=theory_eigs, P=P,
        )
        mse_theory          = torch.mean((y_pred_theory - y_test) ** 2).item()
        learnability_theory = h3_learnability_from_predictions(y_pred_theory, X_test)

        # Trained model predictions
        with torch.no_grad():
            y_pred_model_raw = model(X_test)
        y_pred_model       = collapse_model_prediction(y_pred_model_raw)
        mse_model          = torch.mean((y_pred_model - y_test) ** 2).item()
        learnability_model = h3_learnability_from_predictions(y_pred_model, X_test)

        results.append({
            "run_dir":    str(run_dir),
            "checkpoint": str(checkpoint_path),
            "d": d, "P": P,
            "N":     int(cfg.get("N", cfg.get("n1", 0))),
            "chi":   float(cfg.get("chi", float("nan"))) if cfg.get("chi") is not None else float("nan"),
            "kappa": ridge,
            "seed":  train_seed,
            "test_seed": test_seed,
            "test_mse":                float(mse),
            "h1_sum":                  learnability["h1_sum"],
            "h3_sum":                  learnability["h3_sum"],
            "proj3_target_sum":        learnability["proj3_target_sum"],
            "test_mse_model":          float(mse_model),
            "h1_sum_model":            learnability_model["h1_sum"],
            "h3_sum_model":            learnability_model["h3_sum"],
            "proj3_target_sum_model":  learnability_model["proj3_target_sum"],
            "test_mse_theory":         float(mse_theory),
            "h1_sum_theory":           learnability_theory["h1_sum"],
            "h3_sum_theory":           learnability_theory["h3_sum"],
            "proj3_target_sum_theory": learnability_theory["proj3_target_sum"],
            "lH1T_theory":  theory_eigs.get("lH1T"),
            "lH1P_theory":  theory_eigs.get("lH1P"),
            "lH3T_theory":  theory_eigs.get("lH3T"),
            "lH3P_theory":  theory_eigs.get("lH3P"),
            "learnability_H1T_theory": theory_h1_learnability,
            "learnability_H3T_theory": theory_h3_learnability,
        })

        print(
            f"P={P:4d} | gpr_mse={mse:.4e} | model_mse={mse_model:.4e} | "
            f"theory_mse={mse_theory:.4e} | gpr_h3={learnability['h3_sum']:.4e} | "
            f"model_h3={learnability_model['h3_sum']:.4e} | "
            f"theory_h3={learnability_theory['h3_sum']:.4e} | {run_dir.name}"
        )

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, out_dir: Path):
    if not results:
        raise RuntimeError("No results to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)

    P_arr          = np.array([r["P"]              for r in results])
    h1_arr         = np.array([r["h1_sum"]          for r in results])
    h1_model_arr   = np.array([r["h1_sum_model"]    for r in results])
    h1_theory_arr  = np.array([r["h1_sum_theory"]   for r in results])
    h1_tlearn_arr  = np.array([r["learnability_H1T_theory"] for r in results], dtype=float)
    h3_arr         = np.array([r["h3_sum"]          for r in results])
    h3_model_arr   = np.array([r["h3_sum_model"]    for r in results])
    h3_theory_arr  = np.array([r["h3_sum_theory"]   for r in results])
    h3_tlearn_arr  = np.array([r["learnability_H3T_theory"] for r in results], dtype=float)
    mse_arr        = np.array([r["test_mse"]        for r in results])
    mse_theory_arr = np.array([r["test_mse_theory"] for r in results])
    unique_P       = np.array(sorted(set(P_arr.tolist())))

    def _mean_std(arr, p):
        mask = P_arr == p
        return float(arr[mask].mean()), float(arr[mask].std(ddof=0))

    def _eb(ax, arr, fmt, color, label):
        ms = [_mean_std(arr, p) for p in unique_P]
        means = [x[0] for x in ms]
        stds  = [x[1] for x in ms]
        ax.errorbar(unique_P, means, yerr=stds, fmt=fmt, lw=2, color=color, label=label)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    for p in unique_P:
        mask = P_arr == p
        ax1.scatter(np.full(mask.sum(), p), h1_arr[mask],        alpha=0.35, s=24, color="tab:green")
        ax1.scatter(np.full(mask.sum(), p), h1_model_arr[mask],  alpha=0.35, s=24, color="tab:olive")
        ax1.scatter(np.full(mask.sum(), p), h1_theory_arr[mask], alpha=0.35, s=24, color="tab:purple")
        ax2.scatter(np.full(mask.sum(), p), h3_arr[mask],        alpha=0.35, s=24, color="tab:blue")
        ax2.scatter(np.full(mask.sum(), p), h3_model_arr[mask],  alpha=0.35, s=24, color="tab:cyan")
        ax2.scatter(np.full(mask.sum(), p), h3_theory_arr[mask], alpha=0.35, s=24, color="tab:pink")
        ax3.scatter(np.full(mask.sum(), p), mse_arr[mask],        alpha=0.6,  s=30, color="tab:red")
        ax3.scatter(np.full(mask.sum(), p), mse_theory_arr[mask], alpha=0.4,  s=26, color="tab:orange")

    _eb(ax1, h1_arr,        "o-",  "tab:green",  "GPR mean ± std")
    _eb(ax1, h1_model_arr,  "s--", "tab:olive",  "Model mean ± std")
    _eb(ax1, h1_theory_arr, "^:",  "tab:purple", "Theory-kernel GPR mean ± std")
    _eb(ax1, h1_tlearn_arr, "d-.", "tab:red",    "Theory He1T learnability")
    _eb(ax2, h3_arr,        "o-",  "tab:blue",   "GPR mean ± std")
    _eb(ax2, h3_model_arr,  "s--", "tab:cyan",   "Model mean ± std")
    _eb(ax2, h3_theory_arr, "^:",  "tab:pink",   "Theory-kernel GPR mean ± std")
    _eb(ax2, h3_tlearn_arr, "d-.", "tab:green",  "Theory He3T learnability")
    _eb(ax3, mse_arr,        "o-",  "tab:red",    "Empirical-kernel GPR mean ± std")
    _eb(ax3, mse_theory_arr, "^:",  "tab:orange", "Theory-kernel GPR mean ± std")

    for ax, title, ylabel in [
        (ax1, "Linear Learnability vs P", "linear learnability"),
        (ax2, "h3 Learnability vs P",     "h3 learnability"),
        (ax3, "Test MSE vs P",            "test MSE"),
    ]:
        ax.axvline(20, color="gray", ls="--", alpha=0.6, label="d=20")
        ax.set_xscale("log")
        ax.set_xlabel("P")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    fig_path  = out_dir / "semi_empirical_effective_ridge_correction.png"
    json_path = out_dir / "semi_empirical_effective_ridge_correction.json"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return fig_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Empirical H_Kernel GPR analysis for d=20 scan runs."
    )
    parser.add_argument("--scan-dir",   type=str, default=str(DEFAULT_SCAN_DIR))
    parser.add_argument("--test-size",  type=int, default=3000)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    scan_dir   = Path(args.scan_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else scan_dir / "semi_empirical_effective_ridge_correction"
    )

    results = evaluate_runs(scan_dir, test_size=args.test_size, limit=args.limit)
    fig_path, json_path = plot_results(results, output_dir)
    print(f"Saved figure to {fig_path}")
    print(f"Saved results to {json_path}")
    return results


if __name__ == "__main__":
    main()