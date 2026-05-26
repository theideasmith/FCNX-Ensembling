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
from matplotlib import cm
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
DEFAULT_SCAN_DIR = MILESTONE_DIR / "p_scan_erf_results"
sys.path.insert(0, str(LIB_PATH))

from FCN3Network import FCN3NetworkActivationGeneric
from kappa_eff_solver import compute_kappa_eff

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

DEVICE = torch.device("cuda")
DTYPE = torch.float64

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 140,
    }
)

MODEL_COLOR = "#FF7F0E"
THEORY_COLOR = "#1F77B4"
GPR_COLOR = "#2CA02C"
MSE_COLOR = "#E15759"
ERRORBAR_COLOR = "black"
ALL_D_COLORS = [
    "#7B2CBF",
    "#F77F00",
    "#2A9D8F",
    "#577590",
    "#D62828",
    "#3A86FF",
    "#8AC926",
    "#FF006E",
    "#8338EC",
    "#FB5607",
]
GPR_BASELINE_COLOR = "#59A14F"


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
    candidates = [run_dir / "model_final.pt"]
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


def compute_alpha_eff(eigenvalues: torch.Tensor, k: int, alpha: float) -> float:
    """
    Compute the shrinkage-corrected alpha for a rank-one operator update
    targeting the k-th eigenvector of a finite kernel matrix.

    Args:
        eigenvalues: 1D tensor of eigenvalues (sorted or unsorted, length P)
        k: index of the target eigenvector
        alpha: nominal update strength from operator theory

    Returns:
        alpha_eff = alpha * cos2_theta
    """
    lam = eigenvalues.float()
    lam_k = lam[k]
    lam_rest = torch.cat([lam[:k], lam[k + 1 :]])

    gaps = lam_k - lam_rest
    cos2_theta = 1.0 / (1.0 + (1.0 / len(lam)) * (1.0 / gaps**2).sum().item())

    return float(alpha * cos2_theta)


def kappa_eff_cache_key(cfg: dict) -> tuple:
    return (
        int(cfg["d"]),
        int(cfg["P"]),
        float(cfg.get("chi", float("nan"))),
        float(cfg.get("kappa", float("nan"))),
    )


# ---------------------------------------------------------------------------
# Rank-1 Sherman-Morrison kernel correction
# ---------------------------------------------------------------------------

def _hermite_vecs(X: torch.Tensor) -> dict:
    """
    Raw He1 and He3 eigenfunctions evaluated on the rows of X.

        phi_1(x) = x_0
        phi_3(x) = (x_0^3 - 3*x_0) / sqrt(6)

    No extra normalization — the field-theory convention is phi^T phi / P ≈ 1,
    which these satisfy naturally since E[He_k(x_0)^2] = 1 under the Gaussian
    measure.  All normalization is handled explicitly in gpr_rank1_corrected
    using norm_sq = phi_train^T phi_train computed from the training set.
    """
    x0 = X[:, 0]
    return {
        "h1t": x0,
        "h3t": (x0 ** 3 - 3.0 * x0) / math.sqrt(6.0),
    }


def _sherman_morrison(
    A_inv: torch.Tensor,
    phi_hat: torch.Tensor,
    delta_lambda: float,
) -> torch.Tensor:
    """
    One rank-1 Sherman-Morrison update to A_inv.

    Incorporates the correction  delta_lambda * phi_hat phi_hat^T  into the
    inverse, where phi_hat is a UNIT-NORM vector.

    Identity:
        (A + delta_lambda * phi_hat phi_hat^T)^{-1}
            = A^{-1}
              - (delta_lambda * A^{-1} phi_hat  phi_hat^T A^{-1})
                / (1 + delta_lambda * phi_hat^T A^{-1} phi_hat)

    Cost: O(n^2) versus O(n^3) for a fresh Cholesky.
    """
    u     = A_inv @ phi_hat
    denom = 1.0 + delta_lambda * float(phi_hat @ u)

    if abs(denom) < 1e-10:
        raise ValueError(
            f"Sherman-Morrison denominator ~0 ({denom:.3e}). "
            "The corrected kernel is nearly singular for this mode."
        )
    if denom < 0:
        print(
            f"Warning: SM denominator negative ({denom:.3e}); "
            "corrected kernel is not PD for this mode."
        )

    return A_inv - (delta_lambda / denom) * torch.outer(u, u)

def _assemble_K_all(K_tr, K_cr, K_te):
    n_train = K_tr.shape[0]
    n_test  = K_cr.shape[0]
    K = torch.zeros((n_train + n_test, n_train + n_test),
                    dtype=K_tr.dtype, device=K_tr.device)
    K[:n_train, :n_train] = K_tr
    K[n_train:, :n_train] = K_cr
    K[:n_train, n_train:] = K_cr.T
    K[n_train:, n_train:] = K_te
    return K

def gpr_rank1_corrected(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    ridge:   float,
    theory_eigs: dict,
    P: int,
    method: str = "sr",
    iterations: int = 10,
) -> tuple:

    n_train = X_train.shape[0]
    X_all   = torch.cat([X_train, X_test], dim=0)

    # Bare arcsin kernel
    K_all   = arcsin_kernel(X_all)
    K_train = K_all[:n_train, :n_train]
    K_cross = K_all[n_train:, :n_train].clone()
    
    n_train = X_train.shape[0]

    K_norm = torch.linalg.norm(K_train, ord=2)  # Largest singular value
    K_train = K_train 
    K_cross = K_cross 
    ridge_scaled = ridge 
    

    eye   = torch.eye(n_train, device=K_train.device, dtype=K_train.dtype)
    chol  = torch.linalg.cholesky(K_train + ridge_scaled * eye)
    A_inv = torch.cholesky_solve(eye, chol)


    phi_tr = _hermite_vecs(X_train)
    phi_te = _hermite_vecs(X_test)

    K_train_corr = K_train.clone()

    eigvals, eigvecs = torch.linalg.eigh(K_train)
    # Sort eigvecs by descending eigenvalue
    idx_desc = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]

    # Iterative joint update (no deflation) -- handle as a separate path
# ====================== NEW SMART UPDATE ======================
       # ====================== NEW SMART UPDATE ======================
    if method == "smart_update":
        for vec_key, eig_key in [("h1t", "lH1T"), ("h3t", "lH3T")]:
            lam_ft = theory_eigs.get(eig_key)
            if lam_ft is None or not math.isfinite(lam_ft):
                continue
                
            phi_train_raw = phi_tr[vec_key]
            phi_test_raw = phi_te[vec_key]
            phi_train = phi_train_raw.clone()
            phi_test = phi_test_raw.clone()
            
            # === FIXED NORMALIZATION ===
            norm_sq = phi_train @ phi_train          # keep as tensor
            norm_sq_scalar = float(norm_sq)
            
            if norm_sq_scalar < 1e-8:
                print(f"Warning: {vec_key} degenerate.")
                continue
                
            # Normalized direction (unit norm)
            phi_train_hat = phi_train / torch.sqrt(norm_sq)      # now correct
            phi_test_hat = phi_test / torch.sqrt(norm_sq)
            
            # Find best matching empirical eigenvector via overlap
            overlaps = torch.abs(eigvecs.T @ phi_train)          # or @ phi_train_hat if you prefer
            k_target = int(torch.argmax(overlaps).item())
            overlap_score = float(overlaps[k_target])
            
            lam_gpr = float(eigvals[k_target])
            lam_desired = float(lam_ft) * P
            
            gap = lam_desired - lam_gpr
            delta_lambda = gap
            
            print(
                f"Mode {vec_key:4s} | method=smart_update | P={P:4d} | "
                f"k={k_target:3d} | overlap={overlap_score:.4f} | "
                f"lam_gpr={lam_gpr:8.4f} → {lam_desired:8.4f} | "
                f"delta={delta_lambda:8.4f} | norm_sq={norm_sq_scalar:.2e}"
            )
            
            # Rank-1 update on the inverse
            A_inv = _sherman_morrison(A_inv, phi_train_hat, delta_lambda)
            
            # Update kernels
            K_train_corr += delta_lambda * torch.outer(phi_train_hat, phi_train_hat)
            K_cross += delta_lambda * torch.outer(phi_test_hat, phi_train_hat)
        
    if method == "iterative":
        phi1_tr = phi_tr["h1t"]
        phi1_te = phi_te["h1t"]
        phi3_tr = phi_tr["h3t"]
        phi3_te = phi_te["h3t"]

        norm1 = float(phi1_tr @ phi1_tr)
        norm3 = float(phi3_tr @ phi3_tr)
        if norm1 < 1e-8 or norm3 < 1e-8:
            print("Warning: hermite modes degenerate; falling back")
            return gpr_from_kernel_matrix(arcsin_kernel(torch.cat([X_train, X_test], dim=0)), X_train, y_train, ridge)

        phi1_hat_tr = phi1_tr / math.sqrt(norm1)
        phi1_hat_te = phi1_te / math.sqrt(norm1)
        phi3_hat_tr = phi3_tr / math.sqrt(norm3)
        phi3_hat_te = phi3_te / math.sqrt(norm3)

        # Initial Rayleigh quotients
        lam1_init = float(phi1_hat_tr @ (K_train @ phi1_hat_tr))
        lam3_init = float(phi3_hat_tr @ (K_train @ phi3_hat_tr))

        def build_kernel(a1, a3):
            K_tr = K_train + a1 * torch.outer(phi1_hat_tr, phi1_hat_tr) \
                        + a3 * torch.outer(phi3_hat_tr, phi3_hat_tr)
            K_cr = K_cross + a1 * torch.outer(phi1_hat_te, phi1_hat_tr) \
                        + a3 * torch.outer(phi3_hat_te, phi3_hat_tr)
            K_te = K_all[n_train:, n_train:] \
                        + a1 * torch.outer(phi1_hat_te, phi1_hat_te) \
                        + a3 * torch.outer(phi3_hat_te, phi3_hat_te)
            return K_tr, K_cr, K_te

        # Initialize at theory eigenvalues
        l1_des = theory_eigs.get("lH1T")
        l3_des = theory_eigs.get("lH3T")
        if l1_des is None or l3_des is None:
            print("Warning: theory eigs missing; falling back")
            return gpr_from_kernel_matrix(arcsin_kernel(torch.cat([X_train, X_test], dim=0)), X_train, y_train, ridge)

        lam1 = float(l1_des) * P
        lam3 = float(l3_des) * P

        eps = 1e-6
        y_pred_iter = None

        for it in range(max(1, iterations)):
            a1 = max(lam1 - lam1_init, -lam1_init + eps)
            a3 = max(lam3 - lam3_init, -lam3_init + eps)

            K_tr, K_cr, K_te = build_kernel(a1, a3)
            K_full = _assemble_K_all(K_tr, K_cr, K_te)

            try:
                y_pred_iter, _, _ = gpr_from_kernel_matrix(K_full, X_train, y_train, ridge)
            except Exception as exc:
                print(f"Iterative GPR failed at iter {it+1}: {exc}")
                break

            ld = h3_learnability_from_predictions(y_pred_iter, X_test, X_train)
            eta1 = float(ld.get("h1_sum", 0.0))
            eta3 = float(ld.get("h3_sum", 0.0))

            eta1_c = min(1.0 - eps, eta1)
            eta3_c = min(1.0 - eps, eta3)
            lam1_new = max(eps, eta1_c * ridge / (1.0 - eta1_c))
            lam3_new = max(eps, eta3_c * ridge / (1.0 - eta3_c))
            print(
                f"Iter {it+1:2d}: "
                f"eta1={eta1:.4e} eta3={eta3:.4e} | "
                f"lam1={lam1:.4e}->{lam1_new:.4e} | "
                f"lam3={lam3:.4e}->{lam3_new:.4e}"
            )

            if abs(lam1_new - lam1) < 1e-6 and abs(lam3_new - lam3) < 1e-6:
                print(f"Converged at iter {it+1}")
                lam1, lam3 = lam1_new, lam3_new
                break

            lam1, lam3 = lam1_new, lam3_new

        if y_pred_iter is None:
            return gpr_from_kernel_matrix(K_all, X_train, y_train, ridge)

        a1_final = lam1 - lam1_init
        a3_final = lam3 - lam3_init
        K_train_temp, K_cr_final, K_te_final = build_kernel(a1_final, a3_final)
        K_all_temp = _assemble_K_all(K_train_temp, K_cr_final, K_te_final)

        return y_pred_iter, K_train_temp, K_all_temp

    for vec_key, eig_key in [
        ("h1t", "lH1T"),
        ("h3t", "lH3T"),
    ]:
        lam_ft = theory_eigs.get(eig_key)
        if lam_ft is None or not math.isfinite(lam_ft):
            continue

        # Raw vectors
        phi_train_raw = phi_tr[vec_key]
        phi_test_raw  = phi_te[vec_key]

        phi_train = phi_train_raw.clone()
        phi_test  = phi_test_raw.clone()
        # if vec_key == "h1t":

        #     B_train = X_train[:, 1:]
        #     if B_train.shape[1] > 0:
        #         Q, R = torch.linalg.qr(B_train, mode='reduced')


        #         beta = torch.linalg.pinv(B_train) @ phi_train_raw
        #         phi_train = phi_train_raw - (B_train @ beta)
        #         B_test = X_test[:, 1:]
        #         phi_test = phi_test_raw - (B_test @ beta)

        # if vec_key == "h3t":                               
        #     B_train_list = [X_train]
        #     if X_train.shape[1] > 1:
        #         x_train_perp = X_train[:, 1:]
        #         x_test_perp = X_test[:, 1:]
        #         h3_train_perp = (x_train_perp ** 3 - 3.0 * x_train_perp) / math.sqrt(6.0)
        #         B_train_list.append(h3_train_perp)
            
        #     B_train = torch.cat(B_train_list, dim=1)
        #     Q, R = torch.linalg.qr(B_train, mode='reduced')


        #     beta = torch.linalg.pinv(B_train) @ phi_train_raw
        #     phi_train = phi_train_raw - (B_train @ beta)

        #     B_test_list = [X_test]
        #     if X_test.shape[1] > 1:
        #         x_test_perp = X_test[:, 1:]
        #         h3_test_perp = (x_test_perp ** 3 - 3.0 * x_test_perp) / math.sqrt(6.0)
        #         B_test_list.append(h3_test_perp)
        #     B_test = torch.cat(B_test_list, dim=1)
        #     phi_test = phi_test_raw - (B_test @ beta)
            
        with torch.no_grad():
            overlaps = torch.abs(eigvecs.T @ phi_train)
            k_target = int(torch.argmax(overlaps).item())
            
        # ====================== NORMALIZATION ======================
        norm_sq = float(phi_train @ phi_train)
        if norm_sq < 1e-8:
            print(f"Warning: {vec_key} became degenerate after deflation.")
            continue


        lam_gpr_unscaled = eigvals[k_target]
        lam_gpr = lam_gpr_unscaled
        lam_desired = float(lam_ft) * P

        gap_unscaled = lam_desired - lam_gpr_unscaled

        # Prepare normalized feature vectors
        phi_train_hat = phi_train / math.sqrt(P) #/ math.sqrt(norm_sq)
        phi_test_hat = phi_test / math.sqrt(P) #/ math.sqrt(norm_sq)

        if method == "sr":
            eigvals_eigh, eigvecs_eigh = torch.linalg.eigh(K_train)

            if vec_key == "h3t":
                start_idx = X_train.shape[1]
                end_idx = eigvals_eigh.shape[0]
            else:
                start_idx = 0
                end_idx = X_train.shape[1]

            eigvals_sub = eigvals_eigh[start_idx:end_idx]
            eigvecs_sub = eigvecs_eigh[:, start_idx:end_idx]

            # only include eigenvalues strictly below lam_desired
            mask = eigvals_sub < lam_desired - 1e-10
            if mask.sum() == 0:
                print(f"Warning: no eigenvalues below lam_desired={lam_desired:.4e} for mode {vec_key}; skipping")
                continue

            eigvals_below = eigvals_sub[mask]
            eigvecs_below = eigvecs_sub[:, mask]

            overlaps = (eigvecs_below.T @ phi_train_hat) ** 2

            diffs = lam_desired - eigvals_below  # all positive now

            m_phi = float((overlaps / diffs).sum())  # positive, so delta = -1/m_phi is negative...

            # m_phi is positive here, but we want delta positive (pushing eigenvalue up)
            # the secular equation for an outlier ABOVE the bulk gives delta = 1/m_phi
            delta_lambda = 1.0 / m_phi

            print(
                f"Mode {vec_key:4s} | method=sr_exact | P={P:4d} | start_idx={start_idx} | "
                f"n_below={mask.sum().item()} | "
                f"lam_gpr={lam_gpr:6.4f} | lam_desired={lam_desired:6.4f} | "
                f"m_phi={m_phi:6.4f} | delta={delta_lambda:7.4f}"
            )

            A_inv = _sherman_morrison(A_inv, phi_train_hat, delta_lambda)
            K_cross = K_cross + delta_lambda * torch.outer(phi_test_hat, phi_train_hat)
            K_train_corr = K_train_corr + delta_lambda * torch.outer(phi_train_hat, phi_train_hat)
        elif method == "naive":            
            gap_corrected = gap_unscaled
            delta_lambda = gap_corrected
            print(
                f"Mode {vec_key:4s} | method=naive | P={P:4d} | k={k_target:4d} | lam_gpr={lam_gpr:6.4f} | "
                f"lam_desired={lam_desired:6.4f} | gap={gap_unscaled:6.4f} | delta={delta_lambda:7.4f}"
            )
            A_inv = _sherman_morrison(A_inv, phi_train_hat, delta_lambda)
            K_cross = K_cross + delta_lambda * torch.outer(phi_test_hat, phi_train_hat)
            K_train_corr = K_train_corr + delta_lambda * torch.outer(phi_train_hat, phi_train_hat)

        elif method =="smart_update":
                
            # Final prediction
            alpha = A_inv @ y_train
            y_pred = K_cross @ alpha

            return y_pred, K_train_corr, K_all
        else:
            raise ValueError(f"Unknown correction method: {method}")

        
    # Final prediction
    alpha = A_inv @ y_train
    y_pred = K_cross @ alpha

    return y_pred, K_train_corr, K_all


# ---------------------------------------------------------------------------
# Baseline GPR helpers


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


# def h3_learnability_from_predictions(y_pred: torch.Tensor, X: torch.Tensor, X_train: Optional[torch.Tensor] = None) -> dict:
#     """
#     Compute He1/He3 learnability components for predicted targets `y_pred` on inputs `X`.

#     The normalization components `y_he1` and `y_he3` are computed from the TRAINING
#     inputs as:

#         y = He1_train + 0.074 * He3_train
#         y_he1 = (y^T He1_train) / P_train
#         y_he3 = (y^T He3_train) / P_train

#     These scalars represent the target's projection onto the Hermite features that
#     the model actually sees and are used to normalize the measured projections on
#     the test set.
#     """
#     if X_train is None:
#         raise ValueError("X_train must be provided to compute y_he1 and y_he3 from the training set")

#     # Test-set feature evaluations
#     x0_test = X[:, 0]
#     h3_comp = hermite_h3(x0_test)

#     # Training-set feature evaluations for normalization
#     x0_train = X_train[:, 0]
#     he1_train = x0_train
#     he3_train = hermite_h3(x0_train)
#     P_train = float(X_train.shape[0])

#     # Combined target decomposition on training set
#     y_comb = he1_train + 0.074 * he3_train
#     # Scalar normalizations
#     y_he1 = float((y_comb @ he1_train) / P_train)
#     y_he3 = float((y_comb - he1_train * y_he1) @ he3_train / P_train)
#     # Avoid divide-by-zero
#     if abs(y_he1) < 1e-12 or abs(y_he3) < 1e-12:
#         print("Warning: small normalization constants y_he1 or y_he3; results may be unstable")

#     # Compute projections of predictions on test set
#     y_He1_component = (y_pred * x0_test).mean()
#     linear_coeff = y_He1_component
#     remainder = y_pred - linear_coeff * x0_test
#     proj3 = (remainder * h3_comp).mean()

#     return {
#         "h1_sum": float(linear_coeff.item() / y_he1),
#         "h3_sum": float(proj3.item() / y_he3),
#         "proj3_target_sum": float(proj3.item()),
#     }

def h3_learnability_from_predictions(y_pred: torch.Tensor, X: torch.Tensor, X_train: Optional[torch.Tensor] = None) -> dict:
    x0_test = X[:, 0]
    h3_comp = hermite_h3(x0_test)

    y_true = target_fn(X)

    # True target projections with He1 deflation
    true_he1_comp = (y_true * x0_test).mean()
    true_remainder = y_true - true_he1_comp * x0_test
    true_proj3 = (true_remainder * h3_comp).mean()

    # Prediction projections with same deflation
    pred_he1_comp = (y_pred * x0_test).mean()
    pred_remainder = y_pred - pred_he1_comp * x0_test
    pred_proj3 = (pred_remainder * h3_comp).mean()

    h1_sum = float(pred_he1_comp / true_he1_comp) if abs(float(true_he1_comp)) > 1e-10 else 0.0
    h3_sum = float(pred_proj3 / true_proj3) if abs(float(true_proj3)) > 1e-10 else 0.0

    return {
        "h1_sum": h1_sum,
        "h3_sum": h3_sum,
        "proj3_target_sum": float(pred_proj3),
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


def set_p_axis_limits(ax, p_values, padding_fraction: float = 0.08):
    """Set log-scale x limits slightly outside the min and max P values."""
    p_values = np.asarray(p_values, dtype=float)
    p_values = p_values[np.isfinite(p_values)]
    if p_values.size == 0:
        return

    p_min = float(np.min(p_values))
    p_max = float(np.max(p_values))
    if p_min <= 0.0:
        raise ValueError("P values must be positive to use a log-scaled x-axis")

    if math.isclose(p_min, p_max):
        lower = p_min * (1.0 - padding_fraction)
        upper = p_max * (1.0 + padding_fraction)
    else:
        lower = p_min * (1.0 - padding_fraction)
        upper = p_max * (1.0 + padding_fraction)
    ax.set_xlim(lower, upper)


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
    # Deduplicate per run directory and prefer model_final.pt over model.pt.
    best_by_run_dir = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        run_dir = candidate.parent.resolve()
        score = 0 if candidate.name == "model_final.pt" else 1
        current = best_by_run_dir.get(run_dir)
        if current is None or score < current[0] or (score == current[0] and str(candidate) < str(current[1])):
            best_by_run_dir[run_dir] = (score, candidate.resolve())
    files = sorted((entry[1] for entry in best_by_run_dir.values()), key=str)
    return files


def exclude_first_n_p_values(results, exclude_first_n_p: int = 0):
    """Drop all datapoints whose P belongs to the first n distinct P values."""
    if not results or exclude_first_n_p <= 0:
        return list(results)

    unique_ps = sorted({int(row["P"]) for row in results})
    excluded_ps = set(unique_ps[: int(exclude_first_n_p)])
    return [row for row in results if int(row["P"]) not in excluded_ps]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_runs(scan_dir: Path, test_size: int = 5000, limit: Optional[int] = None, correction_method: str = "sr", iter_steps: int = 10):
    model_files = find_model_files(scan_dir)
    if limit is not None:
        model_files = model_files[:int(limit)]

    print(f"Found {len(model_files)} checkpoint files in {scan_dir}")
    results = []
    kappa_eff_cache: dict[tuple, float] = {}

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

        kappa_key = kappa_eff_cache_key(cfg)
        if kappa_key not in kappa_eff_cache:
            try:
                kappa_eff_cache[kappa_key] = compute_kappa_eff(
                    d=d,
                    P=P,
                    kappa_bare=ridge,
                    n1=N_for_theory,
                    n2=N_for_theory,
                    chi=chi,
                    num_samples=P,
                    device=DEVICE,
                    verbose=False,
                )
                print(f"  Computed kappa_eff={kappa_eff_cache[kappa_key]:.6f} (kappa_bare={ridge:.6f})")
            except Exception as e:
                print(f"  Warning: kappa_eff computation failed ({e}), using bare ridge={ridge:.6f}")
                kappa_eff_cache[kappa_key] = ridge
        else:
            print(f"  Reusing cached kappa_eff={kappa_eff_cache[kappa_key]:.6f} for d={d}, P={P}, chi={chi}, kappa={ridge:.6f}")

        kappa_eff = kappa_eff_cache[kappa_key]
        X_train = make_gaussian_dataset(d, P,         train_seed, device=DEVICE)
        y_train = target_fn(X_train).to(dtype=DTYPE)
        X_test  = make_gaussian_dataset(d, test_size, test_seed,  device=DEVICE)
        y_test  = target_fn(X_test).to(dtype=DTYPE)

        # Bare empirical-kernel GPR
        X_all  = torch.cat([X_train, X_test], dim=0)
        K_all  = arcsin_kernel(X_all)
        y_pred, K_train, K_all = gpr_from_kernel_matrix(K_all, X_train, y_train, ridge=kappa_eff)
        mse          = torch.mean((y_pred - y_test) ** 2).item()
        learnability = h3_learnability_from_predictions(y_pred, X_test, X_train)

        # Theory eigenvalues from Julia
        theory_eigs = compute_theory_eigenvalues(
            d=d, P=P, N=N_for_theory, chi=chi, kappa=kappa_eff, eps=eps
        )
        theory_h1_learnability = learnability_from_eigenvalue(theory_eigs.get("lH1T"), kappa_eff, P)
        theory_h3_learnability = learnability_from_eigenvalue(theory_eigs.get("lH3T"), kappa_eff, P)

        # Rank-1 corrected GPR via Sherman-Morrison
        y_pred_theory, K_train_theory, _ = gpr_rank1_corrected(
            X_train=X_train, y_train=y_train, X_test=X_test,
            ridge=kappa_eff, theory_eigs=theory_eigs, P=P,
            method=correction_method, iterations=iter_steps,
        )
        mse_theory          = torch.mean((y_pred_theory - y_test) ** 2).item()
        learnability_theory = h3_learnability_from_predictions(y_pred_theory, X_test, X_train)

        # Trained model predictions
        with torch.no_grad():
            y_pred_model_raw = model(X_test)
        y_pred_model       = collapse_model_prediction(y_pred_model_raw)
        mse_model          = torch.mean((y_pred_model - y_test) ** 2).item()
        learnability_model = h3_learnability_from_predictions(y_pred_model, X_test, X_train)

        results.append({
            "run_dir":    str(run_dir),
            "checkpoint": str(checkpoint_path),
            "d": d, "P": P,
            "N":     int(cfg.get("N", cfg.get("n1", 0))),
            "chi":   float(cfg.get("chi", float("nan"))) if cfg.get("chi") is not None else float("nan"),
            "kappa": ridge,
            "kappa_eff": kappa_eff,
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

def plot_results(results, out_dir: Path, separate_plots: bool = False, filename_suffix: str = ""):
    if not results:
        raise RuntimeError("No results to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)

    d_value = int(results[0].get("d", -1))
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
        return float(arr[mask].mean()), float(arr[mask].std(ddof=0)/math.sqrt(mask.sum()))

    def _eb(ax, arr, fmt, color, label):
        ms    = [_mean_std(arr, p) for p in unique_P]
        means = [x[0] for x in ms]
        stds  = [x[1] for x in ms]
        ax.errorbar(
            unique_P,
            means,
            yerr=stds,
            fmt=fmt,
            lw=2,
            color=color,
            label=label,
            ecolor=ERRORBAR_COLOR,
            capsize=3,
            elinewidth=1.6,
            capthick=1.6,
        )

    if separate_plots:
        # Create three separate figures
        figs = []
        axs = []
        for _ in range(3):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            figs.append(fig)
            axs.append(ax)
        ax1, ax2, ax3 = axs
    else:
        # Create one figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        figs = [fig]

    for p in unique_P:
        mask = P_arr == p
        # ax1.scatter(np.full(mask.sum(), p), h1_arr[mask],        alpha=0.35, s=24, color=GPR_COLOR)
        # ax1.scatter(np.full(mask.sum(), p), h1_model_arr[mask],  alpha=0.35, s=24, color=MODEL_COLOR)
        # ax1.scatter(np.full(mask.sum(), p), h1_theory_arr[mask], alpha=0.35, s=24, color=THEORY_COLOR)
        # ax2.scatter(np.full(mask.sum(), p), h3_arr[mask],        alpha=0.35, s=24, color=GPR_COLOR)
        # ax2.scatter(np.full(mask.sum(), p), h3_model_arr[mask],  alpha=0.35, s=24, color=MODEL_COLOR)
        # ax2.scatter(np.full(mask.sum(), p), h3_theory_arr[mask], alpha=0.35, s=24, color=THEORY_COLOR)
        # ax3.scatter(np.full(mask.sum(), p), mse_arr[mask],        alpha=0.6,  s=30, color="tab:red")
        # ax3.scatter(np.full(mask.sum(), p), mse_theory_arr[mask], alpha=0.4,  s=26, color=MSE_COLOR)

    _eb(ax1, h1_model_arr,  "s--", MODEL_COLOR,  "Model mean +/- std")
    _eb(ax1, h1_theory_arr, "^:",  THEORY_COLOR, "Corrected Theory +/- std")
    _eb(ax1, h1_tlearn_arr, "d-.", "tab:red",    "Raw Theory")
    _eb(ax2, h3_arr,        "o-",  GPR_COLOR,   "GPR mean +/- std")
    _eb(ax2, h3_model_arr,  "s--", MODEL_COLOR,   "Model mean +/- std")
    _eb(ax2, h3_theory_arr, "^:",  THEORY_COLOR,   "Corrected Theory +/- std")
    _eb(ax2, h3_tlearn_arr, "d-.", "tab:red",  "Raw Theory")
    _eb(ax3, mse_arr,        "o-",  "tab:red",    "Model")
    _eb(ax3, mse_theory_arr, "^:",  MSE_COLOR, "Adapted-kernel GPR +/- std")
    chi=results[0].get("chi", float("nan"))
    kappa=results[0].get("kappa", float("nan"))
    N=results[0].get("N", float("nan"))
    for ax, title, ylabel in [
        (ax1, rf"He1 Learnability vs P, $d={d_value},\kappa={kappa},\chi={chi}$", "linear learnability"),
        (ax2, rf"He3 Learnability vs P, $d={d_value},\kappa={kappa},\chi={chi}$",     "He3 learnability"),
        (ax3, "Test MSE vs P",            "test MSE"),
    ]:
        ax.axvline(d_value, color="gray", ls="--", alpha=0.6, label=f"d={d_value}")
        ax.set_xscale("log")
        ax.set_xlabel("P")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        set_p_axis_limits(ax, P_arr)
        if ax is ax1:
            ax.set_ylim(bottom=0.0)
        ax.legend()

    if separate_plots:
        # Save each figure separately
        fig_paths = []
        for i, (fig, ax) in enumerate(zip(figs, [ax1, ax2, ax3])):
            fig.tight_layout()
            titles = ["linear_learnability", "He3_learnability", "test_mse"]
            fig_path = out_dir / f"semi_empirical_effective_ridge_correction_d{d_value}_{titles[i]}.png"
            fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
            fig_paths.append(fig_path)
            plt.close(fig)
        fig_path = fig_paths  # Return list of paths
    else:
        # Save one combined figure
        plt.tight_layout()
        fig_path  = out_dir / f"semi_empirical_effective_ridge_correction_d{d_value}{filename_suffix}.png"
        plt.savefig(str(fig_path), dpi=300, bbox_inches="tight")
        plt.show()
    
    json_path = out_dir / f"semi_empirical_effective_ridge_correction_d{d_value}{filename_suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return fig_path, json_path


def plot_all_d_model_vs_theory(results, out_dir: Path, separate_plots: bool = False, filename_suffix: str = ""):
    if not results:
        raise RuntimeError("No results to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)

    d_values = sorted({int(r["d"]) for r in results})
    if not d_values:
        raise RuntimeError("No d values found in results.")

    # If there's only one d, behave like the original flow: return the per-d
    # figure(s) produced by `plot_results` so we don't duplicate or change
    # the original single-d plotting semantics.
    if len(d_values) == 1:
        d_single = d_values[0]
        d_dir = out_dir / f"d{d_single}"
        # If separate_plots was requested, return the three per-d figure paths
        if separate_plots:
            titles = ["linear_learnability", "He3_learnability", "test_mse"]
            paths = []
            for t in titles:
                p = d_dir / f"semi_empirical_effective_ridge_correction_d{d_single}_{t}.png"
                if p.exists():
                    paths.append(p)
            if paths:
                return paths
        # Otherwise return the combined per-d figure if it exists
        combined = d_dir / f"semi_empirical_effective_ridge_correction_d{d_single}.png"
        if combined.exists():
            return combined
        # Fall through to compute the all-d figure if per-d outputs are missing

    if len(d_values) <= len(ALL_D_COLORS):
        d_to_color = {d_value: ALL_D_COLORS[i] for i, d_value in enumerate(d_values)}
    else:
        cmap = cm.get_cmap("turbo")
        color_positions = np.linspace(0.1, 0.9, max(len(d_values), 1))
        d_to_color = {d_value: cmap(pos) for d_value, pos in zip(d_values, color_positions)}

    # Enforce specific colors for important d values: d=10 -> purple, d=15 -> orange
    # (keeps other d mappings intact)
    if 10 in d_values:
        d_to_color[10] = "#7B2CBF"  # purple
    if 15 in d_values:
        d_to_color[15] = "#FF7F0E"  # orange (match MODEL_COLOR)

    def mean_over_p(d_results, key, p_values):
        means = []
        stds = []
        for p in p_values:
            vals = np.array([float(r[key]) for r in d_results if int(r["P"]) == p], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=0) / math.sqrt(len(vals))))
        return np.array(means), np.array(stds)

    def mean_over_p_global(key, p_values):
        means = []
        stds = []
        for p in p_values:
            vals = np.array([float(r[key]) for r in results if int(r["P"]) == p], dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=0) / math.sqrt(len(vals))))
        return np.array(means), np.array(stds)

    def decorate_axis(ax, title, ylabel, d_min, d_max):
        ax.set_xscale("log")
        ax.set_xlabel("P")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvline(d_min, color="gray", ls=":", alpha=0.5)
        ax.axvline(d_max, color="gray", ls=":", alpha=0.5)

    def make_legend(ax):
        style_handles = [
            ax.plot([], [], color=MODEL_COLOR, linestyle="-", marker="o", linewidth=2.5, label="Model")[0],
            ax.plot([], [], color=THEORY_COLOR, linestyle="--", marker="s", linewidth=2.5, label="Corrected theory")[0],
            ax.plot([], [], color=GPR_COLOR, linestyle=":", marker=None, linewidth=2.5, label="GPR")[0],
        ]
        d_handles = [
            ax.plot([], [], color=d_to_color[d_value], linestyle="-", linewidth=3, label=f"d={d_value}")[0]
            for d_value in d_values
        ]
        ax.legend(handles=style_handles + d_handles, loc="best", fontsize=13)

    d_min = min(d_values)
    d_max = max(d_values)

    if separate_plots:
        fig_paths = []
        for mode_key, title, ylabel, suffix in [
            ("h1", "He1: Empirical model vs corrected theory", "linear learnability", "model_vs_theory_linear"),
            ("h3", "He3: Empirical model vs corrected theory", "He3 learnability", "model_vs_theory_quadratic"),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            for d_value in d_values:
                d_results = [r for r in results if int(r["d"]) == d_value]
                p_values = np.array(sorted({int(r["P"]) for r in d_results}))
                color = d_to_color[d_value]
                model_mean, model_std = mean_over_p(d_results, f"{mode_key}_sum_model", p_values)
                theory_mean, theory_std = mean_over_p(d_results, f"{mode_key}_sum_theory", p_values)

                ax.errorbar(p_values, model_mean, yerr=model_std, color=color, linestyle="-", marker="o", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
                ax.errorbar(p_values, theory_mean, yerr=theory_std, color=color, linestyle="--", marker="s", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)

                # Per-d empirical GPR baseline (same color as the per-d model, distinct linestyle)
                gpr_mean, gpr_std = mean_over_p(d_results, f"{mode_key}_sum", p_values)
                ax.errorbar(p_values, gpr_mean, yerr=gpr_std, color=color, linestyle=":", marker=None, linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)

            # (Removed aggregated all-d GPR baseline for separate per-d plots)

            decorate_axis(ax, title, ylabel, d_min, d_max)
            # compute p_all for axis limits (all observed P across results)
            p_all = np.array(sorted({int(r["P"]) for r in results}))
            set_p_axis_limits(ax, p_all)
            if mode_key == "h1":
                ax.set_ylim(bottom=0.0)
            # Build legend showing line styles and per-d colors
            style_handles = [
                ax.plot([], [], color=MODEL_COLOR, linestyle="-", marker="o", linewidth=2.5, label="Model")[0],
                ax.plot([], [], color=THEORY_COLOR, linestyle="--", marker="s", linewidth=2.5, label="Corrected Theory")[0],
                ax.plot([], [], color=GPR_COLOR, linestyle=":", linewidth=2.5, label="GPR")[0],
            ]
            d_handles = [
                ax.plot([], [], color=d_to_color[d_val], linestyle="-", linewidth=2.5, label=f"d={d_val}")[0]
                for d_val in d_values
            ]
            ax.legend(handles=style_handles + d_handles, loc="best", fontsize=13, ncol=2)
            fig.tight_layout()
            fig_path = out_dir / f"semi_empirical_effective_ridge_correction_all_d_{suffix}{filename_suffix}.png"
            fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
            plt.close(fig)
            fig_paths.append(fig_path)

        return fig_paths

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for d_value in d_values:
        d_results = [r for r in results if int(r["d"]) == d_value]
        p_values = np.array(sorted({int(r["P"]) for r in d_results}))
        color = d_to_color[d_value]

        h1_model_mean, h1_model_std = mean_over_p(d_results, "h1_sum_model", p_values)
        h1_theory_mean, h1_theory_std = mean_over_p(d_results, "h1_sum_theory", p_values)
        h3_model_mean, h3_model_std = mean_over_p(d_results, "h3_sum_model", p_values)
        h3_theory_mean, h3_theory_std = mean_over_p(d_results, "h3_sum_theory", p_values)

        ax1.errorbar(p_values, h1_model_mean, yerr=h1_model_std, color=color, linestyle="-", marker="o", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
        ax1.errorbar(p_values, h1_theory_mean, yerr=h1_theory_std, color=color, linestyle="--", marker="s", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
        # Per-d GPR baseline for He1
        h1_gpr_mean, h1_gpr_std = mean_over_p(d_results, "h1_sum", p_values)
        ax1.errorbar(p_values, h1_gpr_mean, yerr=h1_gpr_std, color=color, linestyle=":", marker=None, linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
        ax2.errorbar(p_values, h3_model_mean, yerr=h3_model_std, color=color, linestyle="-", marker="o", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
        ax2.errorbar(p_values, h3_theory_mean, yerr=h3_theory_std, color=color, linestyle="--", marker="s", linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)
        # Per-d GPR baseline for He3
        h3_gpr_mean, h3_gpr_std = mean_over_p(d_results, "h3_sum", p_values)
        ax2.errorbar(p_values, h3_gpr_mean, yerr=h3_gpr_std, color=color, linestyle=":", marker=None, linewidth=2.5, capsize=3, ecolor=ERRORBAR_COLOR)

    decorate_axis(ax1, "He1: Empirical model vs corrected theory", "linear learnability", d_min, d_max)
    decorate_axis(ax2, "He3: Empirical model vs corrected theory", "He3 learnability", d_min, d_max)
    set_p_axis_limits(ax1, [r["P"] for r in results])
    set_p_axis_limits(ax2, [r["P"] for r in results])
    ax1.set_ylim(bottom=0.0)
    # Build legend showing line styles and per-d colors
    style_handles = [
        ax1.plot([], [], color=MODEL_COLOR, linestyle="-", marker="o", linewidth=2.5, label="Model")[0],
        ax1.plot([], [], color=THEORY_COLOR, linestyle="--", marker="s", linewidth=2.5, label="Corrected Theory")[0],
        ax1.plot([], [], color=GPR_COLOR, linestyle=":", linewidth=2.5, label="GPR")[0],
    ]
    d_handles = [
        ax1.plot([], [], color=d_to_color[d_val], linestyle="-", linewidth=2.5, label=f"d={d_val}")[0]
        for d_val in d_values
    ]
    ax1.legend(handles=style_handles + d_handles, loc="best", fontsize=13, ncol=2)
    ax2.legend(handles=style_handles + d_handles, loc="best", fontsize=13, ncol=2)

    plt.tight_layout()
    fig_path = out_dir / f"semi_empirical_effective_ridge_correction_all_d_model_vs_theory{filename_suffix}.png"
    plt.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Empirical H_Kernel GPR analysis for mixed-d scan runs."
    )
    parser.add_argument("--scan-dir",   type=str, default=str(DEFAULT_SCAN_DIR))
    parser.add_argument("--test-size",  type=int, default=1000)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--use-cache",  action="store_true", help="Load results from cached JSON instead of recomputing")
    parser.add_argument("--separate-plots", action="store_true", help="Save each plot separately instead of combined")
    parser.add_argument("--exclude-first-n-p", type=int, default=0, help="Exclude datapoints whose P is among the first n distinct P values")
    parser.add_argument("--correction-method", type=str, choices=["sr", "naive", "iterative", "smart_update"], default="sr", help="Which rank-1 correction method to use ('sr' default, 'naive', or 'iterative')")
    parser.add_argument("--iter-steps", type=int, default=10, help="Number of fixed-point iterations for 'iterative' method")
    args = parser.parse_args(argv)

    scan_dir   = Path(args.scan_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else scan_dir / "semi_empirical_effective_ridge_correction"
    )
    filename_suffix = f"_exclude_first_n_p{args.exclude_first_n_p}" if args.exclude_first_n_p > 0 else ""

    results_by_d = {}
    results = []

    if args.use_cache:
        if filename_suffix:
            all_paths = sorted(output_dir.glob("d*/semi_empirical_effective_ridge_correction_d*.json"))
            raw_paths = [p for p in all_paths if "_exclude_first_n_p" not in p.name]
            if raw_paths:
                cached_paths = raw_paths
            else:
                cached_paths = sorted(output_dir.glob(f"d*/semi_empirical_effective_ridge_correction_d*{filename_suffix}.json"))
                if cached_paths:
                    print(
                        f"No raw cache files found; loading filtered cache files for suffix {filename_suffix}."
                    )
        else:
            all_paths = sorted(output_dir.glob("d*/semi_empirical_effective_ridge_correction_d*.json"))
            cached_paths = [p for p in all_paths if "_exclude_first_n_p" not in p.name]
        for cached_json in cached_paths:
            try:
                with open(cached_json, "r") as f:
                    cached_results = json.load(f)
                if not cached_results:
                    continue
                d_value = int(cached_results[0]["d"])
                results_by_d[d_value] = cached_results
                results.extend(cached_results)
                print(f"Loaded cached results for d={d_value} from {cached_json}")
            except Exception as exc:
                print(f"Warning: failed to load cache {cached_json}: {exc}")

    if not results:
        results = evaluate_runs(scan_dir, test_size=args.test_size, limit=args.limit, correction_method=args.correction_method, iter_steps=args.iter_steps)
        if not results:
            print("No runs found.")
            return results

    if args.exclude_first_n_p > 0:
        before = len(results)
        results = exclude_first_n_p_values(results, args.exclude_first_n_p)
        after = len(results)
        print(
            f"Excluded datapoints for the first {args.exclude_first_n_p} distinct P values: "
            f"{before - after} rows removed"
        )

    if not results:
        print("No runs left after applying cache/filter selection.")
        return results

    results_by_d = {}
    for row in results:
        results_by_d.setdefault(int(row["d"]), []).append(row)

    for d_value in sorted(results_by_d):
        d_results = results_by_d[d_value]
        d_output_dir = output_dir / f"d{d_value}"
        fig_path, json_path = plot_results(
            d_results,
            d_output_dir,
            separate_plots=args.separate_plots,
            filename_suffix=filename_suffix,
        )
        if isinstance(fig_path, list):
            print(f"Saved figures for d={d_value} to:")
            for fp in fig_path:
                print(f"  {fp}")
        else:
            print(f"Saved figure for d={d_value} to {fig_path}")
        print(f"Saved results for d={d_value} to {json_path}")

    all_d_fig = plot_all_d_model_vs_theory(
        results,
        output_dir,
        separate_plots=args.separate_plots,
        filename_suffix=filename_suffix,
    )
    if isinstance(all_d_fig, list):
        print("Saved all-d comparison figures to:")
        for fp in all_d_fig:
            print(f"  {fp}")
    else:
        print(f"Saved all-d comparison figure to {all_d_fig}")

    return results


if __name__ == "__main__":
    main()