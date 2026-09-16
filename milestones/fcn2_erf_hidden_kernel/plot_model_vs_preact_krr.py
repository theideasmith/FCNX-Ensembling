#!/usr/bin/env python3
"""Compare trained FCN2 outputs to last-layer KRR / kernel ridge.

For each ensemble member, take the hidden features Phi that the readout sees
(erf(W0 x) by default; raw W0 x with --raw-preact) and form

    K = Phi Phi^T / N

Then

    f_krr = K (K + sigma^2 I)^{-1} y

which is solved in feature space (N x N) as

    f_krr = Phi (Phi^T Phi + sigma^2 N I)^{-1} Phi^T y

If the trained readout A matches this ridge solution, model outputs agree with
f_krr. Disagreement means the last-layer weights are not the kernel-ridge
optimum (undertrained, Langevin noise, wrong prior, etc.).

Evaluates both on the reconstructed training set and on an independent test
set of the same size (override with --P-test). On test, KRR / arcsin ridge are
fit on train features/labels and predicted out-of-sample; model is forwarded on
test inputs. Test figures/summaries use a `_test` suffix.

sigma^2 defaults to kappa_eff from kappa_eff_solver (kappa_bare = T/2).
Pass --sigma2 to override.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric  # noqa: E402
from kappa_eff_solver import compute_kappa_eff  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = SCRIPT_DIR / "action_h0_activation_plots"

_INVARIANT_MODELS_DIR = (
    SCRIPT_DIR / "red_robin_alpha_beta_invariant_P0160_Pmax3674_betamax9" / "models"
)
_INVARIANT_PATTERN = (
    "invariant_beta*_alpha*_d*_P*_N*_sa0*_kappa*_seed*"
)


def slugify(name: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "experiment"


def collect_invariant_model_dirs() -> list[str]:
    if not _INVARIANT_MODELS_DIR.is_dir():
        return []
    return sorted(str(p) for p in _INVARIANT_MODELS_DIR.glob(_INVARIANT_PATTERN) if p.is_dir())


_INVARIANT_PFIXED1500_MODELS_DIR = (
    SCRIPT_DIR
    / "red_robin_alpha_beta_invariant_Pfixed1500_betamax9_nu0.25_N01000_lam_eq_omega_alpha_fixed_ens1_Asnap_eps0.03_schedule"
    / "models"
)


def collect_invariant_pfixed1500_asnap_dirs() -> list[str]:
    if not _INVARIANT_PFIXED1500_MODELS_DIR.is_dir():
        return []
    return sorted(
        str(p)
        for p in _INVARIANT_PFIXED1500_MODELS_DIR.glob(_INVARIANT_PATTERN)
        if p.is_dir()
    )


def collect_n_ablation_p1678_model_dirs() -> list[str]:
    """Baseline N=377 + ablations N=1500, 3000 at fixed P=1678 invariant point."""
    import re

    baseline = (
        _INVARIANT_MODELS_DIR
        / "invariant_beta5.196_alpha110.05_d114_P1678_N377_sa00.6623_kappa1.3806_seed0"
    )
    ablation_root = SCRIPT_DIR / "n_ablation_from_invariant_P1678" / "models"
    dirs = []
    if baseline.is_dir():
        dirs.append(str(baseline))
    if ablation_root.is_dir():
        for p in sorted(ablation_root.glob("ablateN*_from_beta*_P1678_N*_seed*")):
            if p.is_dir():
                dirs.append(str(p))

    def _n_key(path: str) -> int:
        m = re.search(r"_N(\d+)_", Path(path).name)
        return int(m.group(1)) if m else 10**9

    return sorted(dirs, key=_n_key)


# Restarted P-sweep dirs used by SampleComplexityTestI (seed 0; see red_robin_launcher_P_sweep).
_SAMPLE_COMPLEXITY_TEST_I_DIRS = [
    SCRIPT_DIR / "d100_P100_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P400_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P700_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P1000_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P4000_N700_chi_700.0_lr_0.005_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P6000_N700_chi_700.0_lr_0.001_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P8000_N700_chi_700.0_lr_0.001_T_0.1_seed_0_eps_0.074",
    SCRIPT_DIR / "d100_P10000_N700_chi_700.0_lr_0.001_T_0.1_seed_0_eps_0.074",
]


def collect_sample_complexity_test_i_dirs() -> list[str]:
    return [str(p) for p in _SAMPLE_COMPLEXITY_TEST_I_DIRS if p.is_dir()]


def collect_n_chi_eq_n_linear_dirs() -> list[str]:
    import re

    from n_chi_eq_N_linear_schedule import MODELS_DIR

    if not MODELS_DIR.is_dir():
        return []
    pattern = re.compile(
        r"^d\d+_P\d+_N(\d+)_chi_[\d.]+_lr_[\d.eE+-]+_T_[\d.]+_seed_\d+_eps_[\d.]+_schedule$"
    )
    dirs = []
    for p in MODELS_DIR.iterdir():
        if p.is_dir() and pattern.match(p.name):
            dirs.append(str(p))

    def _n_key(path: str) -> int:
        m = re.search(r"_N(\d+)_", Path(path).name)
        return int(m.group(1)) if m else 10**9

    return sorted(dirs, key=_n_key)


EXPERIMENT_GROUP_BY_NAME = {
    "AlphaBetaInvariant": type(
        "ExperimentGroup",
        (),
        {"name": "AlphaBetaInvariant", "model_dirs": collect_invariant_model_dirs()},
    )(),
    "InvariantPfixed1500Asnap": type(
        "ExperimentGroup",
        (),
        {
            "name": "InvariantPfixed1500Asnap",
            "model_dirs": collect_invariant_pfixed1500_asnap_dirs(),
        },
    )(),
    "NChiEqNLinearSchedule": type(
        "ExperimentGroup",
        (),
        {
            "name": "NChiEqNLinearSchedule",
            "model_dirs": collect_n_chi_eq_n_linear_dirs(),
        },
    )(),
    "NAblationP1678": type(
        "ExperimentGroup",
        (),
        {"name": "NAblationP1678", "model_dirs": collect_n_ablation_p1678_model_dirs()},
    )(),
    "SampleComplexityTestI": type(
        "ExperimentGroup",
        (),
        {
            "name": "SampleComplexityTestI",
            "model_dirs": collect_sample_complexity_test_i_dirs(),
        },
    )(),
}


def parse_config_from_dirname(dirname):
    import json
    import re

    dir_path = Path(dirname)
    config_path = dir_path / "config.json"

    def seed_from_dirname(name: str):
        m = re.search(r"seed_?(\d+)", name)
        return int(m.group(1)) if m else None

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        seed = cfg.get("dataset_seed", cfg.get("seed"))
        if seed is None:
            seed = seed_from_dirname(dir_path.name)
        else:
            seed = int(seed)
        return (
            int(cfg["d"]),
            int(cfg["P"]),
            int(cfg["N"]),
            float(cfg["chi"]),
            seed,
            float(cfg["temperature"]),
            cfg.get("eps"),
            cfg.get("s0"),
        )

    name = dir_path.name
    # Standard training-run dirname:
    #   d{d}_P{P}_N{N}_chi_{chi}_lr_{lr}_T_{T}_seed_{seed}_eps_{eps}[_s0_{s0}_sigmaW0_{...}][_schedule]
    std = re.match(
        r"d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)_chi_(?P<chi>[\d.]+)"
        r"_lr_[\d.eE+-]+_T_(?P<T>[\d.]+)_seed_(?P<seed>\d+)"
        r"(?:_eps_(?P<eps>[\d.]+))?(?:_s0_(?P<s0>[\d.]+))?",
        name,
    )
    if std:
        return (
            int(std.group("d")),
            int(std.group("P")),
            int(std.group("N")),
            float(std.group("chi")),
            int(std.group("seed")),
            float(std.group("T")),
            float(std.group("eps")) if std.group("eps") is not None else None,
            float(std.group("s0")) if std.group("s0") is not None else 1.0,
        )
    ba = re.match(
        r"(?:invariant_|learnable_)?beta(?P<beta>[\d.]+)_alpha(?P<alpha>[\d.]+)_d(?P<d>\d+)_P(?P<P>\d+)_N(?P<N>\d+)"
        r"_sa0(?P<sa0>[\d.]+)_kappa(?P<kappa>[\d.]+)_seed(?P<seed>\d+)",
        name,
    )
    if ba:
        n = int(ba.group("N"))
        kappa = float(ba.group("kappa"))
        return (
            int(ba.group("d")),
            int(ba.group("P")),
            n,
            float(n),
            int(ba.group("seed")),
            2.0 * kappa,
            0.0,
            1.0,
        )
    raise ValueError(f"Could not parse config from {dirname}")


def reconstruct_training_inputs(P, d, seed, device):
    if seed is not None:
        torch.manual_seed(int(seed))
    gen_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    X = torch.randn(int(P), int(d), device=gen_device)
    return X.to(device)


def make_training_dataset(P, d, seed, eps, device):
    X = reconstruct_training_inputs(P, d, seed, device)
    z = X[:, 0]
    he3 = (z ** 3 - 3.0 * z) / (6.0 ** 0.5)
    y = z + float(eps) * he3
    return X, y, z, he3


def load_model(model_dir, device, use_a_snapshots=False):
    """Load FCN2 from the final checkpoint (snapshot averaging is call-site specific)."""
    del use_a_snapshots
    d, P, N, chi, seed, *_ = parse_config_from_dirname(model_dir)
    model_dir_path = Path(model_dir)
    for candidate in ("model_final.pt", "model.pt", "checkpoint.pt"):
        model_path = model_dir_path / candidate
        if model_path.exists():
            break
    else:
        print(f"Model not found in {model_dir}")
        return None, None, None
    if candidate == "checkpoint.pt":
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    import json

    cfg_path = model_dir_path / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        sigma_w0 = float(cfg.get("sigmaW0", 1.0 / d))
        sigma_a = float(cfg.get("sigmaA", 1.0 / (N * chi)))
    else:
        sigma_w0 = 1.0 / d
        sigma_a = 1.0 / (N * chi)

    ens = state_dict["W0"].shape[0]
    model = FCN2NetworkActivationGeneric(
        d=d,
        n1=N,
        P=P,
        ens=ens,
        activation="erf",
        weight_initialization_variance=(sigma_w0, sigma_a),
        device=device,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, d, P, seed


def make_dataset(P: int, d: int, seed: int, eps: float, device: torch.device):
    """Replay the training set: CUDA RNG, P points, y = He1 + eps He3."""
    return make_training_dataset(P, d, seed, eps, device)


# Offset so the held-out draw never collides with the training RNG stream.
_TEST_SEED_OFFSET = 1_000_003


def make_test_dataset(
    P: int,
    d: int,
    seed: int,
    eps: float,
    device: torch.device,
    P_test: int | None = None,
):
    """Independent Gaussian test set with the same target rule as training."""
    n = int(P if P_test is None else P_test)
    test_seed = int(seed) + _TEST_SEED_OFFSET
    return make_training_dataset(n, d, test_seed, eps, device)


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Empirical inner product <a, b> = (1/P) sum_i a_i b_i."""
    return torch.mean(a * b)


def he1_residual(values: torch.Tensor, he1: torch.Tensor) -> torch.Tensor:
    """f - <f, He1> He1, using the Gaussian-orthonormal convention E[He1^2] = 1."""
    return values - inner(values, he1) * he1


def cubic_learnability(values: torch.Tensor, he1: torch.Tensor, he3: torch.Tensor, y: torch.Tensor, eps: float) -> float:
    """(<f, He3> - <f, He1> <He1, He3>) / <y, He3>."""
    y_he3 = inner(y, he3)
    if abs(float(y_he3.item())) < 1e-12:
        return float("nan")
    return float(((inner(values, he3) - inner(values, he1) * inner(he1, he3)) / y_he3).item())


def krr_ridge_coef(phi_train: torch.Tensor, y_train: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Solve (Phi^T Phi + sigma2 N I) coef = Phi^T y. phi: (P, N), y: (P,) -> coef (N,)."""
    n1 = phi_train.shape[1]
    gram = phi_train.T @ phi_train
    ridge = sigma2 * n1
    eye = torch.eye(n1, device=phi_train.device, dtype=phi_train.dtype)
    return torch.linalg.solve(gram + ridge * eye, phi_train.T @ y_train)


def krr_from_features(phi: torch.Tensor, y: torch.Tensor, sigma2: float) -> torch.Tensor:
    """In-sample: f = K(K + sigma2 I)^{-1} y with K = phi phi^T / N."""
    return phi @ krr_ridge_coef(phi, y, sigma2)


def krr_predict_from_train(
    phi_train: torch.Tensor,
    y_train: torch.Tensor,
    phi_eval: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """Train ridge on phi_train,y_train; predict on phi_eval."""
    return phi_eval @ krr_ridge_coef(phi_train, y_train, sigma2)


def arcsin_kernel(X: torch.Tensor, X2: torch.Tensor | None = None) -> torch.Tensor:
    """Infinite-width erf KRR: K_uv = (2/π) arcsin( 2 (x_u·x_v)/d / (s_u s_v) )

    with s_i = sqrt(1 + 2 ||x_i||^2 / d). This is E[erf(h) erf(h')] when
    h = W0 x and W0 has variance 1/d. If X2 is given, returns the cross-kernel.
    """
    d = X.shape[1]
    if X2 is None:
        X2 = X
    gram = (X @ X2.T) / d
    scale_x = torch.sqrt(1.0 + 2.0 * torch.sum(X * X, dim=1) / d)
    scale_y = torch.sqrt(1.0 + 2.0 * torch.sum(X2 * X2, dim=1) / d)
    arg = 2.0 * gram / (scale_x[:, None] * scale_y[None, :])
    arg = torch.clamp(arg, -1.0 + 1e-6, 1.0 - 1e-6)
    return (2.0 / torch.pi) * torch.arcsin(arg)


def arcsin_krr_predict(X: torch.Tensor, y: torch.Tensor, sigma2: float) -> torch.Tensor:
    """In-sample f = K (K + sigma^2 I)^{-1} y with the infinite-width arcsin kernel."""
    return arcsin_krr_predict_from_train(X, y, X, sigma2)


def arcsin_krr_predict_from_train(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """Train arcsin ridge on (X_train, y_train); predict on X_eval."""
    K_tt = arcsin_kernel(X_train)
    p = K_tt.shape[0]
    eye = torch.eye(p, device=X_train.device, dtype=X_train.dtype)
    alpha = torch.linalg.solve(K_tt + sigma2 * eye, y_train)
    K_et = arcsin_kernel(X_eval, X_train)
    return K_et @ alpha


def compare_one_model(
    model,
    X_eval,
    y_train,
    sigma2: float,
    raw_preact: bool,
    X_train=None,
):
    """Model forward on X_eval; feature KRR fit on train, predict on eval.

    If X_train is None, fit and evaluate in-sample on X_eval (legacy train path).
    """
    X_fit = X_eval if X_train is None else X_train
    with torch.no_grad():
        model_out = model.forward(X_eval)  # (P_eval, ens)
        if raw_preact:
            feats_eval = model.h0_preactivation(X_eval)
            feats_fit = feats_eval if X_train is None else model.h0_preactivation(X_fit)
        else:
            feats_eval = model.h0_activation(X_eval)
            feats_fit = feats_eval if X_train is None else model.h0_activation(X_fit)

    krr = torch.empty_like(model_out)
    for q in range(model.ens):
        krr[:, q] = krr_predict_from_train(
            feats_fit[:, q, :], y_train, feats_eval[:, q, :], sigma2
        )

    return model_out, krr


def _fill_split_panels(
    *,
    idx: int,
    ncols: int,
    axes_agree,
    axes_target,
    axes_x0,
    axes_he3,
    model_dir: str,
    N: int,
    P_train: int,
    P_eval: int,
    sigma2: float,
    eps_val: float,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    he1_eval: torch.Tensor,
    he3_eval: torch.Tensor,
    model_out: torch.Tensor,
    krr: torch.Tensor,
    split_label: str,
    summary_rows: list,
    he3_summary_rows: list,
):
    model_np = model_out.detach().cpu().numpy()
    krr_np = krr.detach().cpu().numpy()
    y_np = y_eval.detach().cpu().numpy()
    he1_np = he1_eval.detach().cpu().numpy()
    he3_np = he3_eval.detach().cpu().numpy()
    x0_np = he1_np
    y_he3 = float(inner(y_eval, he3_eval).item())
    print(
        f"{split_label} set P_eval={P_eval} (P_train={P_train}) "
        f"X.device={X_eval.device} <y,He3>={y_he3:.4g}"
    )

    model_mean_t = torch.as_tensor(model_np.mean(axis=1), device=he1_eval.device, dtype=he1_eval.dtype)
    krr_mean_t = torch.as_tensor(krr_np.mean(axis=1), device=he1_eval.device, dtype=he1_eval.dtype)
    model_mean = model_mean_t.detach().cpu().numpy()
    krr_mean = krr_mean_t.detach().cpu().numpy()

    if abs(eps_val) < 1e-12 or abs(y_he3) < 1e-12:
        model_cubic = np.full_like(model_mean, np.nan)
        krr_cubic = np.full_like(krr_mean, np.nan)
        target_cubic = np.full_like(y_np, np.nan)
        arcsin_cubic = np.full_like(y_np, np.nan)
        arcsin_learned = float("nan")
        cubic_learned = float("nan")
        he3_corr = float("nan")
        he3_mse = float("nan")
    else:
        denom = y_he3
        model_cubic = (he1_residual(model_mean_t, he1_eval) / denom).detach().cpu().numpy()
        krr_cubic = (he1_residual(krr_mean_t, he1_eval) / denom).detach().cpu().numpy()
        target_cubic = (he1_residual(y_eval, he1_eval) / denom).detach().cpu().numpy()
        arcsin_pred = arcsin_krr_predict_from_train(X_train, y_train, X_eval, sigma2)
        arcsin_cubic = (he1_residual(arcsin_pred, he1_eval) / denom).detach().cpu().numpy()
        arcsin_learned = cubic_learnability(arcsin_pred, he1_eval, he3_eval, y_eval, eps_val)
        cubic_learned = cubic_learnability(model_mean_t, he1_eval, he3_eval, y_eval, eps_val)
        he3_corr = float(np.corrcoef(model_cubic, he3_np)[0, 1])
        he3_mse = float(np.mean((model_cubic - he3_np) ** 2))
    he3_summary_rows.append(
        (Path(model_dir).name, P_train, P_eval, eps_val, y_he3, cubic_learned, he3_corr, he3_mse)
    )
    print(
        f"  He3 ({split_label}): <y,He3>={y_he3:.4g}  model={cubic_learned:.4f}  "
        f"feature KRR={cubic_learnability(krr_mean_t, he1_eval, he3_eval, y_eval, eps_val):.4f}  "
        f"arcsin KRR={arcsin_learned:.4f}  corr vs He3={he3_corr:.4f}"
    )
    mse_model_krr = float(np.mean((model_np - krr_np) ** 2))
    mse_model_y = float(np.mean((model_mean - y_np) ** 2))
    mse_krr_y = float(np.mean((krr_mean - y_np) ** 2))
    corr = float(np.corrcoef(model_np.ravel(), krr_np.ravel())[0, 1])
    summary_rows.append(
        (Path(model_dir).name, P_train, P_eval, sigma2, mse_model_krr, mse_model_y, mse_krr_y, corr)
    )
    print(
        f"{Path(model_dir).name} [{split_label}]: P_train={P_train} P_eval={P_eval} "
        f"sigma2={sigma2:.4g}  MSE(model,krr)={mse_model_krr:.4e}  "
        f"MSE(model,y)={mse_model_y:.4e}  MSE(krr,y)={mse_krr_y:.4e}  corr={corr:.4f}"
    )

    r, c = divmod(idx, ncols)
    ax = axes_agree[r][c]
    lo = min(krr_np.min(), model_np.min())
    hi = max(krr_np.max(), model_np.max())
    ax.scatter(krr_np.ravel(), model_np.ravel(), s=4, alpha=0.25, color="royalblue", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"N={N}, P={P_train} ({split_label})\ncorr={corr:.3f}")
    ax.set_xlabel("KRR  K(K+σ²)⁻¹ y")
    ax.set_ylabel("Model output")
    ax.grid(True, alpha=0.3)

    ax_t = axes_target[r][c]
    lo_t = min(y_np.min(), model_mean.min(), krr_mean.min())
    hi_t = max(y_np.max(), model_mean.max(), krr_mean.max())
    ax_t.scatter(y_np, krr_mean, s=8, alpha=0.55, color="darkorange", label="KRR", rasterized=True)
    ax_t.scatter(y_np, model_mean, s=8, alpha=0.7, color="royalblue", label="model", rasterized=True)
    ax_t.plot([lo_t, hi_t], [lo_t, hi_t], "k--", lw=1)
    ax_t.set_title(f"N={N}, P={P_train} ({split_label})")
    ax_t.set_xlabel("True target y")
    ax_t.set_ylabel("Output")
    ax_t.grid(True, alpha=0.3)
    if idx == 0:
        ax_t.legend(fontsize=8)

    ax2 = axes_x0[r][c]
    ax2.scatter(x0_np, y_np, s=6, alpha=0.35, color="0.6", label="y", rasterized=True)
    ax2.scatter(x0_np, krr_mean, s=8, alpha=0.7, color="darkorange", label="KRR", rasterized=True)
    ax2.scatter(x0_np, model_mean, s=8, alpha=0.7, color="royalblue", label="model", rasterized=True)
    ax2.set_title(f"N={N}, P={P_train} ({split_label})")
    ax2.set_xlabel("X[:,0]")
    ax2.set_ylabel("output")
    ax2.grid(True, alpha=0.3)
    if idx == 0:
        ax2.legend(fontsize=8)

    ax3 = axes_he3[r][c]
    order = np.argsort(x0_np)
    ax3.scatter(x0_np, target_cubic, s=6, alpha=0.35, color="0.6", label="target residual / ε", rasterized=True, zorder=1)
    ax3.scatter(x0_np, krr_cubic, s=8, alpha=0.5, color="darkorange", label="feature KRR", rasterized=True, zorder=2)
    ax3.scatter(x0_np, model_cubic, s=8, alpha=0.55, color="royalblue", label="model", rasterized=True, zorder=3)
    ax3.plot(x0_np[order], he3_np[order], "k--", lw=1.2, label=r"$\mathrm{He}_3(x_0)$", zorder=4)
    ax3.scatter(x0_np, arcsin_cubic, s=10, alpha=0.75, color="seagreen", label="arcsin KRR", rasterized=True, zorder=5)
    ax3.set_title(f"N={N}, P={P_train} ({split_label})  model={cubic_learned:.3f}  arcsin={arcsin_learned:.3f}")
    ax3.set_xlabel(r"$X[:,0]$")
    ax3.set_ylabel(r"$(f - \langle f,\mathrm{He}_1\rangle\mathrm{He}_1)/\langle y,\mathrm{He}_3\rangle$")
    ax3.grid(True, alpha=0.3)
    if idx == 0:
        ax3.legend(fontsize=7)


def _save_split_figures(
    *,
    group_name: str,
    feat_name: str,
    output_dir: Path,
    split_label: str,
    n: int,
    nrows: int,
    ncols: int,
    fig_agree,
    axes_agree,
    fig_target,
    axes_target,
    fig_x0,
    axes_x0,
    fig_he3,
    axes_he3,
    summary_rows: list,
    he3_summary_rows: list,
):
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_agree[r][c].axis("off")
        axes_target[r][c].axis("off")
        axes_x0[r][c].axis("off")
        axes_he3[r][c].axis("off")

    suffix = "" if split_label == "train" else f"_{split_label}"
    split_title = "train" if split_label == "train" else "test (KRR fit on train)"

    fig_agree.suptitle(
        f"{group_name}: model vs KRR [{split_title}] | features = {feat_name}", fontsize=12
    )
    fig_agree.tight_layout()
    agree_path = output_dir / f"model_vs_krr_scatter{suffix}.png"
    fig_agree.savefig(agree_path, dpi=150)
    plt.close(fig_agree)

    fig_target.suptitle(
        f"{group_name}: model / KRR vs true target y [{split_title}] | features = {feat_name}",
        fontsize=12,
    )
    fig_target.tight_layout()
    target_path = output_dir / f"model_vs_target_scatter{suffix}.png"
    fig_target.savefig(target_path, dpi=150)
    plt.close(fig_target)

    fig_x0.suptitle(
        f"{group_name}: y / KRR / model vs X[:,0] [{split_title}] | features = {feat_name}",
        fontsize=12,
    )
    fig_x0.tight_layout()
    x0_path = output_dir / f"model_krr_vs_x0{suffix}.png"
    fig_x0.savefig(x0_path, dpi=150)
    plt.close(fig_x0)

    fig_he3.suptitle(
        rf"{group_name}: $(f - \langle f,\mathrm{{He}}_1\rangle \mathrm{{He}}_1)/\langle y,\mathrm{{He}}_3\rangle$ vs $X[:,0]$"
        f" [{split_title}] | features = {feat_name}",
        fontsize=12,
    )
    fig_he3.tight_layout()
    he3_path = output_dir / f"cubic_he3_learned{suffix}.png"
    fig_he3.savefig(he3_path, dpi=150)
    plt.close(fig_he3)

    summary_path = output_dir / f"summary{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(f"group={group_name}  split={split_label}  features={feat_name}\n")
        f.write(
            f"{'run':<80} {'P_tr':>6} {'P_ev':>6} {'sigma2':>10} "
            f"{'MSE_m_krr':>14} {'MSE_m_y':>12} {'MSE_krr_y':>12} {'corr':>8}\n"
        )
        for row in summary_rows:
            f.write(
                f"{row[0]:<80} {row[1]:6d} {row[2]:6d} {row[3]:10.4g} "
                f"{row[4]:14.4e} {row[5]:12.4e} {row[6]:12.4e} {row[7]:8.4f}\n"
            )
        f.write("\nHe3 cubic: (f - <f,He1> He1) / <y,He3> vs x0; learned = that residual's He3 coeff:\n")
        f.write(
            f"{'run':<80} {'P_tr':>6} {'P_ev':>6} {'eps':>8} {'<y,He3>':>10} "
            f"{'learned':>10} {'corr':>8} {'MSE_vs_He3':>12}\n"
        )
        for row in he3_summary_rows:
            f.write(
                f"{row[0]:<80} {row[1]:6d} {row[2]:6d} {row[3]:8.4g} {row[4]:10.4g} "
                f"{row[5]:10.4f} {row[6]:8.4f} {row[7]:12.4e}\n"
            )
    print(f"Saved {agree_path}")
    print(f"Saved {target_path}")
    print(f"Saved {x0_path}")
    print(f"Saved {he3_path}")
    print(f"Saved {summary_path}")


def plot_group(
    group_name: str,
    device: torch.device,
    sigma2_override: float | None,
    raw_preact: bool,
    P_test: int | None = None,
):
    group = EXPERIMENT_GROUP_BY_NAME[group_name]
    output_dir = OUTPUT_BASE_DIR / slugify(group.name) / "model_vs_preact_krr"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = list(group.model_dirs)
    if not model_dirs:
        print(f"No model dirs in group {group_name}")
        return

    n = len(model_dirs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    feat_name = "W0 x (raw preact)" if raw_preact else "erf(W0 x) (readout features)"
    kappa_eff_cache: dict[tuple, float] = {}

    split_state = {}
    for split_label in ("train", "test"):
        fig_agree, axes_agree = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
        )
        fig_target, axes_target = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
        )
        fig_x0, axes_x0 = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
        )
        fig_he3, axes_he3 = plt.subplots(
            nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False
        )
        split_state[split_label] = {
            "fig_agree": fig_agree,
            "axes_agree": axes_agree,
            "fig_target": fig_target,
            "axes_target": axes_target,
            "fig_x0": fig_x0,
            "axes_x0": axes_x0,
            "fig_he3": fig_he3,
            "axes_he3": axes_he3,
            "summary_rows": [],
            "he3_summary_rows": [],
        }

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        eps_val = float(epsilon) if epsilon is not None else 0.0
        kappa_bare = float(T) / 2.0
        if sigma2_override is not None:
            sigma2 = float(sigma2_override)
            sigma2_label = "override"
        else:
            cache_key = (int(d), int(P), float(kappa_bare), int(N), float(chi))
            if cache_key not in kappa_eff_cache:
                try:
                    kappa_eff_cache[cache_key] = float(
                        compute_kappa_eff(
                            d=int(d),
                            P=int(P),
                            kappa_bare=float(kappa_bare),
                            n1=int(N),
                            chi=float(chi),
                            device=device,
                            verbose=False,
                        )
                    )
                except Exception as exc:
                    print(f"kappa_eff failed ({exc}); falling back to kappa_bare={kappa_bare}")
                    kappa_eff_cache[cache_key] = float(kappa_bare)
            sigma2 = kappa_eff_cache[cache_key]
            sigma2_label = "kappa_eff"
        print(
            f"Ridge sigma2={sigma2:.6g} ({sigma2_label}; kappa_bare={kappa_bare:.6g})"
        )

        X_train, y_train, he1_train, he3_train = make_dataset(P, d, seed, eps_val, device)
        X_test, y_test, he1_test, he3_test = make_test_dataset(
            P, d, seed, eps_val, device, P_test=P_test
        )

        model_train, krr_train = compare_one_model(
            model, X_train, y_train, sigma2, raw_preact, X_train=None
        )
        model_test, krr_test = compare_one_model(
            model, X_test, y_train, sigma2, raw_preact, X_train=X_train
        )

        for split_label, X_eval, y_eval, he1_eval, he3_eval, model_out, krr in (
            ("train", X_train, y_train, he1_train, he3_train, model_train, krr_train),
            ("test", X_test, y_test, he1_test, he3_test, model_test, krr_test),
        ):
            st = split_state[split_label]
            _fill_split_panels(
                idx=idx,
                ncols=ncols,
                axes_agree=st["axes_agree"],
                axes_target=st["axes_target"],
                axes_x0=st["axes_x0"],
                axes_he3=st["axes_he3"],
                model_dir=model_dir,
                N=N,
                P_train=P,
                P_eval=int(X_eval.shape[0]),
                sigma2=sigma2,
                eps_val=eps_val,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                he1_eval=he1_eval,
                he3_eval=he3_eval,
                model_out=model_out,
                krr=krr,
                split_label=split_label,
                summary_rows=st["summary_rows"],
                he3_summary_rows=st["he3_summary_rows"],
            )

    for split_label in ("train", "test"):
        st = split_state[split_label]
        _save_split_figures(
            group_name=group_name,
            feat_name=feat_name,
            output_dir=output_dir,
            split_label=split_label,
            n=n,
            nrows=nrows,
            ncols=ncols,
            fig_agree=st["fig_agree"],
            axes_agree=st["axes_agree"],
            fig_target=st["fig_target"],
            axes_target=st["axes_target"],
            fig_x0=st["fig_x0"],
            axes_x0=st["axes_x0"],
            fig_he3=st["fig_he3"],
            axes_he3=st["axes_he3"],
            summary_rows=st["summary_rows"],
            he3_summary_rows=st["he3_summary_rows"],
        )


def main():
    parser = argparse.ArgumentParser(description="Model vs preactivation-kernel KRR")
    parser.add_argument(
        "--group",
        default="AlphaBetaInvariant",
        choices=list(EXPERIMENT_GROUP_BY_NAME),
        help="Experiment group (default: AlphaBetaInvariant)",
    )
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: cuda if available)")
    parser.add_argument(
        "--sigma2",
        type=float,
        default=None,
        help="Noise variance in K(K + sigma2 I)^{-1} y. Default: kappa_eff from kappa_bare=T/2.",
    )
    parser.add_argument(
        "--raw-preact",
        action="store_true",
        help="Use raw W0 x instead of erf(W0 x). Default uses readout features erf(W0 x).",
    )
    parser.add_argument(
        "--P-test",
        type=int,
        default=None,
        help="Held-out test-set size (default: same as each run's train P).",
    )
    args = parser.parse_args()
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    plot_group(args.group, device, args.sigma2, args.raw_preact, P_test=args.P_test)


if __name__ == "__main__":
    main()
