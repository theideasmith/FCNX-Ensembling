#!/usr/bin/env python3
"""
Diagnostics for FCN2 ERF run: per-channel weight stats, covariance trace, and
weight norm statistics, with theory values recorded.

This script is tailored for the specific run directory and parameters:
- Run dir: /home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P200_N200_chi_200.0_lr_5e-05_T_2.0
- Hardcoded params from filename: d=50, P=200, N=200, chi=200.0, lr=5e-05, T=2.0

Outputs:
- weights_channel_means_std.png: Bar plot of per-channel mean with std error bars
- weight_norms_hist.png: Histogram of per-unit L2 norms with mean/std annotated
- weight_cov_stats.json: JSON with covariance trace and norm comparisons
- theory_used.json: JSON echoing the provided theory dictionary for reference

Notes:
- No command-line args; paths and params are hardcoded for this experiment.
- Ensemble size is inferred from the saved state dict shape.
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure we can import the model class
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN2Network import FCN2NetworkActivationGeneric


def main():
    # Hardcoded run parameters from the directory name (do not parse at runtime)
    RUN_DIR = Path("d150_P600_N700_chi_700.0_lr_5e-06_T_0.1_seed_42/")#Path("/home/akiva/FCNX-Ensembling/milestones/fcn2_erf_hidden_kernel/d50_P200_N200_chi_200.0_lr_5e-05_T_2.0")
    d = 150
    P = 600
    N = 700
    chi = 700.0
    lr = 5e-06
    T = 0.1

    # Record of theory numbers to include alongside outputs
    theory = {
    "P": 600,
    "lWT": 0.13370406894148495,
    "b": 0.4244131815783876,
    "n1": 700,
    "lJ": 0.05231511814347966,
    "delta": 0,
    "chi": 700,
    "lWP": 0.006666666666666667,
    "lk": 0.9968217737264263,
    "kappa": 0.05,
    "lkp": 4.206704429243757e-16,
    "lJP": 0.002829421210522826,
    "d": 150
    }
    # Preferred loading: checkpoint.pt (has metadata) -> model.pt
    ckpt_path = RUN_DIR / "checkpoint.pt"
    model_path = RUN_DIR / "model.pt"
    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location="cpu")
        state_dict = blob["model_state_dict"]
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {RUN_DIR} (checkpoint.pt or model.pt)")

    # Infer ensemble size, n1, d from W0 tensor shape
    if "W0" not in state_dict:
        raise KeyError("State dict missing key 'W0' for first-layer weights")
    W0_shape = tuple(state_dict["W0"].shape)  # (ens, n1, d)
    ens_infer, n1_infer, d_infer = W0_shape

    # Sanity check shapes against hardcoded params
    if n1_infer != N:
        print(f"[warn] Inferred N={n1_infer} differs from hardcoded N={N}")
    if d_infer != d:
        print(f"[warn] Inferred d={d_infer} differs from hardcoded d={d}")

    ens = ens_infer

    # Build model just to restore shapes (activations/paths unused here)
    model = FCN2NetworkActivationGeneric(d=d, n1=N, P=P, ens=ens, activation="erf")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Extract W0 to numpy on CPU: shape (ens, n1, d)
    W0 = model.W0.detach().cpu().numpy()

    # (2) Per-channel distributions: mean and std for each channel index k
    # Define per-channel stats over ensembles and input dimensions
    means = np.zeros(N)
    stds = np.zeros(N)
    for k in range(N):
        vals = W0[:, k, :].reshape(-1)  # ens * d samples for this channel
        means[k] = vals.mean()
        stds[k] = vals.std(ddof=0)

    # Plot bar chart: mean per channel with std as error bar
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(N)
    ax.bar(x, means, yerr=stds, alpha=0.8, ecolor='black', capsize=2, linewidth=0)
    ax.set_title(f"FCN2 W0 Per-Channel Mean ± STD\n{RUN_DIR.name} (ens={ens}, d={d}, N={N})")
    ax.set_xlabel("Channel index k")
    ax.set_ylabel("Mean(W0[:,k,:]) with STD")
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out_bar = RUN_DIR / "weights_channel_means_std.png"
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)

    # (3) Covariance matrix of weights averaged over ensembles, and its trace
    # For each ensemble q, compute C_q = (1/N) sum_k (w_{qk}-mu_q)(...)^T, then average over q
    C_accum = np.zeros((d, d), dtype=np.float64)
    mu_accum = np.zeros(d, dtype=np.float64)
    for q in range(ens):
        Wq = W0[q]  # (N, d)
        mu_q = Wq.mean(axis=0)  # (d,)
        X = Wq - mu_q
        C_q = (X.T @ X) / float(N)
        C_accum += C_q
        mu_accum += mu_q
    C_avg = C_accum / float(ens)
    mu_global = mu_accum / float(ens)  # mean vector averaged over ensembles
    tr_C = float(np.trace(C_avg))
    # Save the averaged covariance and mean vector for further analysis
    cov_npy = RUN_DIR / "weight_cov_avg.npy"
    mu_npy = RUN_DIR / "weight_cov_mean_vec.npy"
    np.save(cov_npy, C_avg)
    np.save(mu_npy, mu_global)
    # Also produce a heatmap visualization
    figC, axC = plt.subplots(figsize=(6, 5))
    im = axC.imshow(C_avg, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
    axC.set_title("Averaged Weight Covariance C (d×d)")
    axC.set_xlabel("dimension j")
    axC.set_ylabel("dimension i")
    figC.tight_layout()
    cov_png = RUN_DIR / "weight_cov_heatmap.png"
    figC.savefig(cov_png, dpi=150)
    plt.close(figC)

    # (4) Norms of weights across all units (q,k)
    norms = np.linalg.norm(W0.reshape(ens * N, d), axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std(ddof=0))
    norms_sq = norms**2
    norm_sq_mean = float(norms_sq.mean())

    # Approximate mean squared norm by trace(C) (+ mean^2 correction)
    mu_sq = float(np.dot(mu_global, mu_global))
    approx_norm_sq_trace_only = tr_C
    approx_norm_sq_trace_plus_mean = tr_C + mu_sq

    # Histogram of norms
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(norms, bins=40, alpha=0.8, color='steelblue', edgecolor='white')
    ax2.axvline(norm_mean, color='red', linestyle='--', label=f"mean={norm_mean:.4f}")
    ax2.set_title(f"W0 Unit L2 Norms (q,k)\nmean={norm_mean:.4f}, std={norm_std:.4f}")
    ax2.set_xlabel("||w_{qk}||₂")
    ax2.set_ylabel("count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_hist = RUN_DIR / "weight_norms_hist.png"
    fig2.savefig(out_hist, dpi=150)
    plt.close(fig2)

    # (extra) Averaged 4/pi/(1 + 2*||w||^2) across all N*ens weight vectors
    # Using previously computed norms_sq
    four_over_pi = 4.0 / np.pi
    svals = four_over_pi / (1.0 + 2.0 * norms_sq)
    s_mean = float(np.mean(svals))
    s_std = float(np.std(svals, ddof=0))
    # Compare against trace-based and theory-based approximations
    s_trace = float(four_over_pi / (1.0 + 2.0 * tr_C))
    # ((d-1) * lWP + lWT)
    lWT_theory = theory.get("lWT", None)
    lWP_theory = theory.get("lWP", None)
    theory_combo = None
    s_theory = None
    if (lWT_theory is not None) and (lWP_theory is not None):
        theory_combo = (d - 1) * float(lWP_theory) + float(lWT_theory)
        s_theory = float(four_over_pi / (1.0 + 2.0 * theory_combo))

    # (extra) x^T Σ x statistics using the training dataset seed (seed=42) and P samples
    # Recreate X ~ N(0, I) with the same seed and dimensions used during training
    rng = np.random.default_rng(42)
    X_samples = rng.standard_normal(size=(P, d))
    x_sigma_x_vals = np.einsum('pi,ij,pj->p', X_samples, C_avg, X_samples)
    x_sigma_x_mean = float(x_sigma_x_vals.mean())
    x_sigma_x_std = float(x_sigma_x_vals.std(ddof=0))

    # Save numeric stats
    # Eigenvalues of C_avg for largest/bulk statistics
    eigs = np.linalg.eigvalsh(C_avg)
    eigs_sorted = np.sort(eigs)[::-1]  # descending
    lmax = float(eigs_sorted[0]) if eigs_sorted.size > 0 else None
    bulk = eigs_sorted[1:] if eigs_sorted.size > 1 else np.array([])
    bulk_mean = float(bulk.mean()) if bulk.size > 0 else None
    bulk_std = float(bulk.std(ddof=0)) if bulk.size > 0 else None

    # Theory comparison fields pulled into the same JSON for convenience
    theory_lWT = theory.get("lWT", None)
    theory_lWP = theory.get("lWP", None)

    stats = {
        "run_dir": str(RUN_DIR),
        "hardcoded_params": {"d": d, "P": P, "N": N, "chi": chi, "lr": lr, "T": T, "ens": ens},
        "per_channel": {
            "mean": means.tolist(),
            "std": stds.tolist(),
        },
        "covariance": {
            "trace": tr_C,
            "mu_sq": mu_sq,
            "cov_path": str(cov_npy),
            "mu_path": str(mu_npy),
            "eigenvalues": eigs_sorted.tolist(),
            "largest": lmax,
            "bulk_mean": bulk_mean,
            "bulk_std": bulk_std,
        },
        "theory": {"lWT": theory_lWT, "lWP": theory_lWP},
        "comparison": {
            "exp_cov_largest": lmax,
            "exp_cov_bulk_mean": bulk_mean,
            "exp_cov_bulk_std": bulk_std,
            "theory_lWT": theory_lWT,
            "theory_lWP": theory_lWP,
            "delta_largest_vs_lWT": (None if (lmax is None or theory_lWT is None) else float(lmax - theory_lWT)),
            "delta_bulk_mean_vs_lWP": (None if (bulk_mean is None or theory_lWP is None) else float(bulk_mean - theory_lWP)),
        },
        "norms": {
            "mean": norm_mean,
            "std": norm_std,
            "mean_sq": norm_sq_mean,
            "approx_mean_sq_trace_only": approx_norm_sq_trace_only,
            "approx_mean_sq_trace_plus_mean": approx_norm_sq_trace_plus_mean,
        },
        "sigma_factor": {
            "per_unit_mean": s_mean,
            "per_unit_std": s_std,
            "trace_based": s_trace,
            "theory_combo": theory_combo,
            "theory_based": s_theory,
        },
        "x_sigma_x": {
            "mean": x_sigma_x_mean,
            "std": x_sigma_x_std,
            "seed": 42,
            "P": P,
            "d": d,
        },
    }
    out_stats = RUN_DIR / "weight_cov_stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    # Save the theory dictionary for reference/comparison
    out_theory = RUN_DIR / "theory_used.json"
    with open(out_theory, "w") as f:
        json.dump(theory, f, indent=2)

    # Console summary
    print("==== Diagnostics Summary ====")
    print(f"Run: {RUN_DIR.name}")
    print(f"Shapes: ens={ens}, N={N} (inferred {n1_infer}), d={d} (inferred {d_infer})")
    print(f"Per-channel plot: {out_bar}")
    print(f"Norm histogram:   {out_hist}")
    print(f"Covariance trace (avg over ensembles): tr(C) = {tr_C:.6f}")
    print(f"Saved C_avg to: {cov_npy}")
    print(f"Saved covariance heatmap: {cov_png}")
    print(f"Mean squared norm (empirical):          E[||w||^2] = {norm_sq_mean:.6f}")
    print(f"Trace-only approximation:               tr(C)      = {approx_norm_sq_trace_only:.6f}")
    print(f"Trace + mean correction:                tr(C)+||μ||^2 = {approx_norm_sq_trace_plus_mean:.6f}")
    if lmax is not None:
        print(f"Largest eigenvalue of C:                λ_max = {lmax:.6f}")
    if bulk.size > 0:
        print(f"Bulk eigenvalues (rest) mean±std:       {bulk_mean:.6f}±{bulk_std:.6f}")
    print(f"Sigma factor 4/pi/(1+2||w||^2):        mean={s_mean:.6f} std={s_std:.6f}")
    print(f"Sigma factor (trace-based):             {s_trace:.6f}")
    if s_theory is not None:
        print(f"Sigma factor (theory-based):            {s_theory:.6f}")
    print(f"x^T Σ x over seed-42 dataset:           mean={x_sigma_x_mean:.6f} std={x_sigma_x_std:.6f}")
    print(f"Stats JSON: {out_stats}")
    print(f"Theory JSON: {out_theory}")

    # Overlay plot comparing eigenvalue stats vs theory
    try:
        fig_cmp, axs = plt.subplots(1, 2, figsize=(10, 4))
        # Left: λ_max vs lWT
        left_vals = []
        left_labels = []
        if lmax is not None:
            left_vals.append(lmax)
            left_labels.append("exp λ_max")
        if theory_lWT is not None:
            left_vals.append(float(theory_lWT))
            left_labels.append("theory lWT")
        axs[0].bar(np.arange(len(left_vals)), left_vals, color=["C0","C1"]) if left_vals else None
        axs[0].set_xticks(np.arange(len(left_vals)))
        axs[0].set_xticklabels(left_labels, rotation=0)
        axs[0].set_title("Largest eigenvalue vs lWT")
        axs[0].grid(True, axis='y', alpha=0.3)
        # Right: bulk mean±std vs lWP
        right_x = []
        right_vals = []
        right_err = []
        right_labels = []
        if bulk_mean is not None:
            right_x.append(0)
            right_vals.append(bulk_mean)
            right_err.append(bulk_std if bulk_std is not None else 0.0)
            right_labels.append("exp bulk mean")
        if theory_lWP is not None:
            right_x.append(1)
            right_vals.append(float(theory_lWP))
            right_err.append(0.0)
            right_labels.append("theory lWP")
        if right_vals:
            axs[1].bar(right_x, right_vals, yerr=right_err, capsize=3, color=["C0","C1"][:len(right_vals)])
            axs[1].set_xticks(right_x)
            axs[1].set_xticklabels(right_labels, rotation=0)
        axs[1].set_title("Bulk mean (±std) vs lWP")
        axs[1].grid(True, axis='y', alpha=0.3)
        fig_cmp.tight_layout()
        out_cmp = RUN_DIR / "cov_eigen_compare.png"
        fig_cmp.savefig(out_cmp, dpi=150)
        plt.close(fig_cmp)
        print(f"Saved eigenvalue comparison plot: {out_cmp}")
    except Exception as e:
        print(f"Warning: could not save eigenvalue comparison plot: {e}")


if __name__ == "__main__":
    main()
