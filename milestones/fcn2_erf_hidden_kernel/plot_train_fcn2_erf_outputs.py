#!/usr/bin/env python3
"""
Standalone plotting utility for outputs produced by train_fcn2_erf.py

Generates figures from:
- eigenvalues_over_time.json
- losses.json

Usage examples:
    python plot_train_fcn2_erf_outputs.py --run-dir d10_P30_N256_chi_1.0_lr_1e-05_T_1.0
    python plot_train_fcn2_erf_outputs.py --run-dir /absolute/path/to/run_dir --no-log-y

The script saves plots back into the same run directory:
- eigenvalues_over_time.png (line plot, all eigenvalues)
- eigenvalues_heatmap.png   (eigenvalue index x epoch heatmap, log color scale)
- eigenvalues_final.png     (final spectrum, sorted)
- loss_curve.png            (mean loss with std band)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Headless-friendly
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float32)
import subprocess
import tempfile
try:
    import imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# Add lib to path to import the network class
LIB_PATH = Path(__file__).parent.parent.parent / "lib"
if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))
from FCN2Network import FCN2NetworkActivationGeneric


def _compute_julia_fcn2_erf_predictions(d: int, n1: int, P: int,
                                        chi: float | None = None,
                                        kappa: float | None = None,
                                        anneal: bool = True,
                                        anneal_steps: int = 30000,
                                        lr: float = 1e-3,
                                        max_iter: int = 50000,
                                        tol: float = 1e-8) -> dict | None:
    """
    Call the Julia script compute_fcn2_erf_eigs.jl to get theoretical eigenvalues.

    Returns a dict like {"lJ": ..., "lk": ..., "lJP": ..., "lkp": ...} or None on failure.
    """
    try:
        julia_script = Path(__file__).parent.parent / "julia_lib" / "compute_fcn2_erf_eigs.jl"
        if not julia_script.exists():
            # Try repository root fallback
            julia_script = Path(__file__).resolve().parents[2] / "julia_lib" / "compute_fcn2_erf_eigs.jl"
        if not julia_script.exists():
            print(f"Warning: Julia script not found: {julia_script}")
            return None

        cmd = [
            "julia", str(julia_script),
            "--d", str(float(d)),
            "--n1", str(float(n1)),
            "--P", str(float(P)),
            "--lr", str(lr),
            "--max_iter", str(int(max_iter)),
            "--tol", str(tol),
            "--anneal_steps", str(int(anneal_steps)),
        ]
        # chi, kappa defaults inside Julia if not provided
        if chi is not None:
            cmd += ["--chi", str(float(chi))]
        if kappa is not None:
            cmd += ["--kappa", str(float(kappa))]
        if anneal is True:
            # anneal by default; julia uses --no-anneal to disable
            pass
        else:
            cmd += ["--no-anneal"]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            out_path = Path(tf.name)
        cmd += ["--output", str(out_path)]

        env = dict(**os.environ)
        # Some environments need DISPLAY unset for headless
        env.pop("DISPLAY", None)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True, timeout=180)
        if res.returncode != 0:
            print("Warning: Julia compute failed:\n", res.stderr)
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        if not out_path.exists():
            print("Warning: Julia did not produce output JSON.")
            return None

        import json
        with open(out_path, "r") as f:
            data = json.load(f)
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Warning: failed to run Julia predictions: {e}")
        return None


def _load_eigenvalues(run_dir: Path):
    ev_path = run_dir / "eigenvalues_over_time.json"
    if not ev_path.exists():
        raise FileNotFoundError(f"No eigenvalues file found at {ev_path}")
    with open(ev_path, "r") as f:
        data = json.load(f)
    epochs = sorted(int(k) for k in data.keys())
    # Handle keys as strings in JSON
    ev = np.array([data[str(e)] for e in epochs], dtype=float)
    return epochs, ev


essential_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def plot_eigenvalues_over_time(epochs, eigenvalues, out_path: Path,
                               log_x: bool = True, log_y: bool = True,
                               alpha: float = 0.6, lw: float = 1.2):
    fig, ax = plt.subplots(figsize=(10, 6))
    num_modes = eigenvalues.shape[1]
    for i in range(num_modes):
        color = essential_colors[i % len(essential_colors)] if num_modes <= 10 else None
        ax.plot(epochs, eigenvalues[:, i], alpha=alpha, linewidth=lw, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("H Eigenvalues over Training")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eigenvalues_heatmap(epochs, eigenvalues, out_path: Path):
    # Avoid log(0) for color scale
    eps = 1e-12
    log_vals = np.log10(np.maximum(eigenvalues, eps))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(log_vals.T, aspect='auto', origin='lower',
                   extent=[min(epochs), max(epochs), 0, eigenvalues.shape[1]],
                   cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(Eigenvalue)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue Index")
    ax.set_title("H Eigenvalues Heatmap (log scale)")
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_spectrum(eigenvalues, out_path: Path, log_y: bool = True):
    final = np.array(eigenvalues[-1], dtype=float)
    final_sorted = np.sort(final)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, final_sorted.size + 1), final_sorted, marker='o', markersize=3, linewidth=1.2)
    ax.set_xlabel("Mode (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Final H Eigenvalue Spectrum")
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_losses(run_dir: Path):
    loss_path = run_dir / "losses.json"
    if not loss_path.exists():
        return None
    with open(loss_path, "r") as f:
        data = json.load(f)
    losses = {int(k): float(v) for k, v in data.get("losses", {}).items()}
    loss_stds = {int(k): float(v) for k, v in data.get("loss_stds", {}).items()}
    if not losses:
        return None
    epochs = sorted(losses.keys())
    mean = np.array([losses[e] for e in epochs], dtype=float)
    std = np.array([loss_stds.get(e, 0.0) for e in epochs], dtype=float)
    return epochs, mean, std


def plot_losses(epochs, mean, std, out_path: Path, log_x: bool = True, log_y: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mean, color="#1f77b4", label="mean loss")
    ax.fill_between(epochs, np.maximum(mean - std, 0.0), mean + std, color="#1f77b4", alpha=0.2, label="std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (sum over ensembles)")
    ax.set_title("Training Loss")
    if log_x:
        ax.set_xscale('log')
    if log_y:
        # Guard against zeros
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_config(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        return json.load(f)


def _parse_run_dir_params(run_dir: Path):
    """Parse parameters from run directory name.

    Expected pattern (from training script):
      d{d}_P{P}_N{N}_chi_{chi}_lr_{lr}_T_{temperature}

    Returns a dict with keys: d, P, N, chi, lr, temperature
    Missing keys are omitted.
    """
    import re
    name = run_dir.name
    params = {}
    # Simple integer params
    for key in ["d", "P", "N"]:
        m = re.search(rf"{key}(\d+)", name)
        if m:
            params[key] = int(m.group(1))
    # Float-like params with explicit key_ prefix
    def parse_float_after(prefix):
        m = re.search(rf"{prefix}_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)
        return float(m.group(1)) if m else None
    for key in ["chi", "lr", "T", "temperature"]:
        val = parse_float_after(key)
        if val is not None:
            k = "temperature" if key in ("T", "temperature") else key
            params[k] = val
    return params


def _load_pred_vs_true(run_dir: Path):
    pred_path = run_dir / "pred_vs_true.json"
    if not pred_path.exists():
        return None
    with open(pred_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def _instantiate_model_matching_state(state: dict, d: int, P: int, N: int, ens: int):
    """Instantiate a model whose shapes match the loaded state dict.

    If the ensemble size (or n1/d) in the saved weights differ from the provided
    parameters, re-instantiate with the inferred shapes to avoid load errors.
    """
    # Infer shapes from 'W0' if available: (ens_loaded, n1_loaded, d_loaded)
    new_d, new_N, new_ens = d, N, ens
    try:
        W0 = state.get("W0", None)
        if isinstance(W0, torch.Tensor):
            shp = tuple(W0.shape)
        elif W0 is not None:
            # Some checkpoints may store tensors as other types; attempt to read .size()
            shp = tuple(W0.size()) if hasattr(W0, "size") else None
        else:
            shp = None
        if shp is not None and len(shp) == 3:
            new_ens, new_N, new_d = shp[0], shp[1], shp[2]
    except Exception:
        pass

    # Recreate model with inferred shapes
    model = FCN2NetworkActivationGeneric(
        new_d, new_N, P, ens=new_ens,
        activation="erf",
        weight_initialization_variance=(1/new_d, 1/new_N)
    )
    # Load with strict to catch unexpected mismatches
    model.load_state_dict(state, strict=True)
    return model


def _load_trained_model(run_dir: Path, d: int, P: int, N: int, ens: int):
    """Instantiate model and load weights from the run directory.

    Preference order for weights: checkpoint.pt -> model_final.pt -> model.pt
    Returns the model on CPU, ensuring ensemble size matches the loaded weights.
    """
    # Try checkpoint first
    ckpt_path = run_dir / "checkpoint.pt"
    final_path = run_dir / "model_final.pt"
    model_path = run_dir / "model.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", None)
        if state is not None:
            return _instantiate_model_matching_state(state, d, P, N, ens)

    # Fallbacks
    if final_path.exists():
        state = torch.load(final_path, map_location="cpu")
        return _instantiate_model_matching_state(state, d, P, N, ens)

    if model_path.exists():
        state = torch.load(model_path, map_location="cpu")
        return _instantiate_model_matching_state(state, d, P, N, ens)

    raise FileNotFoundError("No model weights found (checkpoint.pt/model_final.pt/model.pt)")


def plot_pred_vs_true(run_dir: Path, out_path: Path):
    """Recreate training data, load trained model, and scatter Y_true vs Y_pred.

    The training script uses torch.manual_seed(42) for data and Y = X[:, 0].
    """
    cfg = _load_config(run_dir)
    if cfg is None:
        params = _parse_run_dir_params(run_dir)
        if not all(k in params for k in ("d", "P", "N")):
            raise FileNotFoundError(
                f"Missing config.json and could not parse d/P/N from run dir name: {run_dir.name}"
            )
        d = int(params["d"])
        P = int(params["P"])
        N = int(params["N"])
        ens = 5  # default ensemble size used in training
    else:
        d = int(cfg["d"])
        P = int(cfg["P"])
        N = int(cfg["N"])
        ens = int(cfg.get("ens", 5))

    # Recreate dataset
    torch.manual_seed(42)
    X = torch.randn(3000, d)
    Y = X[:, 0].unsqueeze(-1)

    # Load model
    model = _load_trained_model(run_dir, d, P, N, ens)
    model.eval()
    with torch.no_grad():
        Yhat_ens = model(X)  # (P, ens)
        # Use ensemble mean prediction
        Yhat = Yhat_ens.mean(dim=1).cpu().numpy()
    Y_true = Y.squeeze(-1).cpu().numpy()

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(Y_true, Yhat, s=12, c="#1f77b4", alpha=0.7, edgecolors="none")
    # Identity line
    min_v = float(min(Y_true.min(), Yhat.min()))
    max_v = float(max(Y_true.max(), Yhat.max()))
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.2, label="y = x")
    ax.set_xlabel("True target y")
    ax.set_ylabel("Predicted ŷ (ensemble mean)")
    ax.set_title("Prediction vs. True Target")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Bar plot of eigenvalues with error bars from ensemble std
    result = model.H_eig(X, X, std=True)
    if isinstance(result, tuple):
        eigenvalues, eigenvalues_std = result
        eigenvalues = eigenvalues.cpu().numpy()
        eigenvalues_std = eigenvalues_std.cpu().numpy()
    else:
        eigenvalues = result.cpu().numpy()
        eigenvalues_std = None
    
    # Sort by eigenvalue magnitude
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sort_idx]
    eigenvalues_std_sorted = eigenvalues_std[sort_idx] if eigenvalues_std is not None else None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(1, eigenvalues_sorted.size + 1)
    if eigenvalues_std_sorted is not None:
        ax.errorbar(x_pos, eigenvalues_sorted, yerr=eigenvalues_std_sorted,
                   fmt='o-', markersize=5, linewidth=1.2, capsize=4, capthick=1.5,
                   elinewidth=1.5, alpha=0.8, color='#1f77b4', ecolor='#ff7f0e')
    else:
        ax.plot(x_pos, eigenvalues_sorted, marker='o', markersize=3, linewidth=1.2)
    ax.set_xlabel("Mode (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("H Eigenvalue Spectrum at Final Model (with ensemble std)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add Julia theoretical predictions as horizontal lines (δ=1 and δ=0)
    try:
        # chi defaults to n1 (N) in many of our scripts
        chi_val = float(N)
        pred = _compute_julia_fcn2_erf_predictions(d=d, n1=N, P=P, chi=chi_val)
        if isinstance(pred, dict):
            lk = pred.get("lJ", None)   # delta=1.0
            lkp = pred.get("lJP", None) # delta=0.0
            if isinstance(lk, (int, float)) and np.isfinite(lk):
                ax.axhline(lk, color="k", linestyle="--", linewidth=1.2, alpha=0.9, label="Julia lk (δ=1)")
            if isinstance(lkp, (int, float)) and np.isfinite(lkp):
                ax.axhline(lkp, color="#444", linestyle=":", linewidth=1.2, alpha=0.9, label="Julia lk (δ=0)")
            # Show legend if any prediction lines added
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="best")
    except Exception as e:
        print(f"Warning: could not overlay Julia predictions: {e}")

    fig.tight_layout()
    eig_out_path = out_path.parent / "eigenvalues_final_model.png"
    fig.savefig(eig_out_path, dpi=150)
    # Also save JPG copy for downstream workflows
    eig_out_jpg = out_path.parent / "eigenvalues_final_model.jpg"
    try:
        fig.savefig(eig_out_jpg, dpi=150)
    except Exception:
        pass
    plt.close(fig)

    # Storing final eigenvalues in a JSON file
    eig_json_path = out_path.parent / "eigenvalues_final_model.json"
    print("Logging json to: ", eig_json_path)
    eig_dict = {str(i+1): {"value": float(ev), "std": float(std_val)} 
                if eigenvalues_std_sorted is not None else float(ev)
                for i, (ev, std_val) in enumerate(zip(eigenvalues_sorted, 
                                                        eigenvalues_std_sorted if eigenvalues_std_sorted is not None 
                                                        else [0.0]*len(eigenvalues_sorted)))}
    with open(eig_json_path, "w") as f:
        json.dump(eig_dict, f, indent=4)


def plot_slope_over_epochs(pred_data: dict, out_path: Path, log_x: bool = False):
    epochs = sorted(pred_data.keys())
    slopes = [pred_data[e].get("slope", float('nan')) for e in epochs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, slopes, color="#ff7f0e", lw=1.8, label="slope")
    ax.axhline(1.0, color="k", linestyle="--", lw=1.0, alpha=0.7, label="slope = 1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fit slope (ŷ vs y)")
    ax.set_title("Prediction Alignment Over Training")
    if log_x:
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_alignment_video(pred_data: dict, out_path: Path, fps: int = 8):
    if not HAS_IMAGEIO:
        print("Warning: imageio not available; skipping video generation.")
        return None
    epochs = sorted(pred_data.keys())
    all_vals = []
    for rec in pred_data.values():
        all_vals.extend(rec["y_true"])
        all_vals.extend(rec["y_pred_mean"])
    vmin, vmax = min(all_vals), max(all_vals)
    span = vmax - vmin
    pad = 0.05 * (span if span > 0 else 1.0)
    lo, hi = vmin - pad, vmax + pad
    frames = []
    for e in epochs:
        rec = pred_data[e]
        y_true = np.array(rec["y_true"])
        y_pred = np.array(rec["y_pred_mean"])
        slope = rec.get("slope", float('nan'))
        intercept = rec.get("intercept", float('nan'))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, s=18, c="#1f77b4", alpha=0.75, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
        if not np.isnan(slope) and not np.isnan(intercept):
            ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color="#ff7f0e", lw=1.2, label=f"fit slope={slope:.3f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True target y")
        ax.set_ylabel("Predicted ŷ")
        ax.set_title(f"Epoch {e}")
        ax.legend()
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # Use RGBA buffer and drop alpha channel (robust across backends)
        frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame_rgba[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot outputs from train_fcn2_erf run directory")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory containing JSON outputs")
    parser.add_argument("--no-log-x", action="store_true", help="Disable log scale on x-axis where applicable")
    parser.add_argument("--no-log-y", action="store_true", help="Disable log scale on y-axis where applicable")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--no-pred-vs-true", action="store_true", help="Skip generating predicted vs true scatter plot")
    parser.add_argument("--no-video", action="store_true", help="Skip generating the alignment video")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the alignment video")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    # Load data
    epochs_ev, eigenvalues = _load_eigenvalues(run_dir)
    losses_loaded = _load_losses(run_dir)
    pred_data = _load_pred_vs_true(run_dir)

    # Plot eigenvalues over time (lines)
    out1 = run_dir / "eigenvalues_over_time.png"
    plot_eigenvalues_over_time(
        epochs_ev, eigenvalues, out1,
        log_x=not args.no_log_x, log_y=not args.no_log_y,
    )

    # Heatmap
    out2 = run_dir / "eigenvalues_heatmap.png"
    plot_eigenvalues_heatmap(epochs_ev, eigenvalues, out2)

    # Final spectrum
    out3 = run_dir / "eigenvalues_final.png"
    plot_final_spectrum(eigenvalues, out3, log_y=not args.no_log_y)

    # Loss curve (if available)
    if losses_loaded is not None:
        epochs_loss, mean_loss, std_loss = losses_loaded
        out4 = run_dir / "loss_curve.png"
        plot_losses(epochs_loss, mean_loss, std_loss, out4, log_x=not args.no_log_x, log_y=not args.no_log_y)

    # Prediction vs True target
    out5 = None
    if not args.no_pred_vs_true:
        try:
            out5 = run_dir / "pred_vs_true.png"
            plot_pred_vs_true(run_dir, out5)
        except Exception as e:
            print(f"Warning: could not generate pred-vs-true plot: {e}")

    out6 = None
    if pred_data is not None:
        out6 = run_dir / "slope_over_epochs.png"
        plot_slope_over_epochs(pred_data, out6, log_x=not args.no_log_x)

    out7 = None
    if pred_data is not None and not args.no_video:
        out7 = run_dir / "pred_vs_true_video.mp4"
        res = make_alignment_video(pred_data, out7, fps=args.fps)
        if res is None:
            out7 = None

    print("Saved:")
    print(f"  - {out1}")
    print(f"  - {out2}")
    print(f"  - {out3}")
    if losses_loaded is not None:
        print(f"  - {out4}")
    if out5 is not None:
        print(f"  - {out5}")
    if out6 is not None:
        print(f"  - {out6}")
    if out7 is not None:
        print(f"  - {out7}")


if __name__ == "__main__":
    main()
