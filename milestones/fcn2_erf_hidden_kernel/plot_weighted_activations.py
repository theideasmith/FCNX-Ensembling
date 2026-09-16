#!/usr/bin/env python3
"""Plot weighted hidden activations A_i * erf(w_i · x) on the train set.

For each hidden unit i: A_i * erf(W_i · X[p,:]) vs preactivation h_i = W_i · X,
with points sorted by h_i for readable curves. All units share one color/alpha;
overlapping lines darken where many units cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))

from plot_action_h0_activation import (  # noqa: E402
    EXPERIMENT_GROUP_BY_NAME,
    OUTPUT_BASE_DIR,
    load_model,
    make_training_dataset,
    parse_config_from_dirname,
    slugify,
)


def weighted_activations(model, X: torch.Tensor, ens_index: int = 0) -> torch.Tensor:
    """Return (P, n1): wa[p, i] = A[q, i] * erf(W0[q, i] · X[p,:])."""
    with torch.no_grad():
        act = model.h0_activation(X)  # (P, ens, n1)
        A = model.A[ens_index]  # (n1,)
        return act[:, ens_index, :] * A.unsqueeze(0)


def unit_preactivations(model, X: torch.Tensor, ens_index: int = 0) -> torch.Tensor:
    """Return (P, n1): h[p, i] = W0[q, i] · X[p,:]."""
    with torch.no_grad():
        return model.h0_preactivation(X)[:, ens_index, :]


def top_a_percentile_indices(A: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    """Unit indices with |A_i| in the top top_frac fraction (default top 10%)."""
    thresh = np.percentile(np.abs(A), 100.0 * (1.0 - top_frac))
    return np.flatnonzero(np.abs(A) >= thresh)


def wa_grid_sorted_by_preactivation(h0_np: np.ndarray, wa_np: np.ndarray) -> np.ndarray:
    """Return (N, P) grid with row i sorted by h_i = W_i · X."""
    n_units = wa_np.shape[1]
    wa_grid = np.empty((n_units, wa_np.shape[0]), dtype=np.float64)
    for i in range(n_units):
        order = np.argsort(h0_np[:, i])
        wa_grid[i, :] = wa_np[order, i]
    return wa_grid


UNIT_LINE_COLOR = (0.15, 0.35, 0.65)


def default_line_alpha(n_units: int) -> float:
    return min(0.15, 8.0 / max(n_units, 1))


def line_collection_uniform(
    ax,
    h0_np: np.ndarray,
    wa_np: np.ndarray,
    unit_indices: np.ndarray | None = None,
    linewidth: float = 0.55,
    color: tuple[float, float, float] = UNIT_LINE_COLOR,
    alpha: float = 0.02,
) -> mcollections.LineCollection:
    """LineCollection with each unit sorted by h_i; uniform color, alpha for overdraw."""
    if unit_indices is None:
        unit_indices = np.arange(wa_np.shape[1])
    segments = []
    for i in unit_indices:
        order = np.argsort(h0_np[:, i])
        segments.append(np.column_stack([h0_np[order, i], wa_np[order, i]]))
    segments_arr = np.asarray(segments, dtype=np.float64)
    lc = mcollections.LineCollection(
        segments_arr,
        colors=color,
        alpha=alpha,
        linewidths=linewidth,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(lc)
    return lc


def mean_curve_sorted_by_preactivation(
    h0_np: np.ndarray,
    wa_np: np.ndarray,
    unit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean (h, wa) over units at each rank along h-sorted curves."""
    if unit_indices is None:
        h_sub = h0_np
        wa_sub = wa_np
    else:
        h_sub = h0_np[:, unit_indices]
        wa_sub = wa_np[:, unit_indices]
    n_units = wa_sub.shape[1]
    p = wa_sub.shape[0]
    h_grid = np.empty((n_units, p), dtype=np.float64)
    wa_grid = np.empty((n_units, p), dtype=np.float64)
    for i in range(n_units):
        order = np.argsort(h_sub[:, i])
        h_grid[i, :] = h_sub[order, i]
        wa_grid[i, :] = wa_sub[order, i]
    return h_grid.mean(axis=0), wa_grid.mean(axis=0)


def plot_mean_on_twin_axis(
    ax,
    h0_np: np.ndarray,
    wa_np: np.ndarray,
    unit_indices: np.ndarray | None = None,
    *,
    color: str = "crimson",
    lw: float = 2.0,
    label: str = r"$\langle A_i\,\mathrm{erf}(h_i)\rangle_i$",
) -> plt.Axes:
    """Plot neuron-averaged curve on a right-hand y-axis (separate scale)."""
    h_mean, wa_mean = mean_curve_sorted_by_preactivation(h0_np, wa_np, unit_indices)
    ax_mean = ax.twinx()
    ax_mean.plot(h_mean, wa_mean, color=color, lw=lw, label=label, zorder=5)
    ax_mean.set_ylabel(label, color=color)
    ax_mean.tick_params(axis="y", labelcolor=color)
    wa_pad = 0.05 * (wa_mean.max() - wa_mean.min() + 1e-9)
    ax_mean.set_ylim(wa_mean.min() - wa_pad, wa_mean.max() + wa_pad)
    return ax_mean


def h_axis_limits(h0_np: np.ndarray, unit_indices: np.ndarray | None = None) -> tuple[float, float]:
    if unit_indices is None:
        h_sub = h0_np
    else:
        h_sub = h0_np[:, unit_indices]
    return float(h_sub.min()), float(h_sub.max())


def plot_models(
    model_dirs: list[str],
    output_dir: Path,
    title: str,
    device: torch.device,
    ens_index: int,
    line_alpha: float | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dirs:
        print("No model dirs provided")
        return

    n = len(model_dirs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig_hm, axes_hm = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
    fig_lines, axes_lines = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    fig_top_a, axes_top_a = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)
    fig_max_a, axes_max_a = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)

    for idx, model_dir in enumerate(model_dirs):
        d, P, N, chi, seed, T, epsilon, s0 = parse_config_from_dirname(model_dir)
        model, *_ = load_model(model_dir, device)
        if model is None:
            continue
        eps_val = float(epsilon) if epsilon is not None else 0.0
        X, y, _, _ = make_training_dataset(P, d, seed, eps_val, device)

        wa = weighted_activations(model, X, ens_index=ens_index)  # (P, n1)
        h0 = unit_preactivations(model, X, ens_index=ens_index)  # (P, n1)
        with torch.no_grad():
            out_ens = model.forward(X)[:, ens_index]
        wa_np = wa.detach().cpu().numpy()
        h0_np = h0.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        wa_sum = wa_np.sum(axis=1)
        verify = float(np.max(np.abs(wa_sum - out_ens.detach().cpu().numpy())))
        A_np = model.A[ens_index].detach().cpu().numpy()
        abs_a = np.abs(A_np)
        alpha = line_alpha if line_alpha is not None else default_line_alpha(N)

        r, c = divmod(idx, ncols)

        ax_hm = axes_hm[r][c]
        wa_grid = wa_grid_sorted_by_preactivation(h0_np, wa_np)  # (n1, P), row i sorted by h_i
        vmax = np.percentile(np.abs(wa_grid), 99)
        im = ax_hm.imshow(
            wa_grid,
            aspect="auto",
            origin="upper",
            extent=[0, P, 0, N],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_hm.set_title(f"P={P}  N={N}  ens={ens_index}")
        ax_hm.set_xlabel(r"samples sorted by $h_i$")
        ax_hm.set_ylabel("unit i")
        fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label=r"$A_i\,\mathrm{erf}(w_i\!\cdot\! x)$")

        ax_ln = axes_lines[r][c]

        line_collection_uniform(ax_ln, h0_np, wa_np, linewidth=0.55, alpha=alpha)
        ax_ln.plot(
            [],
            [],
            color=UNIT_LINE_COLOR,
            lw=1.0,
            alpha=min(1.0, alpha * 20),
            label=rf"all {N} units (overdraw)",
        )
        ax_ln_mean = plot_mean_on_twin_axis(ax_ln, h0_np, wa_np)

        y_lo = wa_np.min()
        y_hi = wa_np.max()
        pad = 0.05 * (y_hi - y_lo + 1e-9)
        h_lo, h_hi = h_axis_limits(h0_np)
        ax_ln.set_xlim(h_lo, h_hi)
        ax_ln.set_ylim(y_lo - pad, y_hi + pad)

        ax_ln.set_title(f"P={P}  N={N}  ens={ens_index}")
        ax_ln.set_xlabel(r"$h_i = W_i \cdot X$  (preactivation)")
        ax_ln.set_ylabel(r"$A_i\,\mathrm{erf}(h_i)$  (per unit)")
        ax_ln.grid(True, alpha=0.3)
        if idx == 0:
            h_left, l_left = ax_ln.get_legend_handles_labels()
            h_right, l_right = ax_ln_mean.get_legend_handles_labels()
            ax_ln.legend(h_left + h_right, l_left + l_right, fontsize=7, loc="best")

        top_idx = top_a_percentile_indices(A_np, top_frac=0.10)
        n_top = len(top_idx)
        ax_top = axes_top_a[r][c]
        line_collection_uniform(
            ax_top,
            h0_np,
            wa_np,
            unit_indices=top_idx,
            linewidth=0.8,
            alpha=alpha,
        )
        ax_top.plot(
            [],
            [],
            color=UNIT_LINE_COLOR,
            lw=1.0,
            alpha=min(1.0, alpha * 20),
            label=rf"top 10% $|A_i|$ ({n_top} units)",
        )
        ax_top_mean = plot_mean_on_twin_axis(ax_top, h0_np, wa_np, unit_indices=top_idx)
        y_top_lo = wa_np[:, top_idx].min()
        y_top_hi = wa_np[:, top_idx].max()
        top_pad = 0.05 * (y_top_hi - y_top_lo + 1e-9)
        h_top_lo, h_top_hi = h_axis_limits(h0_np, top_idx)
        ax_top.set_xlim(h_top_lo, h_top_hi)
        ax_top.set_ylim(y_top_lo - top_pad, y_top_hi + top_pad)
        ax_top.set_title(f"P={P}  top 10% $|A_i|$  ({n_top} units)")
        ax_top.set_xlabel(r"$h_i = W_i \cdot X$  (preactivation)")
        ax_top.set_ylabel(r"$A_i\,\mathrm{erf}(h_i)$  (per unit)")
        ax_top.grid(True, alpha=0.3)
        if idx == 0:
            h_left, l_left = ax_top.get_legend_handles_labels()
            h_right, l_right = ax_top_mean.get_legend_handles_labels()
            ax_top.legend(h_left + h_right, l_left + l_right, fontsize=7, loc="best")

        i_max = int(np.argmax(abs_a))
        wa_max = wa_np[:, i_max]
        h_max = h0_np[:, i_max]
        ax_max = axes_max_a[r][c]
        order_h = np.argsort(h_max)
        ax_max.plot(
            h_max[order_h],
            wa_max[order_h],
            color=UNIT_LINE_COLOR,
            lw=1.6,
            label=rf"$A_i\,\mathrm{{erf}}(h)$, $i={i_max}$, $A_i={A_np[i_max]:.4g}$",
            zorder=3,
        )
        h_pad = 0.05 * (wa_max.max() - wa_max.min() + 1e-9)
        ax_max.set_xlim(h_max.min(), h_max.max())
        ax_max.set_ylim(wa_max.min() - h_pad, wa_max.max() + h_pad)
        ax_max.set_title(f"P={P}  max $|A_i|$  i={i_max}  $|A|$={abs_a[i_max]:.4g}")
        ax_max.set_xlabel(r"$h_i = W_i \cdot X$  (preactivation)")
        ax_max.set_ylabel(r"$A_i\,\mathrm{erf}(h_i)$")
        ax_max.grid(True, alpha=0.3)
        ax_max.legend(fontsize=6, loc="best")

        mse_sum = float(np.mean((wa_sum - y_np) ** 2))
        print(
            f"{Path(model_dir).name}: P={P} N={N}  verify max|sum_i wa - f|={verify:.2e}  "
            f"sum_i wa vs y MSE={mse_sum:.4e}  max|wa|={np.abs(wa_np).max():.4g}  "
            f"i_max={i_max}"
        )

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_hm[r][c].axis("off")
        axes_lines[r][c].axis("off")
        axes_top_a[r][c].axis("off")
        axes_max_a[r][c].axis("off")

    fig_hm.suptitle(
        rf"{title}: weighted activations sorted by $h_i$ (heatmap)",
        fontsize=12,
    )
    fig_hm.tight_layout()
    hm_path = output_dir / "weighted_activations_heatmap.png"
    fig_hm.savefig(hm_path, dpi=150)
    plt.close(fig_hm)

    fig_lines.suptitle(
        rf"{title}: all weighted activations vs $h_i$ (sorted, uniform overdraw)",
        fontsize=12,
    )
    fig_lines.tight_layout()
    lines_path = output_dir / "weighted_activations_lines.png"
    fig_lines.savefig(lines_path, dpi=150)
    plt.close(fig_lines)

    fig_top_a.suptitle(
        rf"{title}: top 10% $|A_i|$ weighted activations vs $h_i$ (sorted)",
        fontsize=12,
    )
    fig_top_a.tight_layout()
    top_a_path = output_dir / "weighted_activations_top10pct_A.png"
    fig_top_a.savefig(top_a_path, dpi=150)
    plt.close(fig_top_a)

    fig_max_a.suptitle(
        rf"{title}: max $|A_i|$ unit vs preactivation $h_i=W_i\!\cdot\!X$ (smooth)",
        fontsize=11,
    )
    fig_max_a.tight_layout()
    max_a_path = output_dir / "weighted_activations_max_abs_A.png"
    fig_max_a.savefig(max_a_path, dpi=150)
    plt.close(fig_max_a)

    print(f"Saved {hm_path}")
    print(f"Saved {lines_path}")
    print(f"Saved {top_a_path}")
    print(f"Saved {max_a_path}")


def plot_group(group_name: str, device: torch.device, ens_index: int, line_alpha: float | None):
    group = EXPERIMENT_GROUP_BY_NAME[group_name]
    output_dir = OUTPUT_BASE_DIR / slugify(group.name) / "weighted_activations"
    plot_models(list(group.model_dirs), output_dir, group_name, device, ens_index, line_alpha)


def main():
    parser = argparse.ArgumentParser(description="Plot A_i * erf(w_i·x) vs h_i=W_i·X (sorted)")
    parser.add_argument(
        "--group",
        default="SampleComplexityTestI",
        choices=list(EXPERIMENT_GROUP_BY_NAME),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ens-index",
        type=int,
        default=0,
        help="Ensemble member for W0/A (default: 0)",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=None,
        help="Alpha per unit curve. Default: min(0.15, 8/N).",
    )
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    plot_group(args.group, device, args.ens_index, args.line_alpha)


if __name__ == "__main__":
    main()
