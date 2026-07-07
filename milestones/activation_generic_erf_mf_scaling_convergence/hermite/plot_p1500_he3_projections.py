#!/usr/bin/env python3
"""Plot He3/H1 projections over checkpoints for the Hermite P=1500 run.

This script reuses the same Hermite feature construction as the existing
Hermite analysis helpers:

* target coordinate: x[:, 0]
* perp coordinate: x[:, 1]
* He3(x) = (x^3 - 3x) / sqrt(6)

For each checkpoint it computes the second moments of the projected h1 and
h3 features, then plots the evolution across training checkpoints to make
equilibration visible.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric


RUN_DIR_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "hermite_activation"
    / "d150_P500_N200_chi200_kappa0.1"
)
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "p1500_he3_projection_plots"


def _checkpoint_step(path: Path) -> Tuple[int, str]:
    if path.name == "model.pt":
        return (10**12, path.name)
    if path.name == "model_final.pt":
        return (10**12 - 1, path.name)
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match is None:
        return (-1, path.name)
    return (int(match.group(1)), path.name)


def _sorted_checkpoints(seed_dir: Path) -> List[Path]:
    checkpoints: List[Path] = []
    for pattern in ("model_*.pt", "model.pt", "model_final.pt"):
        checkpoints.extend(path for path in seed_dir.glob(pattern) if path.is_file())

    unique: Dict[str, Path] = {}
    for checkpoint in checkpoints:
        unique[checkpoint.name] = checkpoint

    ordered = list(unique.values())
    ordered.sort(key=_checkpoint_step)
    return ordered


def _sample_checkpoints(checkpoints: Sequence[Path], max_checkpoints: int) -> List[Path]:
    if max_checkpoints <= 0 or len(checkpoints) <= max_checkpoints:
        return list(checkpoints)

    keep_indices = {0, len(checkpoints) - 1}
    if max_checkpoints > 2:
        linspace = torch.linspace(0, len(checkpoints) - 1, steps=max_checkpoints)
        keep_indices.update(int(round(float(idx))) for idx in linspace)
    return [checkpoints[idx] for idx in sorted(keep_indices)] + [checkpoints[0]]


def _load_config(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(config_path, "r") as handle:
        return json.load(handle)


def _strip_singleton_seed_dim(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim in (3, 4) and value.shape[0] == 1:
            cleaned[key] = value.squeeze(0)
        else:
            cleaned[key] = value
    return cleaned


def _load_state_dict(checkpoint_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif not isinstance(payload, dict) and hasattr(payload, "state_dict"):
        try:
            payload = payload.state_dict()
        except Exception:
            return None

    if not isinstance(payload, dict):
        return None
    if not all(key in payload for key in ("W0", "W1", "A")):
        return None
    return _strip_singleton_seed_dim(payload)


def _build_model(config: Dict[str, object], device: torch.device) -> FCN3NetworkActivationGeneric:
    d = int(config.get("d"))
    n1 = int(config.get("n1", config.get("N")))
    n2 = int(config.get("n2", config.get("N", n1)))
    p = int(config.get("P"))
    ens = int(config.get("ens", config.get("ensembles", 1)))
    activation = str(config.get("activation", "hermite3"))

    model = FCN3NetworkActivationGeneric(
        d=d,
        n1=n1,
        n2=n2,
        P=p,
        ens=ens,
        activation=activation,
        device=device,
    ).to(device)
    return model


def _load_checkpoint_model(run_dir: Path, checkpoint_path: Path, device: torch.device) -> FCN3NetworkActivationGeneric:
    config = _load_config(run_dir)
    model = _build_model(config, device)
    state_dict = _load_state_dict(checkpoint_path)
    if state_dict is None:
        raise ValueError(f"Could not load a state dict from {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _moment(values: torch.Tensor) -> float:
    values = values.reshape(-1)
    return float(torch.mean(values ** 2).item())


def _project_h1_h3(
    model: FCN3NetworkActivationGeneric,
    d: int,
    device: torch.device,
    samples: int,
    batch_size: int,
    seed: int = 0,
    target_dim: int = 0,
    perp_dim: int = 1,
) -> Dict[str, float]:
    dtype = torch.float32
    model.to(device=device, dtype=dtype)

    ens = model.ens
    # reproducible RNG like analysis_hermite_scaling.compute_projection_variances
    torch.manual_seed(int(seed))
    # Match the accumulation used in analysis_hermite_scaling.py:
    # - h1_preactivation -> shape (bs, ens, n2) -> reshape to (bs, ens*n2)
    # - accumulate two projections: x0 (He1) and He3 = (x0^3 - 3 x0)/sqrt(6)
    n2 = model.n2
    accumulator = torch.zeros((2, ens * n2), dtype=dtype, device=device)
    x_batch = torch.empty((batch_size, d), dtype=dtype, device=device)
    sqrt6 = float(6.0 ** 0.5)

    def _step(x: torch.Tensor) -> torch.Tensor:
        h1 = model.h1_preactivation(x).reshape(x.shape[0], -1)  # (bs, ens*n2)
        x_target = x[:, target_dim]
        he3_target = (x_target ** 3 - 3.0 * x_target) / sqrt6
        # matmul: (bs,) @ (bs, ens*n2) -> (ens*n2,)
        proj_x = torch.matmul(x_target, h1)
        proj_he3 = torch.matmul(he3_target, h1)
        return torch.stack([proj_x, proj_he3], dim=0)

    with torch.no_grad():
        full_batches = samples // batch_size
        remainder = samples % batch_size
        for batch_index in range(full_batches + (1 if remainder > 0 else 0)):
            current_batch = batch_size if batch_index < full_batches else remainder
            if current_batch == 0:
                break
            current_x = x_batch if current_batch == batch_size else x_batch[:current_batch]
            current_x.normal_()
            accumulator.add_(_step(current_x))

    # normalize by total samples and reshape to (2, ens, n2)
    coeffs = (accumulator / float(samples)).view(2, ens, n2)
    h11_emp = float(coeffs[0].var(unbiased=False).item())
    h33_emp = float(coeffs[1].var(unbiased=False).item())

    return {"H11_emp": h11_emp, "H33_emp": h33_emp}


def _checkpoint_sort_key(entry: Tuple[str, int]) -> Tuple[int, str]:
    name, step = entry
    return (step, name)


def _aggregate(records: List[Dict[str, object]]) -> Dict[str, Dict[str, List[float]]]:
    grouped: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        checkpoint = str(record["checkpoint"])
        for key in ("H11_emp", "H33_emp"):
            value = record.get(key)
            if isinstance(value, (int, float)):
                grouped[checkpoint][key].append(float(value))
    return {checkpoint: dict(metrics) for checkpoint, metrics in grouped.items()}


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() == 1:
        return float(tensor.item()), 0.0
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot He3 projections over Hermite P=1500 checkpoints.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT, help="Hermite run directory (used as template for other P values)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT, help="Directory for plots and JSON output")
    parser.add_argument("--samples", type=int, default=100_000, help="Random samples per checkpoint")
    parser.add_argument("--batch-size", type=int, default=2_000, help="Batch size for projection accumulation")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Maximum checkpoints per seed to sample; 0 uses all checkpoints")
    parser.add_argument("--seed", type=str, default="", help="Optional single seed directory name to inspect")
    parser.add_argument("--p-values", type=int, nargs="+", default=[1500], help="List of P values to overlay (e.g. --p-values 500 1000 1500)")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Determine original P in the default run dir name for replacement
    base_name = args.run_dir.name
    m = re.search(r"_P(\d+)_", base_name)
    origP = m.group(1) if m else None

    p_values = args.p_values
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(p_values) - 1)) for i in range(len(p_values))]

    # Containers to collect per-P summaries for plotting
    per_p_summaries = {}

    for i_p, p in enumerate(p_values):
        # construct run dir for this P by replacing the _P{orig}_ token if possible
        if origP is not None:
            run_dir_p = args.run_dir.parent / base_name.replace(f"_P{origP}_", f"_P{p}_")
        else:
            # fallback: look for a directory that contains _P{p}_
            candidates = list(args.run_dir.parent.glob(f"*P{p}*"))
            run_dir_p = candidates[0] if candidates else None

        if run_dir_p is None or not run_dir_p.exists():
            print(f"Warning: run dir for P={p} not found; skipping")
            continue

        seed_dirs = [path for path in sorted(run_dir_p.glob("seed*")) if path.is_dir()]
        if args.seed:
            seed_dirs = [path for path in seed_dirs if path.name == args.seed]
        if not seed_dirs:
            print(f"Warning: no seed dirs under {run_dir_p}; skipping P={p}")
            continue

        all_seed_records: List[Dict[str, object]] = []
        per_seed_results: List[Dict[str, object]] = []

        for seed_dir in seed_dirs:
            checkpoints = _sorted_checkpoints(seed_dir)
            checkpoints = _sample_checkpoints(checkpoints, args.max_checkpoints)

            seed_records: List[Dict[str, object]] = []
            config = _load_config(seed_dir)
            d = int(config.get("d"))

            print(f"P={p} {seed_dir.name}: processing {len(checkpoints)} checkpoints")
            for checkpoint_path in checkpoints:
                step, _ = _checkpoint_step(checkpoint_path)
                model = _load_checkpoint_model(seed_dir, checkpoint_path, device)
                metrics = _project_h1_h3(
                    model,
                    d=d,
                    device=device,
                    samples=args.samples,
                    batch_size=min(args.batch_size, args.samples),
                    seed=int(config.get("seed", 0)),
                )
                record = {
                    "seed": seed_dir.name,
                    "checkpoint": checkpoint_path.name,
                    "step": step,
                    **metrics,
                }
                seed_records.append(record)
                all_seed_records.append(record)
                print(f"  {checkpoint_path.name}: H11_emp={metrics.get('H11_emp', float('nan')):.4e} H33_emp={metrics.get('H33_emp', float('nan')):.4e}")

            per_seed_results.append({"seed": seed_dir.name, "records": seed_records})

        aggregated = _aggregate(all_seed_records)
        checkpoint_entries = sorted({(record["checkpoint"], int(record["step"])) for record in all_seed_records}, key=_checkpoint_sort_key)

        # compute mean/std per checkpoint for this P
        x_values = list(range(len(checkpoint_entries)))
        checkpoint_labels = [name for name, _ in checkpoint_entries]
        metric_names = ["H11_emp", "H33_emp"]

        summary_rows: List[Dict[str, object]] = []
        for checkpoint_name, _ in checkpoint_entries:
            row: Dict[str, object] = {"checkpoint": checkpoint_name}
            metrics = aggregated.get(checkpoint_name, {})
            for metric_name in metric_names:
                values = metrics.get(metric_name, [])
                if values:
                    mean, std = _mean_std(values)
                    row[metric_name] = {"mean": mean, "std": std, "count": len(values)}
                else:
                    row[metric_name] = {"mean": None, "std": None, "count": 0}
            summary_rows.append(row)

        per_p_summaries[p] = {
            "run_dir": str(run_dir_p),
            "checkpoint_entries": checkpoint_entries,
            "aggregated": aggregated,
            "summary_rows": summary_rows,
            "per_seed_results": per_seed_results,
            "color": colors[i_p],
        }

    # Plot overlay of H11 and H33 for each P
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    for ax, metric_name, title in zip(axes, ("H11_emp", "H33_emp"), ("H1 variance (H11)", "He3 variance (H33)")):
        for p in sorted(per_p_summaries.keys()):
            summary = per_p_summaries[p]
            checkpoint_entries = summary["checkpoint_entries"]
            aggregated = summary["aggregated"]
            x_vals = list(range(len(checkpoint_entries)))
            means = []
            stds = []
            for checkpoint_name, _ in checkpoint_entries:
                vals = aggregated.get(checkpoint_name, {}).get(metric_name, [])
                if vals:
                    mean, std = _mean_std(vals)
                else:
                    mean, std = float("nan"), float("nan")
                means.append(mean)
                stds.append(std)

            means_t = torch.tensor(means, dtype=torch.float32)
            stds_t = torch.tensor(stds, dtype=torch.float32)
            color = summary.get("color")
            ax.plot(x_vals, means_t, label=f"P={p}", color=color)
            ax.fill_between(x_vals, means_t - stds_t, means_t + stds_t, color=color, alpha=0.12)
        ax.set_title(title)
        ax.set_ylabel("second moment")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # set x ticks for the largest checkpoint list
    # find the p with max checkpoints
    if per_p_summaries:
        p_max = max(per_p_summaries.keys(), key=lambda k: len(per_p_summaries[k]["checkpoint_entries"]))
        tick_count = min(10, max(1, len(per_p_summaries[p_max]["checkpoint_entries"])))
        tick_positions = [int(round(i)) for i in torch.linspace(0, len(per_p_summaries[p_max]["checkpoint_entries"]) - 1, steps=tick_count).tolist()]
        tick_labels = [per_p_summaries[p_max]["checkpoint_entries"][pos][0] for pos in tick_positions]
        for ax in axes:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_xlabel("checkpoint")

    fig.suptitle("He3 / H1 projection comparison across P values", y=0.98)
    fig.tight_layout()
    plot_path = args.out_dir / f"{args.run_dir.name}_he_projection_comparison_Ps_{'_'.join(str(p) for p in p_values)}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    json_path = args.out_dir / f"{args.run_dir.name}_he_projection_comparison_Ps_{'_'.join(str(p) for p in p_values)}.json"
    results = {"p_values": p_values, "per_p_summaries": per_p_summaries, "plot_path": str(plot_path)}
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved comparison plot to {plot_path}")
    print(f"Saved comparison summary to {json_path}")


if __name__ == "__main__":
    main()