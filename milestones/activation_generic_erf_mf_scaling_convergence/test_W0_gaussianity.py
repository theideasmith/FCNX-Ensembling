#!/usr/bin/env python3
"""Test Gaussianity of W0 weight distributions.

For each run, performs statistical tests and generates Q-Q plots to assess
whether W0 weights follow a Gaussian distribution. Tests are performed on:
1. The 0th feature column W0[:, :, 0]
2. The mean across features W0.mean(axis=2)

Statistical tests:
- Shapiro-Wilk (for samples < 5000)
- Anderson-Darling
- Kolmogorov-Smirnov

Outputs:
- Q-Q plots with 95% confidence bands
- Histogram with fitted normal overlay
- Statistical test results table

Usage:
    python test_W0_gaussianity.py --base-dir <runs_root> [--dims 150]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

DEVICE_DEFAULT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_run_dirs(base: Path, dims: Optional[List[int]] = None, suffix: str = "") -> List[Tuple[Path, Dict[str, int]]]:
    """Locate seed directories that contain model.pt and match the naming convention."""
    selected: List[Tuple[Path, Dict[str, int]]] = []
    pattern = r"d(\d+)_P(\d+)_N(\d+)_chi(\d+(?:\.\d+)?)"

    model_files = list(base.glob(f"**/*{suffix}*/model.pt")) if suffix else list(base.glob("**/model.pt"))
    for model_file in model_files:
        seed_dir = model_file.parent
        seed_name = seed_dir.name
        m_seed = re.match(r"seed(\d+)", seed_name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))

        cfg_dir = seed_dir.parent
        cfg_name = cfg_dir.name
        m_cfg = re.match(pattern, cfg_name)
        if not m_cfg:
            continue

        d = int(m_cfg.group(1))
        if dims and d not in dims:
            continue
        P = int(m_cfg.group(2))
        N = int(m_cfg.group(3))
        chi = int(float(m_cfg.group(4)))

        cfg = {"d": d, "P": P, "N": N, "chi": chi, "seed": seed}
        selected.append((seed_dir, cfg))

    selected.sort(key=lambda x: (x[1]["d"], x[1]["seed"]))
    print(f"Found {len(selected)} runs with model.pt")
    return selected


def load_model(run_dir: Path, config: Dict[str, int], device: torch.device) -> Optional[FCN3NetworkActivationGeneric]:
    """Load model from run directory."""
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        model_path = run_dir / "model_final.pt"
    if not model_path.exists():
        return None

    d = int(config.get("d"))
    P = int(config.get("P"))
    N = int(config.get("N"))
    chi = int(config.get("chi", N))

    state_dict = torch.load(model_path, map_location=device)
    ens = int(state_dict['W0'].shape[0])

    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return model


def test_normality(data: np.ndarray, label: str) -> Dict[str, any]:
    """Perform normality tests on data.
    
    Returns dict with test statistics and p-values.
    """
    results = {"label": label, "n": len(data)}
    
    # Shapiro-Wilk (subsample if needed, as it's slow for large samples)
    try:
        if len(data) > 5000:
            # Subsample for Shapiro-Wilk (still representative)
            np.random.seed(42)
            sample_data = np.random.choice(data, size=5000, replace=False)
            stat, p = stats.shapiro(sample_data)
            results["shapiro_stat"] = stat
            results["shapiro_p"] = p
            results["shapiro_subsampled"] = True
        else:
            stat, p = stats.shapiro(data)
            results["shapiro_stat"] = stat
            results["shapiro_p"] = p
            results["shapiro_subsampled"] = False
    except Exception as e:
        results["shapiro_stat"] = None
        results["shapiro_p"] = None
        results["shapiro_subsampled"] = False
    
    # Anderson-Darling
    try:
        result = stats.anderson(data, dist='norm')
        results["anderson_stat"] = result.statistic
        results["anderson_critical_5pct"] = result.critical_values[2]  # 5% level
    except Exception as e:
        results["anderson_stat"] = None
        results["anderson_critical_5pct"] = None
    
    # Kolmogorov-Smirnov
    try:
        stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        results["ks_stat"] = stat
        results["ks_p"] = p
    except Exception as e:
        results["ks_stat"] = None
        results["ks_p"] = None
    
    return results


def plot_gaussianity_test(
    run_dir: Path,
    cfg: Dict[str, int],
    model: FCN3NetworkActivationGeneric,
    out_path: Path,
) -> None:
    """Generate Q-Q plots and normality test results for W0 distributions."""
    try:
        W0 = model.W0.detach().cpu().numpy()  # (ens, N, d)
        
        # Two distributions to test
        w0_feature0 = W0[:, :, 0].flatten()
        w0_mean_across_d = W0[:, :, 1:].mean(axis=2).flatten() if W0.shape[2] > 1 else W0.mean(axis=2).flatten()
        
        # Run normality tests
        test_results = []
        test_results.append(test_normality(w0_feature0, "W0[:,:,0]"))
        test_results.append(test_normality(w0_mean_across_d, "W0.mean(axis=2)"))
        
        # Create figure with Q-Q plots and histograms
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        for i, (data, label) in enumerate([(w0_feature0, "W0[:, :, 0]"), 
                                            (w0_mean_across_d, "W0.mean(axis=2)")]):
            # Q-Q plot
            ax_qq = fig.add_subplot(gs[0, i])
            stats.probplot(data, dist="norm", plot=ax_qq)
            ax_qq.set_title(f"Q-Q Plot: {label}")
            ax_qq.grid(True, alpha=0.3)
            
            # Histogram with normal overlay
            ax_hist = fig.add_subplot(gs[1, i])
            n, bins, patches = ax_hist.hist(data, bins=50, density=True, alpha=0.7, 
                                           color='tab:blue', edgecolor='black')
            mu, std = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax_hist.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, 
                        label=f'N({mu:.3e}, {std:.3e})')
            ax_hist.set_xlabel('Weight value')
            ax_hist.set_ylabel('Density')
            ax_hist.set_title(f'Histogram: {label}')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
        
        # Test results table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        table_data = []
        headers = ["Distribution", "n", "Shapiro W", "Shapiro p", 
                  "Anderson A²", "A² (5%)", "K-S D", "K-S p"]
        
        for res in test_results:
            row = [
                res["label"],
                f"{res['n']:,}",
                f"{res['shapiro_stat']:.4f}" if res['shapiro_stat'] is not None else "N/A",
                f"{res['shapiro_p']:.4e}" if res['shapiro_p'] is not None else "N/A",
                f"{res['anderson_stat']:.4f}" if res['anderson_stat'] is not None else "N/A",
                f"{res['anderson_critical_5pct']:.4f}" if res['anderson_critical_5pct'] is not None else "N/A",
                f"{res['ks_stat']:.4f}" if res['ks_stat'] is not None else "N/A",
                f"{res['ks_p']:.4e}" if res['ks_p'] is not None else "N/A",
            ]
            # Add asterisk if Shapiro-Wilk used subsample
            if res.get('shapiro_subsampled', False):
                row[2] = row[2] + "*"
                row[3] = row[3] + "*"
            table_data.append(row)
        
        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add interpretation text
        interpretation = (
            "Interpretation:\n"
            "• Shapiro-Wilk: p > 0.05 → fail to reject normality (* = subsampled to n=5000)\n"
            "• Anderson-Darling: A² < critical value → consistent with normality\n"
            "• Kolmogorov-Smirnov: p > 0.05 → fail to reject normality"
        )
        fig.text(0.5, 0.02, interpretation, ha='center', fontsize=9, 
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(
            f"W0 Gaussianity Tests (d={cfg['d']}, P={cfg['P']}, N={cfg['N']}, chi={cfg['chi']})",
            fontsize=14, fontweight='bold'
        )
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved W0 Gaussianity test plot to {out_path}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"Gaussianity Test Summary for {run_dir.name}")
        print(f"{'='*60}")
        for res in test_results:
            print(f"\n{res['label']} (n={res['n']:,}):")
            if res['shapiro_p'] is not None:
                subsample_note = " (subsampled to 5000)" if res.get('shapiro_subsampled', False) else ""
                print(f"  Shapiro-Wilk{subsample_note}: W={res['shapiro_stat']:.4f}, p={res['shapiro_p']:.4e}")
            if res['anderson_stat'] is not None:
                is_normal = "✓" if res['anderson_stat'] < res['anderson_critical_5pct'] else "✗"
                print(f"  Anderson-Darling: A²={res['anderson_stat']:.4f} vs {res['anderson_critical_5pct']:.4f} (5%) {is_normal}")
            if res['ks_p'] is not None:
                print(f"  Kolmogorov-Smirnov: D={res['ks_stat']:.4f}, p={res['ks_p']:.4e}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"  Warning: failed to test W0 Gaussianity for {run_dir}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test W0 weight Gaussianity for runs.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory to search for runs (default: script directory)")
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="Optional list of d values to include")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix to filter run folders (matches *<suffix>*)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g., cuda:0 or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    base_dir = args.base_dir
    dims = args.dims

    runs = find_run_dirs(base_dir, dims=dims, suffix=args.suffix)
    if not runs:
        print("No runs found. Nothing to test.")
        return

    for run_dir, cfg in runs:
        model = load_model(run_dir, cfg, device)
        if model is None:
            print(f"Skipping {run_dir}: model not found")
            continue
        
        out_path = run_dir / "plots" / "W0_gaussianity_test.png"
        plot_gaussianity_test(run_dir, cfg, model, out_path)


if __name__ == "__main__":
    main()
