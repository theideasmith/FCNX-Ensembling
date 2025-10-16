import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


# =========================
# Data Loading
# =========================
def load_first_three_columns(csv_path):
    Ps, lK1s, lK3s = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                P = float(row[0])
                lK1 = complex(row[1]).real
                lK3 = complex(row[2]).real
            except (ValueError, IndexError):
                continue
            Ps.append(P)
            lK1s.append(lK1)
            lK3s.append(lK3)
    return np.array(Ps), np.array(lK1s), np.array(lK3s)


# =========================
# GPR Fitting
# =========================
def fit_gpr(x, y):
    """Fit a GPR model on finite (x, y) points. Return mean prediction + std function."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        mean = np.nanmean(y[mask]) if np.any(mask) else 0
        def fallback_predict(X):
            return np.full(len(X), mean), np.zeros(len(X))
        return fallback_predict

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.001, n_restarts_optimizer=10, random_state=0)
    gp.fit(x[mask].reshape(-1, 1), y[mask])
    return lambda X: gp.predict(X.reshape(-1, 1), return_std=True)


# =========================
# Plotting
# =========================
def make_arxiv_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (6.0, 4.0),
        'axes.grid': True,
        'grid.alpha': 0.25,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })


def plot_dataset(ax, Ps, lK1s, lK3s, d, color1, color3, label_prefix):
    """Plot a single dataset on an existing Axes object."""
    alphas = np.log(Ps) / np.log(d)

    # Fit GPR models
    gpr1 = fit_gpr(alphas, lK1s)
    gpr3 = fit_gpr(alphas, lK3s)

    # Prediction grid
    x_grid = np.linspace(np.min(alphas), np.max(alphas), 400)
    mu1, std1 = gpr1(x_grid)
    mu3, std3 = gpr3(x_grid)

    # Plot lines + confidence intervals
    ax.plot(x_grid, mu1, color=color1, linewidth=1.0, label=f'{label_prefix} λK¹')
    ax.fill_between(x_grid, mu1 - 2 * std1, mu1 + 2 * std1, color=color1, alpha=0.2)

    ax.plot(x_grid, mu3, color=color3, linestyle='--', linewidth=1.0, label=f'{label_prefix} λK³')
    ax.fill_between(x_grid, mu3 - 2 * std3, mu3 + 2 * std3, color=color3, alpha=0.2)

    # Scatter original points
    ax.scatter(alphas, lK1s, color=color1, s=9, alpha=0.6, linewidths=0)
    ax.scatter(alphas, lK3s, color=color3, s=9, alpha=0.6, linewidths=0)


# =========================
# Main Plot Function
# =========================
def make_overlay_plot(csv_paths, out_dir, d):
    make_arxiv_style()
    fig, ax = plt.subplots()

    # Distinguishable colors for multiple datasets
    base_colors = [
        ('#1f77b4', '#d62728'),
        ('#2ca02c', '#9467bd'),
        ('#ff7f0e', '#8c564b'),
        ('#17becf', '#e377c2'),
    ]
    prfx = ["Standard Scaling", "Mean Field Scaling"]
    for i, csv_path in enumerate(csv_paths):
        Ps, lK1s, lK3s = load_first_three_columns(csv_path)
        if len(Ps) == 0:
            print(f'⚠️ No valid data in {csv_path}, skipping')
            continue
        label_prefix = prfx[i] if i < len(prfx) else f'Dataset {i+1}'
        color1, color3 = base_colors[i % len(base_colors)]
        plot_dataset(ax, Ps, lK1s, lK3s, d, color1, color3, label_prefix)

    ax.set_title('Learnability Scaling (Overlay)')
    ax.set_xlabel(r'$\alpha \;\,(P \sim d^\alpha)$')
    ax.set_ylabel('Learnability')
    subtitle = fr'$n=1000,\; \chi=n,\; \kappa=1,\; d={d},\; \epsilon=0.03$'
    ax.text(0.5, 0.2, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
    ax.legend(frameon=False)

    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, 'learnability_overlay_MF_SSC_d_625_n3000.pdf')
    out_png = os.path.join(out_dir, 'learnability_overlay_MF_SSC_d_625_n3000.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


# =========================
# CLI Entry Point
# =========================
def main():
    parser = argparse.ArgumentParser(description='Overlay learnability plots from multiple CSV files.')
    parser.add_argument('--csv', nargs='+', required=True, help='Paths to one or more CSV files')
    parser.add_argument('--outdir', type=str, default='.', help='Output directory')
    parser.add_argument('--d', type=int, default=625, help='Value of d')
    args = parser.parse_args()

    make_overlay_plot(args.csv, args.outdir, d=args.d)


if __name__ == '__main__':
    main()
