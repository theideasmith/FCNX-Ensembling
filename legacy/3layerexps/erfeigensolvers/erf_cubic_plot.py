import os
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

def load_first_three_columns(csv_path):
    Ps = []
    lK1s = []
    lK3s = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            print(row)
            
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
    return Ps, lK1s, lK3s

def make_arxiv_plot(Ps, lK1s, lK3s, d, out_dir):
    # Transform P -> alpha = log(P/d)/log(d) + 1
    alphas = np.log(np.array(Ps)) / np.log(d)
    
    # GPR kernel
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    
    y1_arr = np.asarray(lK1s, dtype=float)
    y3_arr = np.asarray(lK3s, dtype=float)
    mask_finite1 = np.isfinite(y1_arr) & np.isfinite(alphas)
    mask_finite3 = np.isfinite(y3_arr) & np.isfinite(alphas)
    
    x_all = alphas.reshape(-1, 1)
    
    # Fit on all finite data
    if np.sum(mask_finite1) >= 2:
        gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.001, n_restarts_optimizer=10, random_state=0)
        gp1.fit(x_all[mask_finite1], y1_arr[mask_finite1])
    else:
        mean1 = np.nanmean(y1_arr[mask_finite1])
        gp1 = None  # Will handle in prediction
    
    if np.sum(mask_finite3) >= 2:
        gp3 = GaussianProcessRegressor(kernel=kernel, alpha=0.001, n_restarts_optimizer=10, random_state=0)
        gp3.fit(x_all[mask_finite3], y3_arr[mask_finite3])
    else:
        mean3 = np.nanmean(y3_arr[mask_finite3])
        gp3 = None
    
    # Build plotting grid
    alpha_min = np.min(alphas)
    alpha_max = np.max(alphas)
    x_grid = np.linspace(alpha_min, alpha_max, 400)
    x_grid_2d = x_grid.reshape(-1, 1)
    
    if gp1 is not None:
        mu1, std1 = gp1.predict(x_grid_2d, return_std=True)
    else:
        mu1 = np.full_like(x_grid, mean1)
        std1 = np.zeros_like(x_grid)
    
    if gp3 is not None:
        mu3, std3 = gp3.predict(x_grid_2d, return_std=True)
    else:
        mu3 = np.full_like(x_grid, mean3)
        std3 = np.zeros_like(x_grid)
    
    # Aesthetics for arXiv-quality
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
    fig, ax = plt.subplots()
    
    # GPR means and confidence intervals
    ax.plot(x_grid, mu1, color='#1f77b4', linewidth=1.0, label=r'$\lambda_K^{1} / (\kappa/P + \lambda_K^{1})$')
    ax.fill_between(x_grid, mu1 - 2 * std1, mu1 + 2 * std1, color='#1f77b4', alpha=0.2)
    ax.plot(x_grid, mu3, color='#d62728', linestyle='--', linewidth=1.0, label=r'$\lambda_K^{3} / (\kappa/P + \lambda_K^{3})$')
    ax.fill_between(x_grid, mu3 - 2 * std3, mu3 + 2 * std3, color='#d62728', alpha=0.2)
    
    # Scatter all finite points
    ax.scatter(alphas[mask_finite1], y1_arr[mask_finite1], color='#1f77b4', s=9, alpha=0.6, linewidths=0)
    ax.scatter(alphas[mask_finite3], y3_arr[mask_finite3], color='#d62728', s=9, alpha=0.6, linewidths=0)
    
    ax.set_title('Learnability Scaling')
    ax.set_xlabel(fr'$\alpha\;\,(P \sim d^'+'{\\alpha})$')
    ax.set_ylabel('Learnability')
    
    # Hyperparameter note beneath title (small font)
    subtitle = fr'$n=1000,\; \chi=n,\; \kappa=1,\; d=625,\; \epsilon=0.03$'
    ax.text(0.5, 0.2, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
    ax.legend(frameon=False)
    
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, 'learnability_scaling_chi_1000_d_625_epsilon_3e-2_testplot_SSC.pdf')
    out_png = os.path.join(out_dir, 'learnability_scaling_chi_1000_d_625_epsilon_3e-2_SSC.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')

def main():
    default_csv = os.path.join(os.path.dirname(__file__), 'erf_cubic_solver.csv')
    parser = argparse.ArgumentParser(description='Plot Learnability Scaling from CSV (first three rows).')
    parser.add_argument('--csv', type=str, default=default_csv, help='Path to CSV (default: solver CSV)')
    parser.add_argument('--outdir', type=str, default=os.path.dirname(default_csv), help='Output directory for figures')
    args = parser.parse_args()
    d = 625 # as specified
    Ps, lK1s, lK3s = load_first_three_columns(args.csv)
    if len(Ps) == 0:
        raise RuntimeError(f'No valid data rows found in {args.csv}')
    make_arxiv_plot(Ps, lK1s, lK3s, d=d, out_dir=args.outdir)

if __name__ == '__main__':
    main()