import numpy as np
import os
import csv
import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
class Config:
    # --- Physical Parameters from your script ---
    N_SAMPLES = 250.0
    D = 60.0
    KAPPA = 1.0
    B_CONST = 4 / (3 * np.pi)
    EPSILON = 0.03
    DELTA = 1.0
    
    # --- Solver/Ensemble Parameters ---
    NUM_INITIAL_CONDITIONS = 30000 # The number of random starts for each P
    MAX_WORKERS = 22
    SOLVER_TOLERANCE = 1e-8
    RESIDUAL_THRESHOLD = 0.1

    # --- Scaling Law Scan Parameters ---
    ALPHA_RANGE = np.linspace(0.1, 4, 10)
    BETA_RANGE = np.linspace(0.5, 2.0, 10)
    
    # --- Output Directory ---
    OUTPUT_DIR = "fsolve_random_init_results"

# =============================================================================
# 2. PARALLEL WORKER (must be top-level for multiprocessing)
# =============================================================================
def _solve_worker(args):
    """
    Worker function for a single fsolve call, designed for ProcessPoolExecutor.
    It is self-contained and takes all necessary parameters.
    """
    init_cond, P, config_dict, chi = args
    try:
        # Reconstruct the equations function locally to ensure it's self-contained
        def equations_local(vars, P):
            lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
            lT1=lT1_re+1j*lT1_im; lV1=lV1_re+1j*lV1_im; lT3=lT3_re+1j*lT3_im; lV3=lV3_re+1j*lV3_im
            d, delta, b, n, kappa, epsilon = config_dict['D'], config_dict['DELTA'], config_dict['B_CONST'], config_dict['N_SAMPLES'], config_dict['KAPPA'], config_dict['EPSILON']
            with np.errstate(divide='ignore', invalid='ignore'):
                lWT = 1/(d+delta*b*lV1/n); lWP = 1/d; TrSigma = lWT + lWP*(d-1); term1_Tr = 1+2*TrSigma
                if np.real(term1_Tr)<0: term1_Tr=np.nan
                EChh_term_J3 = (8/np.pi * (1/term1_Tr)**3 * (213*lWT**3 + 9*lWT)); EChh_term_J1 = (4/np.pi*(1/term1_Tr)*lWT)
                EChh = lH1+lH3 + EChh_term_J3*(d-1) + EChh_term_J1*(d-1); gammaYh2 = (4/np.pi)*1/(1+2*EChh)
            lK1=gammaYh2*lH1; lK3=gammaYh2*lH3
            eq1=lT1-(-chi**2/(kappa/P+lK1)**2*delta); eq2=lV1-(1/lJ1**2*lH1-1/lJ1)
            eq3=lH1-(1/(1/lJ1+gammaYh2*lT1/(n*chi))); eq4=lJ1-EChh_term_J1
            eq5=lT3-(-chi**2/(kappa/P+lK3)**2*delta); eq6=lV3-(1/lJ3**2*lH3-1/lJ3)
            eq7=lH3-(1/(1/lJ3+gammaYh2*lT3*epsilon**2/(n*chi))); eq8=lJ3-EChh_term_J3
            flat_eqs = np.array([np.real(e) for e in [eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8]] + [np.imag(e) for e in [eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8]])
            return np.nan_to_num(flat_eqs.flatten()[:12], nan=1e12)

        sol, _, ier, _ = fsolve(equations_local, init_cond, args=(P,), full_output=True, xtol=config_dict['SOLVER_TOLERANCE'])
        if ier == 1 and np.max(np.abs(equations_local(sol, P))) < config_dict['RESIDUAL_THRESHOLD']:
            return sol
    except Exception: pass
    return None

# =============================================================================
# 3. SIMULATION RUNNER CLASS
# =============================================================================
class SimulationRunner:
    def __init__(self, config, chi):
        self.config = config
        self.chi = chi
        self.config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__')}

    def _generate_initial_conditions(self, num):
        """Generates a diverse set of random starting points, as per your code."""
        return [
            (np.random.uniform(-1e8,1e8), np.random.uniform(-1e8,1e8), np.random.uniform(-1e6,1e6), 
             np.random.uniform(-1e6,1e6), np.random.uniform(0,0.05), np.random.uniform(0,0.05),
             np.random.uniform(-1e8,1e8), np.random.uniform(-1e8,1e8), np.random.uniform(-1e6,1e6), 
             np.random.uniform(-1e6,1e6), np.random.uniform(0,0.05), np.random.uniform(0,0.05))
            for _ in range(num)
        ]

    def run_simulation(self, P_values):
        all_results = []

        for P in P_values:
            # Generate fresh random initial conditions for every P value
            initial_conditions = self._generate_initial_conditions(self.config.NUM_INITIAL_CONDITIONS)
            
            with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()) as progress:
                task_id = progress.add_task(f"Solving for P={P:.3e}", total=len(initial_conditions))
                solutions = []
                
                worker_args = [(init, P, self.config_dict, self.chi) for init in initial_conditions]

                with ProcessPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                    for sol in executor.map(_solve_worker, worker_args):
                        if sol is not None:
                            solutions.append(sol)
                        progress.update(task_id, advance=1)

            solutions = np.array(solutions)
            print(solutions)
            filtered_sols = solutions[(solutions[:, 4] >= 0) & (solutions[:, 5] >= 0) & (solutions[:, 10] >= 0) & (solutions[:, 11] >= 0)] if solutions.shape[0] > 0 else np.array([])
            
            if filtered_sols.shape[0] > 0:
                lH1, lH3 = filtered_sols[:, 4], filtered_sols[:, 10]
                lV1_re, lV1_im = filtered_sols[:, 2], filtered_sols[:, 3]
                lV1 = lV1_re + 1j * lV1_im
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    lWT = 1/(self.config.D + self.config.DELTA * self.config.B_CONST * lV1 / self.config.N_SAMPLES)
                    EChh = lH1 + lH3 # Simplified for ratio calculation, full expr is complex and computed inside solver
                    gammaYh2 = (4/np.pi) * 1/(1+2*EChh)
                lK1, lK3 = gammaYh2*lH1, gammaYh2*lH3
                
                ratio1 = np.nanmean(np.real(lK1 / (self.config.KAPPA/P + lK1)))
                ratio3 = np.nanmean(np.real(lK3 / (self.config.KAPPA/P + lK3)))
            else:
                ratio1, ratio3 = float('nan'), float('nan')
                
            all_results.append({'P': P, 'avg_learnability1': ratio1, 'avg_learnability3': ratio3})
        return all_results

# =============================================================================
# 4. PLOTTING AND DATA HANDLING
# =============================================================================
def write_results_to_csv(results, filename, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['P', 'avg_learnability1', 'avg_learnability3'])
        for res in results:
            if res: writer.writerow([res['P'], res['avg_learnability1'], res['avg_learnability3']])
    print(f"Results saved to {filepath}")

def make_scaling_plot(csv_path, d, n_val, chi_str, scaling_type, out_dir):
    Ps, lK1s, lK3s = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f); next(reader, None)
        for row in reader:
            try:
                Ps.append(float(row[0])); lK1s.append(float(row[1])); lK3s.append(float(row[2]))
            except (ValueError, IndexError): continue
    if not Ps: print(f"No valid data to plot from {csv_path}"); return

    if scaling_type == 'alpha':
        x_values = np.array(Ps) / d; xlabel = fr'$\alpha \quad (P = d \cdot \alpha)$'
    else: # beta
        x_values = np.log(np.array(Ps)) / np.log(d); xlabel = fr'$\beta \quad (P = d^\beta)$'

    y1_arr, y3_arr = np.asarray(lK1s), np.asarray(lK3s)
    mask1 = np.isfinite(y1_arr) & np.isfinite(x_values); mask3 = np.isfinite(y3_arr) & np.isfinite(x_values)
    
    if not np.any(mask1) and not np.any(mask3):
        print(f"ERROR: No finite data points found in {csv_path} to plot. Skipping plot.")
        return

    x_all = x_values.reshape(-1, 1)
    kernel = C(1.0) * Matern(length_scale=0.5, nu=2.5)
    x_grid = np.linspace(np.min(x_values[mask1|mask3]), np.max(x_values[mask1|mask3]), 400).reshape(-1, 1)

    plt.style.use('seaborn-v0_8-paper'); fig, ax = plt.subplots(figsize=(6, 4))

    if np.sum(mask1) > 1:
        gp1 = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10).fit(x_all[mask1], y1_arr[mask1])
        mu1, std1 = gp1.predict(x_grid, return_std=True)
        ax.plot(x_grid, mu1, color='#1f77b4', lw=1.5, label=r'Learnability ($\phi_1$)')
        ax.fill_between(x_grid.ravel(), mu1-2*std1, mu1+2*std1, color='#1f77b4', alpha=0.2)
    ax.scatter(x_values[mask1], y1_arr[mask1], color='#1f77b4', s=15, alpha=0.7, zorder=5)
    
    if np.sum(mask3) > 1:
        gp3 = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=10).fit(x_all[mask3], y3_arr[mask3])
        mu3, std3 = gp3.predict(x_grid, return_std=True)
        ax.plot(x_grid, mu3, color='#d62728', ls='--', lw=1.5, label=r'Learnability ($\phi_3$)')
        ax.fill_between(x_grid.ravel(), mu3-2*std3, mu3+2*std3, color='#d62728', alpha=0.2)
    ax.scatter(x_values[mask3], y3_arr[mask3], color='#d62728', s=15, alpha=0.7, zorder=5)

    ax.set_title(f'Learnability Scaling for $\chi = {chi_str}$'); ax.set_xlabel(xlabel)
    ax.set_ylabel('Average Learnability')
    subtitle = f'$n={n_val},\\; \\kappa={Config.KAPPA},\\; d={d},\\; \\epsilon={Config.EPSILON}$'
    ax.text(0.5, -0.2, subtitle, transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.legend(frameon=False); ax.grid(True, alpha=0.3)
    
    filename = f"learnability_{scaling_type}_scaling_chi_{chi_str.replace('/', '_')}.pdf"
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight'); plt.close(fig)
    print(f"Plot saved to {os.path.join(out_dir, filename)}")


# =============================================================================
# 5. MAIN ORCHESTRATOR
# =============================================================================
def main():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    chi_configs = [
        {'name': 'n', 'value': config.N_SAMPLES},
        {'name': 'n_div_10', 'value': config.N_SAMPLES / 10}
    ]
    scaling_configs = [
        {'type': 'alpha', 'values': config.ALPHA_RANGE * config.D},
        {'type': 'beta', 'values': config.D ** config.BETA_RANGE}
    ]

    for chi_conf in chi_configs:
        chi_name, chi_val = chi_conf['name'], chi_conf['value']
        print(f"\n{'='*25} STARTING RUNS FOR CHI = {chi_name} {'='*25}")
        runner = SimulationRunner(config, chi_val)
        for scale_conf in scaling_configs:
            scale_type, P_values = scale_conf['type'], scale_conf['values']
            print(f"\n--- Starting {scale_type} scaling run ---")
            results = runner.run_simulation(P_values)
            csv_filename = f"results_chi_{chi_name}_{scale_type}_scaling.csv"
            write_results_to_csv(results, csv_filename, config.OUTPUT_DIR)
            csv_filepath = os.path.join(config.OUTPUT_DIR, csv_filename)
            make_scaling_plot(csv_filepath, config.D, int(config.N_SAMPLES), chi_name, scale_type, config.OUTPUT_DIR)

    print("\nAll simulations and plotting tasks are complete.")

if __name__ == '__main__':
    main()