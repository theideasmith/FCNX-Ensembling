import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import csv
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import warnings
import multiprocessing as mp

# Suppress specific warnings globally - use category for older Python/versions
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Covers casting and overflow

# Parameters
n = 1000
chi = 1.0
kappa = 1.0
d = 625*10
b = 4 / (3 * np.pi)
epsilon = 0.03
delta = 1

# Define alphas and P_values - sorted ascending for continuation
alphas = np.sort(np.linspace(0.1, 2.0, num=40))
P_values = d ** alphas

# Global params
global_params = {'d': d, 'delta': delta, 'b': b, 'n': n, 'kappa': kappa, 'epsilon': epsilon}

def equations(vars, chi, P, params):
    d, delta, b, n, kappa, epsilon = params['d'], params['delta'], params['b'], params['n'], params['kappa'], params['epsilon']
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
    lT1 = lT1_re + 1j * lT1_im
    lV1 = lV1_re + 1j * lV1_im
    lT3 = lT3_re + 1j * lT3_im
    lV3 = lV3_re + 1j * lV3_im
    lWT = 1 / (d + delta * b * lV1 / n + 1e-12)
    lWP = 1 / d
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh + 1e-12)
    lK1 = gammaYh2 * lH1
    eq1 = lT1 - (-chi**2 / (kappa / P + lK1 + 1e-12)**2 * delta)
    eq2 = lV1 - (1 / (lJ1**2 + 1e-12) * lH1 - 1 / (lJ1 + 1e-12))
    eq3 = lH1 - (1 / (1 / (lJ1 + 1e-12) + gammaYh2 * lT1 / (n * chi + 1e-12)))
    eq4 = lJ1 - (4 / (np.pi * (1 + 2 * TrSigma + 1e-12)) * lWT)
    lK3 = gammaYh2 * lH3
    eq5 = lT3 - (-chi**2 / (kappa / P + lK3 + 1e-12)**2 * delta)
    eq6 = lV3 - (1 / (lJ3**2 + 1e-12) * lH3 - 1 / (lJ3 + 1e-12))
    eq7 = lH3 - (1 / (1 / (lJ3 + 1e-12) + gammaYh2 * lT3 * epsilon**2 / (n * chi + 1e-12)))
    eq8 = lJ3 - ((8 ) / (np.pi * (1 + 2 * TrSigma)**3 + 1e-12) * (213*lWT**3 + 9*lWT))
    return np.array([
        np.real(eq1), np.imag(eq1),
        np.real(eq2), np.imag(eq2),
        eq3,
        eq4,
        np.real(eq5), np.imag(eq5),
        np.real(eq6), np.imag(eq6),
        eq7,
        eq8
    ], dtype=np.float64)  # Force float64 output

def _solve_one_fsolve(args):
    """
    Solve for one initial condition using fsolve.
    args: (init_guess, chi_val, P, params)
    """
    init_guess, chi_val, P, params = args
    # Ensure init is float64
    init_guess = np.asarray(init_guess, dtype=np.float64)
    def residual(vars):
        # Suppress warnings inside residual too
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return equations(vars, chi_val, P, params)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sol, infodict, ier, mesg = fsolve(
                residual, init_guess, full_output=True, xtol=1e-8, maxfev=10000
            )
        if ier == 1:  # Success
            res_norm = np.linalg.norm(residual(sol))
            if res_norm < 1e-5:
                sol = np.clip(sol, -1e10, 1e10)  # Prevent inf
                sol[4] = max(sol[4], 0)
                sol[5] = max(sol[5], 0)
                sol[10] = max(sol[10], 0)
                sol[11] = max(sol[11], 0)
                return sol.astype(np.float64)
    except Exception as e:
        pass
    return None

def _compute_lK1(sol, d, delta, b, n, kappa):
    """Helper to compute lK1 from solution."""
    if sol is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
            lV1 = lV1_re + 1j * lV1_im
            lWT = 1 / (d + delta * b * lV1 / n + 1e-12)
            lWP = 1 / d
            TrSigma = lWT + lWP * (d - 1)
            EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
            gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh + 1e-12)
            lK1 = gammaYh2 * lH1
            return lK1 if lK1 > 1e-6 else None
        except:
            return None

def filter_positive_lH_lJ(solutions):
    if solutions.size == 0:
        return np.empty((0, 12), dtype=np.float64)
    mask = (solutions[:, 4] > 1e-6) & (solutions[:, 5] > 1e-6) & (solutions[:, 10] > 1e-6) & (solutions[:, 11] > 1e-6)
    return solutions[mask]

# Ensemble of 10K initial conditions - biased to small non-trivial, float64
num_samples = 10000
initial_conditions = [
    np.array([
        np.random.uniform(-0.1, 0.1),  # lT1_re
        np.random.uniform(-1e-3, 1e-3),  # lT1_im small
        np.random.uniform(0.1, 1.0),   # lV1_re
        np.random.uniform(-1e-3, 1e-3),  # lV1_im
        np.random.uniform(0.001, 0.01), # lH1
        np.random.uniform(0.001, 0.01), # lJ1
        np.random.uniform(-0.1, 0.1),  # lT3_re
        np.random.uniform(-1e-3, 1e-3),  # lT3_im
        np.random.uniform(0.1, 1.0),   # lV3_re
        np.random.uniform(-1e-3, 1e-3),  # lV3_im
        np.random.uniform(0.001, 0.01), # lH3
        np.random.uniform(0.001, 0.01)  # lJ3
    ], dtype=np.float64) for _ in range(num_samples)
]

fname = os.path.join(os.path.dirname(__file__), f'erfHe3_Scaling_d_{d}_epsilon_{epsilon}_n_{n}_chi_{chi}_fsolve_cont.csv')
state_file = fname.replace('.csv', '_state.pkl')


# Resume logic
start_idx = 0
prev_solutions = [None] * num_samples  # Previous solutions for continuation
lK1_ratios = []
lK3_ratios = []
varsolutions = []
if os.path.exists(state_file):
    with open(state_file, 'rb') as f:
        saved = pickle.load(f)
        start_idx = saved['idx']
        prev_solutions = saved['prev_sols']
        lK1_ratios = saved['lK1']
        lK3_ratios = saved['lK3']
        varsolutions = saved['varsols']
    # Ensure loaded prev_sols are list of arrays float64
    prev_solutions = [None if s is None else np.asarray(s, dtype=np.float64) for s in prev_solutions]
else:
    with open(fname, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['P', 'lK1_ratio', 'lK3_ratio'])

def main():
    # Set start method to 'spawn' to avoid forking issues with rich console
    mp.set_start_method('spawn', force=True)
    
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        for idx in range(start_idx, len(P_values)):
            P = P_values[idx]
            task_id = progress.add_task(f"P={P:.6g} (alpha={alphas[idx]:.2f}) lK1=--", total=num_samples)
            current_solutions = []
            live_lK1_eigs = []
            
            # Prepare initial guesses for this P: use prev if exists, else base
            current_ini = []
            for i in range(num_samples):
                if prev_solutions[i] is not None:
                    # Perturb slightly and ensure float64
                    init = prev_solutions[i] + np.random.uniform(-1e-4, 1e-4, 12).astype(np.float64)
                else:
                    init = initial_conditions[i].copy()
                current_ini.append(init)
            
            # Multiprocess fsolve over samples
            with ProcessPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(_solve_one_fsolve, (current_ini[i], n, P, global_params)): i for i in range(num_samples)}
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    sol = fut.result()
                    sample_idx = futures[fut]
                    if sol is not None:
                        current_solutions.append(sol)
                        prev_solutions[sample_idx] = sol
                        # Compute live lK1
                        lK1 = _compute_lK1(sol, d, delta, b, n, kappa)
                        if lK1 is not None:
                            live_lK1_eigs.append(lK1)
                    # Update progress
                    mean_eig = np.mean(live_lK1_eigs) if live_lK1_eigs else 0
                    current_ratio = mean_eig / (kappa / P + mean_eig) if mean_eig > 0 else 0
                    progress.update(task_id, completed=completed, description=f"P={P:.6g} lK1={current_ratio:.4f}")
            
            # Post-process - ensure array float64
            if current_solutions:
                solutions = np.array(current_solutions, dtype=np.float64)
            else:
                solutions = np.empty((0, 12), dtype=np.float64)
            solutions = filter_positive_lH_lJ(solutions)
            
            lK1_eigs = []
            lK3_eigs = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for sol in solutions:
                    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
                    lV1 = lV1_re + 1j * lV1_im
                    lWT = 1 / (d + delta * b * lV1 / n + 1e-12)
                    lWP = 1 / d
                    TrSigma = lWT + lWP * (d - 1)
                    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d - 1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
                    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh + 1e-12)
                    lK1 = gammaYh2 * lH1
                    lK3 = gammaYh2 * lH3
                    if lK1 > 1e-6 and lK3 > 1e-6:
                        lK1_eigs.append(lK1)
                        lK3_eigs.append(lK3)
            
            mean_lK1 = np.mean(lK1_eigs) if lK1_eigs else float('nan')
            mean_lK3 = np.mean(lK3_eigs) if lK3_eigs else float('nan')
            mean_ratio1 = mean_lK1 / (kappa / P + mean_lK1) if np.isfinite(mean_lK1) and mean_lK1 > 0 else float('nan')
            mean_ratio3 = mean_lK3 / (kappa / P + mean_lK3) if np.isfinite(mean_lK3) and mean_lK3 > 0 else float('nan')
            
            varsolutions.append(solutions.mean(axis=0).tolist() if len(solutions) > 0 else [float('nan')]*12)
            lK1_ratios.append(mean_ratio1)
            lK3_ratios.append(mean_ratio3)
            
            # Save incrementally
            with open(fname, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([P, mean_ratio1, mean_ratio3])
                csvfile.flush()
            
            # Save state
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'idx': idx + 1,
                    'prev_sols': prev_solutions,
                    'lK1': lK1_ratios,
                    'lK3': lK3_ratios,
                    'varsols': varsolutions
                }, f)

if __name__ == '__main__':
    main()