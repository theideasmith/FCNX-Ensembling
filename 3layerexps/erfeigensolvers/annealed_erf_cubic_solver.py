import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import csv
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse as ap
import math

parser = ap.ArgumentParser()
# Add arguments for all parameters
parser.add_argument('--n', type=int, default=1000, help='Number of iterations')
parser.add_argument('--chi', '-c', type=float, default=None, help='Max weight variance scaling factor χ to use in the annealler')
parser.add_argument('--kappa', type=float, default=1.0, help='Kappa parameter')
parser.add_argument('--d', type=int, default=625, help='Dimension parameter')
parser.add_argument('--b', type=float, default=4/(3*np.pi), help='B parameter')
parser.add_argument('--epsilon', type=float, default=0.03, help='Epsilon parameter')
parser.add_argument('--delta', type=int, default=1, help='Delta parameter')

args = parser.parse_args()

# Set variables to parsed values
n = args.n
chi = args.n if args.chi is None else args.chi
kappa = args.kappa
d = args.d
b = args.b
epsilon = args.epsilon
delta = args.delta
# Define P values: P = d^(1/2), d, d^(3/2), d^2
P_labels = [r'$d^{1/2}$', r'$d$', r'$d^{3/2}$', r'$d^2$']
import torch
import numpy as np
from scipy.optimize import fsolve


# Set variables to parsed values
n = 1000
chi = n
kappa = 1.0
d = 250
b = 4 / ( 3* np.pi)
epsilon = 0.03
delta = 1.0

def equations(vars, P, χ, N):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
    lT1 = lT1_re + 1j * lT1_im
    lV1 = lV1_re + 1j * lV1_im
    lT3 = lT3_re + 1j * lT3_im
    lV3 = lV3_re + 1j * lV3_im
    lWT = 1 / (d + delta * b * lV1 / N)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    eq1 = lT1 - (-χ**2 / (kappa / P + lK1)**2 * delta)
    eq2 = lV1 - (1 / lJ1**2 * lH1 - 1 / lJ1)
    eq3 = lH1 - (1 / (1 / lJ1 + gammaYh2 * lT1 / (N * χ)))
    eq4 = lJ1 - (4 / (np.pi * (1 + 2 * TrSigma)) * lWT)
    lK3 = gammaYh2 * lH3
    eq5 = lT3 - (-χ**2 / (kappa / P + lK3)**2 * delta)
    eq6 = lV3 - (1 / lJ3**2 * lH3 - 1 / lJ3)
    eq7 = lH3 - (1 / (lJ3**(-1) + gammaYh2 * lT3 * epsilon**2 / (N * χ)))
    eq8 = lJ3 - ((8 ) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT))
    return [
        np.real(eq1), np.imag(eq1),
        np.real(eq2), np.imag(eq2),
        np.real(eq3),
        np.real(eq4),
        np.real(eq5), np.imag(eq5),
        np.real(eq6), np.imag(eq6),
        np.real(eq7),
        np.real(eq8)
    ]

def compute_lK1(sol):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
    lV1 = lV1_re + 1j * lV1_im
    lWT = 1 / (d + delta * b * lV1 / n)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1

    return lK1

# Parallel worker: solve one initial condition
def _solve_one_init(args):
    init, P, χ, N = args
    try:
        sol, infodict, ier, mesg = fsolve(lambda vars: equations(vars, P, χ, N), init, full_output=True, xtol=1e-8)
        if ier == 1:
            residuals = np.abs(equations(sol, P, χ, N))
            converged = np.max(residuals) < 0.03
            positive_definite = (sol[4] >= 0) & (sol[5] >= 0) & (sol[10] >= 0) & (sol[11] >= 0)
            if converged and positive_definite:
                return sol
    except Exception as e:
        pass
    return None


def rand_initial():
     return (
            np.random.uniform(-1e3, 1e3), # lT1_re
            np.random.uniform(-1e3, 1e3), # lT1_im
            np.random.uniform(-1000, 1000), # lV1_re
            np.random.uniform(-1000, 1000), # lV1_im
            np.random.uniform(0, 0.05), # lH1
            np.random.uniform(0, 0.05), # lJ1
            np.random.uniform(-1e3, 1e3), # lT3_re
            np.random.uniform(-1e3, 1e3), # lT3_im
            np.random.uniform(-100, 100), # lV3_re
            np.random.uniform(-100, 100), # lV3_im
            np.random.uniform(0, 0.05), # lH3
            np.random.uniform(0, 0.05) # lJ3
        )
        
def _anneal(current_solutions, P, chi, N, chi_f, multiplier = 1.2):
    _, previous_solution = current_solutions[-1]
    ags = (previous_solution, ) + (P,chi,N)
    sol = _solve_one_init(ags)
    if sol is None:
        return current_solutions
    # Acyclic graph dependency chain
    if chi >= chi_f:
        return current_solutions + [(chi, sol)]
    else:
        return _anneal(current_solutions + [(chi, sol)], P, chi * multiplier, N, chi_f, multiplier = multiplier)

def anneal(t0, P, chi, N):
    return _anneal([(chi, t0)], P, 1, N, chi)

def get_lKs(sol):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
    lV1 = lV1_re + 1j * lV1_im
    lWT = 1 / (d + delta * b * lV1 / n)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3
    return np.real(lK1), np.imag(lK3)

def gen_initial_conditions(num_samples = 10000):
    initial_conditions = [rand_initial() for _ in range(num_samples)]
    return initial_conditions

fname = os.path.join(os.path.dirname(__file__), f'erfHe3_Scaling_epsilon_{epsilon}_n_{n}_annealling.csv')

if __name__ == '__main__':
    # Store ratios for plotting
    lK1_ratios = []
    lK3_ratios = []
    varsolutions = []
    valid_initial_conditions = []
    initial_conditions = gen_initial_conditions()
    chi_schedule = [math.floor(chi  / n) for n in range(10)] + [1, 5, 10, 30]
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress, open(fname, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['chi', 'P', 'lK1_ratio', 'lK3_ratio']) # Header row
        # Define alphas and P_values
        alphas = np.linspace(1.2, 2.0, num=20)
        P_values = d ** alphas
        for P in P_values:
            # Solve for each initial condition with progress (parallel)
            task_id = progress.add_task(f"P={P:.6g} lK1=--", total=len(initial_conditions))
            solutions = []
            live_lK1_eigs = []
            with ProcessPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(_solve_one_init, (init, P)) for init in initial_conditions]
                for fut in as_completed(futures):
                    sol = fut.result()
                    if sol is None:
                        continue 
                    solutions.append(sol)
                    try:
                        lK1 = compute_lK1(sol)
                        live_lK1_eigs.append(lK1)
                    except Exception:
                        pass
                    # Compute current description
                    if len(live_lK1_eigs) > 0:
                        mean_eig = np.mean(live_lK1_eigs)
                        current_ratio = mean_eig / (kappa / P + mean_eig)
                        desc = f"P={P:.6g} lK1={current_ratio:.4f}"
                    else:
                        desc = f"P={P:.6g} lK1=--"
                    progress.update(task_id, advance=1, description=desc)
            # Convert solutions to numpy array and filter
            solutions = np.array(solutions)
            lK1_eigs = []
            lK3_eigs = []

            for sol in solutions:
                lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
                lV1 = lV1_re + 1j * lV1_im
                lWT = 1 / (d + delta * b * lV1 / n)
                lWP = (1/d)
                TrSigma = lWT + lWP * (d - 1)
                EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
                gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
                lK1 = gammaYh2 * lH1
                lK3 = gammaYh2 * lH3
                lK1_eigs.append(lK1)
                lK3_eigs.append(lK3)

            mean_lK1 = np.mean(lK1_eigs) if len(lK1_eigs) > 0 else float('nan')
            mean_lK3 = np.mean(lK3_eigs) if len(lK3_eigs) > 0 else float('nan')
            mean_ratio1 = mean_lK1 / (kappa / P + mean_lK1) if np.isfinite(mean_lK1) else float('nan')
            mean_ratio3 = mean_lK3 / (kappa / P + mean_lK3) if np.isfinite(mean_lK3) else float('nan')
            varsolutions.append(np.array(solutions).mean(axis=0).tolist())
            lK1_ratios.append(mean_ratio1)
            lK3_ratios.append(mean_ratio3)

            csv_writer.writerow([P, mean_ratio1, mean_ratio3])