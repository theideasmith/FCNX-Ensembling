import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import csv
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameters
n = 1000
chi = 1.0
kappa = 1.0
d = 25
b = 4 / (3 * np.pi)
epsilon = 0.03
delta = 1

# Define P values: P = d^(1/2), d, d^(3/2), d^2
P_labels = [r'$d^{1/2}$', r'$d$', r'$d^{3/2}$', r'$d^2$']

# Define alphas and P_values
alphas = np.linspace(0.1, 2.0, num=40)
P_values = d ** alphas

def _solve_one_init_anneal(args, collect=False):
    i0, d, P, chif, N = args
    chi0 = 1e-8

    # Initialize the solution list with the initial condition
    current_solutions = [(chi0, i0)]
    t=0

    if chi0 == chif:
        sol = _solve_one_init((i0, d, P, chi0, N))
        if collect:
            return current_solutions+ [(chi0, sol)]
        return sol
    
    while True:
        # Compute chi_t
        chi_t =chif + (chi0 - chif) * np.exp(- 5 * t / 100)
        # print(f'chi_t: {chi_t:.8f}')

        # Get the previous solution
        _, previous_solution = current_solutions[-1]

        args = (previous_solution, d, P, chi_t, N)

        # Solve for the current chi_t
        sol = _solve_one_init(args)
        if sol is None:
            return None

        # Append the new solution
        current_solutions.append((chi_t, sol))

        # Check termination condition
        if chi_t >= chif:
            if collect:
                return current_solutions
            return current_solutions[-1][1]

        # Increment t
        t += 1

def filter_positive_lH_lJ(solutions):
    """
        Filter solutions to keep only those where lH1, lJ1, lH3, lJ3 are positive.
        Parameters:
            solutions (np.ndarray): Array of solutions with shape (n_solutions, 12)
                                  where columns are [lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3]
        Returns:
            np.ndarray: Filtered solutions where lH1, lJ1, lH3, lJ3 > 0
    """
    if len(solutions)==0:
        return solutions*0
    return solutions[(solutions[:, 4] >= 0) & (solutions[:, 5] >= 0) & (solutions[:, 10] >= 0) & (solutions[:, 11] >= 0)]

# Define the system of equations with real and imaginary parts separated
# Here we assume that the weights have been learned to be centered with zero mean. 
def equations(vars, chi, P):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
    lT1 = lT1_re + 1j * lT1_im
    lV1 = lV1_re + 1j * lV1_im
    lT3 = lT3_re + 1j * lT3_im
    lV3 = lV3_re + 1j * lV3_im
    lWT = 1 / (d + delta * b * lV1 / n)
    lWP = (1/d)
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + (4 / (np.pi * (1 + 2 * TrSigma)) * (15*lWT**3 - 18*lWT**2 + 10*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    eq1 = lT1 - (-chi**2 / (kappa / P + lK1)**2 * delta)
    eq2 = lV1 - (1 / lJ1**2 * lH1 - 1 / lJ1)
    eq3 = lH1 - (1 / (1 / lJ1 + gammaYh2 * lT1 / (n * chi)))
    eq4 = lJ1 - (4 / (np.pi * (1 + 2 * TrSigma)) * lWT)
    lK3 = gammaYh2 * lH3
    eq5 = lT3 - (-chi**2 / (kappa / P + lK3)**2 * delta)
    eq6 = lV3 - (1 / lJ3**2 * lH3 - 1 / lJ3)
    eq7 = lH3 - (1 / (lJ3**(-1) + gammaYh2 * lT3 * epsilon**2 / (n * chi)))
    eq8 = lJ3 - ((4 ) / (np.pi * (1 + 2 * TrSigma)) * (15*lWT**3 - 18*lWT**2 + 10*lWT))
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

# Parallel worker: solve one initial condition
def _solve_one_init(args):
    i0, d, P, chi0, N = args
    try:
        sol, infodict, ier, mesg = fsolve(lambda vars: equations(vars, chi0, P), i0, full_output=True, xtol=1e-8)
        if ier == 1:
            residuals = np.abs(equations(sol, chi0, P))
            if np.max(residuals) < 0.01:
                return sol
    except Exception:
        pass
    return None

# Ensemble of initial conditions
num_samples = 10000
initial_conditions = [
    (
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
    ) for _ in range(num_samples)
]

fname = os.path.join(os.path.dirname(__file__), f'erfHe3_Scaling_epsilon_{epsilon}_n_{n}_chi_{chi}_d_{d}.csv')
print(fname)

# Store ratios for plotting
lK1_ratios = []
lK3_ratios = []
varsolutions = []
valid_initial_conditions = []

with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress, open(fname, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['P', 'lK1_ratio', 'lK3_ratio']) # Header row
    for P in P_values:
        # Solve for each initial condition with progress (parallel)
        task_id = progress.add_task(f"P={P:.6g} lK1=--", total=len(initial_conditions))
        solutions = []
        live_lK1_eigs = []
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(_solve_one_init, (init, d, P, n, n)) for init in initial_conditions]
            for fut in as_completed(futures):
                sol = fut.result()
                if sol is not None:
                    solutions.append(sol)
                    try:
                        lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
                        lV1 = lV1_re + 1j * lV1_im
                        lWT = 1 / (d + delta * b * lV1 / n)
                        lWP = (1/d)
                        TrSigma = lWT + lWP * (d - 1)
                        EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
                        gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
                        lK1 = gammaYh2 * lH1
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
        solutions = filter_positive_lH_lJ(solutions)
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
            # print(f"Solution lK1: {lK1}, lK3: {lK3}")
            # print(f"Solution lH1: {lH1}, lJ1: {lJ1}, lH3: {lH3}, lJ3: {lJ3}")

        mean_lK1 = np.mean(lK1_eigs) if len(lK1_eigs) > 0 else float('nan')
        mean_lK3 = np.mean(lK3_eigs) if len(lK3_eigs) > 0 else float('nan')
        # Printing stddev and mean of lK1 and lK3
        std_lK1 = np.std(lK1_eigs) if len(lK1_eigs) > 0 else float('nan')
        std_lK3 = np.std(lK3_eigs) if len(lK3_eigs) > 0 else float('nan')
        # print(f"P={P:.6g}: mean_lK1={mean_lK1:.6g} (std={std_lK1:.6g}), mean_lK3={mean_lK3:.6g} (std={std_lK3:.6g}), num_solutions={len(solutions)}")
        # Compute ratios
        mean_ratio1 = mean_lK1 / (kappa / P + mean_lK1) if np.isfinite(mean_lK1) else float('nan')
        mean_ratio3 = mean_lK3 / (kappa / P + mean_lK3) if np.isfinite(mean_lK3) else float('nan')
        varsolutions.append(np.array(solutions).mean(axis=0).tolist())
        lK1_ratios.append(mean_ratio1)
        lK3_ratios.append(mean_ratio3)

        csv_writer.writerow([P, mean_ratio1, mean_ratio3])