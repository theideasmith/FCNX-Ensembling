import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import csv
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# Scikit-learn for the GPR model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# ==============================================================================
# PART 1: GPR SURROGATE MODEL FOR EIGENVALUES (lJ1, lJ3)
# This section contains the adapted DIAGONALIZER logic.
# ==============================================================================

# --- Core components for eigenvalue calculation ---
basis_degrees = [1, 3]
He = {1: lambda z: z, 3: lambda z: z**3 - 3*z}

def get_projection_constant(n):
    if n % 2 == 0: return 0.0
    return (2 / np.sqrt(np.pi)) / np.sqrt(math.factorial(n))

C_n_values = {n: get_projection_constant(n) for n in basis_degrees}

def compute_eigenvalues_worker(lV_real, d, n_param, delta_param, b_const):
    """
    Worker function to compute eigenvalues for a single real lV value.
    This is the core of the DIAGONALIZER, adapted with the SOLVER's parameterization.
    """
    # CRITICAL: Using the precise parameterization from the SOLVER
    a_cov = 1.0 / d
    # Note the variable names: b_const is the scalar from the solver's parameters,
    # while b_cov is the actual covariance parameter we are calculating.
    denominator = d + delta_param * b_const * lV_real / n_param
    if denominator == 0: return (np.nan, np.nan)
    b_cov = 1.0 / denominator

    variance_z = a_cov + b_cov
    if variance_z <= 0: return (np.nan, np.nan)
    std_dev_z = np.sqrt(variance_z)
    
    M = np.zeros((len(basis_degrees), len(basis_degrees)))
    
    for i, n_deg in enumerate(basis_degrees):
        for j, m_deg in enumerate(basis_degrees):
            Cn = C_n_values[n_deg]
            Cm = C_n_values[m_deg]
            
            def integrand(z):
                pdf_z = norm.pdf(z, loc=0, scale=std_dev_z)
                return He[n_deg](z) * He[m_deg](z) * pdf_z
            
            integral_val, _ = quad(integrand, -np.inf, np.inf, limit=100)
            M[i, j] = Cn * Cm * integral_val
            
    eigenvalues = np.linalg.eigh(M)[0]
    # The eigenvalues correspond to lJ1 and lJ3. We assume lJ1 is the larger one.
    return (eigenvalues[1], eigenvalues[0]) # Return as (lJ1, lJ3)

def train_gpr_model(d, n, delta, b_const):
    """
    Generates training data by scanning lV and trains a GPR model.
    """
    print("--- Part 1: Training GPR Surrogate Model for lJ1 and lJ3 ---")
    
    # Define the range of real lV values to scan for training data
    lV_train_values = np.logspace(-6, 8, 200)

    print(f"Generating training data for {len(lV_train_values)} points...")
    
    worker_func = partial(compute_eigenvalues_worker, d=d, n_param=n, delta_param=delta, b_const=b_const)
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(worker_func, lV_train_values), total=len(lV_train_values)))

    # Filter out any failed computations (NaNs)
    valid_indices = ~np.isnan(results).any(axis=1)
    X_train = lV_train_values[valid_indices].reshape(-1, 1)
    y_train = np.array(results)[valid_indices]

    print("Training GPR model...")
    # Define a GPR kernel. RBF for smoothness, WhiteKernel for noise.
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    
    # Fit the model to the data
    gpr.fit(X_train, y_train)
    
    print("GPR model training complete.")

    # --- Optional: Visualize the GPR fit ---
    X_plot = np.logspace(-6, 8, 500).reshape(-1, 1)
    y_pred, sigma = gpr.predict(X_plot, return_std=True)
    
    plt.figure(figsize=(12, 7))
    plt.plot(X_train, y_train[:, 0], 'r.', markersize=10, label='lJ1 Training Data')
    plt.plot(X_train, y_train[:, 1], 'b.', markersize=10, label='lJ3 Training Data')
    plt.plot(X_plot, y_pred[:, 0], 'r-', label='GPR Prediction for lJ1')
    plt.plot(X_plot, y_pred[:, 1], 'b-', label='GPR Prediction for lJ3')
    plt.xscale('log')
    plt.xlabel('lV_real')
    plt.ylabel('Eigenvalue (lJ)')
    plt.title('GPR Surrogate Model Fit')
    plt.legend()
    plt.grid(True)
    plt.show()

    return gpr

# ==============================================================================
# PART 2: SELF-CONSISTENT SOLVER USING THE GPR MODEL
# This section contains the adapted SOLVER logic.
# ==============================================================================

# --- Solver Parameters (as provided) ---
n = 1000
chi = 1.0
kappa = 1.0
d = 625
b_const = 4 / (3 * np.pi) # Renamed to avoid confusion with covariance param
epsilon = 0.03
delta = 1

# Define P values
alphas = np.linspace(0.1, 5, num=6) # Modified to avoid d^1/2 < 625
P_values = d * alphas

def filter_positive_lH_lJ(solutions):
    if len(solutions) == 0: return np.array([])
    return solutions[(solutions[:, 4] >= 0) & (solutions[:, 5] >= 0) & (solutions[:, 10] >= 0) & (solutions[:, 11] >= 0)]

def equations(vars, P, gpr_model):
    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = vars
    lT1 = lT1_re + 1j * lT1_im
    lV1 = lV1_re + 1j * lV1_im
    lT3 = lT3_re + 1j * lT3_im
    lV3 = lV3_re + 1j * lV3_im

    # --- MODIFICATION: GPR model is used here ---
    # We use the real part of lV1 as input to the GPR model.
    # The model predicts the pair (lJ1, lJ3).
    predicted_lJs = gpr_model.predict(np.array([[lV1_re]]))[0]
    lJ1_pred, lJ3_pred = predicted_lJs[0], predicted_lJs[1]
    # ---------------------------------------------

    lWT = 1 / (d + delta * b_const * lV1 / n)
    lWP = 1/d
    TrSigma = lWT + lWP * (d - 1)
    EChh = lH1 + lH3 + ((8) / (np.pi * (1 + 2 * TrSigma)**3) * (213*lWT**3 + 9*lWT)) * (d-1) + (4 / (np.pi * (1 + 2 * TrSigma)) * lWT) * (d - 1)
    gammaYh2 = (4 / np.pi) * 1 / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    eq1 = lT1 - (-chi**2 / (kappa / P + lK1)**2 * delta)
    eq2 = lV1 - (1 / lJ1**2 * lH1 - 1 / lJ1)
    eq3 = lH1 - (1 / (1 / lJ1 + gammaYh2 * lT1 / (n * chi)))
    
    # --- MODIFICATION: Replace explicit equations with GPR error ---
    eq4 = lJ1 - lJ1_pred  # Error between current lJ1 and GPR prediction
    
    lK3 = gammaYh2 * lH3
    eq5 = lT3 - (-chi**2 / (kappa / P + lK3)**2 * delta)
    eq6 = lV3 - (1 / lJ3**2 * lH3 - 1 / lJ3)
    eq7 = lH3 - (1 / (lJ3**(-1) + gammaYh2 * lT3 * epsilon**2 / (n * chi)))
    
    eq8 = lJ3 - lJ3_pred # Error between current lJ3 and GPR prediction
    # -------------------------------------------------------------
    
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

def _solve_one_init(args):
    init, P, gpr_model = args
    try:
        sol, infodict, ier, mesg = fsolve(lambda vars: equations(vars, P, gpr_model), init, full_output=True, xtol=1e-8)
        if ier == 1:
            residuals = np.abs(equations(sol, P, gpr_model))
            if np.max(residuals) < 0.01:
                return sol
    except Exception:
        pass
    return None

def run_solver(gpr_model):
    print("\n--- Part 2: Running Self-Consistent Solver with GPR Model ---")
    num_samples = 10000
    initial_conditions = [(np.random.uniform(-1e3, 1e3), np.random.uniform(-1e3, 1e3), np.random.uniform(-1000, 1000), np.random.uniform(-1000, 1000), np.random.uniform(0, 0.05), np.random.uniform(0, 0.05), np.random.uniform(-1e3, 1e3), np.random.uniform(-1e3, 1e3), np.random.uniform(-100, 100), np.random.uniform(-100, 100), np.random.uniform(0, 0.05), np.random.uniform(0, 0.05)) for _ in range(num_samples)]
    
    fname = 'erf_cubic_solver_gpr.csv'
    print(f"Saving results to: {fname}")
    
    lK1_ratios, lK3_ratios = [], []

    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as progress, open(fname, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['P', 'lK1_ratio', 'lK3_ratio'])
        for P in P_values:
            task_id = progress.add_task(f"P={P:.6g}", total=num_samples)
            solutions = []
            
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(_solve_one_init, (init, P, gpr_model)) for init in initial_conditions]
                for fut in as_completed(futures):
                    sol = fut.result()
                    if sol is not None:
                        solutions.append(sol)
                    progress.update(task_id, advance=1)
            
            solutions = np.array(solutions)
            solutions = filter_positive_lH_lJ(solutions)
            
            lK1_eigs, lK3_eigs = [], []
            if len(solutions) > 0:
                for sol in solutions:
                    lT1_re, lT1_im, lV1_re, lV1_im, lH1, lJ1, lT3_re, lT3_im, lV3_re, lV3_im, lH3, lJ3 = sol
                    lV1 = lV1_re + 1j * lV1_im
                    lWT = 1 / (d + delta * b_const * lV1 / n)
                    lWP = 1/d
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
            
            lK1_ratios.append(mean_ratio1)
            lK3_ratios.append(mean_ratio3)
            csv_writer.writerow([P, mean_ratio1, mean_ratio3])

    print("Solver finished.")
    # You can add plotting for the final ratios here if desired.

if __name__ == "__main__":
    # The main workflow
    # 1. Train the GPR model based on the micro-theory
    gpr_surrogate_model = train_gpr_model(d=d, n=n, delta=delta, b_const=b_const)
    
    # 2. Run the self-consistent solver using the trained model
    run_solver(gpr_surrogate_model)