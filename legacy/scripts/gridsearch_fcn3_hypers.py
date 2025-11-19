"""
SciPy-based script to perform grid search over hyperparameters to maximize lKT/lKp ratio.
Uses updated equations and prints results without saving to a file.
"""
import numpy as np
from scipy.optimize import fsolve
from itertools import product

# Configuration
BOUNDS = {
    'n': (2, 500),
    'dim': (w, 100),
    'P': (1, 1000),
    'k': (0.1, 10.0)
}
GRID_STEPS = {
    'n': 50,      # Steps of 50
    'dim': 50,    # Steps of 10
    'P': 500,     # Steps of 100
    'k': [0.1, 0.316, 1.0, 3.162, 10.0]  # Logarithmic steps
}
TOLERANCE = 1e-8

# def solve_target_equations(params):
#     """Solve the target equations for lH and lK."""
#     n, dim, P, k = params
#     χ = n
#     g = k / (P * χ)

#     def equations(vars):
#         lH, lK = vars
#         term1 = lH / (lH + k / P)
#         eq1 = lK - (g * term1 + term1**2)
#         eq2 = lH - 1.0 / (dim + ((χ**2 * (lK / lH**2 - (1.0 / (k / P + lH))**2) - 1.0 / lH * χ) / (χ * n)))
#         return [eq1, eq2]

#     initial_guess = [1.0 / dim, g * (1.0 / dim) / (1.0 / dim + k / P)]
#     lH, lK = fsolve(equations, initial_guess, xtol=TOLERANCE)
#     return lH, lK

def solve_alternative_lh(params):
    """Solve the alternative lH equation for lKT."""
    n, dim, P, k = params
    χ = n

    def equation(lH):
        term = (χ**2 * (1.0 / (k / P + lH))**2) / (χ * n)
        return lH - 1.0 / (dim - term)

    initial_guess = 1.0 / dim
    lH_alt = fsolve(equation, initial_guess, xtol=TOLERANCE)[0]
    return lH_alt

def compute_ratio(params):
    """Compute lKT/lKp ratio."""
    n, dim, P, k = params
    χ = n
    g = k / (P * χ)

    # Perpendicular equations (analytic)
    lH_perp = 1.0 / dim
    lK_perp = g * lH_perp / (lH_perp + k / P)

    # lKT from alternative lH
    lH_alt = solve_alternative_lh(params)
    lKT = g * lH_alt / (lH_alt + k / P)
    lKp = lK_perp

    ratio = lKT / lKp
    return ratio, lKT, lKp, lH_perp, lH_alt

def grid_search():
    """Perform grid search to maximize lKT/lKp ratio."""
    # Define grid points
    n_vals = np.arange(BOUNDS['n'][0], BOUNDS['n'][1] + GRID_STEPS['n'], GRID_STEPS['n'])
    dim_vals = np.arange(BOUNDS['dim'][0], BOUNDS['dim'][1] + GRID_STEPS['dim'], GRID_STEPS['dim'])
    P_vals = np.arange(BOUNDS['P'][0], BOUNDS['P'][1] + GRID_STEPS['P'], GRID_STEPS['P'])
    k_vals = GRID_STEPS['k']

    best_ratio = np.array(1.0)
    best_params = (n_vals[0], dim_vals[0],P_vals[0],k_vals[0])
    best_lKT = 0.1
    best_lKp = 0.0
    best_lHp = 0.1
    best_lHT = 0.1
    # Iterate over all combinations
    total_combinations = len(n_vals) * len(dim_vals) * len(P_vals) * len(k_vals)
    print(f"Evaluating {total_combinations} combinations...")

    for n, dim, P, k in product(n_vals, dim_vals, P_vals, k_vals):
        params = (n, dim, P, k)
        ratio, lKT, lKp, lH_perp, lH_alt = compute_ratio(params)
        if ratio > best_ratio:
            best_ratio = ratio
            best_params = params
            best_lKT = lKT
            best_lKp = lKp
            best_lHp = lH_perp
            best_lHT = lH_alt

    # Print results
    print("\nGrid search complete. Best parameters:")
    print(f"n: {best_params[0]:.6f}")
    print(f"dim: {best_params[1]:.6f}")
    print(f"P: {best_params[2]:.6f}")
    print(f"k: {best_params[3]:.6f}")
    print(f"lKT: {best_lKT:.6f}")
    print(f"lKp: {best_lKp:.6f}")
    print(f"lHp: {best_lHp: .6f}")
    print(f"lHT: {best_lHT: .6f}")
    print(f"lKT/lKp: {best_ratio}")

def main():
    """Run the grid search."""
    grid_search()

if __name__ == '__main__':
    main()

"""
Output:

Evaluating 112455 combinations...

Grid search complete. Best parameters:
n: 850.000000
dim: 960.000000
P: 100.000000
k: 3.162000
lKT: 0.000005
lKp: 0.000001
lHp:  0.001042
lHT:  0.004827
lKT/lKp: 4.1523153586448736
"""
