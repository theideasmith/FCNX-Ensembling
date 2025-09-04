import numpy as np

def solve_system(delta, n, chi, kappa, P, d):
    """
    Numerically solves the system of coupled equations for a given delta.

    Args:
        delta (float): The hyperparameter, either 0 or 1.
        n (float): The value of n.
        chi (float): The value of chi.
        kappa (float): The value of kappa.
        P (float): The value of P.
        d (float): The value of d.

    Returns:
        A dictionary with the converged values of the variables.
    """
    b = 4 / (3 * np.pi)
    b=1
    # Initial guesses for the variables
    lH, lV, lJ, lK, lT = 0.05, 0.1, 0.1, 0.1, 0.1

    # Iteration parameters
    max_iterations = 10000
    tolerance = 1e-12
    
    print(f"Solving for delta = {delta}...")
    for i in range(max_iterations):
        lH_old, lV_old, lJ_old, lK_old, lT_old = lH, lV, lJ, lK, lT

        # Update dependent variables
        EChh = lH_old + (d - 1) / d
        EwSigmaw = 1/ (d + delta * b * lV_old / n) + (d - 1) / d
        GammaYh2 = 1
        (4 / np.pi) * (1 + 2 * EChh)

        lJ = b * 1 / (d + delta * b * lV_old / n)
        lK = GammaYh2 * lH_old
        ffT = (kappa / (P * chi)) * (lK / (kappa/P + lK)) + (lK / (kappa/P + lK))**2 * delta
        lT =  -(chi**2) / (kappa/P + lK)**2 * delta
        lH = 1 / (1/lJ + GammaYh2 * lT / (n * chi))
        lV = (1 / lJ**2) * lH - (1 / lJ)

        lJ = 1 / (d + delta * lV_old / n)
        lK =  lH_old
        lT =  -(chi**2) / (kappa/P + lK)**2 * delta
        lH = 1 / (1/lJ + GammaYh2 * lT / (n * chi))
        lV = (1 / lJ**2) * lH - (1 / lJ)

        # Check for convergence
        if (abs(lH - lH_old) < tolerance and
            abs(lV - lV_old) < tolerance and
            abs(lJ - lJ_old) < tolerance and
            abs(lK - lK_old) < tolerance and
            abs(lT - lT_old) < tolerance):
            print(f"Converged after {i+1} iterations.")
            break
    else:
        print(f"Did not converge after {max_iterations} iterations. Final values may not be accurate.")

    # Calculate ffT using the final converged values
   
    
    return {
        'lH': lH,
        'lV': lV,
        'lJ': lJ,
        'lK': lK,
        'lT': lT,
        'ffT': ffT,
        'iterations': i+1
    }

# Constants
n = 400.0
chi = 1.0
kappa = 1.0
P = 20.0
d = 20.0

# Solve for delta = 0
print("\n--- MEAN FIELD SCALING")
results_delta_0 = solve_system(delta=0, n=n, chi=n, kappa=kappa, P=P, d=d)
print("\n--- Results for delta = 0 ---")
if results_delta_0:
    for key, value in results_delta_0.items():
        print(f"{key}: {value}")

print("\n" + "="*30 + "\n")

# Solve for delta = 1
results_delta_1 = solve_system(delta=1, n=n, chi=n, kappa=kappa, P=P, d=d)
print("\n--- Results for delta = 1 ---")
if results_delta_1:
    for key, value in results_delta_1.items():
        print(f"{key}: {value}")

print("\n--- STANDARD SCALING")
results_delta_0 = solve_system(delta=0, n=n, chi=1.0, kappa=kappa, P=P, d=d)
print("\n--- Results for delta = 0 ---")
if results_delta_0:
    for key, value in results_delta_0.items():
        print(f"{key}: {value}")

print("\n" + "="*30 + "\n")

# Solve for delta = 1
results_delta_1 = solve_system(delta=1, n=n, chi=1.0, kappa=kappa, P=P, d=d)
print("\n--- Results for delta = 1 ---")
if results_delta_1:
    for key, value in results_delta_1.items():
        print(f"{key}: {value}")