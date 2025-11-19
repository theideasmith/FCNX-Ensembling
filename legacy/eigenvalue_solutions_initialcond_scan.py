import os
import numpy as np
from scipy.optimize import root
import argparse
import matplotlib.pyplot as plt

def eigenvalue_solver_with_guess(params, guess):
    P, k, d, sigma2, N0, N1, chi = params
    initial_guess = np.ones(5) * guess
    def eqsp(vals):
        lKp, lKhp, lHp, lHhp, lJp = vals
        equations = [
            k/(chi*P) * lHp / (lHp + k/P)                           - lKp,
            -lKhp,
            sigma2*((1/chi) * (lKhp/N1) + lJp**-1)**-1       - lHp,
            (-1.0/lJp + lHp/(lJp**2))               - lHhp,
            1/(lHhp / N0 + d/sigma2)                       - lJp
        ]
        return np.array(equations)
    def eqsT(vals):
        lKT, lKhT, lHT, lHhT, lJT= vals
        equations = [
            (k/(chi*P))*(lHT)/(lHT + k/P) + (lHT/(lHT+k/P))**2      - lKT,
            -(chi**2 * (lHT + k/P)**-2)    - lKhT,
            ((1.0/(chi*N1))*lKhT + 1.0/lJT)**-1       - lHT,
            (-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2))               - lHhT,
            1/(lHhT/N0 + d)                      - lJT, 
        ]
        return np.array(equations)
    try:
        solutionT = root(eqsT, initial_guess, method='hybr')
        solutionp = root(eqsp, initial_guess, method='hybr')
        if not (solutionT.success and solutionp.success):
            error_msg = f"Solver did not converge for guess={guess}.\nTarget message: {solutionT.message}\nPerp message: {solutionp.message}"
            raise RuntimeError(error_msg)
        # Compute residuals
        residualsT = np.abs(eqsT(solutionT.x))
        residualsp = np.abs(eqsp(solutionp.x))
        return solutionT.x, solutionp.x, residualsT, residualsp
    except Exception as e:
        print(f"Error for initial guess {guess}: {e}")
        return None, None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid scan initial condition for eigenvalue solver.')
    parser.add_argument('--d', type=int, default=3, help='Input dimension (default: 3)')
    parser.add_argument('--N0', type=int, default=200, help='First hidden layer width (default: 200)')
    parser.add_argument('--N1', type=int, default=200, help='Second hidden layer width (default: 200)')
    parser.add_argument('--P', type=int, default=30, help='Number of training samples (default: 30)')
    parser.add_argument('--kappa', type=float, default=1.0, help='kappa parameter (default: 1.0)')
    parser.add_argument('--chi', type=float, default=200, help='chi parameter (default: 200)')
    parser.add_argument('--sigma2', type=float, default=1.0, help='sigma^2 parameter (default: 1.0)')
    args = parser.parse_args()

    d = args.d
    N0 = args.N0
    N1 = args.N1
    P = args.P
    kappa = args.kappa
    sigma2 = args.sigma2
    chi = args.chi

    initial_guesses = np.arange(0.1, 3.01, 0.5)
    eigperp_vals = []
    eigtarg_vals = []
    failed_guesses = []
    solutionsT = []
    solutionsp = []
    residualsT = []
    residualsp = []

    solver_params = [P, kappa, d, sigma2, N0, N1, chi]

    for guess in initial_guesses:
        solT, solp, resT, resp = eigenvalue_solver_with_guess(solver_params, guess)
        if solT is not None and solp is not None:
            eigtarg_vals.append(solT[2])
            eigperp_vals.append(solp[2])
            solutionsT.append(solT)
            solutionsp.append(solp)
            residualsT.append(resT)
            residualsp.append(resp)
        else:
            eigtarg_vals.append(np.nan)
            eigperp_vals.append(np.nan)
            solutionsT.append([np.nan]*5)
            solutionsp.append([np.nan]*5)
            residualsT.append([np.nan]*5)
            residualsp.append([np.nan]*5)
            failed_guesses.append(guess)

    # Save plot to folder
    save_dir = 'eigenvalue_solutions_scan'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'initial_guess_scan.png')
    plt.figure(figsize=(8,6))
    plt.plot(initial_guesses, eigtarg_vals, label='eigtarg (lHT)')
    plt.plot(initial_guesses, eigperp_vals, label='eigperp (lHp)')
    plt.xlabel('Initial Guess Value')
    plt.ylabel('Solution Value')
    plt.title('Effect of Initial Guess on eigperp and eigtarg Solutions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Save solutions and residuals to file
    np.savez(os.path.join(save_dir, 'initial_guess_solutions_and_residuals.npz'),
             initial_guesses=initial_guesses,
             solutionsT=np.array(solutionsT),
             solutionsp=np.array(solutionsp),
             residualsT=np.array(residualsT),
             residualsp=np.array(residualsp))
    print(f"Solutions and residuals saved to {os.path.join(save_dir, 'initial_guess_solutions_and_residuals.npz')}")

    if failed_guesses:
        print(f"Failed to converge for initial guesses: {failed_guesses}") 