import numpy as np
from scipy.optimize import fsolve, root

def eigenvalue_solver(params):
    P, k, d, sigma2,  N, chi = params
    initial_guess = np.ones(3) * 0.1 # Initial guess for the eigenvalues
    """Solves the eigenvalue equations for the empirical kernels."""
    # This line unpacks the values from the tuple
    # l is a shorthand for lambda
    # the next letter is the kernel type.
    # the kernel type is suffixed by h if it is a helper kernel
    # the final letter, T or p signifies whether the eigenvalue is for the target or the perpendicular kernel
    varnamesT = ['lKT','lKhT', 'lHT']
    varnamesp = ['lKT', 'lKhp', 'lHp']

    def eqsp(vals):
        lKp, lKhp, lHp = vals
        equations = [
            (k/(P*chi))*(lHp/(lHp + kappa/P)) - (lHp/(lHp + kappa/P))**2 - lKp,
            -chi/lHp + chi**2 * (lKp/(lHp**2))   - lKhp,
            1/(lKhp * sigma2 / (chi * N) + d/sigma2)                       - lHp
        ]

        return np.array(equations)
    def eqsT(vals):
        lKT, lKhT, lHT = vals

        equations = [
            (k/(P*chi))*(lHT/(lHT + kappa/P)) - (lHT/(lHT + kappa/P))**2 - lKT,
            -chi/lHT + chi**2 * (lKT/(lHT**2) - (lHT + k/P)**-2)    - lKhT,
            #  chi**2 * ( - (lHT + k/P)**-2)    - lKhT,
            1/(lKhT*(sigma2**2)/(chi*N) + d/sigma2)                      - lHT, 
        ]

        return np.array(equations)

    solutionT = root(eqsT, initial_guess,  method='lm').x
    solutionp = root(eqsp, initial_guess, method='lm').x

    print("-----<<((Eigenvalue Solver Target))>>-----")
    print("Solution found:")
    # Prints out each variable name and its corresponding value
    for varname, value in zip(varnamesT, solutionT):
        print(f"{varname} = {value:.10f}")

    print("-----<<((Eigenvalue Solver Perp))>>-----")
    print("Solution found:")
    # Prints out each variable name and its corresponding value
    for varname, value in zip(varnamesp, solutionp):
        print(f"{varname} = {value:.10f}")

    return solutionT, solutionp

if __name__ == '__main__':
    # Using ens.eigenvalue_solver to solve the analytical eigenvalues
    # Assuming 'standard_hyperparams' and 'eos_solvers' would be imported
    # if this were part of a larger project.
    # For this standalone file, we'll define the parameters directly.

    d: int = 50
    N = 1000
    T = 1.0

    kappa = T
    P: int = 200

    chi = N
    print("kappa/p = ") 
    print(kappa/P)
    solver_params = [P, kappa, d, 1.0, N, chi]
    solution = eigenvalue_solver(solver_params)
    print("-------------")
    if solution is not None:
        lKT, lKhT, lHT  = solution[0]
        lKT, lKhp, lHp = solution[1]
        Cov_KT = (kappa/(P*chi)) * (lHT)/( lHT + kappa/(P))
        Cov_Kp =  (kappa/(P*chi)) * (lHp)/( lHp + kappa/(P))
        print(f"Kappa / P = {kappa/P}")

        print("\nEigenvalues (from eigenvalue_solver):")
        print(f"Cov KT = {Cov_KT:.12}")
        print(f"lHT = {lHT:.10f}")
        print(f"Cov_Kp = {Cov_Kp:.12}")
        print(f"lHp = {lHp:.10f}")
        print(f"CovlHT/CovlHp = {Cov_KT/Cov_Kp: .10f}")

    # solver_params_alt = [P, kappa, d, sigma2, N0, N1, chi]
    # solution_alt = eigenvalue_solver_alternative(solver_params_alt)
    # if solution_alt is not None:
    #     lKT_alt, lKhT_alt, lHT_alt, lHhT_alt, lJT_alt, lKp_alt, lKhp_alt, lHp_alt, lHhp_alt, lJp_alt = solution_alt
    #     print("\nEigenvalues (from eigenvalue_solver_alternative):")
    #     print(f"lKT = {lKT_alt:.10f}")
    #     print(f"lKhT = {lKhT_alt:.10f}")
    #     print(f"lHT = {lHT_alt:.10f}")
    #     print(f"lHhT = {lHhT_alt:.10f}")
    #     print(f"lJT = {lJT_alt:.10f}")
    #     print(f"lKp = {lKp_alt:.10f}")
    #     print(f"lKhp = {lKhp_alt:.10f}")
    #     print(f"lHp = {lHp_alt:.10f}")
    #     print(f"lHhp = {lHhp_alt:.10f}")
    #     print(f"lJp = {lJp_alt:.10f}")