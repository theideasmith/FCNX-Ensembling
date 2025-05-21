import numpy as np
from scipy.optimize import fsolve, root

def eigenvalue_solver_alternative(params):
    P, k, d, sigma2, N0, N1, chi = params
    initial_guess = np.ones(10) * 1.1 # Initial guess for the eigenvalues




    def eqs(vals):
        lKT, lKhT, lHT, lHhT, lJT, lKp, lKhp, lHp, lHhp, lJp = vals
        c1 = (k/(chi*P))
        c2 = (lHT)/(lHT + k/P)
        c3 = c1*c2
        sigma2_f = c3
        mean2_f = c2**2
        equations = [
            -lKT + sigma2_f + mean2_f,
            -chi/lHT + chi**2 * lKT/lHT**2 - chi**2 * (lHT + k/P)**-2 - lKhT,
            sigma2*(1/chi * sigma2/N1*lKT + lJT**-1)**-1           - lHT,
            N1*(-lJT**-1 + sigma2**-1 * lJT**-2 * lHT)              -lHhT,
            1/(lHT*sigma2/N0 + d/sigma2)                           - lJT, 
            k/(chi*P) * lHp / (lHp + k/P) -                        - lKp,
            -chi/lHp + chi**2 * lKp/lHp**2                        -  lKhp,
            sigma2*(1/chi * sigma2/N1*lKp + lJp**-1)**-1          -  lHp,
            N1*(-lJp**-1 + 1/sigma2 * lJp**-2 * lHp)               - lHhp,
            1/(lHhp * sigma2 / N0 + d/sigma2)                      - lJp
        ]
        return np.array(equations)

    sol = root(eqs, initial_guess, method='lm') # Using the Levenberg-Marquardt algorithm
    if sol.success:
        return sol.x
    else:
        print(f"Alternative solver failed: {sol.message}")
        return None

def eigenvalue_solver(params):
    P, k, d, sigma2, N0, N1, chi = params
    initial_guess = np.ones(5) * 0.1 # Initial guess for the eigenvalues
    """Solves the eigenvalue equations for the empirical kernels."""
    # This line unpacks the values from the tuple
    # l is a shorthand for lambda
    # the next letter is the kernel type.
    # the kernel type is suffixed by h if it is a helper kernel
    # the final letter, T or p signifies whether the eigenvalue is for the target or the perpendicular kernel
    varnamesT = ['lKT', 'lKhT', 'lHT', 'lHhT', 'lJT']
    varnamesp = ['lKp', 'lKhp', 'lHp', 'lHhp', 'lJp']

    def eqsp(vals):
        lKp, lKhp, lHp, lHhp, lJp = vals
        equations = [
            k/(chi*P) * lHp / (lHp + k/P)                           - lKp,
            -chi/lHp + chi**2 * lKp/lHp**2                          - lKhp,
            sigma2*((1/chi) * (sigma2/N1)*lKhp + lJp**-1)**-1       - lHp,
            (-1.0/lJp + 1.0/sigma2 * lHp/(lJp**2))               - lHhp,
            1/(lHhp * sigma2 / N0 + d/sigma2)                       - lJp
        ]

        return np.array(equations)
    def eqsT(vals):
        lKT, lKhT, lHT, lHhT, lJT= vals

        equations = [
            (k/(chi*P))*(lHT)/(lHT + k/P) + (lHT/(lHT+k/P))**2      - lKT,
            -chi/lHT + chi**2 * (lKT/(lHT**2) - (lHT + k/P)**-2)    - lKhT,
            sigma2*((1/chi) * (sigma2/N1)*lKhT + lJT**-1)**-1       - lHT,
            (-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2))               - lHhT,
            1/(lHhT*(sigma2**2)/N0 + d/sigma2)                      - lJT, 
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

    d: int = 400
    N0 = 40000
    N1 = 50000
    T = 1

    kappa = T
    P: int = 300
    sigma2 = 1.0

    chi = N0
    print("kappa/p = ") 
    print(kappa/P)
    solver_params = [P, kappa, d, sigma2, N0, N1, chi]
    solution = eigenvalue_solver(solver_params)
    print("-------------")
    if solution is not None:
        lKT, lKhT, lHT, lHhT, lJT = solution[0]
        lKp, lKhp, lHp, lHhp, lJp = solution[1]
        print(f"lHhT = {-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2)}")

        Cov_KT = kappa/(chi * P) * (lHT)/( lHT + kappa/P)
        Cov_KhT = - chi / lHT + chi**2 * (lKT / lHT**2)

        Cov_Kp = kappa/(chi * P) * (lHp)/( lHp + kappa/P)
        print(f"Kappa / P = {kappa/P}")
        # WHEN KAPPA/P IS SMALL, THEN THE COVARIANCES ARE NEARLY EXACTLY EQUAL TO ONE ANOTHER
        # The no feature learning being tracked down to the helper fields might be due to 
        # my mixing up covariances and second moments
        Cov_Khp = - chi / lHp + chi**2 * (lKp / lHp**2)
        print(f"Absolute difference of covariance eigenvalues: {Cov_KT - Cov_Kp: .15f}")
        print(f"Kernel Target-Perp Ratio: {Cov_KT/Cov_Kp: .15f}")
        print("\nEigenvalues (from eigenvalue_solver):")
        print(f"Cov KT = {Cov_KT:.12}")
        print(f"lHT = {lHT:.10f}")
        print(f"lHhT = {lHhT:.10f}")
        print(f"lJT = {lJT:.10f}")
        print(f"Cov_Kp = {Cov_Kp:.12}")
        print(f"lHp = {lHp:.10f}")
        print(f"lHhp = {lHhp:.10f}")
        print(f"lJp = {lJp:.10f}")

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