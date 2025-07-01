import numpy as np
from scipy.optimize import fsolve, root
import argparse

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
            (-lJT**-1 + sigma2**-1 * lJT**-2 * lHT)              -lHhT,
            1/(lHT*sigma2/N0 + d/sigma2)                           - lJT, 
            (k/(chi*P) )* lHp / (lHp + k/P)                         - lKp,
            -chi/lHp + chi**2 * lKp/lHp**2                        -  lKhp,
            (lKhp/(N1*chi) + lJp**-1)**-1          -  lHp,
            (-lJp**-1 + lJp**-2 * lHp)               - lHhp,
            1/(lHhp * 1.0 / N0 + d)                      - lJp
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
    initial_guess = np.ones(5) * 0.5 # Initial guess for the eigenvalues
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
            # -chi/lHp -chi/lHp + chi**2 * lKp/lHp**2                          - lKhp,
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
            # -chi/lHT + chi**2 * (lKT/(lHT**2) - (lHT + k/P)**-2)    - lKhT,
            -(chi**2 * (lHT + k/P)**-2)    - lKhT,
            ((1.0/(chi*N1))*lKhT + 1.0/lJT)**-1       - lHT,
            (-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2))               - lHhT,
            1/(lHhT/N0 + d)                      - lJT, 
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

    return solutionT[2], solutionp[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve eigenvalue equations for empirical kernels.')
    parser.add_argument('--d', type=int, default=3, help='Input dimension (default: 50)')
    parser.add_argument('--N0', type=int, default=200, help='First hidden layer width (default: 1000)')
    parser.add_argument('--N1', type=int, default=200, help='Second hidden layer width (default: 1000)')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature (default: 1.0)')
    parser.add_argument('--P', type=int, default=30, help='Number of training samples (default: 200)')
    parser.add_argument('--kappa', type=float, default=1.0, help='kappa parameter (default: 1.0)')
    parser.add_argument('--chi', type=float, default=200, help='chi parameter (default: N1)')
    parser.add_argument('--sigma2', type=float, default=1.0, help='sigma^2 parameter (default: 1.0)')
    args = parser.parse_args()

    d = args.d
    N0 = args.N0
    N1 = args.N1
    T = args.T
    P = args.P
    kappa = args.kappa
    sigma2 = args.sigma2

    # Define the N values as requested
    N_values = [50, 200, 600, 1000]
    
    eigperp_results = {}
    eigtarg_results = {}

    for N_val in N_values:
        print(f"\n--- Solving for N = {N_val} ---")
        # For the problem statement, chi = N
        chi = N_val 

        solver_params = [P, kappa, d, sigma2, N_val, N_val, chi]
        print(f"solver_params: {solver_params}")
        eigperp_val, eigtarg_val = eigenvalue_solver(solver_params)
        
        eigperp_results[N_val] = eigperp_val
        eigtarg_results[N_val] = eigtarg_val

        # Print detailed results for the current N value if needed
        # (This part can be uncommented if you need to see the intermediate print statements
        # from the original `eigenvalue_solver` function during the loop)
        # print("-------------")
        # if solution is not None:
        #     lKT, lKhT, lHT, lHhT, lJT = solution[0]
        #     lKp, lKhp, lHp, lHhp, lJp = solution[1]
        #     print(f"lHhT = {-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2)}")
        #     Cov_KT = kappa/(chi * P) * (lHT)/( lHT + kappa/P)
        #     Cov_KhT = - chi / lHT + chi**2 * (lKT / lHT**2)
        #     Cov_Kp = kappa/(chi * P) * (lHp)/( lHp + kappa/P)
        #     print(f"Kappa / P = {kappa/P}")
        #     Cov_Khp = - chi / lHp + chi**2 * (lKp / lHp**2)
        #     print(f"Absolute difference of covariance eigenvalues: {Cov_KT - Cov_Kp: .15f}")
        #     print(f"Kernel Target-Perp Ratio: {Cov_KT/Cov_Kp: .15f}")
        #     print("\nEigenvalues (from eigenvalue_solver):")
        #     print(f"Cov KT = {Cov_KT:.12}")
        #     print(f"lHT = {lHT:.10f}")
        #     print(f"lHhT = {lHhT:.10f}")
        #     print(f"lJT = {lJT:.10f}")
        #     print(f"Cov_Kp = {Cov_Kp:.12}")
        #     print(f"lHp = {lHp:.10f}")
        #     print(f"lHhp = {lHhp:.10f}")
        #     print(f"lJp = {lJp:.10f}")

    print("\n--- Final Results ---")
    print("eigperp_results (N: eigperp):", eigperp_results)
    print("eigtarg_results (N: eigtarg):", eigtarg_results)


# import numpy as np
# from scipy.optimize import fsolve, root
# import argparse

# def eigenvalue_solver_alternative(params):
#     P, k, d, sigma2, N0, N1, chi = params
#     initial_guess = np.ones(10) * 1.1 # Initial guess for the eigenvalues




#     def eqs(vals):
#         lKT, lKhT, lHT, lHhT, lJT, lKp, lKhp, lHp, lHhp, lJp = vals
#         c1 = (k/(chi*P))
#         c2 = (lHT)/(lHT + k/P)
#         c3 = c1*c2
#         sigma2_f = c3
#         mean2_f = c2**2
#         equations = [
#             -lKT + sigma2_f + mean2_f,
#             -chi/lHT + chi**2 * lKT/lHT**2 - chi**2 * (lHT + k/P)**-2 - lKhT,
#             sigma2*(1/chi * sigma2/N1*lKT + lJT**-1)**-1           - lHT,
#             (-lJT**-1 + sigma2**-1 * lJT**-2 * lHT)              -lHhT,
#             1/(lHT*sigma2/N0 + d/sigma2)                           - lJT, 
#             (k/(chi*P) )* lHp / (lHp + k/P)                         - lKp,
#             -chi/lHp + chi**2 * lKp/lHp**2                        -  lKhp,
#             (lKhp/(N1*chi) + lJp**-1)**-1          -  lHp,
#             (-lJp**-1 + lJp**-2 * lHp)               - lHhp,
#             1/(lHhp * 1.0 / N0 + d)                      - lJp
#         ]
#         return np.array(equations)

#     sol = root(eqs, initial_guess, method='lm') # Using the Levenberg-Marquardt algorithm
#     if sol.success:
#         return sol.x
#     else:
#         print(f"Alternative solver failed: {sol.message}")
#         return None

# def eigenvalue_solver(params):
#     P, k, d, sigma2, N0, N1, chi = params
#     initial_guess = np.ones(5) * 0.5 # Initial guess for the eigenvalues
#     """Solves the eigenvalue equations for the empirical kernels."""
#     # This line unpacks the values from the tuple
#     # l is a shorthand for lambda
#     # the next letter is the kernel type.
#     # the kernel type is suffixed by h if it is a helper kernel
#     # the final letter, T or p signifies whether the eigenvalue is for the target or the perpendicular kernel
#     varnamesT = ['lKT', 'lKhT', 'lHT', 'lHhT', 'lJT']
#     varnamesp = ['lKp', 'lKhp', 'lHp', 'lHhp', 'lJp']

#     def eqsp(vals):
#         lKp, lKhp, lHp, lHhp, lJp = vals
#         equations = [
#             k/(chi*P) * lHp / (lHp + k/P)                           - lKp,
#             # -chi/lHp -chi/lHp + chi**2 * lKp/lHp**2                          - lKhp,
#             -lKhp,
#             sigma2*((1/chi) * (lKhp/N1) + lJp**-1)**-1       - lHp,
#             (-1.0/lJp + lHp/(lJp**2))               - lHhp,
#             1/(lHhp / N0 + d/sigma2)                       - lJp
#         ]

#         return np.array(equations)
#     def eqsT(vals):
#         lKT, lKhT, lHT, lHhT, lJT= vals

#         equations = [
#             (k/(chi*P))*(lHT)/(lHT + k/P) + (lHT/(lHT+k/P))**2      - lKT,
#             # -chi/lHT + chi**2 * (lKT/(lHT**2) - (lHT + k/P)**-2)    - lKhT,
#             -(chi**2 * (lHT + k/P)**-2)    - lKhT,
#             ((1.0/(chi*N1))*lKhT + 1.0/lJT)**-1       - lHT,
#             (-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2))               - lHhT,
#             1/(lHhT/N0 + d)                      - lJT, 
#         ]

#         return np.array(equations)
#     solutionT = root(eqsT, initial_guess,  method='lm').x
#     solutionp = root(eqsp, initial_guess, method='lm').x

#     print("-----<<((Eigenvalue Solver Target))>>-----")
#     print("Solution found:")
#     # Prints out each variable name and its corresponding value
#     for varname, value in zip(varnamesT, solutionT):
#         print(f"{varname} = {value:.10f}")

#     print("-----<<((Eigenvalue Solver Perp))>>-----")
#     print("Solution found:")
#     # Prints out each variable name and its corresponding value
#     for varname, value in zip(varnamesp, solutionp):
#         print(f"{varname} = {value:.10f}")

#     return solutionT, solutionp

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Solve eigenvalue equations for empirical kernels.')
#     parser.add_argument('--d', type=int, default=3, help='Input dimension (default: 50)')
#     parser.add_argument('--N0', type=int, default=200, help='First hidden layer width (default: 1000)')
#     parser.add_argument('--N1', type=int, default=200, help='Second hidden layer width (default: 1000)')
#     parser.add_argument('--T', type=float, default=1.0, help='Temperature (default: 1.0)')
#     parser.add_argument('--P', type=int, default=30, help='Number of training samples (default: 200)')
#     parser.add_argument('--kappa', type=float, default=1.0, help='kappa parameter (default: 1.0)')
#     parser.add_argument('--chi', type=float, default=200, help='chi parameter (default: N1)')
#     parser.add_argument('--sigma2', type=float, default=1.0, help='sigma^2 parameter (default: 1.0)')
#     args = parser.parse_args()

#     d = args.d
#     N0 = args.N0
#     N1 = args.N1
#     T = args.T
#     P = args.P
#     kappa = args.kappa
#     sigma2 = args.sigma2
#     chi = args.chi if args.chi is not None else N1

#     print("kappa/p = ") 
#     print(kappa/P)
#     solver_params = [P, kappa, d, sigma2, N0, N1, chi]
#     solution = eigenvalue_solver(solver_params)
#     print("-------------")
#     if solution is not None:
#         lKT, lKhT, lHT, lHhT, lJT = solution[0]
#         lKp, lKhp, lHp, lHhp, lJp = solution[1]
#         print(f"lHhT = {-1.0/lJT + 1.0/sigma2 * lHT/(lJT**2)}")

#         Cov_KT = kappa/(chi * P) * (lHT)/( lHT + kappa/P)
#         Cov_KhT = - chi / lHT + chi**2 * (lKT / lHT**2)

#         Cov_Kp = kappa/(chi * P) * (lHp)/( lHp + kappa/P)
#         print(f"Kappa / P = {kappa/P}")
#         # WHEN KAPPA/P IS SMALL, THEN THE COVARIANCES ARE NEARLY EXACTLY EQUAL TO ONE ANOTHER
#         # The no feature learning being tracked down to the helper fields might be due to 
#         # my mixing up covariances and second moments
#         Cov_Khp = - chi / lHp + chi**2 * (lKp / lHp**2)
#         print(f"Absolute difference of covariance eigenvalues: {Cov_KT - Cov_Kp: .15f}")
#         print(f"Kernel Target-Perp Ratio: {Cov_KT/Cov_Kp: .15f}")
#         print("\nEigenvalues (from eigenvalue_solver):")
#         print(f"Cov KT = {Cov_KT:.12}")
#         print(f"lHT = {lHT:.10f}")
#         print(f"lHhT = {lHhT:.10f}")
#         print(f"lJT = {lJT:.10f}")
#         print(f"Cov_Kp = {Cov_Kp:.12}")
#         print(f"lHp = {lHp:.10f}")
#         print(f"lHhp = {lHhp:.10f}")
#         print(f"lJp = {lJp:.10f}")

#     # solver_params_alt = [P, kappa, d, sigma2, N0, N1, chi]
#     # solution_alt = eigenvalue_solver_alternative(solver_params_alt)
#     # if solution_alt is not None:
#     #     lKT_alt, lKhT_alt, lHT_alt, lHhT_alt, lJT_alt, lKp_alt, lKhp_alt, lHp_alt, lHhp_alt, lJp_alt = solution_alt
#     #     print("\nEigenvalues (from eigenvalue_solver_alternative):")
#     #     print(f"lKT = {lKT_alt:.10f}")
#     #     print(f"lKhT = {lKhT_alt:.10f}")
#     #     print(f"lHT = {lHT_alt:.10f}")
#     #     print(f"lHhT = {lHhT_alt:.10f}")
#     #     print(f"lJT = {lJT_alt:.10f}")
#     #     print(f"lKp = {lKp_alt:.10f}")
#     #     print(f"lKhp = {lKhp_alt:.10f}")
#     #     print(f"lHp = {lHp_alt:.10f}")
#     #     print(f"lHhp = {lHhp_alt:.10f}")
#     #     print(f"lJp = {lJp_alt:.10f}")
