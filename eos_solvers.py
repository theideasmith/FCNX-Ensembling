import FCN3 as nnkit 
from torch.utils.data import TensorDataset, random_split
import torch 
import GPKit as gpkit
import ensembling as ens


# Solving the eigenvalue equations is simple, so we can rely on a simply solver: 

from scipy.optimize import fsolve


def empirical_kernel(dataset : TensorDataset): 
    return gpkit.GPR_kernel(dataset.tensors[1], dataset.tensors[1], length_scale=1.0)

def collect_covariances(outputs):
    covs = [torch.cov(output) for output in outputs]
    return sum(covs) / len(covs)

def compute_posterior_kernels(models : list, data : TensorDataset):
    """
    Compute the posterior kernels for each internal layer of the ensemble of models.
    
    Args:
        models (list): A list of trained models.
        data (torch.TensorDataset): The dataset to compute the kernels on.
        
    Returns:
        dict: A dictionary of posterior kernels for each internal layer.
    """
    inner_outputs = ens.ensemble_forward(models, data)
    posterior_kernels = {}
    for k, outputs in inner_outputs.items():
        posterior_kernels[k] = collect_covariances(outputs)
    return posterior_kernels

import numpy as np
from scipy.optimize import minimize

def solve_for_h_given_sigma_f(Sigma_f, P, kappa, d):
    """Solves for H given Sigma_f: Sigma_f = H @ (H + P/kappa * I)_inv."""
    I = np.identity(d)

    def objective(h_flat):
        """Objective: Minimize ||Sigma_f - H @ (H + P/kappa * I)_inv||."""
        H = h_flat.reshape(d, d)
        return np.linalg.norm(Sigma_f - H @ np.linalg.inv(H + P / kappa * I))

    h_initial = np.identity(d).flatten()  # Initial guess
    result = minimize(objective, h_initial, method='L-BFGS-B')
    return result.x.reshape(d, d) if result.success else None  # Returns H or None

# def solve_for_htilde_on_data(X : torch.Tensor, N0 : int, N1 : int , sigma2 : float, d : int):
#      I = np.identity(d)
#      def objective(htilde_flat):
#         Htilde = htilde_flat.reshape(d, d)
#         J = sigma2 * X @ np.linalg.inv(sigma2 / N0 * X.T @ Htilde @ X + d / sigma2 * I) @ X.T
#         return np.linalg.norm((1/N1)*Htilde - 


def eigenvalue_solver_alternative(params):
    P, k, d, sigma2, N0, N1, chi = params
    initial_guess = np.ones(10) * 1.1 # Initial guess for the eigenvalues

    def eqs(vals):
        lKT, lKhT, lHT, lHhT, lJT, lKp, lKhp, lHp, lHhp, lJp = vals

        equations = [
            lKT - (lHT)/(lHT + P/k) - (P*chi/k)**2 * ((lHT)/(lHT + P/k))**2,
            lKhT + chi/lHT - chi**2 * (lKT/lHT**2) + chi**4 * (P/k)**2 * (lHT + P/k)**-2,
            lHT - sigma2*((1/chi)*(sigma2/N1)*lKT + 1/lJT)**-1,
            lHhT -N1*(-1/lJT  + (1/sigma2)*(lJT**-2)* lHT),
            lJT - 1/(lHhT*sigma2/N0 + d/sigma2),
            lKp - (lHp)/(lHp + P/k),
            lKhp + chi / lHp - chi**2 * lKp/lHp**2,
            lHp - sigma2*(1/chi*sigma2/N1*lKp + 1/lJp)**-1,
            lHhp - N1*(-1/lJp + 1/sigma2*lJp**-2*lHp),
            lJp - 1/(lHhp*sigma2/N0 + d/sigma2)
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
    initial_guess = np.ones(10) * 1.1 # Initial guess for the eigenvalues
    """Solves the eigenvalue equations for the empirical kernels."""
    # This line unpacks the values from the tuple
    # l is a shorthand for lambda
    # the next letter is the kernel type. 
    # the kernel type is suffixed by h if it is a helper kernel
    # the final letter, T or p signifies whether the eigenvalue is for the target or the perpendicular kernel
    varnames = ['lKT', 'lKhT', 'lHT', 'lHhT', 'lJT', 'lKp', 'lKhp', 'lHp', 'lHhp', 'lJp']


    def eqs(vals):
        lKT, lKhT, lHT, lHhT, lJT, lKp, lKhp, lHp, lHhp, lJp = vals

        equations = [
            lKT - (lHT)/(lHT + P/k) - (P*chi/k)**2 * ((lHT)/(lHT + P/k))**2,
            lKhT + chi/lHT - chi**2 * (lKT/lHT**2) + chi**4 * (P/k)**2 * (lHT + P/k)**-2,
            lHT - sigma2*((1/chi)*(sigma2/N1)*lKT + 1/lJT)**-1,
            lHhT -N1*(-1/lJT  + (1/sigma2)*(lJT**-2)* lHT),
            lJT - 1/(lHhT*sigma2/N0 + d/sigma2),
            lKp - (lHp)/(lHp + P/k),
            lKhp + chi / lHp - chi**2 * lKp/lHp**2,
            lHp - sigma2*(1/chi*sigma2/N1*lKp + 1/lJp)**-1,
            lHhp - N1*(-1/lJp + 1/sigma2*lJp**-2*lHp),
            lJp - 1/(lHhp*sigma2/N0 + d/sigma2)
        ]
        
        return np.array(equations)

    solution, infodict, ier, mesg = fsolve(eqs, initial_guess, full_output=True)
    if ier == 1:
        print("-----<<((Eigenvalue Solver))>>-----")
        print("Solution found:")
        # Prints out each variable name and its corresponding value
        for varname, value in zip(varnames, solution):
            print(f"{varname} = {value:.10f}")
    else:
        print(f"No solution found or convergence issues: {mesg}")

    return solution
                                  
def eigenvalue_equations(Sigma_f, P, kappa, d, sigma2, N0, N1, chi):
    
    lambda_fs = np.linalg.eigvals(Sigma_f)
    lambda_f_target = np.max(lambda_fs)
    lambda_f_perps = np.delete(lambda_fs, np.argmax(lambda_fs))

    lambda_h_target = lambda_f_target * (P/kappa) / (1 - lambda_f_target * (P/kappa))
    lambda_h_perps = lambda_f_perps * (P/kappa) / (1 - lambda_f_perps * (P/kappa))

    lambda_k_target = lambda_f_target + (P * chi/kappa)**2 * lambda_f_target**2
    lambda_ktilde_target = -chi * (1/lambda_h_target) + chi**2 * (lambda_k_target/lambda_h_target**2) - chi**4 * (P/kappa)**2 * (lambda_h_target + P/kappa)**-2



    lambda_J_target = 1.0/(lambda_fs_target * sigma2/N0 + d/sigma2)
    lambda_J_perps = 1.0/(lambda_fs_perps * sigma2/N0 + d/sigma2)

    

def kernel_backprop_from_empirical_kernel(dataset : TensorDataset, hyperparams : dict): 
    """
    Using the empirical output kernel, compute the empirical kernels for each layer of the FCN3 by "kernel backprop."

    Solves the EoS equations for the empirical kernels using the posterior kernels equations.

    As follows: 

    
        Σ_f = H[H + P/κ * I]_inv
        # K~ = -χ * H_inv + χ**2 * H_inv * K * H_inv - χ**4 * (P / κ)**2 * (H_inv * Sigma_f) * y * y.T * (Sigma_f * H_inv)

        # K = H * (H + P / κ * I)_inv + (P * χ / κ)**2 * H * (H + κ / P * I)_inv * y * y.T * H * (H + κ / P * I)_inv

        # H = σ**2 * ((1 / χ) * (σ**2 / N_1) * K~ + J_inv)_inv

        # (1 / N_1) * H_tilde = -J_inv + (1 / σ**2) * J_inv * H * J_inv

        # J = σ**2 * X * (σ**2 / N_0 * X.T * H_tilde * X + d / σ**2 * I_dx_d)_inv * X.T
        # Σ_ij_w = <w_i_0 * w_j_0>  # Sigma_ij_w is the average of w_i_0 * w_j_0    
    """
    chi = hyperparams['chi']
    N1 = hyperparams['N0']
    N2 = hyperparams['N1']
    sigma = hyperparams['sigma']
    d = hyperparams['d']
    P = hyperparams['P']
    kappa = hyperparams['kappa']

    Σf = empirical_kernel(dataset)
    H = solve_for_h_given_sigma_f(Σf, P, kappa, d)
    if H is None:
        raise ValueError("Failed to solve for H.")
    Hinv = np.linalg.inv(H)
    HinvΣf = Σf @ Hinv
    K = Σf  + (P * chi / kappa)**2 * HinvΣf @ dataset.tensors[1] @ dataset.tensors[1].T @ HinvΣf.T
    Ktilde = -chi * Hinv + chi**2 * Hinv @ K @ Hinv - chi**4 * (P / kappa)**2 * (HinvΣf @ dataset.tensors[1] @ dataset.tensors[1].T @ HinvΣf.T)
    Jinv = Hinv/sigma**2  - (1/chi) * sigma**2 * (1/N1) * Ktilde
    Htilde = N1*(-Jinv + (1/sigma**2) * Jinv @ H @ Jinv)
    
    solution = {
        'H': H,
        'Hinv': Hinv,
        'K': K,
        'Ktilde': Ktilde,
        'Jinv': Jinv,
        'Htilde': Htilde
    }

    return solution
"""
TODO: 

- Diagonalize the kernel matrix
- Train ensemble of FCN3s and compute the posterior kernels for each internal layer
- Compute the empirical kernel for each layer using the posterior kernels equations
- Compute the internal eigenvalues and eigenvectors of the empirical kernels using the eigenvalue equations

Think:  How to compute kernels out of ensembles of FCN3s?
Solution: 
"""