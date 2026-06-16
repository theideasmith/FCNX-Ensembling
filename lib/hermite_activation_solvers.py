import matplotlib.pyplot as plt
from scipy.optimize import root
import numpy as np
# --- 1. Variable Scaling for Balanced Arclength Norm ---
SCALES = np.array([100.0, 1.0, 100.0, 10000.0, 1.0])

def pack_state(l_sig, H11, H13, H33, q):
    v = np.array([l_sig, H11, H13, H33, q])
    return v * SCALES

def unpack_state(v):
    unscaled = v / SCALES
    return unscaled[0], unscaled[1], unscaled[2], unscaled[3], unscaled[4]

def project_to_psd(matrix, min_eig=0.01):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, min_eig, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# --- 2. The Physics Residuals ---
def physics_residuals(l_sig, H11, H13, H33, P, d, N1, N2, chi, kappa, epsilon, alpha=1.0):
    y = np.array([[1.0], [epsilon]])
    I2 = np.eye(2)
    
    l_sig = max(l_sig, 1e-10)
    H11 = max(H11, 1e-10)
    H33 = max(H33, 1e-10)
    
    J = np.array([[l_sig, 3 * l_sig**2], [3 * l_sig**2, 15 * l_sig**3]])
    H = np.array([[H11, H13], [H13, H33]])
    
    TrH = H11 + H33
    beta_val = (3.0 / np.sqrt(6.0)) * (TrH - 1.0)
    alpha_val = (1.0 + beta_val)**2
    
    K = np.array([[alpha_val * H11, 3 * np.sqrt(alpha_val) * H13 * H33],
                  [3 * np.sqrt(alpha_val) * H13 * H33, 15 * H33**3]])
    
    M = np.linalg.inv((kappa / P) * I2 + K)
    T = - (chi**2) * (M @ y @ y.T @ M)
    T_eff = np.array([[T[0,0] * alpha_val, 0.0], [0.0, 0.0]])
    
    J_inv = np.linalg.inv(J)
    H_inv_new = J_inv + (1.0 / (N2 * chi)) * T_eff
    H_inv_new = project_to_psd(H_inv_new, min_eig=0.01)
    H_new = np.linalg.inv(H_inv_new)
    
    V_tilde = J_inv @ (J - H_new) @ J_inv
    l_sig_new = 1.0 / (d + (N2 / N1) * V_tilde[0, 0])
    
    return [l_sig_new - l_sig, 
            H_new[0, 0] - H11, 
            H_new[0, 1] - H13, 
            H_new[1, 1] - H33]

# --- 3. Adaptive Pseudo-Arclength Continuation Solver ---
def solve_pac_eos(d, N1, N2, chi, kappa, epsilon, q_start, q_end, ds_nominal=0.15, max_steps=1000):
    lambda_sigma_curve = []
    H11_curve = []
    H33_curve = []
    P_curve = []
    
    # --- STEP 0: Find starting point v_0 ---
    P_start = 10**q_start
    def start_residuals(vars):
        l_sig, H11, H13, H33 = vars
        return physics_residuals(l_sig, H11, H13, H33, P_start, d, N1, N2, chi, kappa, epsilon)
    
    res = root(start_residuals, [1.0/d, 0.2, 0.0, 15.0/d**3], method='hybr')
    v_0 = pack_state(res.x[0], res.x[1], res.x[2], res.x[3], q_start)
    
    # --- STEP 1: Find second point v_1 ---
    q_1 = q_start + 0.02
    P_1 = 10**q_1
    def step1_residuals(vars):
        l_sig, H11, H13, H33 = vars
        return physics_residuals(l_sig, H11, H13, H33, P_1, d, N1, N2, chi, kappa, epsilon)
    
    res = root(step1_residuals, list(res.x), method='hybr')
    v_1 = pack_state(res.x[0], res.x[1], res.x[2], res.x[3], q_1)
    
    # Store initial points
    for v in [v_0, v_1]:
        l_sig, H11, _, H33, q = unpack_state(v)
        lambda_sigma_curve.append(l_sig)
        H11_curve.append(H11)
        H33_curve.append(H33)
        P_curve.append(10**q)
        
    v_prev = v_1
    t_prev = (v_1 - v_0) / np.linalg.norm(v_1 - v_0)
    
    # --- STEP 2: Tracing Loop ---
    print("Tracing the solution curve via Adaptive Pseudo-Arclength Continuation...")
    ds = ds_nominal
    
    for step in range(max_steps):
        success = False
        attempts = 0
        
        while not success and attempts < 6:
            v_pred = v_prev + ds * t_prev
            
            def pac_system(v_current):
                l_sig, H11, H13, H33, q = unpack_state(v_current)
                P = 10**q
                res_phys = physics_residuals(l_sig, H11, H13, H33, P, d, N1, N2, chi, kappa, epsilon)
                res_arc = np.dot(v_current - v_prev, t_prev) - ds
                return np.append(res_phys, res_arc)
            
            res = root(pac_system, v_pred, method='lm', tol=1e-8)
            
            if res.success:
                v_curr = res.x
                success = True
            else:
                ds /= 2.0  # Halve the step size
                attempts += 1
        
        if not success:
            print(f"PAC step {step} failed to converge after 6 attempts. Stopping.")
            break
            
        l_sig, H11, _, H33, q = unpack_state(v_curr)
        
        lambda_sigma_curve.append(l_sig)
        H11_curve.append(H11)
        H33_curve.append(H33)
        P_curve.append(10**q)
        
        t_curr = (v_curr - v_prev) / np.linalg.norm(v_curr - v_prev)
        
        if q >= q_end:
            print(f"Reached target log(P) = {q:.3f} in {step+1} steps.")
            break
            
        # CRITICAL FIX: Speed up recovery rate (from 1.2 to 1.5) to take larger steps in flat regions
        ds = min(ds * 1.5, ds_nominal)
        
        v_prev = v_curr
        t_prev = t_curr
        
    return np.array(P_curve), np.array(H11_curve), np.array(H33_curve), np.array(lambda_sigma_curve)
