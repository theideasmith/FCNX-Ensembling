import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def arcsin_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute arcsin kernel matrix for inputs X (P, d)."""
    # X shape: (P, d)
    XXT = torch.einsum('ui,vi->uv', X, X) / X.shape[1]
    diag = torch.sqrt((1 + 2 * XXT).diag())
    denom = diag[:, None] * diag[None, :]
    arg = (2 * XXT / denom).clamp(-1 + 1e-7, 1 - 1e-7)
    return (2 / torch.pi) * torch.arcsin(arg)

def compute_lambda3_streaming(d, P_total=10_000_000, n1=1000, batch_size=100_000, device="cuda"):
    """
    Computes empirical variance of projections onto the 3rd Hermite polynomial.
    Includes Jensen's inequality correction for weight norms.
    """
    dtype = torch.float64
    sigma_w2 = 1.0 / d
    # Initialize weights
    W0 = torch.empty(1, n1, d, device=device, dtype=dtype).normal_(0.0, sigma_w2**0.5)
    
    # Weight Stats & Bias Correction
    w_norms_sq = torch.sum(W0**2, dim=2)
    theoretical_damping_factor = torch.mean(1.0 / (1 + 2 * w_norms_sq)**3).item()
    
    projections = torch.zeros(1, n1, dtype=dtype, device=device)
    num_batches = P_total // batch_size
    
    with torch.no_grad():
        for _ in range(num_batches):
            X_batch = torch.randn(batch_size, d, dtype=dtype, device=device)
            # phi_3 is the 3rd Hermite polynomial (normalized)
            phi_3_batch = (X_batch[:, 0]**3 - 3*X_batch[:, 0]) 
            h0 = torch.einsum('qkl,ul->qku', W0, X_batch)
            projections += torch.einsum('qku,u->qk', torch.erf(h0), phi_3_batch)

    projections /= P_total
    l3_emp = torch.var(projections).item()
    
    # Theoretical Calculation
    constant_part = (16 * 15 * (sigma_w2**3)) / (np.pi)
    theory_corrected = constant_part * theoretical_damping_factor
    
    return l3_emp, theory_corrected

def run_comparison():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    P_kernel = 20000 # Memory limit for eigvalsh
    d_values = torch.logspace(1, 3, steps=8).int().tolist()
    
    results = {
        "emp_l3": [],
        "theory_corrected": [],
        "kernel_tail": []
    }

    for d in tqdm(d_values, desc="Scaling d"):
        # 1. Empirical Projection & Corrected Theory
        l3_emp, theory_corr = compute_lambda3_streaming(d, device=device)
        results["emp_l3"].append(l3_emp)
        results["theory_corrected"].append(theory_corr)
        
        # 2. Arcsin Kernel Eigenvalues
        X = torch.randn(P_kernel, d, device=device, dtype=dtype)
        K = arcsin_kernel(X)
        eigs = torch.linalg.eigvalsh(K)
        eigs_desc = torch.sort(eigs, descending=True).values
        
        # Take the tail starting at d and average
        # This represents the higher-frequency power (including cubic)
        tail_val = torch.mean(eigs_desc[(d+1):(d+101)]) / P_kernel
        results["kernel_tail"].append(tail_val.item())
        
        del X, K, eigs
        torch.cuda.empty_cache()

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.loglog(d_values, results["theory_corrected"], 'k--', label='Corrected Theory $\lambda_3$ (Jensen)')
    plt.loglog(d_values, results["emp_l3"], 'ro', label='Empirical Projections (Streaming)')
    plt.loglog(d_values, results["kernel_tail"], 'b-', label='Arcsin Kernel Tail Mean $eigs[d:]/P$')
    
    plt.xlabel('Dimension (d)')
    plt.ylabel('Variance / Eigenvalue Magnitude')
    plt.title('Discrepancy Scaling: Projections vs. Theory vs. Kernel Tail')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig('benchmark_projection_eigvalsh_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_comparison()