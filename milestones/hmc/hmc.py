import jax.numpy as jnp
from jax import random, jit
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median
import numpy as np

# ====================================================================
# 0. Configuration and Data Setup
# ====================================================================

# Network Parameters (based on your input)
D_in = 6         # d
H1 = 50          # n1
H2 = 50          # n2
P_data = 50      # P (Number of data points)
Temperature = 1.0 # T
D_out = 1        # Output dimension

# HMC Sampling Parameters
N_chains = 6            # Number of parallel chains (simulates ensembles)
N_warmup = 4000         # Burn-in steps
N_samples = 5000        # Samples to keep
rng_key = random.PRNGKey(42)

# Derived Prior Standard Deviations (Bayesian/Mean-Field Scaling)
# Prior Variance: sigma^2 = 1/d, 1/n1, 1/n2 (mean-field)
# Prior Std Dev: sigma = 1/sqrt(variance)
prior_std_W0 = 1.0 / jnp.sqrt(D_in)
prior_std_W1 = 1.0 / jnp.sqrt(H1)
prior_std_A = 1.0 / jnp.sqrt(H2)
numpyro.set_host_device_count(4)
# Likelihood standard deviation (sigma^2 = 2 * T, from MSE sum loss)
likelihood_std = jnp.sqrt(Temperature) 

# Use float64 for higher numerical stability
FLOAT_DTYPE = jnp.float64
numpyro.set_platform('cpu') # Or 'gpu', but typically not both

# Generate synthetic data using float64
X_data = np.random.randn(P_data, D_in)
y_data = X_data[:, 0:1]
X_data_jnp = jnp.array(X_data, dtype=FLOAT_DTYPE)
y_data_jnp = jnp.array(y_data, dtype=FLOAT_DTYPE)

# ====================================================================
# 1. The Bayesian Model (Priors and Likelihood)
# ====================================================================

def linear_fcn_bayesian_model(X, y):
    """
    Defines the priors and forward pass for the three-layer FCN.
    
    Args:
        X: Data (P, D_in)
        y: Targets (P, D_out)
    """
    
    # --- Priors ---
    # W0: (n1 x d)
    W0 = numpyro.sample("W0", 
        dist.Normal(0.0, prior_std_W0).expand([H1, D_in]).to_event(2)
    )
    # W1: (n2 x n1)
    W1 = numpyro.sample("W1", 
        dist.Normal(0.0, prior_std_W1).expand([H2, H1]).to_event(2)
    )
    # A:  (n2 x 1)
    A = numpyro.sample("A", 
        dist.Normal(0.0, prior_std_A).expand([H2, D_out]).to_event(2)
    )

    # --- Forward Pass (Linear Activation) ---
    # h1_preactivation: X @ W0.T @ W1.T
    H0_pre = X @ W0.T  # (P, H1)
    H1_pre = H0_pre @ W1.T # (P, H2)
    mu = H1_pre @ A  # (P, D_out)
    
    # --- Likelihood ---
    with numpyro.plate("data", len(y)):
        numpyro.sample("obs", dist.Normal(mu, likelihood_std), obs=y)

# ====================================================================
# 2. HMC Sampler Setup and Execution (REVISED for Adaptiveness and Init)
# ====================================================================

# We will let NUTS tune the step size automatically by removing the initial_step_size argument.
# We also switch the initializer to the mode/mode of the prior (zero).

kernel = NUTS(linear_fcn_bayesian_model, target_accept_prob=0.75)

mcmc = MCMC(
    kernel,
    num_warmup=N_warmup, # Let the 1000 warm-up steps tune the step size and mass matrix
    num_samples=N_samples,
    num_chains=N_chains,
    progress_bar=True,
    chain_method='parallel'
)

print(f"🚀 Starting HMC (NUTS) with adaptive tuning and float64: (T={Temperature})...")
mcmc.run(rng_key, X=X_data_jnp, y=y_data_jnp)

# Get all samples
posterior_samples = mcmc.get_samples(group_by_chain=False)

# Debug: print available keys in posterior_samples
print(f"Available keys in posterior_samples: {list(posterior_samples.keys())}")

W0_samples = posterior_samples['W0']
W1_samples = posterior_samples['W1']

# ====================================================================
# 3. Exact H_eig Kernel Calculation (Mirroring PyTorch Logic)
# ====================================================================

@jit
def calculate_H_eig_kernel_sample(W0, W1, X, H1_width, P_data_size):
    """
    Calculates the P x P kernel matrix (hh_inf) based on the H_eig function, 
    using the h1_preactivation (X @ W0.T @ W1.T).
    
    Args:
        W0 (H1, D_in), W1 (H2, H1): Sampled weights
        X (P, D_in): Data
        H1_width (n1): Used for normalization
        P_data_size (P): Used for normalization
    
    Returns:
        K_H_eig (P, P): The exact kernel matrix for one sample.
    """
    
    # 1. f_inf = h1_preactivation: X @ W0.T @ W1.T
    Z0 = X @ W0.T # (P, H1)
    f_inf = Z0 @ W1.T # (P, H2) -> (This is the final pre-activation vector)
    
    # 2. Calculate the inner product: hh_inf_i = torch.einsum('uim,vim->uvi', f_inf, f_inf)
    # Equivalent JAX operation across the width dimension (H2)
    # We get a (P, P, H2) tensor
    hh_inf_i = f_inf[:, None, :] * f_inf[None, :, :] 

    # 3. Sum over the width (m=H2): K_uv = sum_m f_um * f_vm
    hh_inf_summed = jnp.sum(hh_inf_i, axis=2) # (P, P)

    # 4. Normalization used in your code: 1 / (n1 * X.shape[0])
    K_H_eig = hh_inf_summed / (H1_width * P_data_size)
    
    # Note: The division by self.ens is handled implicitly by the MCMC chain averaging.
    return K_H_eig


# --- Iterate and Calculate Eigenvalues for all Samples ---
N_total_samples = len(W0_samples)
kernel_eigenvalues = np.zeros((N_total_samples, P_data))

print(f"\nCalculating and tracking eigenvalues for {N_total_samples} exact H_eig kernel samples...")

for i in range(len(W0_samples)):
    # Pass the sample's W0, W1, data, width H1, and P_data size
    K_sample = calculate_H_eig_kernel_sample(W0_samples[i], W1_samples[i], X_data_jnp, H1, P_data)
    
    # Eigenvalues
    eigvals = jnp.linalg.eigvalsh(K_sample)
    kernel_eigenvalues[i] = jnp.sort(eigvals)

# ====================================================================
# 4. Final Analysis and Summary (Replicating Checkpoint Logic)
# ====================================================================

# We report the mean and variance across the posterior distribution.
mean_eigenvalues = np.mean(kernel_eigenvalues, axis=0)
lH_max = np.max(mean_eigenvalues)

print("\n--- Final HMC Posterior Eigenvalue Analysis (H_eig Kernel) ---")
print(f"Mean Max Eigenvalue (lH): {lH_max:.6f}")

if P_data > 1:
    # Std Dev of the largest eigenvalue (the last sorted one) across the posterior samples
    lH_std_across_posterior = np.std(kernel_eigenvalues[:, -1])
    
    # Mean and Std of the remaining eigenvalues
    remaining_eigenvalues_samples = kernel_eigenvalues[:, :-1]
    remaining_mean = np.mean(remaining_eigenvalues_samples)
    remaining_std = np.std(remaining_eigenvalues_samples)
    
    print(f"Std Dev of Max Eigenvalue (across posterior): {lH_std_across_posterior:.6f}")
    print(f"Mean of remaining eigenvalues (Eig 1..P-1): {remaining_mean:.6f}")
    print(f"Std Dev of remaining eigenvalues: {remaining_std:.6f}")

# mcmc.print_summary()

# ====================================================================
# 5. Compute Theoretical Predictions using Experiment.py
# ====================================================================

import sys
from pathlib import Path
import tempfile
import json

# Add lib to path
LIB_DIR = str(Path(__file__).parent.parent.parent / "lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import torch
from Experiment import ExperimentLinear

print("\n--- Computing Theoretical Eigenvalues via Experiment ---")

# Convert parameters to match Experiment interface
d = D_in
N = H1  # n1
chi = H2  # n2 (chi parameter)
P = P_data
kappa = 1.0  # Default kappa
device = torch.device("cpu")

with tempfile.TemporaryDirectory() as tmpdir:
    exp = ExperimentLinear(
        file=tmpdir,
        N=N,
        d=d,
        chi=chi,
        P=P,
        ens=1,
        device=device,
        kappa=kappa,
    )
    
    predictions = exp.eig_predictions()

print("Theoretical H eigenvalues (via ExperimentLinear):")
if predictions is not None:
    print(f"  lHT (target):        {predictions.lHT:.10e}")
    print(f"  lHP (perpendicular): {predictions.lHP:.10e}")
    print(f"  lJT:                 {predictions.lJT:.10e}")
    print(f"  lJP:                 {predictions.lJP:.10e}")
else:
    print("  Warning: predictions returned None")

# ====================================================================
# 6. Compare and Save Results to JSON
# ====================================================================

results = {
    "parameters": {
        "d": int(d),
        "N": int(N),
        "chi": int(chi),
        "P": int(P),
        "Temperature": float(Temperature),
        "kappa": float(kappa),
        "n_chains": int(N_chains),
        "n_warmup": int(N_warmup),
        "n_samples": int(N_samples),
    },
    "hmc_results": {
        "mean_max_eigenvalue": float(lH_max),
        "std_max_eigenvalue_across_posterior": float(lH_std_across_posterior) if P_data > 1 else None,
        "mean_remaining_eigenvalues": float(remaining_mean) if P_data > 1 else None,
        "std_remaining_eigenvalues": float(remaining_std) if P_data > 1 else None,
        "all_mean_eigenvalues": mean_eigenvalues.tolist(),
    },
    "theory_predictions": {
        "lHT": float(predictions.lHT) if predictions is not None else None,
        "lHP": float(predictions.lHP) if predictions is not None else None,
        "lJT": float(predictions.lJT) if predictions is not None else None,
        "lJP": float(predictions.lJP) if predictions is not None else None,
        "lKT": float(predictions.lKT) if predictions is not None and predictions.lKT is not None else None,
        "lKP": float(predictions.lKP) if predictions is not None and predictions.lKP is not None else None,
    },
    "comparison": {}
}

# Add comparison metrics
if predictions is not None:
    results["comparison"]["hmc_vs_theory_lHT"] = {
        "hmc_mean_max": float(lH_max),
        "theory_lHT": float(predictions.lHT),
        "absolute_error": float(lH_max - predictions.lHT),
        "relative_error": float((lH_max - predictions.lHT) / predictions.lHT) if predictions.lHT != 0 else None,
    }

# Save to JSON
output_file = Path(__file__).parent / "hmc_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_file}")
print("\n--- Comparison Summary ---")
if predictions is not None:
    print(f"HMC Mean Max Eigenvalue:   {lH_max:.10e}")
    print(f"Theory lHT:                {predictions.lHT:.10e}")
    print(f"Absolute Error:            {lH_max - predictions.lHT:.10e}")
    print(f"Relative Error:            {(lH_max - predictions.lHT) / predictions.lHT:.6%}")