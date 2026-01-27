import numpy as np
import torch
from pathlib import Path
import sys

# Import FCN3NetworkActivationGeneric from the lib directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

def main():
    d = 150
    n_samples = 100_000_000
    batch_size = 50_000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model parameters
    P = 100
    N = 1600
    ens = 5
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N))

    # Initialize model
    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    model.eval()

    n_batches = (n_samples + batch_size - 1) // batch_size
    projections_sum = None
    n_total = 0

    running_variances = []
    running_nsamples = []

    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            X = torch.randn(current_batch_size, d, device=device)
            h1 = model.h1_preactivation(X)  # (batch_size, ens, n1)
            x0 = X[:, 0]
            He3 = (x0 ** 3 - 3.0 * x0).view(-1, 1, 1)  # (batch_size, 1, 1)
            proj = h1 * He3  # (batch_size, ens, n1)
            proj_sum = proj.sum(dim=0)  # (ens, n1)
            if batch_idx == 0:
                projections_sum = proj_sum
            else:
                projections_sum += proj_sum
            n_total += current_batch_size

            mean_proj = projections_sum / n_total  # (ens, n1)
            var = torch.var(mean_proj).item()
            running_variances.append(var)
            running_nsamples.append(n_total)
            print(f"Processed batch {batch_idx+1}/{n_batches} | Samples: {n_total} | Variance: {var:.6e}")

    # Plot variance convergence
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(running_nsamples, running_variances, marker='o')
    plt.xlabel('Number of samples')
    plt.ylabel('Variance of mean projections')
    plt.title('Convergence of Variance with Sample Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_convergence.png', dpi=150)
    plt.close()
    print("Saved variance convergence plot as variance_convergence.png")

    print(f"Final variance of mean projections over (ens, n1): {running_variances[-1]:.3e}")

if __name__ == "__main__":
    main()
