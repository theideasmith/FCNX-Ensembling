import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import argparse
import json
import re
# Import FCN3NetworkActivationGeneric from the lib directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from FCN3Network import FCN3NetworkActivationGeneric

def extract_params_from_model_name(model_path):
    """
    Extract parameters from model filename or config.json in the same directory.
    Supports patterns like: model_d150_n1600_chi50.pt or loads from config.json
    """
    model_path = Path(model_path)
    params = {}
    
    # Try to load from config.json in the same directory
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            # Extract parameters from config
            params['d'] = config.get('d')
            params['n1'] = config.get('n1', config.get('N'))
            params['chi'] = config.get('chi')
            params['P'] = config.get('P')
            if all(v is not None for v in params.values()):
                print(f"✓ Loaded parameters from config.json: {params}")
                return params
        except Exception as e:
            print(f"  Could not load config.json: {e}")
    
    # Try to extract from filename using pattern: d150_n1600_chi50 etc.
    filename = model_path.stem  # Get filename without extension
    param_patterns = {
        'd': r'd(\d+)',
        'n1': r'n1(\d+)',
        'chi': r'chi(\d+)',
        'P': r'P(\d+)'
    }
    
    for key, pattern in param_patterns.items():
        match = re.search(pattern, filename)
        if match:
            params[key] = int(match.group(1))
    
    if params:
        print(f"✓ Extracted parameters from filename: {params}")
        return params
    
    print("  No parameters found in filename or config.json")
    return {}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load FCN3 model and compute He3 projections")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model state dict file")
    parser.add_argument("--d", type=int, default=None, help="Input dimension (auto-detected from model name if not provided)")
    parser.add_argument("--n-samples", type=int, default=100_000_000, help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Batch size")
    parser.add_argument("--n1", type=int, default=None, help="Hidden layer 1 dimension (auto-detected from model name if not provided)")
    parser.add_argument("--chi", type=int, default=None, help="Chi parameter (auto-detected from model name if not provided)")
    parser.add_argument("--P", type=int, default=None, help="P parameter (auto-detected from model name if not provided)")
    args = parser.parse_args()
    
    # Try to extract parameters from model name/config first
    model_path = Path(args.model_path)
    detected_params = extract_params_from_model_name(model_path)
    
    # Use detected params, fall back to command-line args, then use defaults
    defaults = {'d': 150, 'n1': 1600, 'chi': 50, 'P': 3000}
    d = args.d or detected_params.get('d') or defaults['d']
    n_samples = args.n_samples
    batch_size = args.batch_size
    N = args.n1 or detected_params.get('n1') or defaults['n1']
    chi = args.chi or detected_params.get('chi') or defaults['chi']
    P = args.P or detected_params.get('P') or defaults['P']
    
    print(f"\nUsing parameters: d={d}, n1={N}, chi={chi}, P={P}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    ens = int(state_dict['W0'].shape[0])
    weight_var = (1.0 / d, 1.0 / N, 1.0 / (N * chi))

    model = FCN3NetworkActivationGeneric(
        d=d, n1=N, n2=N, P=P, ens=ens,
        activation="erf",
        weight_initialization_variance=weight_var,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    n_batches = (n_samples + batch_size - 1) // batch_size
    running_variances = []
    running_nsamples = []
    projections_sum = None
    n_total = 0

    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            X = torch.randn(current_batch_size, d, device=device)
            h1 = model.h1_preactivation(X)  # (batch_size, ens, n1)
            x0 = X[:, 1]#.view(-1,1,1)
            He3 = (x0 ** 3 - 3.0 * x0).view(-1, 1, 1) / 6**0.5 # (batch_size, 1, 1)
            proj = h1 * He3  # (batch_size, ens, n1)
            proj_sum = proj.sum(dim=0)  # (ens, n1)
            if batch_idx == 0:
                projections_sum = proj_sum
            else:
                projections_sum += proj_sum
            n_total += current_batch_size

            mean_proj = projections_sum / n_total  # (ens, n1)
            var = torch.var(mean_proj - torch.mean(mean_proj)).item()
            running_variances.append(var)
            running_nsamples.append(n_total)
            print(f"Processed batch {batch_idx+1}/{n_batches} | Samples: {n_total} | Variance: {var:.6e}")

    # Plot variance convergence
    plt.figure(figsize=(8, 5))
    plt.plot(running_nsamples, running_variances, marker='o')
    plt.xlabel('Number of samples')
    plt.ylabel('Variance of mean projections')
    plt.title('Convergence of Variance with Sample Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_convergence_loaded_model.png', dpi=150)
    plt.close()
    print("Saved variance convergence plot as variance_convergence_loaded_model.png")

    print(f"Final variance of mean projections over (ens, n1): {running_variances[-1]:.3e}")

if __name__ == "__main__":
    main()
