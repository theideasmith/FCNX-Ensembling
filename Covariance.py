import torch
from opt_einsum import contract

def compute_avg_channel_covariance(f, X, layer_name='fc2'):
    f.eval()
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    layer = dict(f.named_modules())[layer_name]
    handle = layer.register_forward_hook(get_activation(layer_name))
    with torch.no_grad():
        f(X)
    handle.remove()
    fc2_output = activations[layer_name]
    P, N = fc2_output.shape
    fc2_centered = fc2_output - fc2_output.mean(dim=1, keepdim=True)
    cov_matrices = torch.einsum('pn,qn->npq', fc2_centered, fc2_centered) / (P)
    return torch.mean(cov_matrices, dim=0)

def compute_avg_channel_covariance_fcn3(f, X):
    f.eval()
    
    with torch.no_grad():
        h1_out = f.h1_activation(X)

    # P: samples, b: ensembles, N1: layer width
    P, b, N1 = h1_out.shape
    fc2_centered = h1_out - h1_out.mean(dim=1, keepdim=True)
    avged_cov = contract('pin,qin->pq', fc2_centered, fc2_centered, backend='torch') / (N1*b)
    print(avged_cov.shape)
    return avged_cov


def project_onto_target_functions(K, y):
    """
    Project covariance matrix K onto target functions y.

    Args:
        K (torch.Tensor): Covariance matrix of shape (P, P)
        y (torch.Tensor): Target functions (eigenfunctions) of shape (P, M)

    Returns:
        torch.Tensor: Projections y^T K y / (y^T y) of shape (M,)
    """
    # Ensure shapes are compatible
    P = K.shape[0]
    assert K.shape == (P, P), f"Expected K to be ({P}, {P}), got {K.shape}"
    assert y.shape[0] == P, f"Expected y to have {P} rows, got {y.shape[0]}"

    # Compute y^T K y for all target functions using einsum
    yKy = torch.einsum('pm,pq,qn->mn', y, K, y).diagonal()  # Shape: (M,)

    # Compute y^T y for normalization
    yTy = torch.einsum('pm,pm->m', y, y)  # Shape: (M,)

    # Normalize: y^T K y / (y^T y)
    # Add small epsilon to avoid division by zero
    projections = yKy / (yTy + 1e-10)

    return projections
def transform_eigenvalues(eigenvalues, k, chi, P):

    # Ensure inputs are valid
    assert P > 0, "P must be positive"
    assert k >= 0, "k must be non-negative"

    # Compute k/P
    k_over_PX = k / (P * chi)
    # Transform eigenvalues
    transformed = (eigenvalues / (eigenvalues + k / P)) * (k_over_PX)

    return transformed
