import torch
from check_mps import check_mps_available_or_fail
def linear_activation(x: torch.Tensor) -> torch.Tensor:
    """
    The linear activation function: phi(x) = x.
    """


    return x

def erf_activation(x: torch.Tensor) -> torch.Tensor:
    """
    The error function activation: phi(x) = erf(x).
    """
    return torch.erf(x)

def third_degree_irrep_activation(x: torch.Tensor) -> torch.Tensor:
    """
    A third-degree polynomial activation function.
    This is a placeholder and would need to be defined based on the
    specific form of the third-degree hyperspherical irrep relevant
    to the N^(0) inputs of the middle layer.

    For a simple polynomial of degree 3, we can use:
    phi(x) = a*x^3 + b*x^2 + c*x + d

    However, a true hyperspherical irrep has specific transformation
    properties under rotations. Defining it generally requires more
    context about the symmetry group and the specific irrep.

    For this example, we'll use a simple third-degree polynomial element-wise.
    """
    # Example coefficients (can be adjusted)
    a: float = 0.1
    b: float = 0.05
    c: float = 1.0
    d: float = 0.0
    return a * (x**3) + b * (x**2) + c * x + d