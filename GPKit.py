import torch
def GPR_kernel(x1: torch.Tensor, x2: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    """
    Computes the Gaussian Process (GP) kernel between two sets of points.

    Args:
        x1 (torch.Tensor): The first set of points.
        x2 (torch.Tensor): The second set of points.
        length_scale (float): The length scale parameter for the kernel.

    Returns:
        torch.Tensor: The computed kernel matrix.
    """
    corr = torch.mm(x1, x2.t())
    return corr

def matinverse(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(matrix)

def GPInv(kernel: torch.Tensor, a: float = 1e-5) -> torch.Tensor:
    return torch.matmul(kernel, matinverse(kernel + a*torch.eye(kernel.shape[0])))

def GPR(kernel : torch.Tensor, y : torch.Tensor, a: float = 1e-5) -> torch.Tensor:
    """
    Computes the Gaussian Process Regression (GPR) prediction.

    Args:
        kernel (torch.Tensor): The kernel matrix.
        y (torch.Tensor): The target values.
        a (float): A small constant for numerical stability.

    Returns:
        torch.Tensor: The GPR prediction.
    """
    return torch.matmul(GPInv(kernel, a), y)