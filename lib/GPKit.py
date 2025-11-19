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

 
def compute_nngp_kernel(phis_a, phis_b, eigenvalues):
    """
    Computes the NNGP kernel matrix K(A, B) given eigenfunctions and eigenvalues.

    Args:
        phis_a (torch.tensor): Eigenfunctions evaluated at points in set A.
                             Shape (num_points_a, num_eigenfunctions).
        phis_b (torch.tensor): Eigenfunctions evaluated at points in set B.
                             Shape (num_points_b, num_eigenfunctions).
        eigenvalues (np.ndarray): 1D array of NNGP eigenvalues (lambda_k).
                                  Shape (num_eigenfunctions,).

    Returns:
        np.ndarray: The NNGP kernel matrix K(A, B).
                    Shape (num_points_a, num_points_b).
    """
    # More efficient way often, by scaling phis_a columns directly:
    scaled_phis_a = phis_a * torch.sqrt(eigenvalues)
    scaled_phis_b = phis_b * torch.sqrt(eigenvalues)
    return scaled_phis_a @ scaled_phis_b.T


def compute_gpr_nngp_torch(
    phis_train,
    y_train,
    phis_test,
    eigenvalues,
    noise_variance
):
    """
    Computes the GPR predictive mean and variance using the NNGP kernel in PyTorch.

    Args:
        phis_train (torch.Tensor): Eigenfunctions evaluated at training points.
                                   Shape (num_train_points, num_eigenfunctions).
        y_train (torch.Tensor): Training target values.
                                Shape (num_train_points,).
        phis_test (torch.Tensor): Eigenfunctions evaluated at test points.
                                  Shape (num_test_points, num_eigenfunctions).
        eigenvalues (torch.Tensor): 1D tensor of NNGP eigenvalues (lambda_k).
                                    Shape (num_eigenfunctions,).
        noise_variance (float): The observation noise variance (sigma_noise^2).

    Returns:
        tuple: A tuple containing:
            - predictive_mean (torch.Tensor): Mean predictions for test points.
                                              Shape (num_test_points,).
            - predictive_variance (torch.Tensor): Variance predictions for test points.
                                                  Shape (num_test_points,).
    """
    num_train_points = phis_train.shape[0]
    num_test_points = phis_test.shape[0]
    
    # Ensure all tensors are on the same device
    device = phis_train.device
    y_train = y_train.to(device)
    phis_test = phis_test.to(device)
    eigenvalues = eigenvalues.to(device)

    K_xx = compute_nngp_kernel(phis_train, phis_train, eigenvalues) + noise_variance * torch.eye(num_train_points, device=device) 
   
    K_xx_inv = torch.linalg.inv(K_xx)
    alpha = K_xx_inv @ y_train

    K_xD = compute_nngp_kernel(phis_test, phis_train, eigenvalues)

    # Compute Predictive Mean
    predictive_mean = K_xD @ alpha


    return predictive_mean
    
    
def gpr_dot_product_explicit(train_x, train_y, test_x, sigma_0_sq):
    """
    Computes Gaussian Process Regression mean and standard deviation using explicit formulas
    with a Dot Product Kernel.

    Args:
        train_x (torch.Tensor): N_train x D tensor of training input features.
        train_y (torch.Tensor): N_train x 1 tensor of training target values.
        test_x (torch.Tensor): N_test x D tensor of test input features.
        sigma_0_sq (float or torch.Tensor): The sigma_0^2 hyperparameter for the DotProduct kernel.
                                            This is NOT trained, you provide its value.
        noise_var (float or torch.Tensor): The observational noise variance (added to diagonal).
                                           This is NOT trained, you provide its value.

    Returns:
        tuple: (mu_pred, sigma_pred)
            mu_pred (torch.Tensor): N_test x 1 tensor of predicted means.
            sigma_pred (torch.Tensor): N_test x 1 tensor of predicted standard deviations.
    """

    sigma_0_sq = torch.tensor(sigma_0_sq, dtype=torch.float32, device=train_x.device)

    # 1. Define the DotProduct kernel function (helper within the main function)
    def dot_product_kernel_torch(X1, X2):
        """
        Computes the DotProduct kernel matrix K(X1, X2) using PyTorch.
        k(xi, xj) = xi @ xj.T
        """
        a = X1 / X1.shape[1]**0.5
        b = X2 / X2.shape[1]**0.5
        return a @ b.T

    K_xx = dot_product_kernel_torch(train_x, train_x) + sigma_0_sq * torch.eye(train_x.shape[0], device=train_x.device) 

    K_xstar_x = dot_product_kernel_torch(test_x, train_x)

    # jitter = 1e-6 * torch.eye(train_x.shape[0], device=train_x.device)
    try:
        K_xx_inv = torch.linalg.inv(K_xx)
    except torch.linalg.LinAlgError as e:
        print(f"Error: K_xx is singular or ill-conditioned even with jitter: {e}")
        raise

    # 4. Predict Mean (mu_pred)
    # Formula: mu_pred = K(X_test, X_train) @ K(X_train, X_train)^-1 @ y_train
    mu_pred = K_xstar_x @ K_xx_inv @ train_y

    return mu_pred
