import torch

def cumulants4(
    self: torch.Tensor,
    dims: tuple[int, ...],
    keepdim: bool = False,
    correction: int = 1
) -> torch.Tensor:
    """
    Compute the first 4 cumulants along specified dimensions.
    
    Returns:
        tensor of shape (*, 4) containing:
            [κ₁, κ₂, κ₃, κ₄]
        where:
            κ₁ = mean
            κ₂ = variance
            κ₃ = 3rd central moment (related to skewness)
            κ₄ = 4th cumulant (related to excess kurtosis)
    
    Args:
        self: Input tensor
        dims: Tuple of dimensions to reduce over (e.g. (2,3) for H,W)
        keepdim: If True, keeps reduced dims with size 1
        correction: Bessel's correction for variance (1 for unbiased, 0 for population)
    
    Example:
        x = torch.randn(16, 3, 64, 64)
        c = x.cumulants4(dims=(2,3))
        # c.shape -> (16, 3, 4)
        mean, var, skew_moment, kurt_cumul = c.unbind(-1)
    """
    x = self
    dims = tuple(d if d >= 0 else x.ndim + d for d in dims)
    
    if not dims:
        raise ValueError("dims must be non-empty")

    # Compute central moments using torch native functions (most stable)
    mean = x.mean(dim=dims, keepdim=True)          # μ₁ = E[x]
    centered = x - mean
    
    # Raw central moments
    m2 = torch.mean(centered**2, dim=dims, keepdim=True)  # E[(x-μ)²]
    m3 = torch.mean(centered**3, dim=dims, keepdim=True)  # E[(x-μ)³]
    m4 = torch.mean(centered**4, dim=dims, keepdim=True)  # E[(x-μ)⁴]

    # Apply bias correction if requested
    n = torch.tensor([x.shape[d] for d in dims], device=x.device, dtype=x.dtype).prod()
    if correction:
        factor = n / (n - 1) if correction else 1.0
        m2 = m2 * factor
        # Higher moment corrections are more complex — we skip for simplicity (common practice)

    # Cumulants from central moments (exact relations)
    κ1 = mean.squeeze() if not keepdim else mean
    κ2 = m2
    κ3 = m3
    κ4 = m4 - 3 * m2 * m2  # This is the key: κ₄ = μ₄ − 3μ₂²

    result = torch.stack([κ1, κ2.squeeze(), κ3.squeeze(), κ4.squeeze()], dim=-1)

    if keepdim:
        # Restore reduced dimensions
        for d in reversed(sorted(dims)):
            result = result.unsqueeze(d)

    return result