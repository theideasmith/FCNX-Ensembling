using QuadGK
using FastGaussQuadrature
using Printf

# 1. The function we are integrating
# f(w) = w^6 / (T + 2w^2)^3
function integrand_core(w, T)
    return w^6 / (T + 2.0 * w^2)^3
end

# 2. Ground Truth using Adaptive Integration (QuadGK)
# High precision: 1e-12 tolerance
function ground_truth(μ, σ, T)
    # Standard normal variable z
    # Integral of f(μ + σz) * exp(-z^2/2) / sqrt(2π)
    f_z(z) = integrand_core(μ + σ * z, T) * exp(-z^2/2.0) / sqrt(2π)
    # Integrate from -10 to 10 (sufficient for Gaussian measure)
    val, err = quadgk(f_z, -12.0, 12.0, rtol=1e-13)
    return val
end

# 3. Gauss-Hermite Quadrature Implementation
# n: number of points
function gh_quadrature(μ, σ, T, n)
    # FastGaussQuadrature gives nodes x and weights w for weight e^{-x^2}
    x, w = gausshermite(n)
    
    # Change of variables from exp(-x^2) to exp(-z^2/2)/sqrt(2π):
    # z = sqrt(2)x  => dz = sqrt(2)dx
    # Integral[ f(z) exp(-z^2/2)/sqrt(2π) dz ] = Integral[ f(sqrt(2)x) exp(-x^2)/sqrt(π) dx ]
    res = 0.0
    for i in 1:n
        z = sqrt(2.0) * x[i]
        res += (w[i] / sqrt(π)) * integrand_core(μ + σ * z, T)
    end
    return res
end

# 4. Benchmarking Function
function run_benchmark()
    # Test cases: [μ, σ, T]
    test_cases = [
        (0.0,  0.1, 1.1),  # Unimodal, small variance (origin)
        (0.5,  0.2, 1.1),  # Near origin, narrow
        (2.0,  0.5, 1.2),  # Transition region
        (5.0,  0.5, 1.2),  # Saturated region
        (1.0,  2.0, 1.5),  # Very broad well
        (10.0, 1.0, 2.0)   # Deep plateau
    ]

    println("="^85)
    @printf("%-20s | %-15s | %-15s | %-15s\n", "Params (μ, σ, T)", "True Val", "GH-5 Error", "GH-11 Error")
    println("-"^85)

    for (μ, σ, T) in test_cases
        truth = ground_truth(μ, σ, T)
        val5  = gh_quadrature(μ, σ, T, 5)
        val11 = gh_quadrature(μ, σ, T, 11)

        err5  = abs(truth - val5)
        err11 = abs(truth - val11)

        param_str = @sprintf("(%.1f, %.1f, %.1f)", μ, σ, T)
        @printf("%-20s | %-15.10f | %-15.2e | %-15.2e\n", param_str, truth, err5, err11)
    end
    println("="^85)
end

run_benchmark()