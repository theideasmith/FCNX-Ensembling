using SpecialFunctions
# 1. The "Voigt-based" expectation of x^2 / (B + 2x^2)
function v_eff(μ, σ, B)
    
    q = sqrt(B / 2)
    # The complex argument for the Faddeeva function
    z = complex(-μ, q) / (σ * sqrt(2))

    # Faddeeva w(z) = erfcx(-im*z)
    w_z = erfcx(-im * z)

    # The analytical result
    val = 0.5 - (q * sqrt(π) / (2 * σ * sqrt(2))) * real(w_z)
    return val
end

# 2. Jensen's Lower Bound on GMM Entropy
function entropy_bound(μs, σs, πs)
    K = length(πs)
    H_bound = 0.0

    for i in 1:K
        # Sum of overlaps for component i
        overlap_sum = 0.0
        for j in 1:K
            # Overlap of two Gaussians (integral of N_i * N_j)
            # z_ij = N(μ_i; μ_j, σ_i^2 + σ_j^2)
            var_sum = σs[i]^2 + σs[j]^2
            dist_sq = (μs[i] - μs[j])^2
            z_ij = exp(-dist_sq / (2 * var_sum)) / sqrt(2 * π * var_sum)

            overlap_sum += πs[j] * z_ij
        end
        H_bound -= πs[i] * log(overlap_sum)
    end
    return H_bound
end

# Helper: Softmax for mixing weights
softmax(x) = exp.(x) ./ sum(exp.(x))

function free_energy(params, A, B)
    K = length(params) ÷ 3
    μs = params[1:K]
    log_σs = params[(K+1):2K]
    ws = params[(2K+1):3K]

    σs = exp.(log_σs)
    πs = softmax(ws)

    # 1. Expected Action (Energy)
    # Assuming S(x) = A * x^2/(B+2x^2) 
    energy = 0.0
    for k in 1:K
        energy += πs[k] * A * v_eff(μs[k], σs[k], B)
        # Add other terms like confinement if needed:
        # energy += πs[k] * (μs[k]^2 + σs[k]^2) / 2
    end

    # 2. Negative Entropy (using the bound)
    neg_entropy = -entropy_bound(μs, σs, πs)

    return energy + neg_entropy
end

function v_eff(μ, σ, A, B)


    q = sqrt(B / 2)
    z = complex(-μ, q) / (σ * sqrt(2))

    # Faddeeva w(z) = erfcx(-im*z)
    w_z = erfcx(-im * z)

    # Expected value of x^2 / (a + 2x^2)
    return A * 0.5 - A * (q * sqrt(π) / (2 * σ * sqrt(2))) * real(w_z)
end