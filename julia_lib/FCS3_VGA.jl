module FCS3_VGA

using ForwardDiff
using LinearAlgebra
using NLsolve
using SpecialFunctions
using Base: @kwdef

export ProblemParams3, Solution3, residuals_fcn3, solve_FCN3_Erf, populate_solution_fcn3

# ---------------------------------------------------------
# Structs
# ---------------------------------------------------------
@kwdef mutable struct ProblemParams3
    d::Float32 = 150.0f0
    κ::Float32 = 2.0f0
    ϵ::Float32 = 1.0f0
    P::Float32 = 600.0f0
    n1::Float32 = 1600.0f0
    n2::Float32 = 1600.0f0
    χ::Float32 = 1600.0f0
    b::Float32 = 4.0f0 / (3.0f0 * π)
    δ::Float32 = 1.0f0
    phi_nodes::Vector{Float64} = Float64[]
end

@kwdef mutable struct Solution3
    lJ1::Float64 = NaN
    lJ3::Float64 = NaN
    lH1::Float64 = NaN
    lH3::Float64 = NaN
    lK1::Float64 = NaN
    lK3::Float64 = NaN
    lT1::Float64 = NaN
    lT3::Float64 = NaN
    lV1::Float64 = NaN
    lWT::Float64 = NaN
    muW::Float64 = NaN
    sigW::Float64 = NaN
    muH1::Float64 = NaN
    sigH1::Float64 = NaN
    muH3::Float64 = NaN
    sigH3::Float64 = NaN
    learnability1::Float64 = NaN
    learnability3::Float64 = NaN
    kappa_eff::Float64 = NaN
end

# ---------------------------------------------------------
# Quadrature Helpers & Precomputations
# ---------------------------------------------------------
function gausslaguerre(n::Int)
    alpha = [Float64(2i - 1) for i in 1:n]
    beta = [Float64(i) for i in 1:(n-1)]
    T = SymTridiagonal(alpha, beta)
    vals, vecs = eigen(T)
    weights = vecs[1, :] .^ 2
    return vals, weights
end

const GL_NODES_12, GL_WEIGHTS_12 = gausslaguerre(12)

"""
    precompute_phi_nodes(d)

Precomputes the static perpendicular factor phi(t_i) for all 12 Gauss-Laguerre nodes.
This eliminates 24 transcendental log/exp evaluations per residual pass.
"""
function precompute_phi_nodes(d::Real)
    lWP = 1.0 / d
    phi = zeros(Float64, length(GL_NODES_12))
    for i in 1:length(GL_NODES_12)
        t_val = GL_NODES_12[i]
        log_val = -((d - 1.0) / 2.0) * log(1.0 + 4.0 * t_val * lWP) - (d^3 / 2.0) * log(1.0 + 4.0 * t_val * lWP^3)
        phi[i] = exp(log_val)
    end
    return phi
end

function ensure_phi_nodes!(params::ProblemParams3)
    if isempty(params.phi_nodes) || length(params.phi_nodes) != length(GL_NODES_12)
        params.phi_nodes = precompute_phi_nodes(params.d)
    end
    return params.phi_nodes
end

# ---------------------------------------------------------
# Core Residual Function (Optimized & Non-Allocating)
# ---------------------------------------------------------
function residuals_fcn3(x, P, chi, d, kappa, delta, epsilon, n1, n2, b, phi_nodes::Vector{Float64})
    lJ1, lJ3, lH1, lH3, lWT, mu_J1, mu_J3, mu_H1, mu_H3, mu_WT = x
    lWP = 1.0 / d

    J1_mean_sum, J3_mean_sum = 0.0, 0.0
    K11_sum, K1_mean_sum = 0.0, 0.0
    K33_sum, K3_mean_sum = 0.0, 0.0

    @inbounds for i in 1:12
        t_val = GL_NODES_12[i]
        w_val = GL_WEIGHTS_12[i]
        phi = phi_nodes[i]

        # J Kernel First Moments
        denom_J = 4.0 * t_val * lWT + 1.0
        exp_J = exp(-2.0 * t_val * mu_WT^2 / denom_J)
        term_J1_mean = (exp_J / denom_J^1.5) * mu_WT
        J1_mean_sum += w_val * (t_val^(-0.5)) * phi * term_J1_mean

        mu_prime = mu_WT / denom_J
        sig_prime2 = lWT / denom_J
        moment3 = mu_prime^3 + 3.0 * mu_prime * sig_prime2
        term_J3_mean = (exp_J / sqrt(denom_J)) * moment3
        J3_mean_sum += w_val * (t_val^0.5) * phi * term_J3_mean

        # K Kernel Projections
        denom_K1 = 4.0 * t_val * lH1 + 1.0
        denom_K3 = 4.0 * t_val * lH3 + 1.0
        exp_K = exp(-2.0 * t_val * mu_H1^2 / denom_K1)

        term_K11 = (exp_K / (denom_K1^1.5 * sqrt(denom_K3))) * (lH1 + mu_H1^2 / denom_K1)
        K11_sum += w_val * phi * term_K11
        term_K1_mean = (exp_K / (denom_K1^1.5 * sqrt(denom_K3))) * mu_H1
        K1_mean_sum += w_val * (t_val^(-0.5)) * phi * term_K1_mean

        term_K33 = (exp_K / (sqrt(denom_K1) * denom_K3^1.5)) * lH3
        K33_sum += w_val * phi * term_K33
        term_K3_mean = (exp_K / (sqrt(denom_K1) * denom_K3^1.5)) * mu_H3
        K3_mean_sum += w_val * (t_val^0.5) * phi * term_K3_mean
    end

    mu_J1_pred = (2.0 / π) * J1_mean_sum
    mu_J3_pred = -(8.0 / π) * J3_mean_sum

    lK1 = max((4.0 / π) * K11_sum - ((2.0 / π) * K1_mean_sum)^2, 1e-10)
    mu_K1 = (2.0 / π) * K1_mean_sum

    lK3 = max((4.0 / π) * K33_sum - (-(8.0 / π) * K3_mean_sum)^2, 1e-10)
    mu_K3 = -(8.0 / π) * K3_mean_sum

    # Dyson Auxiliary Terms
    TrSigma = lWT + lWP * (d - 1.0)
    EChh = lH1 + lH3 +
           (16.0 / (π * (1.0 + 2.0 * TrSigma)^3) * (15.0 * lWP^3)) * (d - 1.0) +
           (4.0 / (π * (1.0 + 2.0 * TrSigma)) * lWP) * (d - 1.0)
    gammaYh2 = (4.0 / π) / (1.0 + 2.0 * EChh)

    lV1 = - ((lH1 + 2.0 * mu_H1^2) / max(lJ1^2, 1e-10) - 1.0 / max(lJ1, 1e-10))
    lT1 = -(chi^2 / (kappa / P + lK1)^2 * delta)
    lT3 = -(chi^2 / (kappa / P + lK3)^2 * delta)

    # Dyson Residuals
    rj1 = lJ1 - (4.0 / (π * (1.0 + 2.0 * TrSigma)) * lWT)
    rj3 = lJ3 - ((16.0 / (π * (1.0 + 2.0 * TrSigma)^3) * (15.0 * lWT^3))) / 6.0
    rh1 = lH1 - 1.0 / (1.0 / lJ1 + gammaYh2 * lT1 / (n2 * chi))
    rh3 = lH3 - 1.0 / (1.0 / lJ3 + gammaYh2 * lT3 * epsilon^2 / (n2 * chi))
    rlWT = lWT - 1.0 / (d + delta * b * (n2 / n1) * lV1)

    rmu_j1 = mu_J1 - mu_J1_pred
    rmu_j3 = mu_J3 - mu_J3_pred

    # Zero-Allocation Scalar Derivatives for Means
    A_pot1 = -(chi^2 / (kappa / P + lK1)^2 * delta)
    A_pot3 = -(chi^2 / (kappa / P + lK3)^2 * delta) * epsilon^2

    function free_energy_mu_h1(mh1)
        sh1_2 = max(lH1 - mh1^2, 1e-10)
        sh3_2 = max(lH3 - mu_H3^2, 1e-10)
        prior_h = 0.5 * (mh1^2 + sh1_2) / max(lJ1, 1e-10)
        loss_pot = 0.0
        @inbounds for i in 1:12
            t_val = GL_NODES_12[i]
            w_val = GL_WEIGHTS_12[i]
            phi = phi_nodes[i]
            denom_h1 = 4.0 * t_val * sh1_2 + 1.0
            denom_h3 = 4.0 * t_val * sh3_2 + 1.0
            exp_h = exp(-2.0 * t_val * mh1^2 / denom_h1)
            term1 = (A_pot1 / (denom_h1^1.5 * sqrt(denom_h3))) * (sh1_2 + mh1^2 / denom_h1)
            term3 = (A_pot3 / (sqrt(denom_h1) * denom_h3^1.5)) * sh3_2
            loss_pot += w_val * phi * exp_h * (term1 + term3)
        end
        return prior_h + (4.0 / π) * loss_pot + log(1.0 + exp(-2.0 * mh1^2 / sh1_2))
    end

    function free_energy_mu_h3(mh3)
        sh1_2 = max(lH1 - mu_H1^2, 1e-10)
        sh3_2 = max(lH3 - mh3^2, 1e-10)
        prior_h = 0.5 * (mh3^2 + sh3_2) / max(lJ3, 1e-10)
        loss_pot = 0.0
        @inbounds for i in 1:12
            t_val = GL_NODES_12[i]
            w_val = GL_WEIGHTS_12[i]
            phi = phi_nodes[i]
            denom_h1 = 4.0 * t_val * sh1_2 + 1.0
            denom_h3 = 4.0 * t_val * sh3_2 + 1.0
            exp_h = exp(-2.0 * t_val * mu_H1^2 / denom_h1)
            term3 = (A_pot3 / (sqrt(denom_h1) * denom_h3^1.5)) * sh3_2
            loss_pot += w_val * phi * exp_h * term3
        end
        return prior_h + (4.0 / π) * loss_pot + log(1.0 + exp(-2.0 * mh3^2 / sh3_2))
    end

    function free_energy_mu_w(mw)
        sw_2 = max(lWT - mw^2, 1e-10)
        prior_w = (d / 2.0) * (mw^2 + sw_2)
        return prior_w + log(1.0 + exp(-2.0 * mw^2 / sw_2))
    end

    r_fe_mu_h1 = ForwardDiff.derivative(free_energy_mu_h1, mu_H1)
    r_fe_mu_h3 = ForwardDiff.derivative(free_energy_mu_h3, mu_H3)
    r_fe_mu_w = ForwardDiff.derivative(free_energy_mu_w, mu_WT)

    return [rj1, rj3, rh1, rh3, rlWT, rmu_j1, rmu_j3, r_fe_mu_h1, r_fe_mu_h3, r_fe_mu_w]
end

# ---------------------------------------------------------
# Adaptive Continuation Annealing Solver
# ---------------------------------------------------------
"""
    nlsolve_solver_fcn3(initial_guess; params, anneal_chi, anneal_P, anneal_steps, tol)

Optimized solver using adaptive tolerances across the annealing path to accelerate convergence.
"""
function nlsolve_solver_fcn3(initial_guess;
    params::ProblemParams3,
    anneal_chi::Bool=true,
    anneal_P::Bool=false,
    anneal_steps::Int=50,
    tol::Float64=1e-8,
    max_iter::Int=5000)

    phi_nodes = ensure_phi_nodes!(params)

    curr_x = copy(initial_guess)
    if length(curr_x) < 10
        curr_x = [curr_x..., zeros(10 - length(curr_x))...]
    end

    target_P = Float64(params.P)
    target_chi = Float64(params.χ)

    P_start = anneal_P ? Float64(params.d + 5.0) : target_P
    chi_start = anneal_chi ? 1e-6 : target_chi

    P_path = anneal_P ? exp.(range(log(P_start), log(target_P), length=anneal_steps)) : fill(target_P, anneal_steps)
    chi_path = anneal_chi ? exp.(range(log(chi_start), log(target_chi), length=anneal_steps)) : fill(target_chi, anneal_steps)

    for i in 1:anneal_steps
        c_curr = chi_path[i]
        p_curr = P_path[i]

        # Use looser tolerance on intermediate continuation steps, tightening on the final step
        step_tol = (i == anneal_steps) ? tol : max(1e-4, tol * 100.0)
        step_iter = (i == anneal_steps) ? max_iter : min(1000, max_iter)

        f!(F, x) = (F .= residuals_fcn3(
            abs.(x), p_curr, c_curr,
            params.d, params.κ, params.δ, params.ϵ,
            params.n1, params.n2, params.b, phi_nodes
        ))

        sol = nlsolve(f!, curr_x, xtol=step_tol, ftol=step_tol, iterations=step_iter)
        curr_x = abs.(sol.zero)
    end

    return curr_x
end

# ---------------------------------------------------------
# Solution Population
# ---------------------------------------------------------
function populate_solution_fcn3(sol_vec, params::ProblemParams3)
    if isnothing(sol_vec) || any(isnan.(sol_vec))
        return Solution3()
    end

    phi_nodes = ensure_phi_nodes!(params)
    lJ1, lJ3, lH1, lH3, lWT, mu_J1, mu_J3, mu_H1, mu_H3, mu_WT = sol_vec

    sigW = sqrt(max(lWT - mu_WT^2, 0.0))
    sigH1 = sqrt(max(lH1 - mu_H1^2, 0.0))
    sigH3 = sqrt(max(lH3 - mu_H3^2, 0.0))

    # Recompute K projections
    K11_sum, K33_sum = 0.0, 0.0
    @inbounds for i in 1:12
        t_val = GL_NODES_12[i]
        w_val = GL_WEIGHTS_12[i]
        phi = phi_nodes[i]

        denom_K1 = 4.0 * t_val * lH1 + 1.0
        denom_K3 = 4.0 * t_val * lH3 + 1.0
        exp_K = exp(-2.0 * t_val * mu_H1^2 / denom_K1)

        K11_sum += w_val * phi * (exp_K / (denom_K1^1.5 * sqrt(denom_K3))) * (lH1 + mu_H1^2 / denom_K1)
        K33_sum += w_val * phi * (exp_K / (sqrt(denom_K1) * denom_K3^1.5)) * lH3
    end

    lK1 = (4.0 / π) * K11_sum
    lK3 = (4.0 / π) * K33_sum
    kt = params.κ / params.P

    lT1 = -(params.χ^2 / (kt + lK1)^2 * params.δ)
    lT3 = -(params.χ^2 / (kt + lK3)^2 * params.δ)
    lV1 = - ((lH1 + 2.0 * mu_H1^2) / max(lJ1^2, 1e-10) - 1.0 / max(lJ1, 1e-10))

    return Solution3(
        lJ1=lJ1, lJ3=lJ3, lH1=lH1, lH3=lH3,
        lK1=lK1, lK3=lK3, lT1=lT1, lT3=lT3, lV1=lV1, lWT=lWT,
        muW=mu_WT, sigW=sigW,
        muH1=mu_H1, sigH1=sigH1,
        muH3=mu_H3, sigH3=sigH3,
        learnability1=lK1 / (lK1 + kt),
        learnability3=lK3 / (lK3 + kt),
        kappa_eff=params.κ
    )
end

function solve_FCN3_Erf(params::ProblemParams3, initial_guess;
    anneal_chi::Bool=true, anneal_P::Bool=false, anneal_steps::Int=50, tol::Float64=1e-8)

    sol_vec = nlsolve_solver_fcn3(
        initial_guess;
        params=params,
        anneal_chi=anneal_chi,
        anneal_P=anneal_P,
        anneal_steps=anneal_steps,
        tol=tol
    )

    return populate_solution_fcn3(sol_vec, params)
end

end # module