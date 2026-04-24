# # ============================================================================
# #  FCS2Erf_Cubic.jl — Fixed-point Consistency Solver for FCN2 (erf activations)
# # ============================================================================
# # Purpose:
# #   Numerical solvers for 2-layer fully connected networks with erf activations,
# #   including cubic eigenvalue predictions. Simplified from FCS.jl by removing
# #   the second hidden layer.
# #
# # Architecture:
# #   Input (d) -> Hidden (n1, erf) -> Output (linear)
# #   f(x) = A · erf(W0 · x)
# #
# # Variables:
# #   lJ1, lJ3: Hidden layer kernel eigenvalues (linear and cubic features)
# #   lWT: Target direction weight variance
# #
# # Exports:
# #   - residuals_fcn2
# #   - gradient_descent_solver_fcn2
# #   - nlsolve_solver_fcn2
# #   - solve_FCN2_Erf
# #   - populate_solution_fcn2
# #
# # Usage:
# #   using FCS2Erf_Cubic
# #   params = ProblemParams2(d=50.0, κ=1.0, ϵ=0.03, P=600.0, n1=800.0, χ=80.0)
# #   guess = [1.0, 1.0, 0.02]  # [lJ1, lJ3, lWT]
# #   sol = solve_FCN2_Erf(params, guess; verbose=true)
# #
# module FCS2Erf_Cubic

# export residuals_fcn2, gradient_descent_solver_fcn2, nlsolve_solver_fcn2, solve_FCN2_Erf, populate_solution_fcn2, ProblemParams2, Solution2

# using ForwardDiff
# using LinearAlgebra
# using NLsolve
# using Base: @kwdef

# @kwdef mutable struct ProblemParams2
#     d::Float32
#     κ::Float32
#     ϵ::Float32
#     P::Float32
#     n1::Float32
#     χ::Float32
#     b::Float32 = 4/(3*π)
#     δ::Float32 = 0.0
# end

# @kwdef mutable struct Solution2
#     lJ1::Float64 = NaN
#     lJ3::Float64 = NaN
#     lK1::Float64 = NaN
#     lK3::Float64 = NaN
#     lWT::Float64 = NaN
#     learnability1::Float64 = NaN
#     learnability3::Float64 = NaN
# end

# # Helper functions
# lWP(d::Real) = 1 / d

# function TrSigma(lWT::Real, d::Real)
#     lWP_val = lWP(d)
#     return lWT + lWP_val * (d - 1)
# end

# # -------------------------
# # Residual function for FCN2
# # -------------------------
# # Given x = [lJ1, lJ3, lWT], returns residuals of the 3 equations
# function residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
#     lJ1, lJ3, lWT = x
#     lWP_val = 1.0 / d
    
#     TrSigma_val = lWT + lWP_val * (d - 1)
    
#     # Output kernel eigenvalues (K = variance of output weighted by readout)
#     # For FCN2: K relates directly to J through readout variance (chi scaling)
#     # γ_Yh = (4/π) / (1 + 2·E[h·h])
#     # For FCN2, E[h·h] ≈ TrΣ in the NNGP limit
#     gammaYh = (4 / π) / (1 + 2 * TrSigma_val)
    
#     # Training signal (learning rates from kernel theory)
#     lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
#     lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)
    
#     # Equations of state
#     # J1: Linear feature kernel from first layer
#     rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
    
#     # J3: Cubic feature kernel from first layer
#     rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3)) /6.0
    
#     # WT: Target weight variance with training feedback
#     # Training modifies weights: Σ_T = Σ_0 + δ·training_contribution
#     # For FCN2: lV1 = effect of training on kernel = lT1·gammaYh/(n1·chi)

#     rlWT = lWT -  1 / (d + gammaYh * (lT1)/ ( n1 * chi)) 
    
#     return [rj1, rj3, rlWT]
# end

# # -------------------------
# # Residual function with hardcoded TrSigma
# # -------------------------
# # Given x = [lJ1, lJ3], with TrSigma fixed to hardcoded value
# function residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
#     lJ1, lJ3 = x
#     lWP_val = 1.0 / d
    
#     # Compute lWT from fixed TrSigma: TrSigma = lWT + (d-1)*lWP => lWT = TrSigma - (d-1)*lWP
#     lWT = TrSigma_fixed - (d - 1) * lWP_val
#     TrSigma_val = TrSigma_fixed
    
#     # Output kernel eigenvalues
#     gammaYh = (4 / π) / (1 + 2 * TrSigma_val)
    
#     # Training signal
#     lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
#     lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)
    
#     # Equations of state (J equations only, lWT is fixed)
#     rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
#     rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3)) / 6.0
    
#     return [rj1, rj3]
# end

# # -------------------------
# # Gradient Descent Solver
# # -------------------------
# function gradient_descent_solver_fcn2(initial_guess;
#     chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
#     P=nothing,
#     TrSigma_fixed=nothing,
#     lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

#     if P === nothing
#         P = d^1.2
#     end
#     loss = nothing

#     if TrSigma_fixed === nothing
#         # Standard 3-variable solve
#         x = copy(initial_guess)
#         for iter in 1:max_iter
#             res = residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
#             loss = sum(res .^ 2)
            
#             grad = ForwardDiff.gradient(x -> sum(residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b) .^ 2), x)
#             x -= lr * grad
            
#             if loss < tol
#                 if verbose
#                     println("Converged in $iter iterations, loss = $loss")
#                 end
#                 return x
#             end
#         end
#     else
#         # 2-variable solve with fixed TrSigma
#         x = copy(initial_guess[1:2])  # Use only [lJ1, lJ3]
#         lWP_val = 1.0 / d
#         lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
#         for iter in 1:max_iter
#             res = residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
#             loss = sum(res .^ 2)
            
#             grad = ForwardDiff.gradient(x -> sum(residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed) .^ 2), x)
#             x -= lr * grad
            
#             if loss < tol
#                 if verbose
#                     println("Converged in $iter iterations, loss = $loss (fixed TrSigma = $TrSigma_fixed, lWT = $lWT_from_TrSigma)")
#                 end
#                 return [x[1], x[2], lWT_from_TrSigma]  # Return with computed lWT
#             end
#         end
#     end

#     if verbose
#         println("Reached max iterations, final loss = $loss")
#     end
#     if TrSigma_fixed === nothing
#         return x
#     else
#         lWP_val = 1.0 / d
#         lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
#         return [x[1], x[2], lWT_from_TrSigma]
#     end
# end

# # -------------------------
# # NLsolve Solver
# # -------------------------
# function nlsolve_solver_fcn2(initial_guess;
#     anneal=false,
#     chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
#     P=nothing, anneal_steps=30000,
#     TrSigma_fixed=nothing,
#     lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)
    
#     if P === nothing
#         P = d^1.2
#     end
#     print("Solving with chi: $chi, d: $d, kappa: $kappa, delta: $delta, epsilon: $epsilon, n1: $n1, P: $P, anneal: $anneal\n")
#     result = nothing
    
#     if TrSigma_fixed === nothing
#         # Standard 3-variable solve
#         x = copy(initial_guess)
        
#         function res_func!(F, x, c)
#             F[:] = residuals_fcn2(x, P, c, d, kappa, delta, epsilon, n1, b)
#         end

#         if anneal
#             chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
#             prev_sol = x
#             for (j, chit) in enumerate(chi_anneal_list)
#                 f1! = (F, x) -> res_func!(F, x, chit)
#                 try
#                     sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
#                     if (j == anneal_steps)
#                         result = sol.zero
#                     end
#                     prev_sol = sol.zero
#                 catch e
#                     println("Error during annealing at chi=$chit")
#                     showerror(stdout, e, catch_backtrace())
#                 end
#             end
#         else
#             f2!(F, x) = res_func!(F, x, chi)
#             try
#                 result = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
#             catch e
#                 println("Error during solve")
#                 showerror(stdout, e, catch_backtrace())
#                 result = nothing
#             end
#         end
#     else
#         # 2-variable solve with fixed TrSigma
#         x = copy(initial_guess[1:2])  # Use only [lJ1, lJ3]
#         lWP_val = 1.0 / d
#         lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
        
#         function res_func_fixed!(F, x, c)
#             F[:] = residuals_fcn2_fixed_TrSigma(x, P, c, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
#         end

#         if anneal
#             chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
#             prev_sol = x
#             for (j, chit) in enumerate(chi_anneal_list)
#                 f1! = (F, x) -> res_func_fixed!(F, x, chit)
#                 try
#                     sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
#                     if (j == anneal_steps)
#                         result = [sol.zero[1], sol.zero[2], lWT_from_TrSigma]
#                     end
#                     prev_sol = sol.zero
#                 catch e
#                     println("Error during annealing at chi=$chit with fixed TrSigma=$TrSigma_fixed")
#                     showerror(stdout, e, catch_backtrace())
#                 end
#             end
#         else
#             f2!(F, x) = res_func_fixed!(F, x, chi)
#             try
#                 sol_2var = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
#                 result = [sol_2var[1], sol_2var[2], lWT_from_TrSigma]
#             catch e
#                 println("Error during solve with fixed TrSigma=$TrSigma_fixed")
#                 showerror(stdout, e, catch_backtrace())
#                 result = nothing
#             end
#         end
#     end

#     return result
# end

# # -------------------------
# # Populate Solution
# # -------------------------
# function populate_solution_fcn2(sol_vec, params::ProblemParams2)
#     if isnothing(sol_vec) || length(sol_vec) != 3
#         return Solution2()
#     end
    
#     lJ1, lJ3, lWT = sol_vec
#     lWP_val = lWP(params.d)
#     TrSigma_val = TrSigma(lWT, params.d)
    
#     gammaYh = (4 / π) / (1 + 2 * TrSigma_val)
    
#     lK1 = gammaYh * lJ1 
#     lK3 = gammaYh * lJ3 
    
#     learnability1 = lK1 / (lK1 + params.κ / params.P)
#     learnability3 = lK3 / (lK3 + params.κ / params.P)
    
#     return Solution2(
#         lJ1=lJ1,
#         lJ3=lJ3,
#         lK1=lK1,
#         lK3=lK3,
#         lWT=lWT,
#         learnability1=learnability1,
#         learnability3=learnability3
#     )
# end

# # -------------------------
# # Main Solver Interface
# # -------------------------
# function solve_FCN2_Erf(problem_params::ProblemParams2, initial_guess;
#     lr=1e-3, max_iter=5000, tol=1e-8, verbose=false, use_anneal=true, TrSigma_fixed=nothing)

#     sol_vec = nlsolve_solver_fcn2(
#         initial_guess,
#         chi=problem_params.χ,
#         d=problem_params.d,
#         kappa=problem_params.κ,
#         delta=problem_params.δ,
#         epsilon=problem_params.ϵ,
#         n1=problem_params.n1,
#         b=problem_params.b,
#         P=problem_params.P,
#         TrSigma_fixed=TrSigma_fixed,
#         lr=lr,
#         max_iter=max_iter,
#         tol=tol,
#         verbose=verbose,
#         anneal=use_anneal
#     )
    
#     solution = populate_solution_fcn2(sol_vec, problem_params)
#     return solution
# end

# end # module


# ============================================================================
#  FCS2Erf_Cubic.jl — Fixed-point Consistency Solver for FCN2 (erf activations)
# ============================================================================
# Purpose:
#   Numerical solvers for 2-layer fully connected networks with erf activations.
#   Uses Log-space transformation and P/Chi annealing for high stability at 
#   large P and large Chi.
#
# Architecture:
#   Input (d) -> Hidden (n1, erf) -> Output (linear)
#
# Metadata:
#   Project: FCNX-Ensembling
#   Date: 2025-12-02
#
module FCS2Erf_Cubic

export residuals_fcn2, nlsolve_solver_fcn2, gradient_descent_solver_fcn2, solve_FCN2_Erf, populate_solution_fcn2, ProblemParams2, Solution2

using ForwardDiff
using LinearAlgebra
using NLsolve
using Base: @kwdef

@kwdef mutable struct ProblemParams2
    d::Float32
    κ::Float32
    ϵ::Float32
    P::Float32
    n1::Float32
    χ::Float32
    b::Float32 = 4 / (3 * π)
    δ::Float32 = 0.0 # Training contribution parameter
end

@kwdef mutable struct Solution2
    lJ1::Float64 = NaN
    lJ3::Float64 = NaN
    lK1::Float64 = NaN
    lK3::Float64 = NaN
    lWT::Float64 = NaN
    learnability1::Float64 = NaN
    learnability3::Float64 = NaN
    kappa_eff::Float64 = NaN
end

# -------------------------
# Helpers
# -------------------------
lWP(d::Real) = 1 / d

function TrSigma(lWT::Real, d::Real)
    lWP_val = lWP(d)
    return lWT + lWP_val * (d - 1)
end

function compute_effective_ridge(kappa_bare::Real, lambdas::AbstractVector{<:Real}, P::Real;
    max_iter::Int=200, tol::Real=1e-10)
    λ = [x for x in lambdas if isfinite(x) && x > 0]
    if isempty(λ)
        return float(kappa_bare)
    end

    κ = float(kappa_bare)
    for _ in 1:max_iter
        correction = sum((κ / P .* λ) ./ (κ / P .+ λ))
        κ_new = float(kappa_bare) + correction
        if abs(κ_new - κ) < tol * max(1.0, abs(κ))
            return κ_new
        end
        κ = κ_new
    end
    return κ
end

# -------------------------
# Core Residual Functions
# -------------------------

function residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
    lJ1, lJ3, lWT = x
    lWP_val = 1.0 / d
    TrSigma_val = lWT + lWP_val * (d - 1)

    gammaYh = (4 / π) / (1 + 2 * TrSigma_val)

    # Training signals
    lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
    lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)

    # Residuals
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
    rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3)) / 6.0
    rlWT = lWT - 1 / (d + gammaYh * (lT1) / (n1 * chi))

    return [rj1, rj3, rlWT]
end

function residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
    lJ1, lJ3 = x
    lWP_val = 1.0 / d
    lWT = TrSigma_fixed - (d - 1) * lWP_val
    TrSigma_val = TrSigma_fixed

    gammaYh = (4 / π) / (1 + 2 * TrSigma_val)

    lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
    lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)

    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
    rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3)) / 6.0

    return [rj1, rj3]
end

# -------------------------
# Gradient Descent Solver (Fallback)
# -------------------------
function gradient_descent_solver_fcn2(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
    P=nothing, lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    if P === nothing
        P = d^1.2
    end
    # Note: Does not use log-space or annealing, purely for simple optimization
    x = copy(initial_guess)
    loss = Inf
    for iter in 1:max_iter
        res = residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
        loss = sum(res .^ 2)
        if loss < tol
            break
        end

        grad = ForwardDiff.gradient(xt -> sum(residuals_fcn2(xt, P, chi, d, kappa, delta, epsilon, n1, b) .^ 2), x)
        x -= lr * grad
    end
    return x
end

# -------------------------
# NLsolve Solver (Log-based + Annealing)
# -------------------------

function nlsolve_solver_fcn2(initial_guess;
    anneal=false,        # chi annealing
    anneal_P=true,       # P annealing
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
    P=nothing, anneal_steps=50,
    TrSigma_fixed=nothing,
    max_iter=5000, tol=1e-8, verbose=false)

    target_P = (P === nothing) ? d^1.2 : P
    target_chi = chi

    # Path setup
    P_start = d + 5.0
    chi_start = 1e-8
    P_path = anneal_P ? exp.(range(log(P_start), log(target_P), length=anneal_steps)) : fill(target_P, anneal_steps)
    chi_path = anneal ? exp.(range(log(chi_start), log(target_chi), length=anneal_steps)) : fill(target_chi, anneal_steps)

    # Solve in Log Space: prevents negative eigenvalues
    current_x_log = log.(initial_guess .+ 1e-12)
    if TrSigma_fixed !== nothing
        current_x_log = current_x_log[1:2]
    end

    result_phys = nothing

    for i in 1:anneal_steps
        p_curr = P_path[i]
        c_curr = chi_path[i]

        function res_func_log!(F, x_log)
            x_phys = exp.(x_log)
            if TrSigma_fixed === nothing
                F[:] = residuals_fcn2(x_phys, p_curr, c_curr, d, kappa, delta, epsilon, n1, b)
            else
                F[:] = residuals_fcn2_fixed_TrSigma(x_phys, p_curr, c_curr, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
            end
        end

        try
            sol = nlsolve(res_func_log!, current_x_log, xtol=tol, ftol=tol, iterations=max_iter, show_trace=(verbose && i == anneal_steps))
            current_x_log = sol.zero

            if i == anneal_steps
                phys_zero = exp.(sol.zero)
                if TrSigma_fixed === nothing
                    result_phys = phys_zero
                else
                    lWT_fixed = TrSigma_fixed - (d - 1) * (1.0 / d)
                    result_phys = [phys_zero[1], phys_zero[2], lWT_fixed]
                end
            end
        catch e
            @error "Solver failed during annealing step $i (P=$p_curr, chi=$c_curr)"
            return nothing
        end
    end
    return result_phys
end

# -------------------------
# Populate Solution
# -------------------------
function populate_solution_fcn2(sol_vec, params::ProblemParams2)
    if isnothing(sol_vec) || length(sol_vec) != 3
        return Solution2()
    end

    lJ1, lJ3, lWT = sol_vec
    lWP_val = lWP(params.d)
    TrSigma_val = TrSigma(lWT, params.d)

    gammaYh = (4 / π) / (1 + 2 * TrSigma_val)

    lK1 = gammaYh * lJ1
    lK3 = gammaYh * lJ3

    k_term = params.κ / params.P
    learnability1 = lK1 / (lK1 + k_term)
    learnability3 = lK3 / (lK3 + k_term)

    return Solution2(
        lJ1=lJ1, lJ3=lJ3, lK1=lK1, lK3=lK3, lWT=lWT,
        learnability1=learnability1, learnability3=learnability3
    )
end
# -------------------------
# Main Solver Interface
# -------------------------
function solve_FCN2_Erf(problem_params::ProblemParams2, initial_guess;
    max_iter=5000, 
    tol=1e-8, 
    verbose=false, 
    lr=1e-3,             # Added to prevent MethodError
    use_anneal=true,     # Legacy name for chi annealing
    anneal_chi=nothing,  # New name
    anneal_P=true, 
    anneal_steps=3000, 
    TrSigma_fixed=nothing,
    effective_ridge::Bool=false)

    # Maintain backward compatibility: 
    # if anneal_chi isn't set, use the value of use_anneal
    actual_anneal_chi = isnothing(anneal_chi) ? use_anneal : anneal_chi

    function solve_with_kappa(kappa_value)
        return nlsolve_solver_fcn2(
            initial_guess;
            chi=problem_params.χ,
            d=problem_params.d,
            kappa=kappa_value,
            delta=problem_params.δ,
            epsilon=problem_params.ϵ,
            n1=problem_params.n1,
            b=problem_params.b,
            P=problem_params.P,
            TrSigma_fixed=TrSigma_fixed,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            anneal=actual_anneal_chi,
            anneal_P=anneal_P,
            anneal_steps=anneal_steps
        )
    end

    kappa_used = float(problem_params.κ)
    sol_vec = solve_with_kappa(kappa_used)
    if isnothing(sol_vec)
        return Solution2()
    end

    if effective_ridge
        prelim_params = ProblemParams2(
            d=problem_params.d,
            κ=Float32(kappa_used),
            ϵ=problem_params.ϵ,
            P=problem_params.P,
            n1=problem_params.n1,
            χ=problem_params.χ,
            b=problem_params.b,
            δ=problem_params.δ
        )
        prelim_solution = populate_solution_fcn2(sol_vec, prelim_params)
        kappa_used = compute_effective_ridge(kappa_used, [prelim_solution.lK1, prelim_solution.lK3], problem_params.P)

        sol_vec_eff = solve_with_kappa(kappa_used)
        if !isnothing(sol_vec_eff)
            sol_vec = sol_vec_eff
        end
    end

    params_eff = ProblemParams2(
        d=problem_params.d,
        κ=Float32(kappa_used),
        ϵ=problem_params.ϵ,
        P=problem_params.P,
        n1=problem_params.n1,
        χ=problem_params.χ,
        b=problem_params.b,
        δ=problem_params.δ
    )
    populated = populate_solution_fcn2(sol_vec, params_eff)
    populated.kappa_eff = kappa_used
    return populated
end

end # module