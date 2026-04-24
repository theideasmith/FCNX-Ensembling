# ============================================================================

#  FCS.jl — Fixed-point Consistency Solver for FCN3 (erf activations)
# ============================================================================
# Purpose:
#   Numerical solvers and helpers that compute mean-field / fixed-point
#   predictions used by the Python `Experiment` wrapper. This module exposes
#   routines to solve the nonlinear FCS equations (via `NLsolve` or a simple
#   gradient-descent fallback), to compute kernel-derived eigenvalues and
#   learnability metrics, and to sweep parameter ranges.
#
# Exports (most important):
#   - residuals
#   - gradient_descent_solver
#   - compute_lK_ratio
#   - sweep_learnabilities
#   - nlsolve_solver
#   - solve_FCN3_Erf
#   - populate_solution
#
# Quick example (from Julia REPL):
#   using Pkg; Pkg.instantiate()
#   using FCS
#   params = FCS.ProblemParams(d=50.0f0, κ=1.0f0, ϵ=0.03f0, P=4000.0f0, n=200.0f0, χ=5.0f0)
#   guess = [1.0, 1.0, 1.0, 1.0]
#   sol = FCS.solve_FCN3_Erf(params, guess; verbose=true)
#
# Dependencies:
#   ForwardDiff, NLsolve, LinearAlgebra, Plots (optional), Colors
#
# Metadata:
#   Project: FCNX-Ensembling
#   Path:    julia_lib/FCS.jl
#   Author:  repo maintainers
#   Date:    2025-12-02
#
module FCS

export residuals, gradient_descent_solver, compute_lK_ratio, sweep_learnabilities, nlsolve_solver

using ForwardDiff
using LinearAlgebra
using NLsolve







is_physical(sol) = all(sol .> 0)
using Base: @kwdef




@kwdef mutable struct ProblemParams
    d::Float32
    κ::Float32
    ϵ::Float32
    P::Float32
    n1::Float32
    n2::Float32
    χ::Float32
    b::Float32  = 4/(3*π)
end 

@kwdef mutable struct Solution
    lJ1::Float64 = NaN
    lJ3::Float64 = NaN
    lH1::Float64 = NaN
    lH3::Float64 = NaN
    lK1::Float64 = NaN
    lK3::Float64 = NaN
    lT1::Float64 = NaN
    lT3::Float64 = NaN
    lV1::Float64 = NaN
    lV3::Float64 = NaN
    lWT::Float64 = NaN
    learnability1::Float64 = NaN
    learnability3::Float64 = NaN
    kappa_eff::Float64 = NaN
end



# basic helpers
function lV(sol::Solution)
    lV1 = - (sol.lH1 / sol.lJ1^2 - 1 / sol.lJ1)
    lV3 = - (sol.lH3 / sol.lJ3^2 - 1 / sol.lJ3)
    return lV1, lV3
end

lWP(d::Real) = 1 / d

function lWT(sol::Solution; n1::Real=1.0, n2::Real=1.0, d::Real=1.0, delta::Real=1.0, b::Real=4/(3π))
    lV1, _ = lV(sol)
    return 1 / (d + delta * b * n2  * lV1 / n1)
end

function TrSigma(sol::Solution; n1::Real=1.0, n2::Real=1.0, d::Real=1.0, delta::Real=1.0, b::Real=4/(3π))
    return lWT(sol; n1=n1, n2=n2, d=d, delta=delta, b=b) + lWP(d) * (d - 1)
end

function EChh(sol::Solution; n1::Real=1.0, n2::Real=1.0, d::Real=1.0, delta::Real=1.0, b::Real=4/(3π))
    ts = TrSigma(sol; n1=n1, n2=n2, d=d, delta=delta, b=b)
    lp = lWP(d)
    return sol.lH1 + sol.lH3 +
        (16 / (π * (1 + 2 * ts)^3) * (15 * lp^3)) * (d - 1) +
        (4 / (π * (1 + 2 * ts)) * lp) * (d - 1)
end

gammaYh2(sol::Solution; n1::Real=1.0, n2::Real=1.0, d::Real=1.0, delta::Real=1.0, b::Real=4/(3π)) =
    (4 / π) / (1 + 2 * EChh(sol; n1=n1, n2=n2, d=d, delta=delta, b=b))

# compute lK values
function lK(sol::Solution, P; n1::Real=1.0, n2::Real=1.0, chi::Real=1.0, d::Real=1.0, delta::Real=1.0, kappa::Real=1.0, epsilon::Real=1.0, b::Real=4/(3π))
    gy = gammaYh2(sol; n1=n1, n2=n2, d=d, delta=delta, b=b)
    return gy * sol.lH1, gy * sol.lH3
end

# compute lT values (uses same formula as residuals)
function lT(sol::Solution, P; n1::Real=1.0, n2::Real=1.0, chi::Real=1.0, d::Real=1.0, delta::Real=1.0, kappa::Real=1.0, epsilon::Real=1.0, b::Real=4/(3π))
    lK1, lK3 = lK(sol, P; n1=n1, n2=n2, chi=chi, d=d, delta=delta, kappa=kappa, epsilon=epsilon, b=b)
    t1 = -chi^2 / (kappa / P + lK1)^2 * delta
    t3 = -chi^2 / (kappa / P + lK3)^2 * delta
    return t1, t3
end

# convenience conversions
to_vector(sol::Solution) = [sol.lJ1, sol.lJ3, sol.lH1, sol.lH3]

function get_eigenvalues(i0,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, n2=1.0, b=1.0,
    P=nothing)
    Tf = 6_000_000
    lr = 1e-4
    return gradient_descent_solver(
        i0,
        chi=chi, d=d, kappa=1.0, delta=delta,
        epsilon=ϵ, n1=n1, n2=n2, b=4 / (3 * π),
        P=P, lr=lr, max_iter=Tf, verbose=true
    )
end


# -------------------------
# Residual helper functions
# -------------------------

# Legacy residuals function (current implementation, without 1/6 normalization)
function residuals_legacy(x, P, chi, d, kappa, delta, epsilon, n1, n2, b)
    lJ1, lJ3, lH1, lH3, lWT = x  # current variables
    lWP =  1.0 / d
    
    TrSigma = lWT + lWP * (d - 1)

    # Conjugate inter-layer discrepancies by the inverse kernels
    # J : Downstream ↦ Upstream
    # H : Upstream ↦ Preactivation
    # J^-1 : Preimage of H ↦ Preimage of J
    # V: Discrepancy between H1 and J1 in the preimage of J1
    # FCS is the equations of state for FCN3 with erf activations
    lV1 =    - lJ1^(-1) * ( lH1 - lJ1) * lJ1^(-1)
    lV3 =    - lJ3^(-1) * (lH3  - lJ3) * lJ3^(-1)
    b = 4 / (π) * 1/ ( 1 + 2 * TrSigma)

    EChh = lH1 + lH3 +
        (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3))  * (d - 1) +
        (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)

    gammaYh2 = (4 / π) / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3

    lT1 = -(chi^2 / (kappa / P + lK1)^2 * delta) - chi^2 * kappa / (P * chi) * lK1 / (lK1 + kappa / P)
    lT3 =   -(chi^2 / (kappa / P + lK3)^2 * delta) - chi^2 * kappa / (P * chi) * lK3 / (lK3 + kappa / P)

    # Residuals
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma)) * lWT)
    rh1 = lH1 - 1 / (1 / lJ1 + gammaYh2 * lT1 / (n2 * chi))
    rh3 = lH3 - ( 1 / (1 / lJ3 +  gammaYh2 * lT3 * epsilon^2 / (n2 * chi)) )
    rj3 = lJ3 - ( (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)) ) / 6
    rlWT = lWT - 1 / (d + delta * b * (n2 / n1) * lV1)
    return [rj1, rj3, rh1, rh3, rlWT]
end

# Normalized residuals function (with 1/6 factor applied to all l3 eigenvalues)
function residuals_normalized(x, P, chi, d, kappa, delta, epsilon, n1, n2, b)
    lJ1, lJ3, lH1, lH3, lWT = x  # current variables
    lWP =  1.0 / d
    
    # Apply 1/6 normalization to all l3 eigenvalues
    lJ3 = lJ3 
    lH3 = lH3 
    
    TrSigma = lWT + lWP * (d - 1)

    # Conjugate inter-layer discrepancies by the inverse kernels
    lV1 =    - lJ1^(-1) * ( lH1 - lJ1) * lJ1^(-1)   # Apply 1/6 normalization to lV1 as well, since it depends on lH1 and lJ1
    lV3 =    - lJ3^(-1) * (lH3  - lJ3) * lJ3^(-1) 
    b = 4 / (π) * 1/ ( 1 + 2 * TrSigma)

    EChh = lH1 + lH3 +
        (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3))  * (d - 1) +
        (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)

    gammaYh2 = (4 / π) / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1 
    lK3 = gammaYh2 * lH3 

    lT1 = -(chi^2 / (kappa / P + lK1)^2 * delta) - chi^2 * kappa / (P * chi) * lK1 / (lK1 + kappa / P)

    lT3 =   -(chi^2 / (kappa / P + lK3)^2 * delta) - chi^2 * kappa / (P * chi) * lK3 / (lK3 + kappa / P)
    # lT3 = lT3 / 6.0  # Apply 1/6 normalization to lT3 as well, since it depends on lK3 which depends on lH3

    # Residuals
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma)) * lWT)
    rh1 = lH1 - 1 / (1 / lJ1 + gammaYh2 * lT1 / (n2 * chi))
    rh3 = lH3 - ( 1 / (1 / lJ3 +  gammaYh2 * lT3 * epsilon^2 / (n2 * chi)) ) 
    rj3 = lJ3 - ( (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWT^3)) ) / 6.0
    rlWT = lWT - 1 / (d + delta * b * (n2 / n1) * lV1)
    return [rj1, rj3, rh1, rh3, rlWT]
end

# Main residuals dispatcher function
# Given a guess x = [lH1, lJ1, lH3, lJ3], returns the residuals of the 4 equations
# If normalized=true (default), applies 1/6 factor to l3 eigenvalues
# If normalized=false, uses legacy implementation
function residuals(x, P, chi, d, kappa, delta, epsilon, n1, n2, b; normalized::Bool=false)
    normalized = true
    if normalized
        return residuals_normalized(x, P, chi, d, kappa, delta, epsilon, n1, n2, b)
    else
        return residuals_legacy(x, P, chi, d, kappa, delta, epsilon, n1, n2, b)
    end
end

# -------------------------
# Gradient Descent Solver
# -------------------------
function gradient_descent_solver(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, n2=1.0, b=1.0,
    P=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false, normalized::Bool=true)

    x = copy(initial_guess)
    if P === nothing
        P = d^1.2  # default
    end
    loss = nothing

    for iter in 1:max_iter
        # Compute residuals
        res = residuals(x, P, chi, d, kappa, delta, epsilon, n1, n2, b; normalized=normalized)

        # Loss = sum of squares
        loss = sum(res .^ 2) + sum(x .^ 2)

        # Gradient of loss wrt x
        grad = ForwardDiff.gradient(x -> sum(residuals(x, P, chi, d, kappa, delta, epsilon, n1, n2, b; normalized=normalized) .^ 2), x)

        # Update step
        x -= lr * grad

        # Convergence check
        if loss < tol
            if verbose
                println("Converged in $iter iterations, loss = $loss")
            end
            return x
        end

        # if verbose && iter % 500 == 0
        #     println("Iter $iter: loss = $loss")
        # end
    end

    if verbose
        println("Reached max iterations, final loss = $loss")
    end
    return x
end


function compute_lK_ratio(sol, P, n1, n2, chi, d, delta, kappa, epsilon, b)
    if sol === nothing
        return (NaN, NaN)
    end
    lJ1, lJ3, lH1, lH3, lWT = sol
    lWP = 1 / d

    TrSigma = lWT + lWP * (d - 1)
    b = 4 / (π) * 1 / (1 + 2 * TrSigma)
    EChh = lH1 + lH3 +
        (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) * (d - 1) +
        (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)
    gammaYh2 = (4 / π) / (1 + 2 * EChh)
    lK1 = gammaYh2 * lH1
    lK3 = gammaYh2 * lH3
    return (lK1 / (lK1 + kappa / P), lK3 / (lK3 + kappa / P))
end



function compute_lK(sol, P,n1, n2, chi, d, delta, kappa, epsilon, b)


lJ1, lJ3, lH1, lH3,lWT = sol
lV1 = -(lH1 / lJ1^2 - 1 / lJ1)
lV3 = -(lH3 / lJ3^2 - 1 / lJ3)
lWT = 1 / (d + delta * b * n2 * (lV1) / n1)
lWP = 1 / d

TrSigma = lWT + lWP * (d - 1)
EChh = lH1 + lH3 +
       (16 / (π * (1 + 2 * TrSigma)^3) * (15 * lWP^3)) * (d - 1) +
       (4 / (π * (1 + 2 * TrSigma)) * lWP) * (d - 1)
gammaYh2 = (4 / π) / (1 + 2 * EChh)
print(gammaYh2)
lK1 = gammaYh2 * lH1
lK3 = gammaYh2 * lH3

return lK1, lK3

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

function populate_solution(sol::Solution, params::ProblemParams)
    lK1, lK3 = lK(sol, params.P; n1=params.n1, n2=params.n2, chi=params.χ, d=params.d, delta=params.ϵ, kappa=params.κ, epsilon=params.ϵ, b=params.b)
    t1, t3 = lT(sol, params.P; n1=params.n1, n2=params.n2, chi=params.χ, d=params.d, delta=params.ϵ, kappa=params.κ, epsilon=params.ϵ, b=params.b)
    v1, v3 = lV(sol)

    sol.lK1 = lK1
    sol.lK3 = lK3
    sol.lT1 = t1
    sol.lT3 = t3
    sol.lV1 = v1
    sol.lV3 = v3
    sol.learnability1 = lK1 / (lK1 + params.κ / params.P)
    sol.learnability3 = lK3 / (lK3 + params.κ / params.P)
    return sol
end

function nlsolve_solver(initial_guess;
    anneal=false,        # Flag for chi annealing
    anneal_P=true,      # Flag for P annealing
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, n2=1.0, b=1.0,
    P=nothing, anneal_steps=1000,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false, normalized::Bool=true)

    # 1. Transform initial guess to log-space to ensure positivity
    # We add a small epsilon to avoid log(0) if the guess is bad
    x_log = log.(initial_guess .+ 1e-12)

    target_P = (P === nothing) ? d^1.2 : P
    target_chi = chi

    # 2. Internal solver function that works in log-space
    function solve_at_params_log(current_x_log, current_P, current_chi)
        function f_log!(F, x_log_val)
            # Convert back to physical space for the residual calculation
            x_phys = exp.(x_log_val)
            F[:] = residuals(x_phys, current_P, current_chi, d, kappa, delta, epsilon, n1, n2, b; normalized=normalized)
        end
        return nlsolve(f_log!, current_x_log, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
    end

    # Define paths
    P_start = d + 5.0
    chi_start = 1e-8

    P_path = anneal_P ? exp.(range(log(P_start), log(target_P), length=anneal_steps)) : fill(target_P, anneal_steps)
    chi_path = anneal ? exp.(range(log(chi_start), log(target_chi), length=anneal_steps)) : fill(target_chi, anneal_steps)

    result_log = x_log

    if anneal || anneal_P
        if verbose
            println("Annealing in Log-Space: P=$(anneal_P), Chi=$(anneal)...")
        end
        for i in 1:anneal_steps
            p_curr = P_path[i]
            c_curr = chi_path[i]
            try
                sol = solve_at_params_log(result_log, p_curr, c_curr)
                result_log = sol.zero
            catch e
                @error "Log-solver failed at step $i (P=$p_curr)"
                return nothing
            end
        end
        return exp.(result_log) # Convert back to physical space
    else
        try
            sol = solve_at_params_log(x_log, target_P, target_chi)
            return exp.(sol.zero)
        catch e
            @error "Single-shot log-solver failed"
            return nothing
        end
    end
end

function solve_FCN3_Erf(problem_params::ProblemParams, initial_guess;
    anneal_P=false, anneal=true, anneal_steps=100,
    lr=1e-3, max_iter=5000, tol=1e-9, verbose=false, normalized::Bool=true,
    effective_ridge::Bool=false)

    # Ensure initial_guess is length 5: [lJ1, lJ3, lH1, lH3, lWT]
    # If a 4-element guess is passed, append a reasonable lWT
    if length(initial_guess) == 4
        initial_guess = [initial_guess..., 1.0 / problem_params.d]
    end

    function solve_with_kappa(kappa_value)
        return nlsolve_solver(
            initial_guess;
            anneal=anneal,
            anneal_P=anneal_P,
            anneal_steps=anneal_steps,
            chi=problem_params.χ,
            d=problem_params.d,
            kappa=kappa_value,
            delta=problem_params.ϵ,
            epsilon=problem_params.ϵ,
            n1=problem_params.n1,
            n2=problem_params.n2,
            b=problem_params.b,
            P=problem_params.P,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            normalized=normalized
        )
    end

    kappa_used = float(problem_params.κ)
    sol_vec = solve_with_kappa(kappa_used)

    if isnothing(sol_vec)
        return Solution()
    end

    if effective_ridge
        lK1_pre, lK3_pre = compute_lK(
            sol_vec, problem_params.P, problem_params.n1, problem_params.n2,
            problem_params.χ, problem_params.d, problem_params.ϵ,
            kappa_used, problem_params.ϵ, problem_params.b
        )
        kappa_used = compute_effective_ridge(kappa_used, [lK1_pre, lK3_pre], problem_params.P)
        sol_vec_eff = solve_with_kappa(kappa_used)
        if !isnothing(sol_vec_eff)
            sol_vec = sol_vec_eff
        end
    end

    # Map results to Solution object
    res_sol = Solution(
        lJ1=sol_vec[1],
        lJ3=sol_vec[2],
        lH1=sol_vec[3],
        lH3=sol_vec[4],
        lWT=sol_vec[5]
    )

    params_eff = ProblemParams(
        d=problem_params.d,
        κ=Float32(kappa_used),
        ϵ=problem_params.ϵ,
        P=problem_params.P,
        n1=problem_params.n1,
        n2=problem_params.n2,
        χ=problem_params.χ,
        b=problem_params.b
    )

    populated = populate_solution(res_sol, params_eff)
    populated.kappa_eff = kappa_used
    return populated
end

function sweep_learnabilities(initial_guess; alphas, chi, d, kappa, delta, epsilon, n, b, normalized::Bool=true)
    P_vals = d .^ collect(alphas)
    learnabilities_1 = Float64[]
    learnabilities_3 = Float64[]

    x = copy(initial_guess)  # warm-start
    n1 = n
    n2 = n
    for P in P_vals
        sol = nlsolve_solver(
            x,
            chi=chi, d=d, kappa=kappa, delta=delta,
            epsilon=epsilon, n1=n1, n2=n2, b=b,
            P=P,lr=1e-5, max_iter=500000, anneal=true, verbose=false,
            normalized=false
        )

        l1, l3 = compute_lK_ratio(sol, P, n1, n2, chi, d, delta, kappa, epsilon, b)
        push!(learnabilities_1, l1)
        push!(learnabilities_3, l3)

        # warm start next P with the last solution (if valid)
        if !(sol === nothing)
            x = sol
        end
    end

    return P_vals, learnabilities_1, learnabilities_3
end



end 