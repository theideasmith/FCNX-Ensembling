# ============================================================================
#  FCS2Erf_Cubic.jl — Fixed-point Consistency Solver for FCN2 (erf activations)
# ============================================================================
# Purpose:
#   Numerical solvers for 2-layer fully connected networks with erf activations,
#   including cubic eigenvalue predictions. Simplified from FCS.jl by removing
#   the second hidden layer.
#
# Architecture:
#   Input (d) -> Hidden (n1, erf) -> Output (linear)
#   f(x) = A · erf(W0 · x)
#
# Variables:
#   lJ1, lJ3: Hidden layer kernel eigenvalues (linear and cubic features)
#   lWT: Target direction weight variance
#
# Exports:
#   - residuals_fcn2
#   - gradient_descent_solver_fcn2
#   - nlsolve_solver_fcn2
#   - solve_FCN2_Erf
#   - populate_solution_fcn2
#
# Usage:
#   using FCS2Erf_Cubic
#   params = ProblemParams2(d=50.0, κ=1.0, ϵ=0.03, P=600.0, n1=800.0, χ=80.0)
#   guess = [1.0, 1.0, 0.02]  # [lJ1, lJ3, lWT]
#   sol = solve_FCN2_Erf(params, guess; verbose=true)
#
module FCS2Erf_Cubic

export residuals_fcn2, gradient_descent_solver_fcn2, nlsolve_solver_fcn2, solve_FCN2_Erf, populate_solution_fcn2, ProblemParams2, Solution2

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
    b::Float32 = 4/(3*π)
end

@kwdef mutable struct Solution2
    lJ1::Float64 = NaN
    lJ3::Float64 = NaN
    lK1::Float64 = NaN
    lK3::Float64 = NaN
    lWT::Float64 = NaN
    learnability1::Float64 = NaN
    learnability3::Float64 = NaN
end

# Helper functions
lWP(d::Real) = 1 / d

function TrSigma(lWT::Real, d::Real)
    lWP_val = lWP(d)
    return lWT + lWP_val * (d - 1)
end

# -------------------------
# Residual function for FCN2
# -------------------------
# Given x = [lJ1, lJ3, lWT], returns residuals of the 3 equations
function residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
    lJ1, lJ3, lWT = x
    lWP_val = 1.0 / d
    
    TrSigma_val = lWT + lWP_val * (d - 1)
    
    # Output kernel eigenvalues (K = variance of output weighted by readout)
    # For FCN2: K relates directly to J through readout variance (chi scaling)
    # γ_Yh = (4/π) / (1 + 2·E[h·h])
    # For FCN2, E[h·h] ≈ TrΣ in the NNGP limit
    gammaYh = (4 / π) / (1 + 2 * TrSigma_val)
    
    # Training signal (learning rates from kernel theory)
    lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
    lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)
    
    # Equations of state
    # J1: Linear feature kernel from first layer
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
    
    # J3: Cubic feature kernel from first layer
    rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3)) 
    
    # WT: Target weight variance with training feedback
    # Training modifies weights: Σ_T = Σ_0 + δ·training_contribution
    # For FCN2: lV1 = effect of training on kernel = lT1·gammaYh/(n1·chi)

    rlWT = lWT - 1 / (d + epsilon^2 * gammaYh * (lT1)/ ( n1 * chi))
    
    return [rj1, rj3, rlWT]
end

# -------------------------
# Residual function with hardcoded TrSigma
# -------------------------
# Given x = [lJ1, lJ3], with TrSigma fixed to hardcoded value
function residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
    lJ1, lJ3 = x
    lWP_val = 1.0 / d
    
    # Compute lWT from fixed TrSigma: TrSigma = lWT + (d-1)*lWP => lWT = TrSigma - (d-1)*lWP
    lWT = TrSigma_fixed - (d - 1) * lWP_val
    TrSigma_val = TrSigma_fixed
    
    # Output kernel eigenvalues
    gammaYh = (4 / π) / (1 + 2 * TrSigma_val)
    
    # Training signal
    lT1 = -(chi^2 / (kappa / P + lJ1)^2 * delta) - chi^2 * kappa / (P * chi) * lJ1 / (lJ1 + kappa / P)
    lT3 = -(chi^2 / (kappa / P + lJ3)^2 * delta) - chi^2 * kappa / (P * chi) * lJ3 / (lJ3 + kappa / P)
    
    # Equations of state (J equations only, lWT is fixed)
    rj1 = lJ1 - (4 / (π * (1 + 2 * TrSigma_val)) * lWT)
    rj3 = lJ3 - (16 / (π * (1 + 2 * TrSigma_val)^3) * 15 * (lWT^3))
    
    return [rj1, rj3]
end

# -------------------------
# Gradient Descent Solver
# -------------------------
function gradient_descent_solver_fcn2(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
    P=nothing,
    TrSigma_fixed=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    if P === nothing
        P = d^1.2
    end
    loss = nothing

    if TrSigma_fixed === nothing
        # Standard 3-variable solve
        x = copy(initial_guess)
        for iter in 1:max_iter
            res = residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b)
            loss = sum(res .^ 2)
            
            grad = ForwardDiff.gradient(x -> sum(residuals_fcn2(x, P, chi, d, kappa, delta, epsilon, n1, b) .^ 2), x)
            x -= lr * grad
            
            if loss < tol
                if verbose
                    println("Converged in $iter iterations, loss = $loss")
                end
                return x
            end
        end
    else
        # 2-variable solve with fixed TrSigma
        x = copy(initial_guess[1:2])  # Use only [lJ1, lJ3]
        lWP_val = 1.0 / d
        lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
        for iter in 1:max_iter
            res = residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
            loss = sum(res .^ 2)
            
            grad = ForwardDiff.gradient(x -> sum(residuals_fcn2_fixed_TrSigma(x, P, chi, d, kappa, delta, epsilon, n1, b, TrSigma_fixed) .^ 2), x)
            x -= lr * grad
            
            if loss < tol
                if verbose
                    println("Converged in $iter iterations, loss = $loss (fixed TrSigma = $TrSigma_fixed, lWT = $lWT_from_TrSigma)")
                end
                return [x[1], x[2], lWT_from_TrSigma]  # Return with computed lWT
            end
        end
    end

    if verbose
        println("Reached max iterations, final loss = $loss")
    end
    if TrSigma_fixed === nothing
        return x
    else
        lWP_val = 1.0 / d
        lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
        return [x[1], x[2], lWT_from_TrSigma]
    end
end

# -------------------------
# NLsolve Solver
# -------------------------
function nlsolve_solver_fcn2(initial_guess;
    anneal=false,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, epsilon=1.0, n1=1.0, b=1.0,
    P=nothing, anneal_steps=30000,
    TrSigma_fixed=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)
    
    if P === nothing
        P = d^1.2
    end

    result = nothing
    
    if TrSigma_fixed === nothing
        # Standard 3-variable solve
        x = copy(initial_guess)
        
        function res_func!(F, x, c)
            F[:] = residuals_fcn2(x, P, c, d, kappa, delta, epsilon, n1, b)
        end

        if anneal
            chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
            prev_sol = x
            for (j, chit) in enumerate(chi_anneal_list)
                f1! = (F, x) -> res_func!(F, x, chit)
                try
                    sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
                    if (j == anneal_steps)
                        result = sol.zero
                    end
                    prev_sol = sol.zero
                catch e
                    println("Error during annealing at chi=$chit")
                    showerror(stdout, e, catch_backtrace())
                end
            end
        else
            f2!(F, x) = res_func!(F, x, chi)
            try
                result = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
            catch e
                println("Error during solve")
                showerror(stdout, e, catch_backtrace())
                result = nothing
            end
        end
    else
        # 2-variable solve with fixed TrSigma
        x = copy(initial_guess[1:2])  # Use only [lJ1, lJ3]
        lWP_val = 1.0 / d
        lWT_from_TrSigma = TrSigma_fixed - (d - 1) * lWP_val
        
        function res_func_fixed!(F, x, c)
            F[:] = residuals_fcn2_fixed_TrSigma(x, P, c, d, kappa, delta, epsilon, n1, b, TrSigma_fixed)
        end

        if anneal
            chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
            prev_sol = x
            for (j, chit) in enumerate(chi_anneal_list)
                f1! = (F, x) -> res_func_fixed!(F, x, chit)
                try
                    sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
                    if (j == anneal_steps)
                        result = [sol.zero[1], sol.zero[2], lWT_from_TrSigma]
                    end
                    prev_sol = sol.zero
                catch e
                    println("Error during annealing at chi=$chit with fixed TrSigma=$TrSigma_fixed")
                    showerror(stdout, e, catch_backtrace())
                end
            end
        else
            f2!(F, x) = res_func_fixed!(F, x, chi)
            try
                sol_2var = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
                result = [sol_2var[1], sol_2var[2], lWT_from_TrSigma]
            catch e
                println("Error during solve with fixed TrSigma=$TrSigma_fixed")
                showerror(stdout, e, catch_backtrace())
                result = nothing
            end
        end
    end

    return result
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
    
    lK1 = gammaYh * lJ1 / params.χ
    lK3 = gammaYh * lJ3 / params.χ
    
    learnability1 = lK1 / (lK1 + params.κ / params.P)
    learnability3 = lK3 / (lK3 + params.κ / params.P)
    
    return Solution2(
        lJ1=lJ1,
        lJ3=lJ3,
        lK1=lK1,
        lK3=lK3,
        lWT=lWT,
        learnability1=learnability1,
        learnability3=learnability3
    )
end

# -------------------------
# Main Solver Interface
# -------------------------
function solve_FCN2_Erf(problem_params::ProblemParams2, initial_guess;
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false, use_anneal=true, TrSigma_fixed=nothing)

    sol_vec = nlsolve_solver_fcn2(
        initial_guess,
        chi=problem_params.χ,
        d=problem_params.d,
        kappa=problem_params.κ,
        delta=problem_params.ϵ,
        epsilon=problem_params.ϵ,
        n1=problem_params.n1,
        b=problem_params.b,
        P=problem_params.P,
        TrSigma_fixed=TrSigma_fixed,
        lr=lr,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        anneal=use_anneal
    )
    
    solution = populate_solution_fcn2(sol_vec, problem_params)
    return solution
end

end # module
