# ============================================================================
#  FCSLinear.jl — Fixed-point Consistency Solver for FCN3 (linear activations)
# ============================================================================
# Purpose:
#   Numerical solvers and helpers that compute mean-field / fixed-point
#   predictions for linear activation networks. This is a simplified version
#   of FCS.jl since linear networks cannot learn cubic functions (no He3).
#
# Key differences from FCS.jl:
#   - No b (EChh factor) in the equations
#   - No gamma (gammaYh2 factor) in the equations  
#   - Only He1 eigenvalues (no He3)
#   - Simplified residuals with direct formulas
#
# Mathematica equations:
#   lKh == χ^2 lK/lH^2
#   lK == (k/(P χ)) lH/(lH + k/P) - (lH/(lH + k/P))^2 * Δ
#   lHh == (1/lJ - (1/lJ)^2 lH)
#   lH == 1/((lKh/(χ n2)) + 1/lJ)
#   lJ == 1/(d + (n2/n1) lHh)
#
# Exports:
#   - residuals
#   - gradient_descent_solver
#   - nlsolve_solver
#   - compute_lK
#
# Dependencies:
#   ForwardDiff, NLsolve, LinearAlgebra
#
module FCSLinear

export residuals, gradient_descent_solver, nlsolve_solver, compute_lK

using Pkg
using ForwardDiff
using LinearAlgebra
using NLsolve

Pkg.instantiate()

is_physical(sol) = all(sol .> 0)

using Base: @kwdef

@kwdef mutable struct ProblemParams
    d::Float32
    κ::Float32
    P::Float32
    n1::Float32
    n2::Float32
    χ::Float32
end 

@kwdef mutable struct Solution
    lJ::Float64 = NaN
    lH::Float64 = NaN
    lK::Float64 = NaN
    lKh::Float64 = NaN
    lHh::Float64 = NaN
end

function compute_lKh(sol, P, n1, n2, chi, d, delta, kappa)
    if sol === nothing
        return NaN
    end
    
    lJ, lH = sol
    
    # lK == (k/(P χ)) lH/(lH + k/P) - (lH/(lH + k/P))^2 * Δ
    k_over_P = kappa / P
    denominator = lH + k_over_P
    lK = (kappa / (P * chi)) * lH / denominator + (lH / denominator)^2 * delta
    
    # lKh == χ^2 lK/lH^2
    lKh =  - chi^2 * lK 
    
    return lKh
end
# -------------------------
# Residual function for linear network
# -------------------------
# Given a guess x = [lJ, lH], returns the residuals of the 2 equations
function residuals(x, P, chi, d, kappa, delta, n1, n2)
    lJ, lH = x  # current variables
    
    # Compute intermediate values from the Mathematica equations
    # lHh == (1/lJ - (1/lJ)^2 lH)
    lHh = (1/lJ - (1/lJ)^2 * lH)
    
    # lK == (k/(P χ)) lH/(lH + k/P) - (lH/(lH + k/P))^2 * Δ
    k_over_P = kappa / P
    denominator = lH + k_over_P
    lK = (1.0 / denominator)^2 * delta
    
    # lKh == χ^2 lK/lH^2
    lKh =  - chi^2 * lK 
    
    # Now compute residuals:
    # lH == 1/((lKh/(χ n2)) + 1/lJ)
    rh = lH - 1 / ((lKh / (chi * n2)) + 1/lJ)
    
    # lJ == 1/(d + (n2/n1) lHh)
    rj = lJ - 1 / (d + (n2/n1) * lHh)
    
    return [rj, rh]
end

# -------------------------
# Gradient Descent Solver
# -------------------------
function gradient_descent_solver(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, n1=1.0, n2=1.0,
    P=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    x = copy(initial_guess)
    if P === nothing
        P = d^1.2  # default
    end
    loss = nothing

    for iter in 1:max_iter
        # Compute residuals
        res = residuals(x, P, chi, d, kappa, delta, n1, n2)

        # Loss = sum of squares
        loss = sum(res .^ 2) + sum(x .^ 2)

        # Gradient of loss wrt x
        grad = ForwardDiff.gradient(x -> sum(residuals(x, P, chi, d, kappa, delta, n1, n2) .^ 2), x)

        # Update step
        x -= lr * grad

        # Convergence check
        if loss < tol
            if verbose
                println("Converged in $iter iterations, loss = $loss")
            end
            return x
        end
    end

    if verbose
        println("Reached max iterations, final loss = $loss")
    end
    return x
end

# -------------------------
# NLsolve Solver with annealing
# -------------------------
function nlsolve_solver(initial_guess;
    anneal=false,
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, n1=1.0, n2=1.0,
    P=nothing, anneal_steps=30000,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)
    
    x = copy(initial_guess)
    if P === nothing
        P = d^1.2 # default
    end

    function res_func!(F, x, c)
        F[:] = residuals(x, P, c, d, kappa, delta, n1, n2)
    end

    result = nothing
    if anneal 
        chi_anneal_list = exp.(range(log(1e-8), log(chi), length=anneal_steps))
        prev_sol = x
        for (j, chit) in enumerate(chi_anneal_list)
            f1! = (F, x) -> res_func!(F, x, chit)
            try
                sol = nlsolve(f1!, prev_sol, xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose)
                if (j == anneal_steps)
                    result = is_physical(sol.zero) ? sol.zero : nothing
                end
                prev_sol = sol.zero
            catch e
                print("THERE WAS AN ERROR")
                showerror(stdout, e, catch_backtrace())
            end
        end
    else 
        f2!(F, x) = res_func!(F, x, chi)
        result = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
    end 

    return result
end

# -------------------------
# Compute lK value from solution
# -------------------------
function compute_lK(sol, P, n1, n2, chi, d, delta, kappa)
    if sol === nothing
        return NaN
    end
    
    lJ, lH = sol
    
    # lK == (k/(P χ)) lH/(lH + k/P) - (lH/(lH + k/P))^2 * Δ
    k_over_P = kappa / P
    denominator = lH + k_over_P
    lK = (kappa / (P * chi)) * lH / denominator + (lH / denominator)^2 * delta
    
    return [lK]  # Return as array for consistency with FCS.jl interface
end

end  # module FCSLinear
