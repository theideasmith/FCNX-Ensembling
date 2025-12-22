# ============================================================================
#  FCS2Erf.jl — Fixed-point Consistency Solver for FCN2 (erf activations)
# ============================================================================
# Purpose:
#   Numerical solvers for FCN2 with erf activations.
#   Simplification of FCN3Erf by removing the second hidden layer.
#   
#   Key equations for FCN2:
#     lJ = 1/(d + delta * b * lT / (chi * n1))
#     where b = 4/(3*pi) * factor from erf nonlinearity
#     lT = -chi^2 / (kappa/P + lK)^2 * delta
#     lK = gamma * lJ, where gamma = (4/pi) / (1 + 2*TrSigma)
#
# Exports:
#   - residuals
#   - nlsolve_solver
#   - gradient_descent_solver
#
# Dependencies:
#   ForwardDiff, NLsolve, LinearAlgebra
#
module FCS2Erf

export residuals, gradient_descent_solver, nlsolve_solver

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
    ϵ::Float32
    P::Float32
    n1::Float32
    χ::Float32
    b::Float32 = 4/(3*π)
end 

@kwdef mutable struct Solution
    lJ::Float64 = NaN
end

# -------------------------
# Residual function for FCN2 erf network
# -------------------------
# Given a guess x = [lJ], returns the residuals
function residuals(x, P, chi, d, kappa, delta, n1, b)
    lJ , lk, lWT = x[1], x[2], x[3]
    
    lWP = 1.0 / d
    TrSigma = lWT + lWP * (d - 1) 
    b = (4 / (π)) / (1 + 2 * TrSigma)  # erf factor
    # Compute lT
    lT = -chi^2 * lJ^(-2) * lk  - chi / lJ
    
    # Compute TrSigma (trace of covariance matrix)
    rlWT = lWT - 1 / (d + delta * b * lT / (chi * n1))
    
    # Compute gamma factor
    gammaYh2 = (4 / (π)) / (1 + 2 * TrSigma)
    
    # Residual: fixed point for lJ
    rj = lJ - gammaYh2 / (d + delta * b * lT / (chi * n1)) 
    rk = lk -  (lJ / (lJ + kappa / P))^2 * delta
    return [rj, rk, rlWT]
end

# -------------------------
# Gradient Descent Solver
# -------------------------
function gradient_descent_solver(initial_guess;
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, n1=1.0, b=4/(3*π),
    P=nothing,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)

    x = copy(initial_guess)
    if P === nothing
        P = d^1.2
    end
    loss = nothing

    for iter in 1:max_iter
        res = residuals(x, P, chi, d, kappa, delta, n1, b)
        loss = sum(res .^ 2) + sum(x .^ 2)
        grad = ForwardDiff.gradient(y -> sum(residuals(y, P, chi, d, kappa, delta, n1, b) .^ 2), x)
        x -= lr * grad

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
    chi=1.0, d=1.0, kappa=1.0, delta=1.0, n1=1.0, b=4/(3*π),
    P=nothing, anneal_steps=30000,
    lr=1e-3, max_iter=5000, tol=1e-8, verbose=false)
    
    x = copy(initial_guess)
    if P === nothing
        P = d^1.2
    end

    function res_func!(F, x, c)
        F[:] = residuals(x, P, c, d, kappa, delta, n1, b)
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
                print("ERROR in annealing: ")
                showerror(stdout, e, catch_backtrace())
            end
        end
    else 
        f2!(F, x) = res_func!(F, x, chi)
        result = nlsolve(f2!, x; xtol=tol, ftol=tol, iterations=max_iter, show_trace=verbose).zero
    end 

    return result
end

end  # module FCS2Erf
