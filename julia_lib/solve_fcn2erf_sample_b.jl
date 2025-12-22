# solve_fcn2erf_sample_b.jl
#
# Julia script to solve FCN2Erf equations, sampling b from N(0.415, 0.0566)
# Requires: FCN2Erf module in julia_lib/FCN2Erf.jl

using Random
using Distributions
include("FCS2Erf.jl")

function solve_fcn2erf_with_sampled_b(; num_samples=100, seed=42, solver=:nlsolve, initial_guess=[1.0, 1.0, 1.0], kwargs...)
    Random.seed!(seed)
    b_dist = Normal(0.415, 0.0566)
    results = []
    for i in 1:num_samples
        b = rand(b_dist)
        # Pass b as argument, use nlsolve_solver or gradient_descent_solver from FCS2Erf
        if solver == :nlsolve
            sol = FCS2Erf.nlsolve_solver(initial_guess; b=b, kwargs...)
        elseif solver == :gd
            sol = FCS2Erf.gradient_descent_solver(initial_guess; b=b, kwargs...)
        else
            error("Unknown solver: $solver")
        end
        push!(results, (b=b, sol=sol))
    end
    return results
end

# Example CLI usage:
if abspath(PROGRAM_FILE) == @__FILE__
    # Example: d=2, kappa=1.0, delta=1.0, n1=100, chi=1.0, P=40
    num_samples = 10
    d = 2
    kappa = 1.0
    delta = 1.0
    n1 = 100
    chi = 1.0
    P = 40
    results = solve_fcn2erf_with_sampled_b(num_samples=num_samples, d=d, kappa=kappa, delta=delta, n1=n1, chi=chi, P=P)
    for r in results
        println("b=", r.b, ", sol=", r.sol)
    end
end

export solve_fcn2erf_with_sampled_b
