#!/usr/bin/env julia
# Self-consistent solver for kappa_eff given eigenvalues and kappa_bare from JSON


using JSON
using Printf
using NLsolve
using LinearAlgebra

function solve_kappa_eff(lambdas::Vector{Float64}, kappa_bare::Float64, P::Int; tol=1e-10, maxiter=1000)
    function f!(F, x)
        kappa_eff = x[1]
        denom = kappa_eff / P .+ lambdas
        sum_term = sum(lambdas .* kappa_eff ./ P ./ denom)
        F[1] = kappa_eff - (kappa_bare + sum_term)
    end
    x0 = [kappa_bare + sum(lambdas)]
    sol = nlsolve(f!, x0; xtol=tol, ftol=tol, iterations=maxiter)
    if sol.x_converged || sol.f_converged
        return sol.zero[1]
    else
        error("nlsolve did not converge after $maxiter iterations")
    end
end

function arcsin_kernel(X::Matrix{Float64})
    # X is P x d
    norms_sq = sum(X.^2, dims=2)  # P x 1
    dot = X * X'  # P x P
    arg = dot ./ sqrt.((1 .+ norms_sq) * (1 .+ norms_sq)')
    return (2 / π) * asin.(clamp.(arg, -1.0, 1.0))
end

function main()
    if length(ARGS) < 2
        println("Usage: julia self_consistent_kappa_solver.jl <eigen_json> <P>")
        println("Or: julia self_consistent_kappa_solver.jl arcsin <kappa_bare> <P> <d>")
        exit(1)
    end
    if ARGS[1] == "arcsin"
        if length(ARGS) < 4
            println("Usage for arcsin: julia self_consistent_kappa_solver.jl arcsin <kappa_bare> <P> <d>")
            exit(1)
        end
        kappa_bare = parse(Float64, ARGS[2])
        P = parse(Int, ARGS[3])
        d = parse(Int, ARGS[4])
        X = randn(P, d)
        K = arcsin_kernel(X)
        lambdas = sort(eigvals(K), rev=true)
    else
        json_path = ARGS[1]
        P = parse(Int, ARGS[2])
        data = JSON.parsefile(json_path)
        lambdas = Vector{Float64}(data["eigenvalues"])
        kappa_bare = Float64(data["kappa_bare"])
    end
    kappa_eff = solve_kappa_eff(lambdas, kappa_bare, P)
    @printf("kappa_eff = %.10f\n", kappa_eff)
    # Sum of eigenvalues
    sum_lambdas = sum(lambdas)
    @printf("sum of eigenvalues = %.10f\n", sum_lambdas)
end

main()
