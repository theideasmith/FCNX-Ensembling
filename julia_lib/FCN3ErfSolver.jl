#!/usr/bin/env julia
# filepath: /home/akiva/FCNX-Ensembling/julia_lib/solve_fcs_cli.jl

using JSON
using Printf

include("FCS.jl")
using .FCS

function main()
    if length(ARGS) < 2
        println("Usage: julia solve_fcs_cli.jl <output_json> <d> [--kappa=1.0] [--epsilon=0.0] [--P=600] [--n1=450] [--n2=450] [--chi=450] [--delta=1.0] [--b=4/(3*pi)] [--max_iter=5000] [--tol=1e-8]")
        exit(1)
    end

    # Required
    output_json = ARGS[1]
    d = parse(Float64, ARGS[2])

    # Defaults
    κ = 1.0
    ϵ = 0.0
    P = 600.0
    n1 = 450.0
    n2 = 450.0
    χ = 450.0
    δ = 1.0
    b = 4/(3*π)
    max_iter = 5000
    tol = 1e-8
    lr = 1e-3
    verbose = false

    # Parse optional arguments
    for arg in ARGS[3:end]
        if occursin("--kappa=", arg)
            κ = parse(Float64, split(arg, "=")[2])
        elseif occursin("--epsilon=", arg)
            ϵ = parse(Float64, split(arg, "=")[2])
        elseif occursin("--P=", arg)
            P = parse(Float64, split(arg, "=")[2])
        elseif occursin("--n1=", arg)
            n1 = parse(Float64, split(arg, "=")[2])
        elseif occursin("--n2=", arg)
            n2 = parse(Float64, split(arg, "=")[2])
        elseif occursin("--chi=", arg)
            χ = parse(Float64, split(arg, "=")[2])
        elseif occursin("--delta=", arg)
            δ = parse(Float64, split(arg, "=")[2])
        elseif occursin("--b=", arg)
            b = parse(Float64, split(arg, "=")[2])
        elseif occursin("--max_iter=", arg)
            max_iter = parse(Int, split(arg, "=")[2])
        elseif occursin("--tol=", arg)
            tol = parse(Float64, split(arg, "=")[2])
        elseif occursin("--lr=", arg)
            lr = parse(Float64, split(arg, "=")[2])
        elseif occursin("--verbose", arg)
            verbose = true
        end
    end

    # Initial guess (can be improved)
    i0 = [4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 1/d]

    params = FCS.ProblemParams(
        d=Float32(d), κ=Float32(κ), ϵ=Float32(ϵ), P=Float32(P),
        n1=Float32(n1), n2=Float32(n2), χ=Float32(χ), b=Float32(b)
    )

    sol = FCS.nlsolve_solver(
        i0,
        chi=χ, d=d, kappa=κ, delta=δ,
        epsilon=ϵ, n1=n1, n2=n2, b=b,
        P=P, lr=lr, max_iter=max_iter, tol=tol, verbose=verbose
    )

    if sol === nothing
        println("No solution found.")
        exit(2)
    end
    print(sol)
    # Populate solution struct for derived quantities
    solution = FCS.populate_solution(FCS.Solution(sol[1], sol[2], sol[3], sol[4]), params)

    # Prepare output dictionary
    out = Dict(
        "d" => d,
        "kappa" => κ,
        "epsilon" => ϵ,
        "P" => P,
        "n1" => n1,
        "n2" => n2,
        "chi" => χ,
        "delta" => δ,
        "b" => b,
        "lJ1" => solution.lJ1,
        "lJ3" => solution.lJ3,
        "lH1" => solution.lH1,
        "lH3" => solution.lH3,
        "lK1" => solution.lK1,
        "lK3" => solution.lK3,
        "lT1" => solution.lT1,
        "lT3" => solution.lT3,
        "lV1" => solution.lV1,
        "lV3" => solution.lV3,
        "lWT" => solution.lWT,
        "learnability1" => solution.learnability1,
        "learnability3" => solution.learnability3
    )

    open(output_json, "w") do io
        JSON.print(io, out; indent=4)
    end
    println("Solution written to $output_json")
end

main()