using ArgParse
using ForwardDiff
using LinearAlgebra
using Plots
using Colors
using LaTeXStrings
using Printf
using JSON

include("./FCS.jl")
using .FCS

function sigfig(x::Float64; n::Int=4)
    if isnan(x) || isinf(x)
        return string(x)
    end

    if x == 0
        return "0" * (n > 1 ? "." * "0"^(n - 1) : "")
    end

    order = floor(log10(abs(x)))
    if order >= n
        return string(round(Int, x))
    else
        int_digits = order >= 0 ? Int(order) + 1 : 1
        dec_digits = n - int_digits
        dec_digits = max(dec_digits, 0)

        s = @sprintf("%.*f", dec_digits, x)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
        return s
    end
end

sigfig(::Nothing; n::Int=4) = "nothing"

function parse_commandline()
    s = ArgParseSettings(description="FCS Eigenvalue Solver with Target and Perpendicular modes")

    @add_arg_table! s begin
        "--d"
        help = "Dimension parameter"
        arg_type = Float64
        default = 150.0
        "--kappa", "-k"
        help = "Kappa parameter"
        arg_type = Float64
        default = 2.0
        "--epsilon", "-e"
        help = "Epsilon parameter"
        arg_type = Float64
        default = 0.03
        "--P"
        help = "P parameter"
        arg_type = Float64
        default = 1200.0
        "--n1"
        help = "n1 parameter"
        arg_type = Int
        default = 1600
        "--n2"
        help = "n2 parameter"
        arg_type = Int
        default = 1600
        "--chi"
        help = "Chi parameter"
        arg_type = Float64
        default = 80.0
        "--b"
        help = "b parameter"
        arg_type = Float64
        default = 4.0 / (3.0 * π)
        "--lr"
        help = "Learning rate"
        arg_type = Float64
        default = 1e-6
        "--max-iter"
        help = "Maximum iterations"
        arg_type = Int
        default = 6_000_000
        "--no-anneal"
        help = "Disable annealing (annealing is enabled by default)"
        action = :store_false
        dest_name = "anneal"
        default = true
        "--anneal-steps"
        help = "Number of annealing steps"
        arg_type = Int
        default = 30000
        "--tol"
        help = "Tolerance for solver"
        arg_type = Float64
        default = 1e-12
        "--verbose", "-v"
        help = "Verbose output from solver"
        action = :store_true
        "--precision"
        help = "Significant figures for display"
        arg_type = Int
        default = 8
        "--to"
        help = "Output file for JSON results (if not specified, prints to stdout)"
        arg_type = String
        default = nothing
        "--quiet", "-q"
        help = "Suppress progress output, only show JSON result"
        action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # Extract parameters
    d = args["d"]
    κ = args["kappa"]
    ϵ = args["epsilon"]
    P = args["P"]
    n1 = args["n1"]
    n2 = args["n2"]
    χ = args["chi"]
    b = args["b"]
    lr = args["lr"]
    Tf = args["max-iter"]
    anneal = args["anneal"]
    anneal_steps = args["anneal-steps"]
    tol = args["tol"]
    verbose = args["verbose"]
    precision = args["precision"]
    output_file = args["to"]
    quiet = args["quiet"]

    sf = x -> x  # Could use: x -> sigfig(x; n=precision)

    if !quiet
        println("#####################################################################")
        println("##    &*&       THE GREAT FCN3-ERF EOS SOLVER            &*&.        ")
        println("##                       Initializes                                 ")
        println("#####################################################################")
        println()
        println("Parameters:")
        println("  d=$d, κ=$κ, ϵ=$ϵ, P=$P")
        println("  n1=$n1, n2=$n2, χ=$χ, b=$b")
        println("  lr=$lr, max_iter=$Tf")
        println()
    end

    # ========================================================================
    # TARGET EIGENVECTOR RUN (δ = 1.0)
    # ========================================================================
    δ_target = 1.0

    i0_target = [4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2),
        4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 1 / d]

    exp_sol_target = FCS.nlsolve_solver(
        i0_target,
        chi=χ, d=d, kappa=κ, delta=δ_target,
        epsilon=ϵ, n1=n1, n2=n2, b=b,
        P=P, lr=lr, max_iter=Tf, verbose=verbose,
        anneal=anneal, anneal_steps=anneal_steps, tol=tol
    )

    l1_target, l3_target = FCS.compute_lK_ratio(exp_sol_target, P, n1, n2, χ, d, δ_target, κ, ϵ, b)

    if !quiet
        println("-------")
        println("TARGET MODE (δ=$δ_target) - Semi MF Scaling Target Eigenvalues χ=$χ")
    end

    if exp_sol_target === nothing
        if !quiet
            println("exp_sol_target is nothing")
        end
        lJ1T = lJ3T = lH1T = lH3T = lWT = nothing
        lK1T = lK3T = nothing
    else
        lJ1T, lJ3T, lH1T, lH3T, lWT = exp_sol_target
        lV1T = -(lH1T / lJ1T^2 - 1 / lJ1T)

        if !quiet
            println("lH1T: $(sf(lH1T)), lH3T: $(sf(lH3T))")
            println("lJ1T: $(sf(lJ1T)), lJ3T: $(sf(lJ3T))")
            @printf("lWT: %.*g\n", 7, lWT)
            println("lJ1T/lWT: $(lJ1T/(b*lWT))")
            println("lJ3T/lWT^3: $(lJ3T/(15*16/(27*π) * lWT^3))")
            println()
            println("Learnabilities (Target):")
            println("mu1 = $(sf(l1_target))")
            println("mu3 = $(sf(l3_target))")
        end

        lK1T, lK3T = FCS.compute_lK(exp_sol_target, P, n1, n2, χ, d, δ_target, κ, ϵ, b)

        if !quiet
            println()
            println("Readout Eigenvalues (Target):")
            println("lK1T: $lK1T")
            println("lK3T: $lK3T")
        end
    end

    if !quiet
        println()
    end

    # ========================================================================
    # PERPENDICULAR EIGENVECTOR RUN (δ = 0.0)
    # ========================================================================
    δ_perp = 0.0

    i0_perp = [4 / (3 * π) * 1 / d, 1 / d^3,
        4 / (3 * π) * 1 / d, 1 / d^3, 1 / d]

    exp_sol_perp = FCS.nlsolve_solver(
        i0_perp,
        chi=χ, d=d, kappa=κ, delta=δ_perp,
        epsilon=ϵ, n1=n1, n2=n2, b=b,
        P=P, lr=lr, max_iter=Tf,
        anneal=anneal, verbose=verbose, tol=tol
    )

    l1_perp, l3_perp = FCS.compute_lK_ratio(exp_sol_perp, P, n1, n2, χ, d, δ_perp, κ, ϵ, b)

    if !quiet
        println("----------")
        println("PERPENDICULAR MODE (δ=$δ_perp) - Perpendicular Eigenvalues")
    end

    if exp_sol_perp === nothing
        if !quiet
            println("exp_sol_perp is nothing")
        end
        lJ1P = lJ3P = lH1P = lH3P = lWP = nothing
        lK1P = lK3P = nothing
    else
        lJ1P, lJ3P, lH1P, lH3P, lWP = exp_sol_perp
        lV1P = -(lH1P / lJ1P^2 - 1 / lJ1P)

        if !quiet
            println("lH1P: $(sf(lH1P)), lH3P: $(sf(lH3P))")
            println("lJ1P: $(sf(lJ1P)), lJ3P: $(sf(lJ3P))")
            @printf("lWP: %.*g\n", 7, lWP)
            println("lJ1P/lWP: $(lJ1P/(b*lWP))")
            println("lJ3P/lWP^3: $(lJ3P/(15*16/(27*π) * lWP^3))")
            println()
            println("Learnabilities (Perpendicular):")
            println("mu1 = $(sf(l1_perp))")
            println("mu3 = $(sf(l3_perp))")
        end

        lK1P, lK3P = FCS.compute_lK(exp_sol_perp, P, n1, n2, χ, d, δ_perp, κ, ϵ, b)

        if !quiet
            println()
            println("Readout Eigenvalues (Perpendicular):")
            println("lK1P: $lK1P")
            println("lK3P: $lK3P")
        end
    end

    if !quiet
        println("----------")
    end

    # ========================================================================
    # COMPARISON AND JSON OUTPUT
    # ========================================================================
    results = Dict(
        "parameters" => Dict(
            "d" => d,
            "kappa" => κ,
            "epsilon" => ϵ,
            "P" => P,
            "n1" => n1,
            "n2" => n2,
            "chi" => χ,
            "b" => b,
            "lr" => lr,
            "max_iter" => Tf,
            "anneal" => anneal,
            "anneal_steps" => anneal_steps,
            "tol" => tol
        ),
        "target" => Dict(
            "delta" => δ_target,
            "lJ1T" => lJ1T,
            "lJ3T" => lJ3T,
            "lH1T" => lH1T,
            "lH3T" => lH3T,
            "lWT" => lWT,
            "lK1T" => lK1T,
            "lK3T" => lK3T,
            "mu1" => l1_target,
            "mu3" => l3_target
        ),
        "perpendicular" => Dict(
            "delta" => δ_perp,
            "lJ1P" => lJ1P,
            "lJ3P" => lJ3P,
            "lH1P" => lH1P,
            "lH3P" => lH3P,
            "lWP" => lWP,
            "lK1P" => lK1P,
            "lK3P" => lK3P,
            "mu1" => l1_perp,
            "mu3" => l3_perp
        )
    )

    if exp_sol_target !== nothing && exp_sol_perp !== nothing
        results["ratios"] = Dict(
            "lK1T_over_lK1P" => lK1T / lK1P,
            "lK3T_over_lK3P" => lK3T / lK3P
        )

        if !quiet
            println()
            println("Ratio of readout (Target / Perpendicular):")
            println("lK1T / lK1P: $(lK1T/lK1P)")
            println("lK3T / lK3P: $(lK3T/lK3P)")
        end
    end

    # Output JSON
    json_output = JSON.json(results, 2)

    if output_file !== nothing
        open(output_file, "w") do f
            write(f, json_output)
        end
        if !quiet
            println()
            println("Results saved to: $output_file")
        end
    else
        println(json_output)
    end

    if !quiet
        println()
        println("Done!")
    end
end

# Run main function
main()