# ============================================================================
#  run_FCS.jl — CLI Runner for the 10-Variable GMM FCS Solver
# ============================================================================
# Purpose:
#   Command-line utility to solve the FCN3-Erf equations of state under
#   the bimodal GMM variational ansatz.
#
# Usage:
#   julia run_FCS.jl --d 150 --mu 1.0 --kappa 2.0 --P 1200
#
# Metadata:
#   Project: FCNX-Ensembling
#   Path:    run_FCS.jl
#   Author:  repo maintainers
#   Date:    2026-07-21
#
using ArgParse
using ForwardDiff
using LinearAlgebra
using Printf
using JSON

include(joinpath(@__DIR__, "FCS3_VGA.jl"))
using .FCS3_VGA

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
    s = ArgParseSettings(description="FCS 10-Variable GMM Solver with Target and Perpendicular modes")

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
        "--mu", "-m"
        help = "Bimodal GMM mean along target direction"
        arg_type = Float64
        default = 1.0
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
        default = 1e-8
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
    μ_val = args["mu"]
    lr = args["lr"]
    Tf = args["max-iter"]
    anneal = args["anneal"]
    anneal_steps = args["anneal-steps"]
    tol = args["tol"]
    verbose = args["verbose"]
    precision = args["precision"]
    output_file = args["to"]
    quiet = args["quiet"]

    sf = x -> sigfig(x; n=precision)

    if !quiet
        println("#####################################################################")
        println("##    &*&       THE GREAT FCN3-ERF GMM EOS SOLVER       &*&.        ")
        println("##                       Initializes                                 ")
        println("#####################################################################")
        println()
        println("Parameters:")
        println("  d=$d, κ=$κ, ϵ=$ϵ, P=$P")
        println("  n1=$n1, n2=$n2, χ=$χ, b=$b, μ=$μ_val")
        println("  lr=$lr, max_iter=$Tf")
        println()
    end

    # ========================================================================
    # TARGET EIGENVECTOR RUN (δ = 1.0)
    # ========================================================================
    δ_target = 1.0

    # 5-variable initial guess (will be auto-expanded to 10 variables in the solver)
    i0_target = [4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2),
        4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 1 / d]

    exp_sol_target = FCS3_VGA.nlsolve_solver(
        i0_target,
        chi=χ, d=d, kappa=κ, delta=δ_target,
        epsilon=ϵ, n1=n1, n2=n2, b=b, mu=μ_val,
        P=P, lr=lr, max_iter=Tf, verbose=verbose,
        anneal=anneal, anneal_steps=anneal_steps, tol=tol
    )

    l1_target, l3_target = FCS3_VGA.compute_lK_ratio(exp_sol_target, P, n1, n2, χ, d, δ_target, κ, ϵ, b; mu=μ_val)

    if !quiet
        println("-------")
        println("TARGET MODE (δ=$δ_target) - GMM Variational Target Eigenvalues χ=$χ")
    end

    if exp_sol_target === nothing
        if !quiet
            println("exp_sol_target is nothing")
        end
        lJ1T = lJ3T = lH1T = lH3T = lWTT = nothing
        mu_J1T = mu_J3T = mu_H1T = mu_H3T = mu_WTT = nothing
        lK1T = lK3T = nu1T = nu3T = nothing
    else
        lJ1T, lJ3T, lH1T, lH3T, lWTT, mu_J1T, mu_J3T, mu_H1T, mu_H3T, mu_WTT = exp_sol_target
        
        # Auxiliary fields conjugate to the target means
        nu1T = mu_H1T / lJ1T
        nu3T = mu_H3T / lJ3T

        lV1T = - ((lH1T + 2 * mu_H1T^2) / lJ1T^2 - 1 / lJ1T)
        TrH = lH1T + lH3T + (d - 1) * (15 * (1/d)^3) + (d - 1) * 1/d
        
        if !quiet
            println("Variances (Target):")
            println("  lH1T: $(sf(lH1T)), lH3T: $(sf(lH3T))")
            println("  lJ1T: $(sf(lJ1T)), lJ3T: $(sf(lJ3T))")
            println("  lWTT: $(sf(lWTT))")
            println()
            println("Means (Target):")
            println("  mu_H1T: $(sf(mu_H1T)), mu_H3T: $(sf(mu_H3T))")
            println("  mu_J1T: $(sf(mu_J1T)), mu_J3T: $(sf(mu_J3T))")
            println("  mu_WTT: $(sf(mu_WTT))")
            println()
            println("Auxiliary Fields (Target):")
            println("  nu1T: $(sf(nu1T)), nu3T: $(sf(nu3T))")
            println()
            println("Learnabilities (Target):")
            println("  mu1 = $(sf(l1_target))")
            println("  mu3 = $(sf(l3_target))")
        end

        lK1T, lK3T = FCS3_VGA.compute_lK(exp_sol_target, P, n1, n2, χ, d, δ_target, κ, ϵ, b; mu=μ_val)

        if !quiet
            println()
            println("Readout Eigenvalues (Target):")
            println("  lK1T: $lK1T")
            println("  lK3T: $lK3T")
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

    exp_sol_perp = FCS3_VGA.nlsolve_solver(
        i0_perp,
        chi=χ, d=d, kappa=κ, delta=δ_perp,
        epsilon=ϵ, n1=n1, n2=n2, b=b, mu=μ_val,
        P=P, lr=lr, max_iter=Tf,
        anneal=anneal, verbose=verbose, tol=tol
    )

    l1_perp, l3_perp = FCS3_VGA.compute_lK_ratio(exp_sol_perp, P, n1, n2, χ, d, δ_perp, κ, ϵ, b; mu=μ_val)

    if !quiet
        println("----------")
        println("PERPENDICULAR MODE (δ=$δ_perp) - Perpendicular Eigenvalues")
    end

    if exp_sol_perp === nothing
        if !quiet
            println("exp_sol_perp is nothing")
        end
        lJ1P = lJ3P = lH1P = lH3P = lWTP = nothing
        mu_J1P = mu_J3P = mu_H1P = mu_H3P = mu_WTP = nothing
        lK1P = lK3P = nu1P = nu3P = nothing
    else
        lJ1P, lJ3P, lH1P, lH3P, lWTP, mu_J1P, mu_J3P, mu_H1P, mu_H3P, mu_WTP = exp_sol_perp
        
        nu1P = mu_H1P / lJ1P
        nu3P = mu_H3P / lJ3P

        lV1P = - ((lH1P + 2 * mu_H1P^2) / lJ1P^2 - 1 / lJ1P)

        if !quiet
            println("Variances (Perpendicular):")
            println("  lH1P: $(sf(lH1P)), lH3P: $(sf(lH3P))")
            println("  lJ1P: $(sf(lJ1P)), lJ3P: $(sf(lJ3P))")
            println("  lWTP: $(sf(lWTP))")
            println()
            println("Means (Perpendicular):")
            println("  mu_H1P: $(sf(mu_H1P)), mu_H3P: $(sf(mu_H3P))")
            println("  mu_J1P: $(sf(mu_J1P)), mu_J3P: $(sf(mu_J3P))")
            println("  mu_WTP: $(sf(mu_WTP))")
            println()
            println("Auxiliary Fields (Perpendicular):")
            println("  nu1P: $(sf(nu1P)), nu3P: $(sf(nu3P))")
            println()
            println("Learnabilities (Perpendicular):")
            println("  mu1 = $(sf(l1_perp))")
            println("  mu3 = $(sf(l3_perp))")
        end

        lK1P, lK3P = FCS3_VGA.compute_lK(exp_sol_perp, P, n1, n2, χ, d, δ_perp, κ, ϵ, b; mu=μ_val)

        if !quiet
            println()
            println("Readout Eigenvalues (Perpendicular):")
            println("  lK1P: $lK1P")
            println("  lK3P: $lK3P")
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
            "mu" => μ_val,
            "lr" => lr,
            "max_iter" => Tf,
            "anneal" => anneal,
            "anneal_steps" => anneal_steps,
            "tol" => tol
        ),
        "target" => Dict(
            "delta" => δ_target,
            "lJ1" => lJ1T,
            "lJ3" => lJ3T,
            "lH1" => lH1T,
            "lH3" => lH3T,
            "lWT" => lWTT,
            "mu_J1" => mu_J1T,
            "mu_J3" => mu_J3T,
            "mu_H1" => mu_H1T,
            "mu_H3" => mu_H3T,
            "mu_WT" => mu_WTT,
            "nu1" => nu1T,
            "nu3" => nu3T,
            "lK1" => lK1T,
            "lK3" => lK3T,
            "mu1" => l1_target,
            "mu3" => l3_target
        ),
        "perpendicular" => Dict(
            "delta" => δ_perp,
            "lJ1" => lJ1P,
            "lJ3" => lJ3P,
            "lH1" => lH1P,
            "lH3" => lH3P,
            "lWP" => lWTP,
            "mu_J1" => mu_J1P,
            "mu_J3" => mu_J3P,
            "mu_H1" => mu_H1P,
            "mu_H3" => mu_H3P,
            "mu_WT" => mu_WTP,
            "nu1" => nu1P,
            "nu3" => nu3P,
            "lK1" => lK1P,
            "lK3" => lK3P,
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
            println("  lK1T / lK1P: $(lK1T/lK1P)")
            println("  lK3T / lK3P: $(lK3T/lK3P)")
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