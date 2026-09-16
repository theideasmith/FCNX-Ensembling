#!/usr/bin/env julia
# compute_fcn2_erf_cubic_eigs.jl

using ArgParse
using JSON3
using LinearAlgebra

push!(LOAD_PATH, @__DIR__)

# Ensure FCS2_VGA.jl is the Bimodal version [lJ1, lJ3, sigS, muW]
include("FCS2_VGA.jl")
using .FCS2_VGA

# Ensure FCS2Erf_Cubic.jl is the Classical Laplace version [lJ1, lJ3, lWT]
include("FCS2Erf_Cubic.jl")
using .FCS2Erf_Cubic

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--d"
        arg_type = Float64;
        required = true
        "--n1"
        arg_type = Float64;
        required = true
        "--P"
        arg_type = Float64;
        required = true
        "--chi"
        arg_type = Float64;
        default = NaN
        "--kappa"
        arg_type = Float64;
        default = NaN
        "--delta"
        arg_type = Float64;
        default = 1.0
        "--epsilon"
        arg_type = Float64;
        default = 0.03
        "--s0"
        arg_type = Float64;
        default = 1.0
        "--compare"
        help = "Compare Bimodal VGA (4D) with Classical Laplace (3D)"
        action = :store_true
        "--to"
        arg_type = String;
        default = ""
        "--quiet"
        action = :store_true
        "--verbose"
        action = :store_true
        "--anneal_steps"
        arg_type = Int;
        default = 3000
        "--no-anneal"
        action = :store_true
        "--advanced"
        help = "Use exact Gauss–Hermite GMM entropy instead of Hershey–Olsen bound"
        action = :store_true
        "--entropy-rule"
        help = "Gauss–Hermite quadrature order for --advanced entropy"
        arg_type = Int;
        default = 64
    end
    return ArgParse.parse_args(ARGS, s)
end

function main()
    args = parse_cli_args()
    d, n1, P = args["d"], args["n1"], args["P"]
    chi = isnan(args["chi"]) ? n1 : args["chi"]
    kappa = isnan(args["kappa"]) ? 1.0 / chi : args["kappa"]
    epsilon = args["epsilon"]
    s0 = args["s0"]
    advanced = args["advanced"]
    entropy_rule = args["entropy-rule"]

    # --- Initial Guesses ---
    # VGA Bimodal (4D): [lJ1, lJ3, sigS, muW]
    init_mix = [1.0/d, 1.0/d^3, sqrt(0.8/d), 0.1]

    # Classical (3D): [lJ1, lJ3, lWT]
    init_class = [1.0/d, 1.0/d^3, 1.0/d]

    function solve_system(δ_val)
        # 1. Solve Bimodal Mixture (VGA)
        vga_params = FCS2_VGA.ProblemParams2(
            d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon),
            P=Float32(P), n1=Float32(n1), χ=Float32(chi), δ=Float32(δ_val), s0=Float32(s0)
        )
        sol_vga = FCS2_VGA.solve_FCN2_Erf(
            vga_params, init_mix;
            anneal_steps=args["anneal_steps"], use_anneal=(!args["no-anneal"]),
            advanced=advanced, entropy_rule=entropy_rule,
        )

        # 2. Solve Classical (Laplace) if requested
        sol_class = nothing
        if args["compare"]
            class_params = FCS2Erf_Cubic.ProblemParams2(
                d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon),
                P=Float32(P), n1=Float32(n1), χ=Float32(chi), δ=Float32(δ_val)
            )
            sol_class = FCS2Erf_Cubic.solve_FCN2_Erf(
                class_params, init_class;
                verbose=args["verbose"], use_anneal=(!args["no-anneal"])
            )
        end
        return (vga=sol_vga, class=sol_class)
    end

    # Solve for Target (delta=1) and Perpendicular (delta=0)
    target = solve_system(1.0)
    perp = solve_system(0.0)

    # Compile results dictionary
    result = Dict(
        "parameters" => Dict(
            "d"=>d, "n1"=>n1, "P"=>P, "chi"=>chi, "kappa"=>kappa,
            "advanced"=>advanced, "entropy_rule"=>entropy_rule,
        ),
        "vga" => Dict("target" => target.vga, "perp" => perp.vga),
        "derived" => Dict(
            "lJ1_total" => target.vga.lJ1 + perp.vga.lJ1 * (d - 1),
            "TrSigma_target" => target.vga.lWT + (d - 1) / d
        )
    )

    if args["compare"]
        result["classical"] = Dict("target" => target.class, "perp" => perp.class)
    end

    # --- Printing ---
    if !args["quiet"]
        entropy_label = advanced ? "exact GMM entropy (rule=$(entropy_rule))" : "Hershey–Olsen entropy bound"
        println("\n" * "="^90)
        println("FCN2 4D Bimodal Mixture Model (VGA) Results")
        println("Entropy: $entropy_label")
        println("="^90)

        for (label, sol) in [("Target (δ=1.0)", target.vga), ("Perpendicular (δ=0.0)", perp.vga)]
            println("\n$label:")
            println("  Moments    : lWT (E[w^2]) = $(round(sol.lWT, sigdigits=6))")
            println("  Spectral   : lJ1 = $(round(sol.lJ1, sigdigits=6)), lJ3 = $(round(sol.lJ3, sigdigits=6))")
            println("  Kernels    : lK1 = $(round(sol.lK1, sigdigits=6)), lK3 = $(round(sol.lK3, sigdigits=6))")

            # Mixture Breakdown
            println("  Mixture    : σ_side=$(round(sol.sigS, sigdigits=4)), μ_side=$(round(sol.muW, sigdigits=4))")
            println("  Weights    : Symmetric 0.5/0.5 (Bimodal)")

            if args["compare"] && label == "Target (δ=1.0)"
                c_lwt = target.class.lWT
                gap = 100 * (sol.lWT - c_lwt) / c_lwt
                println("  Comparison : Classical lWT = $(round(c_lwt, sigdigits=6)) (Gap: $(round(gap, digits=3))%)")
            end
        end

        println("-"^90)
        println("Learnability (VGA Target): Linear=$(round(target.vga.learnability1, sigdigits=4)), Cubic=$(round(target.vga.learnability3, sigdigits=4))")
        println("="^90 * "\n")
    end

    if !isempty(args["to"])
        open(args["to"], "w") do io
            JSON3.pretty(io, result; allow_inf=true)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end