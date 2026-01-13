#!/usr/bin/env julia
# compute_fcn2_erf_cubic_eigs.jl
# Compute FCN2 (erf) theoretical eigenvalues including cubic features using FCS2Erf_Cubic.
# Usage (CLI):
#   julia compute_fcn2_erf_cubic_eigs.jl --d 50 --n1 800 --P 600 --chi 80 --kappa 1.0 --delta 1.0
#   julia compute_fcn2_erf_cubic_eigs.jl --d 50 --n1 800 --P 600 --chi 80 --kappa 1.0 --delta 0.0 --to results.json

using ArgParse
using JSON3
using LinearAlgebra

push!(LOAD_PATH, @__DIR__)
include("FCS2Erf_Cubic.jl")
using .FCS2Erf_Cubic

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--d"
        help = "input dimension"
        arg_type = Float64
        required = true
        
        "--n1"
        help = "hidden width"
        arg_type = Float64
        required = true
        
        "--P"
        help = "number of samples"
        arg_type = Float64
        required = true
        
        "--chi"
        help = "chi scaling (defaults to n1)"
        arg_type = Float64
        default = NaN
        
        "--kappa"
        help = "kappa (regularization, defaults to 1/chi)"
        arg_type = Float64
        default = NaN
        
        "--delta"
        help = "delta (1.0 for target / 0.0 for perp)"
        arg_type = Float64
        default = 1.0
        
        "--epsilon"
        help = "epsilon (cubic target strength)"
        arg_type = Float64
        default = 0.00
        
        "--b"
        help = "erf constant b"
        arg_type = Float64
        default = 4 / (3 * pi)
        
        "--lr"
        help = "step size for solver"
        arg_type = Float64
        default = 1e-6
        
        "--max_iter"
        help = "max iterations for solver"
        arg_type = Int
        default = 1_000_000
        
        "--tol"
        help = "tolerance for solver"
        arg_type = Float64
        default = 1e-8
        
        "--anneal_steps"
        help = "annealing steps"
        arg_type = Int
        default = 30_000
        
        "--no-anneal"
        help = "disable annealing"
        action = :store_true
        
        "--TrSigma-hardcode"
        help = "hardcode TrSigma value (optional, solves 2D problem for lJ1 & lJ3)"
        arg_type = Float64
        default = NaN
        
        "--to"
        help = "path to save JSON results"
        arg_type = String
        default = ""
        
        "--quiet"
        help = "suppress stdout output"
        action = :store_true
        
        "--verbose"
        help = "verbose solver output"
        action = :store_true
    end
    return ArgParse.parse_args(ARGS, s)
end

function main()
    args = parse_cli_args()
    d = args["d"]
    n1 = args["n1"]
    P = args["P"]
    chi = isnan(args["chi"]) ? n1 : args["chi"]
    kappa = isnan(args["kappa"]) ? 1.0 / chi : args["kappa"]
    delta = args["delta"]
    epsilon = args["epsilon"]
    b = args["b"]
    quiet = args["quiet"]
    verbose = args["verbose"]
    TrSigma_hardcode = isnan(args["TrSigma-hardcode"]) ? nothing : args["TrSigma-hardcode"]
    
    # Initial guess: [lJ1, lJ3, lWT]
    init = [1.0 / d, 1.0 / d^3, 1.0 / d]
    
    # Solve for target (delta=1.0)
    params_target = FCS2Erf_Cubic.ProblemParams2(
        d=Float32(d),
        κ=Float32(kappa),
        ϵ=Float32(1.0),  # delta for target
        P=Float32(P),
        n1=Float32(n1),
        χ=Float32(chi),
        b=Float32(b)
    )
    
    sol_target = FCS2Erf_Cubic.solve_FCN2_Erf(
        params_target,
        init;
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=verbose,
        use_anneal=!args["no-anneal"],
        TrSigma_fixed=TrSigma_hardcode
    )
    
    # Solve for perpendicular (delta=0.0)
    params_perp = FCS2Erf_Cubic.ProblemParams2(
        d=Float32(d),
        κ=Float32(kappa),
        ϵ=Float32(0.0),  # delta for perp
        P=Float32(P),
        n1=Float32(n1),
        χ=Float32(chi),
        b=Float32(b)
    )
    
    sol_perp = FCS2Erf_Cubic.solve_FCN2_Erf(
        params_perp,
        init;
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=verbose,
        use_anneal=!args["no-anneal"],
        TrSigma_fixed=TrSigma_hardcode
    )
    
    # Compute normalized eigenvalues
    lJ_tot = sol_target.lJ1 + sol_perp.lJ1 * (d - 1)
    lJ1_norm = sol_target.lJ1 / lJ_tot
    lJ1P_norm = sol_perp.lJ1 / lJ_tot
    
    # Build result dictionary
    result = Dict(
        "parameters" => Dict(
            "d" => d,
            "n1" => n1,
            "P" => P,
            "chi" => chi,
            "kappa" => kappa,
            "epsilon" => epsilon,
            "b" => b
        ),
        "target" => Dict(
            "lJ1T" => sol_target.lJ1,
            "lJ3T" => sol_target.lJ3,
            "lK1T" => sol_target.lK1,
            "lK3T" => sol_target.lK3,
            "lWT" => sol_target.lWT,
            "learnability1" => sol_target.learnability1,
            "learnability3" => sol_target.learnability3
        ),
        "perpendicular" => Dict(
            "lJ1P" => sol_perp.lJ1,
            "lJ3P" => sol_perp.lJ3,
            "lK1P" => sol_perp.lK1,
            "lK3P" => sol_perp.lK3,
            "lWP" => 1.0 / d,
            "lWTP" => sol_perp.lWT,
            "learnability1" => sol_perp.learnability1,
            "learnability3" => sol_perp.learnability3
        ),
        "derived" => Dict(
            "lJ1_total" => lJ_tot,
            "lJ1T_normalized" => lJ1_norm,
            "lJ1P_normalized" => lJ1P_norm,
            "TrSigma_target" => sol_target.lWT + (d - 1) / d
        )
    )
    
    # Print results
    if !quiet
        println("\n" * "="^70)
        println("FCN2 Erf Cubic Eigenvalue Solver Results")
        println("="^70)
        println("Parameters: d=$d, n1=$n1, P=$P, χ=$chi, κ=$kappa, ε=$epsilon")
        println("\nTarget (δ=1.0):")
        println("  lJ1T = $(round(sol_target.lJ1, sigdigits=6)), lJ3T = $(round(sol_target.lJ3, sigdigits=6))")
        println("  lK1T = $(round(sol_target.lK1, sigdigits=6)), lK3T = $(round(sol_target.lK3, sigdigits=6))")
        println("  lWT = $(round(sol_target.lWT, sigdigits=6))")
        println("  Learnability: linear=$(round(sol_target.learnability1, sigdigits=4)), cubic=$(round(sol_target.learnability3, sigdigits=4))")
        
        println("\nPerpendicular (δ=0.0):")
        println("  lJ1P = $(round(sol_perp.lJ1, sigdigits=6)), lJ3P = $(round(sol_perp.lJ3, sigdigits=6))")
        println("  lK1P = $(round(sol_perp.lK1, sigdigits=6)), lK3P = $(round(sol_perp.lK3, sigdigits=6))")
        println("  lWTP = $(round(sol_perp.lWT, sigdigits=6))")
        println("  Learnability: linear=$(round(sol_perp.learnability1, sigdigits=4)), cubic=$(round(sol_perp.learnability3, sigdigits=4))")
        
        println("\nDerived:")
        println("  Normalized lJ1T = $(round(lJ1_norm, sigdigits=6))")
        println("  Normalized lJ1P = $(round(lJ1P_norm, sigdigits=6))")
        println("  TrΣ (target) = $(round(result["derived"]["TrSigma_target"], sigdigits=6))")
        println("="^70 * "\n")
    end
    
    # Save to file if requested
    if !isempty(args["to"])
        open(args["to"], "w") do io
            JSON3.pretty(io, result; allow_inf=true)
        end
        if !quiet
            println("Results saved to $(args["to"])")
        end
    end
    
    # Always output JSON to stdout for piping
    if isempty(args["to"])
        println(JSON3.write(result; allow_inf=true))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
