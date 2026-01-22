# compute_fcn2_erf_eigs.jl
# Compute FCN2 (erf) theoretical eigenvalues using FCS2Erf.
# Usage (CLI):
#   julia compute_fcn2_erf_eigs.jl --d 20 --n1 200 --P 400 --chi 200 --kappa 0.005 --delta 1.0
#   julia compute_fcn2_erf_eigs.jl --d 20 --n1 200 --P 400 --chi 200 --kappa 1.0 --delta 0.0

using ArgParse
using JSON3
using LinearAlgebra

push!(LOAD_PATH, @__DIR__)
include("FCS2Erf.jl")
using .FCS2Erf

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table s begin
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
        help = "kappa (defaults to 1/chi)"
        arg_type = Float64
        default = NaN
        "--delta"
        help = "delta (1.0 target / 0.0 perp)"
        arg_type = Float64
        default = 1.0
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
        default = 50_000
        "--tol"
        help = "tolerance for solver"
        arg_type = Float64
        default = 1e-7
        "--anneal_steps"
        help = "annealing steps"
        arg_type = Int
        default = 30_000
        "--no-anneal"
        help = "disable annealing"
        action = :store_true
        "--output"
        help = "path to save JSON results"
        arg_type = String
        default = ""
        "--print"
        help = "print JSON results to stdout"
        arg_type = Bool
        default = false
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
    b = args["b"]

    init = [1 / d, 1 / d, 1/d]
    sol = nlsolve_solver(
        init,
        chi=chi,
        d=d,
        kappa=kappa,
        delta=1.0,
        n1=n1,
        b=b,
        P=P,
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=false,
        anneal=!args["no-anneal"],
        anneal_steps=args["anneal_steps"],
    )

    # println("Solved for delta = $delta")
    # println("Solution: ", sol)
    solution = sol
    lJ = solution[1]
    lk = solution[2]
    lWT = solution[3]
    # lJ = sol === nothing ? NaN : sol[1]
    # lk = sol === nothing ? NaN : sol[2]
    # lwT = sol === nothing ? NaN : sol[3]
    sol = nlsolve_solver(
        init,
        chi=chi,
        d=d,
        kappa=kappa,
        delta=0.0,
        n1=n1,
        b=b,
        P=P,
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=false,
        anneal=!args["no-anneal"],
        anneal_steps=args["anneal_steps"],
    )
    solution = sol
    lJP = solution[1]
    lkp = solution[2]
    lWTP = solution[3]
    lJ_tot = lJ + lJP * (d-1)
    lJnorm = lJ / lJ_tot
    lJPnorm = lJP / lJ_tot
    result = Dict(
        "d" => d,
        "n1" => n1,
        "P" => P,
        "chi" => chi,
        "kappa" => kappa,
        "delta" => 0.0,
        "b" => b,
        "lJ" => lJ,
        "lJP" => lJP,
        "lk" => lk,
        "lkp" => lkp,
        "lWP" => 1 / d,
        "lWT" => lWT,
        "lJPnorm" => lJPnorm,
        "lJnorm" => lJnorm,
        "lWTP" => lWTP,
        "TrSigma_delta_1" => lWT + (d - 1) / d,
    )

    # println("Self Consistency Trace Sigmma")
    # println("TrSigma")
    # println( lWT + (d - 1) / d )
    # println("Normalized Eigenvalues (lJ )")
    # lJ_tot = lJ + lJP * (d-1)
    # lJnorm = lJ / lJ_tot
    # lJPnorm = lJP / lJ_tot
    # println("lJ norm (delta=1.0) = $lJnorm, lJ P norm = $lJPnorm")


    # println("Pretty printing results:")
    # # The eigenvalues are printed to 6 sigfigs by default; use JSON output for full precision
    # println("d = $(result["d"]), n1 = $(result["n1"]), P = $(result["P"]), chi = $(result["chi"]), kappa = $(result["kappa"]), b = $(result["b"])")
    # println("lJ (delta=1.0) = $(round(result["lJ"], sigdigits=6)), lk (delta=1.0) = $(round(result["lk"], sigdigits=6))")
    # println("lJ (delta=0.0) = $(round(result["lJP"], sigdigits=6)), lk (delta=0.0) = $(round(result["lkp"], sigdigits=6))") 

    if args["print"] == true
        println(JSON3.pretty(result))
    end
    if !isempty(args["output"])
        JSON3.write(args["output"], result; allow_inf=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()

end

# FCS2Erf.jl