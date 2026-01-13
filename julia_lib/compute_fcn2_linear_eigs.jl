# compute_fcn2_erf_eigs.jl
# Compute FCN2 (erf) theoretical eigenvalues using FCS2Erf.
# Usage (CLI):
#   julia compute_fcn2_erf_eigs.jl --d 20 --n1 200 --P 400 --chi 200 --kappa 0.005 --delta 1.0
#   julia compute_fcn2_erf_eigs.jl --d 20 --n1 200 --P 400 --chi 200 --kappa 1.0 --delta 0.0

using ArgParse
using JSON3
using LinearAlgebra

push!(LOAD_PATH, @__DIR__)
include("FCS2Linear.jl")
using .FCS2Linear


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
            default = 4/(3*pi)
        "--lr"
            help = "step size for solver"
            arg_type = Float64
            default = 1e-3
        "--max_iter"
            help = "max iterations for solver"
            arg_type = Int
            default = 50_000
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
        "--output"
            help = "path to save JSON results"
            arg_type = String
            default = ""
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

    init = [1/d]
    sol = nlsolve_solver(
        init,
        chi=chi,
        d=d,
        kappa=kappa,
        delta=delta,
        n1=n1,
        P=P,
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=false,
        anneal=!args["no-anneal"],
        anneal_steps=args["anneal_steps"],
    )

    lJ = sol === nothing ? NaN : sol[1]

    sol = nlsolve_solver(
        init,
        chi=chi,
        d=d,
        kappa=kappa,
        delta=0.0,
        n1=n1,


        P=P,
        lr=args["lr"],
        max_iter=args["max_iter"],
        tol=args["tol"],
        verbose=false,
        anneal=!args["no-anneal"],
        anneal_steps=args["anneal_steps"],
    )

    lJP = sol === nothing ? NaN : sol[1]

    result = Dict(
        "d"=>d,
        "n1"=>n1,
        "P"=>P,
        "chi"=>chi,
        "kappa"=>kappa,
        "delta"=>0.0,
        "b"=>b,
        "lJ"=>lJ,
        "lJP"=>lJP,


    )

    println("Pretty printing results:")
    println(JSON3.pretty(result))
    if !isempty(args["output"])
        JSON3.write(args["output"], result; allow_inf=true)
        println("Saved to $(args["output"])" )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
