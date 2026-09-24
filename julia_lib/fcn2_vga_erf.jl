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
        help = "Hidden-weight prior scale w0: σ_w² = s0/d"
        "--a0"
        arg_type = Float64;
        default = 1.0
        help = "Readout prior scale sa0: multiplies A = a0 · ℓ_T / (n1 χ). a0=1 recovers classic."
        "--sa0"
        arg_type = Float64;
        default = NaN
        help = "Alias for --a0 (sigma_a^2). If set, overrides --a0."
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
        "--regularized"
        help = "Use m²-preconditioned stationarity residuals (same zeros, better Newton scale)"
        action = :store_true
        "--offdiag"
        help = "HO VGA with He1–He3 off-diagonal discrepancy (rotate teacher into K_H eigenbasis each residual)"
        action = :store_true
        "--matrix"
        help = "2×2 He1–He3 coupling, Q=[λ11 λ13; λ13 λ33], y=(1,ε). On by default for Laplace (V=prior+½cᵀGc). With --vga: matrix discrepancy VGA."
        action = :store_true
        "--saddle"
        help = "δ-well saddle: freeze σ=1e-10, drop variational entropy, solve [lJ1,lJ3,μ] only"
        action = :store_true
        "--laplace"
        help = "Laplace saddle (default mode): μ from V'=0, σ=1/√V''. He1–He3 matrix energy unless --linear."
        action = :store_true
        "--linear"
        help = "Laplace with the linear channel only: V=prior+A₁λ₁(w) (no He1–He3 coupling)"
        action = :store_true
        "--vga"
        help = "Use the variational (entropy) solver instead of the default matrix Laplace saddle. Implied by --offdiag/--advanced/--regularized."
        action = :store_true
        "--laplace-mean"
        help = "Mean-only Laplace: σ frozen at 1e-10, solve [lJ1,lJ3,μ] from V'(μ)=0 only (mode locations of V)"
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
    a0 = isnan(args["sa0"]) ? args["a0"] : args["sa0"]
    advanced = args["advanced"]
    regularized = args["regularized"]
    offdiag = args["offdiag"]
    saddle = args["saddle"]
    laplace_mean = args["laplace-mean"]
    vga = args["vga"] || offdiag || advanced || regularized
    # VGA leaves μ weakly pinned (entropy vs prior); the matrix Laplace saddle
    # ties μ to feature learning and keeps the He1–He3 kernel coupling.
    laplace = args["laplace"] || laplace_mean || !(vga || saddle)
    matrix = args["matrix"] || (laplace && !args["linear"])
    entropy_rule = args["entropy-rule"]
    if saddle && laplace
        error("Choose only one of --saddle and --laplace/--laplace-mean")
    end
    if vga && laplace
        error("--vga/--offdiag/--advanced/--regularized select VGA; drop --laplace/--laplace-mean")
    end

    # --- Initial Guesses ---
    # VGA Bimodal / Laplace (4D): [lJ1, lJ3, sigS, muW]
    # Saddle δ-wells (3D): [lJ1, lJ3, muW]
    # μ₀≈0.1 often collapses onto the μ=0 saddle even when a deep double-well
    # exists. Try a moderate μ grid and keep the largest-|μ| finite solution.
    init_mix = saddle ? [1.0/d, 1.0/d^3, 0.5] : [1.0/d, 1.0/d^3, sqrt(0.8/d), 0.1]

    # Classical (3D): [lJ1, lJ3, lWT]
    init_class = [1.0/d, 1.0/d^3, 1.0/d]

    function solve_vga_best(δ_val)
        vga_params = FCS2_VGA.ProblemParams2(
            d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon),
            P=Float32(P), n1=Float32(n1), χ=Float32(chi), δ=Float32(δ_val),
            s0=Float32(s0), a0=Float32(a0),
        )
        if saddle
            return FCS2_VGA.solve_FCN2_Erf(
                vga_params, init_mix;
                anneal_steps=args["anneal_steps"], use_anneal=(!args["no-anneal"]),
                advanced=advanced, regularized=regularized, offdiag=offdiag,
                matrix=matrix, freeze_U=true, entropy_rule=entropy_rule,
                saddle=true, laplace=false,
            )
        end
        if laplace
            # χ-anneal tracks the μ=0 root; solve directly from a μ₀ grid and
            # keep the largest converged |μ| (deepest self-consistent well).
            best = nothing
            T_lap = 1.0 + 2.0 * (d - 1) * s0 / d
            σ_grid = laplace_mean ? (FCS2_VGA.SIG_SADDLE_DEFAULT,) : (0.05, 0.15, 0.3)
            for μ0 in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.3, 2.0), σ0 in σ_grid
                l1_0 = FCS2_VGA.compute_lambda1(μ0, σ0, T_lap)
                g0 = laplace_mean ? [l1_0, 1.0/d^3, μ0] : [l1_0, 1.0/d^3, σ0, μ0]
                cand = FCS2_VGA.solve_FCN2_Erf(
                    vga_params, g0;
                    anneal_steps=1, use_anneal=false,
                    laplace=!laplace_mean, laplace_mean=laplace_mean,
                    matrix=matrix,
                )
                isnan(cand.muW) && continue
                x_c = laplace_mean ? [cand.lJ1, cand.lJ3, cand.muW] :
                    [cand.lJ1, cand.lJ3, cand.sigS, cand.muW]
                r = FCS2_VGA.residuals_fcn2_laplace(
                    x_c, P, chi, d, kappa, δ_val, n1, s0, epsilon;
                    a0=a0, mean_only=laplace_mean, matrix=matrix,
                )
                sqrt(sum(abs2, r)) < 1e-5 || continue
                if best === nothing || abs(cand.muW) > abs(best.muW)
                    best = cand
                end
            end
            best === nothing || return best
        end
        guesses = Any[
            [1.0/d, 1.0/d^3, max(sqrt(0.4/d), 0.12), 0.25],
            [1.0/d, 1.0/d^3, max(sqrt(0.4/d), 0.12), 0.35],
            [1.0/d, 1.0/d^3, 0.20, 0.40],
            [1.0/d, 1.0/d^3, 0.22, 0.45],
            [1.0/d, 1.0/d^3, 0.18, 0.30],
            init_mix,
        ]
        best = nothing
        for g in guesses
            cand = FCS2_VGA.solve_FCN2_Erf(
                vga_params, Float64.(g);
                anneal_steps=args["anneal_steps"], use_anneal=(!args["no-anneal"]),
                advanced=advanced, regularized=regularized, offdiag=offdiag,
                matrix=matrix, freeze_U=true, entropy_rule=entropy_rule,
                saddle=false, laplace=laplace && !laplace_mean, laplace_mean=laplace_mean,
            )
            isnan(cand.muW) && continue
            if best === nothing || abs(cand.muW) > abs(best.muW)
                best = cand
            end
            abs(cand.muW) > 0.2 && break
        end
        return best === nothing ? FCS2_VGA.solve_FCN2_Erf(
            vga_params, init_mix;
            anneal_steps=args["anneal_steps"], use_anneal=(!args["no-anneal"]),
            advanced=advanced, regularized=regularized, offdiag=offdiag,
            matrix=matrix, freeze_U=true, entropy_rule=entropy_rule,
            saddle=false, laplace=laplace && !laplace_mean, laplace_mean=laplace_mean,
        ) : best
    end

    function solve_system(δ_val)
        # 1. Solve Bimodal Mixture (VGA)
        sol_vga = solve_vga_best(δ_val)

        # 2. Solve Classical (Laplace) if requested
        sol_class = nothing
        if args["compare"]
            # Classical cubic has no a0; leave χ as train χ (a0 enters VGA only).
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
            "d"=>d, "n1"=>n1, "P"=>P, "chi"=>chi, "a0"=>a0, "s0"=>s0,
            "kappa"=>kappa,
            "advanced"=>advanced, "regularized"=>regularized,
            "offdiag"=>offdiag, "matrix"=>matrix, "freeze_U"=>true,
            "saddle"=>saddle, "laplace"=>laplace, "laplace_mean"=>laplace_mean,
            "entropy_rule"=>entropy_rule,
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
        resid_label = regularized ? "m²-preconditioned stationarity" : "raw stationarity"
        println("\n" * "="^90)
        chan = matrix ? "He1–He3 matrix V=prior+½cᵀGc" : "linear channel V=prior+A₁λ₁"
        if laplace_mean
            println("FCN2 mean-only Laplace ($chan; σ=0, V'(μ)=0)")
        elseif laplace
            println("FCN2 Laplace saddle ($chan; σ=1/√V'')")
        elseif saddle
            println("FCN2 δ-well saddle (σ→0, prior+energy only)")
        else
            println("FCN2 4D Bimodal Mixture Model (VGA) Results")
            println("Entropy: $entropy_label")
            println("Residuals: $resid_label")
        end
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