# =============================================================
# FCS2_d_scaling_learnability.jl
#
# Focused sweep: how learnability (linear and cubic mode) scales
# with alpha = log_d(P), across 10 values of d, at fixed chi = N.
#
# N        = 4d
# kappa    = P / 150
# epsilon, delta, P_samples, alpha range: unchanged from the main sweep.
#
# Kept deliberately modular: one function builds the alpha/P grid,
# one function runs a single (d, solver) sweep, one function plots.
# Swapping the solver, the d-range, or the plot styling means editing
# exactly one function, not the whole file.
# =============================================================

include("FCS2_VGA.jl")
include("FCS2Erf_Cubic.jl")
using .FCS2_VGA
using .FCS2Erf_Cubic
using Plots
using LaTeXStrings

gr()

# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------
const OUTDIR = "FCS2 Theory Predictions"
isdir(OUTDIR) || mkpath(OUTDIR)
outpath(name) = joinpath(OUTDIR, name)

# ---------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------
const D_VALS = collect(range(50.0f0, 500.0f0, length=10))
const P_SAMPLES = 30
const EPSILON = 0.03f0
const DELTA = 1.0f0

# chi = N = 4d, kappa = P / 150 (recomputed per P point below)
N_of_d(d) = 4.0f0 * d

# ---------------------------------------------------------
# Build the (P, alpha) grid for a given d
#   P runs logarithmically from d to d^3, so alpha = log_d(P)
#   runs from 1 to 3.
# ---------------------------------------------------------
function alpha_grid(d::Float32, samples::Int)
    P_vals = exp.(range(log(d), log(d^3), length=samples))
    alpha_vals = log.(P_vals) ./ log(d)
    return P_vals, alpha_vals
end

# ---------------------------------------------------------
# Single-d, single-solver sweep. Returns (learnability_linear, learnability_cubic)
# as vectors over the alpha grid. `solver` is :vga or :regular.
# ---------------------------------------------------------
function run_sweep(d::Float32, solver::Symbol)
    P_vals, _ = alpha_grid(d, P_SAMPLES)
    N_val = N_of_d(d)

    learn_lin = Float64[]
    learn_cub = Float64[]

    if solver == :vga
        guess = [0.2, 0.01, 1.0, 0.05]
        for P in P_vals
            kappa = Float32(P) / 150.0f0
            params = FCS2_VGA.ProblemParams2(
                d=d, κ=1.0, ϵ=EPSILON, P=Float32(P), n1=N_val, χ=N_val, δ=DELTA
            )
            sol = FCS2_VGA.solve_FCN2_Erf(params, guess; anneal_steps=3000, use_anneal=true)
            push!(learn_lin, sol.learnability1)
            push!(learn_cub, sol.learnability3)
            if !isnan(sol.lJ1) && !isnan(sol.sigS)
                guess = [sol.lJ1, sol.lJ3, sol.sigS, sol.muW]
            end
        end

    elseif solver == :regular
        for P in P_vals
            kappa = Float32(P) / 150.0f0
            params = FCS2Erf_Cubic.ProblemParams2(
                d=d, κ=1.0  , ϵ=EPSILON, P=Float32(P), n1=N_val, χ=N_val, δ=1.0
            )
            sol = FCS2Erf_Cubic.solve_FCN2_Erf(
                params, [1.0 / d, 1.0 / d^3, 1.0 / d];
                lr=1e-6, max_iter=1_000_000, anneal_steps=30_000, use_anneal=true
            )
            push!(learn_lin, sol.learnability1)
            push!(learn_cub, sol.learnability3)
        end
    else
        error("solver must be :vga or :regular")
    end

    return learn_lin, learn_cub
end

# ---------------------------------------------------------
# Run the full sweep for one solver: 10 values of d, each producing
# a learnability-vs-alpha curve. Returns a matrix (P_SAMPLES x length(D_VALS))
# for each mode, plus the shared alpha grid (identical for every d since
# alpha = log_d(P) always spans [1, 3]).
# ---------------------------------------------------------
function sweep_all_d(solver::Symbol)
    _, alpha_vals = alpha_grid(D_VALS[1], P_SAMPLES)
    Y_lin = Array{Float64}(undef, P_SAMPLES, length(D_VALS))
    Y_cub = Array{Float64}(undef, P_SAMPLES, length(D_VALS))

    for (j, d) in enumerate(D_VALS)
        println("[$solver] d = $(round(d, digits=1)) ($(j)/$(length(D_VALS)))")
        lin, cub = run_sweep(d, solver)
        Y_lin[:, j] = lin
        Y_cub[:, j] = cub
    end

    return alpha_vals, Y_lin, Y_cub
end

# ---------------------------------------------------------
# Production-quality plot: one curve per d, colored by a continuous
# colormap over d (viridis), vector output for arXiv.
# ---------------------------------------------------------
function plot_by_d(alpha_vals, Y, d_vals, ylabel_str, title_str, fname)
    colors = palette(:viridis, length(d_vals))
    p = plot(
        size=(520, 400),
        dpi=300,
        framestyle=:box,
        grid=false,
        legend=:outerright,
        legendfontsize=7,
        guidefontsize=11,
        tickfontsize=9,
        titlefontsize=11,
        fontfamily="Computer Modern",
        xlabel=L"\alpha \ (P = d^\alpha)",
        ylabel=ylabel_str,
        title=title_str,
    )
    for (j, d) in enumerate(d_vals)
        plot!(p, alpha_vals, Y[:, j], color=colors[j], lw=1.8, label=L"d = %$(round(Int, d))")
    end
    savefig(p, outpath(fname))
    return p
end

# ---------------------------------------------------------
# Run and plot for both solvers
# ---------------------------------------------------------
for solver in (:vga, :regular)
    alpha_vals, Y_lin, Y_cub = sweep_all_d(solver)
    tag = solver == :vga ? "VGA" : "Regular"

    plot_by_d(alpha_vals, Y_lin, D_VALS,
        "Learnability", "$tag: Linear-Mode Learnability vs α (χ=N=4d)",
        "learnability_linear_by_d_$(solver).pdf")

    plot_by_d(alpha_vals, Y_cub, D_VALS,
        "Learnability", "$tag: Cubic-Mode Learnability vs α (χ=N=4d)",
        "learnability_cubic_by_d_$(solver).pdf")
end

println("All plots saved to \"$(OUTDIR)/\"")