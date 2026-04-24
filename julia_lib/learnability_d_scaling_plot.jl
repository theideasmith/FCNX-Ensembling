#!/usr/bin/env julia

using Plots
using LaTeXStrings
using JLD2

include("FCS.jl")
using .FCS

# --- Configuration ---
const D_LIST = [150.0, 200, 300, 500, 800, 1000]
const NUM_P_VALUES = 100
const KAPPA = 0.14
const EPSILON = 0.03 * sqrt(6.0)
const DELTA_TARGET = 1.0

const CACHE_FILE = joinpath(@__DIR__, "learnability_d_scaling_cache.jld2")
const PLOTS_DIR = normpath(joinpath(@__DIR__, "..", "plots"))
mkpath(PLOTS_DIR)

recompute = "--recompute" in ARGS

# --- Data Generation / Loading ---
learnability_data = Dict{Int, Dict{String, Vector{Float64}}}()

if !recompute && isfile(CACHE_FILE)
    println("Loading cached data from $(CACHE_FILE)")
    @load CACHE_FILE learnability_data
else
    println("Computing learnability scaling data...")

    for d in D_LIST
        d_int = Int(round(d))
        N = Int(round(3 * d))
        chi = N / 10

        P_values = exp10.(range(log10(d^1.0), log10(d^2.5), length=NUM_P_VALUES))
        alpha_values = log.(P_values) ./ log(d)

        mu1_values = Float64[]
        mu3_values = Float64[]

        println("Running d=$(d_int), N=$(N), chi=$(chi)")

        for P in P_values
            i0_target = [
                4 / (3 * pi) * 1 / sqrt(d),
                1 / d^(3 / 2),
                4 / (3 * pi) * 1 / sqrt(d),
                1 / d^(3 / 2),
                1 / d,
            ]

            exp_sol_target = FCS.nlsolve_solver(
                i0_target;
                chi=chi,
                d=d,
                kappa=KAPPA,
                delta=DELTA_TARGET,
                epsilon=EPSILON,
                n1=N,
                n2=N,
                b=4.0 / (3.0 * pi),
                P=P,
                lr=1e-6,
                max_iter=6_000_000,
                verbose=false,
                anneal=true,
                anneal_steps=3000,
                tol=1e-8,
            )

            l1, l3 = FCS.compute_lK_ratio(
                exp_sol_target,
                P,
                N,
                N,
                chi,
                d,
                DELTA_TARGET,
                KAPPA,
                EPSILON,
                4.0 / (3.0 * pi),
            )

            push!(mu1_values, l1)
            push!(mu3_values, l3)

            println("  P=$(round(P, sigdigits=6)) alpha=$(round(log(P) / log(d), digits=3)) mu1=$(round(l1, digits=6)) mu3=$(round(l3, digits=6))")
        end

        learnability_data[d_int] = Dict(
            "alpha" => collect(alpha_values),
            "mu1" => mu1_values,
            "mu3" => mu3_values,
        )
    end

    @save CACHE_FILE learnability_data
    println("Saved cache to $(CACHE_FILE)")
end

# --- Plotting ---
palette = cgrad(:viridis, length(D_LIST), categorical=true)

common_opts = (
    xlabel=L"$\\alpha$",
    ylabel="Learnability",
    ylim=(0.0, 1.0),
    grid=false,
    frame=:box,
    legend=:right,
    lw=2,
    marker=:circle,
    ms=3,
)

p_he1 = plot(; common_opts..., title=L"He$_1$ Learnability")
p_he3 = plot(; common_opts..., title=L"He$_3$ Learnability")

for (i, d) in enumerate(D_LIST)
    d_int = Int(round(d))
    N = Int(round(3 * d))
    series = learnability_data[d_int]
    color_i = palette[i]

    label = "d=$(d_int), N=$(N), chi=$(round(N/10, digits=2))"

    plot!(p_he1, series["alpha"], series["mu1"], color=color_i, label=label)
    plot!(p_he3, series["alpha"], series["mu3"], color=color_i, label=label)
end

he1_path = joinpath(PLOTS_DIR, "learnability_he1_d_scaling.pdf")
he3_path = joinpath(PLOTS_DIR, "learnability_he3_d_scaling.pdf")
combined_path = joinpath(PLOTS_DIR, "learnability_he1_he3_d_scaling.pdf")

savefig(p_he1, he1_path)
savefig(p_he3, he3_path)

combined = plot(p_he1, p_he3, layout=(2, 1), size=(1000, 900))
savefig(combined, combined_path)

println("Saved He1 plot: $(he1_path)")
println("Saved He3 plot: $(he3_path)")
println("Saved combined plot: $(combined_path)")
