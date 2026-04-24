#!/usr/bin/env julia

using JSON
using Plots
using Printf
using JLD2
using Plots.PlotMeasures

# Include the solver modules
include("FCS.jl")
using .FCS
include("FCS2Erf_Cubic.jl")
using .FCS2Erf_Cubic

# --- Configuration & Flags ---
cache_file = "learnability_cache.jld2"
recompute = "--recompute" in ARGS

# Parameters
d = 150.0
N_list = [200, 500, 1000, 1600]
P_list = exp10.(range(log10(10), log10(30000), length=20))

# --- Data Generation / Loading ---
if !recompute && isfile(cache_file)
    println("Loading data from cache: $cache_file")
    @load cache_file fcn3_data fcn2_data
else
    println("Running learnability analysis (Generating new data)...")
    fcn3_data = Dict()
    fcn2_data = Dict()

    for N in N_list
        chi = 50
        println("Processing N=$N")
        fcn3_mu1, fcn3_mu3 = Float64[], Float64[]
        fcn2_mu1, fcn2_mu3 = Float64[], Float64[]

        for P in P_list
            # FCN3 solver
            try
                print("FCN3: ")
                kappa, epsilon = 0.14, 0.03 * 6^0.5
                n1 = n2 = N
                b, lr, max_iter = 4.0 / (3.0 * π), 1e-6, 6_000_000
                delta_target = 1.0
                i0_target = [4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2),
                    4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 1 / d]

                exp_sol_target = FCS.nlsolve_solver(
                    i0_target; chi=chi, d=d, kappa=kappa, delta=delta_target,
                    epsilon=epsilon, n1=n1, n2=n2, b=b, P=P, lr=lr,
                    max_iter=max_iter, verbose=false, anneal=true,
                    anneal_steps=300, tol=1e-8
                )

                if exp_sol_target !== nothing
                    l1, l3 = FCS.compute_lK_ratio(exp_sol_target, P, n1, n2, chi, d, delta_target, kappa, epsilon, b)
                    print("Learnability1: $l1, l3: $l3, P=$P,N=$N,d=$d,chi=$chi\n")
                    push!(fcn3_mu1, l1)
                    push!(fcn3_mu3, l3)
                else
                    print("FCN3 solver did not converge for P=$P, N=$N\n")
                    push!(fcn3_mu1, NaN)
                    push!(fcn3_mu3, NaN)
                end
            catch e
                push!(fcn3_mu1, NaN)
                push!(fcn3_mu3, NaN)
            end

            # FCN2 solver
            try
                print("FCN2: ")
                kappa, epsilon = 0.14, 0.03 * 6^0.5
                n1 = N
                b, lr, max_iter = 4.0 / (3.0 * π), 1e-6, 6_000_000
                params_target = FCS2Erf_Cubic.ProblemParams2(
                    d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon),
                    P=Float32(P), n1=Float32(N), χ=Float32(chi), b=Float32(4 / (3 * pi)), δ=Float32(1.0)
                )
                sol_target = FCS2Erf_Cubic.solve_FCN2_Erf(
                    params_target, [1.0 / d, 1.0 / d^3, 1.0 / d];
                    lr=1e-6, max_iter=6_000_000, tol=1e-12, verbose=false, use_anneal=true
                )
                push!(fcn2_mu1, sol_target.learnability1)
                push!(fcn2_mu3, sol_target.learnability3)
                print("Learnability1: $(sol_target.learnability1), l3: $(sol_target.learnability3), P=$P,N=$N,d=$d,chi=$chi\n")
            catch e
                push!(fcn2_mu1, NaN)
                push!(fcn2_mu3, NaN)
            end
        end
        fcn3_data[N] = Dict("P" => P_list, "mu1" => fcn3_mu1, "mu3" => fcn3_mu3)
        fcn2_data[N] = Dict("P" => P_list, "mu1" => fcn2_mu1, "mu3" => fcn2_mu3)
    end

    @save cache_file fcn3_data fcn2_data
    println("Results cached to $cache_file")
end

# --- Plotting ---
# Using :turbo or :viridis for better visibility on white backgrounds
c_scheme = :turbo
n_min, n_max = minimum(N_list), maximum(N_list)
alpha_list = log.(P_list) / log(d)

common_opts = (
    grid=false,
    frame=:box,
    xlabel="α",
    thickness_scaling=1.1,
    margin=5mm
)

p1 = plot(; common_opts..., ylabel="Ratio μ₁", title="He₁ Learnability")
p2 = plot(; common_opts..., ylabel="Ratio μ₃", title="He₃ Learnability")

# Hidden scatter for the Colorbar (Labeled N: Layer Width)
scatter!(p1, [NaN], [NaN], zcolor=[n_min, n_max], clabel="N (Layer Width)",
    c=c_scheme, label="", colorbar=true)

for (i, N) in enumerate(N_list)
    # Map N to color
    normalized_val = (N - n_min) / (n_max - n_min == 0 ? 1 : n_max - n_min)
    line_color = cgrad(c_scheme)[normalized_val]

    # Legend labels only for first pass
    l3 = (i == 1) ? "FCN3 (Solid)" : ""
    l2 = (i == 1) ? "FCN2 (Dash)" : ""

    plot!(p1, alpha_list, fcn3_data[N]["mu1"], color=line_color, lw=2, label=l3)
    plot!(p1, alpha_list, fcn2_data[N]["mu1"], color=line_color, lw=2, ls=:dash, label=l2)

    plot!(p2, alpha_list, fcn3_data[N]["mu3"], color=line_color, lw=2, label="")
    plot!(p2, alpha_list, fcn2_data[N]["mu3"], color=line_color, lw=2, ls=:dash, label="")
end

final_plot = plot(p1, p2, layout=(2, 1), size=(850, 900))
savefig(final_plot, "arxiv_learnability_plot.pdf")
println("Publication-ready plot saved as arxiv_learnability_plot.pdf")