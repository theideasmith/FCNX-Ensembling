#!/usr/bin/env julia

using JSON
using Plots
using LaTeXStrings
using Printf
using JLD2
using Plots.PlotMeasures

# Include the solver modules
include("FCS.jl")
using .FCS
include("FCS2Erf_Cubic.jl")
using .FCS2Erf_Cubic

# --- Configuration & Flags ---
# NOTE: If you changed the solver logic, delete the cache file manually:
# rm learnability_fixed_P_cache.jld2

d_list = 100:50:500
P_multipliers = [identity, x -> x^1.5, x -> x^2, x -> x^3]  # P = f(d)

cache_file = "learnability_fixed_P_cache.jld2"
plots_dir = normpath(joinpath(@__DIR__, "..", "plots"))
mkpath(plots_dir)

if isfile(cache_file)
    @load cache_file data_dict
    println("Loaded cached data from $cache_file")
else
    data_dict = Dict()

    for d in d_list
        println("\n--- Processing d=$d ---")
        N = 4 * d
        chi = N#d / 5  # Adjusted chi to scale with d

        P_list = [mult(d) for mult in P_multipliers]
        alpha_list = [log(p) / log(d) for p in P_list]

        fcn3_mu1, fcn3_mu3 = Float64[], Float64[]
        fcn2_mu1, fcn2_mu3 = Float64[], Float64[]
        fcn3_lWT, fcn3_lWP = Float64[], Float64[]
        fcn2_lWT, fcn2_lWP = Float64[], Float64[]
        fcn3_lH1T, fcn3_lH3T, fcn3_lH1P, fcn3_lH3P = Float64[], Float64[], Float64[], Float64[]
        fcn2_lH1T, fcn2_lH3T, fcn2_lH1P, fcn2_lH3P = Float64[], Float64[], Float64[], Float64[]

        for (idx, P) in enumerate(P_list)
            # Setup constants
            epsilon = 0.03 * sqrt(6.0)
            n1 = n2 = Float64(N)
            b = 4.0 / (3.0 * π)

            # CRITICAL: κ should usually be constant to see P-scaling effects
            kappa_val = 1.0

            # FCN3 solver
            print("FCN3 (α=$(round(alpha_list[idx], digits=2))): ")
            delta_target = 1.0
            # Initial guess [lJ1, lJ3, lH1, lH3, lWT]
            i0_target = [1.0 / d, 0.1 / d^3, 1.0 / d, 0.1 / d^3, 1.0 / d]

            # Enable both anneal (for chi) and anneal_P (for P)
            exp_sol_target = FCS.nlsolve_solver(
                i0_target; chi=chi, d=d, kappa=kappa_val, delta=delta_target,
                epsilon=epsilon, n1=n1, n2=n2, b=b, P=P,
                max_iter=5000, verbose=false,
                anneal=true, anneal_P=true, anneal_steps=10000, tol=1e-8
            )

            if isnothing(exp_sol_target)
                l1, l3, lWT, lH1T, lH3T = NaN, NaN, NaN, NaN, NaN
            else
                l1, l3 = FCS.compute_lK_ratio(exp_sol_target, P, n1, n2, chi, d, delta_target, kappa_val, epsilon, b)
                lJ1T, lJ3T, lH1T, lH3T, lWT = exp_sol_target
            end
            push!(fcn3_mu1, l1)
            push!(fcn3_mu3, l3)
            push!(fcn3_lWT, lWT)
            push!(fcn3_lH1T, lH1T)
            push!(fcn3_lH3T, lH3T)
            @printf("L1: %.4f, L3: %.4f\n", l1, l3)

            # FCN3 perpendicular
            delta_perp = 0.0
            i0_perp = [1.0 / d, 0.1 / d^3, 1.0 / d, 0.1 / d^3, 1.0 / d]
            exp_sol_perp = FCS.nlsolve_solver(
                i0_perp; chi=chi, d=d, kappa=kappa_val, delta=delta_perp,
                epsilon=epsilon, n1=n1, n2=n2, b=b, P=P,
                max_iter=5000, verbose=false,
                anneal=true, anneal_P=true, anneal_steps=10000, tol=1e-8
            )
            if isnothing(exp_sol_perp)
                lWP, lH1P, lH3P = NaN, NaN, NaN
            else
                lJ1P, lJ3P, lH1P, lH3P, lWP = exp_sol_perp
            end
            push!(fcn3_lWP, lWP)
            push!(fcn3_lH1P, lH1P)
            push!(fcn3_lH3P, lH3P)

            # FCN2 solver
            print("FCN2: ")
            params_fcn2 = FCS2Erf_Cubic.ProblemParams2(
                d=Float32(d), κ=Float32(kappa_val), ϵ=Float32(epsilon),
                P=Float32(P), n1=Float32(N), χ=Float32(chi), δ=Float32(1.0)
            )
            # FCN2 solve with log-space and P-annealing enabled
            sol_fcn2 = FCS2Erf_Cubic.solve_FCN2_Erf(
                params_fcn2, [1.0 / d, 1.0 / d^3, 1.0 / d];
                max_iter=5000, tol=1e-8, verbose=false,
                anneal_chi=true, anneal_P=true, anneal_steps=10000
            )

            push!(fcn2_mu1, sol_fcn2.learnability1)
            push!(fcn2_mu3, sol_fcn2.learnability3)
            push!(fcn2_lWT, sol_fcn2.lWT)
            push!(fcn2_lH1T, sol_fcn2.lJ1)
            push!(fcn2_lH3T, sol_fcn2.lJ3)
            @printf("L1: %.4f, L3: %.4f\n", sol_fcn2.learnability1, sol_fcn2.learnability3)

            # FCN2 perpendicular
            params_fcn2_perp = FCS2Erf_Cubic.ProblemParams2(
                d=Float32(d), κ=Float32(kappa_val), ϵ=Float32(epsilon),
                P=Float32(P), n1=Float32(N), χ=Float32(chi), δ=Float32(0.0)
            )
            sol_fcn2_perp = FCS2Erf_Cubic.solve_FCN2_Erf(
                params_fcn2_perp, [1.0 / d, 1.0 / d^3, 1.0 / d];
                max_iter=5000, tol=1e-8, verbose=false,
                anneal_chi=true, anneal_P=true, anneal_steps=10000
            )
            push!(fcn2_lWP, sol_fcn2_perp.lWT)
            push!(fcn2_lH1P, sol_fcn2_perp.lJ1)
            push!(fcn2_lH3P, sol_fcn2_perp.lJ3)
        end

        data_dict[d] = Dict(
            "alpha" => alpha_list,
            "fcn3_mu1" => fcn3_mu1, "fcn3_mu3" => fcn3_mu3,
            "fcn2_mu1" => fcn2_mu1, "fcn2_mu3" => fcn2_mu3,
            "fcn3_lWT" => fcn3_lWT, "fcn3_lWP" => fcn3_lWP,
            "fcn2_lWT" => fcn2_lWT, "fcn2_lWP" => fcn2_lWP,
            "fcn3_lH1T" => fcn3_lH1T, "fcn3_lH3T" => fcn3_lH3T,
            "fcn3_lH1P" => fcn3_lH1P, "fcn3_lH3P" => fcn3_lH3P,
            "fcn2_lH1T" => fcn2_lH1T, "fcn2_lH3T" => fcn2_lH3T,
            "fcn2_lH1P" => fcn2_lH1P, "fcn2_lH3P" => fcn2_lH3P
        )
    end

    @save cache_file data_dict
    println("Results cached to $cache_file")
end
# --- ArXiv Publication Styling ---
default(
    fontfamily="Computer Modern",
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    legendfontsize=9,
    linewidth=2.2,
    grid=false,
    frame=:box,
    thickness_scaling=1.0
)

# Colors for the 4 multipliers: P=d, P=d^1.5, P=d^2, P=d^3
colors = [:blue, :green, :orange, :red]
P_labels = [L"P=d", L"P=d^{1.5}", L"P=d^{2}", L"P=d^{3}"]

# Helper to extract a vector of data across d_list for a specific P multiplier (idx)
get_curve(dict, d_list, key, idx) = [dict[d][key][idx] for d in d_list]

# ==========================================================
# FIGURE 1: Primary Learnability Results (μ1 and μ3)
# ==========================================================
p_mu1 = plot(ylabel=L"Learnability $\mu_1$", xlabel=L"Dimension $d$")
p_mu3 = plot(ylabel=L"Learnability $\mu_3$", xlabel=L"Dimension $d$")

for i in 1:4
    col = colors[i]
    # μ1
    plot!(p_mu1, d_list, get_curve(data_dict, d_list, "fcn3_mu1", i), color=col, label="")
    plot!(p_mu1, d_list, get_curve(data_dict, d_list, "fcn2_mu1", i), color=col, ls=:dash, label="")

    # μ3
    plot!(p_mu3, d_list, get_curve(data_dict, d_list, "fcn3_mu3", i), color=col, label="")
    plot!(p_mu3, d_list, get_curve(data_dict, d_list, "fcn2_mu3", i), color=col, ls=:dash, label="")
end

# Combined Legend Construction
plot!(p_mu1, [0], [0], color=:black, lw=1.5, label="FCN3 (3-layer)", linealpha=0.6)
plot!(p_mu1, [0], [0], color=:black, lw=1.5, ls=:dash, label="FCN2 (2-layer)", linealpha=0.6)
for i in 1:4
    plot!(p_mu1, [0], [0], color=colors[i], label=P_labels[i])
end

fig1 = plot(p_mu1, p_mu3, layout=(1, 2), size=(950, 400), margin=5mm)
savefig(fig1, joinpath(plots_dir, "arxiv_mu_scaling.pdf"))


# ==========================================================
# FIGURE 2: Kernel Eigenvalue Alignment (T / Perp Ratios)
# ==========================================================
# This shows how training "pulls" the kernel modes
pr1 = plot(ylabel=L"Alignment $\lambda_1^T / \lambda_1^\perp$", xlabel=L"Dimension $d$", yscale=:log10)
pr3 = plot(ylabel=L"Alignment $\lambda_3^T / \lambda_3^\perp$", xlabel=L"Dimension $d$", yscale=:log10)

for i in 1:4
    col = colors[i]
    # Mode 1 ratio
    r1_3 = get_curve(data_dict, d_list, "fcn3_lH1T", i) ./ get_curve(data_dict, d_list, "fcn3_lH1P", i)
    r1_2 = get_curve(data_dict, d_list, "fcn2_lH1T", i) ./ get_curve(data_dict, d_list, "fcn2_lH1P", i)
    plot!(pr1, d_list, r1_3, color=col, label="")
    plot!(pr1, d_list, r1_2, color=col, ls=:dash, label="")

    # Mode 3 ratio
    r3_3 = get_curve(data_dict, d_list, "fcn3_lH3T", i) ./ get_curve(data_dict, d_list, "fcn3_lH3P", i)
    r3_2 = get_curve(data_dict, d_list, "fcn2_lH3T", i) ./ get_curve(data_dict, d_list, "fcn2_lH3P", i)
    plot!(pr3, d_list, r3_3, color=col, label="")
    plot!(pr3, d_list, r3_2, color=col, ls=:dash, label="")
end

fig2 = plot(pr1, pr3, layout=(1, 2), size=(950, 400), margin=5mm)
savefig(fig2, joinpath(plots_dir, "arxiv_eigenvalue_alignment.pdf"))


# ==========================================================
# FIGURE 3: Weight Alignment Ratio (Sigma Evolution)
# ==========================================================
# Explains why FCN3 outperforms FCN2 via inter-layer feedback
p_sig = plot(ylabel=L"Weight Alignment $\lambda_\Sigma^T / \lambda_\Sigma^\perp$",
    xlabel=L"Dimension $d$", title="Target Signal Feedback to Weights")

for i in 1:4
    col = colors[i]
    rsig_3 = get_curve(data_dict, d_list, "fcn3_lWT", i) ./ get_curve(data_dict, d_list, "fcn3_lWP", i)
    rsig_2 = get_curve(data_dict, d_list, "fcn2_lWT", i) ./ get_curve(data_dict, d_list, "fcn2_lWP", i)

    plot!(p_sig, d_list, rsig_3, color=col, label=P_labels[i])
    plot!(p_sig, d_list, rsig_2, color=col, ls=:dash, label="")
end

fig3 = plot(p_sig, size=(550, 450), margin=5mm, legend=:topright)
savefig(fig3, joinpath(plots_dir, "arxiv_sigma_alignment.pdf"))
# ==========================================================
# FIGURE 4: Scaling Ratio Beta
# ==========================================================
# Beta = log10(lH3T) / log10(lWT)
# This captures the relative scaling of higher-order features vs weights.

pb = plot(ylabel=L"Scaling Ratio $\beta$",
    xlabel=L"Dimension $d$",
    title=L"Feature Scaling $\beta = \log_{10}(\lambda_3^T) / \log_{10}(\lambda_\Sigma^T)$")

for i in 1:4
    col = colors[i]

    # Calculate Beta for FCN3
    lH3T_3 = get_curve(data_dict, d_list, "fcn3_lH3T", i)
    lWT_3 = get_curve(data_dict, d_list, "fcn3_lWT", i)
    # We use .abs because log(tiny_value) is negative; beta is the ratio of these magnitudes
    beta_3 = log10.(lH3T_3) ./ log10.(lWT_3)

    # Calculate Beta for FCN2
    lH3T_2 = get_curve(data_dict, d_list, "fcn2_lH3T", i)
    lWT_2 = get_curve(data_dict, d_list, "fcn2_lWT", i)
    beta_2 = log10.(lH3T_2) ./ log10.(lWT_2)

    plot!(pb, d_list, beta_3, color=col, label=P_labels[i])
    plot!(pb, d_list, beta_2, color=col, ls=:dash, label="")
end

fig4 = plot(pb, size=(600, 450), margin=5mm, legend=:outerright)
savefig(fig4, joinpath(plots_dir, "arxiv_beta_scaling.pdf"))

println("Beta scaling plot saved as arxiv_beta_scaling.pdf")
println("ArXiv-ready PDF plots generated.")



# --- ArXiv Publication Styling ---
default(
    fontfamily="Computer Modern",
    titlefontsize=16,
    guidefontsize=14,
    tickfontsize=12,
    legendfontsize=11,
    linewidth=3.0,      # Thick lines for readability
    grid=false,
    frame=:box,
    thickness_scaling=1.0,
    size=(700, 500)
)

# --- Configuration ---
d_list = 100:50:600
kappa_val = 1.0
const epsilon = 0.03 * sqrt(6.0)
cache_file = "d3_scaling_results.jld2"

# Colors: Blue for 3-layer, Orange for 2-layer
color_3 = :royalblue
color_2 = :darkorange

# Data containers
results = Dict()

if isfile(cache_file)
    @load cache_file results
    println("Loaded results from cache.")
else
    for d in d_list
        println("\n--- Solving d = $d ---")
        N = 4 * d             # Width N = 4d
        chi = N           # chi = N/10
        P = Float64(d)^3      # P ~ d^3
        
        # 1. Solve FCN3
        print("FCN3 Solve... ")
        # [lJ1, lJ3, lH1, lH3, lWT]
        i0_fcn3 = [1.0/d, 0.1/d^3, 1.0/d, 0.1/d^3, 1.0/d]
        sol_3 = FCS.nlsolve_solver(
            i0_fcn3; chi=chi, d=d, kappa=kappa_val, delta=1.0,
            epsilon=epsilon, n1=N, n2=N, b=4/(3π), P=P,
            anneal=true, anneal_P=true, anneal_steps=10000, tol=1e-9
        )
        
        # 2. Solve FCN2
        print("FCN2 Solve... ")
        params_2 = FCS2Erf_Cubic.ProblemParams2(
            d=Float32(d), κ=Float32(kappa_val), ϵ=Float32(epsilon),
            P=Float32(P), n1=Float32(N), χ=Float32(chi), δ=Float32(1.0)
        )
        sol_2_struct = FCS2Erf_Cubic.solve_FCN2_Erf(
            params_2, [1.0/d, 0.1/d^3, 1.0/d];
            anneal_chi=true, anneal_P=true, anneal_steps=10000, tol=1e-9
        )
        
        # Calculate Metrics
        # FCN3 Metrics
        mu1_3, mu3_3 = FCS.compute_lK_ratio(sol_3, P, N, N, chi, d, 1.0, kappa_val, epsilon, 4/(3π))
        beta_3 = log10(sol_3[4]) / log10(sol_3[5]) # log10(lH3T) / log10(lWT)
        
        # FCN2 Metrics
        mu1_2 = sol_2_struct.learnability1
        mu3_2 = sol_2_struct.learnability3
        beta_2 = log10(sol_2_struct.lJ3) / log10(sol_2_struct.lWT) # log10(lH3T) / log10(lWT)
        
        results[d] = (mu1_3, mu3_3, beta_3, mu1_2, mu3_2, beta_2)
        @printf("Done. mu1_3: %.3f, mu3_3: %.3f, beta_3: %.3f\n", mu1_3, mu3_3, beta_3)
    end
    @save cache_file results
end

# --- Extraction for Plotting ---
mu1_3_vec = [results[d][1] for d in d_list]
mu3_3_vec = [results[d][2] for d in d_list]
beta_3_vec = [results[d][3] for d in d_list]

mu1_2_vec = [results[d][4] for d in d_list]
mu3_2_vec = [results[d][5] for d in d_list]
beta_2_vec = [results[d][6] for d in d_list]

# --- Figure 1: Learnabilities ---
p1 = plot(xlabel=L"Dimension $d$", ylabel=L"Learnability $\mu_k$", title="Feature Learning at Sample Saturation")
label_str = L"P \sim d^3, \chi = N/10, N=4d"

# Mode 1 (Linear)
plot!(p1, d_list, mu1_3_vec, color=color_3, label="FCN3 (Linear Mode)", alpha=0.9)
plot!(p1, d_list, mu1_2_vec, color=color_2, label="FCN2 (Linear Mode)", alpha=0.9)

# Mode 3 (Cubic) - using dashed lines of same colors
plot!(p1, d_list, mu3_3_vec, color=color_3, ls=:dash, label="FCN3 (Cubic Mode)")
plot!(p1, d_list, mu3_2_vec, color=color_2, ls=:dash, label="FCN2 (Cubic Mode)")

# Annotate scaling
annotate!(p1, [(400, 0.4, text(label_str, 12, :black, :center))])

# --- Figure 2: Beta Scaling ---
p2 = plot(xlabel=L"Dimension $d$", ylabel=L"Scaling Ratio $\beta$", 
          title=L"Staircasification Exponent $\beta = \frac{\log_{10} \lambda_{He3}^T}{\log_{10} \lambda_\Sigma^T}$")

plot!(p2, d_list, beta_3_vec, color=color_3, label="FCN3 (3-layer)")
plot!(p2, d_list, beta_2_vec, color=color_2, label="FCN2 (2-layer)")

# Save plots
savefig(p1, joinpath(plots_dir, "arxiv_learnability_d3.pdf"))
savefig(p2, joinpath(plots_dir, "arxiv_beta_scaling_d3.pdf"))

println("\nArXiv figures generated successfully:")
println("1. arxiv_learnability_d3.pdf")
println("2. arxiv_beta_scaling_d3.pdf")