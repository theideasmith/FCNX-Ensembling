#!/usr/bin/env julia

using Plots
using LaTeXStrings
using JLD2
using Plots.PlotMeasures
using Printf

# Include our robust modules
include("FCS.jl")
using .FCS
include("FCS2Erf_Cubic.jl")
using .FCS2Erf_Cubic

# --- ArXiv Publication Styling ---
default(
    fontfamily="Computer Modern",
    titlefontsize=16,
    guidefontsize=14,
    tickfontsize=12,
    legendfontsize=10,
    linewidth=3.0,      
    grid=false,
    frame=:box,
    thickness_scaling=1.0,
    size=(750, 550)
)

# --- Configuration ---
d_list = 100:50:1200
kappa_val = 0.1
epsilon = 0.03 * sqrt(6.0)
cache_file = "d3_scaling_results_beta.jld2"
plots_dir = normpath(joinpath(@__DIR__, "..", "plots"))
println("Plots: will be saved to: $plots_dir")

mkpath(plots_dir)

# Colors: Blue for 3-layer, Orange for 2-layer
color_3 = :royalblue
color_2 = :darkorange

# Data containers
results = Dict()

if false #isfile(cache_file)
    @load cache_file results
    println("Loaded results from cache.")
else
    for d in d_list
        println("\n--- Solving d = $d ---")
        N = 4 * d             # Width N = 4d
        chi = N / 10          # chi = N/10
        P = Float64(d)^3      # P ~ d^3
        
        # 1. Solve FCN3
        print("FCN3 Solve... ")
        i0_fcn3 = [1.0/d, 0.1/d^3, 1.0/d, 0.1/d^3, 1.0/d]
        params_3 = FCS.ProblemParams(
            d=Float32(d), κ=Float32(kappa_val), ϵ=Float32(0.074),
            P=Float32(P), n1=Float32(N), n2=Float32(N), χ=Float32(chi), b=Float32(4 / (3 * π))
        )
        sol_3_struct = FCS.solve_FCN3_Erf(
            params_3, i0_fcn3;
            anneal=true, anneal_P=true, anneal_steps=10000, tol=1e-9,
            effective_ridge=false
        )
        
        # 2. Solve FCN2
        print("FCN2 Solve... ")
        params_2 = FCS2Erf_Cubic.ProblemParams2(
            d=Float32(d), κ=Float32(kappa_val), ϵ=Float32(0.074),
            P=Float32(P), n1=Float32(N), χ=Float32(chi), δ=Float32(1.0)
        )
        sol_2_struct = FCS2Erf_Cubic.solve_FCN2_Erf(
            params_2, [1.0/d, 0.1/d^3, 1.0/d];
            anneal_chi=true, anneal_P=true, anneal_steps=10000, tol=1e-9,
            effective_ridge=false
        )
        
        # Calculate Metrics
        # FCN3 Metrics
        lWT = sol_3_struct.lWT
        lWP = 1 / d
        TrSigma = lWT + (d - 1) * lWP
        gamma3 = ((16 / (π * (1 + 2 * TrSigma)^3) * (15))) / 6.0
        gamma1 = 4 / (π * (1 + 2 * TrSigma))
        mu1_3 = sol_3_struct.learnability1
        mu3_3 = sol_3_struct.learnability3
        print(sol_3_struct.lH3, " ", sol_3_struct.lH1, " ", sol_3_struct.lWT, " ")
        beta3_3 = log10(sol_3_struct.lH3) / log10(sol_3_struct.lWT)
        beta1_3 = log10(sol_3_struct.lH1) / log10(sol_3_struct.lWT)
        lWT_3 = sol_3_struct.lWT
        
        lWT = sol_2_struct.lWT
        lWP = 1 / d
        TrSigma = lWT + (d-1) * lWP
        gamma3 = ((16 / (π * (1 + 2 * TrSigma)^3) * (15))) / 6.0
        gamma1 = 4/ ( π * (1+ 2 * TrSigma))
        # FCN2 Metrics
        mu1_2 = sol_2_struct.learnability1
        mu3_2 = sol_2_struct.learnability3
        beta3_2 = log10(sol_2_struct.lJ3 ) / log10(sol_2_struct.lWT) 
        beta1_2 = log10(sol_2_struct.lJ1 ) / log10(sol_2_struct.lWT)
        lWT_2 = sol_2_struct.lWT
        
        results[d] = (mu1_3, mu3_3, beta3_3, beta1_3, mu1_2, mu3_2, beta3_2, beta1_2, lWT_3, lWT_2)
        @printf("Done. d=%d: beta3_3=%.3f, beta1_3=%.3f\n", d, beta3_3, beta1_3)
    end
    @save cache_file results
end

# --- Extraction for Plotting ---
mu1_3_v = [results[d][1] for d in d_list]; mu3_3_v = [results[d][2] for d in d_list]
beta3_3_v = [results[d][3] for d in d_list]; beta1_3_v = [results[d][4] for d in d_list]

mu1_2_v = [results[d][5] for d in d_list]; mu3_2_v = [results[d][6] for d in d_list]
beta3_2_v = [results[d][7] for d in d_list]; beta1_2_v = [results[d][8] for d in d_list]
lWT_ratio_v = [results[d][10] / results[d][9] for d in d_list]

# --- Figure 2: Scaling Exponents Beta ---
p2 = plot(xlabel=L"Dimension $d$", ylabel=L"Scaling Exponent $\beta_k$", 
          title=L"Staircasification: $\beta_k = \log_{10} \lambda_{Hek}^T / \log_{10} \lambda_\Sigma^T$",
          legend=:outerright)

label_info = L"$P \sim d^3, \chi = N/10, N=4d$"

# FCN3 (3-layer) - Blue
plot!(p2, d_list, beta3_3_v, color=color_3, ls=:solid, label=L"FCN3 $\beta_3$")
plot!(p2, d_list, beta1_3_v, color=color_3, ls=:dot,  label=L"FCN3 $\beta_1$")

# FCN2 (2-layer) - Orange
plot!(p2, d_list, beta3_2_v, color=color_2, ls=:solid, label=L"FCN2 $\beta_3$")
plot!(p2, d_list, beta1_2_v, color=color_2, ls=:dot,  label=L"FCN2 $\beta_1$")

# Annotate the configuration
annotate!(p2, [(350, (maximum(beta1_3_v)+minimum(beta3_2_v))/2, text(label_info, 10, :black, :center))])

# Save plot
savefig(p2, joinpath(plots_dir, "arxiv_beta_scaling_comparison_d3.pdf"))

println("\nArXiv scaling exponent plot generated: arxiv_beta_scaling_comparison_d3.pdf")

# --- Figure 3: Width Ratio lWT(FCN3) / lWT(FCN2) ---
p3 = plot(
    d_list,
    lWT_ratio_v,
    xlabel=L"Dimension $d$",
    ylabel=L"$\lambda_{\Sigma,\text{FCN2}}^\star / \lambda_{\Sigma,\text{FCN3}}^\star$",
    title="Target Width-Eigenvalue Ratio with Effective Ridge",
    color=:purple,
    lw=3,
    marker=:circle,
    markersize=4,
    label=L"$P\sim d^3,\;\chi=N/10,\;N=4d$",
    legend=:topright
)

savefig(p3, joinpath(plots_dir, "arxiv_lWT_ratio_fcn3_over_fcn2_d3.pdf"))
println("ArXiv lWT ratio plot generated: arxiv_lWT_ratio_fcn3_over_fcn2_d3.pdf")