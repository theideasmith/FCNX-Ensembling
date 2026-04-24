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
cache_file = "learnability_fixed_N800_chi_cache.jld2"
plots_dir = normpath(joinpath(@__DIR__, "..", "plots"))
mkpath(plots_dir)
recompute = "--recompute" in ARGS

# Parameters
d = 150
N = 800
P_list = exp10.(range(log10(1), log10(300^3.0), length=50))
chi_list = [1.0, 10.0, N / 10.0, Float64(N)]
chi_labels = [L"\chi=1", L"\chi=10", L"\chi=N/10", L"\chi=N"]
chi_colors = [:blue, :green, :red, :orange]
global kappa
kappa, epsilon = 1.4, 0.03 * 6^0.5

# --- Data Generation / Loading ---
if !recompute && isfile(cache_file)
    println("Loading data from cache: $cache_file")
    @load cache_file fcn3_data fcn2_data
else
    println("Running learnability analysis (Generating new data)...")
    fcn3_data = Dict{Float64, Dict{String, Any}}()
    fcn2_data = Dict{Float64, Dict{String, Any}}()

    for chi in chi_list
        println("Processing N=$N, chi=$chi")
        fcn3_mu1, fcn3_mu3 = Float64[], Float64[]
        fcn3_lH1T, fcn3_lH3T = Float64[], Float64[]
        fcn3_lH1P, fcn3_lH3P = Float64[], Float64[]
        fcn3_lWT, fcn3_lWP = Float64[], Float64[]
        fcn2_mu1, fcn2_mu3 = Float64[], Float64[]
        fcn2_lH1T, fcn2_lH3T = Float64[], Float64[]
        fcn2_lH1P, fcn2_lH3P = Float64[], Float64[]
        fcn2_lWT, fcn2_lWP = Float64[], Float64[]

        for P in P_list
            # FCN3 solver
            print("FCN3: ")
            n1 = n2 = N
            b, lr, max_iter = 4.0 / (3.0 * π), 1e-6, 6_000_000
            delta_target = 1.0
            i0_target = [4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2),
                4 / (3 * π) * 1 / d^0.5, 1 / d^(3 / 2), 1 / d]

            exp_sol_target = FCS.nlsolve_solver(
                i0_target; chi=chi, d=d, kappa=kappa, delta=delta_target,
                epsilon=epsilon, n1=n1, n2=n2, b=b, P=P, lr=lr,
                max_iter=max_iter, verbose=false, anneal=true,
                anneal_steps=3000, tol=1e-8
            )

            l1, l3 = FCS.compute_lK_ratio(exp_sol_target, P, n1, n2, chi, d, delta_target, kappa, epsilon, b)
            lJ1T, lJ3T, lH1T, lH3T, lWT = exp_sol_target
            print("Learnability1: $l1, l3: $l3, P=$P,N=$N,d=$d,chi=$chi\n")
            push!(fcn3_mu1, l1)
            push!(fcn3_mu3, l3)
            push!(fcn3_lH1T, lH1T)
            push!(fcn3_lH3T, lH3T)
            push!(fcn3_lWT, lWT)

            # FCN3 perpendicular
            delta_perp = 0.0
            i0_perp = [4 / (3 * π) * 1 / d, 1 / d^3,
                4 / (3 * π) * 1 / d, 1 / d^3, 1 / d]
            exp_sol_perp = FCS.nlsolve_solver(
                i0_perp; chi=chi, d=d, kappa=kappa, delta=delta_perp,
                epsilon=epsilon, n1=n1, n2=n2, b=b, P=P, lr=lr,
                max_iter=max_iter, verbose=false, anneal=true,
                anneal_steps=3000, tol=1e-8
            )
            lJ1P, lJ3P, lH1P, lH3P, lWP = exp_sol_perp
            push!(fcn3_lH1P, lH1P)
            push!(fcn3_lH3P, lH3P)
            push!(fcn3_lWP, lWP)

            # FCN2 solver (target)
            params_target = FCS2Erf_Cubic.ProblemParams2(
                d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon),
                P=Float32(P), n1=Float32(N), χ=Float32(chi), b=Float32(4 / (3 * pi)), δ=Float32(1.0)
            )
            sol_target = FCS2Erf_Cubic.solve_FCN2_Erf(
                params_target, [1.0 / d, 1.0 / d^3, 1.0 / d];
                lr=1e-6, max_iter=6_000_000, tol=1e-8, verbose=false, use_anneal=true, anneal_P=true
            )
            push!(fcn2_mu1, sol_target.learnability1)
            push!(fcn2_mu3, sol_target.learnability3)
            push!(fcn2_lH1T, sol_target.lJ1)
            push!(fcn2_lH3T, sol_target.lJ3)
            push!(fcn2_lWT, sol_target.lWT)
            print("Learnability1: $(sol_target.learnability1), l3: $(sol_target.learnability3), P=$P,N=$N,d=$d,chi=$chi\n")

            # FCN2 perpendicular
            params_perp = FCS2Erf_Cubic.ProblemParams2(
                d=Float32(d), κ=Float32(kappa), ϵ=Float32(epsilon), δ=Float32(0.0),
                P=Float32(P), n1=Float32(N), χ=Float32(chi), b=Float32(4 / (3 * pi))
            )
            sol_perp = FCS2Erf_Cubic.solve_FCN2_Erf(
                params_perp, [1.0 / d, 1.0 / d^3, 1.0 / d];
                lr=1e-6, max_iter=6_000_000, tol=1e-8, verbose=false, use_anneal=true, anneal_P=true
            )
            push!(fcn2_lH1P, sol_perp.lJ1)
            push!(fcn2_lH3P, sol_perp.lJ3)
            push!(fcn2_lWP, sol_perp.lWT)
        end

        fcn3_data[chi] = Dict(
            "P" => P_list,
            "mu1" => fcn3_mu1,
            "mu3" => fcn3_mu3,
            "lH1T" => fcn3_lH1T,
            "lH3T" => fcn3_lH3T,
            "lH1P" => fcn3_lH1P,
            "lH3P" => fcn3_lH3P,
            "lWT" => fcn3_lWT,
            "lWP" => fcn3_lWP,
        )
        fcn2_data[chi] = Dict(
            "P" => P_list,
            "mu1" => fcn2_mu1,
            "mu3" => fcn2_mu3,
            "lH1T" => fcn2_lH1T,
            "lH3T" => fcn2_lH3T,
            "lH1P" => fcn2_lH1P,
            "lH3P" => fcn2_lH3P,
            "lWT" => fcn2_lWT,
            "lWP" => fcn2_lWP,
        )
    end

    @save cache_file fcn3_data fcn2_data
    println("Results cached to $cache_file")
end

using Plots, Measures, LaTeXStrings
# --- Plotting ---
alpha_list = log.(P_list) / log(d)

# 1. Subplot options: Legend must be false here to avoid duplicates
common_opts = (
    grid=false,
    frame=:box,
    xlabel=L"$\alpha$",
    thickness_scaling=1.1,
    margin=5mm,
    legend=false,
)

p1 = plot(; common_opts..., ylabel=L"Ratio $\mu_1$", title=L"He$_1$ Learnability")
p2 = plot(; common_opts..., ylabel=L"Ratio $\mu_3$", title=L"He$_3$ Learnability")
p3 = plot(; common_opts..., ylabel=L"Eigenvalue $\lambda_1$", title=L"He$_1$ Eigenvalues", yscale=:log10)
p4 = plot(; common_opts..., ylabel=L"Eigenvalue $\lambda_3$", title=L"He$_3$ Eigenvalues", yscale=:log10)

# 2. Create the dedicated legend plot
# Use :inside instead of :center
p_legend = plot(axis=false, grid=false, showaxis=false, ticks=false, legend_font_pointsize=15, legend=:inside, background_color_inside=:transparent)

for (i, chi) in enumerate(chi_list)
    line_color = chi_colors[i]

    # Plot data into main panels
    plot!(p1, alpha_list, fcn3_data[chi]["mu1"], color=line_color, lw=2)
    plot!(p1, alpha_list, fcn2_data[chi]["mu1"], color=line_color, lw=2, ls=:dash)

    plot!(p2, alpha_list, fcn3_data[chi]["mu3"], color=line_color, lw=2)
    plot!(p2, alpha_list, fcn2_data[chi]["mu3"], color=line_color, lw=2, ls=:dash)

    plot!(p3, alpha_list, fcn3_data[chi]["lH1T"], color=line_color, lw=2)
    plot!(p3, alpha_list, fcn2_data[chi]["lH1T"], color=line_color, lw=2, ls=:dash)
    # plot!(p3, alpha_list, fcn3_data[chi]["lH1P"], color=line_color, lw=2, ls=:dot)
    # plot!(p3, alpha_list, fcn2_data[chi]["lH1P"], color=line_color, lw=2, ls=:dashdot)

    plot!(p4, alpha_list, fcn3_data[chi]["lH3T"], color=line_color, lw=2)
    plot!(p4, alpha_list, fcn2_data[chi]["lH3T"], color=line_color, lw=2, ls=:dash)
    # plot!(p4, alpha_list, fcn3_data[chi]["lH3P"], color=line_color, lw=2, ls=:dot)
    # plot!(p4, alpha_list, fcn2_data[chi]["lH3P"], color=line_color, lw=2, ls=:dashdot)

    # 3. Add Legend Entries to the dummy p_legend plot
    if i == 1
        plot!(p_legend, [NaN], [NaN], color=:black, lw=2, label="FCN3 T (Solid)")
        plot!(p_legend, [NaN], [NaN], color=:black, lw=2, ls=:dash, label="FCN2 T (Dash)")
        plot!(p_legend, [NaN], [NaN], color=:black, lw=2, ls=:dot, label="FCN3 P (Dot)")
        plot!(p_legend, [NaN], [NaN], color=:black, lw=2, ls=:dashdot, label="FCN2 P (DashDot)")
    end
    plot!(p_legend, [NaN], [NaN], color=line_color, lw=3, label="$(chi_labels[i])")
end

# 4. Final Composition
# This layout puts the 4 plots in a 2x2 grid and the legend in a slim sidebar
lay = @layout [grid(2, 2) l{0.3w}]

final_plot = plot(p1, p2, p3, p4, p_legend, layout=lay, size=(1200, 900))

# Save
savefig(final_plot, joinpath(plots_dir, "arxiv_learnability_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf"))
println("Publication-ready plot saved successfully.")



# New plot for ratios lH1T / lH1P and lH3T / lH3P
pr1 = plot(; common_opts..., ylabel=L"Ratio $\lambda_1^T / \lambda_1^\perp$", title=L"He$_1$ Eigenvalue Ratios")
pr2 = plot(; common_opts..., ylabel=L"Ratio $\lambda_3^T / \lambda_3^\perp$", title=L"He$_3$ Eigenvalue Ratios")

for (i, chi) in enumerate(chi_list)
    line_color = chi_colors[i]

    lr3 = (i == 1) ? "FCN3" : ""
    lr2 = (i == 1) ? "FCN2" : ""

    ratio1_fcn3 = fcn3_data[chi]["lH1T"] ./ fcn3_data[chi]["lH1P"]
    ratio3_fcn3 = fcn3_data[chi]["lH3T"] ./ fcn3_data[chi]["lH3P"]
    plot!(pr1, alpha_list, ratio1_fcn3, color=line_color, lw=2, label=lr3, yscale=:log10)
    plot!(pr2, alpha_list, ratio3_fcn3, color=line_color, lw=2, label="", yscale=:log10)

    ratio1_fcn2 = fcn2_data[chi]["lH1T"] ./ fcn2_data[chi]["lH1P"]
    ratio3_fcn2 = fcn2_data[chi]["lH3T"] ./ fcn2_data[chi]["lH3P"]
    plot!(pr1, alpha_list, ratio1_fcn2, color=line_color, lw=2, ls=:dash, label=lr2, yscale=:log10)
    plot!(pr2, alpha_list, ratio3_fcn2, color=line_color, lw=2, ls=:dash, label="", yscale=:log10)
end

for (i, chi) in enumerate(chi_list)
    plot!(pr1, [NaN], [NaN], color=chi_colors[i], lw=3, label="$(chi_labels[i])")
end

ratio_plot = plot(pr1, pr2, layout=(2, 1), legend=:outerright, size=(1200, 600))
savefig(ratio_plot, joinpath(plots_dir, "eigenvalue_ratios_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf"))
println("Eigenvalue ratios plot saved as eigenvalue_ratios_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf")

# New plot for lWT and lWP scaling with P
pw1 = plot(; common_opts..., ylabel=L"$\lambda_\Sigma^T$", title=L"Target Width Eigenvalues", yscale=:log10)
pw2 = plot(; common_opts..., ylabel=L"$\lambda_\Sigma^\perp$", title=L"Perpendicular Width Eigenvalues", yscale=:log10)

for (i, chi) in enumerate(chi_list)
    line_color = chi_colors[i]

    lw3 = (i == 1) ? "FCN3" : ""
    lw2 = (i == 1) ? "FCN2" : ""

    plot!(pw1, alpha_list, fcn3_data[chi]["lWT"], color=line_color, lw=2, label=lw3)
    plot!(pw1, alpha_list, fcn2_data[chi]["lWT"], color=line_color, lw=2, ls=:dash, label=lw2)

    plot!(pw2, alpha_list, fcn3_data[chi]["lWP"], color=line_color, lw=2, label="")
    plot!(pw2, alpha_list, fcn2_data[chi]["lWP"], color=line_color, lw=2, ls=:dash, label="")
end

for (i, chi) in enumerate(chi_list)
    plot!(pw1, [NaN], [NaN], color=chi_colors[i], lw=3, label="$(chi_labels[i])")
end

width_plot = plot(pw1, pw2, layout=(2, 1), legend=:outerright, size=(1200, 600))
savefig(width_plot, joinpath(plots_dir, "width_eigenvalues_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf"))
println("Width eigenvalues plot saved as width_eigenvalues_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf")

# New plot for lWT / lWP ratios
pwr = plot(; common_opts..., ylabel=L"$\lambda_\Sigma^* / \lambda_\Sigma^\perp$", title=L"$\Sigma$ Target-Perp Eigenvalue Ratios")

for (i, chi) in enumerate(chi_list)
    line_color = chi_colors[i]

    lwr3 = (i == 1) ? "FCN3" : ""
    lwr2 = (i == 1) ? "FCN2" : ""

    ratio_w_fcn3 = fcn3_data[chi]["lWT"] ./ fcn3_data[chi]["lWP"]
    ratio_w_fcn2 = fcn2_data[chi]["lWT"] ./ fcn2_data[chi]["lWP"]
    plot!(pwr, alpha_list, ratio_w_fcn3, color=line_color, lw=2, label=lwr3)
    plot!(pwr, alpha_list, ratio_w_fcn2, color=line_color, lw=2, ls=:dash, label=lwr2)
end

for (i, chi) in enumerate(chi_list)
    plot!(pwr, [NaN], [NaN], color=chi_colors[i], lw=3, label="$(chi_labels[i])")
end

width_ratio_plot = plot(pwr, size=(600, 400), legend=:outerright)
savefig(width_ratio_plot, joinpath(plots_dir, "width_eigenvalue_ratios_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf"))
println("Width eigenvalue ratios plot saved as width_eigenvalue_ratios_plot_d$(d)_N$(N)_kappa$(kappa)_chi_scan.pdf")