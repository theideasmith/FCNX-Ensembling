using ForwardDiff
using LinearAlgebra
using Plots
using Colors
using Serialization
using Statistics
using LaTeXStrings
using Measures
using Printf

include("./FCSLinear.jl")
using .FCSLinear

#########################################################
# d-sweep eigensolver for linear FCN3 networks
#
# This script sweeps over input dimension `d` for different P scalings
# and computes the mean-field solutions for eigenvalue equations
# using the solvers provided by `FCSLinear.jl`.
#
# For each d the script records:
# - lH (Hessian eigenvalue)
# - lJ (NTK eigenvalue)
#
# Configurations:
# - P ~ d with chi=1.0 and chi=N
# - P ~ sqrt(d) with chi=1.0 and chi=N
#
# The script produces separate log-log plots for lH and lJ with
# fitted power-law slopes shown in the legend.
#########################################################

## User-editable sweep parameters
κ = 1.0                      # kappa (noise)
n_factor = 5.0               # n = n_factor * d (N = 4*d as requested)
δ = 0.0                      # delta for the eigenvalue equations (no distinction needed)
lr = 1e-5
Tf = 10_000                 # max iterations for solver

# Sweep range for d (input dimension): 2 to 100
d_values = collect(1:10:1000)

# Output folder for plots
plot_dir = joinpath(@__DIR__, "..", "plots", "d_sweep_linear")
mkpath(plot_dir)
results_file = joinpath(plot_dir, "d_sweep_linear_results.jls")

# Storage arrays
ND = length(d_values)

# P ~ d configurations
lH_Pd_chi1 = fill(NaN, ND)
lJ_Pd_chi1 = fill(NaN, ND)

lH_Pd_chiN = fill(NaN, ND)
lJ_Pd_chiN = fill(NaN, ND)

# P ~ sqrt(d) configurations
lH_Psqrtd_chi1 = fill(NaN, ND)
lJ_Psqrtd_chi1 = fill(NaN, ND)

lH_Psqrtd_chiN = fill(NaN, ND)
lJ_Psqrtd_chiN = fill(NaN, ND)

println("Starting d-sweep for linear FCN3 networks...")
println("d range: $(d_values[1]) to $(d_values[end])")
println("P scalings: P ~ d and P ~ sqrt(d)")
println("chi values: 1.0 and N")
println("delta: $δ (single solve per configuration)")
println()

for (i, d) in enumerate(d_values)
    P_linear = 4 * Float64(d)           # P ~ d
    P_sqrt = 5 * sqrt(Float64(d))       # P ~ sqrt(d)
    
    n = n_factor * d
    n1 = n
    n2 = n
    
    χ1 = 1.0        # chi = 1.0
    χN = Float64(n2)  # chi = N
    
    if i % 10 == 0 || i == 1
        println("Progress: d = $d ($(i)/$(ND))")
    end
    
    # Initial guess heuristics (scale with d)
    i0 = [1.0 / sqrt(d), 1.0 / sqrt(d)]  # [lJ, lH]
    
    # ========================================
    # P ~ d configurations
    # ========================================
    
    # P ~ d, chi=1.0
    try
        sol = FCSLinear.nlsolve_solver(i0;
            chi=χ1, d=d, kappa=κ, delta=1.0,
            n1=n1, n2=n2, P=P_linear, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=10_000)
        
        if sol !== nothing && length(sol) == 2
            lJ_Pd_chi1[i] = sol[1]
            lH_Pd_chi1[i] = sol[2]
        end
    catch e
        @warn "P~d, chi=1 failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # P ~ d, chi=N
    try
        sol = FCSLinear.nlsolve_solver(i0;
            chi=χN, d=d, kappa=κ, delta=1.0,
            n1=n1, n2=n2, P=P_linear, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=1_000)
        
        if sol !== nothing && length(sol) == 2
            lJ_Pd_chiN[i] = sol[1]
            lH_Pd_chiN[i] = sol[2]
        end
    catch e
        @warn "P~d, chi=N failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # ========================================
    # P ~ sqrt(d) configurations
    # ========================================
    
    # P ~ sqrt(d), chi=1.0
    try
        sol = FCSLinear.nlsolve_solver(i0;
            chi=χ1, d=d, kappa=κ, delta=1.0,
            n1=n1, n2=n2, P=P_sqrt, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=10_000)
        
        if sol !== nothing && length(sol) == 2
            lJ_Psqrtd_chi1[i] = sol[1]
            lH_Psqrtd_chi1[i] = sol[2]
        end
    catch e
        @warn "P~sqrt(d), chi=1 failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # P ~ sqrt(d), chi=N
    try
        sol = FCSLinear.nlsolve_solver(i0;
            chi=χN, d=d, kappa=κ, delta=1.0,
            n1=n1, n2=n2, P=P_sqrt, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=10_000)
        
        if sol !== nothing && length(sol) == 2
            lJ_Psqrtd_chiN[i] = sol[1]
            lH_Psqrtd_chiN[i] = sol[2]
        end
    catch e
        @warn "P~sqrt(d), chi=N failed for d=$d" exception=(e, catch_backtrace())
    end
end

println("\nSweep complete. Serializing results...")

# Serialize results
results = Dict(
    "d_values" => d_values,
    # P ~ d configurations
    "lH_Pd_chi1" => lH_Pd_chi1,
    "lJ_Pd_chi1" => lJ_Pd_chi1,
    "lH_Pd_chiN" => lH_Pd_chiN,
    "lJ_Pd_chiN" => lJ_Pd_chiN,
    # P ~ sqrt(d) configurations
    "lH_Psqrtd_chi1" => lH_Psqrtd_chi1,
    "lJ_Psqrtd_chi1" => lJ_Psqrtd_chi1,
    "lH_Psqrtd_chiN" => lH_Psqrtd_chiN,
    "lJ_Psqrtd_chiN" => lJ_Psqrtd_chiN
)

open(results_file, "w") do io
    serialize(io, results)
end
println("Results saved to: $results_file")

#########################################################
# Plotting with fitted power-law slopes
#########################################################

default(titlefont=font(18), guidefont=font(16), tickfont=font(14), legendfontsize=12)

# Fit helper: fit y = A * d^s in log10 space without intercept, returns (s, predicted_y)
# This forces the fit through the origin in log-log space for exact power-law scaling
function fit_powerlaw(d, y; skip_first=5)
    mask = .!isnan.(y) .& isfinite.(y) .& (y .> 0)
    if count(mask) < 3
        return nothing
    end
    
    # Skip first few points for better fits
    mask[1:min(skip_first, length(mask))] .= false
    
    x = log10.(d[mask])
    ylog = log10.(y[mask])
    
    if length(x) < 2
        return nothing
    end
    
    # Fit without intercept: minimize sum((ylog - s*x)^2)
    # This gives s = sum(x * ylog) / sum(x^2)
    s = sum(x .* ylog) / sum(x .^ 2)
    
    # predicted y on full d grid: y = d^s (no multiplicative constant in log space)
    ypred = 10 .^ (s .* log10.(d))
    return (s, ypred)
end

# Helper to add fitted line with slope in legend
function add_fit_line!(plt, d_vals, y_vals, label_base, color; skip_first=5)
    fit = fit_powerlaw(d_vals, y_vals; skip_first=skip_first)
    if fit !== nothing
        (s, ypred) = fit
        slope_str = @sprintf("%.3f", s)
        label = label_base * " (slope: $slope_str)"
        plot!(plt, d_vals, ypred; 
              label=label, 
              linestyle=:dash, 
              color=color, 
              alpha=0.7,
              linewidth=2)
    end
end

#########################################################
# Separate plots for lH and lJ
#########################################################

# Colors for different configurations
color_Pd_chi1 = :blue
color_Pd_chiN = :red
color_Psqrtd_chi1 = :green
color_Psqrtd_chiN = :purple

### Plot 1: lH eigenvalues (all configurations)
println("\nGenerating lH plot...")
plt_lH = plot(xscale=:log10, yscale=:log10, 
              xlabel=L"d", ylabel=L"\lambda_H",
            title="FCN3 Linear Preactivation Kernel Eigs, N = 5d",
            size=(1200, 800),
              legend=:outerright,
              left_margin=10mm)

# P ~ d, chi=1.0
plot!(plt_lH, d_values, lH_Pd_chi1; 
      label=L"P \sim 4d, \chi=1.0", 
      marker=:circle, color=color_Pd_chi1, linewidth=2)
add_fit_line!(plt_lH, d_values, lH_Pd_chi1, 
              L"P \sim 4d, \chi=1.0", color_Pd_chi1)

# P ~ d, chi=N
plot!(plt_lH, d_values, lH_Pd_chiN; 
      label=L"P \sim 4d, \chi=N", 
      marker=:square, color=color_Pd_chiN, linewidth=2)
add_fit_line!(plt_lH, d_values, lH_Pd_chiN, 
              L"P \sim 4d, \chi=N", color_Pd_chiN)

# P ~ sqrt(d), chi=1.0
plot!(plt_lH, d_values, lH_Psqrtd_chi1; 
      label=L"P \sim 5\sqrt{d}, \chi=1.0", 
      marker=:utriangle, color=color_Psqrtd_chi1, linewidth=2)
add_fit_line!(plt_lH, d_values, lH_Psqrtd_chi1, 
              L"P \sim 5\sqrt{d}, \chi=1.0", color_Psqrtd_chi1)

# P ~ sqrt(d), chi=N
plot!(plt_lH, d_values, lH_Psqrtd_chiN; 
      label=L"P \sim 5\sqrt{d}, \chi=N", 
      marker=:hexagon, color=color_Psqrtd_chiN, linewidth=2)
add_fit_line!(plt_lH, d_values, lH_Psqrtd_chiN, 
              L"P \sim 5\sqrt{d}, \chi=N", color_Psqrtd_chiN)

lH_file = joinpath(plot_dir, "lH_scaling_linear.png")
savefig(plt_lH, lH_file)
println("Saved lH plot to: $lH_file")

### Plot 2: lJ eigenvalues (all configurations)
println("Generating lJ plot...")
plt_lJ = plot(xscale=:log10, yscale=:log10, 
              xlabel=L"d", ylabel=L"\lambda_J",
              title="FCN3 Linear Preactivation Kernel Eigs, N = 5d",
              size=(1200, 800),
              legend=:outerright,
              left_margin=10mm)

# P ~ d, chi=1.0
plot!(plt_lJ, d_values, lJ_Pd_chi1; 
      label=L"P \sim4= 4d, \chi=1.0", 
      marker=:circle, color=color_Pd_chi1, linewidth=2)
add_fit_line!(plt_lJ, d_values, lJ_Pd_chi1, 
              L"P \sim 4d, \chi=1.0", color_Pd_chi1)

# P ~ d, chi=N
plot!(plt_lJ, d_values, lJ_Pd_chiN; 
      label=L"P \sim 4d, \chi=N", 
      marker=:square, color=color_Pd_chiN, linewidth=2)
add_fit_line!(plt_lJ, d_values, lJ_Pd_chiN, 
              L"P \sim 4d, \chi=N", color_Pd_chiN)

# P ~ sqrt(d), chi=1.0
plot!(plt_lJ, d_values, lJ_Psqrtd_chi1; 
      label=L"P \sim 5\sqrt{d}, \chi=1.0", 
      marker=:utriangle, color=color_Psqrtd_chi1, linewidth=2)
add_fit_line!(plt_lJ, d_values, lJ_Psqrtd_chi1, 
              L"P \sim 5\sqrt{d}, \chi=1.0", color_Psqrtd_chi1)

# P ~ sqrt(d), chi=N
plot!(plt_lJ, d_values, lJ_Psqrtd_chiN; 
      label=L"P \sim 5\sqrt{d}, \chi=N", 
      marker=:hexagon, color=color_Psqrtd_chiN, linewidth=2)
add_fit_line!(plt_lJ, d_values, lJ_Psqrtd_chiN, 
              L"P \sim 5\sqrt{d}, \chi=N", color_Psqrtd_chiN)

lJ_file = joinpath(plot_dir, "lJ_scaling_linear.png")
savefig(plt_lJ, lJ_file)
println("Saved lJ plot to: $lJ_file")

println("\n" * "="^60)
println("All plots saved to: $plot_dir")
println("="^60)
