using ForwardDiff
using LinearAlgebra
using Plots
using Colors
using Serialization
using Statistics
using LaTeXStrings
using Measures
using Printf

include("./FCS2Linear.jl")
using .FCS2Linear

#########################################################
# d-sweep eigensolver for linear FCN2 networks (1 hidden layer)
#
# This script sweeps over input dimension `d` for different P scalings
# and computes the mean-field solutions for eigenvalue equations
# using the solvers provided by `FCS2Linear.jl`.
#
# For each d the script records:
# - lH (Hessian eigenvalue)
# - lJ (NTK eigenvalue)
#
# The script produces separate log-log plots for lH and lJ with
# fitted power-law slopes shown in the legend.
#########################################################

## User-editable sweep parameters
κ = 1.0                      # kappa (noise)
n_factor = 5.0               # n1 = n_factor * d
δ = 0.0                      # delta for the eigenvalue equations (no distinction needed)
lr = 1e-5
Tf = 50_000                 # max iterations for solver

# Sweep range for d (input dimension): 2 to 10000
d_values = [2,6,8,10, 100, 1000, 10000, 30000]

# Output folder for plots
plot_dir = joinpath(@__DIR__, "..", "plots", "d_sweep_fcs2linear")
mkpath(plot_dir)
results_file = joinpath(plot_dir, "d_sweep_fcs2linear_results.jls")

# Storage arrays
ND = length(d_values)

# P ~ d configurations

lJ_Pd_chi1 = fill(NaN, ND)


lJ_Pd_chiN = fill(NaN, ND)

# P ~ sqrt(d) configurations

lJ_Psqrtd_chi1 = fill(NaN, ND)


lJ_Psqrtd_chiN = fill(NaN, ND)

println("Starting d-sweep for linear FCN2 networks...")
println("d range: $(d_values[1]) to $(d_values[end])")
println("P scalings: P ~ d and P ~ sqrt(d)")
println("chi values: 1.0 and N")

println()

for (i, d) in enumerate(d_values)
    P_linear = 3 * Float64(d)           # P ~ d
    P_sqrt = 5 * sqrt(Float64(d))       # P ~ sqrt(d)
    
    n1 = n_factor * d
    
    χ1 = 1.0        # chi = 1.0
    χN = Float64(n1)  # chi = N
    
    if i % 10 == 0 || i == 1
        println("Progress: d = $d ($(i)/$(ND))")
    end
    
    # Initial guess heuristics (scale with d)
    i0 = [1.0 / sqrt(d)]  # [lJ, lH]
    
    # ========================================
    # P ~ d configurations
    # ========================================
    
    # P ~ d, chi=1.0
    try
        sol = FCS2Linear.nlsolve_solver(i0;
            chi=χ1, d=d, kappa=κ, delta=0.0,tol=1e-8,
            n1=n1, P=P_linear, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=30_000)
        
        if sol !== nothing && length(sol) == 1
            lJ_Pd_chi1[i] = sol[1]

        end
    catch e
        @warn "P~d, chi=1 failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # P ~ d, chi=N
    try
        sol = FCS2Linear.nlsolve_solver(i0;
            chi=χN, d=d, kappa=κ, delta=1.0,tol=1e-8,
            n1=n1, P=P_linear, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=30_000)
        
        if sol !== nothing && length(sol) == 1
            lJ_Pd_chiN[i] = sol[1]

        end
    catch e
        @warn "P~d, chi=N failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # ========================================
    # P ~ sqrt(d) configurations
    # ========================================
    
    # P ~ sqrt(d), chi=1.0
    try
        sol = FCS2Linear.nlsolve_solver(i0;
            chi=χ1, d=d, kappa=κ, delta=0.0,
            n1=n1, P=P_sqrt, lr=lr, max_iter=Tf, tol=1e-8,
            verbose=false, anneal=true, anneal_steps=30_000)
        
        if sol !== nothing && length(sol) == 1
            lJ_Psqrtd_chi1[i] = sol[1]

        end
    catch e
        @warn "P~sqrt(d), chi=1 failed for d=$d" exception=(e, catch_backtrace())
    end
    
    # P ~ sqrt(d), chi=N
    try
        sol = FCS2Linear.nlsolve_solver(i0;
            chi=χN, d=d, kappa=κ, delta=1.0,tol=1e-8,
            n1=n1, P=P_sqrt, lr=lr, max_iter=Tf, 
            verbose=false, anneal=true, anneal_steps=30_000)
        
        if sol !== nothing && length(sol) == 1
            lJ_Psqrtd_chiN[i] = sol[1]

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

    "lJ_Pd_chi1" => lJ_Pd_chi1,

    "lJ_Pd_chiN" => lJ_Pd_chiN,

)

open(results_file, "w") do io
    serialize(io, results)
end
println("Results saved to: $results_file")
println(results)
# ...existing code...

# --- Plotting and Slope Fitting ---
function plot_and_fit(pl, x, y, labelstr; color=:blue, marker=:circle, ylabel="Eigenvalue", title="")
    # Only fit where y is finite and positive
    mask = isfinite.(y) .& (y .> 0)
    xfit = x[mask]
    yfit = y[mask]
    logx = log10.(xfit)
    logy = log10.(yfit)
    # Linear fit in log-log
   # Given logx and logy as vectors
    slope = sum((logx .- mean(logx)) .* (logy .- mean(logy))) / sum((logx .- mean(logx)).^2)
    intercept = mean(logy) - slope * mean(logx)
    yfit_line = 10.0 .^ (slope .* logx .+ intercept)
    println(yfit_line)
    scatter!(pl,xfit, yfit; m=marker, label=labelstr * " (slope=$(round(slope, digits=3)))", color=color)
    plot!(pl,xfit, yfit_line; lw=2, ls=:dash, color=color, label="")
    return slope
end

# Plot lH and lJ for P~d
plt1 = plot(xlabel="d", ylabel="Eigenvalue", xscale=:log10, yscale=:log10, legend=:topleft, title="FCN2 Linear: P~d")
slope_lJ_chi1 = plot_and_fit(plt1,d_values, lJ_Pd_chi1, "lJ, chi=1", color=:red)

slope_lJ_chiN = plot_and_fit(plt1,d_values, lJ_Pd_chiN, "lJ, chi=N", color=:orange)
display(plt1)
savefig(plt1, joinpath(plot_dir, "fcs2linear_Pd_eigs.png"))


println("Plots saved to $plot_dir")