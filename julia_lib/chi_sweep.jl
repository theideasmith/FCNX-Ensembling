#!/usr/bin/env julia
"""
Chi sweep analysis for FCS Linear networks:
- Sweep chi from 1 to 100 (30 values)
- For each chi, compute lH and lJ for d=50 and d=100
- Compute scaling exponents: α = (log lH(d=50) - log lH(d=100)) / (log 50 - log 100)
- Plot lH, lJ, and their scaling exponents vs chi
"""

using Pkg
Pkg.add("Plots")
Pkg.add("StatsPlots")
Pkg.add("NLsolve")
Pkg.add("ForwardDiff")

using Plots, StatsPlots
using NLsolve, ForwardDiff

# Include FCSLinear module
include("FCSLinear.jl")
using .FCSLinear

# Parameters
d1 = 50.0
d2 = 100.0
N = 500.0
P_base = 60
kappa = 1.0 / d1  # kappa = 1/d for d1

# Sweep parameters
chi_values = exp.(range(log(1.0), log(1000.0), length=30))
n_chi = length(chi_values)

# Storage for results
lH_d50 = Float64[]
lJ_d50 = Float64[]
lH_d100 = Float64[]
lJ_d100 = Float64[]
alpha_lH = Float64[]
alpha_lJ = Float64[]

println("Starting chi sweep...")
println("d1=$d1, d2=$d2, N=$N, κ=$kappa")
println()

for (i, chi) in enumerate(chi_values)
    print("Chi $i/$n_chi: chi=$(round(chi, digits=3)) ... ")
    
    # For d=50
    P1 = P_base  # P = 6
    n1 = N
    n2 = N
    
    # Initial guess for training regime (delta=1)
    i0 = [1.0 / sqrt(d1), 1.0 / sqrt(d1)]
    
    try
        sol_d50_T = FCSLinear.nlsolve_solver(
            i0,
            chi=chi, d=d1, kappa=kappa, delta=1.0,
            n1=n1, n2=n2, P=P1, lr=1e-3, max_iter=5000, verbose=false, anneal=true, anneal_steps=3000
        )
        
        if sol_d50_T !== nothing && all(isfinite.(sol_d50_T)) && all(sol_d50_T .> 0)
            lJ_50, lH_50 = sol_d50_T
            push!(lH_d50, lH_50)
            push!(lJ_d50, lJ_50)
        else
            push!(lH_d50, NaN)
            push!(lJ_d50, NaN)
        end
    catch e
        println("Error for d=50: $e")
        push!(lH_d50, NaN)
        push!(lJ_d50, NaN)
    end
    
    # For d=100
    kappa2 = 1.0 / d2  # kappa = 1/d for d2
    i0_d100 = [1.0 / sqrt(d2), 1.0 / sqrt(d2)]
    
    try
        sol_d100_T = FCSLinear.nlsolve_solver(
            i0_d100,
            chi=chi, d=d2, kappa=kappa2, delta=1.0,
            n1=n1, n2=n2, P=P1, lr=1e-3, max_iter=5000, verbose=false, anneal=true, anneal_steps=3000
        )
        
        if sol_d100_T !== nothing && all(isfinite.(sol_d100_T)) && all(sol_d100_T .> 0)
            lJ_100, lH_100 = sol_d100_T
            push!(lH_d100, lH_100)
            push!(lJ_d100, lJ_100)
        else
            push!(lH_d100, NaN)
            push!(lJ_d100, NaN)
        end
    catch e
        println("Error for d=100: $e")
        push!(lH_d100, NaN)
        push!(lJ_d100, NaN)
    end
    
    # Compute scaling exponents if both solutions are valid
    if !isnan(lH_d50[i]) && !isnan(lH_d100[i]) && lH_d50[i] > 0 && lH_d100[i] > 0
        alpha_lH_val = (log(lH_d50[i]) - log(lH_d100[i])) / (log(d1) - log(d2))
        push!(alpha_lH, alpha_lH_val)
    else
        push!(alpha_lH, NaN)
    end
    
    if !isnan(lJ_d50[i]) && !isnan(lJ_d100[i]) && lJ_d50[i] > 0 && lJ_d100[i] > 0
        alpha_lJ_val = (log(lJ_d50[i]) - log(lJ_d100[i])) / (log(d1) - log(d2))
        push!(alpha_lJ, alpha_lJ_val)
    else
        push!(alpha_lJ, NaN)
    end
    
    println("lH: $(round(lH_d50[i], digits=4)) (d=50), $(round(lH_d100[i], digits=4)) (d=100); " *
            "lJ: $(round(lJ_d50[i], digits=4)) (d=50), $(round(lJ_d100[i], digits=4)) (d=100)")
end

println("\n" * "="^70)
println("Chi sweep completed. Creating plots...")
println("="^70)

# Convert chi_values to regular array for plotting
chi_plot = collect(chi_values)

# Plot 1: lH and lJ vs chi (d=50)
p1 = plot(chi_plot, lH_d50, label="lH (d=50)", xlabel="χ (chi)", ylabel="Eigenvalue", 
           title="lH and lJ vs chi (d=50)", legend=:best, linewidth=2, marker=:circle, 
           xscale=:log, yscale=:log)
plot!(p1, chi_plot, lJ_d50, label="lJ (d=50)", linewidth=2, marker=:square)

# Plot 2: lH vs chi for both d values
p2 = plot(chi_plot, lH_d50, label="lH (d=50)", xlabel="χ (chi)", ylabel="lH", 
           title="lH vs chi (comparison)", legend=:best, linewidth=2, marker=:circle, 
           xscale=:log, yscale=:log)
plot!(p2, chi_plot, lH_d100, label="lH (d=100)", linewidth=2, marker=:square)

# Plot 3: lJ vs chi for both d values
p3 = plot(chi_plot, lJ_d50, label="lJ (d=50)", xlabel="χ (chi)", ylabel="lJ", 
           title="lJ vs chi (comparison)", legend=:best, linewidth=2, marker=:circle, 
           xscale=:log, yscale=:log)
plot!(p3, chi_plot, lJ_d100, label="lJ (d=100)", linewidth=2, marker=:square)

# Plot 4: Scaling exponents vs chi
p4 = plot(chi_plot, alpha_lH, label="α_lH = d(log lH)/d(log d)", xlabel="χ (chi)", 
           ylabel="Scaling Exponent", title="Scaling Exponents vs chi", legend=:best, 
           linewidth=2, marker=:circle, xscale=:log)
plot!(p4, chi_plot, alpha_lJ, label="α_lJ = d(log lJ)/d(log d)", linewidth=2, marker=:square)

# Combined plot
p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
savefig(p_combined, "chi_sweep_analysis.png")
println("Saved combined plot to chi_sweep_analysis.png")

# Additional individual plots with better formatting
# Plot lH and lJ separately
p_lh = plot(chi_plot, lH_d50, label="lH (d=50)", xlabel="χ (chi)", ylabel="lH", 
            title="Training Eigenvalue lH vs chi", legend=:best, linewidth=2.5, 
            marker=:circle, markersize=4, xscale=:log, yscale=:log, size=(800, 600))
plot!(p_lh, chi_plot, lH_d100, label="lH (d=100)", linewidth=2.5, marker=:square, markersize=4)
savefig(p_lh, "lH_vs_chi.png")
println("Saved lH plot to lH_vs_chi.png")

p_lj = plot(chi_plot, lJ_d50, label="lJ (d=50)", xlabel="χ (chi)", ylabel="lJ", 
            title="Hidden Eigenvalue lJ vs chi", legend=:best, linewidth=2.5, 
            marker=:circle, markersize=4, xscale=:log, yscale=:log, size=(800, 600))
plot!(p_lj, chi_plot, lJ_d100, label="lJ (d=100)", linewidth=2.5, marker=:square, markersize=4)
savefig(p_lj, "lJ_vs_chi.png")
println("Saved lJ plot to lJ_vs_chi.png")

p_alpha = plot(chi_plot, alpha_lH, label="α_lH (Training)", xlabel="χ (chi)", 
               ylabel="Scaling Exponent α", title="Scaling Exponent vs chi", 
               legend=:best, linewidth=2.5, marker=:circle, markersize=5, 
               xscale=:log, size=(800, 600))
plot!(p_alpha, chi_plot, alpha_lJ, label="α_lJ (Hidden)", linewidth=2.5, marker=:square, markersize=5)
hline!(p_alpha, [0.0], label="α=0", linestyle=:dash, linewidth=1.5, color=:gray)
savefig(p_alpha, "scaling_exponents_vs_chi.png")
println("Saved scaling exponent plot to scaling_exponents_vs_chi.png")

# Save data to file
using DelimitedFiles
data = hcat(chi_plot, lH_d50, lJ_d50, lH_d100, lJ_d100, alpha_lH, alpha_lJ)
writedlm("chi_sweep_data.csv", 
         ["chi" "lH_d50" "lJ_d50" "lH_d100" "lJ_d100" "alpha_lH" "alpha_lJ";
          data], ',')
println("Saved data to chi_sweep_data.csv")

println("\nDone! All plots saved.")
