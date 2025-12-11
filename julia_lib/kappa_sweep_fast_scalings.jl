# kappa_sweep_fast_scalings.jl
# Fast kappa sweep comparing multiple kappa scaling laws (1/d, 1/d^2, 1/d^3) across regimes

using Printf
using LinearAlgebra
using Plots
gr()

include("./FCSLinear.jl")
using .FCSLinear

function sigfig(x::Float64; n::Int=6)
    if isnan(x) || isinf(x) return string(x) end
    if x == 0 return "0" * (n > 1 ? "." * repeat("0", n - 1) : "") end
    order = floor(log10(abs(x)))
    if order >= n
        return string(round(Int, x))
    else
        int_digits = order >= 0 ? Int(order) + 1 : 1
        dec_digits = max(n - int_digits, 0)
        s = @sprintf("%.*f", dec_digits, x)
        return rstrip(rstrip(s, '0'), '.')
    end
end

function run_sweep(d_vals::Vector, chi::Float64, kappa_scaling::Function, regime_name::String)
    """
    Run kappa sweep for given chi and kappa scaling function.
    
    Args:
        d_vals: vector of d values to sweep
        chi: chi parameter (1.0 for standard, N for mean-field)
        kappa_scaling: function that takes d and returns kappa value
        regime_name: name of regime (e.g., "Standard (χ=1)")
    
    Returns:
        results: vector of tuples with (d, kappa, lJ, lH, lK, loss)
    """
    results = []
    delta = 1.0
    
    for (i, d) in enumerate(d_vals)
        initial_guess = [1.0 / sqrt(d), 1.0 / sqrt(d)]
        P = Int(round(10 * d))
        κ = kappa_scaling(d)
        d_local = Int(round(d))
        n_local = Int(round(5 * d))
        
        sol = FCSLinear.nlsolve_solver(
            initial_guess; P=P, chi=chi, d=d_local, kappa=κ, delta=delta,
            n1=Float64(n_local), n2=Float64(n_local), max_iter=50_000, anneal_steps=5_000,
            lr=1e-4, verbose=false, anneal=true
        )
        
        if sol === nothing
            println("  [$i/$(length(d_vals))] $regime_name, κ=$(sigfig(κ, n=3)): FAILED")
            continue
        end
        
        lJ, lH = sol
        lK_arr = FCSLinear.compute_lK(sol, P, Float64(n_local), Float64(n_local), chi, d_local, delta, κ)
        bias = (κ / (lH + κ))^2
        println("bias: ", bias)
        variance = (κ) / (chi_std * P) * 1 / (lH + κ)
        println("variance: ", variance)
        loss = (bias + variance) * chi_std / κ
        
        push!(results, (d=d, κ=κ, lJ=lJ, lH=lH, lK=lK_arr[1], loss=loss, regime=regime_name))
        println("  [$i/$(length(d_vals))] $regime_name, d=$(sigfig(d, n=3)), κ=$(sigfig(κ, n=3)) ✓")
    end
    
    return results
end

function plot_results(all_results::Vector, plot_type::String)
    """
    Plot results from all runs on same axes.
    
    Args:
        all_results: flattened vector of all result tuples
        plot_type: one of "lJ", "lH", "lK", "loss"
    
    Returns:
        plot object
    """
    p = plot(xscale=:log10, yscale=:log10, xlabel="d", ylabel=plot_type)
    
    # Group results by regime
    regimes = unique([r.regime for r in all_results])
    colors = [:blue, :darkblue, :red, :darkred, :green, :darkgreen, :orange, :purple]
    linestyles = [:solid, :dash]
    
    for (color_idx, regime) in enumerate(regimes)
        regime_results = [r for r in all_results if r.regime == regime]
        ds = [r.d for r in regime_results]
        
        if plot_type == "lJ"
            vals = [r.lJ for r in regime_results]
        elseif plot_type == "lH"
            vals = [r.lH for r in regime_results]
        elseif plot_type == "lK"
            vals = [r.lK for r in regime_results]
        else  # loss
            vals = [r.loss for r in regime_results]
        end
        
        color = colors[color_idx]
        plot!(p, ds, vals, linewidth=2.5, color=color, label=regime,
              linestyle=linestyles[mod1(color_idx, length(linestyles))])
    end
    
    return p
end

# ============================================
# Main Script
# ============================================
println("="^70)
println("KAPPA SWEEP: Multiple Scaling Laws vs d")
println("="^70)

N = 200
d_vals = 10.0 .^ range(0.5, 3, length=30)

# Define kappa scaling functions
kappa_1_over_d(d::Float64) = 0.1 / d
kappa_1_over_d2(d::Float64) = 0.1 / d^2
kappa_1_over_d3(d::Float64) = 0.1 / d^3
kappa_d_over_1000(d::Float64) = d / 1000

# Standard scaling (chi=1)
chi_std = 1.0

# Mean-field scaling (chi=N)
chi_mf = Float64(N)

println("\nRunning sweeps for all scaling laws and regimes...")

# 1/d scaling
println("\n[1/6] Standard scaling, κ ∝ 1/d...")
results_std_1d = run_sweep(d_vals, chi_std, kappa_1_over_d, "Standard (χ=1), κ∝1/d")

println("\n[2/6] Mean-field scaling, κ ∝ 1/d...")
results_mf_1d = run_sweep(d_vals, chi_mf, kappa_1_over_d, "Mean-Field (χ=N), κ∝1/d")

# 1/d^2 scaling
println("\n[3/6] Standard scaling, κ ∝ 1/d²...")
results_std_1d2 = run_sweep(d_vals, chi_std, kappa_1_over_d2, "Standard (χ=1), κ∝1/d²")

println("\n[4/6] Mean-field scaling, κ ∝ 1/d²...")
results_mf_1d2 = run_sweep(d_vals, chi_mf, kappa_1_over_d2, "Mean-Field (χ=N), κ∝1/d²")

# 1/d^3 scaling
println("\n[5/6] Standard scaling, κ ∝ 1/d³...")
results_std_1d3 = run_sweep(d_vals, chi_std, kappa_1_over_d3, "Standard (χ=1), κ∝1/d³")

println("\n[6/6] Mean-field scaling, κ ∝ 1/d³...")
results_mf_1d3 = run_sweep(d_vals, chi_mf, kappa_1_over_d3, "Mean-Field (χ=N), κ∝1/d³")

# d/1000 scaling
println("\n[7/8] Standard scaling, κ ∝ d/1000...")
results_std_d = run_sweep(d_vals, chi_std, kappa_d_over_1000, "Standard (χ=1), κ∝d/1000")

println("\n[8/8] Mean-field scaling, κ ∝ d/1000...")
results_mf_d = run_sweep(d_vals, chi_mf, kappa_d_over_1000, "Mean-Field (χ=N), κ∝d/1000")

# Combine all results
all_results = vcat(
    results_std_1d, results_mf_1d,
    results_std_1d2, results_mf_1d2,
    results_std_1d3, results_mf_1d3,
    results_std_d, results_mf_d
)

# ============================================
# Plotting
# ============================================
println("\nGenerating plots...")

p1 = plot_results(all_results, "lJ")
plot!(p1, title="J Eigenvalue (Target)", legend=false)

p2 = plot_results(all_results, "lH")
plot!(p2, title="H Eigenvalue (Target)", legend=false)

p3 = plot_results(all_results, "lK")
plot!(p3, title="Readout Kernel Eigenvalue (Target)", legend=false)

p4 = plot_results(all_results, "loss")
plot!(p4, title="Loss Scaling (Target)", legend=:bottomleft, 
      xscale=:log10, yscale=:log10)

# Build a top row (1x3) and place loss full-width on bottom
top = plot(p1, p2, p3, layout=(1, 3), legend=false)
fig_layout = @layout [a{0.65h}; b{0.35h}]
fig = plot(top, p4, layout=fig_layout, size=(1600, 2000), margin=5Plots.mm)
savefig(fig, "/home/akiva/FCNX-Ensembling/plots/kappa_sweep_all_scalings.png")
println("\nPlot saved: /home/akiva/FCNX-Ensembling/plots/kappa_sweep_all_scalings.png")
println("="^70)
