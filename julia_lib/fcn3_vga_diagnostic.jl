# ============================================================================
# diagnostic_plot_fcn3.jl — Fault-Tolerant P-Sweep Diagnostic Script
# ============================================================================

include("FCS3_VGA.jl")
using .FCS3_VGA
using Plots
using Printf
using Logging

function run_diagnostic_sweep()
    println("="^75)
    println(" Running FCN3 Diagnostic Sweep: Sweeping P from d^1 to d^3")
    println("="^75)

    # Fixed System Parameters
    d_val = 150.0f0
    chi_val = 1600
    n1_val = 1600.0f0
    n2_val = 1600.0f0
    kappa_val = 2.0f0
    epsilon_val = 0.074f0
    delta_val = 1.0f0

    # 1. Sample 100 points for alpha in [1.0, 1.5]
    num_points = 100
    alphas = collect(range(1.0, 1.5, length=num_points))
    P_vals = d_val .^ alphas

    # Storage arrays
    lJ1_vals  = zeros(num_points)
    lJ3_vals  = zeros(num_points)
    lH1_vals  = zeros(num_points)
    lH3_vals  = zeros(num_points)
    lK1_vals  = zeros(num_points)
    lK3_vals  = zeros(num_points)
    lWT_vals  = zeros(num_points)

    muW_vals  = zeros(num_points)
    muH1_vals = zeros(num_points)
    muH3_vals = zeros(num_points)

    learn1_vals = zeros(num_points)
    learn3_vals = zeros(num_points)

    # Initial guess vector: [lJ1, lJ3, lH1, lH3, lWT, mu_J1, mu_J3, mu_H1, mu_H3, mu_WT]
    curr_guess = [0.5, 0.1, 0.5, 0.1, 0.1, 0.1, 0.01, 0.1, 0.01, 0.1]

    println("Beginning upward P-continuation sweep...")

    valid_count = 0
    time_start = time()

    for i in 1:num_points
        alpha = alphas[i]
        P_curr = P_vals[i]

        params = ProblemParams3(
            d = d_val,
            P = Float32(P_curr),
            χ = chi_val,
            n1 = n1_val,
            n2 = n2_val,
            κ = kappa_val,
            ϵ = epsilon_val,
            δ = delta_val
        )

        # Robust evaluation wrapped in try...catch block
        sol = try
            solve_FCN3_Erf(
                params,
                curr_guess;
                anneal_chi = true,
                anneal_steps = 10000,
                tol = 1e-8
            )
        catch e
            @warn @sprintf("Solver failed at step %d/%d (α = %.3f, P = %.1f). Stopping sweep and plotting collected data.",
                           i, num_points, alpha, P_curr)
            println("  Details: ", e)
            break
        end

        # Verify output is finite
        if isnan(sol.lJ1) || isnan(sol.lWT) || isnan(sol.learnability1)
            @warn @sprintf("Solver produced NaN output at step %d/%d (α = %.3f, P = %.1f). Stopping sweep and plotting collected data.",
                           i, num_points, alpha, P_curr)
            break
        end

        # Store successful results
        valid_count = i
        lJ1_vals[i] = sol.lJ1
        lJ3_vals[i] = sol.lJ3
        lH1_vals[i] = sol.lH1
        lH3_vals[i] = sol.lH3
        lK1_vals[i] = sol.lK1
        lK3_vals[i] = sol.lK3
        lWT_vals[i] = sol.lWT

        muW_vals[i]  = sol.muW
        muH1_vals[i] = sol.muH1
        muH3_vals[i] = sol.muH3

        learn1_vals[i] = sol.learnability1
        learn3_vals[i] = sol.learnability3

        if i % 20 == 0 || i == 1 || i == num_points
            @printf("  Step %3d/%d | α = %.3f | P = %9.1f | lWT = %.6f | Learn1 = %.4f\n",
                    i, num_points, alpha, P_curr, sol.lWT, sol.learnability1)
        end
    end

    total_time = time() - time_start
    @printf("\nSweep processed %d/%d steps in %.2f seconds.\n", valid_count, num_points, total_time)

    if valid_count == 0
        error("Solver failed on step 1. No valid data points available to plot.")
    end

    # 2. Slice vectors up to valid_count and generate diagnostic plot
    v = 1:valid_count
    v_alphas = alphas[v]

    println("Generating diagnostic plots for ", valid_count, " valid data points...")

    p1 = plot(
        v_alphas, [lJ1_vals[v] lJ3_vals[v] lH1_vals[v] lH3_vals[v] lK1_vals[v] lK3_vals[v] lWT_vals[v]],
        label = ["λ₁ʲ" "λ₃ʲ" "λ₁ᴴ" "λ₃ᴴ" "λ₁ᴷ" "λ₃ᴷ" "λw*"],
        lw = 2,
        yscale = :log10,
        xlabel = "α = log P / log d",
        ylabel = "Eigenvalue Magnitude (log scale)",
        title = "Kernel Eigenvalues & Weight Norm",
        legend = :outerright,
        grid = :true
    )

    p2 = plot(
        v_alphas, [muW_vals[v] muH1_vals[v] muH3_vals[v]],
        label = ["μw" "μH1" "μH3"],
        lw = 2,
        xlabel = "α = log P / log d",
        ylabel = "GMM Mode Means",
        title = "GMM Mode Centers vs Data Scale P",
        legend = :outerright,
        grid = :true
    )

    p3 = plot(
        v_alphas, [learn1_vals[v] learn3_vals[v]],
        label = ["Linear Mode (L₁)" "Cubic Mode (L₃)"],
        lw = 2,
        xlabel = "α = log P / log d",
        ylabel = "Learnability Metric",
        title = "Target Mode Learnability",
        legend = :outerright,
        grid = :true
    )

    fig = plot(p1, p2, p3, layout = (3, 1), size = (850, 1000), margin = 5Plots.mm)

    output_filename = "fcn3_diagnostic_sweep.png"
    savefig(fig, output_filename)
    println("Diagnostic plot saved to '$output_filename'.")

    return fig
end

# Run the sweep and plot
fig = run_diagnostic_sweep()