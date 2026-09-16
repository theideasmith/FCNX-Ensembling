# 1. Include source files
include("FCS2_VGA.jl")
include("FCS2Erf_Cubic.jl")

# 2. Load the modules
using .FCS2_VGA
using .FCS2Erf_Cubic
using Plots

gr()

function run_simulation_and_plot()
    # ---------------------------------------------------------
    # Simulation Setup
    # ---------------------------------------------------------
    d_val = 150.0f0
    N_val = 500              # chi = N = 500
    kappa_val = 1.0f0        # Regularization scale
    n1_val = 500.0f0
    delta_val = 1.0f0
    epsilon_val = 0.03f0

    P_samples = 30
    P_vals = exp.(range(log(d_val), log(d_val^3), length=P_samples))
    alpha_vals = log.(P_vals) ./ log(d_val)

    # Initialize results arrays explicitly inside local function scope
    vga_N_learnability_linear = Float64[]
    vga_N_learnability_cubic  = Float64[]
    vga_N_lJ1                 = Float64[]
    vga_N_lJ3                 = Float64[]

    vga_1_learnability_linear = Float64[]
    vga_1_learnability_cubic  = Float64[]
    vga_1_lJ1                 = Float64[]
    vga_1_lJ3                 = Float64[]

    cubic_N_learnability_linear = Float64[]
    cubic_N_learnability_cubic  = Float64[]
    cubic_N_lJ1                 = Float64[]
    cubic_N_lJ3                 = Float64[]

    cubic_1_learnability_linear = Float64[]
    cubic_1_learnability_cubic  = Float64[]
    cubic_1_lJ1                 = Float64[]
    cubic_1_lJ3                 = Float64[]

    guess_vga_N = [0.2, 0.01, 1.0, 0.05]
    guess_vga_1 = [0.2, 0.01, 1.0, 0.05]

    println("Running VGA & Cubic solvers for χ = N and χ = 1 across α ∈ [1, 3]...")

    for (i, P_curr) in enumerate(P_vals)
        # 1. VGA Solver (chi = N)
        params_vga_N = FCS2_VGA.ProblemParams2(
            d=d_val, κ=kappa_val, ϵ=epsilon_val, P=Float32(P_curr), n1=n1_val, χ=Float32(N_val), δ=delta_val
        )
        sol_vga_N = FCS2_VGA.solve_FCN2_Erf(params_vga_N, guess_vga_N; anneal_steps=3000, use_anneal=true)
        push!(vga_N_learnability_linear, sol_vga_N.learnability1)
        push!(vga_N_learnability_cubic, sol_vga_N.learnability3)
        push!(vga_N_lJ1, sol_vga_N.lJ1)
        push!(vga_N_lJ3, sol_vga_N.lJ3)
        if !isnan(sol_vga_N.lJ1) && !isnan(sol_vga_N.sigS)
            guess_vga_N = [sol_vga_N.lJ1, sol_vga_N.lJ3, sol_vga_N.sigS, sol_vga_N.muW]
        end

        # 2. VGA Solver (chi = 1)
        params_vga_1 = FCS2_VGA.ProblemParams2(
            d=d_val, κ=kappa_val, ϵ=epsilon_val, P=Float32(P_curr), n1=n1_val, χ=1.0f0, δ=delta_val
        )
        sol_vga_1 = FCS2_VGA.solve_FCN2_Erf(params_vga_1, guess_vga_1; anneal_steps=3000, use_anneal=true)
        push!(vga_1_learnability_linear, sol_vga_1.learnability1)
        push!(vga_1_learnability_cubic, sol_vga_1.learnability3)
        push!(vga_1_lJ1, sol_vga_1.lJ1)
        push!(vga_1_lJ3, sol_vga_1.lJ3)
        if !isnan(sol_vga_1.lJ1) && !isnan(sol_vga_1.sigS)
            guess_vga_1 = [sol_vga_1.lJ1, sol_vga_1.lJ3, sol_vga_1.sigS, sol_vga_1.muW]
        end

        # 3. Cubic Solver (chi = N)
        params_cubic_N = FCS2Erf_Cubic.ProblemParams2(
            d=Float32(d_val), κ=Float32(kappa_val), ϵ=Float32(epsilon_val), P=Float32(P_curr), n1=Float32(N_val), χ=Float32(N_val), δ=1.0
        )
        sol_cubic_N = FCS2Erf_Cubic.solve_FCN2_Erf(
            params_cubic_N, [1.0 / d_val, 1.0 / d_val^3, 1.0 / d_val]; lr=1e-6, max_iter=1_000_000, anneal_steps=30_000, use_anneal=true
        )
        push!(cubic_N_learnability_linear, sol_cubic_N.learnability1)
        push!(cubic_N_learnability_cubic, sol_cubic_N.learnability3)
        push!(cubic_N_lJ1, sol_cubic_N.lJ1)
        push!(cubic_N_lJ3, sol_cubic_N.lJ3)

        # 4. Cubic Solver (chi = 1)
        params_cubic_1 = FCS2Erf_Cubic.ProblemParams2(
            d=Float32(d_val), κ=Float32(kappa_val), ϵ=Float32(epsilon_val), P=Float32(P_curr), n1=Float32(N_val), χ=1.0f0, δ=1.0
        )
        sol_cubic_1 = FCS2Erf_Cubic.solve_FCN2_Erf(
            params_cubic_1, [1.0 / d_val, 1.0 / d_val^3, 1.0 / d_val]; lr=1e-6, max_iter=1_000_000, anneal_steps=30_000, use_anneal=true
        )
        push!(cubic_1_learnability_linear, sol_cubic_1.learnability1)
        push!(cubic_1_learnability_cubic, sol_cubic_1.learnability3)
        push!(cubic_1_lJ1, sol_cubic_1.lJ1)
        push!(cubic_1_lJ3, sol_cubic_1.lJ3)

        println("Step $i/$P_samples (α = $(round(alpha_vals[i], digits=3))): Done.")
    end

    # ---------------------------------------------------------
    # Plotting Layout (Modular & Clean)
    # ---------------------------------------------------------
    default(
        fontfamily="sans-serif", titlefontsize=11, guidefontsize=10, 
        tickfontsize=9, linewidth=2.0, markersize=3.5, grid=true, gridalpha=0.2
    )

    COLOR_LINEAR = :royalblue
    COLOR_CUBIC  = :crimson
    STYLE_VGA     = (style = :solid, marker = :circle)
    STYLE_REGULAR = (style = :dash,  marker = :square)

    # Subplot 1A: Learnability (χ = N)
    p1_chiN = plot(
        alpha_vals,
        [vga_N_learnability_linear vga_N_learnability_cubic cubic_N_learnability_linear cubic_N_learnability_cubic],
        label = ["VGA Linear" "VGA Cubic" "Regular Linear" "Regular Cubic"],
        color = [COLOR_LINEAR COLOR_CUBIC COLOR_LINEAR COLOR_CUBIC],
        linestyle = [STYLE_VGA.style STYLE_VGA.style STYLE_REGULAR.style STYLE_REGULAR.style],
        marker = [STYLE_VGA.marker STYLE_VGA.marker STYLE_REGULAR.marker STYLE_REGULAR.marker],
        xlabel = "α (P = d^α)", ylabel = "Learnability", title = "Learnability (χ = N)", legend = :topleft
    )

    # Subplot 1B: Learnability (χ = 1)
    p1_chi1 = plot(
        alpha_vals,
        [vga_1_learnability_linear vga_1_learnability_cubic cubic_1_learnability_linear cubic_1_learnability_cubic],
        label = ["VGA Linear" "VGA Cubic" "Regular Linear" "Regular Cubic"],
        color = [COLOR_LINEAR COLOR_CUBIC COLOR_LINEAR COLOR_CUBIC],
        linestyle = [STYLE_VGA.style STYLE_VGA.style STYLE_REGULAR.style STYLE_REGULAR.style],
        marker = [STYLE_VGA.marker STYLE_VGA.marker STYLE_REGULAR.marker STYLE_REGULAR.marker],
        xlabel = "α (P = d^α)", ylabel = "Learnability", title = "Learnability (χ = 1)", legend = :topleft
    )

    # Subplot 2A: Eigenvalues lJ1
    p2_lJ1 = plot(
        alpha_vals,
        [vga_N_lJ1 vga_1_lJ1 cubic_N_lJ1 cubic_1_lJ1],
        label = ["VGA (χ=N)" "VGA (χ=1)" "Regular (χ=N)" "Regular (χ=1)"],
        color = [COLOR_LINEAR :teal COLOR_LINEAR :teal],
        linestyle = [STYLE_VGA.style STYLE_VGA.style STYLE_REGULAR.style STYLE_REGULAR.style],
        marker = [STYLE_VGA.marker STYLE_VGA.marker STYLE_REGULAR.marker STYLE_REGULAR.marker],
        xlabel = "α (P = d^α)", ylabel = "lJ1", title = "Linear Eigenvalue (lJ1)", legend = :topleft
    )

    # Subplot 2B: Eigenvalues lJ3
    p2_lJ3 = plot(
        alpha_vals,
        [vga_N_lJ3 vga_1_lJ3 cubic_N_lJ3 cubic_1_lJ3],
        label = ["VGA (χ=N)" "VGA (χ=1)" "Regular (χ=N)" "Regular (χ=1)"],
        color = [COLOR_CUBIC :darkorange COLOR_CUBIC :darkorange],
        linestyle = [STYLE_VGA.style STYLE_VGA.style STYLE_REGULAR.style STYLE_REGULAR.style],
        marker = [STYLE_VGA.marker STYLE_VGA.marker STYLE_REGULAR.marker STYLE_REGULAR.marker],
        xlabel = "α (P = d^α)", ylabel = "lJ3", title = "Cubic Eigenvalue (lJ3)", legend = :topleft
    )

    p_combined = plot(p1_chiN, p1_chi1, p2_lJ1, p2_lJ3, layout=@layout([a b; c d]), size=(1000, 750), margin=5Plots.mm)
    
    savefig(p_combined, "combined_chi_comparison.png")
    println("Saved output plot to combined_chi_comparison.png")
    display(p_combined)
end

# Run execution
run_simulation_and_plot()