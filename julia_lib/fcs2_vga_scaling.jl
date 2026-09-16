# 1. Include both source files
include("FCS2_VGA.jl")
include("FCS2Erf_Cubic.jl")

# 2. Load the modules using local module syntax
using .FCS2_VGA
using .FCS2Erf_Cubic

# 3. Load plotting libraries
using Plots
using LaTeXStrings

# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------
# All plots from this script land here, so nothing is scattered across the
# working directory. Created once, reused for every savefig call below.
const OUTDIR = "FCS2 Theory Predictions"
isdir(OUTDIR) || mkpath(OUTDIR)
outpath(name) = joinpath(OUTDIR, name)

# ---------------------------------------------------------
# Shared plot styling
# ---------------------------------------------------------
# Four series, four colors, solid lines, no markers. Nothing else varies.
const COLOR_VGA_N = :navy
const COLOR_VGA_1 = :steelblue
const COLOR_REG_N = :firebrick
const COLOR_REG_1 = :darkorange
const SERIES_LABELS = ["VGA (χ=N)" "VGA (χ=1)" "Regular (χ=N)" "Regular (χ=1)"]
const SERIES_COLORS = [COLOR_VGA_N COLOR_VGA_1 COLOR_REG_N COLOR_REG_1]

# ---------------------------------------------------------
# Simulation Setup
# ---------------------------------------------------------
d_val = 500.0f0
N_val = 2000              # chi = N = 2000
kappa_val = 1.0f0        # Regularization scale
n1_val = 2000.0f0
delta_val = 1.0f0
epsilon_val = 0.03f0

# 30 samples of P logarithmically spaced from P = d to P = d^3
P_samples = 30
P_vals = exp.(range(log(d_val), log(d_val^3), length=P_samples))

# Compute alpha values: alpha = log_d(P) = log(P) / log(d), running from 1.0 to 3.0
alpha_vals = log.(P_vals) ./ log(d_val)

# Results arrays for VGA
vga_N_learnability_linear = Float64[];
vga_N_learnability_cubic = Float64[]
vga_N_lJ1 = Float64[];
vga_N_lJ3 = Float64[]

vga_1_learnability_linear = Float64[];
vga_1_learnability_cubic = Float64[]
vga_1_lJ1 = Float64[];
vga_1_lJ3 = Float64[]

# Results arrays for the Regular solver (FCS2Erf_Cubic.jl — the module name
# keeps its original filename, but everything user-facing below calls this
# solver "Regular" so it never gets confused with the cubic *mode*)
reg_N_learnability_linear = Float64[];
reg_N_learnability_cubic = Float64[]
reg_N_lJ1 = Float64[];
reg_N_lJ3 = Float64[]

reg_1_learnability_linear = Float64[];
reg_1_learnability_cubic = Float64[]
reg_1_lJ1 = Float64[];
reg_1_lJ3 = Float64[]

# Initial guesses
guess_vga_N = [0.2, 0.01, 1.0, 0.05]
guess_vga_1 = [0.2, 0.01, 1.0, 0.05]

println("Running VGA & Regular solvers for χ = N and χ = 1 across α ∈ [1, 3] (P ∈ [d, d^3])...")

for (i, P_curr) in enumerate(P_vals)
    # ---------------------------------------------------------
    # 1. VGA Solver
    # ---------------------------------------------------------
    # chi = N
    params_vga_N = FCS2_VGA.ProblemParams2(
        d=d_val, κ=kappa_val, ϵ=epsilon_val, P=Float32(P_curr), n1=n1_val, χ=Float32(N_val), δ=delta_val
    )
    sol_vga_N = FCS2_VGA.solve_FCN2_Erf(params_vga_N, guess_vga_N; anneal_steps=3000, use_anneal=true)
    push!(vga_N_learnability_linear, sol_vga_N.learnability1)
    push!(vga_N_learnability_cubic, sol_vga_N.learnability3)
    push!(vga_N_lJ1, sol_vga_N.lJ1)
    push!(vga_N_lJ3, sol_vga_N.lJ3)
    if !isnan(sol_vga_N.lJ1) && !isnan(sol_vga_N.sigS)
        global guess_vga_N = [sol_vga_N.lJ1, sol_vga_N.lJ3, sol_vga_N.sigS, sol_vga_N.muW]
    end

    # chi = 1
    params_vga_1 = FCS2_VGA.ProblemParams2(
        d=d_val, κ=kappa_val, ϵ=epsilon_val, P=Float32(P_curr), n1=n1_val, χ=1.0f0, δ=delta_val
    )
    sol_vga_1 = FCS2_VGA.solve_FCN2_Erf(params_vga_1, guess_vga_1; anneal_steps=3000, use_anneal=true)
    push!(vga_1_learnability_linear, sol_vga_1.learnability1)
    push!(vga_1_learnability_cubic, sol_vga_1.learnability3)
    push!(vga_1_lJ1, sol_vga_1.lJ1)
    push!(vga_1_lJ3, sol_vga_1.lJ3)
    if !isnan(sol_vga_1.lJ1) && !isnan(sol_vga_1.sigS)
        global guess_vga_1 = [sol_vga_1.lJ1, sol_vga_1.lJ3, sol_vga_1.sigS, sol_vga_1.muW]
    end

    # ---------------------------------------------------------
    # 2. Regular Solver (FCS2Erf_Cubic module)
    # ---------------------------------------------------------
    # chi = N
    params_reg_N = FCS2Erf_Cubic.ProblemParams2(
        d=Float32(d_val), κ=Float32(kappa_val), ϵ=Float32(epsilon_val), P=Float32(P_curr), n1=Float32(N_val), χ=Float32(N_val), δ=1.0
    )
    sol_reg_N = FCS2Erf_Cubic.solve_FCN2_Erf(
        params_reg_N, [1.0 / d_val, 1.0 / d_val^3, 1.0 / d_val]; lr=1e-6, max_iter=1_000_000, anneal_steps=30_000, use_anneal=true
    )
    push!(reg_N_learnability_linear, sol_reg_N.learnability1)
    push!(reg_N_learnability_cubic, sol_reg_N.learnability3)
    push!(reg_N_lJ1, sol_reg_N.lJ1)
    push!(reg_N_lJ3, sol_reg_N.lJ3)

    # chi = 1
    params_reg_1 = FCS2Erf_Cubic.ProblemParams2(
        d=Float32(d_val), κ=Float32(kappa_val), ϵ=Float32(epsilon_val), P=Float32(P_curr), n1=Float32(N_val), χ=1.0f0, δ=1.0
    )
    sol_reg_1 = FCS2Erf_Cubic.solve_FCN2_Erf(
        params_reg_1, [1.0 / d_val, 1.0 / d_val^3, 1.0 / d_val]; lr=1e-6, max_iter=1_000_000, anneal_steps=30_000, use_anneal=true
    )
    push!(reg_1_learnability_linear, sol_reg_1.learnability1)
    push!(reg_1_learnability_cubic, sol_reg_1.learnability3)
    push!(reg_1_lJ1, sol_reg_1.lJ1)
    push!(reg_1_lJ3, sol_reg_1.lJ3)

    println("Step $i/$P_samples (α = $(round(alpha_vals[i], digits=3))): Done.")
end

# ---------------------------------------------------------
# Plot 1: Learnability, linear mode only (χ=1 vs χ=N, VGA vs Regular)
# ---------------------------------------------------------
Y_learnability_linear = hcat(
    vga_N_learnability_linear, vga_1_learnability_linear,
    reg_N_learnability_linear, reg_1_learnability_linear
)

p1_linear = plot(
    alpha_vals,
    Y_learnability_linear,
    label=SERIES_LABELS,
    color=SERIES_COLORS,
    lw=2.5,
    xlabel=L"\alpha \ (P = d^\alpha)",
    ylabel="Learnability",
    title="Learnability Scaling — Linear Mode",
    legend=:outerright,
    grid=true
)

# ---------------------------------------------------------
# Plot 2: Learnability, cubic mode only (χ=1 vs χ=N, VGA vs Regular)
# ---------------------------------------------------------
Y_learnability_cubic = hcat(
    vga_N_learnability_cubic, vga_1_learnability_cubic,
    reg_N_learnability_cubic, reg_1_learnability_cubic
)

p1_cubic = plot(
    alpha_vals,
    Y_learnability_cubic,
    label=SERIES_LABELS,
    color=SERIES_COLORS,
    lw=2.5,
    xlabel=L"\alpha \ (P = d^\alpha)",
    ylabel="Learnability",
    title="Learnability Scaling — Cubic Mode",
    legend=:outerright,
    grid=true
)

# ---------------------------------------------------------
# Plot 3: Growth of the linear eigenvalue λ^J_1
# ---------------------------------------------------------
Y_lJ1 = hcat(vga_N_lJ1, vga_1_lJ1, reg_N_lJ1, reg_1_lJ1)

p2 = plot(
    alpha_vals,
    Y_lJ1,
    label=SERIES_LABELS,
    color=SERIES_COLORS,
    lw=2.5,
    xlabel=L"\alpha \ (P = d^\alpha)",
    ylabel=L"\lambda^J_1",
    title=L"Growth of $\lambda^J_1$ vs $\alpha$",
    legend=:topleft,
    grid=true
)

# ---------------------------------------------------------
# Plot 4: Growth of the cubic eigenvalue λ^J_3
# ---------------------------------------------------------
Y_lJ3 = hcat(vga_N_lJ3, vga_1_lJ3, reg_N_lJ3, reg_1_lJ3)

p3 = plot(
    alpha_vals,
    Y_lJ3,
    label=SERIES_LABELS,
    color=SERIES_COLORS,
    lw=2.5,
    xlabel=L"\alpha \ (P = d^\alpha)",
    ylabel=L"\lambda^J_3",
    title=L"Growth of $\lambda^J_3$ vs $\alpha$",
    legend=:topleft,
    grid=true
)

# Each plot is its own file. No combined figure, and linear/cubic never
# share an axes.
savefig(p1_linear, outpath("learnability_linear.png"))
savefig(p1_cubic, outpath("learnability_cubic.png"))
savefig(p2, outpath("eigenvalue_lJ1.png"))
savefig(p3, outpath("eigenvalue_lJ3.png"))

println("All plots saved to \"$(OUTDIR)/\"")