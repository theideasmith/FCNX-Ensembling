

# kappa_sweep_fast.jl
# Fast kappa sweep comparing two scaling regimes: standard (chi=1) and mean-field (chi=N)

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

# Common parameters
d = 100
N = 200
n1, n2 = Float64(N), Float64(N)
delta = 1.0
ds = 10.0 .^ range(0.5, 3, length=30)

println("="^60)
println("KAPPA SWEEP: STANDARD vs MEAN-FIELD SCALING")
println("="^60)
println("d=$(d), N=$(N), delta=$(delta)")
println("Comparing: chi=1 (standard) vs chi=N (mean-field)")
println("="^60)

# ============================================
# Condition 1: Standard Scaling (chi = 1)
# ============================================
println("\n[1/2] Running standard scaling (chi=1)...")
chi_std = 1.0
results_std = []

for (i, d) in enumerate(ds)
    initial_guess = [1.0 / sqrt(d), 1.0 / sqrt(d)]

    P = 10 * d
    κ = 1 / d^3
    d_local = d
    n_local = 5 * d
    sol = FCSLinear.nlsolve_solver(
        initial_guess; P=P, chi=chi_std, d=d_local, kappa=κ, delta=delta,
        n1=n_local, n2=n_local, max_iter=50_000, anneal_steps=20_000,
        lr=1e-4, verbose=false, anneal=true
    )

    solP = FCSLinear.nlsolve_solver(
        initial_guess; P=P, chi=chi_std, d=d_local, kappa=κ, delta=0.0,
        n1=n_local, n2=n_local, max_iter=50_000, anneal_steps=20_000,
        lr=1e-4, verbose=false, anneal=true
    )
    
    if sol === nothing
        println("  [$i/30] κ=$(sigfig(κ, n=3)): FAILED")
        continue
    end
    
    lJ, lH = sol
    lJP, lHP = solP
    lK_arr = FCSLinear.compute_lK(sol, P, n1, n2, chi_std, d_local, delta, κ)

    bias = (κ / (lH + κ))^2
    println("bias: ", bias)
    variance = (κ) / (chi_std * P) * lH/ (lH + κ) 
    println("variance: ", variance)
    loss = (bias + variance) * chi_std / κ 
    push!(results_std, (κ=κ, lJ=lJ, lH=lH, lK=lK_arr[1], disc=bias, loss=loss,d=d_local))
    println("  [$i/30] κ=$(sigfig(κ, n=3)) ✓")
end

# ============================================
# Condition 2: Mean-Field Scaling (chi = N)
# ============================================
println("\n[2/2] Running mean-field scaling (chi=N)...")
chi_mf = Float64(N)
results_mf = []


for (i, d) in enumerate(ds)
    initial_guess = [1.0 / sqrt(d), 1.0 / sqrt(d)]

    P = 10 * d
    κ = 1 / d^3
    d_local = d
    n_local = 5 * d
    sol = FCSLinear.nlsolve_solver(
        initial_guess; P=P, chi=chi_mf, d=d_local, kappa=κ, delta=delta,
        n1=n_local, n2=n_local, max_iter=50_000, anneal_steps=20_000,
        lr=1e-4, verbose=false, anneal=true
    )

    solP = FCSLinear.nlsolve_solver(
        initial_guess; P=P, chi=chi_mf, d=d_local, kappa=κ, delta=0.0,
        n1=n_local, n2=n_local, max_iter=50_000, anneal_steps=20_000,
        lr=1e-4, verbose=false, anneal=true
    )

    if sol === nothing
        println("  [$i/30] κ=$(sigfig(κ, n=3)): FAILED")
        continue
    end

    lJ, lH = sol
    lJP, lHP = solP
    lK_arr = FCSLinear.compute_lK(sol, P, n1, n2, chi_std, d_local, delta, κ)

    bias = (κ / (lH + κ) )^2
    println("bias: ", bias)
    variance = (κ) / (chi_mf * P) * 1 / (lH + κ) 
    println("variance: ", variance)
    loss = (bias + variance) * chi_mf / κ
    push!(results_mf, (κ=κ, lJ=lJ, lH=lH, lK=lK_arr[1], disc=bias, loss=loss,d=d_local))
    println("  [$i/30] κ=$(sigfig(κ, n=3)) ✓")
end

# Extract results
ks_std = [r.κ for r in results_std]
lJs_std = [r.lJ for r in results_std]
lHs_std = [r.lH for r in results_std]
lKs_std = [r.lK for r in results_std]
losses_std = [r.loss for r in results_std]
ds_std = [r.d for r in results_std]

ks_mf = [r.κ for r in results_mf]
lJs_mf = [r.lJ for r in results_mf]
lHs_mf = [r.lH for r in results_mf]
lKs_mf = [r.lK for r in results_mf]
losses_mf = [r.loss for r in results_mf]
ds_mf = [r.d for r in results_mf]
# ============================================
# Overlay Plots
# ============================================
println("\nGenerating overlay plots...")

# Determine y-axis ranges and create better tick marks
all_lJs = vcat(lJs_std, lJs_mf)
all_lHs = vcat(lHs_std, lHs_mf)
all_losses = vcat(losses_std, losses_mf)

yticks_lJ = 10.0 .^ range(floor(log10(minimum(all_lJs))), ceil(log10(maximum(all_lJs))), length=8)
yticks_lH = 10.0 .^ range(floor(log10(minimum(all_lHs))), ceil(log10(maximum(all_lHs))), length=8)
yticks_loss = 10.0 .^ range(floor(log10(minimum(all_losses))), ceil(log10(maximum(all_losses))), length=8)

# p1 = plot(ds_std, lJs_std, xscale=:log10, yscale=:log10, xlabel="d", ylabel="lJ", 
#             title="J Eigenvalue (Target)", legend=:best, linewidth=2.5, 
#             color=:blue, label="Standard (χ=1)", marker=:circle, markersize=3,
#             yticks=yticks_lJ, minorgrid=true)
# plot!(p1, ds_mf, lJs_mf, xscale=:log10, yscale=:log10, 
#       linewidth=2.5, color=:darkblue, label="Mean-Field (χ=N)", 
#       linestyle=:dash, marker=:square, markersize=3)

# p2 = plot(ds_std, lHs_std, xscale=:log10, yscale=:log10, xlabel="d", ylabel="lH", 
#             title="H Eigenvalue (Target)", legend=:best, linewidth=2.5, 
#             color=:red, label="Standard (χ=1)", marker=:circle, markersize=3,
#             yticks=yticks_lH, minorgrid=true)
# plot!(p2, ds_mf, lHs_mf, xscale=:log10, yscale=:log10, 
#       linewidth=2.5, color=:darkred, label="Mean-Field (χ=N)", 
#       linestyle=:dash, marker=:square, markersize=3)

p4 = plot(ds_std, losses_std, xscale=:log10, yscale=:log10, xlabel="d", ylabel="Loss Term (Bias + Variance)", 
    title="Loss Scaling (Target) \n kappa=1 / d^3 P=10*d N=5*d under Standard Scaling", legend=:best, linewidth=2.5,
         color=:purple, label="Standard (χ=1)", marker=:circle, markersize=3,
         yticks=yticks_loss, minorgrid=true)
p3 = plot(ds_mf, losses_mf, xscale=:log10, yscale=:log10, 
        title="Loss Scaling (Target) \n kappa=1 / d^3 P=10*d N=5*d under MF Scaling", xlabel="d", ylabel="Loss Term (Bias + Variance)",
      linewidth=2.5, color=:indigo, label="Mean-Field (χ=N)", 
      linestyle=:dash, marker=:square, markersize=3)

# fig = plot(p1, p2, p3, p4, size=(1400, 1000), layout=(2, 2), margin=5Plots.mm)
fig = plot(p3, p4, size=(1400, 1000), layout=(2, 1), margin=5Plots.mm)
savefig(fig, "/home/akiva/FCNX-Ensembling/plots/kappa_sweep_overlay.png")
println("\nPlot saved: /home/akiva/FCNX-Ensembling/plots/kappa_sweep_overlay.png")
println("="^60)