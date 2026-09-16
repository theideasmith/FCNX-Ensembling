#!/usr/bin/env julia
# Diagnose FCS2_VGA failure on parameters used by plot_action_h0_activation.py

using LinearAlgebra
using ForwardDiff
using NLsolve
using Plots
using Printf

push!(LOAD_PATH, @__DIR__)
include(joinpath(@__DIR__, "FCS2_VGA.jl"))
using .FCS2_VGA

const OUTDIR = joinpath(@__DIR__, "vga_diagnostics")
mkpath(OUTDIR)

# Parameters taken from EXPERIMENT_GROUPS in plot_action_h0_activation.py.
# bare kappa = T/2 as in bare_kappa_from_config.
const CASES = [
    (name="ScaledDown_P250_s0=1",     d=100.0, P=250.0,  n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
    (name="ScaledDown_P500_s0=0.25",  d=100.0, P=500.0,  n1=700.0, chi=700.0, kappa=0.05, s0=0.25,     eps=0.074),
    (name="ScaledDown_P750_s0=0.11",  d=100.0, P=750.0,  n1=700.0, chi=700.0, kappa=0.05, s0=1/9,      eps=0.074),
    (name="ScaledDown_P1000_s0=0.06", d=100.0, P=1000.0, n1=700.0, chi=700.0, kappa=0.05, s0=0.0625,   eps=0.074),
    (name="SampleComp_P100",          d=100.0, P=100.0,  n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
    (name="SampleComp_P400",          d=100.0, P=400.0,  n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
    (name="SampleComp_P700",          d=100.0, P=700.0,  n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
    (name="SampleComp_P1000",         d=100.0, P=1000.0, n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
    (name="SampleComp_P4000",         d=100.0, P=4000.0, n1=700.0, chi=700.0, kappa=0.05, s0=1.0,      eps=0.074),
]

function lT1_current(chi, kappa, P, delta, lJ1)
    return -(chi^2 / (kappa/P + lJ1)^2 * delta) +
        chi / lJ1 +
        (chi * kappa / P) * (lJ1 / (lJ1 + kappa/P))
end

function lT1_fcs_signs(chi, kappa, P, delta, lJ1)
    return -(chi^2 / (kappa/P + lJ1)^2 * delta) -
        chi / lJ1 -
        (chi * kappa / P) * (lJ1 / (lJ1 + kappa/P))
end

function A_pot(chi, kappa, P, delta, n1, lJ1; sign_mode=:current)
    lT = sign_mode === :current ?
        lT1_current(chi, kappa, P, delta, lJ1) :
        lT1_fcs_signs(chi, kappa, P, delta, lJ1)
    return lT / (n1 * chi)
end

function free_energy_fixed_A(s, m, A, T, d, s0)
    prior = (d / s0) * (1 / 2) * (m^2 + s^2)
    energy = A * FCS2_VGA.compute_lambda1(m, s, T)
    return prior + energy - exact_symmetric_gmm_entropy_1d(m, s)
end

function stationarity_grad(s, m; d, P, n1, chi, kappa, s0, delta, sign_mode=:current)
    T = 1.0 + 2.0 * (d - 1.0) * s0 / d
    lJ1 = FCS2_VGA.compute_lambda1(m, s, T)
    A = A_pot(chi, kappa, P, delta, n1, lJ1; sign_mode=sign_mode)
    return ForwardDiff.gradient(v -> free_energy_fixed_A(v[1], v[2], A, T, d, s0), [s, m])
end

function solve_case(case; delta=1.0, anneal_steps=40, tol=1e-8)
    params = ProblemParams2(
        d=Float32(case.d), κ=Float32(case.kappa), ϵ=Float32(case.eps),
        P=Float32(case.P), n1=Float32(case.n1), χ=Float32(case.chi),
        δ=Float32(delta), s0=Float32(case.s0),
    )
    guess = [1.0 / case.d, 1.0 / case.d^3, sqrt(0.8 / case.d), 0.1]
    sol = solve_FCN2_Erf(params, guess; anneal_steps=anneal_steps, use_anneal=true, tol=tol)
    x = [sol.lJ1, sol.lJ3, sol.sigS, sol.muW]
    res = residuals_fcn2(
        x, case.P, case.chi, case.d, case.kappa, delta, case.n1, case.s0,
    )
    return sol, x, res, norm(res)
end

function scan_ds_along_m0(case; delta=1.0, sign_mode=:current)
    ss = range(1e-3, 1.5; length=120)
    grads = Float64[]
    As = Float64[]
    for s in ss
        g = stationarity_grad(s, 0.0; d=case.d, P=case.P, n1=case.n1, chi=case.chi,
            kappa=case.kappa, s0=case.s0, delta=delta, sign_mode=sign_mode)
        push!(grads, g[1])
        T = 1.0 + 2.0 * (case.d - 1.0) * case.s0 / case.d
        lJ1 = FCS2_VGA.compute_lambda1(0.0, s, T)
        push!(As, A_pot(case.chi, case.kappa, case.P, delta, case.n1, lJ1; sign_mode=sign_mode))
    end
    return collect(ss), grads, As
end

function mixture_action(x, m, s; eps=1e-12)
    dens = 0.5 * exp(-0.5 * ((x - m) / s)^2) / (s * sqrt(2π)) +
           0.5 * exp(-0.5 * ((x + m) / s)^2) / (s * sqrt(2π))
    return -log(dens + eps)
end

println("="^80)
println("VGA diagnostics on plotting-code parameters")
println("="^80)

summary_rows = NamedTuple[]
plt_ds = plot(xlabel="σ (sigS)", ylabel="∂F/∂σ |_{m=0}",
    title="Stationarity of free energy (δ=1)", legend=:outertopright, size=(1000, 650))
plt_A = plot(xlabel="σ (sigS)", ylabel="A_pot",
    title="Training amplitude A_pot (δ=1)", legend=:outertopright, size=(1000, 650))

for case in CASES
    @printf("\n--- %s | d=%.0f P=%.0f n1=%.0f χ=%.0f κ=%.3g s0=%.4g ---\n",
        case.name, case.d, case.P, case.n1, case.chi, case.kappa, case.s0)

    sol, x, res, rnorm = solve_case(case; delta=1.0, anneal_steps=30)
    @printf("  returned: lJ1=%.4g lJ3=%.4g sigS=%.4g muW=%.4g lWT=%.4g\n",
        sol.lJ1, sol.lJ3, sol.sigS, sol.muW, sol.lWT)
    @printf("  residual: [%.3g, %.3g, %.3g, %.3g]  ||r||=%.4g\n",
        res[1], res[2], res[3], res[4], rnorm)

    ss, g_cur, A_cur = scan_ds_along_m0(case; delta=1.0, sign_mode=:current)
    _, g_fcs, A_fcs = scan_ds_along_m0(case; delta=1.0, sign_mode=:fcs)
    min_abs_cur = minimum(abs.(g_cur))
    min_abs_fcs = minimum(abs.(g_fcs))
    # zero crossings
    zc(g) = findall(i -> g[i] * g[i+1] < 0, 1:length(g)-1)
    zc_cur = [ss[i] for i in zc(g_cur)]
    zc_fcs = [ss[i] for i in zc(g_fcs)]
    @printf("  current signs: min|∂F/∂σ|=%.4g  zero-crossings at σ≈%s  A(at sol σ)≈%.4g\n",
        min_abs_cur, isempty(zc_cur) ? "NONE" : string(round.(zc_cur; sigdigits=3)),
        A_pot(case.chi, case.kappa, case.P, 1.0, case.n1,
            FCS2_VGA.compute_lambda1(sol.muW, sol.sigS, 1 + 2*(case.d-1)*case.s0/case.d); sign_mode=:current))
    @printf("  FCS signs:     min|∂F/∂σ|=%.4g  zero-crossings at σ≈%s\n",
        min_abs_fcs, isempty(zc_fcs) ? "NONE" : string(round.(zc_fcs; sigdigits=3)))

    push!(summary_rows, (
        name=case.name, P=case.P, s0=case.s0,
        sigS=sol.sigS, muW=sol.muW, lWT=sol.lWT, rnorm=rnorm,
        min_abs_cur=min_abs_cur, min_abs_fcs=min_abs_fcs,
        has_zc_cur=!isempty(zc_cur), has_zc_fcs=!isempty(zc_fcs),
    ))

    plot!(plt_ds, ss, g_cur; label="$(case.name) current", lw=2)
    plot!(plt_ds, ss, g_fcs; label="$(case.name) FCS signs", lw=2, ls=:dash)
    plot!(plt_A, ss, A_cur; label="$(case.name) current", lw=2)
    plot!(plt_A, ss, A_fcs; label="$(case.name) FCS signs", lw=2, ls=:dash)
end

hline!(plt_ds, [0.0]; color=:black, ls=:dot, label="0")
savefig(plt_ds, joinpath(OUTDIR, "stationarity_ds_m0.png"))
savefig(plt_A, joinpath(OUTDIR, "A_pot_vs_sigma.png"))

# Residual-norm summary bar plot
names = [r.name for r in summary_rows]
rnorms = [r.rnorm for r in summary_rows]
plt_bar = bar(names, rnorms;
    xrotation=35, ylabel="‖residual‖ after anneal",
    title="VGA solver residual norms (should be ~0 if converged)",
    legend=false, size=(1100, 550), color=:crimson)
savefig(plt_bar, joinpath(OUTDIR, "residual_norms.png"))

# Action overlay for one representative failing case
case = CASES[findfirst(c -> c.name == "SampleComp_P700", CASES)]
sol, _, res, rnorm = solve_case(case; delta=1.0, anneal_steps=30)
xs = range(-3, 3; length=400)
act = [mixture_action(x, sol.muW, max(sol.sigS, 1e-8)) for x in xs]
# Compare to moment-matched Gaussian with same second moment
σ2 = sol.lWT
act_gauss = [0.5 * x^2 / σ2 + 0.5 * log(2π * σ2) for x in xs]
plt_act = plot(xs, act; label=@sprintf("VGA mixture (μ=%.3g, σ=%.3g)", sol.muW, sol.sigS),
    lw=2, xlabel="w", ylabel="-log p(w)",
    title=@sprintf("Returned VGA action | %s | ‖r‖=%.3g", case.name, rnorm),
    size=(900, 550))
plot!(plt_act, xs, act_gauss; label=@sprintf("Moment-matched N(0, lWT=%.3g)", σ2), lw=2, ls=:dash)
savefig(plt_act, joinpath(OUTDIR, "action_samplecomp_p700.png"))

# Heatmap of |grad F| for one case under both sign conventions
function grad_norm_grid(case; sign_mode=:current, delta=1.0)
    ss = range(0.02, 1.2; length=50)
    ms = range(0.0, 1.2; length=50)
    Z = [norm(stationarity_grad(s, m; d=case.d, P=case.P, n1=case.n1, chi=case.chi,
            kappa=case.kappa, s0=case.s0, delta=delta, sign_mode=sign_mode))
         for m in ms, s in ss]
    return collect(ss), collect(ms), Z
end

ss, ms, Zcur = grad_norm_grid(case; sign_mode=:current)
_, _, Zfcs = grad_norm_grid(case; sign_mode=:fcs)
plt_h1 = heatmap(ss, ms, Zcur; xlabel="σ", ylabel="μ",
    title="‖∇F‖ current lT1 signs (P=700)", colorbar_title="‖∇F‖", size=(700, 550))
plt_h2 = heatmap(ss, ms, Zfcs; xlabel="σ", ylabel="μ",
    title="‖∇F‖ FCS lT1 signs (P=700)", colorbar_title="‖∇F‖", size=(700, 550))
plt_heat = plot(plt_h1, plt_h2; layout=(1, 2), size=(1400, 550))
savefig(plt_heat, joinpath(OUTDIR, "grad_norm_heatmaps_p700.png"))

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
@printf("%-28s %8s %8s %8s %8s %10s %10s %10s %s\n",
    "case", "P", "s0", "sigS", "muW", "‖r‖", "min|ds|cur", "min|ds|fcs", "zc?")
for r in summary_rows
    @printf("%-28s %8.0f %8.4g %8.4g %8.4g %10.3g %10.3g %10.3g cur=%s fcs=%s\n",
        r.name, r.P, r.s0, r.sigS, r.muW, r.rnorm, r.min_abs_cur, r.min_abs_fcs,
        r.has_zc_cur, r.has_zc_fcs)
end

println("\nWrote plots to $OUTDIR")
println("Key finding: if min|∂F/∂σ| stays O(1) and zc=false under current signs,")
println("the free energy has no critical point — NLsolve cannot drive stationarity residuals to 0.")
