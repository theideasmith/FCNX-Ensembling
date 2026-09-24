#!/usr/bin/env julia
"""
Theory-only W0 / H0 action overlays along the invariant (alpha, beta) ray:

    alpha = alpha0 * beta^gamma
    d, N  ~ beta^{eps,nu}
    sa0   = SIGMA_A0 * beta^rho
    P     = P0 * alpha^lambda
    kappa = kappa0 * (alpha/beta)^omega
    with lambda = omega = eps+nu, rho = -omega

Hybrid defaults: steepwell ray (SIGMA_A0=0.03, cold κ₀=1/N0) with
d50-like TASK_EPS=0.5 so cubic learnability is strong while W₀ wells
deepen along β via falling sa0.

Matches red_robin_alpha_beta_invariant_steepwell.py (alpha-growing form).
"""

using Printf
using Plots
using Serialization
using Statistics

push!(LOAD_PATH, @__DIR__)
include(joinpath(@__DIR__, "FCS2_VGA.jl"))
using .FCS2_VGA

gr()

const EPSILON = 0.5
const NU = 0.25
const OMEGA = EPSILON + NU
const RHO = -OMEGA
const LAMBDA = OMEGA

const D0 = 5.0
const N0 = 1000.0
const SIGMA_A0 = 0.03
const KAPPA0 = 1.0 / N0
const P0 = 160.0
const ALPHA0 = 1.0
const GAMMA = 1.25
const S0 = 1.0
# Hybrid: keep steepwell ray (small sa0 / cold κ ⇒ deep W₀ wells) but use
# d50-like cubic strength so L₃ is learnable, not ε=0.03-lazy.
const TASK_EPS = 0.5
const CHI_TRAIN = 1.0
const D_MAX = 50.0
const BETA_MAX = (D_MAX / D0)^(1 / EPSILON)
const NUM_POINTS = 5

const OUTDIR = joinpath(@__DIR__, "invariant_vga_action_overlays")
const OFFDIAG = true

gaussian_action(x, var) = @. 0.5 * x^2 / var + 0.5 * log(2π * var)

function bimodal_action(x, mu, sig; eps=1e-12)
    s = max(sig, 1e-12)
    dens = @. 0.5 * exp(-0.5 * ((x - mu) / s)^2) / (s * sqrt(2π)) +
             0.5 * exp(-0.5 * ((x + mu) / s)^2) / (s * sqrt(2π))
    return @. -log(dens + eps)
end

function scaled_params(beta::Float64)
    alpha = ALPHA0 * beta^GAMMA
    d = D0 * beta^EPSILON
    n = N0 * beta^NU
    a0 = SIGMA_A0 * beta^RHO
    P = P0 * alpha^LAMBDA
    kappa = KAPPA0 * (alpha / beta)^OMEGA
    return (
        beta=beta, alpha=alpha, d=d, N=n, a0=a0, P=P, kappa=kappa, T=2kappa,
        chi=CHI_TRAIN, s0=S0,
    )
end


function fraction_tex(x; maxden=16, tol=1e-8)
    ax = abs(x)
    for den in 1:maxden
        num = round(Int, ax * den)
        if abs(ax - num / den) <= tol
            body = den == 1 ? string(num) : "\\frac{$num}{$den}"
            return x < 0 ? "-" * body : body
        end
    end
    return @sprintf("%g", x)
end


function scaling_caption()
    ε = fraction_tex(EPSILON)
    ν = fraction_tex(NU)
    ω = fraction_tex(OMEGA)
    ρ = fraction_tex(RHO)
    λ = fraction_tex(LAMBDA)
    γ = fraction_tex(GAMMA)
    lines = [
        raw"Action $F = 2P^{2}\sigma_a^{2}/(\pi\kappa^{2}\,d\,N)$ held invariant",
        "Ray \$\\alpha=\\alpha_0\\beta^{\\gamma}\$ with \$\\alpha_0=$(ALPHA0)\$, \$\\gamma=$γ\$",
        raw"$d=d_0\beta^{\epsilon},\quad N=N_0\beta^{\nu},\quad \sigma_a^{2}=\sigma_{a0}\beta^{\rho}$",
        raw"$P=P_0\alpha^{\lambda},\quad \kappa=\kappa_0(\alpha/\beta)^{\omega},\quad T=2\kappa,\quad \chi=" *
            string(Int(CHI_TRAIN)) * raw"$",
        "\$\\epsilon=$ε,\\quad \\nu=$ν,\\quad \\rho=$ρ,\\quad \\lambda=$λ,\\quad \\omega=$ω\$",
        "task \$\\varepsilon=$(TASK_EPS)\$, \$d_0=$(Int(D0))\$, \$N_0=$(Int(N0))\$, \$P_0=$(Int(P0))\$, " *
        "\$\\sigma_{a0}=$SIGMA_A0\$, \$\\kappa_0=$KAPPA0\$, \$s_0=$S0\$" *
        (OFFDIAG ? ", VGA offdiag" : ""),
    ]
    return join(lines, "\n")
end


const LATEX_FOOTNOTE_PY = raw"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

png_path, cap_path = sys.argv[1], sys.argv[2]
fontsize = float(sys.argv[3])
dpi = float(sys.argv[4])
cap = open(cap_path, encoding="utf-8").read().rstrip("\n")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

img = mpimg.imread(png_path)
h, w = img.shape[:2]
n_lines = cap.count("\n") + 1
foot_in = 0.10 + 0.20 * n_lines * (fontsize / 9.0)
fig_w = w / dpi
inner_in = h / dpi
fig_h = inner_in + foot_in

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
ax = fig.add_axes([0.0, foot_in / fig_h, 1.0, inner_in / fig_h])
ax.imshow(img)
ax.set_axis_off()
ax.set_xlim(0, max(w - 1, 1))
ax.set_ylim(max(h - 1, 1), 0)
fig.text(
    0.5, 0.014, cap,
    ha="center", va="bottom",
    fontsize=fontsize, color="0.15",
    linespacing=1.45,
)
fig.savefig(png_path, dpi=dpi, facecolor="white")
plt.close(fig)
"""


function apply_latex_footnote(png_path; fontsize=10, dpi=180)
    cap_path = tempname() * ".tex.txt"
    py_path = tempname() * ".py"
    write(cap_path, scaling_caption())
    write(py_path, LATEX_FOOTNOTE_PY)
    try
        run(`python3 $py_path $png_path $cap_path $fontsize $dpi`)
    finally
        rm(cap_path; force=true)
        rm(py_path; force=true)
    end
end


function savefig_with_scaling(plt, path; width, inner_height, dpi=180, fontsize=10, foot_frac=0.0)
    plot!(plt; size=(width, inner_height), dpi=dpi)
    savefig(plt, path)
    apply_latex_footnote(path; fontsize=fontsize, dpi=dpi)
    println("Saved $path")
end


const RAY_META = (
    EPSILON, NU, RHO, LAMBDA, OMEGA, D0, N0, SIGMA_A0, KAPPA0, P0,
    ALPHA0, GAMMA, TASK_EPS, CHI_TRAIN, S0, OFFDIAG, NUM_POINTS, D_MAX,
)
const SOL_CACHE = joinpath(OUTDIR, "vga_ray_sols.jls")

function solve_target(p; guess=nothing, offdiag::Bool=OFFDIAG)
    init = isnothing(guess) ? [1.0 / p.d, 1.0 / p.d^3, sqrt(0.8 / p.d), 0.3] : guess
    params = ProblemParams2(
        d=Float32(p.d), κ=Float32(p.kappa), ϵ=Float32(TASK_EPS),
        P=Float32(p.P), n1=Float32(p.N), χ=Float32(p.chi),
        δ=1.0f0, s0=Float32(p.s0), a0=Float32(p.a0),
    )
    return solve_FCN2_Erf(
        params, init;
        anneal_steps=3000, use_anneal=true,
        offdiag=offdiag, freeze_U=true,
    )
end

function main()
    mkpath(OUTDIR)
    betas = exp.(range(log(1.0), log(BETA_MAX); length=NUM_POINTS))
    points = [scaled_params(b) for b in betas]

    println("="^72)
    println("Invariant VGA overlay  (α=α₀β^γ, γ=$GAMMA, α₀=$ALPHA0, sa0₀=$SIGMA_A0)")
    println("  eps=$EPSILON, nu=$NU, chi=$CHI_TRAIN, s0=$S0, task_eps=$TASK_EPS, offdiag=$OFFDIAG")
    println("="^72)
    @printf("%8s %8s %5s %5s %5s %8s %10s %10s %10s\n",
            "beta", "alpha", "d", "P", "N", "a0", "kappa", "P/κ", "F")
    for p in points
        F = 2 * p.P^2 * p.a0 / (π * p.kappa^2 * p.d * p.N)
        @printf("%8.3f %8.3f %5.0f %5.0f %5.0f %8.4f %10.4g %10.1f %10.4g\n",
                p.beta, p.alpha, round(p.d), round(p.P), round(p.N),
                p.a0, p.kappa, p.P / p.kappa, F)
    end
    println()

    sols = Any[]
    cache_ok = false
    if isfile(SOL_CACHE)
        try
            data = deserialize(SOL_CACHE)
            if data.meta == RAY_META && length(data.sols) == length(points)
                sols = data.sols
                cache_ok = true
                println("Loaded VGA solutions from $SOL_CACHE")
            end
        catch err
            println("VGA cache unreadable ($err); re-solving")
        end
    end
    if !cache_ok
        for p in points
        println("Solving VGA  β=$(round(p.beta; digits=3))  α=$(round(p.alpha; digits=2))  d=$(round(Int, p.d))  P=$(round(Int, p.P))  a0=$(round(p.a0; digits=4))  κ=$(round(p.kappa; sigdigits=4)) ...")
        # Moderate μ seeds only: too-large μ collapses onto the μ≈0 saddle.
        # Prefer the candidate with largest |μ| among successful solves.
        candidates = Any[]
        for mu0 in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
            for sig0 in (max(sqrt(0.4 / p.d), 0.12), 0.18, 0.22, 0.28)
                push!(candidates, [1.0 / p.d, 1.0 / p.d^3, sig0, mu0])
            end
        end
        sol = nothing
        for guess in candidates
            cand = solve_target(p; guess=guess)
            isnan(cand.muW) && continue
            if sol === nothing || abs(cand.muW) > abs(sol.muW)
                sol = cand
            end
            abs(cand.muW) > 0.2 && break  # solid bimodal hit
        end
        push!(sols, sol)
        @printf(
            "  lJ1=%.4g  lJ3=%.4g  muW=%.4g  sigS=%.4g  lWT=%.4g  L1=%.4g  L3=%.4g%s\n",
            sol.lJ1, sol.lJ3, sol.muW, sol.sigS, sol.lWT,
            sol.learnability1, sol.learnability3,
            abs(sol.muW) > 0.05 ? "  [bimodal]" : "  [UNIMODAL]",
        )
        end
        serialize(SOL_CACHE, (meta=RAY_META, points=points, sols=sols))
        println("Wrote $SOL_CACHE")
    else
        for (p, sol) in zip(points, sols)
            @printf(
                "  β=%.3g  lJ1=%.4g  lJ3=%.4g  muW=%.4g  sigS=%.4g  lWT=%.4g  L1=%.4g  L3=%.4g%s\n",
                p.beta, sol.lJ1, sol.lJ3, sol.muW, sol.sigS, sol.lWT,
                sol.learnability1, sol.learnability3,
                abs(sol.muW) > 0.05 ? "  [bimodal]" : "  [UNIMODAL]",
            )
        end
    end

    cmap = cgrad(:viridis)
    βmin, βmax = extrema(betas)
    col(b) = cmap[clamp((log(b) - log(βmin)) / (log(βmax) - log(βmin) + 1e-12), 0, 1)]

    sigW = [s.sigS for s in sols if !isnan(s.sigS)]
    muW  = [abs(s.muW) for s in sols if !isnan(s.muW)]
    xmax_w = isempty(sigW) ? 1.5 : maximum(muW .+ 4 .* sigW)
    xw = range(-xmax_w, xmax_w; length=500)

    lJ1s = [s.lJ1 for s in sols if s.lJ1 > 0]
    xmax1 = isempty(lJ1s) ? 0.4 : 4 * sqrt(maximum(lJ1s))
    x1 = range(-xmax1, xmax1; length=500)

    lJ3s = [s.lJ3 for s in sols if s.lJ3 > 0]
    xmax3 = isempty(lJ3s) ? 0.05 : 4 * sqrt(maximum(lJ3s))
    x3 = range(-xmax3, xmax3; length=500)

    plt_w = plot(
        title="W₀ action  α=α₀β^γ  γ=$GAMMA",
        xlabel="w", ylabel="S(w)=−log p(w)",
        legend=:topright, size=(900, 550), dpi=180,
    )
    for (p, sol) in zip(points, sols)
        any(isnan, (sol.muW, sol.sigS)) && continue
        yw = bimodal_action(collect(xw), abs(sol.muW), sol.sigS)
        plot!(plt_w, xw, yw; color=col(p.beta), lw=2.2,
              label=@sprintf("β=%.2g  μ=%.2g σ=%.2g  μ/σ=%.2g",
                             p.beta, abs(sol.muW), sol.sigS, abs(sol.muW)/sol.sigS))
    end
    smins = Float64[]; s0s = Float64[]
    for sol in sols
        any(isnan, (sol.muW, sol.sigS)) && continue
        mu = abs(sol.muW)
        push!(smins, bimodal_action([mu], mu, sol.sigS)[1])
        push!(s0s, bimodal_action([0.0], mu, sol.sigS)[1])
    end
    if !isempty(smins)
        plot!(plt_w; ylim=(minimum(smins) - 0.05, maximum(s0s) + 0.25))
    end
    savefig_with_scaling(
        plt_w, joinpath(OUTDIR, "w0_bimodal_action_vs_beta.png");
        width=900, inner_height=550, dpi=180,
    )

    plt_h1 = plot(title="H₀ linear (ℓJ₁)", xlabel="proj", ylabel="S",
                  legend=:topright, size=(900, 550), dpi=180)
    for (p, sol) in zip(points, sols)
        sol.lJ1 > 0 || continue
        y = gaussian_action(collect(x1), sol.lJ1)
        plot!(plt_h1, x1, y; color=col(p.beta), lw=2.2,
              label=@sprintf("β=%.2g  ℓJ₁=%.3g", p.beta, sol.lJ1))
    end
    plot!(plt_h1; ylim=(-2.0, 20))
    savefig_with_scaling(
        plt_h1, joinpath(OUTDIR, "h0_linear_action_vs_beta.png");
        width=900, inner_height=550, dpi=180,
    )

    plt_h3 = plot(title="H₀ cubic (ℓJ₃)", xlabel="proj", ylabel="S",
                  legend=:topright, size=(900, 550), dpi=180)
    for (p, sol) in zip(points, sols)
        sol.lJ3 > 0 || continue
        y = gaussian_action(collect(x3), sol.lJ3)
        plot!(plt_h3, x3, y; color=col(p.beta), lw=2.2,
              label=@sprintf("β=%.2g  ℓJ₃=%.3g", p.beta, sol.lJ3))
    end
    plot!(plt_h3; ylim=(-6.5, 40))
    savefig_with_scaling(
        plt_h3, joinpath(OUTDIR, "h0_cubic_action_vs_beta.png");
        width=900, inner_height=550, dpi=180,
    )

    plt = plot(plt_w, plt_h1, plt_h3; layout=(1, 3), size=(1600, 480), dpi=160)
    savefig_with_scaling(
        plt, joinpath(OUTDIR, "w0_h0_actions_overlay_vs_beta.png");
        width=1600, inner_height=480, dpi=160, fontsize=11,
    )

    println("\nW₀ barrier + floors:")
    @printf("%8s %10s %10s %10s %10s %10s %10s %10s\n",
            "beta", "μ", "σ", "μ/σ", "S(μ)", "S(0)", "barrier", "κ")
    for (p, sol) in zip(points, sols)
        any(isnan, (sol.muW, sol.sigS)) && continue
        mu = abs(sol.muW)
        Smu = bimodal_action([mu], mu, sol.sigS)[1]
        S0v = bimodal_action([0.0], mu, sol.sigS)[1]
        @printf("%8.3f %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g\n",
                p.beta, mu, sol.sigS, mu / sol.sigS, Smu, S0v, S0v - Smu, p.kappa)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
