module FCS2_VGA

using SpecialFunctions
using ForwardDiff
using LinearAlgebra
using NLsolve
using Base: @kwdef

export ProblemParams2, Solution2, residuals_fcn2, residuals_fcn2_advanced,
    solve_FCN2_Erf, exact_symmetric_gmm_entropy_1d

# ---------------------------------------------------------
# ForwardDiff Extension for Complex erfcx
# ---------------------------------------------------------
import SpecialFunctions: erfcx
function erfcx(z::Complex{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    z_val = Complex(ForwardDiff.value(z.re), ForwardDiff.value(z.im))
    f_val = erfcx(z_val)
    df_dz = 2.0 * z_val * f_val - 2.0 / sqrt(π)
    re_p = ForwardDiff.partials(z.re);
    im_p = ForwardDiff.partials(z.im)
    return Complex(
        ForwardDiff.Dual{T,V,N}(f_val.re, df_dz.re * re_p - df_dz.im * im_p),
        ForwardDiff.Dual{T,V,N}(f_val.im, df_dz.re * im_p + df_dz.im * re_p)
    )
end

# ---------------------------------------------------------
# Structs
# ---------------------------------------------------------
@kwdef mutable struct ProblemParams2
    d::Float32;
    κ::Float32;
    ϵ::Float32;
    P::Float32;
    n1::Float32;
    χ::Float32;
    b::Float32 = 4/(3*π);
    δ::Float32 = 0.0
    s0::Float32 = 1.0
end

@kwdef mutable struct Solution2
    lJ1=NaN;
    lJ3=NaN;
    lK1=NaN;
    lK3=NaN;
    lWT=NaN;
    sigS=NaN;
    muW=NaN;
    learnability1=NaN;
    learnability3=NaN;
    kappa_eff=NaN
end

# ---------------------------------------------------------
# Standard-normal Gauss-Hermite rules
# ---------------------------------------------------------

"""
    standard_normal_gh_rule(n)

Return an `n`-point Gauss-Hermite rule for expectations under
`Z ~ Normal(0, 1)`.  The Golub-Welsch construction avoids adding a
quadrature-package dependency.
"""
function standard_normal_gh_rule(n::Int)
    jacobi = SymTridiagonal(
        zeros(Float64, n),
        sqrt.(collect(1:(n - 1))),
    )
    eig = eigen(jacobi)
    return eig.values, vec(eig.vectors[1, :]).^2
end

const STANDARD_NORMAL_GH_RULES = Dict(
    tag => standard_normal_gh_rule(n)
    for (tag, n) in ((11, 11), (16, 16), (32, 32), (64, 64), (128, 128))
)

function gh_rule(rule::Int)
    return get(STANDARD_NORMAL_GH_RULES, rule) do
        throw(ArgumentError(
            "Unsupported quadrature rule $rule; use :11, :16, :32, :64, or :128.",
        ))
    end
end

# ---------------------------------------------------------
# Spectral Integrals
# ---------------------------------------------------------

# Legacy 11-point rule used by compute_lambda3 (restored).

const GH_NODES = [
  -4.45948490915965, -3.40143369785489, -2.43401875620939,
  -1.51593110691244, -0.60576387917106,  0.0,
   0.60576387917106,  1.51593110691244,  2.43401875620939,
   3.40143369785489,  4.45948490915965
];

const GH_WEIGHTS = [
    0.0000156321288126, 0.0007307882280209, 0.0108542240989218,
    0.0714241983888323, 0.2322760200832684, 0.3693994705541348,
    0.2322760200832684, 0.0714241983888323, 0.0108542240989218,
    0.0007307882280209, 0.0000156321288126
  ];

function v_base(μ, σ, T)
    q = sqrt(T / 2.0)
    z = complex(-μ, q) / (max(σ, 1e-10) * sqrt(2.0))
    return (sqrt(π) / (2.0 * max(σ, 1e-10) * sqrt(T))) * real(erfcx(-im * z))
end

function compute_lambda1(μ, σ, T)
    return (2.0 / π) * (1.0 - T * v_base(μ, σ, T))
end

function compute_lambda3(μ, σ, T)
    res = 0.0
    for i in 1:11
        w_val = μ + σ * GH_NODES[i]
        res += GH_WEIGHTS[i] * (w_val^6 / (T + 2.0 * w_val^2)^3)
    end
    return (8.0 / (3.0 * π)) * res
end

# ---------------------------------------------------------
# Symmetric GMM entropy
# ---------------------------------------------------------

"Entropy of a 1D Gaussian with variance `σ²` (nats)."
gaussian_entropy_1d(σ2) = 0.5 * log(2π * exp(1) * max(σ2, 1e-25))

"Stable softplus `log(1 + exp(x))` with a ForwardDiff-safe branch at `x = 0`."
function log1pexp(x)
    if x > zero(x)
        return x + log1p(exp(-x))
    else
        return log1p(exp(x))
    end
end

"""
    exact_symmetric_gmm_entropy_1d(m, s; rule=:64)

Gauss-Hermite estimate of the entropy of
`0.5 * Normal(m, s²) + 0.5 * Normal(-m, s²)`.  Supported standard-normal
quadrature rules are `:11`, `:16`, `:32`, `:64`, and `:128`; in Julia these
numeric quote expressions evaluate to integers.  `:64` is the default used by
the solver.
"""
function exact_symmetric_gmm_entropy_1d(m, s; rule::Int=64)
    nodes, weights = gh_rule(rule)

    s_safe = max(s, 1e-12)
    gaussian_entropy = gaussian_entropy_1d(s_safe^2)
    overlap_correction = zero(m + s)

    for (z, weight) in zip(nodes, weights)
        log_ratio = -2 * m^2 / s_safe^2 - 2 * m * z / s_safe
        overlap_correction += weight * log1pexp(log_ratio)
    end

    return gaussian_entropy + log(2) - overlap_correction
end

# ---------------------------------------------------------
# Core Residual Functions
# ---------------------------------------------------------

"""
    residuals_fcn2(x, P, chi, d, kappa, delta, n1, s0)

Original VGA residuals with the Hershey–Olsen 1D GMM entropy bound.
"""
function residuals_fcn2(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0)
    lJ1, lJ3, sigS, muW = x

    # 1. Concentration of perpendicular space
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    # 2. Update adapted eigenvalues
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    # 3. Training Signal
    lT1 = -(chi^2 / (kappa/P + lJ1_gen)^2 * delta) + chi * lJ1_gen^(-1) + (chi^2 * kappa / (P * chi)) * (lJ1_gen / (lJ1_gen + kappa/P))
    lT3 = -(chi^2 / (kappa/P + lJ3_gen)^2 * delta) + chi * lJ3_gen^(-1) + (chi^2 * kappa / (P * chi)) * (lJ3_gen / (lJ3_gen + kappa/P))
    A_pot = lT1 / (n1 * chi)
    A_pot3 = epsilon^2 * lT3 / (n1 * chi)
    # 4. Variational Free Energy
    function compute_free_energy(vars)
        s, m = vars[1], vars[2]
        s2 = s^2

        # Confinement (Ridge)
        # Expected value of (d/2)||w||^2 = (d/2)(m^2 + s^2)
        prior = (d / s0) * (1.0 / 2.0) * (m^2 + s2)

        # Potential Energy
        # integral of A_pot * (4/pi) * w_par^2 / (1 + 2||w||^2)
        energy = A_pot * compute_lambda1(m, s, T_floor) + A_pot3 * compute_lambda3(m, s, T_floor)
        neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-m^2 / s2))
        return prior + energy + neg_entropy_1d
    end

    grads = ForwardDiff.gradient(compute_free_energy, [sigS, muW])
    rj1 = lJ1 - lJ1_gen
    rj3 = lJ3 - lJ3_gen
    rsig = grads[1]
    rmu = grads[2]

    return [rj1, rj3, rsig, rmu]
end

"""
    residuals_fcn2_advanced(x, P, chi, d, kappa, delta, n1, s0; entropy_rule=:64)

Advanced VGA residuals using a Gauss–Hermite estimate of the exact
symmetric two-component GMM entropy.
"""
function residuals_fcn2_advanced(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    entropy_rule::Int=64)
    lJ1, lJ3, sigS, muW = x

    # 1. Concentration of perpendicular space
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    # 2. Update adapted eigenvalues
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    # 3. Training Signal
    lT1 = -(chi^2 / (kappa/P + lJ1_gen)^2 * delta) + chi * lJ1_gen^(-1) + (chi^2 * kappa / (P * chi)) * (lJ1_gen / (lJ1_gen + kappa/P))
    lT3 = -(chi^2 / (kappa/P + lJ3_gen)^2 * delta) + chi * lJ3_gen^(-1) + (chi^2 * kappa / (P * chi)) * (lJ3_gen / (lJ3_gen + kappa/P))
    A_pot = lT1 / (n1 * chi)
    A_pot3 = lT3 * epsilon^2 / (n1 * chi)
    # 4. Variational Free Energy
    function compute_free_energy(vars)
        s, m = vars[1], vars[2]

        # Confinement (Ridge)
        prior = (d / s0) * (1.0 / 2.0) * (m^2 + s^2)

        # Potential Energy
        energy = A_pot * compute_lambda1(m, s, T_floor)
       # energy += A_pot3 * compute_lambda3(m, s, T_floor)
        # Numerical entropy of the symmetric two-component GMM.
        neg_entropy_1d = -exact_symmetric_gmm_entropy_1d(m, s; rule=entropy_rule)

        return prior + energy + neg_entropy_1d
    end

    # Stationarity
    grads = ForwardDiff.gradient(compute_free_energy, [sigS, muW])

    rj1 = lJ1 - lJ1_gen
    rj3 = lJ3 - lJ3_gen
    rsig = grads[1]
    rmu = grads[2]

    return [rj1, rj3, rsig, rmu]
end

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------

function nlsolve_solver_fcn2(initial_guess; anneal=false, chi=1.0, d=1.0, s0=1.0, kappa=1.0, delta=1.0, n1=1.0, P=nothing, anneal_steps=50, tol=1e-8, advanced::Bool=false, entropy_rule::Int=64, epsilon=1.0)
    curr_P = (P === nothing) ? d^1.2 : P
    chi_path = anneal ? exp.(range(log(1e-5), log(chi), length=anneal_steps)) : fill(chi, anneal_steps)
    curr_x = copy(initial_guess)

    for c in chi_path
        f!(F, x) = begin
            xp = abs.(x)
            if advanced
                F .= residuals_fcn2_advanced(xp, curr_P, c, d, kappa, delta, n1, s0, epsilon; entropy_rule=entropy_rule)
            else
                F .= residuals_fcn2(xp, curr_P, c, d, kappa, delta, n1, s0, epsilon)
            end
        end
        sol = nlsolve(f!, curr_x, xtol=tol, ftol=tol, iterations=6000)
        curr_x = abs.(sol.zero)
    end
    return curr_x
end

function populate_solution_fcn2(sol_vec, params)
    if isnothing(sol_vec) || any(isnan.(sol_vec))
        return Solution2()
    end
    lJ1, lJ3, sigS, muW = sol_vec
    lWT = muW^2 + sigS^2

    ts = lWT + (params.d - 1.0) / params.d
    gy = (4/π) / (1 + 2.0 * ts)
    kt = params.κ / params.P

    return Solution2(
        lJ1=lJ1, lJ3=lJ3, lK1=gy*lJ1, lK3=gy*lJ3, lWT=lWT,
        sigS=sigS, muW=muW,
        learnability1=(lJ1)/(lJ1 + kt),
        learnability3=(lJ3)/(lJ3 + kt)
    )
end

function solve_FCN2_Erf(params, guess; anneal_steps=100, use_anneal=true, tol=1e-8, advanced::Bool=false, entropy_rule::Int=64)
    if length(guess) == 3
        guess = [guess[1], guess[2], sqrt(guess[3]*0.95), sqrt(guess[3]*0.05)]
    end
    sol = nlsolve_solver_fcn2(guess; anneal=use_anneal, chi=params.χ, d=params.d, s0=params.s0, kappa=params.κ, delta=params.δ, n1=params.n1, P=params.P, anneal_steps=anneal_steps, tol=tol, advanced=advanced, entropy_rule=entropy_rule, epsilon=params.ϵ)
    return populate_solution_fcn2(sol, params)
end

end