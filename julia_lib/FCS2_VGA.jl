module FCS2_VGA

using SpecialFunctions
using ForwardDiff
using LinearAlgebra
using NLsolve
using Base: @kwdef

export ProblemParams2, Solution2, residuals_fcn2, solve_FCN2_Erf

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
# High-Precision Spectral Integrals
# ---------------------------------------------------------
const GH_NODES = [-4.45948490915965, -3.40143369785489, -2.43401875620939, -1.51593110691244, -0.60576387917106, 0.0,
    0.60576387917106, 1.51593110691244, 2.43401875620939, 3.40143369785489, 4.45948490915965]
const GH_WEIGHTS = [2.658951684e-5, 0.0010205865, 0.012573199, 0.067468316, 0.182603415, 0.472560987,
    0.182603415, 0.067468316, 0.012573199, 0.0010205865, 2.658951684e-5]

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
# Core Residual Function
# ---------------------------------------------------------

function residuals_fcn2(x, P, chi, d, kappa, delta, n1)
    lJ1, lJ3, sigS, muW = x

    # 1. Concentration of perpendicular space
    T_floor = 1.0 + 2.0 * (d - 1.0) / d

    # 2. Update adapted eigenvalues
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    # 3. Training Signal
    lT1 = -(chi^2 / (kappa/P + lJ1)^2 * delta) + chi * lJ1^(-1) + (chi^2 * kappa / (P * chi)) * (lJ1 / (lJ1 + kappa/P))
    A_pot = lT1 / (n1 * chi)

    # 4. Variational Free Energy
    function compute_free_energy(vars)
        s, m = vars[1], vars[2]
        s2 = s^2

        # Confinement (Ridge)
        # Expected value of (d/2)||w||^2 = (d/2)(m^2 + s^2)
        prior = (d / 2.0) * (m^2 +  s2 + (d-1) / d)

        # Potential Energy 
        # integral of A_pot * (4/pi) * w_par^2 / (1 + 2||w||^2)
        energy = A_pot * compute_lambda1(m, s, T_floor) * π / 4.0

        # 1D Jensen Negative Entropy Bound (Exact as requested)
        # -H = sum xi_i ln(sum xi_j z_ij)
        # z_self = 1/sqrt(4*pi*s^2), z_cross = z_self * exp(-m^2/s^2)
        # For xi = [0.5, 0.5]: -H = ln(0.5) + ln(z_self) + ln(1 + exp(-m^2/s^2))
        # DEPRECATED
        # log_z_self = -0.5 * log(4.0 * π * s2 + 1e-25)
        # neg_entropy_1d = log(0.5) + log_z_self + log(1.0 + exp(-m^2 / s2))

        # Hershey Olsen correction (2007) for 1D GMM entropy bound
        # H = 0.5 * log(2*pi*e*s^2)
        # H <= 0.5 * log(2*pi*e*s^2)
        neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-2.0 * m^2 / s2))


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

function nlsolve_solver_fcn2(initial_guess; anneal=false, chi=1.0, d=1.0, kappa=1.0, delta=1.0, n1=1.0, P=nothing, anneal_steps=50, tol=1e-8)
    curr_P = (P === nothing) ? d^1.2 : P
    chi_path = anneal ? exp.(range(log(1e-5), log(chi), length=anneal_steps)) : fill(chi, anneal_steps)
    curr_x = copy(initial_guess)

    for c in chi_path
        f!(F, x) = (F .= residuals_fcn2(abs.(x), curr_P, c, d, kappa, delta, n1))
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

function solve_FCN2_Erf(params, guess; anneal_steps=100, use_anneal=true, tol=1e-8)
    if length(guess) == 3
        guess = [guess[1], guess[2], sqrt(guess[3]*0.95), sqrt(guess[3]*0.05)]
    end
    sol = nlsolve_solver_fcn2(guess; anneal=use_anneal, chi=params.χ, d=params.d, kappa=params.κ, delta=params.δ, n1=params.n1, P=params.P, anneal_steps=anneal_steps, tol=tol)
    return populate_solution_fcn2(sol, params)
end

end