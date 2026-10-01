module FCS2_VGA

using SpecialFunctions
using ForwardDiff
using LinearAlgebra
using NLsolve
using Base: @kwdef

export ProblemParams2, Solution2, residuals_fcn2, residuals_fcn2_advanced,
    residuals_fcn2_offdiag, residuals_fcn2_matrix, residuals_regularized,
    residuals_fcn2_saddle, residuals_fcn2_laplace, residuals_fcn2_laplace_he57,
    solve_FCN2_Erf,
    exact_symmetric_gmm_entropy_1d, training_signal, training_amplitude,
    compute_lambda13, hermite_kernel_block, hermite_kernel_block_he57,
    training_amplitudes_offdiag,
    lambda1_point, hermite_point_features, hermite_point_features_he57,
    laplace_amplitude_matrix, laplace_amplitude_matrix_he57,
    matrix_learnabilities_he57,
    SIG_SADDLE_DEFAULT

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
    χ::Float32;                 # train χ (T_eff = T/χ); not the readout prior
    b::Float32 = 4/(3*π);
    δ::Float32 = 0.0
    s0::Float32 = 1.0           # w0: hidden prior variance scale, σ_w² = s0/d
    a0::Float32 = 1.0           # sa0: readout prior scale; A ∝ a0 · ℓ_T / (n1 χ)
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
    learnability5=NaN;          # f₅ = ⟨f, He5⟩; teacher has y₅=0
    learnability7=NaN;          # f₇ = ⟨f, He7⟩
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

"""
    lambda1_point(w, T)

Point-mass (σ→0) He1 eigenvalue density along the teacher axis:

    λ₁(w) = (4/π) w² / (T + 2 w²)

with perp floor `T = 1 + 2(d-1)s0/d`. Matches `compute_lambda1(w, 0, T)`.
"""
function lambda1_point(w, T)
    w2 = w * w
    return (2.0 / π) * w2 / (T + 2.0 * w2)
end

"""
    hermite_point_features(w, T) -> (c1, c3)

Single-neuron teacher-axis Hermite coefficients (perp frozen in `T`):

    c1 =  √(4/π)     w  / (T + 2w²)^{1/2}
    c3 = -√(8/(3π))  w³ / (T + 2w²)^{3/2}

so that `c1² = compute_lambda1(w,0,T)`, `c3² = compute_lambda3(w,0,T)`,
`c1 c3 = compute_lambda13(w,0,T)`.
"""
function hermite_point_features(w, T)
    D = T + 2.0 * w * w
    c1 = sqrt(4.0 / π) * w / sqrt(D)
    c3 = -sqrt(8.0 / (3.0 * π)) * w^3 / D^1.5
    return c1, c3
end

function compute_lambda3(μ, σ, T)
    res = 0.0
    for i in 1:11
        w_val = μ + σ * GH_NODES[i]
        res += GH_WEIGHTS[i] * (w_val^6 / (T + 2.0 * w_val^2)^3)
    end
    return (8.0 / (3.0 * π)) * res
end

"""
    compute_lambda13(μ, σ, T)

Off-diagonal Hermite matrix element
`⟨He1, K He3⟩ = E_w[c1(w) c3(w)]` for the erf feature map, with

    c1 =  2 w_∥ / (√π √(1+2|w|²))
    c3 = -4 w_∥³ / (√(6π) (1+2|w|²)^{3/2})

Under the usual `T ≈ 1 + 2|w_⊥|²` concentration this is
`-(8/(π√6)) E[w_∥⁴ / (T + 2 w_∥²)²]` (11-point GH, same as `compute_lambda3`).
Even in `w`, so one Gaussian component of the ±μ mixture suffices.
"""
function compute_lambda13(μ, σ, T)
    res = zero(μ + σ + T)
    for i in 1:11
        w_val = μ + σ * GH_NODES[i]
        D = T + 2.0 * w_val^2
        res += GH_WEIGHTS[i] * (w_val^4 / D^2)
    end
    return -(8.0 / (π * sqrt(6.0))) * res
end

"""
    hermite_kernel_block(μ, σ, T) -> Symmetric{2}

Teacher-axis Hermite block `K_H = [λ11 λ13; λ13 λ33]` under the current
variational Gaussian component `N(μ, σ²)` (and perp concentration `T`).
"""
function hermite_kernel_block(μ, σ, T)
    λ11 = compute_lambda1(μ, σ, T)
    λ33 = compute_lambda3(μ, σ, T)
    λ13 = compute_lambda13(μ, σ, T)
    return Symmetric([λ11 λ13; λ13 λ33])
end

"""
    hermite_point_features_he57(w, T) -> (c1, c3, c5, c7)

M=4 truncation of the erf generating function. He1/He3 match
`hermite_point_features`. Leakage modes:

    c5 =  12/√(30π)  w⁵ / D^{5/2}
    c7 = -20/√(35π)  w⁷ / D^{7/2}

with `D = T + 2w²`. Same as
`c_{2k+1} = (2/√π)(-1)^k γ^{2k+1} (2k)! / (k! √(2k+1)!)` for k=0..3,
`γ = w/√D`.
"""
function hermite_point_features_he57(w, T)
    D = T + 2.0 * w * w
    c1 = sqrt(4.0 / π) * w / sqrt(D)
    c3 = -sqrt(8.0 / (3.0 * π)) * w^3 / D^1.5
    c5 = (12.0 / sqrt(30.0 * π)) * w^5 / D^2.5
    c7 = -(20.0 / sqrt(35.0 * π)) * w^7 / D^3.5
    return c1, c3, c5, c7
end

"""
    hermite_kernel_block_he57(μ, σ, T) -> Symmetric{4}

Teacher-axis Gram `Q_{mn} = E[c_m(w) c_n(w)]` on (He1, He3, He5, He7)
under `N(μ, σ²)`. Products are even, so one well of the ±μ mixture
suffices. Same 11-point GH rule as `compute_lambda3`.
"""
function hermite_kernel_block_he57(μ, σ, T)
    Q = zeros(typeof(μ + σ + T), 4, 4)
    σs = max(σ, 1e-10)
    for i in 1:11
        w_val = μ + σs * GH_NODES[i]
        c1, c3, c5, c7 = hermite_point_features_he57(w_val, T)
        c = [c1, c3, c5, c7]
        wt = GH_WEIGHTS[i]
        for a in 1:4, b in 1:4
            Q[a, b] += wt * c[a] * c[b]
        end
    end
    return Symmetric(Q)
end

"""
    eigen_sym2(K)

Analytic eigen-decomposition of a 2×2 symmetric matrix
`K = [a c; c b]`. ForwardDiff-safe (no LAPACK `eigen`).
Returns `(evals, U)` with columns of `U` the eigenvectors.
"""
function eigen_sym2(K)
    a = K[1, 1]
    b = K[2, 2]
    c = K[1, 2]
    # Stable 2×2 symmetric eigen
    tr = a + b
    det_disc = (a - b)^2 + 4 * c^2
    root = sqrt(max(det_disc, zero(det_disc)))
    λ1 = (tr - root) / 2
    λ2 = (tr + root) / 2
    # Eigenvectors: for [a-λ, c; c, b-λ] 
    # use (c, λ-a) or (λ-b, c) depending on which is larger
    function evec(λ)
        v1 = c
        v2 = λ - a
        if abs(v1) + abs(v2) < abs(λ - b) + abs(c)
            v1 = λ - b
            v2 = c
        end
        nrm = sqrt(v1^2 + v2^2)
        # Degenerate / diagonal case
        if nrm < eps(typeof(nrm)) * (one(nrm) + abs(a) + abs(b))
            return (one(a), zero(a))
        end
        return (v1 / nrm, v2 / nrm)
    end
    u1x, u1y = evec(λ1)
    u2x, u2y = evec(λ2)
    # Ensure right-handed / orthogonal orientation
    if u1x * u2y - u1y * u2x < 0
        u2x, u2y = -u2x, -u2y
    end
    U = [u1x u2x; u1y u2y]
    return [λ1, λ2], U
end

"""
    training_amplitudes_offdiag(chi, n1, kappa, P, delta, epsilon, K_H;
                                a0=1.0, rel_eig_tol=1e-2)

Diagonalize the 2×2 Hermite block, rotate teacher `(1, ε)` into that
eigenbasis, and return channel amplitudes

    A_i = (δ'_i)² · a0 · training_signal(χ,κ,P,δ,d_i) / (n1 χ)

for eigenchannels with `d_i ≥ rel_eig_tol · max|d|`. Near-null channels
get `A_i = 0`: they lie outside the numerical range of `K_H`, so the
teacher component along them is invisible to the kernel and must not
generate a `χ²/m²` training force (which would otherwise diverge).

Also returns eigenvectors `U` and eigenvalues `d` (frozen for the
free-energy gradient, matching the diagonal-theory freeze of `A`).
"""
function training_amplitudes_offdiag(chi, n1, kappa, P, delta, epsilon, K_H;
    a0=1.0, rel_eig_tol=1e-2)
    evals, U = eigen_sym2(K_H)
    δ_rot = U' * [one(epsilon), epsilon]
    A = zeros(eltype(evals), length(evals))
    d_scale = maximum(abs, evals)
    thr = rel_eig_tol * max(d_scale, eps(typeof(d_scale)))
    for i in eachindex(evals)
        di = evals[i]
        abs(di) < thr && continue
        di_safe = max(di, eps(typeof(di)))
        A[i] = (δ_rot[i]^2) * training_amplitude(chi, n1, kappa, P, delta, di_safe; a0=a0)
    end
    return A, U, evals
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
# Training signal (ℓ_T) and free-energy amplitudes
# ---------------------------------------------------------

"""
    training_signal(chi, kappa, P, delta, λ)

Classical FCS2 VGA ℓ_T: discrepancy −χ² δ / m², Onsager +χ (κ/P) η,
and bare +χ/λ. Ridge `m = λ + κ/P`, `η = λ/m`.

`chi` is the train scale (unchanged by `a0` / `sa0`).
"""
function training_signal(chi, kappa, P, delta, λ)
    m = kappa / P + λ
    # Guard only against exact zero from underflow; do not floor away physics.
    m = max(m, eps(typeof(m)))
    η = λ / m
    ret = -(chi^2 * delta) / m^2 # + chi * (kappa / P) * η
    # Bare ±χ/λ term
    ret += chi / max(λ, eps(typeof(λ)))
    return ret
end

"""
A_pot = a0 · ℓ_T(χ/a0, κ/a0) / (n1 χ) for one spectral mode.

Integrating out a readout with prior variance `a0/(n1 χ)` rescales both
`χ → χ/a0` and the ridge `κ/P → κ/(a0 P)` inside ℓ_T.
"""
training_amplitude(chi, n1, kappa, P, delta, λ; a0=1.0) =
    a0 * training_signal(chi / a0, kappa / a0, P, delta, λ) / (n1 * chi)

"""
    laplace_amplitude_matrix(chi, n1, kappa, P, delta, epsilon, Q; a0=1.0)

2×2 generalization of `training_amplitude` on the teacher-axis Hermite block
`Q = [λ11 λ13; λ13 λ33]`, with `χ' = χ/a0`, `ρ' = κ/(a0 P)`, `y = (1, ε)`:

    v = (Q + ρ' I)⁻¹ y
    G = a0/(n1 χ) · [ −χ'² δ v vᵀ + χ' Q⁻¹ ]

For `λ13 = 0` the (1,1) entry is `training_amplitude(λ11)`. The cubic
component `v₃ = −λ13 v₁/(λ33+ρ')` survives at `ε = 0`.
"""
function laplace_amplitude_matrix(chi, n1, kappa, P, delta, epsilon, Q; a0=1.0)
    chi_p = chi / a0
    ρp = kappa / (a0 * P)
    Qm = Matrix(Q)
    m = size(Qm, 1)
    Tq = eltype(Qm)
    y = zeros(Tq, m)
    y[1] = one(Tq)
    m >= 2 && (y[2] = oftype(y[1], epsilon))
    qscale = zero(Tq)
    for i in 1:m
        qscale += abs(Qm[i, i])
    end
    Qreg = Qm + (1e-10 * (one(Tq) + qscale)) * I(m)
    v = (Qreg + ρp * I(m)) \ y
    G = -(chi_p^2 * delta) * (v * v') + chi_p * inv(Qreg)
    return (a0 / (n1 * chi)) * G
end

"""
    laplace_amplitude_matrix_he57(chi, n1, kappa, P, delta, epsilon, Q; a0=1.0)

4×4 cavity kernel on (He1, He3, He5, He7). `Q` must be 4×4
(`hermite_kernel_block_he57`). Replica field
`τ ~ N(i χ' R⁻¹ y, χ' R⁻¹)` with `R = Q + ρ' I₄` and
`y = (1, ε, 0, 0)`:

    v = R⁻¹ y ∈ ℝ⁴
    G = a0/(n1 χ) · [ −χ'² δ v vᵀ + χ' R⁻¹ ] ∈ ℝ⁴ˣ⁴

Same `c ∈ ℝ⁴` as in `V`. Not the 2×2 `Q⁻¹` Onsager.
"""
function laplace_amplitude_matrix_he57(chi, n1, kappa, P, delta, epsilon, Q; a0=1.0)
    Qm = Matrix(Q)
    size(Qm) == (4, 4) || throw(DimensionMismatch(
        "laplace_amplitude_matrix_he57 expects a 4×4 Q, got $(size(Qm))"))
    chi_p = chi / a0
    ρp = kappa / (a0 * P)
    Tq = eltype(Qm)
    y = Tq[one(Tq), oftype(one(Tq), epsilon), zero(Tq), zero(Tq)]
    ρ_safe = max(ρp, eps(typeof(ρp + chi_p)))
    R = Qm + ρ_safe * I(4)
    Rinv = inv(R)
    v = Rinv * y
    G = -(chi_p^2 * delta) * (v * v') + chi_p * Rinv
    return (a0 / (n1 * chi)) * G
end

# ---------------------------------------------------------
# Core Residual Functions
# ---------------------------------------------------------

"""
    residuals_fcn2(x, P, chi, d, kappa, delta, n1, s0)

Original VGA residuals with the Hershey–Olsen 1D GMM entropy bound.
"""
function residuals_fcn2(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0; a0=1.0)
    lJ1, lJ3, sigS, muW = x

    # 1. Concentration of perpendicular space
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    # 2. Update adapted eigenvalues
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    # 3. Training amplitudes (classical ℓ_T; cubic back-reacts via A_pot3)
    A_pot = training_amplitude(chi, n1, kappa, P, delta, lJ1_gen; a0=a0)
    A_pot3 = epsilon^2 * training_amplitude(chi, n1, kappa, P, delta, lJ3_gen; a0=a0)
    



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
    residuals_fcn2_matrix(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0; a0=1.0)

HO VGA residuals with the teacher training signal lifted to the 2×2 Hermite
block `Q = hermite_kernel_block(μ, σ, T)`.

Corrected kernel (discrepancy = square of `Ty`):

    T = (Q + ρ I)⁻¹,  ρ = κ/P,  y = (1, ε),  v = T y
    ℓ = −χ² δ (yᵀ T Q T y) + χ Tr(Q⁻¹ Q)

`T` and `v` are **unfrozen**: rebuilt from the trial `(m,s)` inside the
free energy (same `Q` as in `yᵀ T Q T y`). When `Q` is that trial block,
`Tr(Q⁻¹ Q) = 2` is constant and does not affect stationarity. No classical
`½` on the discrepancy.

Matching residuals still use Hermite diagonals `λ11, λ33`.
"""
function residuals_fcn2_matrix(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0; a0=1.0)
    lJ1, lJ3, sigS, muW = x

    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d
    ρ = kappa / P
    y = [one(epsilon), epsilon]

    Q0 = Matrix(hermite_kernel_block(muW, sigS, T_floor))
    λ11 = Q0[1, 1]
    λ33 = Q0[2, 2]

    function compute_free_energy(vars)
        s, m = vars[1], vars[2]
        s2 = s^2
        prior = (d / s0) * (1.0 / 2.0) * (m^2 + s2)

        Qtr = Matrix(hermite_kernel_block(m, s, T_floor))
        qscale = abs(Qtr[1, 1]) + abs(Qtr[2, 2])
        qfloor = eps(typeof(qscale)) * (one(qscale) + qscale)
        Qreg = Symmetric(Qtr + qfloor * I(2))
        Tmat = inv(Symmetric(Matrix(Qreg) + ρ * I(2)))
        v = Tmat * y
        # −χ² δ yᵀ T Q T y  +  χ Tr(Q⁻¹ Q)   (T, v unfrozen with trial Q)
        disc = -(chi^2 * delta) * dot(v, Qtr * v)
        bare = chi * tr(inv(Qreg) * Qtr)   # → 2χ for full-rank Q (const. in grads)
        ell = disc + bare
        energy = a0 * ell / (n1 * chi)

        neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-m^2 / max(s2, 1e-25)))
        return prior + energy + neg_entropy_1d
    end

    grads = ForwardDiff.gradient(compute_free_energy, [sigS, muW])
    return [lJ1 - λ11, lJ3 - λ33, grads[1], grads[2]]
end

"""
    residuals_fcn2_offdiag(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
                           freeze_U=true)

Hershey–Olsen VGA residuals with **off-diagonal He1–He3 accounting** in the
training discrepancy.

At each residual evaluation:

1. Build `K_H = [λ11 λ13; λ13 λ33]` from `(μ,σ)` (and perp floor `T`).
2. Diagonalize `K_H = U diag(d) Uᵀ` and rotate teacher `(1, ε) → δ' = Uᵀ δ`.
3. Form channel amplitudes `A_i ∝ (δ'_i)² · ℓ_T(d_i)` (same `training_signal`
   as the diagonal theory, but in the eigenbasis).
4. Free energy + HO entropy:
   - `freeze_U=true` (default): freeze `U,A` from the outer `(μ,σ)`, vary only
     projected eigenvalues `uᵢᵀ K_H(m,s) uᵢ`.
   - `freeze_U=false`: rebuild `K_H(m,s)`, re-diagonalize, and recompute `A`
     inside the free energy (ForwardDiff-safe analytic 2×2 eigen).

Hermite-diagonal Rayleigh quotients `λ11, λ33` remain the matching targets
for `(lJ1, lJ3)` so the solution vector stays comparable to `residuals_fcn2`.
"""
function residuals_fcn2_offdiag(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    freeze_U::Bool=true, a0=1.0)
    lJ1, lJ3, sigS, muW = x

    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d
    K_H = hermite_kernel_block(muW, sigS, T_floor)
    λ11 = K_H[1, 1]
    λ33 = K_H[2, 2]
    λ31 = K_H[2, 1]
    if freeze_U
        A_chan, U, _evals = training_amplitudes_offdiag(
            chi, n1, kappa, P, delta, epsilon, K_H; a0=a0,
        )
        u1 = [U[1, 1], U[2, 1]]
        u2 = [U[1, 2], U[2, 2]]
        A1, A2 = A_chan[1], A_chan[2]
        function compute_free_energy_frozen(vars)
            s, m = vars[1], vars[2]
            s2 = s^2
            prior = (d / s0) * (1.0 / 2.0) * (m^2 + s2)
            K_trial = hermite_kernel_block(m, s, T_floor)
            e1 = dot(u1, K_trial * u1)
            e2 = dot(u2, K_trial * u2)
            energy = A1 * e1 + A2 * e2
            neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-m^2 / s2))
            return prior + energy + neg_entropy_1d
        end
        grads = ForwardDiff.gradient(compute_free_energy_frozen, [sigS, muW])
    else
        function compute_free_energy_unfrozen(vars)
            s, m = vars[1], vars[2]
            s2 = s^2
            prior = (d / s0) * (1.0 / 2.0) * (m^2 + s2)
            K_trial = hermite_kernel_block(m, s, T_floor)
            A_chan, _U, evals = training_amplitudes_offdiag(
                chi, n1, kappa, P, delta, epsilon, K_trial; a0=a0,
            )
            energy = A_chan[1] * evals[1] + A_chan[2] * evals[2]
            neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-m^2 / s2))
            return prior + energy + neg_entropy_1d
        end
        grads = ForwardDiff.gradient(compute_free_energy_unfrozen, [sigS, muW])
    end

    return [lJ1 - λ11, lJ3 - λ33, grads[1], grads[2]]
end

"""
    residuals_fcn2_advanced(x, P, chi, d, kappa, delta, n1, s0; entropy_rule=:64)

Advanced VGA residuals using a Gauss–Hermite estimate of the exact
symmetric two-component GMM entropy, classical ℓ_T, and cubic back-reaction.
"""
function residuals_fcn2_advanced(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    entropy_rule::Int=64, a0=1.0)
    lJ1, lJ3, sigS, muW = x

    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    A_pot = training_amplitude(chi, n1, kappa, P, delta, lJ1_gen; a0=a0)
    A_pot3 = epsilon^2 * training_amplitude(chi, n1, kappa, P, delta, lJ3_gen; a0=a0)

    function compute_free_energy(vars)
        s, m = vars[1], vars[2]
        prior = (d / s0) * (1.0 / 2.0) * (m^2 + s^2)
        energy = A_pot * compute_lambda1(m, s, T_floor)
        energy += A_pot3 * compute_lambda3(m, s, T_floor)
        neg_entropy_1d = -exact_symmetric_gmm_entropy_1d(m, s; rule=entropy_rule)
        # Scale like prior/energy (∝ d) so entropy stays in the residual balance.
        return prior + energy + d * neg_entropy_1d
    end

    grads = ForwardDiff.gradient(compute_free_energy, [sigS, muW])
    return [lJ1 - lJ1_gen, lJ3 - lJ3_gen, grads[1], grads[2]]
end

"""
    residuals_regularized(x, P, chi, d, kappa, delta, n1, s0; advanced=false, entropy_rule=64)

Same saddle-point equations as `residuals_fcn2` / `residuals_fcn2_advanced`,
but stationarity residuals are preconditioned by `(m1² + m3²)` where
`mk = λk_gen + κ/P`. Zeros are unchanged; Newton sees O(1) gradients when
`A ∼ 1/m²` would otherwise dominate.
"""
function residuals_regularized(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    advanced::Bool=false, entropy_rule::Int=64, a0=1.0)
    lJ1, lJ3, sigS, muW = x

    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d
    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    A_pot = training_amplitude(chi, n1, kappa, P, delta, lJ1_gen; a0=a0)
    A_pot3 = epsilon^2 * training_amplitude(chi, n1, kappa, P, delta, lJ3_gen; a0=a0)

    function compute_free_energy(vars)
        s, m = vars[1], vars[2]
        prior = (d / s0) * (1.0 / 2.0) * (m^2 + s^2)
        energy = A_pot * compute_lambda1(m, s, T_floor) +
                 A_pot3 * compute_lambda3(m, s, T_floor)
        if advanced
            neg_entropy_1d = -exact_symmetric_gmm_entropy_1d(m, s; rule=entropy_rule)
        else
            s2 = s^2
            neg_entropy_1d = -0.5 * log(s2 + 1e-25) + log(1.0 + exp(-m^2 / s2))
        end
        return prior + energy + neg_entropy_1d
    end

    grads = ForwardDiff.gradient(compute_free_energy, [sigS, muW])
    rj1 = lJ1 - lJ1_gen
    rj3 = lJ3 - lJ3_gen

    # Precondition stationarity: same zeros as (∂_s F, ∂_μ F) = 0
    m1 = max(kappa / P + lJ1_gen, eps(typeof(lJ1_gen)))
    m3 = max(kappa / P + lJ3_gen, eps(typeof(lJ3_gen)))
    scale = m1^2 + m3^2
    return [rj1, rj3, scale * grads[1], scale * grads[2]]
end

# ---------------------------------------------------------
# Laplace saddle: mode of V(w)=prior + A₁ λ₁(w), σ from curvature
# ---------------------------------------------------------

const SIG_SADDLE_DEFAULT = 1e-10

"""
    residuals_fcn2_laplace(x, ...; mean_only=false, sig_saddle=1e-10)

Gaussian Laplace approx around the linear-channel wells (no VGA entropy).
Cubic channel is ignored in the potential (A₃≡0); `lJ3` is still matched
to `compute_lambda3(μ,σ,T)` for reporting.

Microscopic action (teacher axis, perp frozen in `T`):

    V(w) = (d/(2 s0)) w² + A₁ · λ₁(w),
    λ₁(w) = (4/π) w² / (T + 2 w²),
    T = 1 + 2(d-1)s0/d,

with `A₁ = a0 · ℓ_T(λ₁)/(n1 χ)` frozen from the current λ₁.

Default state `x = [lJ1, lJ3, σ, μ]` with residuals

    lJ1 − λ₁(μ,σ),  lJ3 − λ₃(μ,σ),  V'(μ),  σ² V''(μ) − 1

`mean_only=true`: freeze `σ = sig_saddle` (≈0), drop the curvature residual,
state `x = [lJ1, lJ3, μ]`, residuals `(lJ1-λ1, lJ3-λ3, V'(μ))` — mode
locations of `V` only.

`matrix=true`: replace `A₁ λ₁(w)` by the He1–He3 quadratic form

    V(w) = (d/(2 s0)) w² + ½ c(w)ᵀ G c(w),   c = hermite_point_features(w, T),

with `G = laplace_amplitude_matrix(..., Q)` frozen from
`Q = hermite_kernel_block(μ, σ, T)`. The `½` matches `lambda1_point`.
"""
function residuals_fcn2_laplace(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    a0=1.0, mean_only::Bool=false, sig_saddle::Real=SIG_SADDLE_DEFAULT,
    matrix::Bool=false)
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    if mean_only
        lJ1, lJ3, muW = x
        sigS = oftype(muW + lJ1, sig_saddle)
    else
        lJ1, lJ3, sigS, muW = x
    end

    lJ1_gen = compute_lambda1(muW, sigS, T_floor)
    lJ3_gen = compute_lambda3(muW, sigS, T_floor)

    if matrix
        Q = hermite_kernel_block(muW, sigS, T_floor)
        G = laplace_amplitude_matrix(chi, n1, kappa, P, delta, epsilon, Q; a0=a0)
        V = w -> begin
            c1, c3 = hermite_point_features(w, T_floor)
            quad = G[1, 1] * c1^2 + 2.0 * G[1, 2] * c1 * c3 + G[2, 2] * c3^2
            (d / s0) * 0.5 * w^2 + 0.5 * quad
        end
    else
        A_pot = training_amplitude(chi, n1, kappa, P, delta, lJ1_gen; a0=a0)
        V = w -> (d / s0) * 0.5 * w^2 + A_pot * lambda1_point(w, T_floor)
    end

    dV = ForwardDiff.derivative(V, muW)
    rj1 = lJ1 - lJ1_gen
    rj3 = lJ3 - lJ3_gen
    rmu = dV

    if mean_only
        return [rj1, rj3, rmu]
    end

    d2V = ForwardDiff.derivative(w -> ForwardDiff.derivative(V, w), muW)
    # σ = 1/√V''  ⇒  σ² V'' − 1 = 0
    rsig = sigS^2 * d2V - 1.0
    return [rj1, rj3, rsig, rmu]
end

"""
    residuals_fcn2_laplace_he57(x, ...)

M=4 Laplace saddle (He1, He3, He5, He7). Gram and potential use the
same features `c = (c1, c3, c5, c7)`, teacher `y = (1, ε, 0, 0)`:

    Q_{mn} = E[c_m c_n] ∈ ℝ⁴ˣ⁴
    y = (1, ε, 0, 0) ∈ ℝ⁴
    G ∈ ℝ⁴ˣ⁴  from laplace_amplitude_matrix_he57 (replica R⁻¹)
    V(w) = (d/(2 s0)) w² + c(w)ᵀ G c(w)

`Q` and `c` are the same generating-function features, so the energy
is `cᵀ G c` (not the 2×2 `½ cᵀ G c`, which exists only to turn
`c1²` into `lambda1_point`). Frozen-v cavity: `E[cᵀ G c] = tr(G Q)`
equals `a0/(n1 χ) [−χ'² δ vᵀ Q v + χ' tr(R⁻¹ Q)]`.

State is still `x = [lJ1, lJ3, σ, μ]`; dummy residuals match `lJ1, lJ3`
to `Q₁₁, Q₃₃`. `mean_only=true` drops the curvature residual as in
the 2×2 Laplace. Does not change `residuals_fcn2_laplace`.
"""
function residuals_fcn2_laplace_he57(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    a0=1.0, mean_only::Bool=false, sig_saddle::Real=SIG_SADDLE_DEFAULT)
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    if mean_only
        lJ1, lJ3, muW = x
        sigS = oftype(muW + lJ1, sig_saddle)
    else
        lJ1, lJ3, sigS, muW = x
    end

    Q = hermite_kernel_block_he57(muW, sigS, T_floor)
    G = laplace_amplitude_matrix_he57(chi, n1, kappa, P, delta, epsilon, Q; a0=a0)
    V = w -> begin
        c1, c3, c5, c7 = hermite_point_features_he57(w, T_floor)
        c = [c1, c3, c5, c7]
        (d / s0) * 0.5 * w^2 + dot(c, G * c)
    end

    dV = ForwardDiff.derivative(V, muW)
    rj1 = lJ1 - Q[1, 1]
    rj3 = lJ3 - Q[2, 2]
    rmu = dV

    if mean_only
        return [rj1, rj3, rmu]
    end

    d2V = ForwardDiff.derivative(w -> ForwardDiff.derivative(V, w), muW)
    rsig = sigS^2 * d2V - 1.0
    return [rj1, rj3, rsig, rmu]
end

# ---------------------------------------------------------
# Saddle-point (δ-well) residuals: σ → 0, stationarity in μ only
# ---------------------------------------------------------

"""
    residuals_fcn2_saddle(x, ...; sig_saddle=1e-10, offdiag=false, freeze_U=true)

Point-mass (±μ) saddle: freeze `σ = sig_saddle`, drop variational entropy
(diverges as σ→0), and solve

    x = [lJ1, lJ3, μ]

with residuals `(lJ1-λ1, lJ3-λ3, ∂_μ F)` where
`F = prior + energy` only.  Off-diag channel amplitudes match
`residuals_fcn2_offdiag` when `offdiag=true`.
"""
function residuals_fcn2_saddle(x, P, chi, d, kappa, delta, n1, s0, epsilon=1.0;
    sig_saddle::Real=SIG_SADDLE_DEFAULT, offdiag::Bool=false, freeze_U::Bool=true, a0=1.0)
    lJ1, lJ3, muW = x
    sigS = oftype(muW + lJ1, sig_saddle)
    T_floor = 1.0 + 2.0 * (d - 1.0) * s0 / d

    if offdiag
        K_H = hermite_kernel_block(muW, sigS, T_floor)
        λ11 = K_H[1, 1]
        λ33 = K_H[2, 2]
        A_chan, U, _evals = training_amplitudes_offdiag(
            chi, n1, kappa, P, delta, epsilon, K_H; a0=a0,
        )
        u1 = [U[1, 1], U[2, 1]]
        u2 = [U[1, 2], U[2, 2]]
        A1, A2 = A_chan[1], A_chan[2]

        function F_mu_offdiag(m)
            prior = (d / s0) * (1.0 / 2.0) * (m^2 + sigS^2)
            K_trial = hermite_kernel_block(m, sigS, T_floor)
            if freeze_U
                e1 = dot(u1, K_trial * u1)
                e2 = dot(u2, K_trial * u2)
                energy = A1 * e1 + A2 * e2
            else
                A_t, _U, ev = training_amplitudes_offdiag(
                    chi, n1, kappa, P, delta, epsilon, K_trial; a0=a0,
                )
                energy = A_t[1] * ev[1] + A_t[2] * ev[2]
            end
            return prior + energy
        end
        rmu = ForwardDiff.derivative(F_mu_offdiag, muW)
        return [lJ1 - λ11, lJ3 - λ33, rmu]
    else
        lJ1_gen = compute_lambda1(muW, sigS, T_floor)
        lJ3_gen = compute_lambda3(muW, sigS, T_floor)
        A_pot = training_amplitude(chi, n1, kappa, P, delta, lJ1_gen; a0=a0)
        A_pot3 = epsilon^2 * training_amplitude(chi, n1, kappa, P, delta, lJ3_gen; a0=a0)

        function F_mu(m)
            prior = (d / s0) * (1.0 / 2.0) * (m^2 + sigS^2)
            energy = A_pot * compute_lambda1(m, sigS, T_floor) +
                     A_pot3 * compute_lambda3(m, sigS, T_floor)
            return prior + energy
        end
        rmu = ForwardDiff.derivative(F_mu, muW)
        return [lJ1 - lJ1_gen, lJ3 - lJ3_gen, rmu]
    end
end

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------

function nlsolve_solver_fcn2(initial_guess; anneal=false, chi=1.0, a0=1.0, d=1.0, s0=1.0, kappa=1.0, delta=1.0, n1=1.0, P=nothing, anneal_steps=50, tol=1e-8, advanced::Bool=false, regularized::Bool=false, offdiag::Bool=false, matrix::Bool=false, freeze_U::Bool=true, entropy_rule::Int=64, epsilon=1.0, saddle::Bool=false, laplace::Bool=false, laplace_mean::Bool=false, sig_saddle::Real=SIG_SADDLE_DEFAULT, he57::Bool=false)
    curr_P = (P === nothing) ? d^1.2 : P
    chi_path = anneal ? exp.(range(log(1e-5), log(chi), length=anneal_steps)) : fill(chi, anneal_steps)
    curr_x = copy(initial_guess)

    for c in chi_path
        f!(F, x) = begin
            xp = abs.(x)
            if saddle
                # 3D: [lJ1, lJ3, μ] with σ frozen
                F .= residuals_fcn2_saddle(
                    xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                    sig_saddle=sig_saddle, offdiag=offdiag, freeze_U=freeze_U, a0=a0,
                )
            elseif laplace_mean
                # 3D: [lJ1, lJ3, μ], σ→0, V' only (linear channel)
                if he57
                    F .= residuals_fcn2_laplace_he57(
                        xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                        a0=a0, mean_only=true, sig_saddle=sig_saddle,
                    )
                else
                    F .= residuals_fcn2_laplace(
                        xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                        a0=a0, mean_only=true, sig_saddle=sig_saddle, matrix=matrix,
                    )
                end
            elseif laplace
                # 4D: [lJ1, lJ3, σ, μ] from V' = 0 and σ² V'' = 1
                if he57
                    F .= residuals_fcn2_laplace_he57(
                        xp, curr_P, c, d, kappa, delta, n1, s0, epsilon; a0=a0,
                    )
                else
                    F .= residuals_fcn2_laplace(
                        xp, curr_P, c, d, kappa, delta, n1, s0, epsilon; a0=a0,
                        matrix=matrix,
                    )
                end
            elseif regularized
                F .= residuals_regularized(
                    xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                    advanced=advanced, entropy_rule=entropy_rule, a0=a0,
                )
            elseif matrix
                F .= residuals_fcn2_matrix(
                    xp, curr_P, c, d, kappa, delta, n1, s0, epsilon; a0=a0,
                )
            elseif offdiag
                F .= residuals_fcn2_offdiag(
                    xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                    freeze_U=freeze_U, a0=a0,
                )
            elseif advanced
                F .= residuals_fcn2_advanced(
                    xp, curr_P, c, d, kappa, delta, n1, s0, epsilon;
                    entropy_rule=entropy_rule, a0=a0,
                )
            else
                F .= residuals_fcn2(xp, curr_P, c, d, kappa, delta, n1, s0, epsilon; a0=a0)
            end
        end
        sol = nlsolve(f!, curr_x, xtol=tol, ftol=tol, iterations=6000)
        curr_x = abs.(sol.zero)
    end
    return curr_x
end

"""
    matrix_learnabilities(μ, σ, params) -> (L1, L3)

Kernel-ridge prediction on the teacher-axis Hermite block,
`f = Q (Q + ρ' I)⁻¹ y` with `Q = hermite_kernel_block(μ, σ, T)`,
`ρ' = κ/(a0 P)`, `y = (1, ε)`. Returns `(f₁, f₃/ε)`. Because `λ13 ≠ 0`
locks He3 output to He1, `f₃/ε` can be negative when `ε` is small.
"""
function matrix_learnabilities(μ, σ, params)
    T = 1.0 + 2.0 * (params.d - 1.0) * params.s0 / params.d
    Q = Matrix(hermite_kernel_block(μ, σ, T))
    ρp = params.κ / (params.a0 * params.P)
    y = [1.0, Float64(params.ϵ)]
    f = Q * ((Q + ρp * I(2)) \ y)
    L3 = params.ϵ == 0 ? f[2] : f[2] / params.ϵ
    return f[1], L3
end

"""
    matrix_learnabilities_he57(μ, σ, params) -> (L1, L3, f5, f7)

Kernel ridge on the 4×4 He1–He3–He5–He7 block,
`f = Q (Q + ρ' I)⁻¹ y` with `y = (1, ε, 0, 0)`. `L3 = f₃/ε`;
`f5, f7` are raw leakages (teacher has no He5/He7).
"""
function matrix_learnabilities_he57(μ, σ, params)
    T = 1.0 + 2.0 * (params.d - 1.0) * params.s0 / params.d
    Q = Matrix(hermite_kernel_block_he57(μ, σ, T))
    ρp = params.κ / (params.a0 * params.P)
    y = [1.0, Float64(params.ϵ), 0.0, 0.0]
    f = Q * ((Q + ρp * I(4)) \ y)
    L3 = params.ϵ == 0 ? f[2] : f[2] / params.ϵ
    return f[1], L3, f[3], f[4]
end

function populate_solution_fcn2(sol_vec, params; sig_saddle::Real=SIG_SADDLE_DEFAULT,
    matrix::Bool=false, he57::Bool=false)
    if isnothing(sol_vec) || any(isnan.(sol_vec))
        return Solution2()
    end
    if length(sol_vec) == 3
        lJ1, lJ3, muW = sol_vec
        sigS = float(sig_saddle)
    else
        lJ1, lJ3, sigS, muW = sol_vec
    end
    lWT = muW^2 + sigS^2

    ts = lWT + (params.d - 1.0) / params.d
    gy = (4/π) / (1 + 2.0 * ts)
    kt = params.κ / params.P
    L5, L7 = NaN, NaN
    if he57
        L1, L3, L5, L7 = matrix_learnabilities_he57(abs(muW), abs(sigS), params)
    elseif matrix
        L1, L3 = matrix_learnabilities(abs(muW), abs(sigS), params)
    else
        L1, L3 = lJ1 / (lJ1 + kt), lJ3 / (lJ3 + kt)
    end

    return Solution2(
        lJ1=lJ1, lJ3=lJ3, lK1=gy*lJ1, lK3=gy*lJ3, lWT=lWT,
        sigS=sigS, muW=muW,
        learnability1=L1,
        learnability3=L3,
        learnability5=L5,
        learnability7=L7,
    )
end

function solve_FCN2_Erf(params, guess; anneal_steps=100, use_anneal=true, tol=1e-8,
    advanced::Bool=false, regularized::Bool=false, offdiag::Bool=false,
    matrix::Bool=false, freeze_U::Bool=true, entropy_rule::Int=64,
    saddle::Bool=false, laplace::Bool=false, laplace_mean::Bool=false,
    sig_saddle::Real=SIG_SADDLE_DEFAULT, he57::Bool=false)
    if saddle || laplace_mean
        # Expect [lJ1, lJ3, μ].  Old 4D / lWT-style guesses are remapped.
        if length(guess) == 4
            guess = [guess[1], guess[2], max(abs(guess[4]), 0.1)]
        elseif length(guess) == 3
            # Ambiguous: classical [lJ1,lJ3,lWT] vs saddle [lJ1,lJ3,μ].
            # If third entry looks like a variance ≪ 1 and ≪ sqrt scale, treat as lWT→μ.
            g3 = abs(guess[3])
            if g3 < 0.05
                guess = [guess[1], guess[2], sqrt(max(g3, 1e-6))]
            end
        end
    elseif length(guess) == 3
        guess = [guess[1], guess[2], sqrt(guess[3]*0.95), sqrt(guess[3]*0.05)]
    end
    # χ stays as train χ; sa0 (a0) multiplies A = a0 · ℓ_T / (n1 χ).
    sol = nlsolve_solver_fcn2(
        guess;
        anneal=use_anneal, chi=params.χ, a0=params.a0, d=params.d, s0=params.s0,
        kappa=params.κ, delta=params.δ, n1=params.n1, P=params.P,
        anneal_steps=anneal_steps, tol=tol, advanced=advanced,
        regularized=regularized, offdiag=offdiag, matrix=matrix,
        freeze_U=freeze_U, entropy_rule=entropy_rule, epsilon=params.ϵ,
        saddle=saddle, laplace=laplace, laplace_mean=laplace_mean, sig_saddle=sig_saddle,
        he57=he57,
    )
    return populate_solution_fcn2(sol, params; sig_saddle=sig_saddle, matrix=matrix || he57,
        he57=he57)
end

end