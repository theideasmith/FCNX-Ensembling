#!/usr/bin/env julia
# Fast solver for (E1)–(E8) of notes/fcn3_matrix_eos.tex.
#
# Same method as the FCN2 true-action solver
# (milestones/fcn2_collapsed_hmc/readin_action_theory_q.py, and the matrix
# action used under milestones/fcn2_erf_hidden_kernel):
#
#   * Features are functions of the neuron coordinates only. They are built on a
#     fixed quadrature and are not rebuilt inside an inner step.
#   * The exact, non-Gaussian action enters only as Boltzmann weights.
#   * The readout (E4) is the gradient of the convex potential F(t). Newton on t
#     uses the analytic Hessian; each trial t is a reweighting of one feature table.
#   * The read-in state (J, q̄_h, M) is a low-dimensional unknown. Newton–Broyden
#     drives encode(state out) − encode(state in) to zero.
#
# At χ = N₂ the fixed point is solved there. χ is held fixed, and both layers are
# one Boltzmann integral on the full grid. A second minimum is part of that
# integrand. It is not an unknown and it is not given its own weight.
#
#   julia -t 16 milestones/fcn3_matrix_theory/fcn3_fast.jl --check
#   julia -t 16 milestones/fcn3_matrix_theory/fcn3_fast.jl --nmax 3 --kappa 0.1
#   julia -t 16 milestones/fcn3_matrix_theory/fcn3_fast.jl --nmax 3 --branch central
#   julia -t 16 milestones/fcn3_matrix_theory/fcn3_fast.jl --nmax 3 --chi N
#
# `--chi N` (or `N2`) sets χ = N2, so g̃ = σ_A², and solves the fixed point at that χ.
#
using LinearAlgebra, SpecialFunctions, Printf

const INV_SQRT_PI2 = 2 / sqrt(π)   # erf'(0); also 2/√π

# ------------------------------------------------------------------------------------------------ arguments
argval(name, default) = (i = findfirst(==("--" * name), ARGS); i === nothing ? default : ARGS[i + 1])
hasflag(name) = ("--" * name) in ARGS

mutable struct Params
    d::Float64
    P::Float64
    N1::Float64
    N2::Float64
    chi::Float64
    kappa::Float64
    eps::Float64
    sigA::Float64
    sigB::Float64
    sigw::Float64
    nmax::Int
    ks::Vector{Int}
    rscale::Float64     # multiplies r_B; 0 freezes the read-in at the prior
end

mutable struct Numerics
    nw::Int
    wmax::Float64
    wa::Float64
    nq::Int
    lq_below::Float64
    lq_above::Float64
    pair_bw::Int
    pair_bq::Int
    pair_prune::Float64
    nu::Int
    nx::Int
    ds::Float64
    smax::Float64
    nqh::Int
    lqh_below::Float64
    lqh_above::Float64
    zeta_levels::Vector{Tuple{Float64,Int}}
    max_nz::Int
    newton_tol::Float64
    outer_tol::Float64
    basin_floor::Float64
    central::Bool
    basins::Bool
    hidden_basins::Bool
end

beta(p::Params) = p.d / (2 * p.sigw^2)
gtilde(p::Params) = p.sigA^2 * p.chi / p.N2
lam(p::Params) = p.kappa / p.P
rB(p::Params) = (isinf(p.N1) ? 0.0 : p.sigB^2 * p.N2 / p.N1) * p.rscale
nS(p::Params) = length(p.ks)

function target_y(p::Params)
    y = zeros(nS(p))
    y[1] = 1.0
    nS(p) > 1 && (y[2] = p.eps)
    return y
end

function make_params()
    nmax = parse(Int, argval("nmax", "3"))
    N1s = argval("N1", "1600")
    N1 = N1s == "Inf" ? Inf : parse(Float64, N1s)
    N2 = parse(Float64, argval("N2", "1600"))
    chis = argval("chi", "10")
    chi = (chis == "N" || chis == "N2") ? N2 : parse(Float64, chis)
    p = Params(
        parse(Float64, argval("d", "150")),
        parse(Float64, argval("P", "3000")),
        N1,
        N2,
        chi,
        parse(Float64, argval("kappa", "0.1")),
        parse(Float64, argval("eps", "0.074")),
        parse(Float64, argval("sigA", "1")),
        parse(Float64, argval("sigB", "1")),
        parse(Float64, argval("sigw", "1")),
        nmax,
        collect(1:2:nmax),
        1.0,
    )
    branch = argval("branch", "full")
    basins_on = argval("basins", branch == "central" ? "0" : "1") != "0"
    nu = Numerics(
        parse(Int, argval("nw", "601")),
        parse(Float64, argval("wmax", "6")),
        parse(Float64, argval("wa", "0.5")),
        parse(Int, argval("nq", "401")),
        parse(Float64, argval("lq-below", "14")),
        parse(Float64, argval("lq-above", "1.5")),
        6, 7, 1e-14,
        parse(Int, argval("nu", "8")),
        parse(Int, argval("nx", "48")),
        parse(Float64, argval("ds", "0.35")),
        parse(Float64, argval("smax", "40")),
        parse(Int, argval("nqh", "64")),
        parse(Float64, argval("lqh-below", "12")),
        parse(Float64, argval("lqh-above", "2")),
        [(1e-2, 24), (1e-4, 6), (1e-6, 3), (0.0, 2)],
        parse(Int, argval("max-nz", "1024")),
        1e-10, 1e-8, -120.0,
        branch == "central",
        basins_on,
        basins_on,
    )
    return p, nu
end

# ------------------------------------------------------------------------------------------------ quadrature
function trapz_weights(x::AbstractVector)
    w = zeros(length(x))
    length(x) == 1 && (w[1] = 1.0; return w)
    dx = diff(x)
    @inbounds for i in eachindex(dx)
        w[i] += dx[i] / 2
        w[i + 1] += dx[i] / 2
    end
    return w
end

function logsumexp(x)
    m = -Inf
    @inbounds for v in x
        v > m && (m = v)
    end
    m == -Inf && return -Inf
    s = 0.0
    @inbounds for v in x
        s += exp(v - m)
    end
    return m + log(s)
end

"""Probabilists' Gauss–Hermite: ∫ f(x) e^{-x²/2} dx ≈ Σ wᵢ f(xᵢ), Σ w = √(2π)."""
function gausshermite_e(n::Int)
    T = SymTridiagonal(zeros(n), sqrt.(1.0:n - 1))
    F = eigen(T)
    w = sqrt(2π) .* (F.vectors[1, :] .^ 2)
    idx = sortperm(F.values)
    return F.values[idx], w[idx]
end

"""∫ f(x) (1-x)^α (1+x)^β dx on [-1, 1]."""
function gaussjacobi(n::Int, α::Float64, β::Float64)
    ab = α + β
    d = zeros(n)
    d[1] = (β - α) / (ab + 2)
    @inbounds for i in 2:n
        d[i] = (β^2 - α^2) / ((2i + ab - 2) * (2i + ab))
    end
    s = zeros(n - 1)
    s[1] = 2 * sqrt((α + 1) * (β + 1) / (ab + 3)) / (ab + 2)
    @inbounds for i in 2:n - 1
        s[i] = 2 * sqrt(i * (i + α) * (i + β) * (i + ab) /
                        ((2i + ab - 1) * (2i + ab)^2 * (2i + ab + 1)))
    end
    F = eigen(SymTridiagonal(d, s))
    μ0 = exp((α + β + 1) * log(2) + loggamma(α + 1) + loggamma(β + 1) - loggamma(α + β + 2))
    w = μ0 .* (F.vectors[1, :] .^ 2)
    idx = sortperm(F.values)
    return F.values[idx], w[idx]
end

function product_rule(counts::Vector{Int})
    isempty(counts) && return zeros(1, 0), [1.0]
    nodes = Vector{Vector{Float64}}(undef, length(counts))
    weights = Vector{Vector{Float64}}(undef, length(counts))
    for (j, n) in enumerate(counts)
        x, w = gausshermite_e(n)
        nodes[j] = x
        weights[j] = w ./ sum(w)
    end
    nz = prod(counts)
    Z = zeros(nz, length(counts))
    wz = ones(nz)
    stride = 1
    for j in eachindex(counts)
        nj = counts[j]
        @inbounds for t in 1:nz
            i = ((t - 1) ÷ stride) % nj + 1
            Z[t, j] = nodes[j][i]
            wz[t] *= weights[j][i]
        end
        stride *= nj
    end
    return Z, wz
end

function zeta_count(ratio, levels)
    for (thr, n) in levels
        ratio >= thr && return n
    end
    return levels[end][2]
end

function cap_counts!(counts, max_nz)
    while !isempty(counts) && prod(counts) > max_nz
        k = argmax(counts)
        counts[k] <= 2 && break
        counts[k] -= 1
    end
    return counts
end

# ------------------------------------------------------------------------------------------------ features
function cn_const(n::Int)
    m = (n - 1) ÷ 2
    df = 1.0
    k = n - 2
    while k > 0
        df *= k
        k -= 2
    end
    return INV_SQRT_PI2 * ((-2.0)^m) * df * exp(-0.5 * loggamma(n + 1))
end

function hermite_phi(ks, x)
    maxn = maximum(ks)
    He = Vector{Vector{Float64}}(undef, maxn + 1)
    He[1] = ones(length(x))
    maxn >= 1 && (He[2] = collect(x))
    for n in 1:maxn - 1
        He[n + 2] = x .* He[n + 1] .- n .* He[n]
    end
    φ = zeros(length(ks), length(x))
    for (i, k) in enumerate(ks)
        φ[i, :] .= He[k + 1] .* exp(-0.5 * loggamma(k + 1))
    end
    return φ
end

function features_grid!(C, W, Q, ks, cnst)
    nw, nq = size(W)
    @inbounds for j in 1:nq, i in 1:nw
        Tw = 1 + 2 * (W[i, j]^2 + Q[i, j])
        w = W[i, j]
        for (a, k) in enumerate(ks)
            C[a, i, j] = cnst[a] * w^k / Tw^(k / 2)
        end
    end
    return C
end

function features_atoms!(C, w0, q, ks, cnst)
    @inbounds for i in eachindex(w0)
        Tw = 1 + 2 * (w0[i]^2 + q[i])
        w = w0[i]
        for (a, k) in enumerate(ks)
            C[a, i] = cnst[a] * w^k / Tw^(k / 2)
        end
    end
    return C
end

jxx(n2) = (2 / π) * asin(2 * n2 / (1 + 2 * n2))

"""e_n(h, qh) by Gauss–Hermite in x0. h is a length-S vector."""
function en_at(h, qh, φ, wx)
    S, nx = size(φ)
    ρ = 1 / sqrt(1 + 2 * qh)
    e = zeros(S)
    @inbounds for ix in 1:nx
        arg = 0.0
        for n in 1:S
            arg += h[n] * φ[n, ix]
        end
        f = erf(arg * ρ)
        for n in 1:S
            e[n] += f * φ[n, ix] * wx[ix]
        end
    end
    return e
end

# ------------------------------------------------------------------------------------------------ read-in grid
struct ReadIn
    w::Vector{Float64}
    lq::Vector{Float64}
    W::Matrix{Float64}
    Q::Matrix{Float64}
    C::Array{Float64,3}       # (S, nw, nq)
    base::Matrix{Float64}
    knorm::Matrix{Float64}
    Pperp::Matrix{Float64}
    logm::Matrix{Float64}
    q0::Float64
    seed::Int
    u::Vector{Float64}
    wu::Vector{Float64}
    cnst::Vector{Float64}
    ks::Vector{Int}
end

function build_readin(p::Params, nu::Numerics)
    q0 = p.sigw^2 * (p.d - 1) / p.d
    w = nu.wa .* sinh.(range(0, asinh(nu.wmax / nu.wa), length=nu.nw))
    lq = range(log(q0) - nu.lq_below, log(q0) + nu.lq_above, length=nu.nq)
    lqv = collect(lq)
    W = [w[i] for i in 1:nu.nw, _ in 1:nu.nq]
    LQ = [lqv[j] for _ in 1:nu.nw, j in 1:nu.nq]
    Q = exp.(LQ)
    ks = p.ks
    cnst = cn_const.(ks)
    C = zeros(length(ks), nu.nw, nu.nq)
    features_grid!(C, W, Q, ks, cnst)
    base = beta(p) .* (W .^ 2 .+ Q) .- 0.5 * (p.d - 1) .* LQ
    kn = jxx.(W .^ 2 .+ Q)
    Pperp = kn .- dropdims(sum(C .^ 2, dims=1), dims=1)
    tw, tq = trapz_weights(w), trapz_weights(lqv)
    logm = log.(tw) .+ log.(tq)'
    seedq = argmin(abs.(lqv .- log(q0)))
    seed = 1 + (seedq - 1) * nu.nw                 # column-major (i=1, j=seedq)
    α = (p.d - 4) / 2
    uj, wj = gaussjacobi(nu.nu, α, α)
    return ReadIn(w, lqv, W, Q, C, base, kn, Pperp, logm, q0, seed, uj, wj ./ sum(wj), cnst, ks)
end

# ------------------------------------------------------------------------------------------------ basins
"""Label every cell by the linear index of the local minimum of A reached by
steepest descent over the 8 neighbours. Row 1 is a mirror line (w0 = 0 or s = 0)."""
function basin_labels(A::AbstractMatrix)
    n0, n1 = size(A)
    N = n0 * n1
    ptr = collect(1:N)
    @inbounds for j in 1:n1, i in 1:n0
        bi, bj, bv = i, j, A[i, j]
        for dj in -1:1, di in -1:1
            ii = i + di
            jj = j + dj
            if ii == 0 && n0 > 1
                ii = 2
            end
            (ii < 1 || ii > n0 || jj < 1 || jj > n1) && continue
            if A[ii, jj] < bv
                bv = A[ii, jj]
                bi, bj = ii, jj
            end
        end
        ptr[i + (j - 1) * n0] = bi + (bj - 1) * n0
    end
    for _ in 1:(ceil(Int, log2(max(N, 2))) + 2)
        ptr = ptr[ptr]
    end
    return ptr
end

"""Far basins whose log-mass ratio to the seed basin exceeds `floor`.
Returns g (0 = central, 1..m = far) and m."""
function make_groups(A, logmass, seed, floor)
    lab = basin_labels(A)
    r0 = lab[seed]
    roots = Int[]
    masses = Float64[]
    # log mass of the seed basin
    l0 = group_lse_lab(logmass, lab, r0)
    seen = Dict{Int,Int}()
    for r in unique(lab)
        r == r0 && continue
        lr = group_lse_lab(logmass, lab, r)
        if lr - l0 > floor
            push!(roots, r)
            push!(masses, lr - l0)
        end
    end
    if length(roots) > 4
        keep = sortperm(masses, rev=true)[1:4]
        roots = roots[keep]
    end
    g = zeros(Int, length(lab))
    for (k, r) in enumerate(roots)
        @inbounds for i in eachindex(lab)
            lab[i] == r && (g[i] = k)
        end
    end
    return g, length(roots)
end

"""One far read-in basin. The outer well is born near w0 ≈ 1.3 with a log-mass ratio
near −70 and moves out as it deepens; minima at w0 ≤ wcut are grid wiggles of the
central valley and stay in B_0. Every minimum past wcut is that one well."""
function make_readin_groups(A, logmass, w, seed, floor, wcut)
    lab = basin_labels(A)
    n0 = length(w)
    r0 = lab[seed]
    l0 = group_lse_lab(logmass, lab, r0)
    far = Int[]
    for r in unique(lab)
        r == r0 && continue
        i = mod1(r, n0)
        w[i] <= wcut && continue
        lr = group_lse_lab(logmass, lab, r)
        lr - l0 > floor && push!(far, r)
    end
    g = zeros(Int, length(lab))
    isempty(far) && return g, 0
    mark = falses(length(lab))
    for r in far
        mark[r] = true
    end
    @inbounds for i in eachindex(lab)
        mark[lab[i]] && (g[i] = 1)
    end
    return g, 1
end

function group_lse_lab(logmass, lab, r)
    m = -Inf
    @inbounds for i in eachindex(lab)
        lab[i] == r && logmass[i] > m && (m = logmass[i])
    end
    m == -Inf && return -Inf
    s = 0.0
    @inbounds for i in eachindex(lab)
        lab[i] == r && (s += exp(logmass[i] - m))
    end
    return m + log(s)
end

function group_lse_g(logmass, g, k)
    m = -Inf
    @inbounds for i in eachindex(g)
        g[i] == k && logmass[i] > m && (m = logmass[i])
    end
    m == -Inf && return -Inf
    s = 0.0
    @inbounds for i in eachindex(g)
        g[i] == k && (s += exp(logmass[i] - m))
    end
    return m + log(s)
end

"""Reweight log-masses so basin k carries softmax mass. Returns (Lr, delta)."""
function apply_groups(Lc, g, m, ell)
    m == 0 && return Lc, Float64[]
    lse = [group_lse_g(Lc, g, k) for k in 0:m]
    delta = lse[2:end] .- lse[1]
    ell === nothing && return Lc, delta
    logits = vcat(0.0, ell[1:m])
    lp = logits .- logsumexp(logits)
    Lr = similar(Lc)
    @inbounds for i in eachindex(Lc)
        k = g[i]
        Lr[i] = Lc[i] - lse[k + 1] + lp[k + 1]
    end
    return Lr, delta
end

"""First local minimum of the marginal log-mass along axis 1 (the barrier)."""
function barrier_index(L::AbstractMatrix)
    n0 = size(L, 1)
    ls = Vector{Float64}(undef, n0)
    for i in 1:n0
        ls[i] = logsumexp(@view L[i, :])
    end
    for i in 2:n0 - 1
        if ls[i] < ls[i - 1] && ls[i] <= ls[i + 1]
            return i
        end
    end
    return n0
end

# ------------------------------------------------------------------------------------------------ hidden table
mutable struct Hidden
    s::Vector{Float64}
    logws::Vector{Float64}
    qh::Vector{Float64}
    logwq::Vector{Float64}
    rq::Vector{Float64}
    zeta::Matrix{Float64}
    logwz::Vector{Float64}
    Zz2::Matrix{Float64}
    u::Vector{Float64}
    Qp::Matrix{Float64}
    a::Vector{Float64}
    Aζ::Matrix{Float64}
    B::Array{Float64,4}
    Je::Array{Float64,5}
    He::Array{Float64,6}
    deriv::Bool
    counts::Vector{Int}
    g::Vector{Int}
    m::Int
    qbar::Float64
    M::Float64
end

function Hidden()
    z = zeros(0)
    return Hidden(z, z, z, z, z, zeros(0, 0), z, zeros(0, 0), z, zeros(0, 0), z, zeros(0, 0),
        zeros(0, 0, 0, 0), zeros(0, 0, 0, 0, 0), zeros(0, 0, 0, 0, 0, 0), false, Int[], Int[], 0, NaN, NaN)
end

function setup_frame!(H::Hidden, L, tdir, qbar, M, φ, wx, nu::Numerics)
    S = size(L, 1)
    v = L' * tdir
    if !(norm(v) > 0)
        v = zeros(S)
        v[1] = 1.0
    end
    u = v ./ norm(v)
    if S == 1
        Qp = zeros(S, 0)
        counts = Int[]
        zeta, wz = zeros(1, 0), [1.0]
    else
        Qp = nullspace(reshape(u, 1, :))
        sw = sqrt.(wx)
        Sens = Qp' * (L' * φ)                    # (nspec, nx)
        @inbounds for ix in 1:size(Sens, 2)
            Sens[:, ix] .*= sw[ix]
        end
        F = svd(Sens)
        Qp = Qp * F.U[:, 1:length(F.S)]
        scale = norm(φ' * (L * u) .* sw)
        counts = [zeta_count(F.S[i] / scale, nu.zeta_levels) for i in eachindex(F.S)]
        cap_counts!(counts, nu.max_nz)
        zeta, wz = product_rule(counts)
    end
    a = φ' * (L * u)
    nspec = size(Qp, 2)
    Aζ = zeros(size(zeta, 1), size(φ, 2))
    if nspec > 0
        Mdir = φ' * (L * Qp)                     # (nx, nspec)
        @inbounds for iz in 1:size(zeta, 1), ix in 1:size(φ, 2)
            s = 0.0
            for j in 1:nspec
                s += zeta[iz, j] * Mdir[ix, j]
            end
            Aζ[iz, ix] = s
        end
    end
    ns = length(H.s)
    nqh = nu.nqh
    k = M / 2
    lqh = log(qbar) .+ range(-nu.lqh_below, nu.lqh_above, length=nqh)
    qh = exp.(lqh)
    lw = log.(trapz_weights(lqh)) .+ k .* (lqh .- log(qbar)) .- k .* qh ./ qbar
    logwq = lw .- logsumexp(lw)
    rq = 1 ./ sqrt.(1 .+ 2 .* qh)
    Zz2 = zeros(nspec, nspec)
    @inbounds for iz in 1:size(zeta, 1)
        for a in 1:nspec, b in 1:nspec
            Zz2[a, b] += wz[iz] * zeta[iz, a] * zeta[iz, b]
        end
    end
    nz = size(zeta, 1)
    bytes = ns * nqh * nz * (S^2 + S^3) * 8
    # Moment form of V = G^{-1}(G−H)G^{-1} is rounding noise once J_nn spans many
    # decades (n_max = 11). Stein's form has no G^{-1}. Keep it up to He11 when it fits.
    deriv = bytes < 1.2e9 && S <= 6
    if size(H.B) != (ns, nqh, nz, S) || H.deriv != deriv
        H.B = zeros(ns, nqh, nz, S)
        if deriv
            H.Je = zeros(ns, nqh, nz, S, S)
            H.He = zeros(ns, nqh, nz, S, S, S)
        end
    end
    H.qh, H.logwq, H.rq = qh, logwq, rq
    H.zeta, H.logwz, H.Zz2 = zeta, log.(wz), Zz2
    H.u, H.Qp, H.a, H.Aζ = u, Qp, a, Aζ
    H.deriv, H.counts = deriv, counts
    H.qbar, H.M = qbar, M
    return H
end

"""New L, q̄, M on a frozen (u, Q⊥, ζ) frame. The z-grid stays a standard normal, so the
residual is smooth in J; only the map z ↦ e(Lz, q_h) changes."""
function rebind!(H::Hidden, L, qbar, M, φ, nu::Numerics)
    H.a = φ' * (L * H.u)
    nspec = size(H.Qp, 2)
    if nspec > 0
        Mdir = φ' * (L * H.Qp)
        nz, nx = size(H.Aζ)
        @inbounds for iz in 1:nz, ix in 1:nx
            s = 0.0
            for j in 1:nspec
                s += H.zeta[iz, j] * Mdir[ix, j]
            end
            H.Aζ[iz, ix] = s
        end
    end
    k = M / 2
    lqh = log(qbar) .+ range(-nu.lqh_below, nu.lqh_above, length=nu.nqh)
    H.qh = exp.(lqh)
    lw = log.(trapz_weights(lqh)) .+ k .* (lqh .- log(qbar)) .- k .* H.qh ./ qbar
    H.logwq = lw .- logsumexp(lw)
    H.rq = 1 ./ sqrt.(1 .+ 2 .* H.qh)
    H.qbar, H.M = qbar, M
    return H
end

function build_features!(H::Hidden, φ, Φw)
    ns, nqh, nz, S = size(H.B)
    deriv = H.deriv
    deriv && (fill!(H.Je, 0); fill!(H.He, 0))
    fill!(H.B, 0)
    Threads.@threads for iz in 1:nz
        @inbounds for iq in 1:nqh
            ρ = H.rq[iq]
            ρ2 = ρ * ρ
            for is in 1:ns
                ss = H.s[is]
                for ix in 1:size(φ, 2)
                    arg = (ss * H.a[ix] + H.Aζ[iz, ix]) * ρ
                    eφ = erf(arg)
                    for n in 1:S
                        H.B[is, iq, iz, n] += eφ * Φw[n, ix]
                    end
                    deriv || continue
                    ep = INV_SQRT_PI2 * exp(-arg * arg)
                    epp = -2 * arg * ep
                    for n in 1:S
                        c1 = Φw[n, ix] * ep * ρ
                        c2 = Φw[n, ix] * epp * ρ2
                        for k in 1:S
                            pk = φ[k, ix]
                            H.Je[is, iq, iz, n, k] += c1 * pk
                            for j in 1:S
                                H.He[is, iq, iz, n, k, j] += c2 * pk * φ[j, ix]
                            end
                        end
                    end
                end
            end
        end
    end
    return H
end

function init_s_grid!(H::Hidden, nu::Numerics)
    ns = ceil(Int, nu.smax / nu.ds) + 1
    H.s = nu.ds .* collect(0:ns - 1)
    H.logws = fill(log(2 * nu.ds), ns)
    H.logws[1] = log(nu.ds)
    H.g = zeros(Int, ns * nu.nqh)
    H.m = 0
    return H
end

"""Trapezoid weights on s ≥ 0, with s > 0 doubled for the ±s pair. Uniform ds reproduces log(ds), log(2ds), …."""
function half_line_logw(s::Vector{Float64})
    n = length(s)
    n <= 1 && return zeros(n)
    w = zeros(n)
    w[1] = s[2] - s[1]
    @inbounds for i in 2:n - 1
        w[i] = s[i + 1] - s[i - 1]
    end
    w[n] = 2 * (s[n] - s[n - 1])
    return log.(max.(w, 1e-300))
end

function uniq_sorted!(s::Vector{Float64})
    sort!(s)
    isempty(s) && return s
    n = 1
    @inbounds for i in 2:length(s)
        if s[i] > s[n] + 1e-6
            n += 1
            s[n] = s[i]
        end
    end
    resize!(s, n)
    return s
end

"""|t · e| ≤ Σ_x |t · Φw[:,x]| because |erf| ≤ 1."""
function boost_bound(t, Φw)
    b = 0.0
    Sd, nx = size(Φw)
    @inbounds for ix in 1:nx
        s = 0.0
        for n in 1:Sd
            s += t[n] * Φw[n, ix]
        end
        b += abs(s)
    end
    return b
end

"""Max over a few q_h of −s²/2 + (g̃/2)(t·e)² along ζ = 0. Locates the data-term well."""
function s_profile(S, hu, t, qbar, smax)
    gt = gtilde(S.p)
    ds = 0.25
    ns = ceil(Int, smax / ds) + 1
    sc = Vector{Float64}(undef, ns)
    lw = fill(-Inf, ns)
    φ, wx = S.φ, S.wx
    Sd, nx = size(φ)
    e = zeros(Sd)
    @inbounds for is in 1:ns
        sc[is] = (is - 1) * ds
    end
    @inbounds for lq in (-12.0, -8.0, -4.0, -2.0, 0.0, 1.0)
        ρ = 1 / sqrt(1 + 2 * qbar * exp(lq))
        for is in 1:ns
            ss = sc[is]
            for n in 1:Sd
                e[n] = 0.0
            end
            for ix in 1:nx
                arg = 0.0
                for n in 1:Sd
                    arg += ss * hu[n] * φ[n, ix]
                end
                wf = erf(arg * ρ) * wx[ix]
                for n in 1:Sd
                    e[n] += wf * φ[n, ix]
                end
            end
            te = 0.0
            for n in 1:Sd
                te += t[n] * e[n]
            end
            v = -0.5 * ss * ss + 0.5 * gt * te * te
            v > lw[is] && (lw[is] = v)
        end
    end
    return sc, lw
end

function nodes_from_profile(sc, lw, ds0)
    peak = maximum(lw)
    ds = ds0
    @inbounds for i in 2:length(sc) - 1
        lw[i] >= peak - 40 || continue
        (lw[i] >= lw[i - 1] && lw[i] >= lw[i + 1]) || continue
        d1 = sc[i] - sc[i - 1]
        d2 = sc[i + 1] - sc[i]
        g1 = (lw[i] - lw[i - 1]) / d1
        g2 = (lw[i + 1] - lw[i]) / d2
        curv = -2 * (g2 - g1) / (d1 + d2)
        if curv > 0
            ds = min(ds, max(0.02, 0.25 / sqrt(curv)))
        end
    end
    span = 0.0
    step = length(sc) > 1 ? sc[2] - sc[1] : ds
    for i in eachindex(sc)
        (lw[i] > peak - 50 || sc[i] <= 6) && (span += step)
    end
    ds = max(ds, span / 1500)
    nodes = Float64[]
    i = 1
    nsc = length(sc)
    while i <= nsc
        hot = lw[i] > peak - 50 || sc[i] <= 6
        if !hot
            i += 1
            continue
        end
        lo = sc[i]
        while i <= nsc && (lw[i] > peak - 50 || sc[i] <= 6)
            i += 1
        end
        hi = sc[i - 1]
        append!(nodes, collect(max(0.0, lo - 1.5):ds:(hi + 1.5)))
    end
    push!(nodes, 0.0)
    return uniq_sorted!(nodes)
end

function install_s!(S, s::Vector{Float64})
    H = S.H
    H.s = s
    H.logws = half_line_logw(s)
    H.g = zeros(Int, length(s) * S.nu.nqh)
    H.m = 0
    S.frame_ready = false
    S.freeze = false
    S.cache = Float64[]
    return length(s)
end

"""Grow the s-mesh when the data-term bound no longer fits on it. The user `--smax` is a floor."""
function adapt_s!(S, x, force_cap=0.0)
    H = S.H
    isempty(H.s) && return false
    J, qbar, _ = decode_state(x, nS(S.p))
    ch = scaled_cholesky(S.p.sigB^2 * J)
    ch === nothing && return false
    t = norm(S.t) == 0 ? S.y : S.t
    b = boost_bound(t, S.Φw)
    gt = gtilde(S.p)
    # Sufficient coverage: even the largest admissible boost is 60 nats below the prior at s_end.
    if force_cap <= H.s[end] + 1e-9 && 0.5 * H.s[end]^2 > 0.5 * gt * b * b + 60
        return false
    end
    v = ch[1]' * t
    nv = norm(v)
    u = zeros(length(t))
    if nv == 0
        u[1] = 1.0
    else
        u .= v ./ nv
    end
    s_cap = sqrt(max(gt * b * b + 160.0, 64.0))
    s_cap = clamp(max(s_cap, force_cap, 8.0), 8.0, 2000.0)
    sc, lw = s_profile(S, ch[1] * u, t, qbar, s_cap)
    peak = maximum(lw)
    ihi = 1
    for i in eachindex(lw)
        (lw[i] > peak - 60 || sc[i] <= 6) && (ihi = i)
    end
    s_hi = min(s_cap, sc[ihi] + 3)
    s_hi = max(s_hi, force_cap)
    if s_hi <= H.s[end] + 0.5 && force_cap <= H.s[end] + 1e-9
        return false
    end
    nodes = nodes_from_profile(sc, lw, S.nu.ds)
    if isempty(nodes) || nodes[end] + 1e-6 < s_hi
        push!(nodes, s_hi)
        uniq_sorted!(nodes)
    end
    # Keep the mesh that is already there when it already covers the mass.
    if nodes[end] <= H.s[end] + 0.5 && length(nodes) >= length(H.s) && force_cap <= H.s[end]
        return false
    end
    ns = install_s!(S, nodes)
    if S.verbose
        im = argmax(lw)
        @printf("  s-grid → %d nodes on [0, %.2f], probe mode s=%.2f, gt=%.3g\n",
            ns, nodes[end], sc[im], gt)
        flush(stdout)
    end
    return true
end

function matrix_edge(M)
    size(M, 1) == 0 && return -Inf
    return logsumexp(view(M, size(M, 1), :)) - logsumexp(M)
end

function readin_w_edge(S)
    hasproperty(S.last, :logmass_r) || return -Inf
    return matrix_edge(S.last.logmass_r)
end

"""The read-in mass is using the outer part of the w0 grid, so the well is not enclosed."""
function readin_cramped(S)
    wmax = S.R.w[end]
    S.last.w0sq > 0.15 * wmax^2 && return true
    return readin_w_edge(S) > -40
end

function extend_readin!(S)
    wm = min(40.0, max(S.nu.wmax * 1.8, S.nu.wmax + 4))
    wm <= S.nu.wmax + 0.5 && return false
    S.nu.wmax = wm
    S.nu.nw = max(S.nu.nw, ceil(Int, 601 * wm / 6))
    S.R = build_readin(S.p, S.nu)
    S.ri_g = zeros(Int, length(S.R.base))
    S.ri_m = 0
    S.verbose && (@printf("  read-in grid → wmax=%.2f (nw=%d)\n", wm, S.nu.nw); flush(stdout))
    return true
end

function s_mode(S)
    Lc, s = S.last.Lc, S.H.s
    (size(Lc, 1) == 0 || size(Lc, 1) != length(s)) && return NaN
    best, ib = -Inf, 1
    for i in 1:size(Lc, 1)
        v = logsumexp(view(Lc, i, :))
        if v > best
            best = v
            ib = i
        end
    end
    return s[ib]
end

# ------------------------------------------------------------------------------------------------ reweighting
struct HidMoments
    K::Matrix{Float64}
    Cov::Matrix{Float64}
    Ezz::Matrix{Float64}
    Eq::Float64
    V::Matrix{Float64}
    delta::Vector{Float64}
    Lc::Matrix{Float64}
    edge::Float64          # log mass in the last s-bin, relative to the total
end

function hidden_moments(H::Hidden, t, gt, Linv; ell=nothing, central=false)
    ns, nqh, nz, S = size(H.B)
    Lc = zeros(ns, nqh)
    tb = zeros(nz)
    lw = zeros(nz)
    @inbounds for iq in 1:nqh, is in 1:ns
        for iz in 1:nz
            s = 0.0
            for n in 1:S
                s += H.B[is, iq, iz, n] * t[n]
            end
            tb[iz] = s
            lw[iz] = H.logwz[iz] + 0.5 * gt * s * s
        end
        Lc[is, iq] = logsumexp(lw) + H.logwq[iq] + H.logws[is] - 0.5 * H.s[is]^2
    end
    if central
        cut = barrier_index(Lc)
        Lc[cut + 1:end, :] .= -Inf
    end
    g = reshape(H.g, ns, nqh)
    Lr, delta = apply_groups(Lc, g, ell === nothing ? 0 : H.m, ell)
    mL = maximum(Lr)
    Z = 0.0
    @inbounds for v in Lr
        Z += exp(v - mL)
    end
    # prior control on s
    lp0 = H.logws .- 0.5 .* H.s .^ 2
    m0 = maximum(lp0)
    p0 = exp.(lp0 .- m0)
    p0 ./= sum(p0)
    Es2_0 = dot(p0, H.s .^ 2)
    nspec = size(H.Qp, 2)
    K = zeros(S, S)
    tB1 = zeros(S)
    t2BB = zeros(S, S)
    Es2 = 0.0
    Esz = zeros(nspec)
    Ezzζ = zeros(nspec, nspec)
    Eq = 0.0
    V = zeros(S, S)
    gu = zeros(S)
    Hu = zeros(S, S)
    e = zeros(S)
    # within-group covariance pieces
    mgroups = ell === nothing ? 0 : H.m
    mt = [zeros(S) for _ in 0:mgroups]
    mk = zeros(mgroups + 1)
    Ct = [zeros(S, S) for _ in 0:mgroups]
    @inbounds for iq in 1:nqh, is in 1:ns
        pi = exp(Lr[is, iq] - mL) / Z
        pi == 0 && continue
        for iz in 1:nz
            s = 0.0
            for n in 1:S
                e[n] = H.B[is, iq, iz, n]
                s += e[n] * t[n]
            end
            tbv = s
            lw[iz] = H.logwz[iz] + 0.5 * gt * tbv * tbv
        end
        mf = maximum(lw)
        sf = 0.0
        for iz in 1:nz
            lw[iz] = exp(lw[iz] - mf)
            sf += lw[iz]
        end
        for iz in 1:nz
            f = lw[iz] / sf
            w = pi * f
            tbv = 0.0
            for n in 1:S
                e[n] = H.B[is, iq, iz, n]
                tbv += e[n] * t[n]
            end
            for n in 1:S
                tB1[n] += w * tbv * e[n]
                for m in 1:S
                    K[n, m] += w * e[n] * e[m]
                    t2BB[n, m] += w * tbv * tbv * e[n] * e[m]
                end
            end
            if nspec > 0
                for a in 1:nspec
                    Esz[a] += w * H.s[is] * H.zeta[iz, a]
                    for b in 1:nspec
                        Ezzζ[a, b] += w * H.zeta[iz, a] * H.zeta[iz, b]
                    end
                end
            end
            if H.deriv
                for k in 1:S
                    s = 0.0
                    for n in 1:S
                        s += H.Je[is, iq, iz, n, k] * t[n]
                    end
                    gu[k] = s
                end
                for j in 1:S, i in 1:S
                    s = 0.0
                    for n in 1:S
                        s += H.He[is, iq, iz, n, i, j] * t[n]
                    end
                    Hu[i, j] = s
                end
                c = 1 + gt * tbv * tbv
                for j in 1:S, i in 1:S
                    V[i, j] += w * (-gt) * (c * gu[i] * gu[j] + tbv * Hu[i, j])
                end
            end
            if mgroups > 0
                k = g[is, iq]
                mk[k + 1] += w
                for n in 1:S
                    mt[k + 1][n] += w * tbv * e[n]
                    for m in 1:S
                        Ct[k + 1][n, m] += w * tbv * tbv * e[n] * e[m]
                    end
                end
            end
        end
        Es2 += pi * H.s[is]^2
        Eq += pi * H.qh[iq]
    end
    # Ezz
    cross = nspec == 0 ? zeros(S) : H.Qp * Esz
    Ezz = Es2 * (H.u * H.u') .+ H.u * cross' .+ cross * H.u'
    nspec > 0 && (Ezz .+= H.Qp * Ezzζ * H.Qp')
    Ezz0 = Es2_0 * (H.u * H.u')
    nspec > 0 && (Ezz0 .+= H.Qp * H.Zz2 * H.Qp')
    Ezzc = Ezz - Ezz0 + I(S)
    Ezzc = (Ezzc + Ezzc') ./ 2
    # <qh> control
    Eq0 = sum(exp.(H.logwq) .* H.qh)
    Eqc = Eq - Eq0 + H.qbar
    if !H.deriv
        V = Linv' * (I(S) - Ezzc) * Linv
    end
    V = (V + V') ./ 2
    if mgroups == 0
        Cov = t2BB - tB1 * tB1'
    else
        Cov = zeros(S, S)
        for k in 0:mgroups
            mk[k + 1] <= 0 && continue
            mtt = mt[k + 1] ./ mk[k + 1]
            Cov .+= Ct[k + 1] .- mk[k + 1] .* (mtt * mtt')
        end
    end
    Cov = (Cov + Cov') ./ 2
    edge = logsumexp(Lc[end, :]) - logsumexp(Lc)
    return HidMoments(K, Cov, Ezzc, Eqc, V, delta, Lc, edge)
end

# ------------------------------------------------------------------------------------------------ read-in moments
struct ReadMoments
    J::Matrix{Float64}
    EJxx::Float64
    EJ2::Float64
    Jat::Matrix{Float64}
    w0sq::Float64
    qmean::Float64
    delta::Vector{Float64}
    action::Matrix{Float64}
    logmass::Matrix{Float64}
end

function quad_form(C, V)
    S, nw, nq = size(C)
    Cr = reshape(C, S, nw * nq)
    Tmp = V * Cr
    out = zeros(nw, nq)
    @inbounds for i in 1:nw * nq
        s = 0.0
        for k in 1:S
            s += Cr[k, i] * Tmp[k, i]
        end
        out[i] = s
    end
    return out
end

function readin_moments(R::ReadIn, p::Params, nu::Numerics, V, mcoef, g, m;
        ell=nothing, central=false)
    action = R.base .+ 0.5 * rB(p) .* quad_form(R.C, V) .+ 0.5 * rB(p) * mcoef .* R.Pperp
    Lc = -action .+ R.logm
    if central
        cut = barrier_index(Lc)
        Lc[cut + 1:end, :] .= -Inf
    end
    gv = g === nothing ? zeros(Int, length(Lc)) : g
    mm = g === nothing ? 0 : m
    Lr, delta = apply_groups(Lc, gv, ell === nothing ? 0 : mm, ell)
    mL = maximum(Lr)
    Z = 0.0
    @inbounds for v in Lr
        Z += exp(v - mL)
    end
    S = size(R.C, 1)
    J = zeros(S, S)
    EJxx = 0.0
    w0sq = 0.0
    qmean = 0.0
    pw = similar(Lr)
    @inbounds for j in 1:size(Lr, 2), i in 1:size(Lr, 1)
        pij = exp(Lr[i, j] - mL) / Z
        pw[i, j] = pij
        EJxx += pij * R.knorm[i, j]
        w0sq += pij * R.W[i, j]^2
        qmean += pij * R.Q[i, j]
        for b in 1:S, a in 1:S
            J[a, b] += pij * R.C[a, i, j] * R.C[b, i, j]
        end
    end
    EJ2, Jat = pair_trace(R, nu, pw)
    return ReadMoments((J + J') ./ 2, EJxx, EJ2, Jat, w0sq, qmean, delta, action, Lc)
end

function pair_trace(R::ReadIn, nu::Numerics, pw)
    n0, n1 = size(pw)
    bw, bq = nu.pair_bw, nu.pair_bq
    nb0, nb1 = cld(n0, bw), cld(n1, bq)
    m = zeros(nb0, nb1)
    sw = zeros(nb0, nb1)
    sqv = zeros(nb0, nb1)
    @inbounds for j in 1:n1, i in 1:n0
        bi, bj = cld(i, bw), cld(j, bq)
        pij = pw[i, j]
        m[bi, bj] += pij
        sw[bi, bj] += pij * R.W[i, j]
        sqv[bi, bj] += pij * R.Q[i, j]
    end
    mv, wv, qv = vec(m), vec(sw), vec(sqv)
    keep = mv .> nu.pair_prune * maximum(mv)
    mv, wv, qv = mv[keep], wv[keep], qv[keep]
    w0 = wv ./ mv
    q = qv ./ mv
    mv = mv ./ sum(mv)
    nat = length(mv)
    S = length(R.ks)
    Ca = zeros(S, nat)
    features_atoms!(Ca, w0, q, R.ks, R.cnst)
    Jat = (Ca .* mv') * Ca'
    T = 1 .+ 2 .* (w0 .^ 2 .+ q)
    sq = sqrt.(q)
    acc = 0.0
    nuu = length(R.u)
    @inbounds for a in 1:nat
        sa = 0.0
        for b in 1:nat
            den = sqrt(T[a] * T[b])
            kb = 0.0
            for iu in 1:nuu
                z = (2 * w0[a] * w0[b] + 2 * sq[a] * sq[b] * R.u[iu]) / den
                z = z > 1 ? 1.0 : (z < -1 ? -1.0 : z)
                as = asin(z)
                kb += R.wu[iu] * (2 / π * as)^2
            end
            sa += mv[b] * kb
        end
        acc += mv[a] * sa
    end
    return acc, (Jat + Jat') ./ 2
end

function state_from(ri::ReadMoments, p::Params)
    qbar = p.sigB^2 * (ri.EJxx - tr(ri.J))
    trG2 = p.sigB^4 * (ri.EJ2 - sum(abs2, ri.Jat))
    (qbar <= 0 || trG2 <= 0) && return nothing
    return ri.J, qbar, qbar^2 / trG2
end

# ------------------------------------------------------------------------------------------------ encode
nstate(S) = S + S * (S - 1) ÷ 2 + 2

function encode_state(J, qbar, M)
    S = size(J, 1)
    d = diag(J)
    sd = sqrt.(max.(d, 1e-300))
    x = zeros(nstate(S))
    x[1:S] .= log.(d)
    k = S
    @inbounds for j in 2:S, i in 1:j - 1
        k += 1
        r = J[i, j] / (sd[i] * sd[j])
        x[k] = atanh(clamp(r, -1 + 1e-15, 1 - 1e-15))
    end
    x[end - 1] = log(qbar)
    x[end] = log(M)
    return x
end

function decode_state(x, S)
    d = exp.(@view x[1:S])
    sd = sqrt.(d)
    R = Matrix{Float64}(I, S, S)
    k = S
    @inbounds for j in 2:S, i in 1:j - 1
        k += 1
        R[i, j] = R[j, i] = tanh(x[k])
    end
    J = (sd * sd') .* R
    return (J + J') ./ 2, exp(x[end - 1]), exp(x[end])
end

function scaled_cholesky(G)
    d = sqrt.(diag(G))
    any(x -> !(x > 0), d) && return nothing
    R = G ./ (d * d')
    R = (R + R') ./ 2
    # tiny negative eigenvalues from the quadrature
    ev = eigvals(Symmetric(R))
    if ev[1] <= 0
        R = R + (1e-12 - ev[1]) * I(length(d))
    end
    C = try
        cholesky(Symmetric(R))
    catch
        return nothing
    end
    Lr = C.L
    Lrinv = Lr \ I(length(d))
    return d .* Lr, Lrinv ./ d'
end

# ------------------------------------------------------------------------------------------------ solver
mutable struct Solver
    p::Params
    nu::Numerics
    R::ReadIn
    H::Hidden
    φ::Matrix{Float64}
    Φw::Matrix{Float64}
    wx::Vector{Float64}
    x::Vector{Float64}
    t::Vector{Float64}
    ell::Vector{Float64}
    ri_g::Vector{Int}
    ri_m::Int
    y::Vector{Float64}
    nbuild::Int
    npass::Int
    res::Float64
    last::NamedTuple
    verbose::Bool
    freeze::Bool          # reuse the hidden-field frame (u, ζ rule) inside one Newton
    frame_ready::Bool
    cache::Vector{Float64}
end

function Solver(p::Params, nu::Numerics; verbose=true)
    R = build_readin(p, nu)
    xg, wg = gausshermite_e(nu.nx)
    wx = wg ./ sqrt(2π)
    φ = hermite_phi(p.ks, xg)
    Φw = φ .* wx'
    H = Hidden()
    init_s_grid!(H, nu)
    S = nS(p)
    # prior state
    V0 = zeros(S, S)
    ri0 = readin_moments(R, p, nu, V0, 0.0, nothing, 0)
    st0 = state_from(ri0, p)
    x0 = encode_state(st0...)
    return Solver(p, nu, R, H, φ, Φw, wx, x0, zeros(S), Float64[],
        zeros(Int, length(R.base)), 0, target_y(p), 0, 0, Inf,
        (t=zeros(S), K=zeros(S, S), Hm=zeros(S, S), V=V0, qbar=st0[2], M=st0[3],
            Eq=st0[2], w0sq=ri0.w0sq, qmean=ri0.qmean, J=st0[1],
            readin_delta=Float64[], hidden_delta=Float64[], edge=0.0, res=Inf),
        verbose, false, false, Float64[])
end

function prior_state(S::Solver)
    V0 = zeros(nS(S.p), nS(S.p))
    ri0 = readin_moments(S.R, S.p, S.nu, V0, 0.0, nothing, 0)
    return state_from(ri0, S.p)
end

# ------------------------------------------------------------------------------------------------ forward residual
function split_ell(S::Solver, ell)
    mr = S.ri_m
    return ell[1:mr], ell[mr + 1:end]
end

"""Map the encoded read-in state (and logits) to the residual of (E6)–(E8) and (E4)'s fixed point
is taken inside solve_t, so the residual in t is zero by construction."""
function forward(S::Solver, x, ell)
    p = S.p
    J, qbar, M = decode_state(x, nS(p))
    ch = scaled_cholesky(p.sigB^2 * J)
    ch === nothing && return nothing
    L, Linv = ch
    mr = S.ri_m
    ell_r = ell[1:mr]
    ell_h = ell[mr + 1:end]
    # stash hidden logits for newton_t!
    S.ell = ell
    t_st = solve_t_with_ell!(S, L, Linv, x, qbar, M, S.t, ell_h)
    if t_st === nothing
        return nothing
    end
    t, st = t_st
    mcoef = M * (qbar - st.Eq) / qbar^2
    ri = readin_moments(S.R, p, S.nu, st.V, mcoef, S.ri_g, S.ri_m;
        ell=mr == 0 ? nothing : ell_r, central=S.nu.central)
    new = state_from(ri, p)
    if new === nothing
        S.verbose && @printf("  read-in state rejected  E||Φ||²=%.3e  trJ=%.3e  EJ2=%.3e\n",
            ri.EJxx, tr(ri.J), ri.EJ2)
        return nothing
    end
    Jn, qn, Mn = new
    Hmat = L * st.Ezz * L'
    S.t = t
    S.last = (t=t, K=st.K, Hm=Hmat, V=st.V, qbar=qbar, M=M, Eq=st.Eq,
        w0sq=ri.w0sq, qmean=ri.qmean, J=Jn, readin_delta=ri.delta,
        hidden_delta=st.delta, edge=st.edge, res=0.0,
        action_r=ri.action, logmass_r=ri.logmass, Lc=st.Lc, mcoef=mcoef)
    Rx = encode_state(Jn, qn, Mn) .- x
    Rell = vcat(ri.delta, st.delta) .- ell
    return vcat(Rx, Rell)
end

function ensure_features!(S, L, x, qbar, M)
    H = S.H
    if S.freeze && S.frame_ready && length(S.cache) == length(x) && S.cache == x
        return
    end
    if S.freeze && S.frame_ready
        rebind!(H, L, qbar, M, S.φ, S.nu)
    else
        tdir = norm(S.t) == 0 ? S.y : S.t
        setup_frame!(H, L, tdir, qbar, M, S.φ, S.wx, S.nu)
        S.frame_ready = true
    end
    build_features!(H, S.φ, S.Φw)
    S.nbuild += 1
    S.cache = copy(x)
    return
end

function solve_t_with_ell!(S, L, Linv, x, qbar, M, t0, ell_h)
    t = copy(t0)
    norm(t) == 0 && (t = copy(S.y))
    ensure_features!(S, L, x, qbar, M)
    return _newton_t(S, Linv, t, ell_h)
end

function readout_value(t, st, λ, y, pref)
    return 0.5 * λ * dot(t, t) - dot(y, t) + pref * logsumexp(vec(st.Lc))
end

function _newton_t(S, Linv, t, ell_h)
    p = S.p
    y, gt, λ = S.y, gtilde(p), lam(p)
    pref = p.sigA^2 / gt
    use_ell = S.H.m > 0 && ell_h !== nothing && length(ell_h) == S.H.m
    ell = use_ell ? ell_h : nothing
    st = hidden_moments(S.H, t, gt, Linv; ell=ell, central=S.nu.central)
    S.npass += 1
    IS = I(length(t))
    for _ in 1:40
        g = λ .* t .+ p.sigA^2 .* (st.K * t) .- y
        ng = norm(g)
        ng < S.nu.newton_tol && return t, st
        Hm = λ .* IS .+ p.sigA^2 .* (st.K .+ gt .* st.Cov)
        Hm = (Hm + Hm') ./ 2
        step = try
            -(cholesky(Symmetric(Hm)) \ g)
        catch
            try
                -(cholesky(Symmetric(λ .* IS .+ p.sigA^2 .* st.K)) \ g)
            catch
                -g
            end
        end
        slope = dot(g, step)
        if slope >= 0
            step = -g
            slope = -dot(g, g)
        end
        # A capped step still has to decrease F. ||g|| need not fall on a partial step.
        ns = norm(step)
        ns > 8 * (norm(t) + 1) && (step .*= 8 * (norm(t) + 1) / ns)
        F0 = readout_value(t, st, λ, y, pref)
        a = 1.0
        tn, stn = t, st
        accepted = false
        while a >= 1e-8
            tn = t .+ a .* step
            stn = hidden_moments(S.H, tn, gt, Linv; ell=ell, central=S.nu.central)
            S.npass += 1
            Fn = readout_value(tn, stn, λ, y, pref)
            if Fn <= F0 + 1e-4 * a * slope
                accepted = true
                break
            end
            a *= 0.5
        end
        if !accepted
            S.verbose && @printf("  readout line search failed |g|=%.3e  |t|=%.3e  edge=%.1f\n",
                ng, norm(t), st.edge)
            return nothing
        end
        t, st = tn, stn
    end
    g = λ .* t .+ p.sigA^2 .* (st.K * t) .- y
    ng = norm(g)
    if S.verbose && ng >= 1e-6
        @printf("  readout Newton unfinished |g|=%.3e  |t|=%.3e  edge=%.1f\n", ng, norm(t), st.edge)
    end
    return ng < 1e-6 ? (t, st) : nothing
end

"""Label basins from a true-weight (no logit) pass, and return the initial logits."""
function refresh_groups!(S::Solver, x)
    saved_m, saved_rm = S.H.m, S.ri_m
    saved_g, saved_rg = copy(S.H.g), copy(S.ri_g)
    S.H.m = 0
    S.ri_m = 0
    R = forward(S, x, Float64[])
    # restore if the probe failed
    if R === nothing
        S.H.m, S.ri_m = saved_m, saved_rm
        S.H.g, S.ri_g = saved_g, saved_rg
        return nothing
    end
    lt = S.last
    if S.nu.basins && !S.nu.central
        if S.nu.hidden_basins
            Ag = -lt.Lc
            g, m = make_groups(Ag, lt.Lc, 1 + (argmin(abs.(log.(S.H.qh) .- log(S.H.qbar))) - 1) * length(S.H.s),
                S.nu.basin_floor)
            S.H.g, S.H.m = g, m
        else
            S.H.g .= 0
            S.H.m = 0
        end
        Ar = lt.action_r
        gr, mr = make_readin_groups(Ar, lt.logmass_r, S.R.w, S.R.seed, S.nu.basin_floor, 0.5)
        S.ri_g, S.ri_m = gr, mr
    else
        S.H.g .= 0
        S.H.m = 0
        S.ri_g .= 0
        S.ri_m = 0
    end
    # deltas of the TRUE distribution (the probe used no logits)
    ell = vcat(lt.readin_delta, lt.hidden_delta)
    # the probe was run with m = 0, so those deltas are empty. Recompute from the labelled groups
    # by one more weighted pass if any basin was found — the unweighted deltas are the log mass ratios.
    if S.H.m + S.ri_m > 0
        # mass ratios from the probe's log mass, which is the full distribution
        ell = vcat(basin_deltas(lt.logmass_r, S.ri_g, S.ri_m), basin_deltas(lt.Lc, S.H.g, S.H.m))
    else
        ell = Float64[]
    end
    return ell
end

function basin_deltas(logmass, g, m)
    m == 0 && return Float64[]
    l0 = group_lse_g(logmass, g, 0)
    return [group_lse_g(logmass, g, k) - l0 for k in 1:m]
end

struct NoConvergence <: Exception
    msg::String
end

function newton!(S::Solver, x, ell)
    u = vcat(x, ell)
    S.freeze = false
    R = forward(S, x, ell)
    R === nothing && throw(NoConvergence("singular at the start"))
    S.freeze = true                 # one smooth frame for the whole Newton
    rinf = maximum(abs, R)
    rinf < S.nu.outer_tol && return u, R, 0
    Jm = fd_jacobian(S, u, R)
    refreshes = 0
    r0 = rinf
    best = rinf
    stall = 0
    for it in 1:40
        rinf = maximum(abs, R)
        rinf < S.nu.outer_tol && return u, R, it
        # He3 (and higher) make the residual ill-conditioned, so a real step can take
        # many iterations. Give up only when it has stopped falling.
        if rinf < 0.85 * best
            best = rinf
            stall = 0
        else
            stall += 1
            if stall >= 8 && it >= 10 && rinf > 1e-4
                throw(NoConvergence(@sprintf("Newton not descending at |R|∞ = %.2e", rinf)))
            end
        end
        # Finite-difference noise sits near 1e-8. A residual already there is a root.
        if it >= 2 && rinf < 1e-6 && rinf > 0.5 * r0
            return u, R, it
        end
        r = norm(R)
        step = try
            -(Jm \ R)
        catch
            -(pinv(Jm) * R)
        end
        # Long steps are fine — Armijo shortens them. Only stop a step that left the domain of Float64.
        ns = norm(step)
        ns > 80 && (step .*= 80 / ns)
        a = 1.0
        un, Rn = u, R
        ok = false
        while a >= 1e-4
            un = u .+ a .* step
            Rn = forward(S, un[1:length(x)], un[length(x) + 1:end])
            if Rn !== nothing && length(Rn) == length(R) && norm(Rn) < (1 - 1e-4 * a) * r
                ok = true
                break
            end
            a *= 0.5
        end
        if !ok && rinf < 1.0
            # J33 sits near 1/d^3 here, so the Newton step is not a descent direction even
            # though the root is nearby. A damped least-squares step still descends.
            n = length(u)
            for μ in (1e-6, 1e-4, 1e-2, 1.0, 1e2)
                step = try
                    -((Jm' * Jm + μ * I(n)) \ (Jm' * R))
                catch
                    continue
                end
                ns = norm(step)
                ns > 5 && (step .*= 5 / ns)
                a = 1.0
                while a >= 1e-3
                    un = u .+ a .* step
                    Rn = forward(S, un[1:length(x)], un[length(x) + 1:end])
                    if Rn !== nothing && length(Rn) == length(R) && norm(Rn) < (1 - 1e-4 * a) * r
                        ok = true
                        break
                    end
                    a *= 0.5
                end
                ok && break
            end
        end
        if !ok
            rinf < 1e-6 && return u, R, it
            # The Jacobian was just built. A failed line search at large residual means the
            # continuation step left the basin; rebuilding it does not put it back.
            (rinf > 1.0 || refreshes >= 1) &&
                throw(NoConvergence(@sprintf("line search failed at |R|∞ = %.2e", rinf)))
            refreshes += 1
            Jm = fd_jacobian(S, u, R)
            Rn0 = forward(S, u[1:length(x)], u[length(x) + 1:end])
            Rn0 === nothing && throw(NoConvergence("lost the point during Jacobian refresh"))
            R = Rn0
            continue
        end
        sv = a .* step
        den = dot(sv, sv)
        Jm = Jm .+ (Rn .- R .- Jm * sv) * (sv' ./ den)
        u, R = un, Rn
        x = u[1:length(x)]
        # t is solved inside the forward. If it has rotated off the frozen frame, the
        # features are in the wrong direction and the reduced residual has no root.
        if drive_misaligned(S, x)
            S.freeze = false
            Rf = forward(S, x, u[length(x) + 1:end])
            Rf === nothing && throw(NoConvergence("frame rebuild left the domain"))
            S.freeze = true
            R = Rf
            Jm = fd_jacobian(S, u, R)
        end
    end
    maximum(abs, R) < 1e-5 && return u, R, 40
    throw(NoConvergence(@sprintf("Newton stalled at |R| = %.2e  R=%s",
        maximum(abs, R), string(round.(R, sigdigits=4)))))
end

function fd_jacobian(S, u, R0)
    nx = nstate(nS(S.p))
    Jm = zeros(length(R0), length(u))
    for j in eachindex(u)
        # 1e-6 is below the quadrature noise once a well is deep; the column is then pure roundoff.
        h = 1e-4 * (1 + abs(u[j]))
        up = copy(u)
        up[j] += h
        Rp = forward(S, up[1:nx], up[nx + 1:end])
        if Rp === nothing || length(Rp) != length(R0)
            up[j] -= 2h
            Rp = forward(S, up[1:nx], up[nx + 1:end])
            h = -h
        end
        (Rp === nothing || length(Rp) != length(R0)) && throw(NoConvergence("Jacobian probe left the domain"))
        Jm[:, j] .= (Rp .- R0) ./ h
    end
    return Jm
end

function solve_at!(S::Solver, x; force_cap=0.0, depth=0, logit_init=:keep)
    adapt_s!(S, x, force_cap)
    S.freeze = false
    ell_true = refresh_groups!(S, x)
    ell_true === nothing && throw(NoConvergence("group probe failed"))
    # A brand-new far minimum starts no heavier than the central one. Newton may
    # leave that interval. An existing logit is kept: replacing it by the true
    # mass ratio jumps the weights onto whichever well currently dominates.
    if logit_init == :zero
        ell = zeros(length(ell_true))
    elseif length(ell_true) == length(S.ell)
        ell = copy(S.ell)
    else
        ell = clamp.(ell_true, S.nu.basin_floor, 0.0)
    end
    u, R, it = newton!(S, x, ell)
    nx = nstate(nS(S.p))
    x1 = u[1:nx]
    ell1 = u[nx + 1:end]
    # The frame was frozen at the incoming drive. If Newton rotated t, rebuild once.
    if drive_misaligned(S, x1)
        S.freeze = false
        u, R, it = newton!(S, x1, ell1)
        x1 = u[1:nx]
        ell1 = u[nx + 1:end]
    end
    if depth < 2 && isfinite(S.last.edge) && S.last.edge > -30 && S.H.s[end] < 1800
        cap = S.H.s[end] + max(8.0, 0.5 * S.H.s[end])
        if S.verbose
            @printf("  hidden mass at s=%.1f (edge %.1f); extending the s-grid\n", S.H.s[end], S.last.edge)
            flush(stdout)
        end
        S.ell = ell1
        return solve_at!(S, x1; force_cap=cap, depth=depth + 1)
    end
    if depth < 3 && readin_cramped(S) && S.nu.wmax < 40
        extend_readin!(S) || (S.res = maximum(abs, R); return x1, ell1, S.res, it)
        S.ell = ell1
        return solve_at!(S, x1; force_cap=force_cap, depth=depth + 1)
    end
    S.res = maximum(abs, R)
    return x1, ell1, S.res, it
end

function drive_misaligned(S::Solver, x)
    J, _, _ = decode_state(x, nS(S.p))
    ch = scaled_cholesky(S.p.sigB^2 * J)
    ch === nothing && return false
    v = ch[1]' * S.t
    nv = norm(v)
    nv == 0 && return false
    length(S.H.u) == length(v) || return true
    return dot(S.H.u, v ./ nv) < 1 - 1e-4
end

function log_state(S, tag, res, it, t0)
    S.verbose || return
    lt = S.last
    f = S.p.sigA^2 .* (lt.K * lt.t)
    L1 = f[1]
    L3 = nS(S.p) > 1 ? f[2] / S.p.eps : NaN
    @printf("%s gt=%.3g rB=%.3g |R|=%.1e it=%d L1=%.4f L3=%.4f d<w0²>=%.3f basins %d+%d s*=%.2f/%.1f n=%d edge=%.1f wedge=%.1f builds=%d %.1fs\n",
        tag, gtilde(S.p), rB(S.p), res, it, L1, L3, S.p.d * lt.w0sq, S.ri_m, S.H.m,
        s_mode(S), S.H.s[end], length(S.H.s), lt.edge, readin_w_edge(S), S.nbuild, time() - t0)
    flush(stdout)
end

"""Continue a positive parameter from its current value up to `v1`, in log steps.
`x` is a converged state at the current value. The setter writes the trial value onto `S.p`."""
function march!(S, x, t0, tag, v0, v1, setv!; give_up=100)
    v0 >= v1 - 1e-14 * max(v1, 1.0) && return x
    lv = log(v0)
    lt = log(v1)
    step = log(sqrt(2))
    have = false
    predicted = true
    fails = 0
    last_it = 0
    x_prev = copy(x)
    t_prev = copy(S.t)
    lv_prev = lv
    while lv < lt - 1e-12
        nbasin = S.ri_m + S.H.m
        cap = nbasin == 0 ? log(2.0) : (last_it <= 4 ? log(1.5) : log(1.12))
        step = min(step, cap)
        nxt = min(lv + step, lt)
        x_at = copy(x)
        t_at = copy(S.t)
        ell_at = copy(S.ell)
        x_guess = x
        if have && predicted && lv > lv_prev + 1e-15
            α = (nxt - lv) / (lv - lv_prev)
            dx = x .- x_prev
            nd = abs(α) * norm(dx)
            nd > 5 && (α *= 5 / nd)
            x_guess = x .+ α .* dx
            dt = S.t .- t_prev
            S.t = S.t .+ α .* dt
        end
        setv!(exp(nxt))
        try
            x_new, ell_new, res, it = solve_at!(S, x_guess)
            x_prev = x_at
            t_prev = t_at
            lv_prev = lv
            x = x_new
            S.x = x
            S.ell = ell_new
            lv = nxt
            have = true
            predicted = true
            fails = 0
            last_it = it
            step = min(step * 1.35, cap)
            log_state(S, @sprintf("%s=%.4g", tag, exp(lv)), res, it, t0)
        catch e
            e isa NoConvergence || rethrow()
            setv!(exp(lv))
            x = x_at
            S.x = x
            S.t = t_at
            S.ell = ell_at
            predicted = false
            if S.verbose
                println("$(tag)=$(round(exp(nxt), sigdigits=4)): $(e.msg); halving the step")
                flush(stdout)
            end
            step /= 2
            fails += 1
            if fails >= give_up || step < 1e-3
                throw(NoConvergence("$(tag) continuation stalled near $(round(exp(lv), sigdigits=4)): $(e.msg)"))
            end
        end
    end
    return x
end

function continue_chi!(S, chi_t, t0; chi_start=nothing, give_up=100)
    p = S.p
    chi_lo = 1e-4 * p.N2 / p.sigA^2
    chi0 = something(chi_start, min(chi_t / 64, chi_lo))
    chi0 = clamp(chi0, 1e-6, chi_t)
    p.chi = chi0
    x = copy(S.x)
    x, ell, res, it = solve_at!(S, x)
    S.x = x
    S.ell = ell
    log_state(S, @sprintf("chi=%.4g", p.chi), res, it, t0)
    return march!(S, x, t0, "chi", chi0, chi_t, v -> (p.chi = v); give_up)
end

function continue_rscale!(S, x, scale_t, t0)
    p = S.p
    v = min(1e-3, scale_t)
    x_at, t_at, ell_at = copy(x), copy(S.t), copy(S.ell)
    while true
        p.rscale = v
        try
            x, ell, res, it = solve_at!(S, x_at)
            S.x = x
            S.ell = ell
            log_state(S, @sprintf("rscale=%.4g", v), res, it, t0)
            break
        catch e
            e isa NoConvergence || rethrow()
            S.x = x_at
            S.t = t_at
            S.ell = ell_at
            if S.verbose
                println("rscale=$(round(v, sigdigits=3)): $(e.msg); halving")
                flush(stdout)
            end
            v /= 2
            v < 1e-8 && throw(NoConvergence("read-in coupling does not turn on: $(e.msg)"))
        end
    end
    return march!(S, x, t0, "rscale", v, scale_t, z -> (p.rscale = z); give_up=1)
end

"""Zero one encoded coordinate by secant, holding the others fixed. Used when the
joint Newton step is not a descent but a single variable (usually log M) still is."""
function secant_component!(S, x, comp)
    S.freeze = false
    S.H.m = 0
    S.ri_m = 0
    probe(z) = forward(S, z, Float64[])
    R = probe(x)
    R === nothing && return x, Inf
    x0 = copy(x)
    a = x0[comp]
    fa = R[comp]
    bestx, bestr = copy(x0), maximum(abs, R)
    b = a - fa
    xb = copy(x0)
    xb[comp] = b
    Rb = probe(xb)
    Rb === nothing && return bestx, bestr
    fb = Rb[comp]
    if maximum(abs, Rb) < bestr
        bestx, bestr = copy(xb), maximum(abs, Rb)
    end
    for _ in 1:8
        abs(fb) < 1e-6 && return xb, maximum(abs, Rb)
        den = fb - fa
        abs(den) < 1e-14 && break
        c = b - fb * (b - a) / den
        c = clamp(c, min(a, b) - 3, max(a, b) + 3)
        a, fa, b = b, fb, c
        xb = copy(x0)
        xb[comp] = c
        Rb = probe(xb)
        Rb === nothing && break
        fb = Rb[comp]
        r = maximum(abs, Rb)
        if r < bestr
            bestx, bestr = copy(xb), r
        end
    end
    return bestx, bestr
end

"""Past a fold the continuation cannot move. Jump r_B to its target and damp
x toward the image of the full Boltzmann map, which is where the outer well lives."""
function damp_to!(S, x, t0)
    p = S.p
    for it in 1:25
        adapt_s!(S, x, 0.0)
        S.freeze = false
        S.H.m = 0
        S.ri_m = 0
        R = forward(S, x, Float64[])
        R === nothing && throw(NoConvergence("damping probe left the domain"))
        rinf = maximum(abs, R)
        S.res = rinf
        S.x = x
        log_state(S, @sprintf("damp rscale=%.4g", p.rscale), rinf, it, t0)
        rinf < S.nu.outer_tol && return x
        if it <= 6 && readin_cramped(S) && extend_readin!(S)
            continue
        end
        if it <= 3 && isfinite(S.last.edge) && S.last.edge > -30 && S.H.s[end] < 1800
            adapt_s!(S, x, S.H.s[end] + max(8.0, 0.5 * S.H.s[end]))
            continue
        end
        r0 = norm(R)
        a = 1.0
        moved = false
        while a >= 1e-3
            xt = x .+ a .* R
            S.freeze = false
            R2 = forward(S, xt, Float64[])
            if R2 !== nothing && norm(R2) < (1 - 0.05 * a) * r0
                x = xt
                S.x = x
                moved = true
                break
            end
            a *= 0.5
        end
        if !moved
            if S.verbose
                @printf("  Picard stuck, R = %s\n", string(round.(R, sigdigits=3)))
                flush(stdout)
            end
            for comp in sortperm(abs.(R), rev=true)
                x, rinf = secant_component!(S, x, comp)
                S.x = x
                S.res = rinf
                log_state(S, @sprintf("secant[%d] rscale=%.4g", comp, p.rscale), rinf, comp, t0)
                rinf < S.nu.outer_tol && return x
            end
            try
                x, ell, res, itn = solve_at!(S, x)
                S.x = x
                S.ell = ell
                log_state(S, @sprintf("damp-newton rscale=%.4g", p.rscale), res, itn, t0)
                return x
            catch e
                e isa NoConvergence || rethrow()
                throw(NoConvergence(@sprintf("damping stopped at |R|∞ = %.2e", S.res)))
            end
        end
    end
    throw(NoConvergence("damping did not reach the root"))
end

"""Newton on the read-in state at a frozen logit. The basin masses stay fixed, so the
map is the smooth reweighting of the one measure; the logit is not a residual here."""
function newton_state!(S::Solver, x, ell)
    nx = length(x)
    S.freeze = false
    Rfull = forward(S, x, ell)
    Rfull === nothing && throw(NoConvergence("singular at the start of the inner solve"))
    S.freeze = true
    R = Rfull[1:nx]
    rinf = maximum(abs, R)
    rinf < S.nu.outer_tol && return x, R, 0
    Jm = zeros(nx, nx)
    for j in 1:nx
        h = 1e-4 * (1 + abs(x[j]))
        xp = copy(x)
        xp[j] += h
        Rp = forward(S, xp, ell)
        if Rp === nothing
            xp[j] -= 2h
            Rp = forward(S, xp, ell)
            h = -h
        end
        Rp === nothing && throw(NoConvergence("inner Jacobian left the domain"))
        Jm[:, j] .= (Rp[1:nx] .- R) ./ h
    end
    for it in 1:10
        rinf = maximum(abs, R)
        rinf < 1e-5 && return x, R, it
        step = try
            -(Jm \ R)
        catch
            -(pinv(Jm) * R)
        end
        ns = norm(step)
        ns > 40 && (step .*= 40 / ns)
        r = norm(R)
        a = 1.0
        un, Rn = x, R
        ok = false
        while a >= 1e-3
            un = x .+ a .* step
            Rf = forward(S, un, ell)
            if Rf !== nothing && norm(Rf[1:nx]) < (1 - 1e-4 * a) * r
                Rn = Rf[1:nx]
                ok = true
                break
            end
            a *= 0.5
        end
        if !ok
            rinf < 1e-4 && return x, R, it
            throw(NoConvergence(@sprintf("inner line search failed at |Rx|∞ = %.2e", rinf)))
        end
        sv = a .* step
        den = dot(sv, sv)
        Jm = Jm .+ (Rn .- R .- Jm * sv) * (sv' ./ den)
        x, R = un, Rn
    end
    maximum(abs, R) < 1e-4 && return x, R, 10
    throw(NoConvergence(@sprintf("inner solve stalled at |Rx| = %.2e", maximum(abs, R))))
end

"""Move the read-in logit until it equals the true log-mass ratio of the one measure.
Each trial freezes the logit, solves the state, then reads the true ratio at that state."""
function match_mass!(S::Solver, x, t0)
    nx = nstate(nS(S.p))
    for _ in 1:3
        adapt_s!(S, x, 0.0)
        S.freeze = false
        S.H.m = 0
        S.ri_m = 0
        S.ell = Float64[]
        refresh_groups!(S, x) === nothing && throw(NoConvergence("group probe failed during the mass match"))
        S.ri_m == 0 && break
        try
            x, _, _ = newton_state!(S, x, zeros(S.ri_m))
        catch err
            err isa NoConvergence || rethrow()
            S.verbose && (println("  mass  grid probe at ell=0 failed: $(err.msg)"); flush(stdout))
        end
        grown = false
        if readin_cramped(S) && S.nu.wmax < 40
            extend_readin!(S)
            grown = true
        end
        if isfinite(S.last.edge) && S.last.edge > -30 && S.H.s[end] < 1800
            adapt_s!(S, x, S.H.s[end] + max(8.0, 0.5 * S.H.s[end]))
            grown = true
        end
        grown || break
    end
    S.freeze = false
    S.H.m = 0
    S.ri_m = 0
    S.ell = Float64[]
    refresh_groups!(S, x) === nothing && throw(NoConvergence("group probe failed during the mass match"))
    if S.ri_m != 1 || S.H.m != 0
        println("mass match: $(S.ri_m) read-in and $(S.H.m) hidden far minima; joint Newton")
        flush(stdout)
        return solve_at!(S, x; logit_init=:zero)
    end
    xcur = Ref(x)
    best = Ref((gap=Inf, e=0.0, x=copy(x), rx=Inf, delta=0.0))
    function ev(e)
        xn, R, it = newton_state!(S, xcur[], [e])
        xcur[] = xn
        Rf = forward(S, xn, [e])
        Rf === nothing && throw(NoConvergence("forward failed during the mass match"))
        δ = Rf[nx + 1] + e
        gap = δ - e
        rx = maximum(abs, Rf[1:nx])
        if abs(gap) < abs(best[].gap)
            best[] = (gap=gap, e=e, x=copy(xn), rx=rx, delta=δ)
        end
        if S.verbose
            @printf("  mass  ell=%.3f  δ=%.3f  gap=%.3e  |Rx|=%.2e  d⟨w0²⟩=%.3f  it=%d  %.0fs\n",
                e, δ, gap, rx, S.p.d * S.last.w0sq, it, time() - t0)
            flush(stdout)
        end
        return gap
    end
    # gap > 0: the true outer mass is heavier than the logit. The two ends of this
    # range sit on opposite sides of ell = δ for the χ=N well.
    probes = [-12.0, -6.0, -3.0, -1.0, 0.0, 1.5, 3.0]
    gaps = Float64[]
    es = Float64[]
    for e in probes
        g = try
            ev(e)
        catch err
            err isa NoConvergence || rethrow()
            S.verbose && (println("  mass  ell=$(round(e, digits=2)) failed: $(err.msg)"); flush(stdout))
            NaN
        end
        push!(es, e)
        push!(gaps, g)
    end
    lo = hi = NaN
    gL = gH = NaN
    for i in 1:length(es) - 1
        a, b = gaps[i], gaps[i + 1]
        (isnan(a) || isnan(b)) && continue
        if a == 0
            lo, hi, gL, gH = es[i], es[i], a, a
            break
        end
        if a * b < 0
            lo, hi, gL, gH = es[i], es[i + 1], a, b
            break
        end
    end
    if isnan(lo)
        last = findlast(isfinite, gaps)
        extra = last === nothing || gaps[last] > 0 ? [5.0, 8.0, 12.0] : [-20.0, -30.0, -40.0]
        prev = last === nothing ? NaN : gaps[last]
        preve = last === nothing ? es[end] : es[last]
        for e in extra
            g = try
                ev(e)
            catch err
                err isa NoConvergence || rethrow()
                NaN
            end
            if isfinite(prev) && isfinite(g) && prev * g < 0
                lo, hi, gL, gH = preve, e, prev, g
                break
            end
            if isfinite(g)
                prev, preve = g, e
            end
        end
    end
    if isnan(lo)
        throw(NoConvergence(@sprintf("mass match found no sign change; best |gap| = %.3e at ell = %.3f",
            best[].gap, best[].e)))
    end
    for _ in 1:18
        abs(hi - lo) < 1e-3 && abs(gL) < 1e-3 && break
        mid = 0.5 * (lo + hi)
        gm = ev(mid)
        if gL * gm <= 0
            hi, gH = mid, gm
        else
            lo, gL = mid, gm
        end
    end
    e = abs(gL) <= abs(gH) ? lo : hi
    x, _, it = newton_state!(S, xcur[], [e])
    # Joint polish. If the finite-difference step is wider than the layer, keep the
    # matched point: there |Rx| is small and the logit equals the true mass ratio.
    S.ell = [e]
    local x1, ell1, res, itj
    try
        u, Rf, itj = newton!(S, x, [e])
        x1 = u[1:nx]
        ell1 = u[nx + 1:end]
        res = maximum(abs, Rf)
    catch err
        err isa NoConvergence || rethrow()
        Rf = forward(S, x, [e])
        Rf === nothing && rethrow()
        x1, ell1, res, itj = x, [e], maximum(abs, Rf), 0
        S.verbose && (println("  mass  polish skipped: $(err.msg)"); flush(stdout))
    end
    S.res = res
    S.x = x1
    S.ell = ell1
    return x1, ell1, res, it + itj
end

"""Residual of the one Boltzmann measure: encode(moments) − x, with no basin reweighting."""
function boltzmann_residual(S::Solver, x)
    S.freeze = false
    S.H.m = 0
    S.ri_m = 0
    S.ell = Float64[]
    return forward(S, x, Float64[])
end

"""Grow the hidden mesh until the readout Newton at this state sits inside it.

At χ = N₂ the data term along the prior can peak far from the origin. A mesh that
ends on that peak truncates the integral, and the readout Newton then has no descent."""
function seat_grid!(S::Solver, x)
    R = nothing
    for _ in 1:8
        adapt_s!(S, x, 0.0)
        S.t .= 0
        R = boltzmann_residual(S, x)
        R !== nothing && S.last.edge < -40 && return R
        cap = max(S.H.s[end] * 1.5, S.H.s[end] + 30)
        cap > 1800 && break
        adapt_s!(S, x, cap)
    end
    return R
end

"""Trust-region step on the one-measure residual. The Picard image at χ = N₂ sits in the
far read-in well; a full step there overshoots the fixed point, so the radius stays finite."""
function lm_boltzmann!(S::Solver, x, R, t0; maxit=50)
    nx = length(x)
    μ = 1e-2
    bestx, bestr, bestR = copy(x), norm(R), copy(R)
    for it in 1:maxit
        rinf = maximum(abs, R)
        S.res = rinf
        S.x = x
        log_state(S, @sprintf("chi=%.4g", S.p.chi), rinf, it, t0)
        rinf < S.nu.outer_tol && return x
        Jm = zeros(length(R), nx)
        for j in 1:nx
            h = 1e-4 * (1 + abs(x[j]))
            xp = copy(x)
            xp[j] += h
            Rp = boltzmann_residual(S, xp)
            if Rp === nothing
                xp[j] -= 2h
                Rp = boltzmann_residual(S, xp)
                h = -h
            end
            Rp === nothing && throw(NoConvergence("Jacobian probe left the domain"))
            Jm[:, j] .= (Rp .- R) ./ h
        end
        # the probes moved S.t; put the residual back on x before the step
        R = boltzmann_residual(S, x)
        R === nothing && throw(NoConvergence("lost the base point"))
        accepted = false
        local xnew, Rnew
        for _ in 1:8
            step = try
                -((Jm' * Jm + μ * I(nx)) \ (Jm' * R))
            catch
                μ *= 10
                continue
            end
            ns = norm(step)
            ns > 0.75 && (step .*= 0.75 / ns)
            xnew = x .+ step
            Rnew = boltzmann_residual(S, xnew)
            if Rnew !== nothing && norm(Rnew) < norm(R)
                accepted = true
                break
            end
            μ *= 8
            μ > 1e6 && break
        end
        if !accepted
            norm(bestR) < 10 * S.nu.outer_tol && break
            # a pure short Picard step, in case the Jacobian averaged over a steep face
            a = 0.05
            moved = false
            r0 = norm(R)
            while a >= 1e-4
                xt = x .+ a .* R
                Rt = boltzmann_residual(S, xt)
                if Rt !== nothing && norm(Rt) < r0
                    x, R = xt, Rt
                    moved = true
                    break
                end
                a *= 0.5
            end
            moved || break
        else
            x, R = xnew, Rnew
            μ = max(μ / 4, 1e-6)
        end
        if norm(R) < norm(bestR)
            bestx, bestR = copy(x), copy(R)
            bestr = norm(bestR)
        end
    end
    Rb = boltzmann_residual(S, bestx)
    if Rb !== nothing
        S.res = maximum(abs, Rb)
        S.x = bestx
        log_state(S, @sprintf("chi=%.4g", S.p.chi), S.res, maxit, t0)
    end
    return bestx
end

"""Fixed point of the full integrals at the χ already stored on `S`. No continuation."""
function fixed_chi!(S::Solver, t0)
    S.nu.basins = false
    S.nu.hidden_basins = false
    S.nu.outer_tol = 1e-6
    x = copy(S.x)
    println("fixed χ=$(S.p.chi): one integral for each layer, g̃=$(round(gtilde(S.p), sigdigits=4)), r_B=$(round(rB(S.p), sigdigits=4))")
    flush(stdout)
    R0 = seat_grid!(S, x)
    if R0 === nothing
        throw(NoConvergence("the Boltzmann integral at the prior did not fit on the grid"))
    end
    for _ in 1:4
        readin_cramped(S) || break
        extend_readin!(S) || break
        R0 = boltzmann_residual(S, x)
        R0 === nothing && throw(NoConvergence("read-in grid extension left the domain"))
    end
    rinf0 = maximum(abs, R0)
    S.res = rinf0
    S.x = x
    log_state(S, @sprintf("chi=%.4g prior", S.p.chi), rinf0, 0, t0)
    rinf0 < S.nu.outer_tol && return x
    return lm_boltzmann!(S, x, R0, t0)
end

function solve!(S::Solver; chi_start=nothing)
    t0 = time()
    p = S.p
    chi_t = p.chi
    scale_t = p.rscale
    gt_t = p.sigA^2 * chi_t / p.N2
    # g̃ = O(1) is the phase being solved. χ stays there; both layers are one integral.
    feature = gt_t >= 0.05 && scale_t > 0 && !isinf(p.N1)
    if feature
        x = fixed_chi!(S, t0)
        S.x = x
        p.chi = chi_t
        p.rscale = scale_t
        return results(S, time() - t0)
    end
    continue_chi!(S, chi_t, t0; chi_start, give_up=100)
    p.chi = chi_t
    p.rscale = scale_t
    return results(S, time() - t0)
end

function results(S::Solver, wall)
    p, lt = S.p, S.last
    f = p.sigA^2 .* (lt.K * lt.t)
    g = λ_res = p.sigA^2 .* (lt.K * lt.t) .+ lam(p) .* lt.t .- S.y
    frac(ds) = [exp(d - logsumexp(vcat(0.0, ds))) for d in ds]
    return Dict(
        "converged" => S.res < 10 * S.nu.outer_tol,
        "walltime_s" => wall,
        "feature_builds" => S.nbuild,
        "reweights" => S.npass,
        "chi" => p.chi,
        "gtilde" => gtilde(p),
        "rB" => rB(p),
        "smax" => S.H.s[end],
        "ns" => length(S.H.s),
        "s_mode" => s_mode(S),
        "L1" => f[1],
        "L3" => nS(p) > 1 ? f[2] / p.eps : nothing,
        "d_w0sq" => p.d * lt.w0sq,
        "qperp_over_prior" => lt.qmean / S.R.q0,
        "qbar" => lt.qbar,
        "M" => lt.M,
        "qh_over_qbar" => lt.Eq / lt.qbar,
        "readout_residual" => norm(λ_res),
        "edge_logmass" => lt.edge,
        "readin_far_mass" => frac(lt.readin_delta),
        "hidden_far_mass" => frac(lt.hidden_delta),
        "J" => lt.J,
        "H" => lt.Hm,
        "K" => lt.K,
        "t" => lt.t,
        "fbar" => f,
    )
end

# ------------------------------------------------------------------------------------------------ checks
function check_quadrature(p, nu)
    println("quadrature and prior checks")
    x, w = gausshermite_e(20)
    wn = w ./ sum(w)
    @printf("  GH  E[x²]=%.6f (1)   E[x⁴]=%.6f (3)\n", dot(wn, x .^ 2), dot(wn, x .^ 4))
    xl, wl = gaussjacobi(4, 0.0, 0.0)
    @printf("  Legendre n=4 nodes %s\n", string(round.(xl, digits=6)))
    α = (p.d - 4) / 2
    uj, wj = gaussjacobi(nu.nu, α, α)
    @printf("  Jacobi α=%.1f  Σw=%.6e  E[u²]=%.6e\n", α, sum(wj), dot(wj, uj .^ 2) / sum(wj))
    c1 = cn_const(1)
    @printf("  c₁ const %.8f vs 2/√π %.8f\n", c1, 2 / sqrt(π))
    R = build_readin(p, nu)
    V0 = zeros(nS(p), nS(p))
    ri = readin_moments(R, p, nu, V0, 0.0, nothing, 0)
    st = state_from(ri, p)
    @printf("  prior E[w0²]=%.6e (σ²/d=%.6e)  E[q]/q0=%.6f\n",
        ri.w0sq, p.sigw^2 / p.d, ri.qmean / R.q0)
    @printf("  prior q̄=%.6f  M=%.2f  J11=%.6e  J33=%.6e  ρ13=%.4f\n",
        st[2], st[3], ri.J[1, 1], size(ri.J, 1) > 1 ? ri.J[2, 2] : NaN,
        size(ri.J, 1) > 1 ? ri.J[1, 2] / sqrt(ri.J[1, 1] * ri.J[2, 2]) : NaN)
    xg, wg = gausshermite_e(nu.nx)
    wx = wg ./ sqrt(2π)
    φ = hermite_phi(p.ks, xg)
    for (h1, qh) in ((0.3, 0.5), (1.2, 0.2))
        h = zeros(nS(p)); h[1] = h1
        e = en_at(h, qh, φ, wx)
        Tw = 1 + 2 * (h1^2 + qh)
        cerr = maximum(abs(e[i] - cn_const(p.ks[i]) * h1^p.ks[i] / Tw^(p.ks[i] / 2)) for i in 1:nS(p))
        @printf("  e_n(h1=%.2f, qh=%.2f) vs c_n  max |err|=%.2e\n", h1, qh, cerr)
    end
    J, q, M = st
    xr = encode_state(J, q, M)
    J2, q2, M2 = decode_state(xr, nS(p))
    @printf("  encode roundtrip |ΔJ|=%.2e |Δq̄|=%.2e |ΔM|=%.2e\n",
        norm(J2 - J), abs(q2 - q), abs(M2 - M))
    return maximum(abs, [ri.w0sq * p.d / p.sigw^2 - 1, ri.qmean / R.q0 - 1]) < 1e-3
end

# ------------------------------------------------------------------------------------------------ main
function report(r)
    @printf("\nχ = %.6g   g̃ = %.6g   r_B = %.6g\n", r["chi"], r["gtilde"], r["rB"])
    @printf("L1 = %.6f    L3 = %s\n", r["L1"], r["L3"] === nothing ? "—" : @sprintf("%.6f", r["L3"]))
    @printf("f̄ = %s\n", string(round.(r["fbar"], digits=5)))
    @printf("d⟨w0²⟩ = %.4f   ⟨q⟩/q0 = %.4f   ⟨qh⟩/q̄ = %.6f\n",
        r["d_w0sq"], r["qperp_over_prior"], r["qh_over_qbar"])
    @printf("s mode = %.3f   s grid ends at %.2f (%d nodes)\n", r["s_mode"], r["smax"], r["ns"])
    @printf("q̄ = %.6f   M = %.3f   readout |residual| = %.2e   edge log-mass = %.2f\n",
        r["qbar"], r["M"], r["readout_residual"], r["edge_logmass"])
    @printf("far-basin mass  read-in %s   hidden %s\n",
        string(round.(r["readin_far_mass"], sigdigits=3)),
        string(round.(r["hidden_far_mass"], sigdigits=3)))
    @printf("feature builds %d   reweights %d   %.2f s\n",
        r["feature_builds"], r["reweights"], r["walltime_s"])
    println("H =")
    show(stdout, "text/plain", r["H"])
    println()
end

function main()
    BLAS.set_num_threads(1)
    p, nu = make_params()
    if Threads.nthreads() == 1
        println("single-threaded; start with  julia -t 16  for the feature build")
    end
    if hasflag("check")
        ok = check_quadrature(p, nu)
        println(ok ? "checks passed" : "checks FAILED")
        ok || exit(1)
        hasflag("solve") || return
    end
    if hasflag("sweep")
        ks = parse.(Float64, split(argval("sweep", "0.1"), ","))
    else
        ks = [p.kappa]
    end
    S = Solver(p, nu; verbose=!hasflag("quiet"))
    out = Dict{Float64,Any}()
    for (i, k) in enumerate(ks)
        p.kappa = k
        println("\n=== d=$(p.d)  κ=$k  nmax=$(p.nmax)  χ=$(p.chi)  g̃=$(gtilde(p))  r_B=$(rB(p))  branch=$(nu.central ? "central" : "full") ===")
        if i == 1
            r = solve!(S)
        else
            # warm: try the current state at the new κ, otherwise continue again
            try
                t0 = time()
                x, ell, res, it = solve_at!(S, S.x)
                S.x, S.ell = x, ell
                log_state(S, "warm", res, it, t0)
                r = results(S, time() - t0)
            catch e
                e isa NoConvergence || rethrow()
                println("warm start failed ($(e.msg)); continuation")
                S.x = encode_state(prior_state(S)...)
                S.t .= 0
                r = solve!(S)
            end
        end
        # keep the solved state for the next κ
        S.x = encode_state(S.last.J, S.last.qbar, S.last.M)
        report(r)
        out[k] = r
    end
    if hasflag("out")
        path = argval("out", "fcn3_fast.json")
        # matrices to lists
        open(path, "w") do io
            println(io, "{")
            for (i, k) in enumerate(ks)
                r = out[k]
                print(io, "  \"$k\": {")
                print(io, "\"L1\": $(r["L1"]), \"L3\": $(something(r["L3"], "null")), ")
                print(io, "\"d_w0sq\": $(r["d_w0sq"]), \"M\": $(r["M"]), \"qbar\": $(r["qbar"]), ")
                print(io, "\"walltime_s\": $(r["walltime_s"])}")
                println(io, i == length(ks) ? "" : ",")
            end
            println(io, "}")
        end
        println("wrote $path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
