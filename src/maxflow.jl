"""
Phase 1 — Nonlinear AMG for the maximum-flow problem via the self-consistent
graph Laplacian L(f)·φ = d. See `doc/nonlinear_gauss_seidel.md` for the
derivation.

Edge model:
   capacities  c⁺_e, c⁻_e  ≥ 0     forward / reverse
   resistance  R_e(f) = 1/(c⁺_e − f)² + 1/(c⁻_e + f)²    (asymmetric barrier)
   conductance W_e(f) = 1 / R_e(f)
   Ohm's law   f_e = W_e(f_e) · (φ_tail − φ_head)        (implicit in f)
   conservation B·f = -d                                  (B is signed incidence)

Per-node Newton relaxation: at each node u, hold all neighbors fixed and take
one Newton step on the local residual r_u(φ_u) = 0, where

   r_u(φ_u) = Σ_{e ∋ u}  g_e(φ_u)  −  d_u
   g_e(φ_u) = signed flow "out of u" along e
            = W_e(g_e) · (φ_u − φ_v)     (implicit; 1D root find)

Linearization recovers exactly the linear-Laplacian GS update when far from
saturation; near saturation the barrier derivative dominates and keeps the
update inside the feasible interval.
"""

"""
    edge_resistance(c⁺::Real, c⁻::Real, f::Real) -> Float64

The asymmetric barrier resistance R_e(f) = 1/(c⁺−f)² + 1/(c⁻+f)².
"""
function edge_resistance(c⁺::Real, c⁻::Real, f::Real)
    @assert c⁺ >= 0 && c⁻ >= 0
    a = c⁺ - f
    b = c⁻ + f
    (a <= 0 || b <= 0) && return Inf
    return 1.0 / (a * a) + 1.0 / (b * b)
end

"""
    edge_conductance(c⁺, c⁻, f) -> Float64
"""
edge_conductance(c⁺::Real, c⁻::Real, f::Real) =
    1.0 / edge_resistance(c⁺, c⁻, f)

"""
    edge_resistance_derivative(c⁺, c⁻, f) -> Float64

R'_e(f) = 2/(c⁺−f)³ − 2/(c⁻+f)³. Needed for the local Newton derivative.
"""
function edge_resistance_derivative(c⁺::Real, c⁻::Real, f::Real)
    a = c⁺ - f
    b = c⁻ + f
    return 2.0 / (a * a * a) - 2.0 / (b * b * b)
end

"""
    solve_implicit_flow(c_fwd, c_rev, Δφ;
                        max_iter=30, tol=1e-12, f0=0.0) -> (f, R, R′)

Solve  f * R_u(f) = Δφ  for the "out-of-u" flow `f`, where

   R_u(f) = 1/(c_fwd − f)² + 1/(c_rev + f)²

and `c_fwd` / `c_rev` are the u-oriented forward / reverse capacities.
Returns `(f, R_u(f), R_u'(f))` for downstream Newton-derivative use.

Uses damped Newton starting from `f0` (default 0), clamped to the feasible
interval `(-c_rev, c_fwd)`.

Edge-local — does not depend on any global graph structure. Tabulating
this 1D solve in `(c_fwd, c_rev, Δφ)` is the "windowing" that makes the
nonlinear AMG O(m); see `PROJECT_PLAN.md` §2 and
`doc/nonlinear_gauss_seidel.md`.
"""
function solve_implicit_flow(c_fwd::Real, c_rev::Real, Δφ::Real;
                             max_iter::Int = 30, tol::Real = 1e-12,
                             f0::Real = 0.0)
    f = float(f0)
    margin = 1e-9
    f = clamp(f, -c_rev + margin, c_fwd - margin)
    for _ in 1:max_iter
        a = c_fwd - f
        b = c_rev + f
        Rf = 1.0 / (a * a) + 1.0 / (b * b)
        Rfp = 2.0 / (a * a * a) - 2.0 / (b * b * b)
        g = f * Rf - Δφ
        gp = Rf + f * Rfp
        abs(gp) < 1e-30 && break
        Δf = g / gp
        f_new = f - Δf
        # Damped step: if stepping leaves the feasible interval, halve.
        ν = 1.0
        while (f_new <= -c_rev + margin || f_new >= c_fwd - margin) && ν > 1e-12
            ν *= 0.5
            f_new = f - ν * Δf
        end
        f = f_new
        abs(Δf) < tol && break
    end
    a = c_fwd - f
    b = c_rev + f
    Rf = 1.0 / (a * a) + 1.0 / (b * b)
    Rfp = 2.0 / (a * a * a) - 2.0 / (b * b * b)
    return f, Rf, Rfp
end

"""
    NodeIncidence

Per-node incidence record. For node u and incident edge e with neighbor v:
   `e_idx`  :: global edge index
   `v`      :: the OTHER endpoint
   `σ`      :: +1 if u is the tail of e in B's reference orientation, -1 if head
   `c_fwd`  :: u-oriented forward capacity (= c⁺[e] if σ=+1, c⁻[e] if σ=-1)
   `c_rev`  :: u-oriented reverse capacity
"""
struct NodeIncidence
    e_idx::Int
    v::Int
    σ::Int8
    c_fwd::Float64
    c_rev::Float64
end

"""
    MaxFlowLaplacian(B, c_plus, c_minus, d; f0=nothing)

State holder for the nonlinear Laplacian L(f)·φ = d.

Fields:
- `B`        :: n × m signed incidence matrix (+1 at head, −1 at tail).
- `c_plus`   :: forward (head→tail-direction) capacity per edge.
- `c_minus`  :: reverse capacity per edge.
- `d`        :: source/sink vector (positive = source, negative = sink).
- `f`        :: current edge flow in B's reference orientation. Mutated by
                the relaxer.
- `incident` :: per-node `NodeIncidence` list — pre-computed at construction
                so the relaxer never touches `B` at hot-path level.
"""
mutable struct MaxFlowLaplacian
    B::SparseMatrixCSC{Float64,Int}
    c_plus::Vector{Float64}
    c_minus::Vector{Float64}
    d::Vector{Float64}
    f::Vector{Float64}
    incident::Vector{Vector{NodeIncidence}}
end

function MaxFlowLaplacian(B::SparseMatrixCSC, c_plus::AbstractVector,
                          c_minus::AbstractVector, d::AbstractVector;
                          f0::Union{Nothing,AbstractVector} = nothing)
    n = size(B, 1); m = size(B, 2)
    @assert length(c_plus) == m
    @assert length(c_minus) == m
    @assert length(d) == n
    f = f0 === nothing ? zeros(Float64, m) : collect(Float64.(f0))

    # Build per-node incidence by scanning B's columns (each column = one edge).
    incident = [NodeIncidence[] for _ in 1:n]
    rows = rowvals(B); vals = nonzeros(B)
    for e in 1:m
        endpoints = Int[]
        signs = Int[]
        for k in nzrange(B, e)
            push!(endpoints, rows[k])
            push!(signs, Int(vals[k]))
        end
        @assert length(endpoints) == 2 "edge $e has $(length(endpoints)) endpoints, expected 2"
        u, v = endpoints
        σu, σv = signs
        # σ at u "out-of-u" sense: B[u,e] = +1 means u is head ⇒ "out of u" is
        # opposite to B's reference ⇒ σ_u = -1. Conversely if u is tail.
        s_u = -σu
        s_v = -σv
        # u-oriented capacities: c_fwd = c⁺ if σ_u = +1 (u is tail), else c⁻.
        cu_fwd = s_u == +1 ? c_plus[e] : c_minus[e]
        cu_rev = s_u == +1 ? c_minus[e] : c_plus[e]
        cv_fwd = s_v == +1 ? c_plus[e] : c_minus[e]
        cv_rev = s_v == +1 ? c_minus[e] : c_plus[e]
        push!(incident[u], NodeIncidence(e, v, Int8(s_u), cu_fwd, cu_rev))
        push!(incident[v], NodeIncidence(e, u, Int8(s_v), cv_fwd, cv_rev))
    end
    return MaxFlowLaplacian(B, collect(Float64.(c_plus)), collect(Float64.(c_minus)),
                            collect(Float64.(d)), f, incident)
end

"""
    NonlinearGSRelaxer(state::MaxFlowLaplacian)

One Newton step per node per sweep. Conforms to the `Relaxer` interface so
the existing multilevel cycle infrastructure works unchanged.
"""
struct NonlinearGSRelaxer <: Relaxer
    state::MaxFlowLaplacian
end

function relax!(rx::NonlinearGSRelaxer, x::AbstractVector,
                b::AbstractVector; sweeps::Int = 1)
    state = rx.state
    n = size(state.B, 1)
    @assert length(x) == n
    @assert length(b) == n
    for _ in 1:sweeps
        @inbounds for u in 1:n
            inc = state.incident[u]
            isempty(inc) && continue
            # Solve the local 1D nodal residual r_u(φ_u) = 0 with neighbors
            # fixed. r_u is monotonic in φ_u (r'_u > 0 always — it is a sum
            # of conductance-like positive terms). Damped Newton converges
            # robustly; we iterate until |r| is small or we hit a cap.
            for _ in 1:10
                r, rp, _ = _local_residual_and_derivative!(state, x, b, u, inc)
                rp <= 0 && break
                Δ = r / rp
                abs(Δ) < 1e-12 * max(1.0, abs(x[u])) && break
                # Damped step: if the next φ_u would push f_e out of the
                # feasible interval at any incident edge, halve until safe.
                ω = 1.0
                while ω > 1e-12 && !_step_feasible(state, x, u, inc, ω * Δ)
                    ω *= 0.5
                end
                x[u] -= ω * Δ
            end
        end
    end
    return x
end

# Compute the local residual r_u(φ_u) = Σ g_e − b_u and its derivative
# r'_u(φ_u) = Σ 1 / (R_e + g_e · R'_e), while updating `state.f` with the
# freshly-solved implicit flows. Returns (r, r', sum |g|).
function _local_residual_and_derivative!(state::MaxFlowLaplacian,
                                         x::AbstractVector, b::AbstractVector,
                                         u::Int, inc::Vector{NodeIncidence})
    r = -b[u]
    rp = 0.0
    sum_abs_g = 0.0
    for inc_rec in inc
        e = inc_rec.e_idx
        σ = inc_rec.σ
        Δφ = x[u] - x[inc_rec.v]
        f0_u = σ * state.f[e]
        g, Rf, Rfp = solve_implicit_flow(inc_rec.c_fwd, inc_rec.c_rev, Δφ;
                                         f0 = f0_u)
        state.f[e] = σ * g
        r += g
        denom = Rf + g * Rfp
        denom > 0 && (rp += 1.0 / denom)
        sum_abs_g += abs(g)
    end
    return r, rp, sum_abs_g
end

# Check that taking a step Δ at node u would not push the implicit flow on
# any incident edge to within `margin` of the capacity.
function _step_feasible(state::MaxFlowLaplacian, x::AbstractVector, u::Int,
                        inc::Vector{NodeIncidence}, Δ::Float64;
                        margin::Float64 = 1e-6)
    φu_new = x[u] - Δ
    for inc_rec in inc
        Δφ = φu_new - x[inc_rec.v]
        # Saturation guard: if Δφ is so large that even a fully-saturated
        # edge couldn't balance it, the implicit-flow solver will struggle.
        # A simple guard: Δφ should not exceed (c_fwd + c_rev) * 10.
        bound = (inc_rec.c_fwd + inc_rec.c_rev) * 10.0
        abs(Δφ) > bound && return false
    end
    return true
end
