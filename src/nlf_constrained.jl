"""
Constrained-linear max-flow relaxation and FAS infrastructure.

Per the user's formulation (and the LAMG paper's §3 multilevel pattern):

   max α  s.t.  A φ = α d,    low_e ≤ (Bᵀφ)_e ≤ high_e.

The smoother per cycle is:
   1. **Gauss-Seidel on Aφ = α·d** (one forward sweep over nodes).
   2. **Kaczmarz on capacity constraints** (one pass over edges; for each
      violated edge, distributively redistribute φ between its endpoints
      to restore feasibility).

The FAS cycle restricts both the operator residual and the constraint
violations to the coarse level. Capacity constraints get **FAS
τ-corrected** (not Galerkin-coarsened) — they are inequalities, not a
linear operator.
"""

# ---------------------------------------------------------- GS+Kaczmarz sweep

"""
    relax_gs_kaczmarz!(mfp::NLFProblem, φ, b, α; sweeps=1, ω=1.0)

One GS sweep on Aφ = α·b followed by one Kaczmarz pass on capacity
constraints, repeated `sweeps` times. Mutates φ in place.

`b` is the right-hand-side at this level. At the finest level usually
`b = d` (the source/sink vector); at coarse levels it's the FAS-restricted
RHS plus `A_c · T φ_f`.

`α` is the global flow value; the relaxation treats it as fixed.
"""
function relax_gs_kaczmarz!(mfp::NLFProblem, φ::AbstractVector,
                            b::AbstractVector, α::Real;
                            sweeps::Int = 1, ω::Real = 1.0,
                            low::Union{Nothing,AbstractVector} = nothing,
                            high::Union{Nothing,AbstractVector} = nothing,
                            enforce_constraints::Bool = true)
    A = mfp.A
    n = size(A, 1)
    m = length(mfp.head)
    @assert length(φ) == n
    @assert length(b) == n
    αd = α .* b
    lo = something(low, mfp.low)
    hi = something(high, mfp.high)
    rows = rowvals(A); vals = nonzeros(A)
    # Pre-extract row index for fast row access via Aᵀ (since A is symmetric,
    # row i = column i — but iterating over column gives row indices).
    for _ in 1:sweeps
        # ─── Phase 1: forward Gauss-Seidel on Aφ = αd ───────────────────
        @inbounds for u in 1:n
            d_uu = 0.0
            sum_off = 0.0
            for k in nzrange(A, u)
                j = rows[k]; v = vals[k]
                if j == u
                    d_uu = v
                else
                    sum_off += v * φ[j]
                end
            end
            if d_uu != 0
                # GS update: φ_u ← (αd_u − Σ_{j≠u} A_{uj}φ_j) / A_{uu}
                φ[u] = (αd[u] - sum_off) / d_uu
            end
        end
        # ─── Phase 2: Kaczmarz on capacity constraints ──────────────────
        # For each edge e: violation = (Bᵀφ)_e − [lo_e, hi_e]
        # If above hi: subtract δ/2 from head, add δ/2 to tail (decrease grad)
        # If below lo: opposite.
        # This is the "distributive" Kaczmarz of the user's prescription:
        # changes only the difference φ_head − φ_tail.
        # Skipped at coarse levels (enforce_constraints=false): the
        # τ-corrected bounds at a coarse level don't always admit the
        # operator-equation solution as feasible, so Kaczmarz there fights
        # the operator GS sweep. Fine-level Kaczmarz on the original bounds
        # is what ultimately enforces feasibility.
        if enforce_constraints
            @inbounds for e in 1:m
                h = mfp.head[e]; ta = mfp.tail[e]
                grad = φ[h] - φ[ta]
                if grad > hi[e]
                    δ = ω * 0.5 * (grad - hi[e])
                    φ[h]  -= δ
                    φ[ta] += δ
                elseif grad < lo[e]
                    δ = ω * 0.5 * (lo[e] - grad)
                    φ[h]  += δ
                    φ[ta] -= δ
                end
            end
        end
    end
    return φ
end

"""
    relax_block_gs!(mfp, φ, b, α; sweeps=1, ω=1.0, low=nothing, high=nothing,
                      enforce_constraints=true)

Node-block constrained Gauss-Seidel smoother. At each node u:

  1. Compute the unconstrained GS update φ_u^new = (α b_u − Σ_{j ≠ u} A_{uj} φ_j) / A_{uu}.
  2. Compute the feasibility interval [lo_u, hi_u] for φ_u given the box
     constraints on all incident edges:
        for each edge e = (h_e, ta_e) incident to u with low_e ≤ φ_{h_e} − φ_{ta_e} ≤ high_e
        if u == h_e:  φ_u ∈ [φ_{ta_e} + low_e,  φ_{ta_e} + high_e]
        if u == ta_e: φ_u ∈ [φ_{h_e} − high_e,  φ_{h_e} − low_e]
     Intersect over all incident edges. If empty (box infeasible w.r.t.
     current neighbors), pick the midpoint of the most-relaxed bound.
  3. Project φ_u^new onto [lo_u, hi_u] (relaxed by ω if specified).

Compared with the edge-by-edge `relax_gs_kaczmarz!`, this avoids the
GS-push / Kaczmarz-pull oscillation on highly-active boxes. Cost is
O(deg(u)) per node, same asymptotic as GS+Kaczmarz.

`mfp.adj_edges` (lazily built node→edge index) is cached on the
`NLFProblem` if available; otherwise built once per call.
"""
function relax_block_gs!(mfp::NLFProblem, φ::AbstractVector,
                          b::AbstractVector, α::Real;
                          sweeps::Int = 1, ω::Real = 1.0,
                          low::Union{Nothing,AbstractVector} = nothing,
                          high::Union{Nothing,AbstractVector} = nothing,
                          enforce_constraints::Bool = true)
    A = mfp.A
    n = size(A, 1); m = length(mfp.head)
    @assert length(φ) == n && length(b) == n
    αd = α .* b
    lo = something(low, mfp.low); hi = something(high, mfp.high)
    rows = rowvals(A); vals = nonzeros(A)
    # Build node→incident-edges index once. For modest m this is O(m).
    # Each edge contributes two entries (head and tail).
    adj_starts = zeros(Int, n + 1)
    @inbounds for e in 1:m
        adj_starts[mfp.head[e] + 1] += 1
        adj_starts[mfp.tail[e] + 1] += 1
    end
    cumsum!(adj_starts, adj_starts)   # prefix-sum into starts
    adj_edges = Vector{Int}(undef, 2m)
    cursor = copy(adj_starts)
    @inbounds for e in 1:m
        h = mfp.head[e]; ta = mfp.tail[e]
        adj_edges[cursor[h] + 1] = e; cursor[h] += 1
        adj_edges[cursor[ta] + 1] = e; cursor[ta] += 1
    end
    for _ in 1:sweeps
        @inbounds for u in 1:n
            # 1. unconstrained GS update
            d_uu = 0.0; sum_off = 0.0
            for k in nzrange(A, u)
                j = rows[k]; v = vals[k]
                j == u ? (d_uu = v) : (sum_off += v * φ[j])
            end
            d_uu == 0 && continue
            φ_new = (αd[u] - sum_off) / d_uu
            # 2. tighten by incident-edge boxes
            if enforce_constraints
                lo_u = -Inf; hi_u = +Inf
                for k in adj_starts[u]:(adj_starts[u + 1] - 1)
                    e = adj_edges[k + 1]
                    if mfp.head[e] == u
                        # edge u→v: low ≤ φ_u − φ_v ≤ high
                        v = mfp.tail[e]
                        lo_e = φ[v] + lo[e]; hi_e = φ[v] + hi[e]
                    else
                        # edge v→u: low ≤ φ_v − φ_u ≤ high  ⇒  φ_v − hi ≤ φ_u ≤ φ_v − lo
                        v = mfp.head[e]
                        lo_e = φ[v] - hi[e]; hi_e = φ[v] - lo[e]
                    end
                    lo_e > lo_u && (lo_u = lo_e)
                    hi_e < hi_u && (hi_u = hi_e)
                end
                # If infeasible (lo_u > hi_u), take midpoint — gives the
                # local least-squares projection onto the polytope of incident
                # box constraints.
                if lo_u > hi_u
                    φ_new = 0.5 * (lo_u + hi_u)
                else
                    φ_new = clamp(φ_new, lo_u, hi_u)
                end
            end
            # 3. ω-relaxation
            φ[u] = (1 - ω) * φ[u] + ω * φ_new
        end
    end
    return φ
end

# ---------------------------------------------------------- shrinkage

"""
    shrinkage_gs(L::SparseMatrixCSC; sweeps_max=20, num_examples=4, rng=...)

GS shrinkage on Lx = 0 (no constraints) for comparison with the constrained
version. Returns the geometric-mean per-sweep residual reduction.
"""
function shrinkage_gs(L::SparseMatrixCSC; sweeps_max::Int = 20,
                      num_examples::Int = 4,
                      rng = Random.default_rng())
    n = size(L, 1)
    rx = GaussSeidelRelaxer(L)
    factors = Float64[]
    for _ in 1:num_examples
        x = randn(rng, n); x .-= sum(x) / n
        b = zeros(n)
        norms = Float64[norm(L * x)]
        for _ in 1:sweeps_max
            relax!(rx, x, b; sweeps = 1)
            x .-= sum(x) / n
            r = norm(L * x)
            push!(norms, r)
            r < 1e-12 * norms[1] && break
        end
        ratios = norms[2:end] ./ max.(norms[1:end - 1], 1e-30)
        tail = ratios[max(end - 3, 1):end]
        push!(factors, exp(mean(log.(max.(tail, 1e-30)))))
    end
    return mean(factors)
end

"""
    shrinkage_gs_kaczmarz(mfp; sweeps_max=20, num_examples=4, α=0.0, rng=...)
        -> (μ_op, μ_full)

Shrinkage of the GS+Kaczmarz smoother on `Aφ = α·d` with capacity
constraints. Returns two factors:
- `μ_op`   : per-sweep reduction of the **operator residual** `‖Aφ − αd‖`.
             Directly comparable with `shrinkage_gs(A)` on the plain
             Laplacian (no constraints).
- `μ_full` : per-sweep reduction of the **combined residual + constraint
             violation** norm √(‖Aφ−αd‖² + Σ_e clip(grad − [lo,hi])²).
             What ultimately matters for convergence to the constrained
             optimum.

With α = 0 and lo < 0 < hi, φ = 0 is interior-feasible. Both factors
should be well below 1 for the smoother to be effective.
"""
function shrinkage_gs_kaczmarz(mfp::NLFProblem;
                               sweeps_max::Int = 20, num_examples::Int = 4,
                               α::Real = 0.0, rng = Random.default_rng())
    A = mfp.A; B = mfp.B; Bt = sparse(B')
    n = size(A, 1)
    lo = mfp.low; hi = mfp.high
    αd = α .* mfp.d

    function violation_norm(φ)
        grad = Bt * φ
        v = 0.0
        @inbounds for e in eachindex(grad)
            if grad[e] > hi[e]
                v += (grad[e] - hi[e])^2
            elseif grad[e] < lo[e]
                v += (lo[e] - grad[e])^2
            end
        end
        return sqrt(v)
    end

    op_factors = Float64[]; full_factors = Float64[]
    for _ in 1:num_examples
        φ = randn(rng, n); φ .-= sum(φ) / n
        op_hist = Float64[norm(A * φ - αd)]
        full_hist = Float64[sqrt(op_hist[1]^2 + violation_norm(φ)^2)]
        for _ in 1:sweeps_max
            relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = 1)
            φ .-= sum(φ) / n
            r_op = norm(A * φ - αd)
            r_full = sqrt(r_op^2 + violation_norm(φ)^2)
            push!(op_hist, r_op)
            push!(full_hist, r_full)
            r_full < 1e-12 * full_hist[1] && break
        end
        function geomean_tail(hist)
            ratios = hist[2:end] ./ max.(hist[1:end - 1], 1e-30)
            tail = ratios[max(end - 3, 1):end]
            return exp(mean(log.(max.(tail, 1e-30))))
        end
        push!(op_factors, geomean_tail(op_hist))
        push!(full_factors, geomean_tail(full_hist))
    end
    return mean(op_factors), mean(full_factors)
end

# ---------------------------------------------------------- FAS constraint coarsening

"""
    coarsen_constraints(low_f, high_f, φ_f, B, P, T, B_c) -> (low_c, high_c)

FAS τ-correction for capacity inequalities (user's formulation):

  fine:   low_e ≤ grad(φ_f)_e ≤ high_e        per fine edge e
  coarse: low^c_E ≤ grad^c(φ_c)_E ≤ high^c_E   per coarse edge E

The coarse constraint at edge E aggregates fine constraints via
  low^c_E  = (Pᵀ ((low - grad(φ_f))) + grad^c(T φ_f))_E
  high^c_E = (Pᵀ ((high - grad(φ_f))) + grad^c(T φ_f))_E

This is FAS: it preserves the linearization that "if the coarse
correction equals the restricted fine error, the bound is exactly the
fine bound translated to coarse coordinates" — and is *not* a Galerkin
coarsening of the inequalities (which would be meaningless for box
constraints).

Inputs:
- `low_f`, `high_f`     :: fine-level bound vectors (length m_f)
- `φ_f`                 :: current fine-level approximation
- `B`                   :: fine incidence (n_f × m_f)
- `P`                   :: fine-to-coarse interpolation (n_f × n_c)
- `T`                   :: coarse-variable type (= R, n_c × n_f)
- `B_c`                 :: coarse incidence (n_c × m_c)

The reduction P → m_f-to-m_c happens via `Pᵀ` applied to *edge-level*
quantities. Practically, we compute (low_f − grad(φ_f)) on each fine edge
and aggregate to coarse edges using the **edge-aggregation** induced by
P (a coarse edge E pools all fine edges between aggregates I and J).
"""
function coarsen_constraints(low_f::AbstractVector, high_f::AbstractVector,
                             φ_f::AbstractVector,
                             B::SparseMatrixCSC, P::SparseMatrixCSC,
                             T::SparseMatrixCSC, B_c::SparseMatrixCSC,
                             fine_head::Vector{Int}, fine_tail::Vector{Int},
                             aggregate::Vector{Int};
                             coarse_head::Vector{Int} = Int[],
                             coarse_tail::Vector{Int} = Int[])
    m_f = length(low_f)
    grad_f = B' * φ_f        # length m_f
    # Slack per fine edge — both bounds reformulated.
    slack_lo = low_f .- grad_f
    slack_hi = high_f .- grad_f

    # Coarse representation of φ_f.
    Tφ = T * φ_f
    grad_Tφ_c = B_c' * Tφ    # length m_c

    # Aggregate fine slacks to coarse edges.
    # Each fine edge (u, v) maps to a coarse edge (aggregate[u], aggregate[v])
    # IF the two endpoints land in different aggregates; otherwise the fine
    # edge is "absorbed" inside an aggregate and we drop its slack (cf.
    # standard LAMG caliber-1 P).
    # We need a map: (I, J) → coarse edge index. Build from coarse_head/tail.
    coarse_edge_map = Dict{Tuple{Int,Int}, Int}()
    for (e, (h, t)) in enumerate(zip(coarse_head, coarse_tail))
        coarse_edge_map[(h, t)] = e
        coarse_edge_map[(t, h)] = e
    end
    m_c = length(coarse_head)
    slack_lo_c = zeros(m_c)
    slack_hi_c = zeros(m_c)
    coarse_count = zeros(Int, m_c)
    for e_f in 1:m_f
        I = aggregate[fine_head[e_f]]
        J = aggregate[fine_tail[e_f]]
        I == J && continue
        e_c = get(coarse_edge_map, (I, J), 0)
        e_c == 0 && continue
        # Sign correction: the fine edge orientation may be opposite to the
        # coarse edge's reference orientation.
        sign = (coarse_head[e_c] == I) ? +1.0 : -1.0
        slack_lo_c[e_c] += sign * slack_lo[e_f]
        slack_hi_c[e_c] += sign * slack_hi[e_f]
        coarse_count[e_c] += 1
    end
    # Coarse bounds: low_c = slack_lo_c + grad^c(T φ_f),  high_c = slack_hi_c + ...
    low_c  = slack_lo_c .+ grad_Tφ_c
    high_c = slack_hi_c .+ grad_Tφ_c
    # Sanity: low_c should be ≤ high_c after the τ-correction.
    for e in 1:m_c
        if low_c[e] > high_c[e]
            # Defensive — set to the midpoint with zero slack.
            mid = 0.5 * (low_c[e] + high_c[e])
            low_c[e] = high_c[e] = mid
        end
    end
    return low_c, high_c
end

# ---------------------------------------------------------- coarse problem build

"""
    build_coarse_problem(mfp; rng=...) -> (mfp_c, P, T, aggregate)

Build a coarse max-flow problem via affinity aggregation on the
graph-Laplacian A. The coarse problem's operator is Galerkin:
   A_c = PᵀAP   with  P = caliber-1 piecewise-constant interpolation.

A coarse edge (I, J) exists iff at least one fine edge has its endpoints
in aggregates I and J (I ≠ J). The coarse capacities are placeholders
(unused — capacity *bounds* on (Bᶜ)ᵀφᶜ are computed per cycle via
`coarsen_constraints`'s FAS τ-correction).
"""
function build_coarse_problem(mfp::NLFProblem;
                              rng = Random.default_rng())
    # 1. Aggregate (LAMG affinity).
    ag = aggregate(mfp.A; rng = rng)
    aggregate_vec = ag.aggregate
    n_c = ag.n_coarse

    # 2. Caliber-1 P, R (= T), Q (= Pᵀ).
    P, R, _Q = piecewise_constant_interpolation(aggregate_vec)
    T = R

    # 3. Coarse Galerkin operator A_c = PᵀAP.
    A_c = sparse(P' * mfp.A * P)

    # 4. Coarse edges = canonical-pair coarse-edge set.
    pair_set = Set{Tuple{Int,Int}}()
    canon(I, J) = I <= J ? (I, J) : (J, I)
    for e in eachindex(mfp.head)
        I = aggregate_vec[mfp.head[e]]
        J = aggregate_vec[mfp.tail[e]]
        I != J && push!(pair_set, canon(I, J))
    end
    coarse_edges = Tuple{Int,Int}[(I, J) for (I, J) in pair_set]
    m_c = length(coarse_edges)
    B_c = incidence_from_edge_list(n_c, coarse_edges)
    head_c = [e[1] for e in coarse_edges]
    tail_c = [e[2] for e in coarse_edges]

    # 5. d at coarse — restrict (mostly placeholder; the FAS cycle's RHS at the
    #    coarse level doesn't read d_c directly).
    d_c = R * mfp.d

    # Identify coarse source/sink as the aggregates containing the fine
    # source and sink (if both fall in the same aggregate, drop both —
    # signals an over-aggregated problem).
    s_c = aggregate_vec[mfp.s]
    t_c = aggregate_vec[mfp.t]

    mfp_c = NLFProblem(A_c, B_c, fill(-Inf, m_c), fill(+Inf, m_c),
                            d_c, s_c, t_c, mfp.name * "/coarse",
                            head_c, tail_c)
    return mfp_c, P, T, aggregate_vec
end

# ---------------------------------------------------------- 2-level FAS cycle

"""
    fas_2level_cycle!(mfp, φ; α=0.0, ν_pre=2, ν_post=2, ν_coarsest=20, rng=...)

Single 2-level FAS cycle for max-flow:
  1. Pre-relax `ν_pre` GS+Kaczmarz sweeps at the fine level.
  2. Build coarse problem.
  3. Coarsen bound constraints via FAS τ-correction (`coarsen_constraints`).
  4. Restrict residual + τ-correct: b_c = R(α·d − A φ_f) + A_c · T φ_f.
  5. Relax `ν_coarsest` sweeps at the coarse level (solving A_c φ_c = b_c
     with coarse bounds).
  6. Correct: φ_f ← φ_f + P (φ_c − T φ_f).
  7. Post-relax `ν_post` sweeps at the fine level.

Mutates φ in place; returns the relative residual reduction over this cycle.
"""
function fas_2level_cycle!(mfp::NLFProblem, φ::AbstractVector;
                           α::Real = 0.0, ν_pre::Int = 2, ν_post::Int = 2,
                           ν_coarsest::Int = 20,
                           rng = Random.default_rng(),
                           mfp_c::Union{Nothing,NLFProblem} = nothing,
                           P::Union{Nothing,SparseMatrixCSC} = nothing,
                           T::Union{Nothing,SparseMatrixCSC} = nothing,
                           agg::Union{Nothing,Vector{Int}} = nothing)
    n_f = size(mfp.A, 1)
    αd = α .* mfp.d

    r_before = norm(mfp.A * φ - αd)

    # 1. Pre-relax.
    relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = ν_pre)

    # 2. Coarse problem (cached if provided).
    if mfp_c === nothing
        mfp_c, P, T, agg = build_coarse_problem(mfp; rng = rng)
    end

    # 3. FAS τ-corrected bounds.
    lo_c, hi_c = coarsen_constraints(mfp.low, mfp.high, φ, mfp.B, P, T,
                                     mfp_c.B, mfp.head, mfp.tail, agg;
                                     coarse_head = mfp_c.head,
                                     coarse_tail = mfp_c.tail)

    # 4. FAS operator-side RHS: b_c = Pᵀ(α·d - A φ_f) + A_c (T φ_f).
    #    Matches LAMG.jl's solve_cycle.jl convention: the residual is
    #    SUM-restricted (Q = Pᵀ), not AVG-coarsened (R = T = (PᵀP)⁻¹ Pᵀ).
    Tφ = T * φ
    r_f = αd .- mfp.A * φ
    b_c = P' * r_f .+ mfp_c.A * Tφ

    # 5. Coarse relaxation: pure GS on operator equation A_c φ_c = b_c.
    #    We deliberately do NOT run Kaczmarz at the coarse level — the
    #    sign-corrected τ-aggregation of inequality slacks can make the
    #    coarse bounds inconsistent with the operator solution, causing
    #    the two sub-sweeps to oscillate (debug: r_c 0.08 → 15.8 over 20
    #    coarsest sweeps). Constraint enforcement happens via fine-level
    #    Kaczmarz in the pre- and post-relax phases.
    φ_c = copy(Tφ)
    relax_gs_kaczmarz!(mfp_c, φ_c, b_c, 1.0;
                       sweeps = ν_coarsest,
                       low = lo_c, high = hi_c,
                       enforce_constraints = false)

    # 6. FAS correction.
    φ .+= P * (φ_c .- Tφ)

    # 7. Post-relax.
    relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = ν_post)

    r_after = norm(mfp.A * φ - αd)
    return r_after / max(r_before, 1e-30)
end

# ============================================================================
# Polymorphic Relaxer subtype for the constrained-linear max-flow path.
# This is the ONLY place that knows the max-flow domain. The cycle code
# (solve_cycle.jl) dispatches per-level through this type's `relax!` and
# `update_fas!` — Level itself remains a pure structural carrier.
# ============================================================================

"""
    GeneralizedConstraint

One row constraint produced when a fine edge incident to an eliminated (F)
node is rewritten in terms of surviving (C) nodes via the Schur exact-
interpolation row `x_F[k] = P[k, :] * x_C  +  q[k] * b_F[k]` (LAMG paper
§3.3 Algorithm 2). The constraint reads
  `low_active ≤ vals' * x_C[cols] ≤ high_active`
where the per-cycle active bounds are
  `low_active  = low_static  + b_correction_coeff * b_F[b_correction_index]`
  `high_active = high_static + b_correction_coeff * b_F[b_correction_index]`
refreshed by `update_fas_elim!` from the FAS-restricted RHS.
"""
# Runtime toggle for the coarse-ELIM generalized-row projection.
#
# DEFAULT OFF. The anchored PFAS τ-shift (update_fas_elim!) is mathematically
# correct — it preserves stationarity (one cycle from the converged solution
# returns it; verified in test_nlf_unified_stationarity.jl-style probes)
# and the no-op property (test_box_elim_tau_shift.jl). HOWEVER, enabling the
# projection from a COLD START destabilises 3D transients: the constraint-blind
# linear hierarchy lets the moving free boundary over-constrain the coarse
# correction (grid3d/12³ regresses μ 0.08→0.85; under-relaxation does not help).
# It is also INERT on the headline grid2d/256² stall (μ=0.886 with or without).
#
# Per the Brandt sub-agent, the real fix for both is constraint-AWARE coarsening
# (PFASMD: aggregate/eliminate respecting the active set), a separate larger
# task. Until then the projection stays gated off to preserve the working
# baseline; the τ-shift machinery + tests remain as the correct foundation.
# Set ENV["LAMG_PROJ_GEN"]="1" to enable for experiments.
const _PROJ_GEN = Ref(get(ENV, "LAMG_PROJ_GEN", "0") != "0")

struct GeneralizedConstraint
    cols::Vector{Int}                 # C-local indices the row touches
    vals::Vector{Float64}             # corresponding coefficients
    low_static::Float64               # low_f[e]
    high_static::Float64              # high_f[e]
    b_correction_coeff::Float64       # +q[h_F] (F-C case) or -q[t_F] (C-F case)
    b_correction_index::Int           # F-local index of the F endpoint
    # Endpoints of the originating edge in the FINER level's node numbering.
    # Used by the anchored PFAS τ-shift (update_fas_elim!): the coarse active
    # bound is anchored to the current fine edge gap φ[fine_head]-φ[fine_tail].
    fine_head::Int
    fine_tail::Int
end

"""
    MaxFlowGSKaczmarzRelaxer(mfp; low, high, box_edge_map, enforce_constraints)

Box-aware Gauss-Seidel + Kaczmarz smoother for the constrained-linear
max-flow problem. Holds all per-level problem-domain state, hiding it
from the cycle code:

- `mfp`           : per-level `NLFProblem` (A, B, head, tail).
- `low`, `high`   : CURRENT (possibly FAS τ-corrected) per-edge bounds.
- `low0`, `high0` : STATIC snapshots from setup time. Coarse levels are
                     τ-corrected per cycle via `update_fas!` which rebuilds
                     `low/high` from these snapshots + the current fine
                     iterate. The finest level keeps its original bounds.
- `box_edge_map`  : fine→coarse edge map (one entry per FINE edge, length
                     m_fine): +E = same orientation as coarse edge E,
                     -E = flipped, 0 = absorbed inside an aggregate.
                     `nothing` on the finest and on ELIM-coarse stubs —
                     `update_fas!` is a no-op then.
- `enforce_constraints` : whether the Kaczmarz pass projects. True only
                     at the finest level — at coarse levels the τ-corrected
                     bounds can fight the operator-GS sweep.
"""
mutable struct MaxFlowGSKaczmarzRelaxer <: Relaxer
    mfp::NLFProblem
    low::Vector{Float64}
    high::Vector{Float64}
    low0::Vector{Float64}
    high0::Vector{Float64}
    box_edge_map::Union{Nothing, Vector{Int}}
    enforce_constraints::Bool
    # Generalized row constraints emitted by Schur substitution at ELIM
    # levels. Each row reads
    #   low_active[g] ≤ vals_g' * x[cols_g] ≤ high_active[g]
    # where low_active = g.low_static + g.b_correction_coeff * b_F[g.b_correction_index]
    # is refreshed per cycle by `update_fas_elim!` from the stage's bstages
    # restricted RHS. Empty on the finest and on AGG levels.
    generalized::Vector{GeneralizedConstraint}
    low_active::Vector{Float64}        # per-generalized-row lower bound (cycle state)
    high_active::Vector{Float64}       # per-generalized-row upper bound
    # Pre-elimination indices of the stage's F nodes that produced
    # `generalized`. `stage_f_indices[k]` is the PRE-ELIM index of the
    # F-local-k F-node, so that `b_F[k] = bstages[1][stage_f_indices[k]]`.
    # `nothing` on AGG / finest / multi-stage ELIM stubs.
    stage_f_indices::Union{Nothing, Vector{Int}}
end

function MaxFlowGSKaczmarzRelaxer(mfp::NLFProblem;
                                   low::AbstractVector = mfp.low,
                                   high::AbstractVector = mfp.high,
                                   box_edge_map::Union{Nothing,Vector{Int}} = nothing,
                                   enforce_constraints::Bool = true,
                                   generalized::Vector{GeneralizedConstraint} =
                                       GeneralizedConstraint[],
                                   stage_f_indices::Union{Nothing,Vector{Int}} =
                                       nothing)
    lo = collect(Float64.(low))
    hi = collect(Float64.(high))
    # Initialize active bounds to the static values (b_F correction added per-cycle).
    nG = length(generalized)
    low_active  = Float64[generalized[g].low_static  for g in 1:nG]
    high_active = Float64[generalized[g].high_static for g in 1:nG]
    return MaxFlowGSKaczmarzRelaxer(mfp, lo, hi, copy(lo), copy(hi),
                                     box_edge_map, enforce_constraints,
                                     generalized, low_active, high_active,
                                     stage_f_indices)
end

# Concrete relax! — dispatches to GS+Kaczmarz against the level's own
# (possibly τ-corrected) bounds and additionally projects against any
# generalized row constraints emitted by Schur elimination on ELIM levels.
#
# enforce_constraints controls the STANDARD ±1 incidence Kaczmarz at this
# level. It's true at the finest (original problem bounds) and false at
# coarse levels (τ-corrected bounds fight operator GS). The generalized-
# row projection (from Schur elimination on ELIM coarse) is governed
# independently: those rows live in a span the operator GS does not
# touch directly (no x_C row is locally adjustable to satisfy them), so
# they DO NOT fight GS and are always projected when present.
function relax!(rx::MaxFlowGSKaczmarzRelaxer, x::AbstractVector,
                b::AbstractVector; sweeps::Int = 1)
    if isempty(rx.generalized)
        # No Schur-elim generalized rows on this level.
        relax_gs_kaczmarz!(rx.mfp, x, b, 1.0;
                           sweeps = sweeps,
                           low = rx.low, high = rx.high,
                           enforce_constraints = rx.enforce_constraints)
    else
        # ELIM-coarse path: generalized Schur-substituted rows are present.
        # Run the standard operator GS+Kaczmarz, then project onto the
        # PFAS-anchored generalized rows. The anchor (update_fas_elim!) makes
        # low_active/high_active consistent with the current fine edge gaps,
        # so the projection is inert when the fine state is feasible and only
        # engages as the coarse correction would push a fine edge out of box.
        for _ in 1:sweeps
            relax_gs_kaczmarz!(rx.mfp, x, b, 1.0;
                               sweeps = 1,
                               low = rx.low, high = rx.high,
                               enforce_constraints = rx.enforce_constraints)
            _PROJ_GEN[] && _project_generalized!(rx, x)
        end
    end
    return x
end

"""
Project `x` onto every generalized row constraint `vals' x[cols] ∈ [low_active, high_active]`
via Kaczmarz row projection. Standard formula:
    s = vals' * x[cols]
    if s < lo: dx = (lo - s) / ||vals||² · vals; x[cols] .+= dx
    if s > hi: dx = (hi - s) / ||vals||² · vals; x[cols] .+= dx
(no-op when in-bounds).
"""
function _project_generalized!(rx::MaxFlowGSKaczmarzRelaxer, x::AbstractVector)
    G = rx.generalized
    lo_act = rx.low_active
    hi_act = rx.high_active
    @inbounds for k in eachindex(G)
        g = G[k]
        cols = g.cols; vals = g.vals
        s = 0.0
        for j in eachindex(cols)
            s += vals[j] * x[cols[j]]
        end
        lo = lo_act[k]; hi = hi_act[k]
        dx = if s < lo
            lo - s
        elseif s > hi
            hi - s
        else
            0.0
        end
        if dx != 0.0
            w2 = 0.0
            for j in eachindex(vals)
                w2 += vals[j] * vals[j]
            end
            w2 < 1e-30 && continue
            α = dx / w2
            for j in eachindex(cols)
                x[cols[j]] += α * vals[j]
            end
        end
    end
    return x
end

"""
    update_fas_elim!(coarse::MaxFlowGSKaczmarzRelaxer, x_C0, φ_fine) -> coarse

Refresh the per-cycle `low_active / high_active` of the coarse ELIM relaxer
using the **anchored PFAS τ-shift** (Brandt & Cryer 1983, "Projected FAS").

Each generalized constraint `g` reads `g_C(x_C) := vals' x_C[cols]` — the
Schur-substituted coarse expression for the original fine edge gap
`φ[fine_head] − φ[fine_tail]`. In FAS the coarse iterate is initialized to the
restricted fine state `x_C0` (= `p.x[c]`), and a coarse correction `Δx_C` is
sought. Interpolated back, that correction shifts the fine edge gap by
`g_C(x_C) − g_C(x_C0)`. Keeping the *fine* edge inside its box requires

    low_e ≤ (φ_h−φ_t)_fine + g_C(x_C) − g_C(x_C0) ≤ high_e

i.e. `low_e + c0 ≤ g_C(x_C) ≤ high_e + c0` with the anchor offset

    c0 = g_C(x_C0) − (φ[fine_head] − φ[fine_tail]).

So `low_active = low_static + c0`, `high_active = high_static + c0`.

NO-OP PROPERTY (unit-tested): when the fine edge currently satisfies its box,
the FAS-initial coarse iterate `x_C0` satisfies the generalized constraint
exactly — `g_C(x_C0) = c0 + (φ_h−φ_t)_fine ∈ [low_static+c0, high_static+c0]`
⟺ `(φ_h−φ_t)_fine ∈ [low_static, high_static]`. Thus projection is inert at
the start of each coarse visit and only engages as the coarse correction would
push the fine edge out of its box. This is what makes the projection stable in
deep hierarchies (the static-only variant enforced an RHS-inconsistent bound).

`x_C0` is the coarse-initial iterate (`p.x[c]`, coarse numbering, indexed by
`cols`); `φ_fine` is the finer-level iterate (`p.x[l]`, indexed by the
constraint's `fine_head/fine_tail`). No-op when `generalized` is empty.
"""
function update_fas_elim!(coarse::MaxFlowGSKaczmarzRelaxer,
                          x_C0::AbstractVector,
                          φ_fine::AbstractVector)
    G = coarse.generalized
    isempty(G) && return coarse
    @inbounds for k in eachindex(G)
        g = G[k]
        gC0 = 0.0
        for j in eachindex(g.cols)
            gC0 += g.vals[j] * x_C0[g.cols[j]]
        end
        fine_gap = φ_fine[g.fine_head] - φ_fine[g.fine_tail]
        c0 = gC0 - fine_gap
        lo = g.low_static + c0
        hi = g.high_static + c0
        if lo > hi
            mid = 0.5 * (lo + hi)
            lo = hi = mid
        end
        coarse.low_active[k]  = lo
        coarse.high_active[k] = hi
    end
    return coarse
end

# (The no-op `update_fas_elim!(::Relaxer, ...)` fallback for the linear path is defined in
# the LAMG+ package, `relaxer.jl`; this max-flow package only adds the specialised method
# above and extends `LAMG.update_fas_elim!` via the import in `NLF.jl`.)

"""
    update_fas!(coarse::MaxFlowGSKaczmarzRelaxer,
                fine::MaxFlowGSKaczmarzRelaxer, φ_f, P, T) -> coarse

FAS τ-correction for AGG-coarse box bounds. Called by the cycle BEFORE
recursing into the coarse level. Rebuilds `coarse.low/.high` from the
static snapshots + sign-summed fine slack at the current fine iterate:

    low_c[E]  = low0[E]  + (B_c^T (T φ_f))[E] − Σ_{e ∈ bundle(E)} σ_e (Bᵀ φ_f)_e
    high_c[E] = high0[E] + (B_c^T (T φ_f))[E] − Σ_{e ∈ bundle(E)} σ_e (Bᵀ φ_f)_e

where bundle + signs come from `coarse.box_edge_map`. When the map is
`nothing` (finest level, ELIM-coarse stubs), this is a no-op.

Linear-path relaxers fall through to the default no-op defined in
`relaxer.jl`.
"""
function update_fas!(coarse::MaxFlowGSKaczmarzRelaxer,
                     fine::MaxFlowGSKaczmarzRelaxer,
                     φ_f::AbstractVector, P, T)
    edge_map = coarse.box_edge_map
    edge_map === nothing && return coarse
    head_c = coarse.mfp.head; tail_c = coarse.mfp.tail
    head_f = fine.mfp.head;   tail_f = fine.mfp.tail
    low0 = coarse.low0; high0 = coarse.high0
    m_c = length(head_c)
    m_f = length(head_f)
    @assert length(edge_map) == m_f
    fine_grad_sum = zeros(Float64, m_c)
    @inbounds for e in 1:m_f
        E_signed = edge_map[e]
        E_signed == 0 && continue
        σ = E_signed > 0 ? 1.0 : -1.0
        E = abs(E_signed)
        fine_grad_sum[E] += σ * (φ_f[head_f[e]] - φ_f[tail_f[e]])
    end
    Tφ = T === nothing ? φ_f : T * φ_f
    low_c = coarse.low; high_c = coarse.high
    @inbounds for E in 1:m_c
        grad_c = Tφ[head_c[E]] - Tφ[tail_c[E]]
        delta = grad_c - fine_grad_sum[E]
        low_c[E]  = low0[E]  + delta
        high_c[E] = high0[E] + delta
        if low_c[E] > high_c[E]
            mid = 0.5 * (low_c[E] + high_c[E])
            low_c[E] = high_c[E] = mid
        end
    end
    return coarse
end


"""
    box_projected_gs!(mfp, φ; pinned=falses(n), sweeps=1) -> φ

Per-node **box-projected Gauss–Seidel** for the constrained-linear problem
`Aφ = 0` (interior) with the edge box `low_e ≤ (Bᵀφ)_e ≤ high_e` (Rule A /
[LOP34] §2.1). Each non-pinned node is relaxed toward its `Aφ=0` GS value and
then clamped into the interval permitted *simultaneously* by all its incident
edge boxes; pinned nodes (e.g. Dirichlet `s,t` for voltage-driving) are held.

This is the smoother the voltage-driven continuation needs (the older
`relax_gs_kaczmarz!` does not propagate a Dirichlet boundary nor clamp the box
tightly). It is correct and box-feasible at the SINGLE-GRID level for moderate
drive; reaching the saturation value `F*` requires running it inside the
multilevel cycle, because the flow limit is set by a (possibly distant)
downstream bottleneck that a local sweep cannot see (see `doc/maxflow_fas_plan.md`).
Incidence is rebuilt each call unless `inc` is supplied.
"""
function box_projected_gs!(mfp::NLFProblem, φ::AbstractVector;
                           pinned::AbstractVector{Bool} = falses(size(mfp.A,1)),
                           sweeps::Int = 1,
                           inc::Union{Nothing,Vector{Vector{Int}}} = nothing)
    A = mfp.A; n = size(A,1); rows = rowvals(A); vals = nonzeros(A)
    if inc === nothing
        inc = [Int[] for _ in 1:n]
        @inbounds for e in 1:length(mfp.head)
            push!(inc[mfp.head[e]], e); push!(inc[mfp.tail[e]], e)
        end
    end
    @inbounds for _ in 1:sweeps, i in 1:n
        pinned[i] && continue
        diag = 0.0; off = 0.0
        for k in nzrange(A,i)
            j = rows[k]; (j == i) ? (diag = vals[k]) : (off += vals[k]*φ[j])
        end
        diag == 0 && continue
        φgs = -off/diag
        lo = -Inf; hi = Inf
        for e in inc[i]
            if mfp.head[e] == i
                j = mfp.tail[e]
                lo = max(lo, φ[j] + mfp.low[e]); hi = min(hi, φ[j] + mfp.high[e])
            else
                j = mfp.head[e]
                lo = max(lo, φ[j] - mfp.high[e]); hi = min(hi, φ[j] - mfp.low[e])
            end
        end
        φ[i] = lo > hi ? 0.5*(lo+hi) : clamp(φgs, lo, hi)   # lo>hi ⇒ over-constrained (cut)
    end
    return φ
end
