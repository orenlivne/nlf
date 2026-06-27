"""
Exact elimination of low-degree nodes for the constrained-linear
max-flow problem (analogue of LAMG §4.1 / `src/elimination.jl`).

The constrained-linear formulation is

    max α  s.t.  Aφ = αd,  low_e ≤ (Bᵀφ)_e ≤ high_e ∀e.

For a node `u` of degree 1 with single incident edge `e = (h, ta)`,
the operator row at `u` reads `φ_u − φ_{nbr(u)} = α · d_u` (unit-weight
unit-degree Laplacian). The edge gradient on `e` is then linear in α
alone (`σ_u · α · d_u`, where `σ_u = +1` if `u = h`, `−1` if `u = ta`),
so the *fine* box constraint `low_e ≤ (Bᵀφ)_e ≤ high_e` becomes a
constraint on α only:

    σ_u · low_e   ≤   α · d_u   ≤   σ_u · high_e        (if σ_u = +1)
    σ_u · high_e  ≤   α · d_u   ≤   σ_u · low_e         (if σ_u = -1)

i.e. one *α-interval* per eliminated degree-1 node. The reduced problem
holds on the surviving `c` nodes:

    A^red φ_c = α · d^red,    low_e ≤ (B_cᵀ φ_c)_e ≤ high_e  for surviving e,
    α ∈ ⋂_{u ∈ f} α-interval(u).

with `A^red`, `d^red` the Schur complement / restricted demand.

Degree-0 (isolated) nodes are simply dropped: their `d` entry must be
zero for the original system to be solvable, and they have no edges.

Degree ≥ 2 elimination introduces fictitious edges with bounds that
depend on the original edge bounds and α — TODO (see open items in
doc/maxflow_constrained.md).

Source / sink nodes are never eliminated, even if they are degree-1.
Eliminating them would require carrying a non-trivial `d^red` and
changes the identity of `s, t` — instead we simply mark them as
"protected".
"""

# ---------------------------------------------------------- Stage struct

"""
    MaxFlowEliminationStage

One stage of degree-≤1 elimination on a `NLFProblem`. The reduced
problem holds on `c = setdiff(1:n, vcat(z, f))` (zero-degree + the
eliminated low-degree set, in original numbering).

To recover the eliminated values once `(φ_c, α)` are known:
* Zero-degree nodes: `φ_u` is free (any value); we use `0`.
* Degree-1 nodes: `φ_u = φ_{nbr(u)} + σ_u · α · d_u`.

Fields
- `n`         : original node count.
- `z`         : zero-degree node indices (orig numbering).
- `f`         : degree-1 eliminated node indices (orig numbering).
- `c`         : surviving node indices (orig numbering).
- `f_neighbor`: for each `u ∈ f`, its single neighbor (in orig numbering).
- `f_edge`    : for each `u ∈ f`, the original edge index incident to it.
- `f_sigma`   : `σ_u ∈ {+1, -1}` — orientation of `u` on its edge.
- `f_d`       : original `d[u]` (cached for fast interpolation).
- `alpha_lo`  : tightest lower bound on α from f-nodes' box constraints.
- `alpha_hi`  : tightest upper bound on α.
"""
struct MaxFlowEliminationStage
    n::Int
    z::Vector{Int}
    f::Vector{Int}
    c::Vector{Int}
    f_neighbor::Vector{Int}
    f_edge::Vector{Int}
    f_sigma::Vector{Int}
    f_d::Vector{Float64}
    alpha_lo::Float64
    alpha_hi::Float64
end

# ---------------------------------------------------------- Identify candidates

"""
    low_degree_nlf_nodes(mfp; protect=Int[]) -> (z, f1)

Return the indices of:
- `z`  : nodes with no incident edges.
- `f1` : nodes with exactly one incident edge, excluding any node in
         `protect` (typically `[mfp.s, mfp.t]`).

For our use the degree-1 set is automatically mutually independent
(two degree-1 nodes sharing an edge would both want to be eliminated;
we just pick the first one in node order — the other becomes degree-1
*in the reduced problem* and can be eliminated by a later stage).
"""
function low_degree_nlf_nodes(mfp::NLFProblem;
                                  protect::Vector{Int} = Int[])
    n = size(mfp.A, 1)
    degree = zeros(Int, n)
    for e in eachindex(mfp.head)
        degree[mfp.head[e]] += 1
        degree[mfp.tail[e]] += 1
    end
    protect_set = Set(protect)
    z = Int[]; f1 = Int[]
    # Greedy: pick u as f1 only if its single neighbor is not already in f1.
    in_f = falses(n)
    for u in 1:n
        u ∈ protect_set && continue
        if degree[u] == 0
            push!(z, u)
        elseif degree[u] == 1
            # Find its neighbor.
            nbr = 0
            for e in eachindex(mfp.head)
                if mfp.head[e] == u
                    nbr = mfp.tail[e]; break
                elseif mfp.tail[e] == u
                    nbr = mfp.head[e]; break
                end
            end
            (nbr == 0 || in_f[nbr]) && continue
            push!(f1, u)
            in_f[u] = true
        end
    end
    return z, f1
end

# ---------------------------------------------------------- One elimination stage

"""
    eliminate_low_degree(mfp::NLFProblem; protect=[mfp.s, mfp.t])
        -> (mfp_reduced::NLFProblem, stage::MaxFlowEliminationStage)

One pass of degree-0/1 elimination. The reduced problem has

  A^red = A[c,c] - A[c,f] A[f,f]⁻¹ A[f,c]   (graph Laplacian on c)
  d^red = d[c]  - A[c,f] A[f,f]⁻¹ d[f]      (demand re-attaches eliminated d_u to nbr(u))

Edges incident to eliminated `z ∪ f` nodes are dropped from `B^red`.
Bounds on the surviving edges are unchanged.

The α-interval `[alpha_lo, alpha_hi]` is intersected over all f-nodes'
box constraints (initially `[-Inf, +Inf]`).

If `f` is empty (no candidates), returns `(mfp, stage)` with
`stage.f == []` and the surviving set `c = 1:n \\ z`.
"""
function eliminate_low_degree(mfp::NLFProblem;
                              protect::Vector{Int} =
                                  mfp.s == 0 ? Int[] : Int[mfp.s, mfp.t])
    n_orig = size(mfp.A, 1)
    z, f = low_degree_nlf_nodes(mfp; protect = protect)
    c_set = trues(n_orig)
    for i in z; c_set[i] = false; end
    for i in f; c_set[i] = false; end
    c = findall(c_set)

    # For each u ∈ f, find its single neighbor + incident edge index.
    f_neighbor = zeros(Int, length(f))
    f_edge     = zeros(Int, length(f))
    f_sigma    = zeros(Int, length(f))
    f_d        = zeros(Float64, length(f))
    alpha_lo = -Inf
    alpha_hi = +Inf
    f_to_pos = Dict{Int,Int}()
    for (k, u) in enumerate(f); f_to_pos[u] = k; end
    @inbounds for e in eachindex(mfp.head)
        h, ta = mfp.head[e], mfp.tail[e]
        for (u, σ, nbr) in ((h, +1, ta), (ta, -1, h))
            pos = get(f_to_pos, u, 0)
            pos == 0 && continue
            # σ = +1 if u = head, -1 if u = tail.
            f_neighbor[pos] = nbr
            f_edge[pos]     = e
            f_sigma[pos]    = σ
            f_d[pos]        = mfp.d[u]
            # α-interval from box on this edge: σ · α · d_u ∈ [low_e, high_e]
            lo_e = mfp.low[e]; hi_e = mfp.high[e]
            a_u = σ * mfp.d[u]
            if a_u > 0
                alpha_lo = max(alpha_lo, lo_e / a_u)
                alpha_hi = min(alpha_hi, hi_e / a_u)
            elseif a_u < 0
                # Flips: divide by negative number → reverse.
                alpha_lo = max(alpha_lo, hi_e / a_u)
                alpha_hi = min(alpha_hi, lo_e / a_u)
            else
                # a_u = 0: σ · 0 · α = 0, constraint reduces to lo_e ≤ 0 ≤ hi_e
                @assert lo_e <= 0 <= hi_e "degenerate degree-1 box constraint"
            end
        end
    end

    stage = MaxFlowEliminationStage(n_orig, z, f, c,
                                    f_neighbor, f_edge, f_sigma, f_d,
                                    alpha_lo, alpha_hi)

    # ─── Build the reduced NLFProblem ──────────────────────────────
    # 1. Reduced operator A^red (Schur complement).
    #    Since A[f,f] is diagonal (degree-1 nodes are independent and have
    #    A_uu = 1 each), A^red = A[c,c] - A[c,f] * A[f,c] (= -A[c,f] A[c,f]ᵀ
    #    since A[f,c] = A[c,f]ᵀ for the symmetric A).
    Afc = mfp.A[f, c]              # |f| × |c|
    Acf = mfp.A[c, f]              # |c| × |f|
    Aff_diag_inv = ones(length(f))  # = 1 ./ diag(A[f,f]); diag(A) = degree, = 1 for u ∈ f
    # More carefully: diag(A) entries for f-nodes — they are 1 if unit weights.
    for (k, u) in enumerate(f)
        Aff_diag_inv[k] = 1.0 / mfp.A[u, u]
    end
    A_red = mfp.A[c, c] - Acf * Diagonal(Aff_diag_inv) * Afc

    # 2. Reduced demand.
    d_red = mfp.d[c] .- Acf * (Aff_diag_inv .* mfp.d[f])

    # 3. Surviving edges = edges not incident to z ∪ f.
    elim_set = Set{Int}(); union!(elim_set, z); union!(elim_set, f)
    surv_edges = Int[]
    for e in eachindex(mfp.head)
        h, ta = mfp.head[e], mfp.tail[e]
        if h ∉ elim_set && ta ∉ elim_set
            push!(surv_edges, e)
        end
    end
    m_red = length(surv_edges)
    # Re-number node ids: c[k] → k.
    orig_to_red = zeros(Int, n_orig)
    for (k, u) in enumerate(c); orig_to_red[u] = k; end
    head_red = [orig_to_red[mfp.head[e]] for e in surv_edges]
    tail_red = [orig_to_red[mfp.tail[e]] for e in surv_edges]
    low_red  = mfp.low[surv_edges]
    high_red = mfp.high[surv_edges]

    # 4. Reduced incidence.
    edge_pairs = Tuple{Int,Int}[(head_red[i], tail_red[i]) for i in 1:m_red]
    B_red = incidence_from_edge_list(length(c), edge_pairs)

    # 5. Source / sink in reduced numbering.
    s_red = mfp.s == 0 ? 0 : orig_to_red[mfp.s]
    t_red = mfp.t == 0 ? 0 : orig_to_red[mfp.t]

    mfp_red = NLFProblem(sparse(A_red), B_red, low_red, high_red,
                              Vector{Float64}(d_red), s_red, t_red,
                              mfp.name * "/elim", head_red, tail_red)
    return mfp_red, stage
end

# ---------------------------------------------------------- Interpolation

"""
    interpolate_eliminated!(φ_full::AbstractVector, φ_c::AbstractVector,
                             α::Real, stage::MaxFlowEliminationStage)

Fill `φ_full` (length stage.n) from `φ_c` (length |stage.c|) and the
flow value `α`:
- `φ_full[c[k]] = φ_c[k]`
- `φ_full[u]    = 0`              for u ∈ z
- `φ_full[u]    = φ_full[nbr(u)] + σ_u · α · d_u`  for u ∈ f
"""
function interpolate_eliminated!(φ_full::AbstractVector, φ_c::AbstractVector,
                                  α::Real, stage::MaxFlowEliminationStage)
    @assert length(φ_full) == stage.n
    @assert length(φ_c) == length(stage.c)
    # 1. Survivors.
    @inbounds for (k, u) in enumerate(stage.c)
        φ_full[u] = φ_c[k]
    end
    # 2. Zero-degree nodes pinned to zero.
    @inbounds for u in stage.z
        φ_full[u] = 0.0
    end
    # 3. Degree-1 nodes via the operator row at u (unit-weight unit-degree
    #    Laplacian): A_uu φ_u + A_{u,v} φ_v = α d_u with A_uu = 1, A_{u,v} = -1,
    #    therefore φ_u = φ_v + α · d_u. The sign σ_u (head vs. tail of the
    #    leaf edge) appears in the edge-gradient identity (gradient = σ_u · α · d_u)
    #    and hence in the α-interval — but *not* in the substitution itself.
    @inbounds for k in eachindex(stage.f)
        u = stage.f[k]
        v = stage.f_neighbor[k]
        φ_full[u] = φ_full[v] + α * stage.f_d[k]
    end
    return φ_full
end

# ---------------------------------------------------------- 2-level FAS-elim cycle

"""
    fas_2level_cycle_elimination!(mfp, φ; α, ν_pre, ν_post, ν_coarsest, protect)

Single 2-level FAS cycle using *exact Schur elimination* as the coarsening,
in the style of LAMG paper eq. (3.2). Differs from `fas_2level_cycle!`
(which uses caliber-1 piecewise-constant aggregation) only in the
restriction/interpolation operators and the coarse problem build —
they're exact here rather than approximate.

Steps
-----
1.  Pre-relax `ν_pre` GS+Kaczmarz sweeps on the fine problem.
2.  `(mfp_red, stage) = eliminate_low_degree(mfp; protect)`.
    All fine c-c edges are inherited at the coarse level with their
    original bounds; all c-f (leaf) edges contribute to `stage.alpha_lo`,
    `stage.alpha_hi` (α-interval — an outer-loop constraint, NOT a
    coarse-level edge constraint).
3.  Restrict `φ_c = φ[c]`. No τ-correction: elimination is exact, so the
    coarse RHS is simply `α · d_red` (LAMG paper eq. 3.2), which is
    already encoded in `mfp_red.d`. (Equivalent to the FAS form
    `b_c = R r_f + A_c T φ_f` when `R = [I_c, −A_cf A_ff⁻¹]` is the
    Schur restriction — see doc/maxflow_constrained.md §7.)
4.  Coarse relax `ν_coarsest` sweeps on `A_red φ_c = α d_red` with c-c
    edge constraints (Kaczmarz enforced — these bounds are exact).
5.  Interpolate `φ[f] = φ[nbr(f)] + α · d_f`, `φ[z] = 0` (matches the
    operator row at each f-node).
6.  Post-relax `ν_post` fine-level GS+Kaczmarz sweeps.

Returns `‖Aφ − αd‖_after / ‖Aφ − αd‖_before` (relative-residual ratio).

Stationarity property
---------------------
If `φ_in` exactly solves `A φ = α d` with all fine edges interior-
feasible, then every step leaves `φ` unchanged: pre-relax is no-op
(zero residual, no violations), coarse start `φ_c = φ[c]` exactly
solves the coarse equation (proven in doc/maxflow_constrained.md §7
via the row-by-row substitution), coarse relax is no-op, interpolation
reproduces the f-rows of `φ_in` via the operator row at each f-node,
and post-relax is no-op.
"""
function fas_2level_cycle_elimination!(mfp::NLFProblem,
                                       φ::AbstractVector;
                                       α::Real = 0.0,
                                       ν_pre::Int = 2, ν_post::Int = 2,
                                       ν_coarsest::Int = 20,
                                       protect::Vector{Int} =
                                           mfp.s == 0 ? Int[] :
                                           Int[mfp.s, mfp.t])
    αd = α .* mfp.d
    r_before = norm(mfp.A * φ - αd)

    # 1. Pre-relax.
    relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = ν_pre)

    # 2. Build elimination "coarse level".
    mfp_red, stage = eliminate_low_degree(mfp; protect = protect)
    if isempty(stage.f) && isempty(stage.z)
        # No eliminable nodes — fall back to post-relax only.
        relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = ν_post)
        return norm(mfp.A * φ - αd) / max(r_before, 1e-30)
    end

    # 3. Restrict iterate to coarse. Schur is exact ⇒ b_c = α · d_red is
    #    already encoded in mfp_red.d; no τ-correction term needed.
    φ_c = φ[stage.c]

    # 4. Coarse relax on A_red φ_c = α · d_red with inherited c-c bounds.
    relax_gs_kaczmarz!(mfp_red, φ_c, mfp_red.d, α; sweeps = ν_coarsest)

    # 5. Interpolate back to the full φ.
    interpolate_eliminated!(φ, φ_c, α, stage)

    # 6. Post-relax.
    relax_gs_kaczmarz!(mfp, φ, mfp.d, α; sweeps = ν_post)

    return norm(mfp.A * φ - αd) / max(r_before, 1e-30)
end
