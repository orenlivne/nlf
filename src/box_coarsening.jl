"""
Box-constraint coarsening for max-flow (unified-hierarchy refactor).

The linear LAMG hierarchy is built from `setup(A)` and coarsens the operator A
through ELIM (Schur) and AGG (caliber-1 PC) levels. The max-flow problem also
carries a side data structure of *box constraints* on edge gradients:

    low_e ≤ (B^T φ)_e ≤ high_e   ∀ edge e.

This file provides pure helpers that coarsen those box constraints in lock-step
with the operator hierarchy. There are two variants — one per level type:

* `coarsen_box_agg`  — bundles fine edges whose endpoint aggregates differ into
  coarse edges. Caliber-1 PC ⇒ a coarse edge is the *sum* of its fine bundle
  (with sign correction when the fine edge orientation disagrees with the
  reference coarse orientation).

* `coarsen_box_elim` — re-indexes c-c fine edges through the Schur c-mapping.
  Edges touching an F-node would in general produce *generalized* (multi-edge)
  constraints; for the current pass we DROP them and stash them under
  `extra_terms` (returned but empty if the stub branch is taken). The unified-
  hierarchy SETUP path doesn't yet plumb generalized constraints into the
  cycle, so dropping them keeps the box side well-formed at coarse levels.
"""

# ---------------------------------------------------------- AGG coarsening

"""
    coarsen_box_agg(low_f, high_f, head_f, tail_f, aggregate,
                     head_c = nothing, tail_c = nothing)
        -> (low_c, high_c, head_c_emit, tail_c_emit)

AGG-level box coarsening. The coarse edge set is the set of pairs (I, J) such
that some fine edge `e` has `(aggregate[head_f[e]], aggregate[tail_f[e]])` =
`(I, J)` (or `(J, I)`).

For each coarse edge `E` with reference orientation `I → J`:

    low_c[E]  = Σ_{e ∈ bundle(E)} σ_e · low_f[e]
    high_c[E] = Σ_{e ∈ bundle(E)} σ_e · high_f[e]

where `σ_e = +1` if `(aggregate[head_f[e]], aggregate[tail_f[e]]) = (I, J)`,
and `σ_e = -1` if it is `(J, I)`. Fine edges with both endpoints in the same
aggregate are *skipped* (they cannot be represented under caliber-1 P; their
gradient is identically zero).

If a sign-summed `low_c[E]` exceeds `high_c[E]` (which can happen when forward
and reverse capacities differ across the bundle), the entry is collapsed to
its midpoint (defensive). This matches the same defense used in the legacy
per-cycle `coarsen_constraints` helper.

If `head_c` and `tail_c` are provided (e.g. taken from the coarse A's existing
edge structure), emit values aligned to that orientation. Otherwise emit
freshly-built coarse edges in canonical sort order with `I < J`.
"""
function coarsen_box_agg(low_f::AbstractVector{<:Real},
                          high_f::AbstractVector{<:Real},
                          head_f::Vector{Int}, tail_f::Vector{Int},
                          aggregate::Vector{Int},
                          head_c::Union{Nothing,Vector{Int}} = nothing,
                          tail_c::Union{Nothing,Vector{Int}} = nothing)
    m_f = length(head_f)
    @assert length(tail_f) == m_f
    @assert length(low_f) == m_f
    @assert length(high_f) == m_f

    if head_c === nothing || tail_c === nothing
        # Build coarse edges from scratch in canonical sort order with I < J.
        pair_index = Dict{Tuple{Int,Int}, Int}()
        head_emit = Int[]
        tail_emit = Int[]
        low_emit = Float64[]
        high_emit = Float64[]
        @inbounds for e in 1:m_f
            I = aggregate[head_f[e]]
            J = aggregate[tail_f[e]]
            I == J && continue
            # Canonical orientation: smaller→larger (head = max for parity
            # with our existing canon I≤J + head=I/tail=J semantics).
            if I < J
                Ihc, Jtc, σ = I, J, +1.0
            else
                Ihc, Jtc, σ = J, I, -1.0
            end
            key = (Ihc, Jtc)
            E = get(pair_index, key, 0)
            if E == 0
                push!(head_emit, Ihc)
                push!(tail_emit, Jtc)
                push!(low_emit,  σ * low_f[e])
                push!(high_emit, σ * high_f[e])
                pair_index[key] = length(head_emit)
            else
                lo_term = σ * low_f[e]
                hi_term = σ * high_f[e]
                low_emit[E]  += lo_term
                high_emit[E] += hi_term
            end
        end
        # Defensive: if low > high after summation, collapse to midpoint.
        @inbounds for E in 1:length(low_emit)
            if low_emit[E] > high_emit[E]
                mid = 0.5 * (low_emit[E] + high_emit[E])
                low_emit[E] = high_emit[E] = mid
            end
        end
        return low_emit, high_emit, head_emit, tail_emit
    else
        # Align to provided coarse edge order: map (Ihc, Jtc) → E. Allow either
        # orientation of (head_c, tail_c) per edge — fine-edge sign is computed
        # against the provided orientation.
        m_c = length(head_c)
        @assert length(tail_c) == m_c
        pair_index = Dict{Tuple{Int,Int}, Int}()
        @inbounds for E in 1:m_c
            pair_index[(head_c[E], tail_c[E])] = E
            pair_index[(tail_c[E], head_c[E])] = -E   # store reversed marker
        end
        low_c = zeros(Float64, m_c)
        high_c = zeros(Float64, m_c)
        @inbounds for e in 1:m_f
            I = aggregate[head_f[e]]
            J = aggregate[tail_f[e]]
            I == J && continue
            E = get(pair_index, (I, J), 0)
            if E == 0
                # Coarse-edge structure refuses this fine edge.
                # (Should not happen if the provided head_c/tail_c was built
                # from the same aggregate map; we drop silently.)
                continue
            end
            σ = E > 0 ? +1.0 : -1.0
            E_abs = abs(E)
            low_c[E_abs]  += σ * low_f[e]
            high_c[E_abs] += σ * high_f[e]
        end
        @inbounds for E in 1:m_c
            if low_c[E] > high_c[E]
                mid = 0.5 * (low_c[E] + high_c[E])
                low_c[E] = high_c[E] = mid
            end
        end
        return low_c, high_c, head_c, tail_c
    end
end

"""
    coarsen_box_agg_with_map(low_f, high_f, head_f, tail_f, aggregate)
        -> (low_c, high_c, head_c, tail_c, edge_map)

Same as `coarsen_box_agg(...)` (no `head_c`/`tail_c` provided) but ALSO
returns `edge_map`: a length-m_f signed vector where `edge_map[e]` is the
1-based coarse edge index that bundles fine edge `e` (positive = same
orientation, negative = flipped) or `0` if the fine edge is absorbed
inside an aggregate (both endpoints in the same coarse node).

The map is what the per-cycle FAS τ-correction needs to walk fine edges
and accumulate sign-corrected slack into coarse-edge slots without
re-deriving the (I, J) → coarse-edge dictionary every cycle.
"""
function coarsen_box_agg_with_map(low_f::AbstractVector{<:Real},
                                   high_f::AbstractVector{<:Real},
                                   head_f::Vector{Int}, tail_f::Vector{Int},
                                   aggregate::Vector{Int})
    m_f = length(head_f)
    @assert length(tail_f) == m_f
    @assert length(low_f) == m_f
    @assert length(high_f) == m_f
    pair_index = Dict{Tuple{Int,Int}, Int}()
    head_emit = Int[]
    tail_emit = Int[]
    low_emit = Float64[]
    high_emit = Float64[]
    edge_map = zeros(Int, m_f)
    @inbounds for e in 1:m_f
        I = aggregate[head_f[e]]
        J = aggregate[tail_f[e]]
        if I == J
            edge_map[e] = 0
            continue
        end
        if I < J
            Ihc, Jtc, σ = I, J, +1
        else
            Ihc, Jtc, σ = J, I, -1
        end
        key = (Ihc, Jtc)
        E = get(pair_index, key, 0)
        if E == 0
            push!(head_emit, Ihc)
            push!(tail_emit, Jtc)
            push!(low_emit,  σ * low_f[e])
            push!(high_emit, σ * high_f[e])
            E = length(head_emit)
            pair_index[key] = E
        else
            low_emit[E]  += σ * low_f[e]
            high_emit[E] += σ * high_f[e]
        end
        edge_map[e] = σ * E    # signed coarse-edge index
    end
    # Defensive: if low > high after summation, collapse to midpoint.
    @inbounds for E in 1:length(low_emit)
        if low_emit[E] > high_emit[E]
            mid = 0.5 * (low_emit[E] + high_emit[E])
            low_emit[E] = high_emit[E] = mid
        end
    end
    return low_emit, high_emit, head_emit, tail_emit, edge_map
end

# ---------------------------------------------------------- ELIM coarsening

"""
    coarsen_box_elim(low_f, high_f, head_f, tail_f,
                      stage::EliminationStage, A_pre_elim)
        -> (low_c, high_c, head_c, tail_c, generalized)

ELIM-level box coarsening for ONE Schur stage. Implements LAMG paper §3.3
Algorithm 2 (exact interpolation) for the box-constraint side: each fine
edge incident to an F (eliminated) node is rewritten as a generalized row
constraint on C-nodes by substituting `x_F = P*x_C + q*b_F`.

Returns five arrays:
- `head_c, tail_c, low_c, high_c` — the simple ±1 incidence edges that
  survived unchanged (CASE C-C only), re-indexed through `c → 1:|c|`.
- `generalized::Vector{GeneralizedConstraint}` — the rewritten F-incident
  edges (CASE F-C and C-F).

CASES:
  * h, t ∈ C    : copy with re-indexed (hc, tc).
  * h ∈ F, t ∈ C: emit GeneralizedConstraint with
                  `w[k] = P[h_F, k]` for `k ≠ t_C`, `w[t_C] = P[h_F, t_C] - 1`,
                  `b_correction_coeff = +q[h_F]`, `b_correction_index = h_F`.
  * h ∈ C, t ∈ F: emit GeneralizedConstraint with
                  `w[k] = -P[t_F, k]` for `k ≠ h_C`, `w[h_C] = 1 - P[t_F, h_C]`,
                  `b_correction_coeff = -q[t_F]`, `b_correction_index = t_F`.
  * h, t ∈ F    : impossible (F is independent under `low_degree_nodes`); asserts.

`A_pre_elim` is unused (the stage's `P` and `q` carry all the math); kept
in the signature for forward compatibility.
"""
function coarsen_box_elim(low_f::AbstractVector{<:Real},
                           high_f::AbstractVector{<:Real},
                           head_f::Vector{Int}, tail_f::Vector{Int},
                           stage::EliminationStage,
                           A_pre_elim::SparseMatrixCSC)
    m_f = length(head_f)
    @assert length(tail_f) == m_f
    @assert length(low_f) == m_f
    @assert length(high_f) == m_f

    n_pre = stage.n
    # c_pos[u] = C-local index (1..|c|) if u ∈ c, else 0.
    # f_pos[u] = F-local index (1..|f|) if u ∈ f, else 0.
    c_pos = zeros(Int, n_pre)
    f_pos = zeros(Int, n_pre)
    @inbounds for (k, u) in enumerate(stage.c); c_pos[u] = k; end
    @inbounds for (k, u) in enumerate(stage.f); f_pos[u] = k; end

    head_c = Int[]
    tail_c = Int[]
    low_c  = Float64[]
    high_c = Float64[]
    generalized = GeneralizedConstraint[]

    @inbounds for e in 1:m_f
        h = head_f[e]; t = tail_f[e]
        hc = c_pos[h]; tc = c_pos[t]
        hf = f_pos[h]; tf = f_pos[t]
        lo = float(low_f[e]); hi = float(high_f[e])

        if hc > 0 && tc > 0
            # CASE C-C: copy with re-indexed endpoints.
            push!(head_c, hc)
            push!(tail_c, tc)
            push!(low_c,  lo)
            push!(high_c, hi)
        elseif hf > 0 && tc > 0
            # CASE F-C: x_h - x_t = (P[h_F,:] - δ_{tc}) ⋅ x_C + q[h_F] * b_F[h_F]
            #   w[k]  = P[h_F, k]      for k ≠ tc
            #   w[tc] = P[h_F, tc] - 1
            push!(generalized, _row_from_P(stage, hf, tc, lo, hi;
                                          tc_sign = -1,
                                          b_coeff = stage.q[hf],
                                          b_idx = hf,
                                          fine_head = h, fine_tail = t))
        elseif hc > 0 && tf > 0
            # CASE C-F: x_h - x_t = (δ_{hc} - P[t_F,:]) ⋅ x_C - q[t_F] * b_F[t_F]
            #   w[k]  = -P[t_F, k]     for k ≠ hc
            #   w[hc] = 1 - P[t_F, hc]
            push!(generalized, _row_from_P(stage, tf, hc, lo, hi;
                                          tc_sign = +1, neg_P = true,
                                          b_coeff = -stage.q[tf],
                                          b_idx = tf,
                                          fine_head = h, fine_tail = t))
        elseif hf > 0 && tf > 0
            # CASE F-F: impossible per low_degree_nodes (F is independent set).
            @assert false "coarsen_box_elim: edge $(e) has both endpoints in F"
        else
            @assert false "coarsen_box_elim: edge $(e) endpoint not classified"
        end
    end

    return low_c, high_c, head_c, tail_c, generalized
end

# Helper: build a GeneralizedConstraint row from one row of stage.P.
#   F-C case: `neg_P = false`, `tc_sign = -1` (subtract 1 at position `pivot`).
#   C-F case: `neg_P = true`,  `tc_sign = +1` (add 1 at position `pivot`).
function _row_from_P(stage::EliminationStage, f_row::Int, pivot::Int,
                     lo::Float64, hi::Float64;
                     tc_sign::Int, neg_P::Bool = false,
                     b_coeff::Float64, b_idx::Int,
                     fine_head::Int, fine_tail::Int)
    Prow = stage.P[f_row, :]              # SparseVector over |c| C-local columns
    cols = Int[]
    vals = Float64[]
    pivot_added = false
    @inbounds for (k, v) in zip(Prow.nzind, Prow.nzval)
        coeff = neg_P ? -v : v
        if k == pivot
            coeff += tc_sign              # ±1 adjustment at the C-end of the edge
            pivot_added = true
        end
        if coeff != 0.0
            push!(cols, k); push!(vals, coeff)
        end
    end
    if !pivot_added
        # P had no nonzero at the pivot column → still inject the ±1.
        # Insert in sorted order.
        idx = searchsortedfirst(cols, pivot)
        insert!(cols, idx, pivot)
        insert!(vals, idx, Float64(tc_sign))
    end
    return GeneralizedConstraint(cols, vals, lo, hi, b_coeff, b_idx,
                                 fine_head, fine_tail)
end

# ---------------------------------------------------------- setup(mfp) overload

"""
    setup(mfp::NLFProblem; options::LAMGOptions = LAMGOptions()) -> Multilevel

Build the **same** linear LAMG hierarchy as `setup(mfp.A; options)`, then walk
the levels populating each with max-flow box-constraint metadata
(`head, tail, low, high`). This is the unified-hierarchy SETUP path: the
max-flow solver can reuse the linear-LAMG operator hierarchy verbatim and
read its box side off the levels.

NOTES
- ELIM-level box coarsening uses the `coarsen_box_elim` STUB which DROPS
  F-incident edges. This is conservative but well-formed (it never produces
  invalid bounds). The hierarchy-equality test does not depend on the box
  side being exact at ELIM levels.
- For AGG levels, the aggregate map is recovered from the level's `p` matrix:
  each row of P has exactly one nonzero in caliber-1 PC, identifying its
  aggregate.
"""
function setup(mfp::NLFProblem; options::LAMGOptions = LAMGOptions())
    # 1. Build the linear hierarchy on mfp.A — identical sequence to setup(A).
    mlh = setup(mfp.A; options = options)
    # 2. Finest: swap the GS relaxer for a box-aware MaxFlowGSKaczmarzRelaxer.
    #    All problem-domain state lives INSIDE the relaxer. The Level itself
    #    remains a pure structural carrier — `solve_cycle.jl` is oblivious to
    #    box state.
    mlh[1].relaxer = MaxFlowGSKaczmarzRelaxer(mfp;
                                               low = mfp.low,
                                               high = mfp.high,
                                               box_edge_map = nothing,
                                               enforce_constraints = true)
    # 3. Walk down, building per-level coarse NLFProblem views and
    #    attaching a coarse MaxFlowGSKaczmarzRelaxer carrying the
    #    fine→coarse edge map so its `update_fas!` hook can run the
    #    τ-correction.
    for l in 2:length(mlh)
        prev_rx   = mlh[l - 1].relaxer::MaxFlowGSKaczmarzRelaxer
        cur       = mlh[l]
        fine_head = prev_rx.mfp.head
        fine_tail = prev_rx.mfp.tail
        edge_map  = nothing                   # default → update_fas! no-op
        generalized_l = GeneralizedConstraint[]
        stage_f_indices_l::Union{Nothing,Vector{Int}} = nothing
        if is_elimination(cur)
            head_cur = fine_head; tail_cur = fine_tail
            low_cur  = prev_rx.low;       high_cur = prev_rx.high
            A_cur    = mlh[l - 1].a
            nstages = length(cur.elim_stages)
            if nstages == 1
                # SINGLE-STAGE — cols emitted in the FINAL coarse-A C-local
                # numbering. Safe to attach.
                stage = cur.elim_stages[1]
                low_cur, high_cur, head_cur, tail_cur, gens =
                    coarsen_box_elim(low_cur, high_cur, head_cur, tail_cur,
                                     stage, A_cur)
                append!(generalized_l, gens)
                stage_f_indices_l = collect(stage.f)
            else
                # MULTI-STAGE — generalized rows from stage 1 are in stage-1's
                # c-local numbering, but the level's coarse A is in stage-N's
                # c-local numbering (with N > 1). Composing the rows through
                # subsequent stages' P matrices is correct but non-trivial.
                # FALLBACK to the prior stub: just re-index C-C survivors and
                # drop F-incident edges. Box constraints at this level are
                # approximated (same accuracy as before the refactor) but the
                # A-side is bit-identical. Single-stage ELIM levels (the
                # common case at the finest ELIM) still get the exact rows.
                for stage in cur.elim_stages
                    low_cur, high_cur, head_cur, tail_cur, _gens =
                        coarsen_box_elim(low_cur, high_cur, head_cur, tail_cur,
                                         stage, A_cur)
                end
            end
            head_c = head_cur; tail_c = tail_cur
            low_c  = low_cur;  high_c = high_cur
        else
            # AGG level. Recover aggregate map from cur.p (caliber-1 PC).
            P = cur.p
            n_f = size(P, 1)
            agg_vec = zeros(Int, n_f)
            rows = rowvals(P); vals = nonzeros(P)
            for j in 1:size(P, 2)
                for k in nzrange(P, j)
                    i = rows[k]
                    if vals[k] != 0
                        agg_vec[i] = j
                    end
                end
            end
            @assert all(agg_vec .> 0) "could not recover aggregate map from P at level $l"
            low_c, high_c, head_c, tail_c, edge_map =
                coarsen_box_agg_with_map(prev_rx.low, prev_rx.high,
                                          fine_head, fine_tail, agg_vec)
        end
        # Build a coarse-level NLFProblem view.
        n_c = size(cur.a, 1)
        m_c = length(head_c)
        coarse_edges = Tuple{Int,Int}[(head_c[e], tail_c[e]) for e in 1:m_c]
        B_c = incidence_from_edge_list(n_c, coarse_edges)
        d_c = zeros(Float64, n_c)         # placeholder; cycle overrides RHS
        mfp_c = NLFProblem(cur.a, B_c,
                                copy(low_c), copy(high_c),
                                d_c, 1, n_c, mfp.name * "/lvl$l",
                                copy(head_c), copy(tail_c))
        # Coarse relaxer: enforce_constraints=false (τ-corrected bounds can
        # fight GS) and carry the fine→coarse edge map for `update_fas!`,
        # plus generalized Schur-row constraints + their stage-F indices for
        # `update_fas_elim!`. AGG coarse: enforce_constraints=true if the
        # generalized set is empty (default for box-active grid problems).
        cur.relaxer = MaxFlowGSKaczmarzRelaxer(mfp_c;
                                                low = low_c, high = high_c,
                                                box_edge_map = edge_map,
                                                enforce_constraints = false,
                                                generalized = generalized_l,
                                                stage_f_indices = stage_f_indices_l)
    end
    return mlh
end
