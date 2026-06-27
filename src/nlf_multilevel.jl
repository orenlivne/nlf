"""
Multilevel FAS cycle for the constrained-linear max-flow problem.

Builds a hierarchy by recursive aggregation (`build_coarse_problem`)
until the coarsest level is small enough, then runs a V-cycle that
threads FAS τ-corrected bounds + Schur restriction through every
level.

Stopping criterion for setup: stop when a coarsening fails to shrink
the node count or hits a minimum size (`n_min`).

Cycle conventions (matches the LAMG linear solve and our 2-level FAS):
- The fine level enforces capacity constraints via Kaczmarz; coarser
  levels do operator-only GS (we disable Kaczmarz at non-finest
  because sign-summed τ-aggregation can produce inconsistent bounds —
  documented in doc/maxflow_constrained.md §5).
- Restriction is `Pᵀ` (sum-restriction), consistent with
  `src/solve_cycle.jl` and the 2-level cycle.
"""

# ---------------------------------------------------------- Hierarchy

"""
    NLFHierarchy

Linear chain of aggregation-coarsened max-flow problems.
- `levels[l]` :: coarse problem at level `l` (level 1 = fine).
- `P[l]`      :: n_l × n_{l+1} prolongation from level l+1 to l.
- `T[l]`      :: n_{l+1} × n_l restriction (the standard caliber-1 T = R).
- `agg[l]`    :: length n_l, aggregate index per fine-l node.
"""
struct NLFHierarchy
    levels::Vector{NLFProblem}
    P::Vector{SparseMatrixCSC{Float64,Int}}
    T::Vector{SparseMatrixCSC{Float64,Int}}
    agg::Vector{Vector{Int}}
end

num_levels(hier::NLFHierarchy) = length(hier.levels)

"""
    setup_nlf_hierarchy(mfp; n_min=20, max_levels=20, rng=...) -> NLFHierarchy

Build the multilevel aggregation hierarchy for `mfp`. Stops when the
coarsest level has ≤ `n_min` nodes, or coarsening stalls (next level
isn't smaller than current), or `max_levels` is reached.
"""
function setup_nlf_hierarchy(mfp::NLFProblem;
                                  n_min::Int = 20,
                                  max_levels::Int = 20,
                                  rng = Random.default_rng())
    levels = NLFProblem[mfp]
    Ps = SparseMatrixCSC{Float64,Int}[]
    Ts = SparseMatrixCSC{Float64,Int}[]
    aggs = Vector{Int}[]
    while length(levels) < max_levels && size(levels[end].A, 1) > n_min
        mfp_c, P, T, agg = build_coarse_problem(levels[end]; rng = rng)
        size(mfp_c.A, 1) >= size(levels[end].A, 1) && break   # stalled
        push!(levels, mfp_c)
        push!(Ps, P); push!(Ts, T); push!(aggs, agg)
    end
    return NLFHierarchy(levels, Ps, Ts, aggs)
end

# ---------------------------------------------------------- Cycle

"""
    fas_multilevel_cycle!(hier, φ; α, ν_pre=2, ν_post=2, ν_coarsest=40,
                          low=hier.levels[1].low, high=hier.levels[1].high)
        -> Float64

Single V-cycle on the max-flow hierarchy. Mutates `φ` (length =
`size(hier.levels[1].A, 1)`); returns the relative-residual ratio
`‖Aφ_after − αd‖ / ‖Aφ_before − αd‖`.

Internally calls `_fas_visit!(hier, l, φ_l, b_l, low_l, high_l, …)` at
each level `l`. At the finest, `b_1 = α·d`, `low_1 = mfp.low`,
`high_1 = mfp.high`. Each coarse level receives FAS-corrected `b_c`
and τ-corrected `low_c, high_c` computed from the current iterate.
"""
function fas_multilevel_cycle!(hier::NLFHierarchy, φ::AbstractVector;
                                α::Real = 0.0,
                                ν_pre::Int = 1, ν_post::Int = 2,
                                ν_coarsest::Int = 40,
                                low::AbstractVector = hier.levels[1].low,
                                high::AbstractVector = hier.levels[1].high,
                                smoother::Symbol = :gs_kaczmarz,
                                γ_vec::Union{Nothing,Vector{Float64}} = nothing,
                                history::Union{Nothing,Vector{<:Any}} = nothing,
                                do_recomb::Bool = true)
    mfp = hier.levels[1]
    αd = α .* mfp.d
    r_before = norm(mfp.A * φ - αd)
    L = num_levels(hier)
    γ = γ_vec === nothing ? fill(1.0, max(L - 1, 0)) : γ_vec
    _fas_visit!(hier, 1, φ, Vector{Float64}(αd), Vector{Float64}(low),
                 Vector{Float64}(high), ν_pre, ν_post, ν_coarsest, true;
                 smoother = smoother, γ_vec = γ,
                 history = history, do_recomb = do_recomb,
                 visits = fill(0, max(L - 1, 0)))
    return norm(mfp.A * φ - αd) / max(r_before, 1e-30)
end

"""
    _fas_visit!(hier, l, φ_l, b_l, low_l, high_l, ν_pre, ν_post, ν_coarsest, is_finest)

Recursive worker for `fas_multilevel_cycle!`. At each level:
1. pre-relax;
2. if `l == num_levels(hier)` — solve via many GS+Kaczmarz sweeps
   (Kaczmarz disabled here because the τ-corrected coarse bounds are
   inherited from the finer level, not the original problem);
3. else: τ-coarsen bounds, build FAS RHS, restrict iterate, recurse,
   correct, post-relax.

`b_l` is the RIGHT-HAND SIDE at level `l` (already includes any `α`
factor — `relax_gs_kaczmarz!` is called with `α = 1.0` so it solves
`A_l φ_l = b_l` directly).
"""
function _fas_visit!(hier::NLFHierarchy, l::Int, φ::AbstractVector,
                     b::AbstractVector, low::AbstractVector, high::AbstractVector,
                     ν_pre::Int, ν_post::Int, ν_coarsest::Int,
                     is_finest::Bool;
                     smoother::Symbol = :gs_kaczmarz,
                     γ_vec::Vector{Float64} = Float64[],
                     history::Union{Nothing,Vector{<:Any}} = nothing,
                     do_recomb::Bool = true,
                     visits::Vector{Int} = Int[])
    mfp = hier.levels[l]
    enforce_box = is_finest    # only at fine level
    relax_fn! = smoother == :block_gs ? relax_block_gs! : relax_gs_kaczmarz!
    # Pre-relax (or coarsest solve).
    if l == num_levels(hier)
        relax_fn!(mfp, φ, b, 1.0;
                  sweeps = ν_coarsest, low = low, high = high,
                  enforce_constraints = false)
        return
    end
    relax_fn!(mfp, φ, b, 1.0;
              sweeps = ν_pre, low = low, high = high,
              enforce_constraints = enforce_box)
    # Coarsen constraints via FAS τ-correction (uses current φ).
    mfp_c = hier.levels[l + 1]
    P = hier.P[l]; T = hier.T[l]; agg = hier.agg[l]
    low_c, high_c = coarsen_constraints(low, high, φ, mfp.B, P, T,
                                         mfp_c.B, mfp.head, mfp.tail, agg;
                                         coarse_head = mfp_c.head,
                                         coarse_tail = mfp_c.tail)
    # FAS RHS: b_c = Pᵀ(b − Aφ) + A_c · Tφ.    (NO 4/3 — paper-exact FAS)
    Tφ = T * φ
    r = b .- mfp.A * φ
    b_c = P' * r .+ mfp_c.A * Tφ
    φ_c = Vector{Float64}(Tφ)
    # Clear coarse-level history at descent (LAMG solve_cycle.jl:136).
    if do_recomb && history !== nothing && (l + 1) <= length(history) &&
       history[l + 1] !== nothing
        LAMG.clear_history!(history[l + 1])
    end
    # γ-controlled coarse visits. visits[l] tracks descents from level l;
    # max_visits at level l+1 = γ_vec[l] * visits[l].
    n_coarse_visits = 0
    while true
        n_coarse_visits += 1
        _fas_visit!(hier, l + 1, φ_c, b_c, low_c, high_c,
                    ν_pre, ν_post, ν_coarsest, false;
                    smoother = smoother, γ_vec = γ_vec,
                    history = history, do_recomb = do_recomb,
                    visits = visits)
        # NOTE: We do NOT save at level l+1 here. The recursive
        # `_fas_visit!(l+1, …)` call already saves its own iterate at
        # level l+1 at the right (PRE-correction) point, mirroring
        # MATLAB ProcessorSolve.m's `saveIterate(l, xf, rf)`. Saving
        # again here would (a) double-fill history[l+1] per visit, and
        # (b) capture the POST-EVERYTHING-at-l+1 iterate (post post-
        # relax), which is a DIFFERENT state from the pre-correction
        # iterate linear LAMG saves. Doing both pollutes the recomb LS
        # basis with mixed-state iterates.
        # Update visit counter at level l (the parent).
        if l <= length(visits); visits[l] += 1; end
        # Decide whether to do another coarse visit.
        γ_here = (l <= length(γ_vec)) ? γ_vec[l] : 1.0
        # max_visits at the coarse level = γ * (parent visits).
        # In a single _fas_visit! call we model one parent visit, so:
        max_visits = γ_here
        if n_coarse_visits >= max_visits
            break
        end
    end
    # Iterate recombination at the coarse level before correction
    # (LAMG solve_cycle.jl:160-162 / MATLAB ProcessorSolve.m:237-241).
    if do_recomb && history !== nothing && (l + 1) <= length(history) &&
       history[l + 1] !== nothing
        r_c = b_c .- mfp_c.A * φ_c
        LAMG.min_res!(history[l + 1], φ_c, r_c)
    end
    # Save the PRE-CORRECTION iterate at level l. Mirrors LAMG
    # solve_cycle.jl:165-167 / MATLAB ProcessorSolve.m:245-247:
    #   if (l > obj.finest); obj.saveIterate(l, xf, rf); end
    # Two changes from the previous (buggy) version:
    #   (i) save happens BEFORE the FAS correction (was after),
    #   (ii) save is SKIPPED at the finest level — the finest is
    #        saved at cycle start (in `fas_multilevel_solve!`) and
    #        recombined at cycle end, never saved here.
    if !is_finest && do_recomb && history !== nothing &&
       l <= length(history) && history[l] !== nothing
        r_l = b .- mfp.A * φ
        LAMG.save_iterate!(history[l], φ, r_l)
    end
    # FAS correction.
    φ .+= P * (φ_c .- Tφ)
    # Post-relax at level l.
    relax_fn!(mfp, φ, b, 1.0;
              sweeps = ν_post, low = low, high = high,
              enforce_constraints = enforce_box)
    return
end

# ---------------------------------------------------------- Convergence driver

"""
    fas_multilevel_solve!(hier, φ; α, tol=1e-8, max_cycles=50, ν_pre=2,
                          ν_post=2, ν_coarsest=40) -> (cycles, residuals)

Run multilevel FAS cycles until the relative residual
`‖Aφ − αd‖ / ‖αd‖` drops below `tol` or `max_cycles` is reached.
Returns the iteration count and the history of residual norms.
"""
function fas_multilevel_solve!(hier::NLFHierarchy, φ::AbstractVector;
                                α::Real = 0.0,
                                tol::Real = 1e-8,
                                max_cycles::Int = 50,
                                ν_pre::Int = 1, ν_post::Int = 2,
                                ν_coarsest::Int = 40,
                                smoother::Symbol = :gs_kaczmarz,
                                do_recomb::Bool = true,
                                history_size::Int = 4,
                                γ::Real = 1.5,
                                γ_coarse::Real = 1.5,
                                γ_coarse_growth::Real = 0.7)
    mfp = hier.levels[1]
    αd = α .* mfp.d
    αd_norm = max(norm(αd), 1e-30)
    n = size(mfp.A, 1)
    L = num_levels(hier)
    res_vec = αd .- mfp.A * φ
    res = norm(res_vec)
    history = Float64[res]
    cycles = 0
    # Per-level iterate history (persists across cycles at the finest;
    # cleared at descent for coarse levels — mirrors LAMG solve_cycle.jl).
    level_hist = if do_recomb
        h = Vector{Any}(undef, L)
        for k in 1:L
            n_k = size(hier.levels[k].A, 1)
            h[k] = IterateHistory(n_k, history_size)
        end
        h
    else
        nothing
    end
    do_recomb && save_iterate!(level_hist[1], φ, res_vec)
    # Per-level γ schedule (paper §3.6). γ_vec[k] = cycle index at level k
    # → level k+1 transition.
    γ_vec = Float64[]
    if L > 1
        γ_vec = Vector{Float64}(undef, L - 1)
        γ_vec[1] = Float64(γ)
        @inbounds for i in 2:(L - 1)
            # Bounded by ratio between consecutive levels to keep work O(m).
            τ = size(hier.levels[i + 1].A, 1) / max(1, size(hier.levels[i].A, 1))
            γ_cap = τ > 0 ? 0.95 / τ : 3.0
            γ_vec[i] = min(γ_coarse * γ_coarse_growth ^ (i - 2), γ_cap, 3.0)
            γ_vec[i] = max(γ_vec[i], γ_coarse)
        end
    end
    while res > tol * αd_norm && cycles < max_cycles
        fas_multilevel_cycle!(hier, φ; α = α, ν_pre = ν_pre, ν_post = ν_post,
                               ν_coarsest = ν_coarsest, smoother = smoother,
                               γ_vec = γ_vec, history = level_hist,
                               do_recomb = do_recomb)
        res_vec = αd .- mfp.A * φ
        if do_recomb
            min_res!(level_hist[1], φ, res_vec)
            res_vec = αd .- mfp.A * φ
            save_iterate!(level_hist[1], φ, res_vec)
        end
        res = norm(res_vec)
        push!(history, res)
        cycles += 1
    end
    return cycles, history
end

# ---------------------------------------------------------- FMG-FAS

"""
    fmg_fas!(hier, φ; α = 0.0, ν_pre = 2, ν_post = 2,
              ν_per_level = 1, ν_coarsest = 40,
              coarsest_tol::Union{Nothing,Real} = nothing)
        -> (φ_finest, level_residuals)

Classic 1-FMG-FAS initial-guess pass for the constrained-linear
max-flow problem. Provides a good starting point for an outer
nonlinear iteration (or just a single high-quality solve).

Algorithm:
  1. **Solve at the coarsest level** to tolerance: relax `ν_coarsest`
     times (or `until‖A_L φ_L − α·d_L‖ ≤ coarsest_tol · ‖α·d_L‖`).
     Starts from φ_L = 0.
  2. **For each finer level** `l = L−1, L−2, …, 1`:
     a. **Interpolate** the previous-level iterate up: φ_l = P_l · φ_{l+1}.
     b. **Do exactly `ν_per_level` V-cycle(s)** rooted at level `l`.
        Classic 1-FMG = `ν_per_level = 1`.
  3. Return φ at the finest level.

`φ` is mutated in place at the finest. The level-by-level residual
history is returned for diagnostics.

This is the *initial-guess* pass; nonlinear outer iterations (e.g.,
the α-update loop) should call `fas_multilevel_solve!` after FMG-FAS
to converge to higher accuracy.

For the linear case (no constraints active), `fmg_fas!` typically
reduces the residual to within 1-2 cycles' worth of tol immediately
— giving the nonlinear outer loop a head start.
"""
function fmg_fas!(hier::NLFHierarchy, φ::AbstractVector;
                  α::Real = 0.0,
                  ν_pre::Int = 2, ν_post::Int = 2,
                  ν_per_level::Int = 1,
                  ν_coarsest::Int = 40,
                  coarsest_tol::Union{Nothing,Real} = nothing)
    L = num_levels(hier)
    @assert length(φ) == size(hier.levels[1].A, 1)
    level_residuals = Float64[]

    # 1. Solve at coarsest. Start from zero.
    mfp_L = hier.levels[L]
    φ_L = zeros(size(mfp_L.A, 1))
    αd_L = α .* mfp_L.d
    # ν_coarsest GS+Kaczmarz sweeps (Kaczmarz off — coarse bounds aren't
    # inherited at the standalone-FMG coarsest, no τ-correction has been
    # computed yet; treat as operator-only solve like fas cycle does).
    relax_gs_kaczmarz!(mfp_L, φ_L, mfp_L.d, α; sweeps = ν_coarsest,
                       enforce_constraints = false)
    push!(level_residuals, norm(mfp_L.A * φ_L - αd_L))

    # 2. Prolongate level-by-level upward, doing ν_per_level V-cycles.
    φ_prev = φ_L
    for l in (L - 1):-1:1
        mfp_l = hier.levels[l]
        P_l = hier.P[l]
        # Prolongate: φ_l = P_l · φ_{l+1}.
        φ_l = Vector{Float64}(P_l * φ_prev)
        # Do ν_per_level V-cycles rooted at level l. For the FAS visit
        # we need to pretend `hier` starts at level l; the cleanest way
        # is to do a partial _fas_visit! with the slice of the
        # hierarchy. But our hier is laid out finest-first. Simplest:
        # use the full hierarchy and do a V-cycle anchored at level l —
        # which means walking down from level l using b_l = α · d_l and
        # bounds = mfp_l.low, mfp_l.high.
        for _ in 1:ν_per_level
            _fas_visit_from_level!(hier, l, φ_l, α; ν_pre = ν_pre,
                                    ν_post = ν_post, ν_coarsest = ν_coarsest)
        end
        αd_l = α .* mfp_l.d
        push!(level_residuals, norm(mfp_l.A * φ_l - αd_l))
        φ_prev = φ_l
        # On reaching the finest, copy into the user's φ buffer.
        if l == 1
            φ .= φ_l
        end
    end
    return φ, level_residuals
end

"""
    _fas_visit_from_level!(hier, l, φ_l, α; ν_pre, ν_post, ν_coarsest)

Run a single V-cycle anchored at level `l` of `hier`, using
`b_l = α · d_l` as the RHS and `mfp_l.low/high` as the original bounds.
Used internally by `fmg_fas!`.
"""
function _fas_visit_from_level!(hier::NLFHierarchy, l::Int,
                                 φ_l::AbstractVector, α::Real;
                                 ν_pre::Int, ν_post::Int, ν_coarsest::Int)
    mfp_l = hier.levels[l]
    b_l = Vector{Float64}(α .* mfp_l.d)
    low_l = Vector{Float64}(mfp_l.low)
    high_l = Vector{Float64}(mfp_l.high)
    _fas_visit!(hier, l, φ_l, b_l, low_l, high_l,
                ν_pre, ν_post, ν_coarsest, l == 1)
    return φ_l
end

# ---------------------------------------------------------- α-LP at coarsest

"""
    solve_alpha_max(mfp::NLFProblem) -> (α_max::Float64, φ::Vector{Float64})

!!! warning "This is the gradient/electrical *relaxation*, NOT the true max-flow."
    It restricts the flow to a potential gradient `f = Bᵀφ` (the minimal-dissipation
    electrical flow), a strict relaxation of max-flow: on heterogeneous-capacity graphs it
    returns only `0.02–0.20 × F*`. It equals the true max-flow only when the optimum is itself
    a gradient flow (e.g. a 1-D chain). For the **true** combinatorial max-flow use
    [`nlf_maxflow`](@ref) (smooth-ρ α-continuation). Kept as a fast lower bound / coarsest-level
    estimate.

Solve the small LP

    maximize α
    s.t.  A φ = α · d
          low_e ≤ (B^T φ)_e ≤ high_e   ∀ e ∈ E.

Since `A` is a graph Laplacian (rank n − 1, nullspace = constant) and
`d = e_t − e_s` is in range(A), the solution `φ(α)` is **unique up to
a constant** for each α. The constant shift doesn't affect the edge
gradients `(B^T φ)_e` (because `B^T·1 = 0`), so the box constraints
reduce to scalar inequalities in α alone.

Concretely: solve once for the "unit" potential `u = A⁺ · d` (sum-zero
solution of `A u = d`). Then `φ(α) = α · u + c·1` for any `c`, and
the edge gradient is `(B^T φ)(α) = α · (B^T u)`. The feasibility
constraints become

    low_e ≤ α · (B^T u)_e ≤ high_e   ∀ e.

For each edge, this gives an interval `[α_lo(e), α_hi(e)]` for α
(with sign flips when `(B^T u)_e < 0`). The intersection is an
interval `[α_min, α_max_box]`; the answer is `α_max_box` (or
`+Inf` if the box is unbounded above).

**Cost**: O(n³) for one Cholesky/LU of A_L on the coarsest (n ≤ 20)
+ O(m) for the per-edge intersection. Negligible.

Returns `(α_max, φ)` with `φ` zero-mean.
"""
function solve_alpha_max(mfp::NLFProblem)
    A = mfp.A; B = mfp.B
    n = size(A, 1); m = length(mfp.head)
    d = mfp.d
    d_proj = d .- (sum(d) / n)    # ensure exactly sum-zero (numerical hygiene)
    # Solve A · u = d_proj in the constant-orthogonal complement.
    # Use a tiny ridge: (A + ε·I) u = d_proj. For Laplacians this projects
    # away the constant null direction at O(ε) precision. Robust to
    # singular A[2:n, 2:n] (e.g., when index 1 is a hub).
    ε = 1e-12 * (1 + sum(abs, nonzeros(A)) / max(nnz(A), 1))
    A_reg = A + ε * sparse(I, n, n)
    u = A_reg \ Vector{Float64}(d_proj)
    # Zero-mean (kills the residual ε contribution).
    u .-= sum(u) / n
    # Edge gradients per unit α.
    Bu = B' * u                   # length m
    # Scan each edge to bracket α.
    α_lo = -Inf; α_hi = +Inf
    @inbounds for e in 1:m
        coeff = Bu[e]
        if abs(coeff) < 1e-14
            # Bound trivially satisfied iff low_e ≤ 0 ≤ high_e.
            (mfp.low[e] > 0.0 || mfp.high[e] < 0.0) && return (0.0, zeros(n))
            continue
        end
        # low ≤ α·coeff ≤ high
        if coeff > 0
            α_lo = max(α_lo, mfp.low[e] / coeff)
            α_hi = min(α_hi, mfp.high[e] / coeff)
        else
            # coeff < 0: flip
            α_lo = max(α_lo, mfp.high[e] / coeff)
            α_hi = min(α_hi, mfp.low[e] / coeff)
        end
    end
    α_hi < α_lo && return (NaN, zeros(n))   # infeasible
    # We want max α; if α_hi = +Inf the box is unbounded above (unbounded LP).
    α_max = isfinite(α_hi) ? α_hi : 1.0     # fallback: any feasible α
    φ = α_max .* u
    φ .-= sum(φ) / n
    return (α_max, φ)
end

# ---------------------------------------------------------- FMG-FAS with α-update

"""
    fmg_fas_alpha_opt!(hier, φ; ν_pre = 2, ν_post = 2,
                       ν_per_level = 1, ν_coarsest = 40)
        -> (α_optimal::Float64, φ_finest, level_residuals)

Variant of `fmg_fas!` that *optimizes α at the coarsest level*. The
returned α (and the corresponding φ) is the max-flow solution
projected to the coarsest grid; the finer levels refine φ at that
α via standard V-cycles.

Algorithm:
  1. **Coarsest**: solve the small LP `solve_alpha_max(mfp_L)`.
     Gives `(α, φ_L)`.
  2. **For each finer level** `l = L−1, L−2, …, 1`:
       a. Prolongate `φ_l = P_l · φ_{l+1}` (α is just copied — α is
          a scalar, no prolongation needed).
       b. Do `ν_per_level` V-cycles at level `l` with this α.
  3. Return `α`, the finest `φ`, and the per-level residual history.

This fixes the μ=1 stalls in genrmf, acyclic, and similar saturating
test cases where a guessed fixed α was either too high (immediate
saturation) or too low (wasted iterations). The coarse LP gives the
*exact* maximal α for the coarsest topology; prolongation may
slightly exceed it on the finer grid (boundary effects of caliber-1
PC), and the V-cycles handle that mismatch.
"""
function fmg_fas_alpha_opt!(hier::NLFHierarchy, φ::AbstractVector;
                            ν_pre::Int = 2, ν_post::Int = 2,
                            ν_per_level::Int = 1,
                            ν_coarsest::Int = 40)
    L = num_levels(hier)
    @assert length(φ) == size(hier.levels[1].A, 1)
    level_residuals = Float64[]

    # 1. Solve the small LP at the coarsest level.
    mfp_L = hier.levels[L]
    α, φ_L = solve_alpha_max(mfp_L)
    # If LP infeasible (e.g., disconnected coarse), fall back to α = 0.
    !isfinite(α) && (α = 0.0)
    push!(level_residuals, norm(mfp_L.A * φ_L .- α .* mfp_L.d))

    # Single-level hierarchy: the LP is on the only level — just copy out.
    if L == 1
        φ .= φ_L
        return α, φ, level_residuals
    end

    # 2. Prolongate level-by-level; do ν_per_level V-cycles at each.
    φ_prev = φ_L
    for l in (L - 1):-1:1
        mfp_l = hier.levels[l]
        P_l = hier.P[l]
        φ_l = Vector{Float64}(P_l * φ_prev)
        for _ in 1:ν_per_level
            _fas_visit_from_level!(hier, l, φ_l, α;
                                    ν_pre = ν_pre, ν_post = ν_post,
                                    ν_coarsest = ν_coarsest)
        end
        αd_l = α .* mfp_l.d
        push!(level_residuals, norm(mfp_l.A * φ_l .- αd_l))
        φ_prev = φ_l
        l == 1 && (φ .= φ_l)
    end
    return α, φ, level_residuals
end

# ---------------------------------------------------------- α-continuation

"""
    fmg_fas_alpha_continuation!(hier, φ; n_outer = 10, tol_outer = 1e-6,
                                ν_cycles_per_outer = 5, ν_pre = 2, ν_post = 2,
                                ν_coarsest = 40)
        -> (α, φ, history)

α-continuation outer loop. Each outer iteration:
  1. Run FMG-FAS-α-opt to get a starting α.
  2. Refine φ at that α via `ν_cycles_per_outer` finest V-cycles.
  3. Check feasibility + residual on fine.
  4. Adjust α (shrink if infeasible, grow if feasible-but-not-max).

For multi-level cases where the coarsest LP's α overshoots the fine
feasibility, this outer loop tightens it.
"""
function fmg_fas_alpha_continuation!(hier::NLFHierarchy, φ::AbstractVector;
                                     n_outer::Int = 10,
                                     tol_outer::Real = 1e-6,
                                     ν_cycles_per_outer::Int = 5,
                                     ν_pre::Int = 2, ν_post::Int = 2,
                                     ν_coarsest::Int = 40)
    mfp = hier.levels[1]
    history = Vector{NamedTuple}()
    α, _, _ = fmg_fas_alpha_opt!(hier, φ; ν_pre = ν_pre, ν_post = ν_post,
                                  ν_per_level = 1, ν_coarsest = ν_coarsest)
    α_best = α; φ_best = copy(φ)
    for it in 1:n_outer
        for _ in 1:ν_cycles_per_outer
            fas_multilevel_cycle!(hier, φ; α = α, ν_pre = ν_pre,
                                   ν_post = ν_post, ν_coarsest = ν_coarsest)
        end
        grad = mfp.B' * φ
        αd_norm = max(norm(α .* mfp.d), 1e-30)
        rel_res = norm(mfp.A * φ - α .* mfp.d) / αd_norm
        n_viol = sum((grad .< mfp.low .- 1e-6) .| (grad .> mfp.high .+ 1e-6))
        push!(history, (iter = it, α = α, rel_res = rel_res, n_viol = n_viol))
        if rel_res < tol_outer && n_viol == 0
            α_best = α; φ_best .= φ
            break
        end
        if n_viol > 0
            α *= 0.7      # shrink — too much α saturates bounds
        else
            α_best = α; φ_best .= φ
            α *= 1.1     # grow — feasible, try harder
        end
    end
    φ .= φ_best
    return α_best, φ, history
end

"""
    nlf_alpha_from_below(mfp; n_min=20, max_cycles=60, tol_res=1e-5,
                             tol_box=1e-4, tol_alpha=1e-4, rng=...) -> (α, φ)

!!! warning "Computes the gradient/electrical *relaxation*, NOT the true max-flow."
    This bisects α on the **constrained-linear** problem `Aφ=αd`, `low≤Bᵀφ≤high` (fixed
    `A=BᵀB`), whose flow `f=Bᵀφ` is a potential gradient — the electrical relaxation, only
    `0.02–0.20 × F*` on heterogeneous graphs (exact only for 1-D chains). For the **true**
    combinatorial max-flow use [`nlf_maxflow`](@ref). Retained as a fast lower bound.

Compute the relaxation value `α` and a feasible potential `φ` by **α-continuation from
below**, using the constrained-linear FAS multilevel solver as the inner kernel.

Rationale (Brandt-school, see `doc/maxflow_fas_plan.md`): for the constrained-
linear problem `Aφ = α d`, `low ≤ Bᵀφ ≤ high`, the operator `A` is fixed, so a
single hierarchy is valid for all `α` (no per-α recoarsening needed). A fixed-`α`
solve is *feasible* iff `α ≤ F*`; for `α > F*` the box cannot sustain the demand
and the equation residual stalls. We therefore bisect `α` on `[0, α_hi]` (with
`α_hi` = total capacity incident to the source — a trivial s-cut upper bound) and
accept `α` iff the fixed-`α` solve both (i) converges (`‖Aφ−αd‖` reduced below
`tol_res`) AND (ii) leaves the flow inside the box (`max box violation < tol_box`).
The largest accepted `α` is `F*`. This cannot overshoot `F*` (unlike optimizing
`α` at the coarsest level, whose coarse min-cut can exceed the fine `F*`).

The box check (ii) is essential: the projected smoother does not always clamp the
box tightly on bottleneck-dominated graphs, so residual convergence alone is not
sufficient to certify feasibility.

This is the correct working baseline; the V-driven continuation (well-conditioned
near the `α=F*` limit point) is the planned refinement.
"""
function nlf_alpha_from_below(mfp::NLFProblem;
                                  n_min::Int = 20,
                                  max_cycles::Int = 60,
                                  tol_res::Real = 1e-5,
                                  tol_box::Real = 1e-4,
                                  tol_alpha::Real = 1e-4,
                                  rng = Random.MersenneTwister(0xfa11))
    n = size(mfp.A, 1)
    hier = setup_nlf_hierarchy(mfp; n_min = n_min, rng = rng)
    s = mfp.s
    # Upper bound on F*: total capacity of edges incident to the source.
    α_hi = 0.0
    @inbounds for e in 1:length(mfp.head)
        if mfp.head[e] == s || mfp.tail[e] == s
            α_hi += max(mfp.high[e], -mfp.low[e])
        end
    end
    α_lo = 0.0
    φ_best = zeros(n)
    while α_hi - α_lo > tol_alpha * max(α_hi, 1e-9)
        α = 0.5 * (α_lo + α_hi)
        φ = zeros(n)
        _, rs = fas_multilevel_solve!(hier, φ; α = α, tol = tol_res,
                                      max_cycles = max_cycles)
        g = mfp.B' * φ
        boxviol = maximum(max.(g .- mfp.high, mfp.low .- g, 0.0))
        rel = rs[end] / max(rs[1], 1e-30)
        if rel < tol_res && boxviol < tol_box
            α_lo = α; φ_best = φ
        else
            α_hi = α
        end
    end
    return α_lo, φ_best
end
