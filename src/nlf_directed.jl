"""
Phase 3 — classical max-flow on directed networks via primal-dual
FMG-FAS (Option B from doc/maxflow_directed_fas_options.md).

Variables: (f, alpha, phi) where
  f in R^m       = edge flows
  alpha in R     = flow value (global; updated at coarsest only)
  phi in R^n     = node potentials (Lagrange multiplier for B f = alpha d)

Lagrangian (saddle-point):
    L(f, alpha, phi) = alpha + phi^T (B f - alpha d)
    max over (f, alpha) with 0 <= f <= c, min over phi.

Smoother: Arrow-Hurwicz with augmented-Lagrangian stabilisation. At
fixed alpha (non-coarsest levels), each sweep:

  1. f update (projected gradient ascent, augmented-Lagrangian):
       f_e <- clip(f_e + w * (B^T phi)_e - w*rho*(B^T (B f - alpha d))_e,
                   low_e, high_e)
  2. phi update (Kaczmarz on equality):
       phi_v <- phi_v - w * (B f - alpha d)_v / A_vv     for v in V.

alpha is updated only at the coarsest level (via a small LP).

Kept separate from the undirected nlf_constrained.jl solver. The
undirected solver (phi-only, gradient-flow) is unchanged.
"""

# ---------------------------------------------------------- struct

"""
    DirectedMaxFlowState

State for the directed primal-dual solver. Holds the iterate (f, α, φ)
and (optionally) workspace arrays.
"""
mutable struct DirectedMaxFlowState
    f::Vector{Float64}          # edge flows (length m)
    α::Float64                  # scalar flow value
    φ::Vector{Float64}          # node potentials (length n)
end

DirectedMaxFlowState(mfp::NLFProblem) =
    DirectedMaxFlowState(zeros(length(mfp.head)), 0.0, zeros(size(mfp.A, 1)))

# ---------------------------------------------------------- smoother

"""
    relax_arrow_hurwicz!(mfp, state; α = state.α, sweeps = 1,
                         ω = 0.25, ρ = 1.0)

Primal-dual smoother sweep (Arrow-Hurwicz + augmented-Lagrangian
stabilisation) at a *fixed* α. Updates `state.f` and `state.φ` in
place.

Defaults: `ω = 0.25` (damping; safe for ||B|| ≤ sqrt(8) on
typical graphs), `ρ = 1.0` (augmented-Lagrangian penalty).
"""
function relax_arrow_hurwicz!(mfp::NLFProblem, state::DirectedMaxFlowState;
                              α::Real = state.α,
                              sweeps::Int = 1,
                              ω::Real = 0.25,
                              ρ::Real = 1.0)
    B = mfp.B
    n = size(B, 1); m = length(state.f)
    @assert length(state.φ) == n
    @assert length(state.f) == m
    αd = α .* mfp.d

    # Diagonal of A = B B^T (degree, possibly weighted).
    A_diag = [mfp.A[v, v] for v in 1:n]
    A_diag .= max.(A_diag, 1e-12)

    @inbounds for _ in 1:sweeps
        # 1. f update.
        # First compute r = B f − α d, then (B^T r) and (B^T φ).
        r = B * state.f .- αd
        Btr = B' * r
        Btφ = B' * state.φ
        for e in 1:m
            df = ω * (Btφ[e] - ρ * Btr[e])
            state.f[e] = clamp(state.f[e] + df, mfp.low[e], mfp.high[e])
        end

        # 2. φ update.
        r = B * state.f .- αd
        for v in 1:n
            state.φ[v] -= ω * r[v] / A_diag[v]
        end
    end
    return state
end

# ---------------------------------------------------------- Chambolle-Pock variant

"""
    relax_chambolle_pock!(mfp, state; α, sweeps = 1, σ = 1.0, τ = 0.25, θ = 1.0)

Chambolle-Pock 2011 primal-dual with extrapolation. Provable O(1/k)
ergodic rate; typically far better than Arrow-Hurwicz on indefinite
saddle systems. Setup:

  primal (max): (f, alpha)  with f in [low, high]
  dual   (min): phi (free, n-dim)
  coupling: phi^T (B f - alpha d)

Algorithm (per sweep, with theta = 1 standard):
  1. phi^{k+1} = phi^k + sigma * (B f_bar^k - alpha_bar^k * d) / A_diag
  2. f^{k+1}_e = clip(f^k_e + tau * (B^T phi^{k+1})_e, low_e, high_e)
  3. alpha^{k+1} = alpha^k + tau * (1 - phi^{k+1}^T d)        [coarsest only]
  4. f_bar^{k+1} = f^{k+1} + theta * (f^{k+1} - f^k)          (extrapolation)
     alpha_bar^{k+1} = alpha^{k+1} + theta * (alpha^{k+1} - alpha^k)
  5. (also can extrapolate phi instead — symmetric variants exist)

Requires sigma * tau * ||K||^2 < 1 with K = B. Conservative: sigma * tau < 1/(2 * max_degree).

`α` is held fixed at the call site (non-coarsest levels); update at
coarsest only.
"""
function relax_chambolle_pock!(mfp::NLFProblem, state::DirectedMaxFlowState;
                                α::Real = state.α,
                                sweeps::Int = 1,
                                σ::Real = 1.0,
                                τ::Real = 0.25,
                                θ::Real = 1.0)
    B = mfp.B
    n = size(B, 1); m = length(state.f)
    αd = α .* mfp.d
    A_diag = [mfp.A[v, v] for v in 1:n]
    A_diag .= max.(A_diag, 1e-12)

    # Workspace: previous f and its extrapolation.
    f_prev = copy(state.f)
    f_bar  = copy(state.f)

    @inbounds for _ in 1:sweeps
        # 1. phi update via Bf_bar - alpha d residual.
        r_bar = B * f_bar .- αd
        for v in 1:n
            state.φ[v] += σ * r_bar[v] / A_diag[v]
        end
        # 2. f update.
        Btφ = B' * state.φ
        for e in 1:m
            f_new = clamp(state.f[e] + τ * Btφ[e], mfp.low[e], mfp.high[e])
            f_prev[e] = state.f[e]
            state.f[e] = f_new
        end
        # 3. extrapolation: f_bar = f_new + theta * (f_new - f_prev).
        @. f_bar = state.f + θ * (state.f - f_prev)
    end
    return state
end

"""
    shrinkage_chambolle_pock(mfp; α, sweeps_max=40, num_examples=2,
                              σ_list=…, τ_list=…)
        -> (best_μ_op, best_μ_full, best_σ, best_τ)

Same idea as `shrinkage_arrow_hurwicz` but for Chambolle-Pock. Sweeps
over (σ, τ) pairs to find the best for each instance.
"""
function shrinkage_chambolle_pock(mfp::NLFProblem;
                                   α::Real,
                                   sweeps_max::Int = 40,
                                   num_examples::Int = 2,
                                   σ_list = (2.0, 1.0, 0.5, 0.25),
                                   τ_list = (0.5, 0.25, 0.125),
                                   rng = Random.default_rng())
    B = mfp.B
    n = size(B, 1); m = length(mfp.head)
    αd = α .* mfp.d
    function box_violation_norm(f)
        v = 0.0
        @inbounds for e in 1:m
            if f[e] < mfp.low[e]
                v += (mfp.low[e] - f[e])^2
            elseif f[e] > mfp.high[e]
                v += (f[e] - mfp.high[e])^2
            end
        end
        return sqrt(v)
    end
    best = (Inf, Inf, NaN, NaN)
    for σ in σ_list, τ in τ_list
        # Stability: στ ||K||² < 1; ||K||² ≈ max_degree ≈ A_max_diag.
        max_deg = maximum(mfp.A[v, v] for v in 1:n)
        σ * τ * max_deg >= 1 && continue
        op_factors = Float64[]; full_factors = Float64[]
        diverged = false
        for _ in 1:num_examples
            state = DirectedMaxFlowState(mfp); state.α = α
            @inbounds for e in 1:m
                state.f[e] = 0.5 * (mfp.low[e] + mfp.high[e])
            end
            state.φ .= randn(rng, n); state.φ .-= sum(state.φ)/n
            op_hist  = Float64[norm(B * state.f .- αd)]
            full_hist = Float64[sqrt(op_hist[1]^2 + box_violation_norm(state.f)^2)]
            for _ in 1:sweeps_max
                relax_chambolle_pock!(mfp, state; α = α, sweeps = 1,
                                       σ = σ, τ = τ)
                r_op = norm(B * state.f .- αd)
                r_full = sqrt(r_op^2 + box_violation_norm(state.f)^2)
                push!(op_hist, r_op)
                push!(full_hist, r_full)
                if !isfinite(r_op) || r_op > 1e10 * op_hist[1]
                    diverged = true; break
                end
                r_full < 1e-12 * full_hist[1] && break
            end
            diverged && break
            function geomean_tail(hist)
                ratios = hist[2:end] ./ max.(hist[1:end - 1], 1e-30)
                tail = ratios[max(end - 3, 1):end]
                return exp(mean(log.(max.(tail, 1e-30))))
            end
            push!(op_factors, geomean_tail(op_hist))
            push!(full_factors, geomean_tail(full_hist))
        end
        diverged && continue
        isempty(full_factors) && continue
        μ_op = mean(op_factors); μ_full = mean(full_factors)
        if μ_full < best[2]
            best = (μ_op, μ_full, σ, τ)
        end
    end
    return best
end

# ---------------------------------------------------------- shrinkage

"""
    shrinkage_arrow_hurwicz(mfp; α, sweeps_max = 30, num_examples = 4,
                            ω = 0.25, ρ = 1.0, rng = …)
        -> (μ_op, μ_full)

Asymptotic per-sweep convergence factor of the Arrow-Hurwicz smoother
at fixed `α`. Returns:
- `μ_op`   : reduction rate of the operator residual `‖Bf − α·d‖`.
- `μ_full` : reduction rate of √(‖Bf − α·d‖² + ‖box-violation‖²).

For each of `num_examples` random feasible starting points (f_0, phi_0),
run up to `sweeps_max` Arrow-Hurwicz sweeps and compute the
geometric mean of per-sweep residual ratios over the last few.
"""
function shrinkage_arrow_hurwicz(mfp::NLFProblem;
                                  α::Real,
                                  sweeps_max::Int = 30,
                                  num_examples::Int = 4,
                                  ω::Real = 0.25,
                                  ρ::Real = 1.0,
                                  rng = Random.default_rng())
    B = mfp.B
    n = size(B, 1); m = length(mfp.head)
    αd = α .* mfp.d

    function box_violation_norm(f)
        v = 0.0
        @inbounds for e in 1:m
            if f[e] < mfp.low[e]
                v += (mfp.low[e] - f[e])^2
            elseif f[e] > mfp.high[e]
                v += (f[e] - mfp.high[e])^2
            end
        end
        return sqrt(v)
    end

    op_factors = Float64[]; full_factors = Float64[]
    for _ in 1:num_examples
        state = DirectedMaxFlowState(mfp)
        state.α = α
        # Initialize f to mid-box (low + 0.5·(high−low)), φ random.
        @inbounds for e in 1:m
            state.f[e] = 0.5 * (mfp.low[e] + mfp.high[e])
        end
        state.φ .= randn(rng, n); state.φ .-= sum(state.φ)/n
        op_hist  = Float64[norm(B * state.f .- αd)]
        full_hist = Float64[sqrt(op_hist[1]^2 + box_violation_norm(state.f)^2)]
        for _ in 1:sweeps_max
            relax_arrow_hurwicz!(mfp, state; α = α, sweeps = 1, ω = ω, ρ = ρ)
            r_op = norm(B * state.f .- αd)
            r_full = sqrt(r_op^2 + box_violation_norm(state.f)^2)
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
