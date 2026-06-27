"""
True max-flow via **smooth-ρ α-continuation**.

This is the NLF max-flow solver. It computes the *true* (combinatorial) max-flow
`F*`, unlike the gradient/electrical relaxation `solve_alpha_max` /
`nlf_alpha_from_below` (which restrict `f = Bᵀφ` to potential gradients and yield
only `0.1–0.6 × F*` on heterogeneous-capacity graphs).

## Formulation

Smooth saturating flux law (per edge, asymmetric forward/reverse capacities):

    ρ_e(g) = c_e · tanh(g / c_e),   c_e = high_e if g ≥ 0 else −low_e
    f_e    = ρ_e((Bᵀφ)_e)              ∈ (low_e, high_e)  (never saturates exactly)

Conservation with throughput `α`:  `B ρ(Bᵀφ) = α d`,  `d = e_t − e_s`.
Because `f = ρ(Bᵀφ)` is a *nonlinear* function of the gradient it is **not** a
gradient flow, so it can realise the true max-flow. The energy
`E_α(φ) = Σ_e Ψ_e((Bᵀφ)_e) − α dᵀφ`, `Ψ_e = ∫₀ᵍ ρ_e` (a `c²logcosh` soft barrier)
has recession slope `c_e` along every edge direction *independent of any sharpness*,
so a finite minimiser exists **iff `α < min-cut = F*`** (max-flow/min-cut duality).

## Why α-continuation, not voltage-driving

Voltage-driving (pin `φ_s = V`, read off `α`) converges to the Thomson / minimal-
dissipation *electrical* flow, which is sub-maximal — the electrical current does not
saturate every min-cut edge. Forcing the throughput `α` (this solver) reaches `F*`
exactly: a fixed-`α` smooth-ρ solve is feasible iff `α < F*`, so the largest feasible
`α` is `F*`. We find it by predictor–corrector continuation in `α` (geometric step,
warm-started Newton; grow the step on success, halve on the first infeasible `α > F*`).

## The α→F* limit point (benign in practice)

As `α ↑ F*` the min-cut edges saturate (`ρ'_e → 0`), `J = B diag(ρ') Bᵀ` becomes
singular along the **min-cut indicator χ**, `φ → ∞` along χ, and `κ(J) ∼ 1/(F*−α)`,
`V ∼ −log(F*−α)`. Two facts keep this benign: (i) the warm-started Newton corrector
converges in a handful of iterations down to `F*−α ∼ 10⁻⁶` (a direct factorisation is
oblivious to `κ` until `~10¹⁵`); and (ii) `J` is a bona-fide weighted graph Laplacian
whose only *weak* edges are the cut, so LAMG+'s relaxation-based affinity refuses to
aggregate across the `ρ'≈0` cut and the inner solve converges at `μ ≈ 0.01` *uniformly*
in `α/F*` (`inner = :multigrid`). The single genuinely-singular direction is the
*scalar* cut amplitude; Brandt's cut-mode bordering ("subtraction of the singularity",
1984 Guide §5.6/§5.7, Bratu turning point §8.3.2) removes it exactly and is validated
in `scripts/bench_env/run_nlf_cusp.jl` — but it is unnecessary here precisely because
the inner solver already represents the cut mode.

Everything runs in the zero-mean gauge (`Σφ = 0`); `J` is then a singular graph
Laplacian (null space = constants) that LAMG consumes directly.
"""

# Smooth saturating law and its derivative. Writes into f, dρ in place.
@inline function _nlf_rho_drho!(f::Vector{Float64}, dρ::Vector{Float64},
                                 g::AbstractVector{Float64},
                                 low::Vector{Float64}, high::Vector{Float64})
    @inbounds for e in eachindex(g)
        c = g[e] >= 0 ? high[e] : -low[e]
        th = tanh(g[e] / c)
        f[e]  = c * th
        dρ[e] = 1 - th * th        # ρ'(g) = sech²(g/c) ∈ (0,1]
    end
    return nothing
end

# Weighted graph Laplacian J = B diag(ρ') Bᵀ (floor weak weights so it stays connected).
_nlf_jacobian(B, dρ; floor = 1e-12) = B * Diagonal(max.(dρ, floor)) * B'

_nlf_zeromean!(x) = (x .-= sum(x) / length(x); x)

"""
    nlf_maxflow(mfp; tol=1e-7, max_steps=200, α0_frac=0.1, inner=:direct,
                 verbose=false) -> (F, φ, f, info)

Compute the true max-flow value `F` (= `F*`), zero-mean node potentials `φ`, and the
feasible edge flow `f = ρ(Bᵀφ)` (`B f = F d` to tolerance, `low ≤ f ≤ high`).

`inner` selects the Newton inner linear engine — the framework is engine-agnostic:
- `:direct`     — augmented sparse Cholesky, refactorised each solve (reference);
- `:multigrid`  — the built-in cut-respecting LAMG+ caliber-2 hierarchy (the default engine);
- a `NLFLinearEngine(build, apply, name)` — any caller-injected solver (e.g. approxChol),
  WITHOUT the LAMG package depending on it (dependency inversion). `build(J)->state` builds the
  reusable frozen factor/hierarchy; `apply(state, J, b)->x` solves `J x = b` (x ⊥ 1).
For every reusable engine (all but `:direct`), `refresh` controls lazy reuse: the frozen state is
rebuilt only when the observed per-step residual factor exceeds `refresh` or after a stale-state
stall (`refresh = 0` rebuilds every step).

`info = (steps, residual, converged, voltage, alphas)`.
"""
function nlf_maxflow(mfp::NLFProblem;
                      tol::Real = 1e-7,
                      max_steps::Int = 200,
                      α0_frac::Real = 0.1,
                      inner = :direct,
                      method::Symbol = :arclength,
                      refresh::Real = 0.25,
                      verbose::Bool = false)
    B = mfp.B; n = size(B, 1); m = size(B, 2)
    d = Vector{Float64}(mfp.d); low = mfp.low; high = mfp.high
    f = zeros(m); dρ = zeros(m)

    # crude s-cut upper bound on F* (total capacity incident to the source)
    α_hi = 0.0
    @inbounds for e in 1:m
        (mfp.head[e] == mfp.s || mfp.tail[e] == mfp.s) &&
            (α_hi += max(high[e], -low[e]))
    end
    tol_rel = max(float(tol), 1e-7)
    feasible(α, φ) = (g = B' * φ; _nlf_rho_drho!(f, dρ, g, low, high);
                      norm(B * f .- α .* d) <= tol_rel * max(α, 1.0))

    # ---- grid-FMG (the one shot): the graph hierarchy IS the continuation ----
    # Solve the small max-flow on the coarsest level (near F*, sharp cut mode), prolongate
    # (φ,α) up, finish each level with a few arclength steps from the seed — no per-level
    # parameter chain. α is the global unknown carried through the levels.
    if method === :fmg
        α, φ, steps, converged = _nlf_fmg!(mfp; inner = inner, tol = tol,
                                            max_steps = max_steps, verbose = verbose)
        φ = _nlf_zeromean!(φ)
        g = B' * φ; _nlf_rho_drho!(f, dρ, g, low, high)
        info = (steps = steps, residual = norm(B * f .- α .* d), converged = converged,
                voltage = φ[mfp.s] - φ[mfp.t], alphas = Float64[α])
        return α, φ, copy(f), info
    end

    # ---- 1) warm start: a feasible α, then CLIMB to bracket F* ----
    # The continuation must not start far below F* (the tangent/cut direction is then
    # ill-defined — e.g. washington stalls). First halve to a feasible α (α_hi can far
    # exceed F*); then climb geometrically, warm-started, keeping the largest feasible
    # (α,φ), until a step turns infeasible (α ≥ F*). This lands the continuation near
    # F* in O(log) cheap solves — a 1-D nested iteration in the parameter.
    α = α0_frac * α_hi
    φ = zeros(n); found = false
    for _ in 1:40
        φt, ok = _nlf_alpha_newton(B, d, α, zeros(n), low, high, f, dρ; inner = inner, refresh = refresh)
        if ok && feasible(α, φt); φ = φt; found = true; break; end
        α *= 0.5
    end
    found || return (0.0, zeros(n), zeros(m),
                     (steps = 0, residual = Inf, converged = false, voltage = 0.0,
                      alphas = Float64[]))
    for _ in 1:60
        αn = α * 1.7
        φt, ok = _nlf_alpha_newton(B, d, αn, φ, low, high, f, dρ; inner = inner, refresh = refresh)
        if ok && feasible(αn, φt); α = αn; φ = φt; else; break; end
    end

    # ---- 2) continuation to F* ----
    converged = false; alphas = Float64[α]; steps = 0
    if method === :arclength
        # Pseudo-arclength: parameter = arc-length (≈ cut-mode amplitude ψ=χᵀφ near F*),
        # α FREED, step direction = the curve tangent (auto-aligns with the cut mode χ as
        # α→F*). Sanely defined along the whole path; reaches F* in a handful of steps.
        α, φ, steps, converged = _nlf_arclength!(B, d, α, φ, low, high, f, dρ;
                                                  inner = inner, tol = tol, refresh = refresh,
                                                  max_steps = max_steps, verbose = verbose)
        alphas = Float64[α]
    else  # :alpha — predictor–corrector in the throughput (the crowding chain; baseline)
        α_best = α; φ_best = copy(φ); Δ = 0.5 * α
        for step in 1:max_steps
            steps = step
            α_try = α_best + Δ
            φt, ok = _nlf_alpha_newton(B, d, α_try, φ_best, low, high, f, dρ; inner = inner, refresh = refresh)
            if ok && feasible(α_try, φt)
                α_best = α_try; φ_best = φt; push!(alphas, α_try); Δ *= 1.4
            else
                Δ *= 0.5
            end
            Δ <= tol * max(α_best, 1.0) && (converged = true; break)
        end
        α = α_best; φ = _nlf_zeromean!(φ_best)
    end

    φ = _nlf_zeromean!(φ)
    g = B' * φ; _nlf_rho_drho!(f, dρ, g, low, high)
    residual = norm(B * f .- α .* d)
    info = (steps = steps, residual = residual, converged = converged,
            voltage = φ[mfp.s] - φ[mfp.t], alphas = alphas)
    return α, φ, copy(f), info
end

# ---- pseudo-arclength continuation (parameter = arc-length, α freed) ----
# Walks the solution curve {(φ,α): Bρ(Bᵀφ)=αd} from a feasible α0 to the fold at F*.
# Tangent (φ̇,α̇) solves J φ̇ = α̇ d (the curve direction); near F* it rotates to (χ,0),
# so the step is automatically along the cut mode. The bordered corrector matrix
# [J −d; φ̇ᵀ α̇] stays nonsingular through the fold (κ stays O(1e3) while κ(J)→∞).
function _nlf_arclength!(B, d, α, φ, low, high, f, dρ;
                          inner = :direct, tol = 1e-7, max_steps = 200, verbose = false,
                          refresh::Real = 0.25,
                          setups::Union{Nothing,Base.RefValue{Int}} = nothing)
    n = size(B, 1)
    φ = _nlf_zeromean!(copy(φ))
    # Lazy hierarchy refresh: the LAMG+ hierarchy is shared across the tangent solves and the
    # corrector iterations, rebuilt only when the corrector's residual factor exceeds `refresh`
    # or after a stale-hierarchy stall (rebuild + retry). `refresh = 0` rebuilds every solve.
    hier = nothing; stale = false
    rebuildA! = () -> begin
        hier = _nlf_build_engine(_nlf_jacobian(B, dρ), inner); stale = false
        setups !== nothing && (setups[] += 1); hier
    end
    # tangent at the start: J φ̇ = d, α̇ = 1, normalised, oriented toward increasing α.
    # The tangent solve J⁺d ∝ χ/λ_min is the direction most sensitive to J's drift near the
    # fold, and it runs only once per accepted step — so it always uses a fresh hierarchy
    # (the lazy reuse applies to the corrector iterations, where the cost is).
    tangent = (φc) -> begin
        g = B' * φc; _nlf_rho_drho!(f, dρ, g, low, high)
        local φd
        if _nlf_reusable(inner)
            rebuildA!()
            φd = _nlf_lap_solve(nothing, Vector(d); inner = inner, hier = hier)
        else
            φd = _nlf_lap_solve(_nlf_jacobian(B, dρ), Vector(d); inner = inner, hier = nothing)
        end
        s = sqrt(dot(φd, φd) + 1.0)
        (φd ./ s, 1.0 / s)
    end
    φdot, αdot = tangent(φ)
    α_best = α; φ_best = copy(φ)
    Δs = 0.5 * sqrt(dot(φ, φ) + α^2); Δs = max(Δs, 0.25)
    steps = 0; converged = false
    for step in 1:max_steps
        steps = step
        φ0 = copy(φ); α0 = α
        # corrector, up to two attempts: if the first fails having used a STALE hierarchy, retry
        # once at the SAME Δs with a fresh one (staleness must not shrink the arc-step); only a
        # fresh-hierarchy failure halves Δs. With refresh = 0 every solve is fresh (old behaviour).
        ok = false; usedstale = false
        for attempt in 1:2
            attempt == 2 && ((_nlf_reusable(inner) && usedstale) ? rebuildA!() : break)
            φ = _nlf_zeromean!(φ0 .+ Δs .* φdot); α = α0 + Δs * αdot   # predictor
            res_prev = Inf; usedstale = false
            for _ in 1:30
                g = B' * φ; _nlf_rho_drho!(f, dρ, g, low, high); G = B * f .- α .* d
                N = dot(φdot, φ .- φ0) + αdot * (α - α0) - Δs
                res = sqrt(norm(G)^2 + N^2)
                if res < tol * max(α, 1.0); ok = true; break; end
                J = _nlf_jacobian(B, dρ)     # cheap O(m); needed by the deflated 2×2 even when reusing
                if _nlf_reusable(inner) && (hier === nothing || refresh <= 0 ||
                                            (isfinite(res_prev) && res > refresh * res_prev))
                    rebuildA!()
                end
                usedstale |= (_nlf_reusable(inner) && stale)
                res_prev = res
                δφ, δα = _nlf_arclength_solve(J, Vector(d), φdot, αdot, -G, -N;
                                               inner = inner, hier = hier)
                τ = 1.0; moved = false
                for _ in 1:30
                    φt = _nlf_zeromean!(φ .+ τ .* δφ); αt = α + τ * δα
                    gt = B' * φt; _nlf_rho_drho!(f, dρ, gt, low, high); Gt = B * f .- αt .* d
                    Nt = dot(φdot, φt .- φ0) + αdot * (αt - α0) - Δs
                    if sqrt(norm(Gt)^2 + Nt^2) <= res; φ = φt; α = αt; moved = true; break; end
                    τ *= 0.5
                end
                if !moved && _nlf_reusable(inner) && stale
                    rebuildA!(); continue                # stale stall mid-corrector: rebuild + retry
                end
                moved || break
                _nlf_reusable(inner) && (stale = true)
            end
            ok && break
        end
        if !ok
            φ = φ0; α = α0; Δs *= 0.5                    # fresh-hierarchy failure: shrink arc-step
            Δs < tol * max(α, 1.0) && break
            continue
        end
        (α > α_best) && (α_best = α; φ_best = copy(φ))
        # refresh + reorient the tangent (continuity)
        φdot_n, αdot_n = tangent(φ)
        (dot(φdot_n, φdot) + αdot_n * αdot < 0) && (φdot_n .= .-φdot_n; αdot_n = -αdot_n)
        verbose && @info "arclen" step α αdot Δs
        # at the fold the tangent is horizontal in α (α̇→0) and α stops climbing ⇒ F*
        if abs(α - α0) < tol * max(α, 1.0) || abs(αdot_n) < tol
            converged = true; break
        end
        φdot, αdot = φdot_n, αdot_n
        Δs *= 1.6                                        # grow arc-step on success
    end
    return α_best, _nlf_zeromean!(φ_best), steps, converged
end

# ---- α-driven Newton (zero-mean gauge). Returns (φ, converged). ----
# Lazy hierarchy refresh (`refresh`): the LAMG+ hierarchy is rebuilt only when first needed,
# when the observed per-step residual factor exceeds `refresh` (the frozen operator has
# drifted too far), or when the line search stalls on a stale hierarchy (rebuild + retry).
# `refresh = 0` reproduces the old rebuild-every-step behaviour; `setups` counts rebuilds.
function _nlf_alpha_newton(B, d, α, φ0, low, high, f, dρ; inner = :direct,
                            nmax = 60, tol = 1e-10, refresh::Real = 0.25,
                            setups::Union{Nothing,Base.RefValue{Int}} = nothing)
    n = size(B, 1); φ = _nlf_zeromean!(copy(φ0)); hier = nothing; conv = false
    nr_prev = Inf; stale = false
    rebuild! = () -> begin
        hier = _nlf_build_engine(_nlf_jacobian(B, dρ), inner); stale = false
        setups !== nothing && (setups[] += 1); hier
    end
    for _ in 1:nmax
        g = B' * φ; _nlf_rho_drho!(f, dρ, g, low, high)
        r = B * f .- α .* d
        nr = norm(r)
        if nr < tol * max(α, 1.0)
            conv = true; break
        end
        if _nlf_reusable(inner) && (hier === nothing || refresh <= 0 ||
                                    (isfinite(nr_prev) && nr > refresh * nr_prev))
            rebuild!()
        end
        nr_prev = nr
        J = inner === :direct ? _nlf_jacobian(B, dρ) : nothing
        δ = _nlf_lap_solve(J, -r; inner = inner, hier = hier)   # J δ = −r, δ⊥1
        φ, moved = _nlf_damped!(φ, δ, B, d, α, low, high, f, dρ, nr)
        if !moved && _nlf_reusable(inner) && stale
            rebuild!()                                           # stale stall: rebuild + retry once
            δ = _nlf_lap_solve(nothing, -r; inner = inner, hier = hier)
            φ, moved = _nlf_damped!(φ, δ, B, d, α, low, high, f, dρ, nr)
        end
        moved || break                                          # genuine stall ⇒ α≥F*
        stale = true
    end
    return φ, conv
end

# Damped Newton update; returns (φ, moved?) — moved=false if no τ reduces the residual.
function _nlf_damped!(φ, δ, B, d, α, low, high, f, dρ, nrm0)
    τ = 1.0
    for _ in 1:40
        φt = _nlf_zeromean!(φ .+ τ .* δ)
        gt = B' * φt; _nlf_rho_drho!(f, dρ, gt, low, high)
        norm(B * f .- α .* d) <= nrm0 && return φt, true
        τ *= 0.5
    end
    return φ, false
end

# ---------------------------------------------------------------- inner solves
# Solve the singular graph-Laplacian system J x = b (b ⊥ 1) for x ⊥ 1.
function _nlf_lap_solve(J, b; inner = :direct, hier = nothing)
    n = length(b)        # not size(J,1): J may be `nothing` on the lazy multigrid path
    if inner == :direct
        # (J + ridge·I) is SPD (J ⪰ 0, ridge > 0) ⇒ never singular, even when J's cut has
        # fully formed and the graph disconnects. For b ⊥ 1 this is J⁺b to O(ridge); the
        # tangent direction (∝ χ near F*) is preserved, its magnitude merely capped.
        ridge = 1e-9 * (1 + sum(abs, diag(J)) / n)
        x = cholesky(Symmetric(J + ridge * sparse(I, n, n))) \ Vector(b)  # PD ⇒ robust
        x .-= sum(x) / n
        return x
    else
        return _nlf_engine_solve(hier, J, b, inner)   # :multigrid (LAMG+) or injected engine
    end
end

# Tangent-bordered solve  [J  −d; tφᵀ  tα] [δφ; δα] = [b1; b2],  δφ ⊥ 1.
# (the pseudo-arclength corrector). Nonsingular through the fold.
function _nlf_arclength_solve(J, d, tφ, tα, b1, b2; inner = :direct, hier = nothing)
    n = size(J, 1)
    if inner == :direct
        # Cholesky-Schur (robust through the fold): J+εI is SPD; the gauge drops because
        # b1,d ⊥ 1. δφ = J⁺(b1 + d·δα), and the tangent border fixes δα.
        ridge = 1e-9 * (1 + sum(abs, diag(J)) / n)
        F = cholesky(Symmetric(J + ridge * sparse(I, n, n)))
        yb = F \ Vector(b1); yb .-= sum(yb) / n
        yd = F \ Vector(d);  yd .-= sum(yd) / n
        δα = (b2 - dot(tφ, yb)) / (dot(tφ, yd) + tα)
        δφ = yb .+ δα .* yd; δφ .-= sum(δφ) / n
        return δφ, δα
    else
        # deflated Schur (subtraction of the singularity): δφ = ξ·t + w, w ⊥ {1,t},
        # with the near-null cut amplitude ξ carried as a scalar so J's inner solves
        # stay well-conditioned. S(c) = deflated LAMG solve on {1,t}^⊥.
        t = tφ ./ max(norm(tφ), 1e-30)
        S(c) = begin
            cc = Vector(c); cc .-= sum(cc) / n; cc .-= t .* dot(t, cc)
            y = _nlf_engine_solve(hier, J, cc, inner); y .-= t .* dot(t, y); _nlf_zeromean!(y)
        end
        yc = S(b1); yd = S(Vector(d))
        λ = dot(t, J * t)
        # 2×2 in (ξ, δα):  [λ  −tᵀd; ‖tφ‖  tφᵀyd+tα] [ξ; δα] = [tᵀb1; b2 − tφᵀyc]
        A2 = [λ              -dot(t, d);
              norm(tφ)        dot(tφ, yd) + tα]
        ξ, δα = A2 \ [dot(t, b1); b2 - dot(tφ, yc)]
        δφ = ξ .* t .+ yc .+ δα .* yd
        return δφ, δα
    end
end

# -------- LAMG+ multigrid inner kernel (cut-respecting caliber-2) --------
_nlf_mg_opts() = LAMGOptions(caliber2_1d = true, cal2_τ = 0.5, max_levels = 20)
_nlf_mg_hier(J) = setup(sparse(J); options = _nlf_mg_opts())

# Solve J y = b (b ⊥ 1) with LAMG, return zero-mean y. `hier` is the solver.
function _nlf_mg_solve(hier, J, b)
    h = hier === nothing ? _nlf_mg_hier(J) : hier
    bz = Vector(b); bz .-= sum(bz) / length(bz)
    y, _ = solve(h, bz; options = _nlf_mg_opts())
    return _nlf_zeromean!(y)
end

# ---- Pluggable linear engine (dependency inversion) -------------------------------------------
# NLF's inner Newton solve is engine-agnostic. The package ships two built-in `inner` symbols —
# `:direct` (augmented sparse Cholesky, refactorised each solve) and `:multigrid` (LAMG+, the
# cut-respecting caliber-2 hierarchy, frozen + lazily refreshed) — and lets a caller INJECT any
# other engine (e.g. approxChol) via a `NLFLinearEngine` WITHOUT the LAMG package taking a
# dependency on it. An engine supplies `build(J) -> state` (the reusable frozen factor/hierarchy)
# and `apply(state, J, b) -> x` (x ⊥ 1 solving J x = b; J may be `nothing` on the lazy path).
struct NLFLinearEngine
    build::Function
    apply::Function
    name::String
end

# Every engine except `:direct` keeps a reusable state shared across corrector iterations and
# lazily refreshed (the `refresh` trigger); `:direct` refactorises each solve.
_nlf_reusable(inner) = inner !== :direct

# Build the reusable engine state for the current frozen Jacobian (only called when reusable).
_nlf_build_engine(J, inner::Symbol) = inner === :direct ? nothing : _nlf_mg_hier(J)
_nlf_build_engine(J, eng::NLFLinearEngine) = eng.build(sparse(J))

# Apply a reusable engine `state` to solve J y = b (b ⊥ 1); return zero-mean y.
_nlf_engine_solve(state, J, b, ::Symbol) = _nlf_mg_solve(state, J, b)   # :multigrid (LAMG+)
function _nlf_engine_solve(state, J, b, eng::NLFLinearEngine)
    bz = Vector(b); bz .-= sum(bz) / length(bz)
    return _nlf_zeromean!(eng.apply(state, J, bz))
end

# ---------------------------------------------------------------- grid-FMG
# LAMG aggregation of the graph, with s and t forced to be singleton aggregates (so the
# coarse problem keeps a source and a sink in distinct nodes). Returns (agg, n_coarse).
function _agg_with_st(mfp, rng)
    a = aggregate(mfp.A; rng = rng).aggregate
    n = length(a); a2 = zeros(Int, n); rel = Dict{Int,Int}(); k = 0
    s = mfp.s; t = mfp.t
    @inbounds for i in 1:n
        (i == s || i == t) && continue
        ai = a[i]; haskey(rel, ai) || (k += 1; rel[ai] = k); a2[i] = rel[ai]
    end
    k += 1; a2[s] = k; k += 1; a2[t] = k
    return a2, k
end

# Coarsen the max-flow PROBLEM: aggregate nodes, sum the forward/reverse capacities of the
# fine edges crossing each aggregate boundary into one coarse edge per ordered pair.
function _coarsen_nlf(mfp, agg)
    nc = maximum(agg)
    acc = Dict{Tuple{Int,Int}, Vector{Float64}}()      # (a,b), a<b → [cap a→b, cap b→a]
    @inbounds for e in 1:length(mfp.head)
        ch = agg[mfp.head[e]]; ct = agg[mfp.tail[e]]   # fine forward (f>0) is tail→head = ct→ch
        ch == ct && continue
        a, b = ct < ch ? (ct, ch) : (ch, ct)
        v = get!(acc, (a, b), [0.0, 0.0])
        if ct < ch                                     # fine forward ct→ch is a→b
            v[1] += mfp.high[e]; v[2] += -mfp.low[e]
        else                                           # fine forward is b→a
            v[1] += -mfp.low[e]; v[2] += mfp.high[e]
        end
    end
    edges = Tuple{Int,Int}[]; cplus = Float64[]; cminus = Float64[]
    for ((a, b), v) in acc
        push!(edges, (b, a)); push!(cplus, v[1]); push!(cminus, v[2])  # head=b,tail=a ⇒ f>0: a→b
    end
    make_problem(nc, edges, cplus, cminus, agg[mfp.s], agg[mfp.t]; name = mfp.name * "/c")
end

# Build the FMG hierarchy of coarsened max-flow problems (finest first).
function _nlf_fmg_levels(mfp; n_min = 30, max_levels = 20,
                          rng = Random.MersenneTwister(0xfa11))
    levels = NLFProblem[mfp]; aggs = Vector{Int}[]; cur = mfp
    while size(cur.A, 1) > n_min && length(levels) < max_levels
        agg, nc = _agg_with_st(cur, rng)
        nc >= size(cur.A, 1) && break
        push!(aggs, agg); push!(levels, _coarsen_nlf(cur, agg)); cur = levels[end]
    end
    return levels, aggs
end

# Reduce α from a prolongated seed until the fixed-α Newton solve is feasible (α < F*_level).
function _nlf_back_off!(B, d, φ, α, low, high, f, dρ; inner = :direct, tol = 1e-7)
    αc = α
    for _ in 1:40
        φt, ok = _nlf_alpha_newton(B, d, αc, φ, low, high, f, dρ; inner = inner)
        if ok
            g = B' * φt; _nlf_rho_drho!(f, dρ, g, low, high)
            norm(B * f .- αc .* d) <= max(float(tol), 1e-7) * max(αc, 1.0) && return αc, φt
        end
        αc *= 0.85
    end
    return αc, _nlf_zeromean!(copy(φ))
end

# Grid-FMG driver: solve the coarsest level, then prolongate (φ,α) up, correcting each
# level with a few arclength steps from the (near-F*) seed.
function _nlf_fmg!(mfp; inner = :direct, tol = 1e-7, max_steps = 200, verbose = false)
    levels, aggs = _nlf_fmg_levels(mfp)
    L = length(levels)
    αc, φc, _, infoc = nlf_maxflow(levels[L]; tol = tol, inner = inner, method = :arclength)
    total = infoc.steps
    for l in (L - 1):-1:1
        ml = levels[l]; agg = aggs[l]
        Bl = ml.B; dl = Vector{Float64}(ml.d); lowl = ml.low; highl = ml.high
        mlm = size(Bl, 2); fl = zeros(mlm); dρl = zeros(mlm)
        φseed = _nlf_zeromean!(φc[agg])                         # prolongate coarse → fine
        αf, φf = _nlf_back_off!(Bl, dl, φseed, αc, lowl, highl, fl, dρl;
                                 inner = inner, tol = tol)
        αc, φc, sl, _ = _nlf_arclength!(Bl, dl, αf, φf, lowl, highl, fl, dρl;
                                         inner = inner, tol = tol,
                                         max_steps = max_steps, verbose = verbose)
        total += sl
        verbose && @info "fmg" level=l α=αc steps=sl
    end
    return αc, φc, total, true
end
