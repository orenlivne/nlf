# Generic source-form nonlinear Laplacian flow solver (NLF core).
#
# Solves the nonlinear graph-Laplacian equilibrium
#
#         B ρ(Bᵀ x) = b ,                                     (∗)
#
# for node potentials `x`, where `B` is the n×m node–edge incidence and the monotone edge
# law is supplied as a callback  `law!(f, dρ, g)`  that fills, for edge differences g = Bᵀx,
#
#         f  .= ρ(g)        (edge flows)
#         dρ .= ρ'(g)       (edge conductances, ρ'(g) ≥ 0).
#
# The solver is a damped, inexact **chord-Newton** iteration on the frozen linearization
# J = B diag(ρ'(g)) Bᵀ (a weighted graph Laplacian), whose inner linear solves are delegated
# to the near-linear LAMG+ engine with a lazily refreshed hierarchy — exactly the machinery
# validated for congestion/max-flow in the NLF paper, here exposed generically so that ANY
# monotone edge law (BPR, p-Laplacian, Huber, …) drops in without touching the solver.
#
# `b` must be range-compatible (Σ b = 0); the constant null space of (∗) is fixed by the
# zero-mean gauge. Reaction/mass terms are handled by the caller via a grounded node with
# linear edges (see the GraphSSL layer), so the core stays a pure source-form flow.
#
# This GENERALIZES `newton_fixed!`/`nlf_bpr` from the paper scripts: the BPR congestion solve
# is the special case law! = BPR, b = α d.

using LinearAlgebra, SparseArrays
import LAMG: LAMGOptions, setup, solve

_zeromean!(x) = (x .-= sum(x) / length(x); x)

# Symmetrise + enforce exact zero row sums so LAMG+ setup sees a clean graph Laplacian
# (B diag(w) Bᵀ drifts by ~1e-16). Rebuild the diagonal from the symmetrised off-diagonals.
function _laplacian_clean(J)
    off = J - spdiagm(0 => diag(J)); off = (off + off') / 2; dropzeros!(off)
    off + spdiagm(0 => -vec(sum(off, dims = 2)))
end

"""
    FlowSolve

Result of a nonlinear flow solve: node potentials `x`, chord-Newton `steps`, final
`residual` = ‖B ρ(Bᵀx) − b‖, number of LAMG hierarchy `setups`, and `converged`.
"""
struct FlowSolve
    x::Vector{Float64}
    steps::Int
    residual::Float64
    setups::Int
    converged::Bool
end

"""
    newton_flow!(x, B, law!, b; inner=:multigrid, ...) -> FlowSolve

Damped inexact chord-Newton for `B ρ(Bᵀx) = b`, warm-started from `x` (mutated in place).
`inner=:multigrid` uses the LAMG+ near-linear engine (lazy hierarchy refresh); `inner=:direct`
uses a pinned sparse Cholesky (small graphs / ground truth). The hierarchy refs
(`H,SC,setups,GG`) may be shared across a continuation chain to freeze the setup across solves.
"""
function newton_flow!(x, B, law!, b; inner = :multigrid, nmax = 80, tol = 1e-8,
                      refresh = 0.25, eta = 0.05, tlim = Inf,
                      build_solver = nothing,
                      H = Ref{Any}(nothing), SC = Ref(1.0), ST = Ref(false),
                      setups = Ref(0), GG = Ref(1.0))
    n = size(B, 1); m = size(B, 2)
    f = zeros(m); dρ = zeros(m); steps = 0; nr_prev = Inf; t0 = time()
    keep = 2:n                                   # pin node 1 for the :direct SPD reduced solve
    bn = max(norm(b), 1.0)
    # `build_solver`, when supplied, injects a swappable inner engine: it maps the scaled clean
    # graph Laplacian L to an apply-closure `rhs -> x` solving Lx=rhs (e.g. approximate Cholesky).
    # When it is `nothing`, the default LAMG+ hierarchy is used. This keeps the inner solve
    # engine-agnostic, exactly as in the NLF framework.
    use_ext = build_solver !== nothing
    rebuild! = (dρf) -> begin
        SC[] = maximum(dρf); Lc = _laplacian_clean((B * Diagonal(dρf) * B') ./ SC[])
        H[] = use_ext ? build_solver(Lc) : setup(Lc; options = LAMGOptions())
        ST[] = false; setups[] += 1
    end
    mg_step = (r) -> begin
        rhs = _zeromean!(-Vector(r) ./ SC[])
        if use_ext
            δ = H[](rhs)
        else
            δ, info = solve(H[], rhs; options = LAMGOptions(tol = eta, γ_coarse_growth = GG[]))
            get(info, :gamma_escalated, false) && (GG[] = 1.15)
        end
        _zeromean!(δ)
    end
    for it in 1:nmax
        steps = it
        g = B' * x; law!(f, dρ, g); r = B * f .- b; nr = norm(r)
        nr < tol * bn && return FlowSolve(_zeromean!(x), steps, nr, setups[], true)
        (time() - t0 > tlim) && return FlowSolve(_zeromean!(x), steps, nr, setups[], false)
        dρf = max.(dρ, 1e-12 * maximum(dρ))       # Jacobian floor: keeps J SPD w/o distorting Newton
        local δ
        if inner == :direct
            J = B * Diagonal(dρf) * B'
            δ = zeros(n); δ[keep] = J[keep, keep] \ (-Vector(r)[keep]); _zeromean!(δ)
        else
            (H[] === nothing || nr > refresh * nr_prev) && rebuild!(dρf)
            δ = mg_step(r)
        end
        nr_prev = nr
        τ = 1.0; ok = false
        for _ in 1:60
            xt = _zeromean!(x .+ τ .* δ); gt = B' * xt
            ft = similar(f); dt = similar(dρ); law!(ft, dt, gt)
            if norm(B * ft .- b) <= nr; copyto!(x, xt); ok = true; break; end
            τ *= 0.5
        end
        if !ok && inner != :direct && ST[]          # stale stall: rebuild + retry once
            rebuild!(dρf); δ = mg_step(r)
            for _ in 1:60
                xt = _zeromean!(x .+ τ .* δ); gt = B' * xt
                ft = similar(f); dt = similar(dρ); law!(ft, dt, gt)
                if norm(B * ft .- b) <= nr; copyto!(x, xt); ok = true; break; end
                τ *= 0.5
            end
        end
        ok || break
        ST[] = true
    end
    g = B' * x; law!(f, dρ, g)
    FlowSolve(_zeromean!(x), steps, norm(B * f .- b), setups[], false)
end

"""
    flow_continuation!(x, B, laws, b; inner=:multigrid, ...) -> Vector{FlowSolve}

Numerical continuation through a sequence of edge laws `laws = [law!₁, law!₂, …]`, warm-starting
each solve from the previous solution and SHARING one lazily refreshed LAMG+ hierarchy across the
whole chain. The canonical use is continuation in the exponent `p` from the p=2 linear solve
(`laws[1]`) down toward the target `p` (`laws[end]`), so each nonlinear solve starts inside the
basin of the previous, cheaper one. Returns the per-stage `FlowSolve` list.
"""
function flow_continuation!(x, B, laws, b; inner = :multigrid, tol = 1e-8, nmax = 80,
                            refresh = 0.25, tlim = Inf, build_solver = nothing, setups = Ref(0),
                            H = Ref{Any}(nothing), SC = Ref(1.0), ST = Ref(false), GG = Ref(1.0))
    out = FlowSolve[]
    for law! in laws
        res = newton_flow!(x, B, law!, b; inner = inner, nmax = nmax, tol = tol,
                           refresh = refresh, tlim = tlim, build_solver = build_solver,
                           H = H, SC = SC, ST = ST, setups = setups, GG = GG)
        push!(out, res)
    end
    out
end
