"""
test_rho_nlf_equivalence.jl — the saturating-conductance (ρ) formulation is
EQUIVALENT to combinatorial max-flow.

ρ-formulation: nonlinear resistor network, edge law f_e = ρ_e(g_e) with
g_e = φ_head − φ_tail, ρ_e monotone, slope 1 at 0, asymptotes ℓ_e (g→−∞),
u_e (g→+∞). Drive s low, t at 0; solve conservation (B ρ(Bᵀφ))_i = 0 at every
interior node (damped Newton); throughput = |current into t|. As V→∞ every
min-cut edge saturates, so the throughput → the combinatorial max-flow value.

We assert α_ρ(V→∞) == Ford–Fulkerson max-flow on small grid AND non-grid
instances (solved by a DIRECT Newton method — the equivalence is a property of
the FORMULATION, independent of the eventual FAS solver).
"""

using Test, NLF, LinearAlgebra, SparseArrays, Random
import NLF: NLFProblem, make_problem, nlf_grid2d, nlf_random

function _ff_maxflow(mfp::NLFProblem)
    n = size(mfp.A, 1); m = length(mfp.head); s = mfp.s; t = mfp.t
    adj = [Vector{Tuple{Int,Int,Int}}() for _ in 1:n]
    for e in 1:m
        push!(adj[mfp.tail[e]], (mfp.head[e], e, +1))
        push!(adj[mfp.head[e]], (mfp.tail[e], e, -1))
    end
    res = Dict{Tuple{Int,Int},Float64}()
    for e in 1:m
        res[(mfp.tail[e], mfp.head[e])] = mfp.high[e]
        res[(mfp.head[e], mfp.tail[e])] = -mfp.low[e]
    end
    tot = 0.0
    while true
        parent = fill(0, n); vis = falses(n); vis[s] = true; q = [s]; found = false
        while !isempty(q) && !found
            u = popfirst!(q)
            for (v, _e, _sg) in adj[u]
                if !vis[v] && res[(u, v)] > 1e-12
                    vis[v] = true; parent[v] = u
                    v == t && (found = true; break); push!(q, v)
                end
            end
        end
        !found && break
        b = Inf; nd = t; while nd != s; b = min(b, res[(parent[nd], nd)]); nd = parent[nd]; end
        nd = t; while nd != s; p = parent[nd]; res[(p, nd)] -= b; res[(nd, p)] += b; nd = p; end
        tot += b
    end
    return tot
end

_rho(g, lo, hi) = g >= 0 ? hi * tanh(g / hi) : (-lo) * tanh(g / (-lo))
_drho(g, lo, hi) = (a = g >= 0 ? hi : (-lo); s = sech(g / a); s * s)

function _rho_throughput(mfp; V = 1e5, iters = 400, tol = 1e-11)
    n = size(mfp.A, 1); m = length(mfp.head); s = mfp.s; t = mfp.t
    lo = mfp.low; hi = mfp.high; B = mfp.B
    interior = setdiff(1:n, [s, t])
    φ = zeros(n); φ[s] = -V               # drive s LOW → forward caps hi_e
    fvec(p) = (g = B' * p; [_rho(g[e], lo[e], hi[e]) for e in 1:m])
    for _ in 1:iters
        g = B' * φ; f = [_rho(g[e], lo[e], hi[e]) for e in 1:m]
        rI = (B * f)[interior]; nr = norm(rI); nr < tol && break
        c = [_drho(g[e], lo[e], hi[e]) for e in 1:m]
        J = B * spdiagm(0 => c) * B'
        Δ = (J[interior, interior] + 1e-13 * I) \ (-rI)
        step = 1.0; φn = copy(φ)
        for _ in 1:40
            φn[interior] = φ[interior] .+ step .* Δ
            norm((B * fvec(φn))[interior]) < nr && break; step *= 0.5
        end
        φ[interior] = φn[interior]
    end
    f = fvec(φ)
    return abs((B * f)[t]), norm((B * f)[interior]), maximum(max.(lo .- f, f .- hi))
end

@testset "ρ-formulation == combinatorial max-flow" begin
    cases = [
        ("grid2d/3²",  nlf_grid2d(3;  rng = MersenneTwister(1))),
        ("grid2d/5²",  nlf_grid2d(5;  rng = MersenneTwister(2))),
        ("random/12",  nlf_random(12, 0.4; rng = MersenneTwister(4))),
        ("random/20",  nlf_random(20, 0.3; rng = MersenneTwister(5))),
    ]
    for (name, mfp) in cases
        F = _ff_maxflow(mfp)
        α, res, viol = _rho_throughput(mfp)
        @test res < 1e-6            # conservation solved
        @test viol < 1e-7           # flow within capacities (feasible)
        @test isapprox(α, F; rtol = 1e-3)   # throughput == max-flow value
    end
end
