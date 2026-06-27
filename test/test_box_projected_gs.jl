"""
test_box_projected_gs.jl — the per-node box-projected Gauss–Seidel smoother
(the voltage-driving foundation). At a single grid level and moderate drive V it
must (a) keep the flow strictly inside the box, (b) propagate the Dirichlet
voltage into the interior, and (c) produce a flux α(V) that rises monotonically
and stays at/below F* (approaching the max-flow from below). Reaching F* itself
needs the multilevel cycle (the downstream-bottleneck limit); that is a separate
development item (doc/maxflow_fas_plan.md).
"""

using Test, NLF, SparseArrays, LinearAlgebra, Random
import NLF: nlf_grid2d, solve_alpha_max, box_projected_gs!

@testset "box-projected GS: feasible, voltage-propagating, α(V) monotone" begin
    res = nlf_grid2d(8; cap_lo = 0.1, cap_hi = 10.0, rng = MersenneTwister(1))
    mfp = res isa Tuple ? res[1] : res
    n = size(mfp.A, 1); αstar, _ = solve_alpha_max(mfp)
    s, t = mfp.s, mfp.t
    pinned = falses(n); pinned[s] = true; pinned[t] = true
    flux(V) = begin
        φ = zeros(n); φ[s] = V; φ[t] = 0.0
        box_projected_gs!(mfp, φ; pinned = pinned, sweeps = 8000)
        g = mfp.B' * φ
        boxviol = maximum(max.(g .- mfp.high, mfp.low .- g, 0.0))
        (abs((mfp.A * φ)[t]), boxviol, maximum(abs.(φ)))
    end
    Vs = [0.5, 1.0, 2.0, 5.0]
    αs = Float64[]
    for V in Vs
        α, bv, mx = flux(V)
        @test bv < 1e-6                      # (a) box feasible
        @test mx ≈ V                         # (b) voltage propagated (max |φ| = V at s)
        @test 0.0 < α <= αstar * (1 + 1e-6)  # (c) below F*
        push!(αs, α)
    end
    @test issorted(αs)                       # α(V) monotone increasing
    @test αs[end] > αs[1]                     # genuinely rising
end
