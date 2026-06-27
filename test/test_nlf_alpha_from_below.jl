"""
test_nlf_alpha_from_below.jl — the α-continuation-from-below max-flow solver
must recover the maximum flow value F* (verified against a direct LP) on a range
of graph types, and must NOT overshoot F* (the bug the coarsest-level
α-optimization had). The box-violation feasibility check is essential — without
it the bottleneck-chain case overshoots ~10x.
"""

using Test, NLF, SparseArrays, LinearAlgebra, Random
import NLF: nlf_grid2d, nlf_grid3d, nlf_random, nlf_bottleneck_chain,
             solve_alpha_max, nlf_alpha_from_below

@testset "max-flow α-from-below recovers F* (no overshoot)" begin
    cases = Any[
        ("grid2d/10", nlf_grid2d(10; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(2))),
        ("grid2d/14", nlf_grid2d(14; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(5))),
        ("grid3d/5",  nlf_grid3d(5;  cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(3))),
        ("rand/200",  nlf_random(200, 0.04; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(200))),
        ("bottleneck/8",  nlf_bottleneck_chain(8)),
        ("bottleneck/12", nlf_bottleneck_chain(12)),
    ]
    for (name, res) in cases
        mfp = res isa Tuple ? res[1] : res
        α_lp, _ = solve_alpha_max(mfp)
        α, φ = nlf_alpha_from_below(mfp)
        @testset "$name" begin
            # matches the LP max-flow value
            @test isapprox(α, α_lp; rtol = 0.03)
            # never exceeds F* (feasibility / no overshoot)
            @test α <= α_lp * (1 + 0.02)
            # the returned φ is box-feasible
            g = mfp.B' * φ
            @test maximum(max.(g .- mfp.high, mfp.low .- g, 0.0)) < 1e-3
        end
    end
end
