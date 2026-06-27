using Test
using LinearAlgebra
using Random
using NLF

@testset "max-flow canonical examples + realistic-capacity FAS tests" begin
    # ─────────────────────────────────────────── topology
    @testset "nlf_clrs builds the CLRS Fig. 26.1 network" begin
        mfp, mf = nlf_clrs()
        @test mf == 23.0
        @test size(mfp.A, 1) == 6
        @test length(mfp.head) == 9
        @test mfp.s == 1 && mfp.t == 6
        # Directed variant: c⁻ = 0.
        @test all(mfp.low .== 0.0)
        @test mfp.high == [16.0, 13.0, 4.0, 12.0, 9.0, 14.0, 7.0, 20.0, 4.0]
        # A = BBᵀ.
        @test mfp.A == mfp.B * mfp.B'
    end

    @testset "nlf_clrs undirected variant has c⁻ = c⁺" begin
        mfp_u, _ = nlf_clrs(undirected = true)
        @test mfp_u.low == -mfp_u.high      # negation matches
        @test mfp_u.high == [16.0, 13.0, 4.0, 12.0, 9.0, 14.0, 7.0, 20.0, 4.0]
    end

    @testset "nlf_bottleneck_chain has correct min-cut" begin
        mfp, mf = nlf_bottleneck_chain(6; cap_chain = 10.0,
                                            cap_bottle = 1.0, bottle_pos = 3)
        @test mf == 1.0
        @test mfp.high == [10.0, 10.0, 1.0, 10.0, 10.0]
        @test all(mfp.low .== 0.0)
        @test mfp.s == 1 && mfp.t == 6
    end

    @testset "nlf_image_seg_2x2 builds a 6-node s-t cut graph" begin
        mfp, mf = nlf_image_seg_2x2()
        @test mf == 10.0
        @test size(mfp.A, 1) == 6
        @test mfp.s == 5 && mfp.t == 6
    end

    # ─────────────────────────────────────────── FAS-elim stationarity on CLRS
    @testset "FAS-elim cycle is stationary at interior-feasible φ* on CLRS-undirected" begin
        # Realistic-capacity test: heterogeneous bounds [4..20], not uniform.
        # Pick α well below max flow so the electrical-flow is interior-
        # feasible (verified: at α = 1 the smallest slack is ≈ 3.55).
        # Note: CLRS has no degree-≤1 nodes once s, t are protected — the
        # graph is too dense for elimination. Augment with a leaf for the
        # elimination test.
        mfp_base, _ = nlf_clrs(undirected = true)
        # Add a degree-1 leaf node 7 attached to v₁ (= node 2). Leaf has
        # d_7 = 0, c⁺ = c⁻ = 2 — small but non-trivial capacity.
        edges_aug = vcat([(mfp_base.head[e], mfp_base.tail[e]) for e in eachindex(mfp_base.head)],
                          [(2, 7)])
        cplus_aug  = vcat(mfp_base.high, [2.0])
        cminus_aug = vcat(-mfp_base.low, [2.0])     # undirected → c⁻ = c⁺
        mfp = make_problem(7, edges_aug, cplus_aug, cminus_aug, 1, 6;
                           name = "CLRS-undirected+leaf")
        α = 1.0
        x, _info = solve(mfp.A, α .* mfp.d; options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / 7
        grad = mfp.B' * x
        # Verify interior feasibility on the original 9 CLRS edges. (Leaf
        # edge has gradient = α · d_leaf = 0 for d_7 = 0 — trivially feasible.)
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        # Run FAS-elim cycle (the leaf is eliminable).
        φ = copy(x)
        ratio = fas_2level_cycle_elimination!(mfp, φ; α = α,
                                               ν_pre = 2, ν_post = 2,
                                               ν_coarsest = 30)
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        # Stationarity: cycle leaves the residual at zero (modulo numerics).
        @test r_after < 1e-6 * norm(α .* mfp.d) + 1e-8
    end

    # ─────────────────────────────────────────── FAS-elim stationarity on bottleneck
    @testset "FAS-elim cycle is stationary on bottleneck-chain at α below cap_bottle" begin
        # All-pass chain at α = 0.3: gradient = 0.3 on every edge, well below
        # the bottleneck cap = 1.0 and the surrounding caps = 10. Realistic
        # "bottleneck at the middle, but operating point still below it".
        # Bottleneck chain has degree-1 leaves at both ends — but those are s/t,
        # protected. No elimination possible by default; pass protect=[1] to
        # eliminate node k (the sink leaf) and check stationarity.
        mfp, mf = nlf_bottleneck_chain(8; cap_chain = 10.0,
                                            cap_bottle = 1.0, bottle_pos = 4)
        α = 0.3
        x, _info = solve(mfp.A, α .* mfp.d; options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / 8
        grad = mfp.B' * x
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        φ = copy(x)
        fas_2level_cycle_elimination!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                                       ν_coarsest = 30,
                                       protect = [1])      # release sink t=8
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        @test r_after < 1e-6 * norm(α .* mfp.d) + 1e-8
    end

    # ─────────────────────────────────────────── FAS-agg stationarity on CLRS
    @testset "FAS-agg cycle is stationary at φ* on CLRS-undirected" begin
        mfp, _ = nlf_clrs(undirected = true)
        α = 1.0
        x, _info = solve(mfp.A, α .* mfp.d; options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / 6
        grad = mfp.B' * x
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        φ = copy(x)
        # build_coarse_problem may produce a tiny aggregate; that's fine.
        mfp_c, P, T, agg = build_coarse_problem(mfp;
                                                 rng = Random.MersenneTwister(0x51))
        fas_2level_cycle!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                          ν_coarsest = 20,
                          mfp_c = mfp_c, P = P, T = T, agg = agg)
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        @test r_after < 1e-6 * norm(α .* mfp.d) + 1e-8
    end
end
