using Test
using LinearAlgebra
using SparseArrays
using Random
using NLF

@testset "max-flow constrained formulation" begin
    # ─────────────────────────────────────────── data structures
    @testset "make_problem builds A = BBᵀ + correct incidence" begin
        # 3-node path: edges 1→2 and 2→3 (head = head, tail = tail).
        edges = [(2, 1), (3, 2)]
        c⁺ = [1.0, 2.0]; c⁻ = [0.5, 1.0]
        mfp = make_problem(3, edges, c⁺, c⁻, 1, 3; name = "tiny")
        # Incidence sanity.
        @test mfp.B[2, 1] == +1.0   # node 2 is head of edge 1 (2,1)
        @test mfp.B[1, 1] == -1.0   # node 1 is tail of edge 1
        @test mfp.B[3, 2] == +1.0
        @test mfp.B[2, 2] == -1.0
        # A = BBᵀ for unit weights.
        @test mfp.A == mfp.B * mfp.B'
        # Bounds.
        @test mfp.high == c⁺
        @test mfp.low == -c⁻
        @test mfp.d == [-1.0, 0.0, 1.0]
    end

    # ─────────────────────────────────────────── GS + Kaczmarz
    @testset "relax_gs_kaczmarz! reduces residual on Aφ = 0" begin
        mfp = nlf_grid2d(8; cap_lo = 0.5, cap_hi = 2.0,
                             rng = MersenneTwister(0x1234))
        rng = MersenneTwister(0xa)
        n = size(mfp.A, 1)
        φ = randn(rng, n); φ .-= sum(φ) / n
        r0 = norm(mfp.A * φ)
        relax_gs_kaczmarz!(mfp, φ, mfp.d, 0.0; sweeps = 10)
        φ .-= sum(φ) / n
        @test norm(mfp.A * φ) < r0
    end

    @testset "Kaczmarz enforces capacity bounds when active" begin
        # Tight bounds: any random φ exceeds them. The smoother should
        # drive the gradient back into [low, high].
        edges = [(2, 1), (3, 2), (4, 3)]
        c⁺ = [0.1, 0.1, 0.1]; c⁻ = [0.1, 0.1, 0.1]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 4; name = "tight")
        rng = MersenneTwister(0xb)
        φ = randn(rng, 4); φ .-= sum(φ) / 4
        relax_gs_kaczmarz!(mfp, φ, mfp.d, 0.0; sweeps = 50)
        # All edge gradients within [-0.1, 0.1].
        grad = mfp.B' * φ
        @test all(-0.1 - 1e-6 .<= grad .<= 0.1 + 1e-6)
    end

    # ─────────────────────────────────────────── shrinkage measurements
    @testset "shrinkage_gs and shrinkage_gs_kaczmarz return values in [0, 1]" begin
        mfp = nlf_grid2d(8; rng = MersenneTwister(0x12))
        μ_gs = shrinkage_gs(mfp.A; sweeps_max = 10, num_examples = 2,
                            rng = MersenneTwister(0x12))
        μ_op, μ_full = shrinkage_gs_kaczmarz(mfp; sweeps_max = 10, num_examples = 2,
                                              rng = MersenneTwister(0x12))
        @test 0 <= μ_gs <= 1
        @test 0 <= μ_op <= 1
        @test 0 <= μ_full <= 1
        # Kaczmarz with α = 0 + loose caps should give μ_op ≈ μ_gs.
        @test abs(μ_op - μ_gs) < 0.05
    end

    # ─────────────────────────────────────────── FAS τ-correction
    @testset "coarsen_constraints: low_c ≤ high_c after τ-correction" begin
        mfp = nlf_grid2d(8; rng = MersenneTwister(0x21))
        mfp_c, P, T, agg = build_coarse_problem(mfp; rng = MersenneTwister(0x21))
        n_f = size(mfp.A, 1)
        φ = randn(MersenneTwister(0x22), n_f); φ .-= sum(φ) / n_f
        lo_c, hi_c = coarsen_constraints(mfp.low, mfp.high, φ, mfp.B, P, T,
                                         mfp_c.B, mfp.head, mfp.tail, agg;
                                         coarse_head = mfp_c.head,
                                         coarse_tail = mfp_c.tail)
        @test all(lo_c .<= hi_c)
    end

    # ─────────────────────────────────────────── FAS 2-level stationary test
    @testset "FAS cycle reduces operator residual on Aφ = α d" begin
        mfp = nlf_grid2d(8; cap_lo = 1.0, cap_hi = 4.0,
                             rng = MersenneTwister(0x31))
        rng = MersenneTwister(0x32)
        n = size(mfp.A, 1)
        # Start from φ₀ = 0, α = 0 — already stationary, residual zero.
        # Test instead: solve Aφ = α d for α small (interior-feasible).
        α = 0.1
        φ = zeros(n)
        r0 = norm(mfp.A * φ - α .* mfp.d)
        # Pre-build coarse so the cycle is deterministic.
        mfp_c, P, T, agg = build_coarse_problem(mfp; rng = MersenneTwister(0x31))
        # One cycle should reduce the residual.
        ratio = fas_2level_cycle!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                                  ν_coarsest = 20,
                                  mfp_c = mfp_c, P = P, T = T, agg = agg)
        @test ratio < 1.0
    end

    @testset "FAS cycle: if φ already exactly solves Aφ = α d (interior-feasible), cycle is ≈ stationary" begin
        # Construct φ* by solving Aφ = α d via LAMG (no constraints).
        # If φ* is interior-feasible (no active capacity constraints), the
        # FAS cycle with τ-corrected constraints should keep φ ≈ φ*.
        mfp = nlf_grid2d(8; cap_lo = 2.0, cap_hi = 5.0,
                             rng = MersenneTwister(0x41))
        n = size(mfp.A, 1)
        α = 0.05    # small so flow is far below capacity
        # Plain LAMG solve of A φ = α d (works since A is a graph Laplacian).
        x, _info = solve(mfp.A, α .* mfp.d; options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / n
        # Verify the constructed φ is interior-feasible.
        grad = mfp.B' * x
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        # Now run one FAS cycle from this point.
        φ = copy(x)
        mfp_c, P, T, agg = build_coarse_problem(mfp; rng = MersenneTwister(0x41))
        r_before = norm(mfp.A * φ .- α .* mfp.d)
        fas_2level_cycle!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                          ν_coarsest = 20,
                          mfp_c = mfp_c, P = P, T = T, agg = agg)
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        # Both should be tiny (started from near-exact and the cycle is
        # stationary on interior-feasible points).
        @test r_after < 1e-6 * norm(α .* mfp.d) + 1e-8
    end
end
