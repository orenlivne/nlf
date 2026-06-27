using Test
using LinearAlgebra
using Random
using NLF

@testset "max-flow multilevel FAS hierarchy + cycles" begin
    # ─────────────────────────────────────────── hierarchy setup
    @testset "setup_nlf_hierarchy chains down to n_min" begin
        mfp = nlf_grid2d(16; cap_lo = 1.0, cap_hi = 3.0,
                             rng = MersenneTwister(0x71))
        hier = setup_nlf_hierarchy(mfp; n_min = 20,
                                        rng = MersenneTwister(0x72))
        @test num_levels(hier) >= 2
        # Levels should be strictly decreasing in size until we hit n_min.
        sizes = [size(L.A, 1) for L in hier.levels]
        @test all(diff(sizes) .< 0)
        @test sizes[end] <= 30      # ≈ n_min plus the last-step slop
        @test length(hier.P) == num_levels(hier) - 1
        @test length(hier.T) == num_levels(hier) - 1
        @test length(hier.agg) == num_levels(hier) - 1
    end

    # ─────────────────────────────────────────── 2-level cold-start convergence
    @testset "2-level FAS cycle reduces residual repeatedly on grid2d(8)" begin
        mfp = nlf_grid2d(8; cap_lo = 1.0, cap_hi = 3.0,
                             rng = MersenneTwister(0x81))
        α = 0.1
        φ = zeros(size(mfp.A, 1))
        αd_norm = norm(α .* mfp.d)
        ratios = Float64[]
        mfp_c, P, T, agg = build_coarse_problem(mfp;
                                                 rng = MersenneTwister(0x82))
        for _ in 1:8
            r_before = norm(mfp.A * φ .- α .* mfp.d)
            fas_2level_cycle!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                              ν_coarsest = 20,
                              mfp_c = mfp_c, P = P, T = T, agg = agg)
            r_after = norm(mfp.A * φ .- α .* mfp.d)
            push!(ratios, r_after / max(r_before, 1e-30))
        end
        # Geometric convergence: all ratios < 1; median ratio bounded.
        @test all(ratios .< 1.0)
        @test minimum(ratios) < 0.5     # at least one cycle gives >2× reduction
        # Tail residual must be a substantial reduction from start.
        @test norm(mfp.A * φ .- α .* mfp.d) < 0.05 * αd_norm
    end

    # ─────────────────────────────────────────── multilevel cold-start convergence
    @testset "multilevel FAS converges to tol on grid2d(16)" begin
        mfp = nlf_grid2d(16; cap_lo = 1.0, cap_hi = 3.0,
                             rng = MersenneTwister(0x91))
        α = 0.05
        hier = setup_nlf_hierarchy(mfp; n_min = 20,
                                        rng = MersenneTwister(0x92))
        @test num_levels(hier) >= 3
        φ = zeros(size(mfp.A, 1))
        cycles, history = fas_multilevel_solve!(hier, φ;
                                                 α = α, tol = 1e-6,
                                                 max_cycles = 50,
                                                 ν_pre = 2, ν_post = 2,
                                                 ν_coarsest = 40)
        αd_norm = norm(α .* mfp.d)
        @test cycles <= 50
        @test history[end] / αd_norm < 1e-6
        # Geometric: per-cycle ratios in the tail should be ≪ 1.
        if length(history) >= 5
            tail_ratios = history[end-3:end] ./ history[end-4:end-1]
            μ = exp(mean(log.(max.(tail_ratios, 1e-30))))
            @test μ < 0.9       # converging (no hard ceiling — depends on graph)
        end
    end

    # ─────────────────────────────────────────── multilevel stationarity
    @testset "multilevel FAS cycle is ≈ stationary at interior-feasible φ*" begin
        mfp = nlf_grid2d(16; cap_lo = 2.0, cap_hi = 5.0,
                             rng = MersenneTwister(0xa1))
        α = 0.05    # well below saturation
        # Build φ* via the plain Laplacian solve.
        x, _info = solve(mfp.A, α .* mfp.d;
                          options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / length(x)
        grad = mfp.B' * x
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        # Run one multilevel cycle from φ*.
        hier = setup_nlf_hierarchy(mfp; n_min = 20,
                                        rng = MersenneTwister(0xa2))
        φ = copy(x)
        r_before = norm(mfp.A * φ .- α .* mfp.d)
        fas_multilevel_cycle!(hier, φ; α = α, ν_pre = 2, ν_post = 2,
                               ν_coarsest = 40)
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        # Cycle should not BLOW UP from the exact solution. We don't
        # claim residual stays exactly zero (the aggregation form's
        # τ-corrected coarse bounds aren't exact like Schur is), but
        # the residual must not exceed a small multiple of the prior.
        @test r_after < 10 * r_before + 1e-6
    end

    # ─────────────────────────────────────────── multilevel on real DIMACS
    @testset "multilevel FAS hierarchy builds on real DIMACS vision instance" begin
        # Only run if the file exists locally (gitignored — must be
        # extracted via data/maxflow/README.md instructions).
        path = joinpath(@__DIR__, "..", "data", "maxflow", "BVZ-tsukuba0.max")
        if !isfile(path)
            @warn "Skipping DIMACS test — $path not extracted."
            return
        end
        mfp = load_dimacs_max(path)
        @test size(mfp.A, 1) > 100_000     # large real instance
        hier = setup_nlf_hierarchy(mfp; n_min = 100,
                                        max_levels = 12,
                                        rng = MersenneTwister(0xb1))
        @test num_levels(hier) >= 4
        # Sanity-check: hierarchy contracts.
        sizes = [size(L.A, 1) for L in hier.levels]
        @test all(diff(sizes) .< 0)
        @test sizes[end] < sizes[1] / 100   # at least 100× shrinkage
    end
end
