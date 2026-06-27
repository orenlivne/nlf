using Test, NLF, LinearAlgebra, SparseArrays, Random, Statistics

@testset "linear setup(A) == new setup(mfp_loose) on A side" begin
    for K in (16, 32, 64)
        mfp = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0,
                                   rng = MersenneTwister(0xfa11))
        # Loosen caps to make the problem effectively linear.
        mfp_loose = NLFProblem(mfp.A, mfp.B,
                                    fill(-1e8, length(mfp.head)),
                                    fill(+1e8, length(mfp.head)),
                                    mfp.d, mfp.s, mfp.t, mfp.name,
                                    mfp.head, mfp.tail)
        opts = LAMGOptions(elim_max_degree = 4, do_recomb = true,
                           γ = 1.5, rhs_correction = 1.0,
                           ν_pre = 1, ν_post = 2, history_size = 4,
                           seed = 0x3039)  # PIN THE SEED — both setups share it
        h_lin = setup(mfp_loose.A; options = opts)
        h_mf  = setup(mfp_loose;   options = opts)
        @test length(h_lin) == length(h_mf)
        for l in 1:length(h_lin)
            @test h_lin[l].level_type == h_mf[l].level_type
            @test size(h_lin[l].a) == size(h_mf[l].a)
            @test nnz(h_lin[l].a) == nnz(h_mf[l].a)
            @test maximum(abs.(h_lin[l].a - h_mf[l].a)) < 1e-12
            if h_lin[l].p !== nothing
                @test h_lin[l].p == h_mf[l].p
            end
            if !isempty(h_lin[l].elim_stages)
                @test length(h_lin[l].elim_stages) == length(h_mf[l].elim_stages)
            end
        end
        # The max-flow path additionally populates a box-aware relaxer.
        @test h_mf[1].relaxer isa NLF.MaxFlowGSKaczmarzRelaxer
        @test length(h_mf[1].relaxer.low) == length(mfp_loose.low)
    end
end

@testset "max-flow setup also populates box at coarse levels" begin
    mfp = NLF.nlf_grid2d(32; cap_lo = 1.0, cap_hi = 1.0,
                                rng = MersenneTwister(0xfa11))
    opts = LAMGOptions(elim_max_degree = 4, seed = 0x3039)
    h = setup(mfp; options = opts)
    for l in 1:length(h)
        rx = h[l].relaxer
        @test rx isa NLF.MaxFlowGSKaczmarzRelaxer
        @test length(rx.low) == length(rx.mfp.head)
        @test length(rx.high) == length(rx.mfp.head)
    end
end
