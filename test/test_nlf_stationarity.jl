using Test, NLF, LinearAlgebra, SparseArrays, Random, Statistics

# Stationarity of the unified max-flow FAS cycle at the exact linear-LAMG
# solution, with capacity bounds loose enough to be inactive everywhere.
# In that regime the constrained problem reduces to A φ = α·b, and a
# single cycle started from φ* must leave the residual at roundoff —
# proof that the FAS cycle's box τ-correction does not inject error when
# the constraints are slack.

@testset "max-flow FAS stationarity at exact solution (loose constraints)" begin
    for K in (16, 32, 64)
        mfp = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0,
                                   rng = MersenneTwister(0xfa11))
        n = size(mfp.A, 1)
        rng = MersenneTwister(0xface)
        x_true = randn(rng, n); x_true .-= mean(x_true)
        b = mfp.A * x_true   # compatible RHS
        # Loosen constraints to make max-flow behave linearly.
        mfp_b = NLFProblem(mfp.A, mfp.B,
                                fill(-1e8, length(mfp.head)),
                                fill(+1e8, length(mfp.head)),
                                b, mfp.s, mfp.t, mfp.name,
                                mfp.head, mfp.tail)
        # AGG-only hierarchy first (no ELIM-coarse box stub interference).
        for emd in (0, 4)
            opts = LAMGOptions(tol = 1e-10, max_cycles = 1,
                               elim_max_degree = emd,
                               ν_pre = 1, ν_post = 2,
                               γ = 1.5, do_recomb = true, history_size = 4,
                               rhs_correction = 1.0, seed = 12345)
            h = setup(mfp_b; options = opts)
            x_out, info = solve(h, mfp_b; α = 1.0, options = opts,
                                          x0 = copy(x_true))
            rel = info.residual_history[end] / max(norm(b), 1e-30)
            @test rel < 1e-10
        end
    end
end
