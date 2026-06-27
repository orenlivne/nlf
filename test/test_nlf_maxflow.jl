# Tests for nlf_maxflow: true max-flow via smooth-ρ α-continuation + cut-mode bordering.
using Test, NLF, SparseArrays, LinearAlgebra, Random
import NLF: nlf_maxflow, make_problem, nlf_bottleneck_chain, nlf_grid2d,
            nlf_grid3d, nlf_random, solve_alpha_max

# some generators return mfp, others (mfp, F_known); unwrap uniformly
_mfp(x) = x isa Tuple ? first(x) : x

# conservation + box-feasibility of a returned flow
function _conserved_feasible(mfp, F, f; rtol = 1e-5)
    cons = norm(mfp.B * f .- F .* mfp.d) / max(abs(F), 1)
    box  = all(mfp.low .- 1e-6 .<= f .<= mfp.high .+ 1e-6)
    cons < rtol, box
end

@testset "nlf_maxflow" begin

    @testset "bottleneck chain (known F* = cap_bottle)" begin
        for cb in (1.0, 2.5, 7.0)
            mfp, F_known = nlf_bottleneck_chain(12; cap_chain = 20.0, cap_bottle = cb)
            F, φ, f, info = nlf_maxflow(mfp; tol = 1e-9)
            @test isapprox(F, F_known; rtol = 1e-4)
            cons, box = _conserved_feasible(mfp, F, f)
            @test cons
            @test box
        end
    end

    @testset "two parallel paths (hand-computed F* = 3)" begin
        # s=1, a=2, b=3, t=4.  s→a(2) a→t(3) | s→b(4) b→t(1).  F* = min(2,3)+min(4,1)=3.
        edges   = [(2, 1), (4, 2), (3, 1), (4, 3)]
        c_plus  = [2.0, 3.0, 4.0, 1.0]
        c_minus = zeros(4)
        mfp = make_problem(4, edges, c_plus, c_minus, 1, 4; name = "parallel")
        F, φ, f, info = nlf_maxflow(mfp; tol = 1e-9)
        @test isapprox(F, 3.0; rtol = 1e-3)
        cons, box = _conserved_feasible(mfp, F, f)
        @test cons
        @test box
    end

    @testset "single edge (F* = capacity)" begin
        mfp = make_problem(2, [(2, 1)], [4.2], [0.0], 1, 2; name = "edge")
        F, _, f, _ = nlf_maxflow(mfp; tol = 1e-10)
        @test isapprox(F, 4.2; rtol = 1e-4)
    end

    @testset "computes TRUE max-flow, not the gradient/electrical relaxation" begin
        # On a heterogeneous grid the gradient LP (solve_alpha_max) is a severe relaxation
        # (≈0.17×F*). nlf_maxflow must far exceed it and land on the true max-flow.
        mfp = _mfp(nlf_grid2d(8; cap_lo = 0.2, cap_hi = 5.0, rng = MersenneTwister(2)))
        F, φ, f, info = nlf_maxflow(mfp; tol = 1e-9)
        F_grad, _ = solve_alpha_max(mfp)
        @test F > 1.5 * F_grad                  # not the relaxation
        @test isapprox(F, 5.8498; rtol = 1e-3)  # the true max-flow (LP + push-relabel)
        cons, box = _conserved_feasible(mfp, F, f)
        @test cons
        @test box
        @test info.converged
    end

    @testset "conservation + feasibility on assorted instances" begin
        for mfp in (_mfp(nlf_grid2d(10; cap_lo = 0.5, cap_hi = 3.0, rng = MersenneTwister(21))),
                    _mfp(nlf_grid3d(4;  cap_lo = 0.2, cap_hi = 5.0, rng = MersenneTwister(22))),
                    _mfp(nlf_random(120, 0.06; cap_lo = 0.3, cap_hi = 4.0, rng = MersenneTwister(23))))
            F, φ, f, info = nlf_maxflow(mfp; tol = 1e-9)
            @test F > 0
            cons, box = _conserved_feasible(mfp, F, f)
            @test cons
            @test box
        end
    end

    @testset "multigrid inner solve reaches true max-flow" begin
        # inner=:multigrid uses cut-respecting LAMG+ caliber-2 on the anisotropic J.
        mfp = _mfp(nlf_grid2d(8; cap_lo = 0.2, cap_hi = 5.0, rng = MersenneTwister(2)))
        F, φ, f, info = nlf_maxflow(mfp; tol = 1e-8, inner = :multigrid)
        @test isapprox(F, 5.8498; rtol = 2e-3)
        cons, box = _conserved_feasible(mfp, F, f; rtol = 1e-4)
        @test cons
        @test box
        mfpb, Fk = nlf_bottleneck_chain(10; cap_chain = 15.0, cap_bottle = 3.0)
        Fb, _, _, _ = nlf_maxflow(mfpb; tol = 1e-8, inner = :multigrid)
        @test isapprox(Fb, Fk; rtol = 1e-3)
    end

    @testset "α cannot exceed the s-cut capacity bound" begin
        mfp = _mfp(nlf_grid2d(8; cap_lo = 0.2, cap_hi = 5.0, rng = MersenneTwister(2)))
        F, _, _, _ = nlf_maxflow(mfp)
        scut = 0.0
        for e in 1:length(mfp.head)
            (mfp.head[e] == mfp.s || mfp.tail[e] == mfp.s) &&
                (scut += max(mfp.high[e], -mfp.low[e]))
        end
        @test F <= scut + 1e-6
    end
end
