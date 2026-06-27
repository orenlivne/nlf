# Lazy hierarchy refresh (nlf_maxflow refresh kwarg): the LAMG+ hierarchy is rebuilt only when
# the per-step residual factor degrades or a stale stall occurs. Correctness: the lazily refreshed
# multigrid solver must reach the same F* as rebuild-every-step and as the direct inner solve.
using Test, Random
using NLF
import NLF: nlf_grid2d, nlf_bottleneck_chain, nlf_maxflow

@testset "lazy hierarchy refresh" begin
    # grid with heterogeneous capacities: nontrivial cut, several continuation steps
    g = nlf_grid2d(12; cap_lo = 0.5, cap_hi = 3.0, rng = MersenneTwister(7))
    mfp = g isa Tuple ? g[1] : g

    Fd, _, fd, infod = nlf_maxflow(mfp; inner = :direct)
    @test infod.converged

    # rebuild-every-step (refresh = 0) == old behaviour; lazy (default 0.25); frozen-greedy (0.9)
    F0, _, _, info0 = nlf_maxflow(mfp; inner = :multigrid, refresh = 0.0)
    Fl, _, fl, infol = nlf_maxflow(mfp; inner = :multigrid)            # default refresh = 0.25
    Fg, _, _, infog = nlf_maxflow(mfp; inner = :multigrid, refresh = 0.9)

    @test info0.converged && infol.converged && infog.converged
    @test isapprox(F0, Fd; rtol = 1e-4)
    @test isapprox(Fl, Fd; rtol = 1e-4)      # lazy refresh changes cost, not the answer
    @test isapprox(Fg, Fd; rtol = 1e-4)      # even near-frozen stays correct (stall safeguard)

    # flows stay capacity-feasible under lazy refresh
    @test all(mfp.low .- 1e-6 .<= fl .<= mfp.high .+ 1e-6)

    # chain (single bottleneck cut, F* known exactly = bottleneck capacity)
    r = nlf_bottleneck_chain(30; cap_chain = 1.0, cap_bottle = 1.0)
    mfp2, Fstar = r[1], r[2]
    Fl2, _, _, info2 = nlf_maxflow(mfp2; inner = :multigrid)
    @test info2.converged
    @test isapprox(Fl2, Fstar; rtol = 1e-3)

    # machine precision at 0.995 F* under lazy refresh: the staleness safeguard must not stall
    B = mfp.B; n = size(B, 1); m2 = size(B, 2)
    d = Vector{Float64}(mfp.d); low = mfp.low; high = mfp.high
    α = 0.995 * Fd
    φw, _ = NLF._nlf_alpha_newton(B, d, 0.9α, zeros(n), low, high, zeros(m2), zeros(m2);
                                    inner = :direct)
    φp, convp = NLF._nlf_alpha_newton(B, d, α, φw, low, high, zeros(m2), zeros(m2);
                                        inner = :multigrid, nmax = 200, tol = 1e-13)
    fp = zeros(m2); dρp = zeros(m2)
    NLF._nlf_rho_drho!(fp, dρp, B' * φp, low, high)
    @test convp
    @test norm(B * fp .- α .* d) / max(α, 1.0) < 1e-12     # machine precision, no stall

    # tight-tolerance continuation must not stall either (the two-attempt corrector safeguard)
    Ft, _, _, infot = nlf_maxflow(mfp; inner = :multigrid, tol = 1e-9)
    @test infot.converged
    @test isapprox(Ft, Fd; rtol = 1e-6)
end
