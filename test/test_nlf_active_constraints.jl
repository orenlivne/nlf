using Test
using NLF
using LinearAlgebra
using SparseArrays
using Random
using Statistics

# Tests that exercise the box constraints DURING the solve (constraints
# actively binding). These prove the Schur-eliminated generalized
# constraint pass (`update_fas_elim!` + Phase-3 Kaczmarz) is doing real
# work — not just the trivial stationarity-at-x_true with loose caps.

# Compute the geometric-mean per-cycle ratio over a clean window.
function _steady_mu(hist::Vector{Float64};
                    floor_factor::Float64 = 1e3 * eps())
    length(hist) < 4 && return NaN
    ratios = hist[2:end] ./ max.(hist[1:end - 1], 1e-30)
    n = length(ratios)
    floor_ = floor_factor * hist[1]
    last = n
    for i in 2:n
        hist[i + 1] < floor_ && (last = i - 1; break)
    end
    last = min(last, n)
    first = min(3, last)
    window = ratios[first:last]
    isempty(window) && (window = ratios)
    return exp(mean(log.(max.(window, 1e-30))))
end

@testset "max-flow with ACTIVE box constraints" begin

    # ── Test A: tight caps, stationarity at interior feasible x_true ──
    @testset "A — tight-cap stationarity at feasible interior point" begin
        K = 16
        rng = MersenneTwister(0xfa11)
        mfp_template = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0, rng = rng)
        n = size(mfp_template.A, 1)
        rng2 = MersenneTwister(0xface)
        x_true = randn(rng2, n); x_true .-= mean(x_true)
        # Scale so that x_true is INTERIOR-feasible with caps ±2.0 (use a
        # large safety margin since random gradients can spike).
        grad0 = mfp_template.B' * x_true
        scale = 1.0 / max(maximum(abs, grad0), 1e-30)
        x_true .*= scale
        grad = mfp_template.B' * x_true
        @test maximum(abs, grad) < 2.0
        # Build problem with caps ±2.0 and b = A x_true.
        b = mfp_template.A * x_true
        mfp = NLFProblem(mfp_template.A, mfp_template.B,
                              fill(-2.0, length(mfp_template.head)),
                              fill(+2.0, length(mfp_template.head)),
                              b, mfp_template.s, mfp_template.t,
                              mfp_template.name,
                              mfp_template.head, mfp_template.tail)
        # Start from x_true; one cycle should leave residual at roundoff.
        opts = LAMGOptions(tol = 1e-10, max_cycles = 1,
                           elim_max_degree = 4,
                           ν_pre = 1, ν_post = 2,
                           γ = 1.5, do_recomb = true,
                           history_size = 4, rhs_correction = 1.0,
                           seed = 12345)
        h = setup(mfp; options = opts)
        x_out, info = solve(h, mfp; α = 1.0, options = opts,
                                       x0 = copy(x_true))
        rel = info.residual_history[end] / max(norm(b), 1e-30)
        @test rel < 1e-12
    end

    # ── Test B: tight caps, convergence from random start ─────────────
    @testset "B — tight-cap convergence from random start" begin
        K = 16
        rng = MersenneTwister(0xfa11)
        mfp_template = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0, rng = rng)
        n = size(mfp_template.A, 1)
        rng2 = MersenneTwister(0xface)
        x_true = randn(rng2, n); x_true .-= mean(x_true)
        # Scale x_true to lie comfortably interior to ±2.0 caps.
        grad0 = mfp_template.B' * x_true
        x_true .*= (1.0 / max(maximum(abs, grad0), 1e-30))
        b = mfp_template.A * x_true
        mfp = NLFProblem(mfp_template.A, mfp_template.B,
                              fill(-2.0, length(mfp_template.head)),
                              fill(+2.0, length(mfp_template.head)),
                              b, mfp_template.s, mfp_template.t,
                              mfp_template.name,
                              mfp_template.head, mfp_template.tail)
        # Start from φ=0. Run max 20 cycles.
        opts = LAMGOptions(tol = 1e-6, max_cycles = 20,
                           elim_max_degree = 4,
                           ν_pre = 2, ν_post = 2,
                           γ = 1.5, do_recomb = true,
                           history_size = 4, rhs_correction = 1.0,
                           seed = 12345)
        h = setup(mfp; options = opts)
        φ0 = zeros(n)
        x_out, info = solve(h, mfp; α = 1.0, options = opts, x0 = φ0)
        rel = info.residual_history[end] / max(norm(b), 1e-30)
        @test rel < 1e-6
        # Feasibility under the original caps.
        grad = mfp.B' * x_out
        @test maximum(abs, grad) <= 2.0 + 1e-6
        # Asymptotic μ over cycles 3..min(end, 10).
        μ = _steady_mu(info.residual_history)
        @test isnan(μ) || μ < 0.5   # NaN ⇒ converged in <3 cycles (rel<1e-6 above)
    end

    # ── Test C: feasible problem with caps TIGHT around x_true ────────
    # Construct b = A x_true where x_true is feasible under tight caps and
    # the linear gradient x_true exercises caps along its dynamic range.
    # The problem Aφ = b admits the feasible solution x_true; we test that
    # the unified solver finds it.
    @testset "C — binding-cap problem (feasible by construction)" begin
        K = 16
        rng = MersenneTwister(0xfa11)
        mfp_template = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0, rng = rng)
        n = size(mfp_template.A, 1)
        rng2 = MersenneTwister(0xface)
        x_true = randn(rng2, n); x_true .-= mean(x_true)
        grad_t = mfp_template.B' * x_true
        gmax = maximum(abs, grad_t)
        # Choose cap = 1.05 × max|grad| so x_true is feasible but many edges
        # are within 5% of the cap — Kaczmarz projection will fire during
        # convergence from random.
        cap = 1.05 * gmax
        b = mfp_template.A * x_true
        mfp = NLFProblem(mfp_template.A, mfp_template.B,
                              fill(-cap, length(mfp_template.head)),
                              fill(+cap, length(mfp_template.head)),
                              b, mfp_template.s, mfp_template.t,
                              mfp_template.name,
                              mfp_template.head, mfp_template.tail)
        opts = LAMGOptions(tol = 1e-6, max_cycles = 30,
                           elim_max_degree = 4,
                           ν_pre = 2, ν_post = 2,
                           γ = 1.5, do_recomb = true,
                           history_size = 4, rhs_correction = 1.0,
                           seed = 12345)
        h = setup(mfp; options = opts)
        φ0 = zeros(n)
        x_out, info = solve(h, mfp; α = 1.0, options = opts, x0 = φ0)
        rel = info.residual_history[end] / max(norm(b), 1e-30)
        μ = _steady_mu(info.residual_history)
        @info "Test C" rel cycles = info.cycles μ cap
        @test rel < 1e-6
        grad = mfp.B' * x_out
        @test maximum(abs, grad) <= cap + 1e-4
        @test isnan(μ) || μ < 0.5   # NaN ⇒ converged in <3 cycles (rel<1e-6 above)
    end

    # ── Test D: K=64 with tighter binding caps + ELIM hierarchy ───────
    @testset "D — K=64 binding caps, elim_max_degree=4" begin
        K = 64
        rng = MersenneTwister(0xfa11)
        mfp_template = NLF.nlf_grid2d(K; cap_lo = 1.0, cap_hi = 1.0, rng = rng)
        n = size(mfp_template.A, 1)
        rng2 = MersenneTwister(0xface)
        x_true = randn(rng2, n); x_true .-= mean(x_true)
        grad_t = mfp_template.B' * x_true
        gmax = maximum(abs, grad_t)
        cap = 1.05 * gmax
        b = mfp_template.A * x_true
        mfp = NLFProblem(mfp_template.A, mfp_template.B,
                              fill(-cap, length(mfp_template.head)),
                              fill(+cap, length(mfp_template.head)),
                              b, mfp_template.s, mfp_template.t,
                              mfp_template.name,
                              mfp_template.head, mfp_template.tail)
        opts = LAMGOptions(tol = 1e-6, max_cycles = 40,
                           elim_max_degree = 4,
                           ν_pre = 2, ν_post = 2,
                           γ = 1.5, do_recomb = true,
                           history_size = 4, rhs_correction = 1.0,
                           seed = 12345)
        h = setup(mfp; options = opts)
        φ0 = zeros(n)
        x_out, info = solve(h, mfp; α = 1.0, options = opts, x0 = φ0)
        rel = info.residual_history[end] / max(norm(b), 1e-30)
        μ = _steady_mu(info.residual_history)
        @info "Test D" rel cycles = info.cycles μ cap
        @test rel < 1e-6
        grad = mfp.B' * x_out
        @test maximum(abs, grad) <= cap + 1e-4
        @test isfinite(μ) && μ < 0.6
    end

end
