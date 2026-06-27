"""
test_box_elim_tau_shift.jl — the anchored PFAS τ-shift for Schur-substituted
box constraints (update_fas_elim! + _project_generalized!).

The coarse generalized row reads  low_active ≤ vals' x_C[cols] ≤ high_active.
update_fas_elim! anchors the active bounds to the CURRENT fine edge gap:

    c0          = vals' x_C0[cols] - (φ[fine_head] - φ[fine_tail])
    low_active  = low_static  + c0
    high_active = high_static + c0

KEY PROPERTY (PFAS / Brandt-Cryer): at the FAS-initial coarse iterate x_C0 the
projection is INERT iff the fine edge currently satisfies its original box, and
when it engages it corrects the coarse iterate by exactly the amount needed to
pull the *fine* edge back to its box boundary.
"""

using Test
using NLF
using LinearAlgebra
using SparseArrays
using Random
import NLF: MaxFlowGSKaczmarzRelaxer, GeneralizedConstraint,
             update_fas_elim!, _project_generalized!, make_problem,
             nlf_grid3d, solve_alpha_max, _PROJ_GEN

# A single generalized row g_C(x_C) = x_C[1] - x_C[2], box [-1, +1], standing
# for fine edge (fine_head=1, fine_tail=2). The coarse 2-node NLFProblem is
# only a structural carrier — the projection acts purely on the generalized row.
function _make_rx()
    mfp = make_problem(2, [(1, 2)], [1.0], [1.0], 1, 2; name = "tau-shift-unit")
    g = GeneralizedConstraint([1, 2], [1.0, -1.0], -1.0, +1.0, 0.0, 1, 1, 2)
    return MaxFlowGSKaczmarzRelaxer(mfp; generalized = [g])
end

@testset "τ-shift: anchor offset c0" begin
    rx = _make_rx()
    x_C0   = [2.0, 0.0]           # restricted fine iterate (coarse numbering)
    φ_fine = [0.3, 0.0]           # current fine iterate; gap = 0.3 (feasible)
    update_fas_elim!(rx, x_C0, φ_fine)
    # c0 = (2.0-0.0) - (0.3-0.0) = 1.7
    @test rx.low_active[1]  ≈ -1.0 + 1.7
    @test rx.high_active[1] ≈ +1.0 + 1.7
end

@testset "τ-shift: no-op when fine edge feasible" begin
    rx = _make_rx()
    x_C0   = [2.0, 0.0]
    φ_fine = [0.3, 0.0]           # gap 0.3 ∈ [-1,1] → feasible
    update_fas_elim!(rx, x_C0, φ_fine)
    x = copy(x_C0)
    _project_generalized!(rx, x)
    # g_C(x_C0) = 2.0 lies inside the shifted band [0.7, 2.7] ⇒ no change,
    # even though 2.0 is OUTSIDE the static band [-1,1].
    @test x ≈ x_C0
end

@testset "τ-shift: engages when fine edge infeasible, pulls fine to boundary" begin
    rx = _make_rx()
    x_C0   = [2.0, 0.0]
    φ_fine = [1.5, 0.0]           # gap 1.5 ∉ [-1,1] → infeasible (over by 0.5)
    update_fas_elim!(rx, x_C0, φ_fine)
    # c0 = 2.0 - 1.5 = 0.5 ⇒ band [-0.5, 1.5]
    @test rx.low_active[1]  ≈ -0.5
    @test rx.high_active[1] ≈ +1.5
    x = copy(x_C0)
    _project_generalized!(rx, x)
    s_new = x[1] - x[2]
    @test s_new ≈ 1.5          # projected onto upper active bound
    # PFAS guarantee: implied fine-gap after this coarse correction =
    #   fine_gap + (g_C(x) - g_C(x_C0)) = 1.5 + (1.5 - 2.0) = 1.0  (back in box)
    implied_fine_gap = 1.5 + (s_new - (x_C0[1] - x_C0[2]))
    @test implied_fine_gap ≈ 1.0
    @test -1.0 - 1e-12 ≤ implied_fine_gap ≤ 1.0 + 1e-12
end

@testset "τ-shift: empty generalized set is a fast no-op" begin
    mfp = make_problem(2, [(1, 2)], [1.0], [1.0], 1, 2; name = "empty")
    rx = MaxFlowGSKaczmarzRelaxer(mfp)        # no generalized rows
    @test update_fas_elim!(rx, [0.0, 0.0], [0.0, 0.0]) === rx
    @test isempty(rx.generalized)
end

# In-situ correctness: enabling the coarse-ELIM projection must NOT move the
# converged solution. This exercises the full hierarchy (real generalized rows
# on grid3d ELIM levels) and proves the anchored τ-shift is consistent at the
# fixed point — even though the projection is gated OFF by default (it
# destabilises cold-start 3D transients; see _PROJ_GEN docs / handoff §4).
@testset "τ-shift: projection is inert at the converged solution (grid3d)" begin
    mfp = nlf_grid3d(8; rng = MersenneTwister(2024))
    αmax, _ = solve_alpha_max(mfp)
    α = 0.05 * αmax
    opts = LAMGOptions(tol = 1e-12, max_cycles = 100, ν_pre = 2, ν_post = 2,
                       γ = 1.5, do_recomb = true, history_size = 4,
                       rhs_correction = 1.0, elim_max_degree = 4, seed = 0xfa11)
    saved = _PROJ_GEN[]
    try
        _PROJ_GEN[] = false                      # converge on the stable baseline
        h = setup(mfp; options = opts)
        φ, _ = solve(h, mfp; α = α, options = opts)
        A = h[1].a; b = α .* mfp.d
        @test norm(b .- A * φ) / norm(b) < 1e-8  # genuinely converged

        _PROJ_GEN[] = true                       # one projected cycle from φ*
        h2 = setup(mfp; options = opts)
        opts1 = LAMGOptions(tol = 1e-12, max_cycles = 1, ν_pre = 2, ν_post = 2,
                            γ = 1.5, do_recomb = true, history_size = 4,
                            rhs_correction = 1.0, elim_max_degree = 4,
                            seed = 0xfa11)
        φ2, _ = solve(h2, mfp; α = α, options = opts1, x0 = copy(φ))
        # The projected cycle barely moves φ* — stationarity preserved.
        @test norm(φ2 - φ) / max(norm(φ), 1e-30) < 1e-6
        @test norm(b .- A * φ2) / norm(b) < 1e-6
    finally
        _PROJ_GEN[] = saved
    end
end
