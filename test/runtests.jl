using Test
using NLF   # re-exports the LAMG+ inner-solver API (laplacian, setup, solve, Multilevel, ...)

# LAMG+ linear-solver tests now live in the LAMG+ package (github.com/orenlivne/lamgplus).
# This suite covers only the NLF max-flow / network-flow layer built on top of it.
@testset "NLF" begin
    include("test_maxflow.jl")
    include("test_nlf_constrained.jl")
    include("test_nlf_elimination.jl")
    include("test_box_coarsening.jl")
    include("test_box_elim_generalized.jl")    # Schur-substituted box rows
    include("test_box_elim_tau_shift.jl")      # anchored PFAS τ-shift (#134)
    include("test_unified_hierarchy.jl")
    include("test_nlf_stationarity.jl")
    include("test_nlf_unified_stationarity.jl")
    include("test_nlf_examples.jl")
    include("test_nlf_multilevel.jl")
    include("test_nlf_correctness.jl")
    include("test_nlf_active_constraints.jl")   # active-box-constraint solves
    include("test_rho_maxflow_equivalence.jl")      # saturating-ρ == combinatorial max-flow
    include("test_nlf_alpha_from_below.jl")     # α-from-below recovers F* (no overshoot)
    include("test_box_projected_gs.jl")             # voltage-driving smoother (moderate V)
    include("test_nlf_maxflow.jl")                 # TRUE max-flow: α-continuation + cut-mode bordering
    include("test_lazy_refresh.jl")                 # lazy LAMG+ hierarchy refresh (same F*, fewer setups)
    include("test_nlf_flow.jl")                     # generic source-form nonlinear-flow solver + hooks
end
