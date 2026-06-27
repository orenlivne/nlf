"""
    NLF

Flow Algebraic Multigrid: a nonlinear, box-constrained max-flow / network-flow solver
built ON TOP of the LAMG+ linear graph-Laplacian solver.

LAMG+ (the `LAMG` package, github.com/orenlivne/lamgplus) is the inner linear engine and is
consumed as a dependency -- its source is no longer vendored here. NLF reuses LAMG+'s
multilevel hierarchy, cycle, and relaxation framework, and extends them with:

  * a box-projected Gauss--Seidel/Kaczmarz relaxer (`MaxFlowGSKaczmarzRelaxer`),
  * per-level box metadata + an anchored PFAS τ-shift on the coarse active set,
  * a nonlinear FAS / FMG-FAS cycle and α-continuation for the flow value.

NLF extends exactly these LAMG generics (imported below so the bare method
definitions in the NLF sources extend rather than shadow them):
`relax!`, `setup`, `solve`, `update_fas!`, `update_fas_elim!`, `num_levels`.
"""
module NLF

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using Base.Threads: @threads, nthreads

# The LAMG+ inner solver (dependency). `using` brings its public API into scope; the
# explicit `import` list is the set of LAMG generics the NLF sources add methods to.
using LAMG
import LAMG: relax!, setup, solve, update_fas!, update_fas_elim!, num_levels

# NLF sources (the max-flow / network-flow layer). Order matters: problems & relaxers
# before the cycle/elimination/box machinery that uses them; nlf_solve last (extends
# `solve` and needs NLFProblem + MaxFlowGSKaczmarzRelaxer).
include("maxflow.jl")
include("nlf_problems.jl")
include("nlf_examples.jl")
include("nlf_constrained.jl")
include("nlf_elimination.jl")
include("box_coarsening.jl")
include("nlf_multilevel.jl")
include("nlf_test_set.jl")
include("nlf_directed.jl")
include("nlf_maxflow.jl")
include("nlf_solve.jl")

# Re-export the entire LAMG+ public API so `using NLF` also exposes the inner solver
# (laplacian, Level, Multilevel, LAMGOptions, setup, solve, ...), preserving the old
# single-import ergonomics now that LAMG+ lives in its own package.
for n in names(LAMG)
    n === :LAMG && continue
    @eval export $n
end

export
    # maxflow (Phase 1 — barrier formulation, legacy)
    MaxFlowLaplacian, NonlinearGSRelaxer, edge_resistance, edge_conductance,
    # box-projected relaxer (NLF's Relaxer subtype)
    MaxFlowGSKaczmarzRelaxer,
    # maxflow (Phase 2 — constrained-linear formulation, current)
    NLFProblem, make_problem, incidence_from_edge_list,
    nlf_grid2d, nlf_grid3d, nlf_random, load_dimacs_max,
    nlf_genrmf, nlf_washington, nlf_acyclic_dense,
    nlf_clrs, nlf_bottleneck_chain, nlf_bipartite_matching,
    nlf_image_seg_2x2,
    relax_gs_kaczmarz!, relax_block_gs!, shrinkage_gs, shrinkage_gs_kaczmarz,
    coarsen_constraints, build_coarse_problem, fas_2level_cycle!,
    coarsen_box_agg, coarsen_box_elim, GeneralizedConstraint,
    MaxFlowEliminationStage, low_degree_nlf_nodes,
    eliminate_low_degree, interpolate_eliminated!,
    fas_2level_cycle_elimination!,
    NLFHierarchy, setup_nlf_hierarchy,
    fas_multilevel_cycle!, fas_multilevel_solve!, fmg_fas!,
    solve_alpha_max, fmg_fas_alpha_opt!, fmg_fas_alpha_continuation!,
    nlf_alpha_from_below, box_projected_gs!,
    nlf_maxflow,
    MaxFlowTestCase, build_test_set,
    DirectedMaxFlowState, relax_arrow_hurwicz!, shrinkage_arrow_hurwicz,
    relax_chambolle_pock!, shrinkage_chambolle_pock

end # module
