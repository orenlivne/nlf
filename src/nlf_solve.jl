# Max-flow FAS solve reusing the UNIFIED LAMG+ multilevel cycle.
#
# This method was previously appended to LAMG+'s `setup.jl`; it now lives on the NLF side
# and EXTENDS `LAMG.solve` (imported in NLF.jl). It reaches two non-exported LAMG internals,
# `LAMG.result` and `LAMG._build_gamma_vec`, by qualified name -- everything else is part of
# LAMG+'s public surface.

"""
    solve(h::Multilevel, mfp::NLFProblem;
          α::Real = 1.0,
          options::LAMGOptions = LAMGOptions(),
          x0::Union{Nothing,AbstractVector} = nothing) -> (φ, info)

Max-flow FAS solve reusing the UNIFIED multilevel cycle. Internally identical to
`solve(h, b)` except:
- RHS is `α · mfp.d` (the source/sink vector scaled by the flow).
- Convergence is measured against `‖α · mfp.d‖`.
- The cycle runs through `SolveCycleProcessor` as on the linear path; box-aware
  relaxation + per-cycle τ-correction are activated by the per-level box metadata
  attached by `setup(mfp)`.

`h` must have been built via `setup(mfp; options)` (so every level carries
`head/tail/low/high/low0/high0/mfp`). The linear `setup(A)` chain does NOT produce a
box-aware hierarchy.
"""
function solve(h::Multilevel, mfp::NLFProblem;
               α::Real = 1.0,
               options::LAMGOptions = LAMGOptions(),
               x0::Union{Nothing,AbstractVector} = nothing)
    @assert finest_level(h).relaxer isa MaxFlowGSKaczmarzRelaxer ""*
        "hierarchy `h` is not box-equipped; build it with " *
        "`setup(mfp::NLFProblem)`."
    n = size(finest_level(h))
    A = finest_level(h).a
    b = α .* mfp.d
    @assert length(b) == n "RHS size mismatch (α·d has length $(length(b)), level expects $n)"
    b_norm = norm(b)
    b_norm == 0 && return zeros(n), (cycles = 0, residual_history = [0.0],
                                      conv_factors = Float64[],
                                      final_residual = 0.0,
                                      solve_time = 0.0,
                                      gamma_escalated = false)
    x = x0 === nothing ? zeros(Float64, n) : collect(Float64.(x0))
    x_init = copy(x)
    proc = SolveCycleProcessor(h, b;
                               ν_pre = options.ν_pre, ν_post = options.ν_post,
                               ν_coarsest = options.ν_coarsest,
                               do_recomb = options.do_recomb,
                               recomb_above_elim = options.elim_sample_rho > 0,
                               history_size = options.history_size,
                               use_direct_coarsest = (options.ν_coarsest == -1),
                               rhs_correction = options.rhs_correction)
    γ_vec = LAMG._build_gamma_vec(options.γ, options.γ_coarse,
                                  options.γ_coarse_growth, length(h), h)
    cyc = Cycle(proc, γ_vec, length(h))
    residual_history = Float64[b_norm]
    conv_factors = Float64[]
    t_solve = @elapsed begin
        for k in 1:options.max_cycles
            run_cycle!(cyc, x)
            x = copy(LAMG.result(proc, 1))
            r = norm(b .- A * x)
            push!(residual_history, r)
            push!(conv_factors, r / residual_history[end - 1])
            r <= options.tol * b_norm && break
        end
    end
    info = (cycles = length(residual_history) - 1,
            residual_history = residual_history,
            conv_factors = conv_factors,
            final_residual = residual_history[end],
            solve_time = t_solve)
    return x, info
end
