# Settle the foundational question: is solve_alpha_max's F* (gradient-flow LP, f=B'phi)
# the TRUE combinatorial max-flow, or a relaxation? And where does V-driving (0.72) sit?
#
#   true max-flow LP : max alpha  s.t.  B f = alpha d,  low <= f       <= high,  f FREE in R^m
#   gradient   LP    : max alpha  s.t.  A phi = alpha d, low <= (B'phi) <= high   (f = B'phi)
#   solve_alpha_max  : closed form of the gradient LP  (phi = alpha * A^+ d)
#
# If true > gradient, the constrained-linear (shipped) formulation is itself a relaxation.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
using JuMP, HiGHS
import LAMG: nlf_grid2d, nlf_grid3d, nlf_random, nlf_bottleneck_chain, solve_alpha_max
un(x) = x isa Tuple ? x[1] : x

# True max-flow over FREE edge flows f.
function true_maxflow(mfp)
    B = mfp.B; d = mfp.d; m = size(B,2); n = size(B,1)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, f[1:m]); @variable(model, α)
    @constraint(model, B*f .== α .* d)
    @constraint(model, [e=1:m], f[e] >= mfp.low[e])
    @constraint(model, [e=1:m], f[e] <= mfp.high[e])
    @objective(model, Max, α)
    optimize!(model)
    ts = termination_status(model)
    (ts == MOI.OPTIMAL) ? value(α) : NaN
end

# Gradient-flow LP (f restricted to be a potential gradient): should equal solve_alpha_max.
function gradient_maxflow(mfp)
    B = mfp.B; A = mfp.A; d = mfp.d; m = size(B,2); n = size(B,1)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, φ[1:n]); @variable(model, α)
    @constraint(model, A*φ .== α .* d)
    g = B' * φ
    @constraint(model, [e=1:m], g[e] >= mfp.low[e])
    @constraint(model, [e=1:m], g[e] <= mfp.high[e])
    @constraint(model, φ[mfp.t] == 0)         # gauge
    @objective(model, Max, α)
    optimize!(model)
    ts = termination_status(model)
    (ts == MOI.OPTIMAL) ? value(α) : NaN
end

println("-- true max-flow (free f) vs gradient LP (f=B'phi) vs solve_alpha_max --")
@printf("%-14s %-5s %-12s %-12s %-12s %-8s\n",
        "instance","n","true MF","gradient LP","solve_a_max","grad/true")
function run(name, mfp)
    n = size(mfp.A,1)
    Ft = true_maxflow(mfp)
    Fg = gradient_maxflow(mfp)
    Fs, _ = solve_alpha_max(mfp)
    @printf("%-14s %-5d %-12.5f %-12.5f %-12.5f %-8.4f\n",
            name, n, Ft, Fg, Fs, Fg/max(Ft,1e-12))
end
run("grid2d-8",   un(nlf_grid2d(8;  cap_lo=0.2, cap_hi=5.0, rng=MersenneTwister(2))))
run("grid2d-16",  un(nlf_grid2d(16; cap_lo=0.2, cap_hi=5.0, rng=MersenneTwister(3))))
run("grid3d-4",   un(nlf_grid3d(4;  cap_lo=0.2, cap_hi=5.0, rng=MersenneTwister(4))))
run("random-200", un(nlf_random(200,0.05; cap_lo=0.2, cap_hi=5.0, rng=MersenneTwister(5))))
run("bottleneck", un(nlf_bottleneck_chain(40)))
println("DONE_TRUE")
