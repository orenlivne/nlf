# Reproducible numbers for Part II Table: nlf_maxflow (smooth-rho alpha-continuation) vs TRUE
# max-flow, with the gradient/electrical relaxation ratio for contrast. Both inner solves.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
using JuMP, HiGHS
import GraphsFlows, Graphs
import LAMG: nlf_grid2d, nlf_grid3d, nlf_random, nlf_bottleneck_chain,
            nlf_genrmf, nlf_washington, nlf_acyclic_dense, nlf_maxflow, solve_alpha_max
un(x)=x isa Tuple ? first(x) : x
function true_lp(mfp)
    B=mfp.B; d=mfp.d; m=size(B,2)
    md=Model(HiGHS.Optimizer); set_silent(md)
    @variable(md,f[1:m]); @variable(md,a)
    @constraint(md,B*f .== a.*d)
    @constraint(md,[e=1:m],f[e]>=mfp.low[e]); @constraint(md,[e=1:m],f[e]<=mfp.high[e])
    @objective(md,Max,a); optimize!(md); value(a)
end
function true_gr(mfp)
    n=size(mfp.A,1); g=Graphs.DiGraph(n); cap=zeros(n,n)
    for e in 1:size(mfp.B,2)
        h=mfp.head[e]; ta=mfp.tail[e]
        Graphs.add_edge!(g,ta,h); cap[ta,h]+=mfp.high[e]
        Graphs.add_edge!(g,h,ta); cap[h,ta]+=-mfp.low[e]
    end
    GraphsFlows.maximum_flow(g,mfp.s,mfp.t,cap)[1]
end
println("Part II results table (LaTeX-ready):")
@printf("%-13s %-6s %-9s %-7s %-9s %-7s %-9s %-7s\n",
        "graph","n","F*(LP)","=PR?","grad/F*","dir/F*","dirSteps","mg/F*")
function run(name,mfp)
    n=size(mfp.A,1); Flp=true_lp(mfp); Fgr=true_gr(mfp)
    Fg,_=solve_alpha_max(mfp)
    Fd,_,_,id=nlf_maxflow(mfp; tol=1e-8, inner=:direct)
    Fm,_,_,im=nlf_maxflow(mfp; tol=1e-8, inner=:multigrid)
    @printf("%-13s %-6d %-9.4f %-7s %-9.4f %-7.4f %-9d %-7.4f\n",
            name,n,Flp, (abs(Flp-Fgr)<1e-3 ? "yes" : "NO"),
            Fg/Flp, Fd/Flp, id.steps, Fm/Flp)
end
run("grid2d/16",  un(nlf_grid2d(16; cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(3))))
run("grid2d/24",  un(nlf_grid2d(24; cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(7))))
run("grid3d/6",   un(nlf_grid3d(6;  cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(8))))
run("random/200", un(nlf_random(200,0.05; cap_lo=0.2,cap_hi=5.0,rng=MersenneTwister(5))))
run("washington", un(nlf_washington(8,8; rng=MersenneTwister(12))))
run("genrmf",     un(nlf_genrmf(4,4,4; rng=MersenneTwister(13))))
run("bottleneck", un(nlf_bottleneck_chain(40)))
println("DONE_PAPER_TABLE")
