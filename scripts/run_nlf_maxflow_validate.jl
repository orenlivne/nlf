# Validate the new nlf_maxflow (smooth-rho alpha-continuation + cut-mode bordering, inner=:direct)
# against the TRUE max-flow (HiGHS LP over free flows AND GraphsFlows push-relabel).
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
using JuMP, HiGHS
import GraphsFlows, Graphs
import LAMG: nlf_grid2d, nlf_grid3d, nlf_random, nlf_bottleneck_chain,
            nlf_genrmf, nlf_washington, nlf_acyclic_dense, nlf_maxflow
un(x) = x isa Tuple ? x[1] : x

function true_lp(mfp)
    B=mfp.B; d=mfp.d; m=size(B,2)
    model=Model(HiGHS.Optimizer); set_silent(model)
    @variable(model,f[1:m]); @variable(model,α)
    @constraint(model,B*f .== α.*d)
    @constraint(model,[e=1:m],f[e]>=mfp.low[e]); @constraint(model,[e=1:m],f[e]<=mfp.high[e])
    @objective(model,Max,α); optimize!(model)
    termination_status(model)==MOI.OPTIMAL ? value(α) : NaN
end
function true_graphs(mfp)
    n=size(mfp.A,1); m=size(mfp.B,2)
    g=Graphs.DiGraph(n); cap=zeros(n,n)
    for e in 1:m
        h=mfp.head[e]; ta=mfp.tail[e]
        Graphs.add_edge!(g,ta,h); cap[ta,h]+=mfp.high[e]
        Graphs.add_edge!(g,h,ta); cap[h,ta]+=-mfp.low[e]
    end
    GraphsFlows.maximum_flow(g,mfp.s,mfp.t,cap)[1]
end

println("===== nlf_maxflow (alpha-continuation + cut-mode bordering) vs TRUE max-flow =====")
@printf("%-16s %-6s %-10s %-10s %-10s %-8s %-7s %-9s\n",
        "instance","n","trueLP","trueGr","nlf","ratio","steps","resid")
function run(name,mfp)
    n=size(mfp.A,1)
    Flp=true_lp(mfp); Fgr=true_graphs(mfp)
    Fd,_,fd,id=nlf_maxflow(mfp; tol=1e-8, inner=:direct)
    Fm,_,fm,im=nlf_maxflow(mfp; tol=1e-8, inner=:multigrid)
    boxd = all(mfp.low .- 1e-7 .<= fd .<= mfp.high .+ 1e-7)
    boxm = all(mfp.low .- 1e-6 .<= fm .<= mfp.high .+ 1e-6)
    @printf("%-14s %-5d %-9.4f %-9.4f | dir %-9.4f %-7.4f %-2d%s | mg %-9.4f %-7.4f %-2d%s\n",
            name,n,Flp,Fgr,
            Fd,Fd/max(Fgr,1e-12),id.steps,boxd ? "" : "!",
            Fm,Fm/max(Fgr,1e-12),im.steps,boxm ? "" : "!")
end
run("grid2d-8",   un(nlf_grid2d(8;  cap_lo=0.2,cap_hi=5.0,rng=MersenneTwister(2))))
run("grid2d-16",  un(nlf_grid2d(16; cap_lo=0.2,cap_hi=5.0,rng=MersenneTwister(3))))
run("grid2d-24",  un(nlf_grid2d(24; cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(7))))
run("grid3d-4",   un(nlf_grid3d(4;  cap_lo=0.2,cap_hi=5.0,rng=MersenneTwister(4))))
run("grid3d-6",   un(nlf_grid3d(6;  cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(8))))
run("random-200", un(nlf_random(200,0.05; cap_lo=0.2,cap_hi=5.0,rng=MersenneTwister(5))))
run("random-500", un(nlf_random(500,0.02; cap_lo=0.1,cap_hi=10.0,rng=MersenneTwister(9))))
run("acyclic-200",un(nlf_acyclic_dense(200,0.05; rng=MersenneTwister(11))))
run("washington", un(nlf_washington(8,8; rng=MersenneTwister(12))))
run("genrmf",     un(nlf_genrmf(4,4,4; rng=MersenneTwister(13))))
run("bottleneck", un(nlf_bottleneck_chain(40)))
println("DONE_VALIDATE")
