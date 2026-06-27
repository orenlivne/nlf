# Scaling study for the FULL nlf_maxflow algorithm (setup+solve combined — the hierarchy is
# rebuilt within the algorithm, so they are not separable). Times total wall-clock on grids of
# increasing size; emits CSV for the point-cloud + regression-line figure (cf. LAMG+ Fig 4.1).
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
import LAMG: nlf_grid2d, nlf_grid3d, nlf_random, nlf_maxflow
un(x)=x isa Tuple ? first(x) : x

INNER  = length(ARGS)>=1 ? Symbol(ARGS[1]) : :multigrid
METHOD = length(ARGS)>=2 ? Symbol(ARGS[2]) : :arclength
OUT    = length(ARGS)>=3 ? ARGS[3] : "/tmp/nlf_scaling.csv"

cases = Tuple{String,Function}[]
for k in (8,12,16,24,32,48,64)
    push!(cases, ("grid2d/$k", ()->nlf_grid2d(k; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(100+k))))
end
for k in (4,6,8,10,12)
    push!(cases, ("grid3d/$k", ()->nlf_grid3d(k; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(200+k))))
end
for (n,p) in ((200,0.05),(500,0.02),(1000,0.01),(2000,0.005))
    push!(cases, ("random/$n", ()->nlf_random(n,p; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(300+n))))
end

open(OUT,"w") do io
    println(io, "instance,n,m,total_s,steps,converged,F,ok")
    @printf("%-12s %-7s %-8s %-10s %-6s %-5s\n","instance","n","m","total_s","steps","conv")
    for (name,mk) in cases
        mfp = un(mk()); n=size(mfp.A,1); m=size(mfp.B,2)
        nlf_maxflow(mfp; tol=1e-6, inner=INNER, method=METHOD)            # warm JIT (small effect)
        t = @elapsed (F,φ,f,info) = nlf_maxflow(mfp; tol=1e-6, inner=INNER, method=METHOD)
        ok = info.converged && all(mfp.low.-1e-5 .<= f .<= mfp.high.+1e-5)
        @printf("%-12s %-7d %-8d %-10.4f %-6d %-5s\n", name,n,m,t,info.steps, ok ? "y" : "n")
        @printf(io, "%s,%d,%d,%.6f,%d,%d,%.6f,%d\n", name,n,m,t,info.steps,
                Int(info.converged), F, Int(ok))
        flush(io)
    end
end
println("WROTE $OUT")
println("DONE_SCALING")
