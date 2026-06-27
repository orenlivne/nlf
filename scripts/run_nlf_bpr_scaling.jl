# SCALING: NLF (BPR congestion, LAMG+ multigrid inner -> O(m)) vs Ipopt (interior-point, sparse-
# direct KKT). Two families: (i) REAL road networks (Transportation Networks for Research) -- planar,
# good separators, real BPR data; (ii) SYNTHETIC Erdos-Renyi graphs with BPR params -- poorly
# separable, where the IPM's direct core blows up. Same convex Beckmann program; costs must match.
include(joinpath(@__DIR__, "bpr_common.jl"))
import LAMG: nlf_random

# synthetic poorly-separable graph carrying BPR parameters (random caps, free-flow times ~U[1,5])
function synth_bpr(n, p; rng)
    mfp = nlf_random(n, p; cap_lo=0.5, cap_hi=3.0, rng=rng)
    B = mfp.B; m = size(B,2)
    c  = [max(mfp.high[e], -mfp.low[e]) for e in 1:m]
    t0 = [1.0 + 4.0*rand(rng) for _ in 1:m]
    B, c, t0, fill(0.15,m), fill(4.0,m), Vector(mfp.d)
end

OUT = length(ARGS)>=1 ? ARGS[1] : "/tmp/nlf_bpr_scaling.csv"
dir = "/tmp/tn_probe"
real_nets = ["SiouxFalls","Anaheim","Chicago-Sketch","Austin","chicago-regional","Sydney"]
synth = [(500,0.024),(1000,0.012),(2000,0.006),(5000,0.0024),(10000,0.0012),(20000,0.0006),(40000,0.0003)]

function runcase(io, fam, name, B, c, t0, b, p, d, α)
    n=size(B,1); m=size(B,2); L=make_law(c,t0,b,p)
    nlf_bpr(B,L,d,α; inner=:multigrid)                          # warm (JIT + caches)
    tF=@elapsed ((φ,fF,it,res)=nlf_bpr(B,L,d,α; inner=:multigrid))
    cF=ccost(L,fF)
    fI,tI,st = ipopt_bpr(B,L,d,α; tlim=120.0)
    okI=!any(isnan,fI); cI = okI ? ccost(L,fI) : NaN
    match = okI ? (abs(cF-cI)/max(abs(cI),1e-9)<1e-3) : false
    @printf("%-8s %-16s %-7d %-8d NLF %7.3fs/%-3d  Ipopt %-9s  %s\n",
            fam,name,n,m,tF,it, okI ? @sprintf("%.3fs",tI) : "FAIL", okI ? (match ? "match" : "MISMATCH") : st)
    @printf(io,"%s,%s,%d,%d,%.6f,%d,%.6f,%.6f,%.6f,%s,%d\n",
            fam,name,n,m,tF,it,cF, okI ? tI : NaN, cI, st, Int(match)); flush(io)
end

open(OUT,"w") do io
    println(io,"family,instance,n,m,nlf_s,nlf_steps,nlf_cost,ipopt_s,ipopt_cost,ipopt_status,match")
    for name in real_nets
        B0,c0,t00,b0,p0,_ = parse_tntp(dir,name)
        B,c,t0,b,p = largest_component(B0,c0,t00,b0,p0)
        s,t=far_pair(B); d=zeros(size(B,1)); d[s]=-1.0; d[t]=+1.0
        runcase(io,"road",name,B,c,t0,b,p,d, 0.3*median_(c))
    end
    for (nn,pp) in synth
        B,c,t0,b,p,d = synth_bpr(nn,pp; rng=MersenneTwister(700+nn))
        runcase(io,"random","random/$nn",B,c,t0,b,p,d, 0.3*median_(c))
    end
end
println("WROTE $OUT"); println("DONE_BPR_SCALING")
