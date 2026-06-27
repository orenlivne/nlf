# FULL-CORPUS 3-solver comparison (all 2003 SuiteSparse graphs, nonlinear BPR CONGESTION, NO FOLD).
# Same undirected convex Beckmann program for all three paradigms:
#   NLF (multigrid-Newton)  |  Ipopt (interior-point / sparse-direct KKT)  |  L-BFGS-on-dual (first-order)
# Per-graph RELATIVE cap: each competitor is capped at 5x the measured NLF wall-clock (a competitor
# slower than that "loses"). NLF runs on every graph (O(m)); competitors are size-guarded to avoid
# hard OOM crashes (recorded as too-large, which is itself the scaling result). Writes live CSV.
include(joinpath(@__DIR__, "bpr_common.jl"))
import Optim; using SparseArrays, LinearAlgebra, Random, Printf

IPOPT_NMAX = 300_000        # sparse-direct KKT OOMs on larger poorly-separable graphs
LBFGS_NMAX = 2_000_000      # matrix-free but the history + B get large
LOAD_NNZMAX = 25_000_000    # cap loader memory (the 2003 corpus is <= 1.8e7 edges)

function load_mtx(path)
    started=false; seen=Set{Tuple{Int,Int}}(); ne=0; I=Int[]; Jc=Int[]; V=Float64[]; nr=nc=0
    for ln in eachline(path)
        (isempty(ln) || ln[1]=='%') && continue
        t=split(ln)
        if !started; nr=parse(Int,t[1]); nc=parse(Int,t[2]); started=true; continue; end
        length(t)<2 && continue
        i=parse(Int,t[1]); j=parse(Int,t[2]); i==j && continue
        key=i<j ? (i,j) : (j,i); (key in seen) && continue; push!(seen,key)
        ne+=1; push!(I,key[1]);push!(Jc,ne);push!(V,-1.0); push!(I,key[2]);push!(Jc,ne);push!(V,+1.0)
    end
    sparse(I,Jc,V,max(nr,nc),ne)
end
instance(B0; seed=1)=(rng=MersenneTwister(seed); m0=size(B0,2);
    largest_component(B0,[0.5+2.5*rand(rng) for _ in 1:m0],[1.0+4.0*rand(rng) for _ in 1:m0],fill(0.15,m0),fill(4.0,m0)))
function lbfgs_dual(B,L,d,α; tol=1e-9,tlim=60.0)
    n=size(B,1);m=size(B,2);f=zeros(m);dρ=zeros(m);bn=max(α*norm(d),1.0)
    E(φ)=dual_energy(B,L,d,α,φ);g!(G,φ)=(g=B'*φ;rho_drho!(f,dρ,g,L);r=B*f.-α.*d;r.-=sum(r)/n;copyto!(G,r);G)
    t=@elapsed res=Optim.optimize(E,g!,zeros(n),Optim.LBFGS(m=20),Optim.Options(g_tol=tol*bn,iterations=10_000_000,time_limit=tlim))
    φ=Optim.minimizer(res);g=B'*φ;rho_drho!(f,dρ,g,L);(it=Optim.iterations(res),t=t,r=norm(B*f.-α.*d)/bn,c=Optim.converged(res))
end

DATA=joinpath(@__DIR__,"..","data"); SIZES=joinpath(DATA,"mtx_sizes.csv")
rows=[(split(l,',')[1], parse(Int,split(l,',')[4])) for l in Iterators.drop(eachline(SIZES),1) if length(split(l,','))>=4]
sort!(rows, by=x->x[2])                       # small -> large, so the bulk finishes first
OUT="/tmp/nlf_corpus_3solver.csv"
open(OUT,"w") do io; println(io,"graph,n,m,nlf_t,nlf_steps,nlf_ok,cap5x,ipopt_t,ipopt_ok,ipopt_note,lbfgs_t,lbfgs_it,lbfgs_ok,lbfgs_note"); end
# JIT warm
let B0=load_mtx(joinpath(DATA,"AG-Monien__airfoil1.mtx")); B,c,t0,b,p=instance(B0); L=make_law(c,t0,b,p);
    s,t=far_pair(B); d=zeros(size(B,1)); d[s]=-1;d[t]=1; nlf_bpr(B,L,d,0.3median_(c);inner=:multigrid); lbfgs_dual(B,L,d,0.3median_(c);tlim=2.0); end

done=0; nfail=0
for (fn,nnz) in rows
    path=joinpath(DATA,fn); isfile(path) || continue
    nnz>LOAD_NNZMAX && continue
    try
        B0=load_mtx(path); B,c,t0,b,p=instance(B0); n=size(B,1); m=size(B,2)
        (n<10 || m<5) && continue
        L=make_law(c,t0,b,p); s,t=far_pair(B); d=zeros(n); d[s]=-1.0;d[t]=1.0; α=0.3*median_(c)
        GC.gc()
        φ,fF,st,r=nlf_bpr(B,L,d,α;inner=:multigrid,tlim=600.0); tNLF=@elapsed nlf_bpr(B,L,d,α;inner=:multigrid,tlim=600.0)
        nlf_ok = r < 1e-6; cap = max(5*tNLF, 1.0)
        # Ipopt (size-guarded; capped at 5x NLF)
        ipt=NaN; ipok=false; ipnote="ok"
        if n<=IPOPT_NMAX
            try; fI,tI,sti=ipopt_bpr(B,L,d,α;tlim=cap); ipt=tI; ipok=!any(isnan,fI); ipnote = ipok ? "ok" : "timeout/fail"
            catch; ipnote="error"; end
        else; ipnote="too-large"; end
        # L-BFGS (size-guarded; capped at 5x NLF)
        lbt=NaN; lbit=-1; lbok=false; lbnote="ok"
        if n<=LBFGS_NMAX
            try; lb=lbfgs_dual(B,L,d,α;tlim=cap); lbt=lb.t; lbit=lb.it; lbok=lb.c; lbnote = lbok ? "ok" : "timeout(>5xNLF)"
            catch; lbnote="error"; end
        else; lbnote="too-large"; end
        open(OUT,"a") do io; @printf(io,"%s,%d,%d,%.4f,%d,%d,%.4f,%.4f,%d,%s,%.4f,%d,%d,%s\n",
            fn,n,m,tNLF,st,Int(nlf_ok),cap, ipt,Int(ipok),ipnote, lbt,lbit,Int(lbok),lbnote); end
        global done+=1
        done % 25 == 0 && (@printf("[%d graphs done] last: %s n=%d m=%d NLF %.2fs/%dst ipopt=%s lbfgs=%s\n",
            done,first(fn,28),n,m,tNLF,st,ipnote,lbnote); flush(stdout))
    catch e
        global nfail+=1
        open(OUT,"a") do io; println(io,"$fn,0,0,NaN,0,0,NaN,NaN,0,loaderr,NaN,-1,0,loaderr"); end
    end
end
@printf("DONE: %d graphs run, %d load failures. -> %s\n", done, nfail, OUT)
