# FOLD inner-solver bake-off (for completeness, the OTHER axis): fix NLF's outer method (chord-Newton
# + pseudo-arclength continuation to the max-flow fold F*) and swap the INNER linear Laplacian solver.
# Near the fold J=B diag(rho')B' is near-singular (kappa->inf), so this stresses the inner kernel's
# robustness. Competitors: LAMG+ (multigrid) | approxChol (Laplacians.jl) | direct Cholesky | diagonal-PCG.
# Question: which inner solver lets the continuation reach F* robustly and fast? (Not to crown LAMG+ --
# for completeness: do all reach the same F*; do the naive ones stall on the near-singular systems.)
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LinearAlgebra, SparseArrays, Printf, Random
import NLF, Laplacians
import NLF: make_problem, nlf_maxflow, NLFLinearEngine

# ---- read .mtx -> undirected edge list (head,tail), largest connected component ----
function load_edges(path)
    started=false; seen=Set{Tuple{Int,Int}}(); E=Tuple{Int,Int}[]; nr=nc=0
    for ln in eachline(path)
        (isempty(ln)||ln[1]=='%') && continue; t=split(ln)
        if !started; nr=parse(Int,t[1]); nc=parse(Int,t[2]); started=true; continue; end
        length(t)<2 && continue
        i=parse(Int,t[1]); j=parse(Int,t[2]); i==j && continue
        k=i<j ? (i,j) : (j,i); (k in seen)&&continue; push!(seen,k); push!(E,k)
    end
    max(nr,nc), E
end
function lcc_and_farpair(n, E)
    adj=[Int[] for _ in 1:n]; for (a,b) in E; push!(adj[a],b); push!(adj[b],a); end
    comp=fill(0,n); nc=0
    for s in 1:n; comp[s]==0||continue; nc+=1; comp[s]=nc; q=[s]
        while !isempty(q); u=popfirst!(q); for v in adj[u]; comp[v]==0&&(comp[v]=nc;push!(q,v)); end; end; end
    best=argmax([count(==(c),comp) for c in 1:nc]); keep=findall(==(best),comp)
    nid=fill(0,n); for (i,u) in enumerate(keep); nid[u]=i; end
    E2=[(nid[a],nid[b]) for (a,b) in E if comp[a]==best&&comp[b]==best]
    nk=length(keep); adj2=[Int[] for _ in 1:nk]; for (a,b) in E2; push!(adj2[a],b); push!(adj2[b],a); end
    bfs(s)=(dist=fill(-1,nk);dist[s]=0;q=[s];while !isempty(q);u=popfirst!(q);for v in adj2[u];dist[v]<0&&(dist[v]=dist[u]+1;push!(q,v));end;end;dist)
    s=argmax(bfs(1)); t=argmax(bfs(s)); nk, E2, s, t
end
function mtx_maxflow(path; seed=1)
    n,E = load_edges(path); nk,E2,s,t = lcc_and_farpair(n,E); rng=MersenneTwister(seed)
    cp=[0.5+2.5*rand(rng) for _ in 1:length(E2)]   # forward caps; symmetric reverse
    make_problem(nk, E2, cp, cp, s, t; name=basename(path))
end

# ---- inner-solver engines ----
adj_from_lap(J)=(Jo=J-spdiagm(0=>diag(J)); W=-Jo; for r in 1:nnz(W); W.nzval[r]<0&&(W.nzval[r]=0.0); end; dropzeros!(W))
approxchol_engine = NLFLinearEngine(J->Laplacians.approxchol_lap(adj_from_lap(sparse(J));verbose=false),
                                     (F,_,b)->F(Vector(b)), "approxChol")
function pcg_jacobi(J,b;tol=1e-8,maxit=3000)
    D=diag(J); n=length(b); x=zeros(n); r=Vector(b); z=r./D; p=copy(z); rz=dot(r,z); bn=max(norm(b),1e-30)
    for _ in 1:maxit
        Jp=J*p; den=dot(p,Jp); den==0 && break; α=rz/den; @. x+=α*p; @. r-=α*Jp
        norm(r)<=tol*bn && break
        z=r./D; rz2=dot(r,z); β=rz2/rz; rz=rz2; @. p=z+β*p
    end
    x
end
diagpcg_engine = NLFLinearEngine(J->sparse(J), (Js,_,b)->pcg_jacobi(Js,b), "diagPCG")
INNERS = [("LAMG+",:multigrid),("approxChol",approxchol_engine),("direct",:direct),("diagPCG",diagpcg_engine)]

DATA=get(ENV,"GRAPH_DATA",joinpath(@__DIR__,"..","data"))
DIRECT_NMAX = 150_000   # Cholesky OOMs on larger poorly-separable graphs
FOLD_NMAX   = 300_000   # cap graph size for the (heavy, 4-solver) fold continuation
# Comprehensive span: curated structured+multiscale graphs, PLUS a size-spread sample across the corpus.
CURATED=["AG-Monien__grid2.mtx","AG-Monien__airfoil1.mtx","Pothen__bodyy5.mtx","FEMLAB__poisson3Da.mtx",
   "Boeing__bcsstk38.mtx","AMD__G2_circuit.mtx","Bomhof__circuit_1.mtx","Bomhof__circuit_3.mtx",
   "Bomhof__circuit_4.mtx","Hamm__scircuit.mtx","LAW__enron.mtx","Gleich__wb-cs-stanford.mtx",
   "SNAP__p2p-Gnutella31.mtx","DIMACS10__caidaRouterLevel.mtx","DIMACS10__citationCiteseer.mtx"]
let szf=joinpath(DATA,"mtx_sizes.csv")
    sizes=[(split(l,',')[1],parse(Int,split(l,',')[2])) for l in Iterators.drop(eachline(szf),1) if length(split(l,','))>=4]
    filter!(x-> x[2]<=FOLD_NMAX && isfile(joinpath(DATA,x[1])), sizes); sort!(sizes,by=x->x[2])
    samp=[sizes[round(Int,k)][1] for k in range(1,length(sizes),length=min(28,length(sizes)))]
    global G=unique(vcat([g for g in CURATED if isfile(joinpath(DATA,g))], samp))
end
OUT="/tmp/nlf_fold_bakeoff.csv"
open(OUT,"w") do io; println(io,"graph,n,m,inner,Fstar,steps,time_s,converged"); end
# warm
let mfp=mtx_maxflow(joinpath(DATA,"AG-Monien__airfoil1.mtx")); for (_,ic) in INNERS; try; nlf_maxflow(mfp;inner=ic,max_steps=30); catch; end; end; end

@printf("FOLD inner-solver bake-off (max-flow continuation to F*). Same F* across solvers = consistent.\n")
for fn in G
    path=joinpath(DATA,fn); isfile(path)||(println("(skip $fn)");continue)
    local mfp
    try; mfp=mtx_maxflow(path); catch e; @printf("%-26s LOAD/BUILD ERR\n",first(fn,26)); continue; end
    n=size(mfp.B,1); m=size(mfp.B,2)
    @printf("%-26s n=%-7d m=%-8d\n", first(fn,26), n, m)
    for (nm,ic) in INNERS
        if nm=="direct" && n>DIRECT_NMAX
            @printf("    %-11s too-large (Cholesky OOM)\n",nm)
            open(OUT,"a") do io; @printf(io,"%s,%d,%d,%s,NaN,-2,NaN,0\n",fn,n,m,nm); end; continue
        end
        res = try
            t=@elapsed ((Fs,φ,f,info)=nlf_maxflow(mfp; inner=ic, max_steps=120, tol=1e-7))
            (Fs=Fs, st=info.steps, t=t, c=info.converged)
        catch e; (Fs=NaN, st=-1, t=NaN, c=false); end
        @printf("    %-11s F*=%-10.4f steps=%-4d %8.2fs %s\n", nm, res.Fs, res.st, res.t, res.c ? "ok" : "STALL/ERR")
        open(OUT,"a") do io; @printf(io,"%s,%d,%d,%s,%.6f,%d,%.4f,%d\n", fn,n,m,nm,res.Fs,res.st,res.t,Int(res.c)); end
        flush(stdout)
    end
end
println("DONE_FOLD_BAKEOFF -> $OUT")
