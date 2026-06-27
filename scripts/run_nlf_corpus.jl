# FULL-CORPUS robustness: run NLF = Newton + LAMG+ congestion on the entire real-world SuiteSparse
# graph corpus LAMG+ was benchmarked on (data/*.mtx, indexed by data/mtx_sizes.csv), to show NLF
# inherits LAMG+'s O(m) graph-class robustness over ~10^3 graphs, not a hand-picked few. Streamlined:
# no warm-up (JIT amortises over the sweep), no Ipopt; record per-graph Newton count, per-edge time,
# convergence. Incremental CSV (resumable). Sorted small->large; capped at nnz<=25M.
include(joinpath(@__DIR__, "bpr_common.jl"))

function load_mtx(path)
    nr=nc=0; started=false; seen=Set{Tuple{Int,Int}}(); ne=0
    I=Int[]; Jc=Int[]; V=Float64[]
    for ln in eachline(path)
        (isempty(ln) || ln[1]=='%') && continue
        t=split(ln)
        if !started; nr=parse(Int,t[1]); nc=parse(Int,t[2]); started=true; continue; end
        length(t)<2 && continue
        i=parse(Int,t[1]); j=parse(Int,t[2]); i==j && continue
        key = i<j ? (i,j) : (j,i); (key in seen) && continue; push!(seen,key)
        ne+=1; push!(I,key[1]);push!(Jc,ne);push!(V,-1.0); push!(I,key[2]);push!(Jc,ne);push!(V,+1.0)
    end
    sparse(I,Jc,V,max(nr,nc),ne)
end
function graph_instance(B0; seed=1)
    rng=MersenneTwister(seed); m0=size(B0,2)
    c0=[0.5+2.5*rand(rng) for _ in 1:m0]; t0=[1.0+4.0*rand(rng) for _ in 1:m0]
    largest_component(B0,c0,t0,fill(0.15,m0),fill(4.0,m0))
end

DATA = joinpath(@__DIR__, "..", "data")
OUT  = length(ARGS)>=1 ? ARGS[1] : "/tmp/nlf_corpus.csv"
NNZ_CAP = 25_000_000

# corpus list from mtx_sizes.csv (name, n, nnz), sorted by nnz ascending, capped
rows = Tuple{String,Int,Int}[]
for (k,ln) in enumerate(eachline(joinpath(DATA,"mtx_sizes.csv")))
    k==1 && continue
    t=split(ln,','); length(t)<4 && continue
    nm=t[1]; n=parse(Int,t[2]); nnz=parse(Int,t[4])
    (n>=10 && 0<nnz<=NNZ_CAP) && push!(rows,(nm,n,nnz))
end
sort!(rows, by=x->x[3])
println("corpus: $(length(rows)) graphs (nnz<=$(NNZ_CAP)); writing $OUT")

done = Set{String}()
if isfile(OUT)
    for ln in eachline(OUT); t=split(ln,','); !isempty(t) && push!(done, t[1]); end
else
    open(OUT,"w") do io; println(io,"graph,n,m,nlf_s,nlf_steps,per_edge_us,residual,converged"); end
end

nok=0; nfail=0
for (idx,(nm,n0,nnz)) in enumerate(rows)
    nm in done && continue
    path=joinpath(DATA,nm); isfile(path) || continue
    try
        B,c,t0,b,p = graph_instance(load_mtx(path))
        n=size(B,1); m=size(B,2); (n<3 || m<2) && continue
        L=make_law(c,t0,b,p); s,t=far_pair(B); d=zeros(n); d[s]=-1.0; d[t]=+1.0; α=0.3*median_(c)
        tf=@elapsed ((φ,fF,it,res)=nlf_bpr(B,L,d,α; inner=:multigrid, tlim=90.0))
        conv = res < 1e-6*max(α*norm(d),1.0) ? 1 : 0; conv==1 ? (global nok+=1) : (global nfail+=1)
        open(OUT,"a") do io
            @printf(io,"%s,%d,%d,%.6f,%d,%.4f,%.3e,%d\n",nm,n,m,tf,it,tf/m*1e6,res,conv)
        end
        idx % 25 == 0 && @printf("[%4d/%4d] %-34s m=%-9d it=%-3d %.2fus/edge %s  (ok=%d fail=%d)\n",
                                 idx,length(rows),nm[1:min(end,34)],m,it,tf/m*1e6,conv==1 ? "" : "NOCONV",nok,nfail)
    catch e
        global nfail+=1
        open(OUT,"a") do io; @printf(io,"%s,%d,0,0,0,0,0,-1\n",nm,n0); end
    end
end
@printf("DONE_CORPUS  ok=%d fail/skip=%d total=%d\n", nok, nfail, length(rows))
