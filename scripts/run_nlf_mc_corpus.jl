# FULL-CORPUS multicommodity robustness: K=4 commodities (k-center-separated demand dipoles) on
# every graph of the SuiteSparse corpus used for the single-commodity sweep (same instance protocol:
# random caps/times seed=1, largest component, α = 0.3·median(c) per commodity, nnz ≤ 25M, sorted
# small→large). Incremental CSV (resumable). Shardable for parallel workers:
#   julia run_nlf_mc_corpus.jl OUT.csv [shard_index shard_count]
# (shard i of S takes graphs with sorted-rank % S == i-1; each shard writes its own CSV.)
include(joinpath(@__DIR__, "mc_common.jl"))

const KCOMM   = 4
const INNER   = Symbol(get(ENV, "MC_INNER", "mg"))
const HWEIGHT = Symbol(get(ENV, "MC_HWEIGHT", "geo"))
const TLIM    = parse(Float64, get(ENV, "MC_TLIM", "1800.0"))

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

DATA  = joinpath(@__DIR__, "..", "data")
OUT   = length(ARGS)>=1 ? ARGS[1] : "/tmp/nlf_mc_corpus.csv"
SHARD = length(ARGS)>=3 ? parse(Int,ARGS[2]) : 1
NSH   = length(ARGS)>=3 ? parse(Int,ARGS[3]) : 1
NNZ_CAP = 25_000_000

rows = Tuple{String,Int,Int}[]
for (k,ln) in enumerate(eachline(joinpath(DATA,"mtx_sizes.csv")))
    k==1 && continue
    t=split(ln,','); length(t)<4 && continue
    nm=t[1]; n=parse(Int,t[2]); nnz=parse(Int,t[4])
    (n>=10 && 0<nnz<=NNZ_CAP) && push!(rows,(nm,n,nnz))
end
sort!(rows, by=x->x[3])
println("MC corpus: $(length(rows)) graphs, K=$(KCOMM), inner=$(INNER), hweight=$(HWEIGHT); shard $SHARD/$NSH -> $OUT")

done = Set{String}()
if isfile(OUT)
    for ln in eachline(OUT); t=split(ln,','); !isempty(t) && push!(done, t[1]); end
else
    open(OUT,"w") do io; println(io,"graph,n,m,K,nlf_s,nlf_steps,setups,per_edge_us,residual,converged"); end
end

nok=0; nfail=0
for (idx,(nm,n0,nnz)) in enumerate(rows)
    (idx-1) % NSH == SHARD-1 || continue
    nm in done && continue
    path=joinpath(DATA,nm); isfile(path) || continue
    try
        B,c,t0,b,p = graph_instance(load_mtx(path))
        n=size(B,1); m=size(B,2); (n<3 || m<2) && continue
        L=make_law(c,t0,b,p); D=k_dipoles(B,KCOMM); α=0.3*median_(c)
        su=Ref(0)
        tf=@elapsed ((Φ,F,it,res)=nlf_mc(B,L,D,α; inner=INNER, hweight=HWEIGHT, tlim=TLIM, setups=su))
        conv = res < 1e-6*max(α*norm(D),1.0) ? 1 : 0; conv==1 ? (global nok+=1) : (global nfail+=1)
        open(OUT,"a") do io
            @printf(io,"%s,%d,%d,%d,%.6f,%d,%d,%.4f,%.3e,%d\n",nm,n,m,KCOMM,tf,it,su[],tf/m*1e6,res,conv)
        end
        idx % 25 == 0 && @printf("[%4d/%4d] %-34s m=%-9d it=%-3d su=%d %.2fus/edge %s  (ok=%d fail=%d)\n",
                                 idx,length(rows),nm[1:min(end,34)],m,it,su[],tf/m*1e6,conv==1 ? "" : "NOCONV",nok,nfail)
    catch e
        global nfail+=1
        open(OUT,"a") do io; @printf(io,"%s,%d,0,%d,0,0,0,0,0,-1\n",nm,n0,KCOMM); end
    end
end
@printf("DONE_MC_CORPUS shard=%d/%d ok=%d fail/skip=%d\n", SHARD, NSH, nok, nfail)
