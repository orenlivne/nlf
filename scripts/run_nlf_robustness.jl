# ROBUSTNESS across graph classes: NLF = Newton + LAMG+ congestion solve on the SAME diverse graphs
# LAMG+ was validated on (structured, FEM/anisotropic, web, social, road, citation; SuiteSparse .mtx in
# data/). Each graph is turned into a single-commodity BPR congestion instance (random caps/free-flow
# times, a far s-t demand). We report the Newton-step count (should be flat across classes/sizes) and
# the per-edge wall-clock (should be flat = O(m)), demonstrating NLF inherits LAMG+'s graph-class
# robustness on the full test set, not just road networks.
include(joinpath(@__DIR__, "bpr_common.jl"))

# ---- MatrixMarket loader: build the undirected graph incidence B from a .mtx sparsity pattern ----
function load_mtx(path)
    nr=nc=0; started=false; seen=Set{Tuple{Int,Int}}(); ne=0
    I=Int[]; Jc=Int[]; V=Float64[]
    for ln in eachline(path)
        (isempty(ln) || ln[1]=='%') && continue
        t=split(ln)
        if !started
            nr=parse(Int,t[1]); nc=parse(Int,t[2]); started=true; continue
        end
        length(t)<2 && continue
        i=parse(Int,t[1]); j=parse(Int,t[2]); i==j && continue
        key = i<j ? (i,j) : (j,i)
        (key in seen) && continue; push!(seen,key)
        ne+=1; push!(I,key[1]);push!(Jc,ne);push!(V,-1.0); push!(I,key[2]);push!(Jc,ne);push!(V,+1.0)
    end
    n=max(nr,nc)
    sparse(I,Jc,V,n,ne)
end
# turn a bare graph into a BPR congestion instance (random caps in [0.5,3], free-flow times in [1,5])
function graph_instance(B0; seed=1)
    rng=MersenneTwister(seed); m0=size(B0,2)
    c0=[0.5+2.5*rand(rng) for _ in 1:m0]; t0=[1.0+4.0*rand(rng) for _ in 1:m0]
    b=fill(0.15,m0); p=fill(4.0,m0)
    B,c,tt,bb,pp = largest_component(B0,c0,t0,b,p)
    B,c,tt,bb,pp
end

DATA = joinpath(@__DIR__, "..", "data")
# (class, label, filename) -- representatives across the LAMG+ classes, spanning sizes
graphs = [
 ("structured","grid2-monien","AG-Monien__grid2.mtx"),
 ("FEM/aniso", "bodyy5",       "Pothen__bodyy5.mtx"),
 ("FEM",       "poisson3Da",   "FEMLAB__poisson3Da.mtx"),
 ("web",       "wb-cs-stanford","Gleich__wb-cs-stanford.mtx"),
 ("road",      "ak2010",       "DIMACS10__ak2010.mtx"),
 ("social",    "enron",        "LAW__enron.mtx"),
 ("social",    "amazon0302",   "SNAP__amazon0302.mtx"),
 ("web",       "Stanford",     "Kamvar__Stanford.mtx"),
 ("citation",  "citationCiteseer","DIMACS10__citationCiteseer.mtx"),
 ("social",    "flickr",       "Gleich__flickr.mtx"),
]

OUT = length(ARGS)>=1 ? ARGS[1] : "/tmp/nlf_robustness.csv"
open(OUT,"w") do io
    println(io,"class,graph,n,m,nlf_s,nlf_steps,per_edge_us,match_ipopt")
    @printf("%-10s %-18s %-8s %-9s %-9s %-5s %-9s %-6s\n","class","graph","n","m","NLF_s","it","us/edge","match")
    for (cls,lab,fn) in graphs
        path=joinpath(DATA,fn)
        isfile(path) || (@printf("%-10s %-18s  (missing)\n",cls,lab); continue)
        try
            B0=load_mtx(path); B,c,t0,b,p=graph_instance(B0)
            n=size(B,1); m=size(B,2); n<3 && continue
            L=make_law(c,t0,b,p); s,t=far_pair(B); d=zeros(n); d[s]=-1.0; d[t]=+1.0; α=0.3*median_(c)
            nlf_bpr(B,L,d,α; inner=:multigrid)                          # warm
            tf=@elapsed ((φ,fF,it,res)=nlf_bpr(B,L,d,α; inner=:multigrid))
            mtch="-"
            if m <= 60000                                               # validate vs Ipopt on the small ones
                fI,_,st=ipopt_bpr(B,L,d,α; tlim=120.0)
                if !any(isnan,fI); mtch = abs(ccost(L,fF)-ccost(L,fI))/max(ccost(L,fI),1e-9)<1e-3 ? "yes" : "NO"; end
            end
            @printf("%-10s %-18s %-8d %-9d %-9.3f %-5d %-9.2f %-6s\n",cls,lab,n,m,tf,it,tf/m*1e6,mtch)
            @printf(io,"%s,%s,%d,%d,%.6f,%d,%.4f,%s\n",cls,lab,n,m,tf,it,tf/m*1e6,mtch); flush(io)
        catch e
            @printf("%-10s %-18s  ERROR %s\n",cls,lab,sprint(showerror,e)[1:min(end,60)])
        end
    end
end
println("WROTE $OUT"); println("DONE_ROBUSTNESS")
