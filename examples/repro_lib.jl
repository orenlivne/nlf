# repro_lib.jl — reproducibility helpers for the NLF package (github.com/orenlivne/nlf).
#
# A clean, self-contained library (NO `Pkg.activate`) that bundles the working logic from the
# benchmark scripts so the reproducibility notebook / verify script can call it directly:
#
#   * FOLD max-flow loader:  load_edges, lcc_and_farpair, mtx_maxflow
#   * inner-solver engines:  approxchol_engine, diagpcg_engine, pcg_jacobi, INNERS
#   * NO-FOLD congestion:    load_mtx, instance, make_law, far_pair, largest_component,
#                            nlf_bpr, ipopt_bpr, lbfgs_dual, median_, rho_drho!, dual_energy, ...
#
# Logic extracted verbatim (logic-preserving) from:
#   scripts/run_fold_bakeoff.jl, scripts/run_corpus_3solver.jl, scripts/bpr_common.jl
# with the `import Pkg; Pkg.activate(...)` lines stripped (the env is provided by the caller).

using NLF, LAMG, Laplacians, JuMP, SparseArrays, LinearAlgebra, Random, Printf
import Ipopt
import Optim
import NLF: make_problem, nlf_maxflow, NLFLinearEngine
import LAMG: LAMGOptions, setup, solve

# =====================================================================================
# PART A — FOLD max-flow on .mtx graphs  (from scripts/run_fold_bakeoff.jl)
# =====================================================================================

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

# ---- inner-solver engines (the FOLD axis: swap the inner Laplacian solve) ----
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

# =====================================================================================
# PART B — NO-FOLD congestion (BPR) helpers  (from scripts/bpr_common.jl)
# =====================================================================================

# ---------- BPR-family edge law: t(f)=r f + k (f/c)|f/c|^{p-1}; rho=t^{-1}, rho'=1/t'(f) ----------
struct Law; r::Vector{Float64}; k::Vector{Float64}; c::Vector{Float64}; p::Vector{Float64}; end
median_(x)=(y=sort(x); n=length(y); iseven(n) ? 0.5*(y[n÷2]+y[n÷2+1]) : y[(n+1)÷2])
function make_law(c,t0,b,p)
    tpos = filter(>(0), t0); tref = isempty(tpos) ? 1.0 : median_(tpos)
    t0f = max.(t0, 1e-4*tref)              # floor zero/centroid free-flow times (keeps r_e>0)
    cap = 30*median_(c)                    # sanitize "uncapacitated" sentinels (e.g. 999999) to a
    cf  = min.(c, cap)                     # large finite capacity -> bounds the conductance range
    Law(t0f ./ cf, b .* t0f ./ cf, cf, p)
end
@inline function pterm(L,e,F)              # F>=0: returns congestion (t_cong, dt_cong/dF), 0 at F=0
    x = F/L.c[e]
    x <= 0 && return (0.0, 0.0)            # guard: avoids 0*Inf in x^{p-1} for any power p
    xp1 = x^(L.p[e]-1)
    (L.k[e]*x*xp1, L.k[e]*L.p[e]*xp1/L.c[e])
end
tof(L,e,F)  = (tc=pterm(L,e,F)[1]; L.r[e]*F + tc)                         # t_e(F), F>=0
dtdf(L,e,F) = (dc=pterm(L,e,F)[2]; L.r[e] + dc)                          # t_e'(F) >= r_e > 0
function rho_drho!(f,dρ,g,L)                                              # f=rho(g), dρ=rho'(g)
    @inbounds for e in eachindex(g)
        s = sign(g[e]); G = abs(g[e])
        F = G / L.r[e]                                                    # linear guess = upper bracket (t(F)>=G)
        for _ in 1:60
            res = tof(L,e,F) - G
            abs(res) <= 1e-12*(G+1) && break
            F -= res / dtdf(L,e,F); F < 0 && (F = 0.0)
        end
        f[e] = s*F
        dρ[e] = 1.0 / dtdf(L,e,F)
    end
end
Phi(L,e,f) = 0.5*L.r[e]*f^2 + L.k[e]*L.c[e]/(L.p[e]+1)*abs(f/L.c[e])^(L.p[e]+1)
ccost(L,f) = sum(Phi(L,e,f[e]) for e in eachindex(f))
function dual_energy(B,L,d,α,φ)
    g=B'*φ; f=similar(g); dρ=similar(g); rho_drho!(f,dρ,g,L)
    s=0.0; @inbounds for e in eachindex(g); s += g[e]*f[e]-Phi(L,e,f[e]); end
    s - α*dot(d,φ)
end

# ---------- NLF: damped Newton on B rho(B'phi)=alpha d, zero-mean gauge ----------
zm!(x)=(x.-=sum(x)/length(x); x)
function laplacian_clean(J)
    off = J - spdiagm(0=>diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0=>-vec(sum(off, dims=2)))
end
function newton_fixed!(φ, B, L, d, α, keepix; inner=:direct, nmax=40, tol=1e-9, t0=time(), tlim=Inf,
                       refresh=0.25, eta=0.05,
                       H=Ref{Any}(nothing), SC=Ref(1.0), ST=Ref(false), setups=Ref(0), GG=Ref(1.0))
    n=size(B,1); m=size(B,2); f=zeros(m); dρ=zeros(m); steps=0; nr_prev=Inf
    rebuild! = (dρf) -> begin
        SC[]=maximum(dρf); J=B*Diagonal(dρf)*B'
        H[]=setup(laplacian_clean(J./SC[]); options=LAMGOptions()); ST[]=false; setups[]+=1
    end
    mg_step = (r) -> begin
        δ,info = solve(H[], zm!(-Vector(r)./SC[]); options=LAMGOptions(tol=eta, γ_coarse_growth=GG[]))
        get(info, :gamma_escalated, false) && (GG[] = 1.15)   # sticky: later solves start on the grown schedule
        zm!(δ)
    end
    for it in 1:nmax
        steps=it
        g=B'*φ; rho_drho!(f,dρ,g,L); r=B*f .- α.*d; nr=norm(r)
        nr < tol*max(α*norm(d),1.0) && break
        (time()-t0 > tlim) && break               # wall-clock guard (skip pathologically slow graphs)
        dρf=max.(dρ, 1e-11*maximum(dρ))     # tiny Jacobian floor: keeps J SPD without distorting Newton
        local δ
        if inner==:direct                    # pin one node -> SPD reduced Laplacian (Cholesky-robust)
            J=B*Diagonal(dρf)*B'
            δ=zeros(n); δ[keepix]=J[keepix,keepix]\(-Vector(r)[keepix]); zm!(δ)
        else
            (H[]===nothing || nr > refresh*nr_prev) && rebuild!(dρf)
            δ=mg_step(r)
        end
        nr_prev=nr
        τ=1.0; ok=false
        for _ in 1:50
            φt=zm!(φ.+τ.*δ); gt=B'*φt; ft=similar(f); dt=similar(dρ); rho_drho!(ft,dt,gt,L)
            if norm(B*ft.-α.*d) <= nr; copyto!(φ,φt); ok=true; break; end; τ*=0.5
        end
        if !ok && inner!=:direct && ST[]          # stale stall: rebuild + retry once
            rebuild!(dρf); δ=mg_step(r)
            for _ in 1:50
                φt=zm!(φ.+τ.*δ); gt=B'*φt; ft=similar(f); dt=similar(dρ); rho_drho!(ft,dt,gt,L)
                if norm(B*ft.-α.*d) <= nr; copyto!(φ,φt); ok=true; break; end; τ*=0.5
            end
        end
        ok||break
        ST[]=true
    end
    steps
end
function nlf_bpr(B, L, d, α; inner=:direct, loads=(0.25,0.5,1.0), tol=1e-9, tlim=Inf,
                  refresh=0.25, setups=Ref(0))
    n=size(B,1); m=size(B,2)
    deg=vec(sum(abs.(B),dims=2)); pin=argmax(deg); keepix=setdiff(1:n,pin)
    φ=zeros(n); tot=0; t0=time()
    H=Ref{Any}(nothing); SC=Ref(1.0); ST=Ref(false); GG=Ref(1.0)
    for ℓ in loads
        tot += newton_fixed!(φ, B, L, d, ℓ*α, keepix; inner=inner, tol=tol, t0=t0, tlim=tlim,
                             refresh=refresh, H=H, SC=SC, ST=ST, setups=setups, GG=GG)
    end
    f=zeros(m); dρ=zeros(m); g=B'*φ; rho_drho!(f,dρ,g,L)
    zm!(φ), copy(f), tot, norm(B*f.-α.*d)
end

# ---------- Ipopt competitor on the same Beckmann program ----------
function ipopt_bpr(B, L, d, α; tlim=120.0)
    n=size(B,1); m=size(B,2)
    rr=L.r; cc=L.c; pp=L.p; kc=[L.k[e]*L.c[e]/(L.p[e]+1) for e in 1:m]; ph=[(L.p[e]+1)/2 for e in 1:m]
    model=Model(Ipopt.Optimizer); set_silent(model); set_optimizer_attribute(model,"max_cpu_time",tlim)
    @variable(model, f[1:m]); @constraint(model, B*f .== α.*d)
    @NLobjective(model, Min, sum(0.5*rr[e]*f[e]^2 + kc[e]*((f[e]/cc[e])^2)^ph[e] for e in 1:m))
    t=@elapsed optimize!(model); st=termination_status(model)
    ok = st in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    (ok ? value.(f) : fill(NaN,m)), t, string(st)
end

# ---------- restrict to the largest connected component (drop isolated/centroid nodes) ----------
function adjacency(B)
    n=size(B,1); m=size(B,2); adj=[Int[] for _ in 1:n]; rows=rowvals(B)
    for e in 1:m
        nb=rows[nzrange(B,e)]; length(nb)==2 || continue
        push!(adj[nb[1]],nb[2]); push!(adj[nb[2]],nb[1])
    end
    adj
end
function largest_component(B, c, t0, b, p)
    n=size(B,1); adj=adjacency(B); comp=fill(0,n); nc=0
    for s in 1:n
        comp[s]==0 || continue; nc+=1; comp[s]=nc; q=[s]
        while !isempty(q); u=popfirst!(q); for v in adj[u]; comp[v]==0 && (comp[v]=nc; push!(q,v)); end; end
    end
    best=argmax([count(==(k),comp) for k in 1:nc])
    keep=findall(==(best),comp); newid=fill(0,n); for (i,u) in enumerate(keep); newid[u]=i; end
    rows=rowvals(B); m=size(B,2); I=Int[];Jc=Int[];V=Float64[]; cc=Float64[];tt=Float64[];bb=Float64[];pp=Float64[]; ne=0
    for e in 1:m
        nb=rows[nzrange(B,e)]; length(nb)==2 || continue
        (comp[nb[1]]==best && comp[nb[2]]==best) || continue
        ne+=1; push!(I,newid[nb[1]]);push!(Jc,ne);push!(V,-1.0); push!(I,newid[nb[2]]);push!(Jc,ne);push!(V,+1.0)
        push!(cc,c[e]);push!(tt,t0[e]);push!(bb,b[e]);push!(pp,p[e])
    end
    sparse(I,Jc,V,length(keep),ne), cc,tt,bb,pp
end

# ---------- single-commodity demand: BFS-far s,t ----------
function far_pair(B)
    n=size(B,1); m=size(B,2)
    adj=[Int[] for _ in 1:n]
    rows=rowvals(B)
    for e in 1:m
        nb=rows[nzrange(B,e)]; length(nb)==2 || continue
        push!(adj[nb[1]],nb[2]); push!(adj[nb[2]],nb[1])
    end
    bfs(s)=(dist=fill(-1,n); dist[s]=0; q=[s]; while !isempty(q); u=popfirst!(q); for v in adj[u]; dist[v]<0 && (dist[v]=dist[u]+1; push!(q,v)); end; end; dist)
    d1=bfs(1); s=argmax(d1); d2=bfs(s); t=argmax(d2)
    s,t
end

# =====================================================================================
# PART C — NO-FOLD loaders / competitors  (from scripts/run_corpus_3solver.jl)
# =====================================================================================

# ---- read .mtx -> signed node-arc incidence B (n x m), largest component handled by `instance` ----
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
# Build a BPR instance from an incidence B0: random caps/free-flow times, restrict to LCC.
instance(B0; seed=1)=(rng=MersenneTwister(seed); m0=size(B0,2);
    largest_component(B0,[0.5+2.5*rand(rng) for _ in 1:m0],[1.0+4.0*rand(rng) for _ in 1:m0],fill(0.15,m0),fill(4.0,m0)))
# L-BFGS-on-dual competitor (first-order, matrix-free).
function lbfgs_dual(B,L,d,α; tol=1e-9,tlim=60.0)
    n=size(B,1);m=size(B,2);f=zeros(m);dρ=zeros(m);bn=max(α*norm(d),1.0)
    E(φ)=dual_energy(B,L,d,α,φ);g!(G,φ)=(g=B'*φ;rho_drho!(f,dρ,g,L);r=B*f.-α.*d;r.-=sum(r)/n;copyto!(G,r);G)
    t=@elapsed res=Optim.optimize(E,g!,zeros(n),Optim.LBFGS(m=20),Optim.Options(g_tol=tol*bn,iterations=10_000_000,time_limit=tlim))
    φ=Optim.minimizer(res);g=B'*φ;rho_drho!(f,dρ,g,L);(it=Optim.iterations(res),t=t,r=norm(B*f.-α.*d)/bn,c=Optim.converged(res))
end

# data/ directory of the NLF repo (this file lives in examples/)
const REPRO_DATA = abspath(joinpath(@__DIR__, "..", "data"))
data_path(fn) = joinpath(REPRO_DATA, fn)
