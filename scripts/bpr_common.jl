# Shared code for NLF congestion-flow on real road networks (BPR): TNTP parser, BPR edge law,
# load-continuation damped Newton (NLF), Ipopt competitor, component/demand helpers.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
using JuMP; import Ipopt
import LAMG: LAMGOptions, setup, solve

# ---------- TNTP parser: undirected graph + per-edge (c, t0, b, p) ----------
function parse_tntp(dir, name)
    nd = joinpath(dir, name)
    netf = first(filter(f->endswith(lowercase(f),"_net.tntp"), readdir(nd, join=true)))
    lines = readlines(netf)
    # find data start (after <END OF METADATA>)
    istart = findfirst(l->occursin("END OF METADATA", l), lines)
    nnodes = 0
    for l in lines
        occursin("NUMBER OF NODES", l) && (nnodes = parse(Int, strip(replace(l, r"[^0-9]"=>" ")|>x->split(x)[end])))
    end
    # collect undirected edges keyed by sorted (i,j)
    E = Dict{Tuple{Int,Int},NTuple{4,Float64}}()  # (i,j)->(cap,t0,b,p)
    for l in lines[istart+1:end]
        s = strip(l)
        (isempty(s) || startswith(s, "~")) && continue
        s = replace(s, ";"=>"")
        t = split(s)
        length(t) < 7 && continue
        i = parse(Int, t[1]); j = parse(Int, t[2])
        cap = parse(Float64, t[3]); t0 = parse(Float64, t[5])
        b = parse(Float64, t[6]); p = parse(Float64, t[7])
        nnodes = max(nnodes, i, j)
        key = i < j ? (i,j) : (j,i)
        if haskey(E, key)
            c0,t00,b0,p0 = E[key]; E[key] = (max(c0,cap), min(t00,t0), b0, p0)  # merge opposing arcs
        else
            E[key] = (cap, t0, b, p)
        end
    end
    keys_e = collect(keys(E))
    m = length(keys_e); n = nnodes
    I = Int[]; Jc = Int[]; V = Float64[]
    c = zeros(m); t0 = zeros(m); bb = zeros(m); pp = zeros(m)
    for (e,(i,j)) in enumerate(keys_e)
        cap,tt,b,p = E[(i,j)]
        c[e]=cap; t0[e]=tt; bb[e]=b; pp[e]=p
        push!(I,i); push!(Jc,e); push!(V,-1.0)
        push!(I,j); push!(Jc,e); push!(V,+1.0)
    end
    B = sparse(I, Jc, V, n, m)
    return B, c, t0, bb, pp, keys_e
end

# ---------- BPR-family edge law: t(f)=r f + k (f/c)|f/c|^{p-1}; rho=t^{-1}, rho'=1/t'(f) ----------
struct Law; r::Vector{Float64}; k::Vector{Float64}; c::Vector{Float64}; p::Vector{Float64}; end
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
# Beckmann DUAL energy E(φ)=Σ_e Φ*_e(g_e) - α d'φ (convex; ∇E = B ρ(B'φ)-αd = residual). Newton
# minimises E, so the line search seeks E-descent -- the principled globaliser for this convex solve.
function dual_energy(B,L,d,α,φ)
    g=B'*φ; f=similar(g); dρ=similar(g); rho_drho!(f,dρ,g,L)
    s=0.0; @inbounds for e in eachindex(g); s += g[e]*f[e]-Phi(L,e,f[e]); end
    s - α*dot(d,φ)
end

# ---------- NLF: damped Newton on B rho(B'phi)=alpha d, zero-mean gauge ----------
zm!(x)=(x.-=sum(x)/length(x); x)
median_(x)=(y=sort(x); n=length(y); iseven(n) ? 0.5*(y[n÷2]+y[n÷2+1]) : y[(n+1)÷2])
# enforce exact symmetry + zero row sums (B diag(w) B' drifts by ~1e-16; LAMG+ setup asserts a clean
# Laplacian). Rebuild diagonal from the symmetrised off-diagonals.
function laplacian_clean(J)
    off = J - spdiagm(0=>diag(J)); off = (off + off')/2; dropzeros!(off)
    off + spdiagm(0=>-vec(sum(off, dims=2)))
end
# damped Newton at a FIXED load α, warm-started from φ (residual line search). Returns (φ, #steps).
# Lazy hierarchy refresh: the LAMG+ hierarchy (H/SC refs, shared across load steps) is rebuilt only
# when first needed, when the per-step residual factor exceeds `refresh`, or after a stale stall
# (rebuild + retry). `refresh=0` reproduces rebuild-every-step. `setups` counts rebuilds.
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
# NLF with LOAD CONTINUATION: ramp α up, warm-starting each Newton solve (the framework's own
# α-continuation; congestion flow has no fold, so a handful of warm-started steps converge).
# The lazily refreshed hierarchy is shared across the whole load chain.
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
# Phi_e(f) = 0.5 r_e f^2 + (k_e c_e/(p+1)) |f/c_e|^{p+1};  with p=4 the congestion term is
# kc_e * ((f/c_e)^2)^{(p+1)/2}, base (f/c)^2>=0 so the fractional power is smooth for Ipopt.
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

# ---------- single-commodity demand: BFS-far s,t, magnitude D ----------
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
