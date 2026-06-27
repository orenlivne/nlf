# ACCURACY COMPLEXITY: how does NLF's cost depend on the desired solution accuracy eps?
# Theory: each inner LAMG+ solve to relative tol delta costs O(m log 1/delta) (bounded mg factor);
# with inexact Newton (inner forcing eta_k ~ ||r_k||) the per-step cycle counts form a geometric
# series dominated by the final solve (||r|| ~ eps), so the TOTAL inner cycles ~ O(log 1/eps), giving
# O(m log 1/eps) overall. We verify by sweeping eps and counting total V-cycles for a few graphs.
include(joinpath(@__DIR__, "bpr_common.jl"))
function load_mtx(path)
    nr=nc=0; started=false; seen=Set{Tuple{Int,Int}}(); ne=0; I=Int[]; Jc=Int[]; V=Float64[]
    for ln in eachline(path)
        (isempty(ln) || ln[1]=='%') && continue; t=split(ln)
        if !started; nr=parse(Int,t[1]); nc=parse(Int,t[2]); started=true; continue; end
        length(t)<2 && continue; i=parse(Int,t[1]); j=parse(Int,t[2]); i==j && continue
        key=i<j ? (i,j) : (j,i); (key in seen) && continue; push!(seen,key)
        ne+=1; push!(I,key[1]);push!(Jc,ne);push!(V,-1.0); push!(I,key[2]);push!(Jc,ne);push!(V,+1.0)
    end
    sparse(I,Jc,V,max(nr,nc),ne)
end
function graph_instance(B0; seed=1)
    rng=MersenneTwister(seed); m=size(B0,2)
    largest_component(B0,[0.5+2.5*rand(rng) for _ in 1:m],[1.0+4.0*rand(rng) for _ in 1:m],fill(0.15,m),fill(4.0,m))
end
# inexact-Newton NLF: inner relative tol = eta (forcing term); count total inner V-cycles.
function nlf_acc(B,L,d,α; η=0.1, εout=1e-8, nmax=400, loads=(0.25,0.5,1.0))
    n=size(B,1); m=size(B,2); deg=vec(sum(abs.(B),dims=2)); pin=argmax(deg); keepix=setdiff(1:n,pin)
    φ=zeros(n); steps=0; cyc=0; iopt=LAMGOptions(); iopt_tol=η
    for (li,ℓ) in enumerate(loads)
        αℓ=ℓ*α; f=zeros(m); dρ=zeros(m); last=(li==length(loads))
        for it in 1:nmax
            g=B'*φ; rho_drho!(f,dρ,g,L); r=B*f.-αℓ.*d; nr=norm(r)
            tt = last ? εout : 1e-3                            # earlier loads: fixed path tol
            nr < tt*max(αℓ*norm(d),1.0) && break
            steps+=1
            dρf=max.(dρ,1e-11*maximum(dρ)); J=B*Diagonal(dρf)*B'; sc=maximum(dρf)
            h=setup(laplacian_clean(J./sc); options=LAMGOptions())
            δ,info=solve(h, zm!(-Vector(r)./sc); options=LAMGOptions(tol=η)); zm!(δ); cyc+=info.cycles
            τ=1.0; ok=false
            for _ in 1:50
                φt=zm!(φ.+τ.*δ); gt=B'*φt; ft=similar(f); dt=similar(dρ); rho_drho!(ft,dt,gt,L)
                if norm(B*ft.-αℓ.*d)<=nr; copyto!(φ,φt); ok=true; break; end; τ*=0.5
            end
            ok||break
        end
    end
    g=B'*φ; f=similar(g); dρ=similar(g); rho_drho!(f,dρ,g,L)
    steps, cyc, norm(B*f.-α.*d)
end

DATA=joinpath(@__DIR__,"..","data")
graphs=[("web","Gleich__wb-cs-stanford.mtx"),("social","LAW__enron.mtx"),
        ("road","DIMACS10__ak2010.mtx"),("FEM","Pothen__bodyy5.mtx")]
eps_list=[1e-2,1e-3,1e-4,1e-6,1e-8,1e-10,1e-12]
OUT=length(ARGS)>=1 ? ARGS[1] : "/tmp/nlf_accuracy.csv"
open(OUT,"w") do io
    println(io,"class,graph,m,eps,newton_steps,total_cycles,time_s,residual")
    @printf("%-8s %-18s %-8s %-7s %-6s %-7s %-8s\n","class","graph","m","eps","steps","cycles","time")
    for (cls,fn) in graphs
        B,c,t0,b,p=graph_instance(load_mtx(joinpath(DATA,fn))); m=size(B,2)
        L=make_law(c,t0,b,p); s,t=far_pair(B); d=zeros(size(B,1));d[s]=-1.0;d[t]=+1.0; α=0.3*median_(c)
        nlf_acc(B,L,d,α; εout=1e-4)                          # warm
        for ε in eps_list
            tm=@elapsed ((st,cy,res)=nlf_acc(B,L,d,α; εout=ε))
            @printf("%-8s %-18s %-8d %.0e %-6d %-7d %-8.3f res=%.1e\n",cls,fn[1:min(end,18)],m,ε,st,cy,tm,res)
            @printf(io,"%s,%s,%d,%.0e,%d,%d,%.6f,%.3e\n",cls,fn,m,ε,st,cy,tm,res); flush(io)
        end
    end
end
println("DONE_ACCURACY")
