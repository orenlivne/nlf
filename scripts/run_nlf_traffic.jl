# NLF as an O(m) solver for CONVEX CONGESTION (traffic) network flow.
#
# Edge cost (travel time) grows with flow:  t_e(f)=c_e sinh(f/c_e)  (psi ~ 1/c_e congestion).
# Equilibrium = min sum_e Phi_e(f_e) s.t. B f = alpha d,  Phi_e(f)=c_e^2(cosh(f/c_e)-1).
# Its dual (node potentials phi) is the nonlinear Laplacian  B rho(B'phi)=alpha d  with
# rho_e=t_e^{-1}=c_e asinh(g/c_e), rho'_e=1/sqrt(1+(g/c_e)^2) in (0,1].  NO feasibility fold
# (flow is unbounded; any demand is feasible), so NLF is PURE Newton: J=B diag(rho')B' is a
# well-conditioned weighted Laplacian solved by LAMG+; O(1) Newton steps x O(m) inner = O(m).
#
# Competitor: Ipopt (interior-point) on the same convex program -- a sparse-direct KKT core, so
# superlinear / OOM on poorly-separable graphs, exactly where NLF's O(m) wins.
import Pkg; Pkg.activate(joinpath(@__DIR__, "..", "examples", "repro_env"))
using LAMG, SparseArrays, LinearAlgebra, Printf, Random
using JuMP
import LAMG: nlf_grid2d, nlf_grid3d, nlf_random, LAMGOptions, setup, solve
un(x)=x isa Tuple ? first(x) : x

# ---- congestion law (asinh) ----
function rho_drho!(f,dρ,g,c)
    @inbounds for e in eachindex(g)
        x=g[e]/c[e]; f[e]=c[e]*asinh(x); dρ[e]=1/sqrt(1+x*x)
    end
end

# ---- NLF: damped Newton on B rho(B'phi)=alpha d, zero-mean gauge (both r,d perp 1) ----
zm!(x)=(x.-=sum(x)/length(x); x)
function nlf_traffic(mfp, α; inner=:direct, nmax=60, tol=1e-9, verbose=false)
    B=mfp.B; n=size(B,1); m=size(B,2); d=Vector(mfp.d)
    c=[max(mfp.high[e], -mfp.low[e]) for e in 1:m]      # per-edge congestion scale
    φ=zeros(n); f=zeros(m); dρ=zeros(m); steps=0; o=ones(n)
    for it in 1:nmax
        steps=it
        g=B'*φ; rho_drho!(f,dρ,g,c); r=B*f .- α.*d
        nr=norm(r); verbose && @printf("  newton %d  |r|=%.3e\n",it,nr)
        nr < tol*max(α,1.0) && break
        J=B*Diagonal(max.(dρ,1e-12))*B'
        if inner==:direct                                # augmented: solve J y=-r, y perp 1
            A=[J reshape(o,n,1); reshape(o,1,n) zeros(1,1)]
            sol=A\[-Vector(r); 0.0]; δ=sol[1:n]
        else                                             # LAMG+ multigrid on the graph Laplacian J
            h=setup(sparse(J); options=LAMGOptions())
            δ,_=solve(h, zm!(-Vector(r)); options=LAMGOptions()); zm!(δ)
        end
        τ=1.0; ok=false
        for _ in 1:30
            φt=zm!(φ .+ τ.*δ); gt=B'*φt; ft=similar(f); dt=similar(dρ); rho_drho!(ft,dt,gt,c)
            if norm(B*ft.-α.*d)<=nr; φ=φt; ok=true; break; end
            τ*=0.5
        end
        ok||break
    end
    g=B'*φ; rho_drho!(f,dρ,g,c)
    zm!(φ), copy(f), steps, norm(B*f .- α.*d)
end

# ---- Ipopt competitor: min sum c^2(cosh(f/c)-1) s.t. B f = alpha d ----
function ipopt_traffic(mfp, α)
    B=mfp.B; n=size(B,1); m=size(B,2); d=Vector(mfp.d)
    c=[max(mfp.high[e], -mfp.low[e]) for e in 1:m]
    model=Model(Ipopt.Optimizer); set_silent(model)
    @variable(model, f[1:m])
    @constraint(model, B*f .== α.*d)
    @NLobjective(model, Min, sum(c[e]^2*(cosh(f[e]/c[e])-1) for e in 1:m))
    t=@elapsed optimize!(model)
    (termination_status(model)==MOI.LOCALLY_SOLVED || termination_status(model)==MOI.OPTIMAL) ?
        (value.(f), t) : (fill(NaN,m), t)
end

println("== NLF congestion-flow vs Ipopt (validation: same equilibrium?) ==")
@printf("%-12s %-6s %-6s %-9s %-9s %-9s %-9s\n",
        "graph","n","m","NLF cost","Ipopt cost","|f_F-f_I|","NLF it")
function run(name,mfp,α)
    n=size(mfp.A,1); m=size(mfp.B,2)
    c=[max(mfp.high[e], -mfp.low[e]) for e in 1:m]
    φ,fF,it,res = nlf_traffic(mfp,α; inner=:direct)
    cost(f)=sum(c[e]^2*(cosh(f[e]/c[e])-1) for e in 1:m)
    fI,_ = ipopt_traffic(mfp,α)
    df = any(isnan,fI) ? NaN : norm(fF-fI)/max(norm(fI),1e-9)
    @printf("%-12s %-6d %-6d %-9.4f %-9.4f %-9.2e %-9d\n",
            name,n,m,cost(fF),any(isnan,fI) ? NaN : cost(fI),df,it)
end
import Ipopt
run("grid2d/6",  un(nlf_grid2d(6;cap_lo=0.5,cap_hi=3.0,rng=MersenneTwister(1))), 2.0)
run("grid2d/10", un(nlf_grid2d(10;cap_lo=0.5,cap_hi=3.0,rng=MersenneTwister(2))), 3.0)
run("grid2d/14", un(nlf_grid2d(14;cap_lo=0.5,cap_hi=3.0,rng=MersenneTwister(3))), 4.0)
println("DONE_TRAFFIC")
