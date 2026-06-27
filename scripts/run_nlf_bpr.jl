# VALIDATION: NLF (BPR congestion, real road networks) reaches the same equilibrium as Ipopt.
# Data: Transportation Networks for Research (Sioux Falls, Anaheim, Chicago, ...) -- the field-
# standard benchmark with REAL topology, capacities, free-flow times and BPR parameters (b, power).
# Shared solver/parser/law live in bpr_common.jl.
include(joinpath(@__DIR__, "bpr_common.jl"))

dir = length(ARGS)>=1 ? ARGS[1] : "/tmp/tn_probe"
println("== NLF (BPR congestion, real road nets) vs Ipopt -- same equilibrium? ==")
@printf("%-16s %-6s %-6s %-11s %-11s %-9s %-5s\n","network","n","m","NLF cost","Ipopt cost","|fF-fI|","it")
for name in ["SiouxFalls","Anaheim","Chicago-Sketch","Austin","Berlin-Center","chicago-regional","Philadelphia","Sydney"]
    B0,c0,t00,b0,p0,_ = parse_tntp(dir,name)
    B,c,t0,b,p = largest_component(B0,c0,t00,b0,p0)
    n=size(B,1); m=size(B,2); L=make_law(c,t0,b,p)
    s,t = far_pair(B); d=zeros(n); d[s]=-1.0; d[t]=+1.0
    α = 0.3*median_(c)
    φ,fF,it,res = nlf_bpr(B,L,d,α; inner=:direct)
    fI,ti,st = ipopt_bpr(B,L,d,α)
    df = any(isnan,fI) ? NaN : norm(fF-fI)/max(norm(fI),1e-9)
    @printf("%-16s %-6d %-6d %-11.4f %-11.4f %-9.2e it=%-3d maxutil=%.2f %s\n",
            name,n,m,ccost(L,fF), any(isnan,fI) ? NaN : ccost(L,fI), df, it, maximum(abs.(fF)./c), st)
end
println("DONE_BPR")
