# verify_repro.jl — reproduce (and PRINT next to the paper's numbers) the headline NLF results.
# Run:  julia --project=<env> examples/verify_repro.jl
# Uses only the small graphs shipped in data/, so it finishes in a few minutes.

include(joinpath(@__DIR__, "repro_lib.jl"))

hr() = println("="^78)

# -------------------------------------------------------------------------------------
# 1) MAX-FLOW EXACTNESS  (paper Table tab:mf). Reference F* is the paper's LP/push-relabel
#    value; NLF (chord-Newton + arclength continuation) must reach it to ~1e-4.
# -------------------------------------------------------------------------------------
hr(); println("1) MAX-FLOW EXACTNESS  (paper Table tab:mf)"); hr()
mf_cases = [
    ("grid2d/16",  () -> nlf_grid2d(16; cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(3)),  4.973),
    ("grid3d/6",   () -> nlf_grid3d(6;  cap_lo=0.1, cap_hi=10.0, rng=MersenneTwister(8)),  17.478),
    ("washington", () -> nlf_washington(8, 8; rng=MersenneTwister(12)),                    500.745),
]
@printf("%-12s %5s  %-12s  %-12s  %-9s  %-9s\n", "graph","n","paper F*","NLF F*","NLF/F*","conv")
println("-"^70)
mf_rows = []
for (name, gen, paperF) in mf_cases
    mfp = gen()
    n = size(mfp.A, 1)
    Fstar, φ, f, info = nlf_maxflow(mfp; tol=1e-8, inner=:direct)
    ratio = Fstar / paperF
    @printf("%-12s %5d  %-12.4f  %-12.4f  %-9.4f  %-9s\n", name, n, paperF, Fstar, ratio, info.converged)
    push!(mf_rows, (name=name, n=n, paperF=paperF, nlfF=Fstar, ratio=ratio))
end
println()

# -------------------------------------------------------------------------------------
# 2) FOLD INNER-SOLVER SWAP  (paper Table tab:innerfold). Hold the outer continuation
#    fixed; swap the inner Laplacian solver across all four engines. Every converging
#    solver must return the IDENTICAL F*; ordering: direct fastest, approxChol next.
# -------------------------------------------------------------------------------------
hr(); println("2) FOLD INNER-SOLVER SWAP  (paper Table tab:innerfold)"); hr()
fold_cases = [
    ("grid2",     "AG-Monien__grid2.mtx",   3.003),
    ("circuit_1", "Bomhof__circuit_1.mtx",  0.944),
]
# warm-up (JIT) on a small instance, all engines
let mfp = mtx_maxflow(data_path("AG-Monien__grid2.mtx"))
    for (_, ic) in INNERS
        try; nlf_maxflow(mfp; inner=ic, max_steps=30); catch; end
    end
end
fold_rows = []
for (label, fn, paperF) in fold_cases
    mfp = mtx_maxflow(data_path(fn))
    n = size(mfp.B, 1); m = size(mfp.B, 2)
    println("\n$label  ($fn)   n=$n  m=$m   paper F*=$paperF")
    @printf("    %-11s  %-12s  %-7s  %-9s  %s\n", "inner","F*","steps","time_s","converged")
    println("    ", "-"^58)
    for (nm, ic) in INNERS
        t = @elapsed ((Fs, φ, f, info) = nlf_maxflow(mfp; inner=ic, max_steps=120, tol=1e-7))
        @printf("    %-11s  %-12.6f  %-7d  %-9.3f  %s\n", nm, Fs, info.steps, t, info.converged)
        push!(fold_rows, (label=label, inner=nm, Fstar=Fs, steps=info.steps, t=t, conv=info.converged, paperF=paperF))
        flush(stdout)
    end
end
println()

# -------------------------------------------------------------------------------------
# 3) NO-FOLD CONGESTION vs. COMPETITORS. Build a small BPR instance and time NLF
#    (multigrid-Newton) vs Ipopt (interior-point) vs L-BFGS-on-dual (first-order).
# -------------------------------------------------------------------------------------
hr(); println("3) NO-FOLD CONGESTION vs. COMPETITORS  (NLF / Ipopt / L-BFGS)"); hr()
# Bomhof__circuit_1 — small enough to finish fast (n ~ 2.6k after LCC).
B0 = load_mtx(data_path("Bomhof__circuit_1.mtx"))
B, c, t0, b, p = instance(B0)
n = size(B, 1); m = size(B, 2)
L = make_law(c, t0, b, p)
s, t = far_pair(B); d = zeros(n); d[s] = -1.0; d[t] = 1.0
α = 0.3 * median_(c)
println("graph = Bomhof__circuit_1 (LCC)   n=$n  m=$m   α=$(round(α; digits=4))")

# warm-up (JIT) each solver
nlf_bpr(B, L, d, α; inner=:multigrid)
lbfgs_dual(B, L, d, α; tlim=5.0)
ipopt_bpr(B, L, d, α; tlim=30.0)

# NLF (multigrid-Newton)
φ, fF, st, r = nlf_bpr(B, L, d, α; inner=:multigrid, tlim=600.0)
tNLF = @elapsed nlf_bpr(B, L, d, α; inner=:multigrid, tlim=600.0)
nlf_ok = r < 1e-6
# Ipopt
fI, tIP, stIP = ipopt_bpr(B, L, d, α; tlim=60.0)
ip_ok = !any(isnan, fI)
# L-BFGS-on-dual
lb = lbfgs_dual(B, L, d, α; tlim=60.0)

println()
@printf("%-16s  %-9s  %-7s  %-9s  %s\n", "solver","time_s","steps","resid","converged")
println("-"^60)
@printf("%-16s  %-9.4f  %-7d  %-9.2e  %s\n", "NLF (multigrid)", tNLF, st, r, nlf_ok)
@printf("%-16s  %-9.4f  %-7s  %-9s  %s\n", "Ipopt (IPM)",    tIP, "-", "-", string(stIP))
@printf("%-16s  %-9.4f  %-7d  %-9.2e  %s\n", "L-BFGS (dual)",  lb.t, lb.it, lb.r, lb.c)
nofold = (n=n, m=m, tNLF=tNLF, st=st, r=r, nlf_ok=nlf_ok,
          tIP=tIP, ip_ok=ip_ok, stIP=string(stIP),
          tLB=lb.t, itLB=lb.it, rLB=lb.r, cLB=lb.c)
println()

# -------------------------------------------------------------------------------------
# 4) REPRODUCED vs PAPER summary
# -------------------------------------------------------------------------------------
hr(); println("4) REPRODUCED vs PAPER"); hr()
println("[tab:mf] max-flow exactness (NLF/F* should be ~1.0000):")
for r in mf_rows
    @printf("   %-12s paper F*=%-9.3f  NLF F*=%-9.4f  NLF/F*=%.4f\n", r.name, r.paperF, r.nlfF, r.ratio)
end
println("\n[tab:innerfold] every converging inner solver returns the SAME F*:")
for label in unique(getfield.(fold_rows, :label))
    rows = filter(x -> x.label == label, fold_rows)
    paperF = rows[1].paperF
    Fs = [r.Fstar for r in rows if r.conv]
    spread = isempty(Fs) ? NaN : maximum(Fs) - minimum(Fs)
    @printf("   %-10s paper F*=%-7.3f  reproduced F*=%.4f  (max spread across solvers=%.2e)\n",
            label, paperF, isempty(Fs) ? NaN : Fs[1], spread)
    # ordering check
    conv = filter(x -> x.conv, rows)
    sort!(conv, by=x->x.t)
    @printf("       fastest->slowest: %s\n", join(["$(r.inner)=$(round(r.t;digits=2))s" for r in conv], "  "))
end
println("\n[no-fold] NLF vs competitors on Bomhof__circuit_1:")
@printf("   NLF=%.3fs (resid %.1e, %s)   Ipopt=%.3fs (%s)   L-BFGS=%.3fs (converged=%s)\n",
        nofold.tNLF, nofold.r, nofold.nlf_ok ? "ok" : "FAIL",
        nofold.tIP, nofold.stIP, nofold.tLB, nofold.cLB)
hr(); println("DONE_VERIFY_REPRO"); hr()
