# Multicommodity NLF: K commodities share one network; congestion couples them through the
# Euclidean magnitude of the per-edge commodity-flow vector. Modular layer over bpr_common.jl
# (reuses the Law struct, per-edge inversion, hierarchy machinery, component/demand helpers);
# nothing in the single-commodity code changes.
#
# Model: potentials Φ = [φ^1 .. φ^K] (n×K); per-edge gradient vector g_e = (B'Φ)_{e,:} ∈ R^K;
#   flow vector   f_e = ρ_e(‖g_e‖) g_e/‖g_e‖    (ρ_e = the scalar edge law; K=1 reduces exactly),
#   equilibrium   B f^k = α d^k,  k = 1..K  — stationarity of the convex energy
#   E(Φ) = Σ_e Φ*_e(‖g_e‖) − α Σ_k (d^k)'φ^k.
# Per-edge Jacobian block: c_e I_K + (ρ'_e − c_e) u_e u_e',  c_e = ρ_e(‖g_e‖)/‖g_e‖, u_e = g_e/‖g_e‖.
# Eigenvalues {c_e (⊥ u_e, multiplicity K−1), ρ'_e (∥ u_e)} — SPD for monotone laws. One frozen
# SCALAR LAMG+ hierarchy on w_e = √(c_e ρ'_e) preconditions the whole K-commodity block system:
# per-edge preconditioned spectrum {c/w, ρ'/w} = {√(c/ρ'), √(ρ'/c)}, condition ≤ c/ρ' ≤ p.

include(joinpath(@__DIR__, "bpr_common.jl"))

# scalar law at one edge: F = ρ_e(s), ρ'_e(s), for s = ‖g_e‖ ≥ 0 (same bracketed Newton as rho_drho!)
@inline function rho_scalar(L, e, s)
    F = s / L.r[e]                       # linear guess = upper bracket (t(F) >= s)
    for _ in 1:60
        res = tof(L, e, F) - s
        abs(res) <= 1e-12*(s + 1) && break
        F -= res / dtdf(L, e, F); F < 0 && (F = 0.0)
    end
    F, 1.0/dtdf(L, e, F)
end

# vector law over all edges: G, F are m×K. Fills F, c (isotropic conductance ρ/‖g‖), dρ (radial ρ').
function mc_law!(F, c, dρ, G, L)
    m, K = size(G)
    @inbounds for e in 1:m
        s2 = 0.0
        for k in 1:K; s2 += G[e,k]*G[e,k]; end
        s = sqrt(s2)
        Fm, rp = rho_scalar(L, e, s)
        ce = s > 0 ? Fm/s : rp           # ρ(s)/s → ρ'(0) as s → 0
        c[e] = ce; dρ[e] = rp
        for k in 1:K; F[e,k] = ce*G[e,k]; end
    end
end

zm2!(X) = (n = size(X,1); for k in 1:size(X,2); X[:,k] .-= sum(@view X[:,k])/n; end; X)

mc_res(B, F, D, α) = B*F .- α.*D         # residual matrix R = BF − αD (n×K), zero column sums

# ---------- frozen full-block Jacobian J_H (for the FCG inner solver and the direct reference) ----
struct FrozenJ
    B::SparseMatrixCSC{Float64,Int}
    c::Vector{Float64}                   # isotropic conductance at the build state
    a::Vector{Float64}                   # ρ' − c at the build state
    U::Matrix{Float64}                   # m×K unit gradient directions at the build state
end
function frozenJ(B, G, c, dρ)
    m, K = size(G); U = zeros(m, K)
    @inbounds for e in 1:m
        s = 0.0; for k in 1:K; s += G[e,k]*G[e,k]; end; s = sqrt(s)
        s > 0 && (for k in 1:K; U[e,k] = G[e,k]/s; end)
    end
    FrozenJ(B, copy(c), dρ .- c, U)
end
function mulJ(J::FrozenJ, X)             # J_H X, X n×K (zero-mean gauge per commodity)
    Gx = J.B' * X; m, K = size(Gx)
    @inbounds for e in 1:m
        t = 0.0; for k in 1:K; t += J.U[e,k]*Gx[e,k]; end
        t *= J.a[e]
        for k in 1:K; Gx[e,k] = J.c[e]*Gx[e,k] + t*J.U[e,k]; end
    end
    zm2!(J.B * Gx)
end
# assembled Kn×Kn block matrix (validation / :direct only)
function assembleJ(J::FrozenJ, K)
    n = size(J.B,1)
    blocks = Matrix{Any}(undef, K, K)
    for k in 1:K, l in 1:K
        w = (k == l ? J.c : zeros(length(J.c))) .+ J.a .* (@view J.U[:,k]) .* (@view J.U[:,l])
        blocks[k,l] = J.B * Diagonal(w) * J.B'
    end
    reduce(vcat, [reduce(hcat, blocks[k,:]) for k in 1:K])
end

# ---------- inner solvers for the frozen correction equation J_H Δ = −R ----------
# (a) :mg — block-diagonal chord: Δ^k = LAMG+ solve on the scalar w-hierarchy, per commodity.
function mc_mg!(Δ, R, H, SC, GG; eta=0.05)
    for k in 1:size(R,2)
        δ, info = solve(H[], zm!(-Vector(@view R[:,k])./SC[]); options=LAMGOptions(tol=eta, γ_coarse_growth=GG[]))
        get(info, :gamma_escalated, false) && (GG[] = 1.15)
        Δ[:,k] .= zm!(δ)
    end
    Δ
end
# (b) :mgcg — flexible PCG on the full frozen block J_H, preconditioned by (a) at loose tolerance.
function mc_fcg!(Δ, JF::FrozenJ, R, H, SC, GG; eta=0.05, ptol=0.1, itmax=30)
    Δ .= 0.0
    res = zm2!(-copy(R)); nb = norm(res)
    nb == 0 && return Δ, 0
    prec(Rv) = (Z = similar(Rv);
        for k in 1:size(Rv,2)
            δ, info = solve(H[], zm!(Vector(@view Rv[:,k])./SC[]); options=LAMGOptions(tol=ptol, γ_coarse_growth=GG[]))
            get(info, :gamma_escalated, false) && (GG[] = 1.15)
            Z[:,k] .= zm!(δ)
        end; Z)
    Z = prec(res); P = copy(Z); zr = dot(Z, res); its = 0
    for j in 1:itmax
        its = j
        Y = mulJ(JF, P); pAp = dot(P, Y)
        pAp <= 0 && break
        αc = zr / pAp
        Δ .+= αc .* P
        resn = res .- αc .* Y
        norm(resn) <= eta*nb && (res = resn; break)
        Zn = prec(resn)
        β = dot(Zn, resn .- res) / zr    # flexible (Polak–Ribière) update
        zr = dot(Zn, resn); res = resn
        P .= Zn .+ β .* P; Z = Zn
    end
    Δ, its
end

# ---------- chord-Newton at a FIXED load α (the K-commodity Algorithm 1) ----------
function mc_newton_fixed!(Φ, B, L, D, α, keepix; inner=:mg, nmax=80, tol=1e-9, t0=time(), tlim=Inf,
                          refresh=0.25, eta=0.05, hweight=:geo,
                          H=Ref{Any}(nothing), SC=Ref(1.0), ST=Ref(false), setups=Ref(0),
                          GG=Ref(1.0), JF=Ref{Any}(nothing), cgits=Ref(0))
    n, K = size(Φ); m = size(B,2)
    F = zeros(m,K); c = zeros(m); dρ = zeros(m); steps = 0; nr_prev = Inf
    nd = max(α*norm(D), 1.0)
    # Adaptive refresh threshold: the block-diagonal frozen operator has an intrinsic per-step rate
    # floor set by the law's anisotropy ρ'/c (independent of staleness); rebuilding cannot beat it.
    # Measure the rate on the step right after each rebuild and only call the operator stale when
    # the observed factor exceeds that fresh-rate baseline.
    thr = refresh; fresh_prev = false
    rebuild! = (G) -> begin
        w = hweight === :iso  ? copy(c)  :
            hweight === :rhop ? copy(dρ) : sqrt.(c .* dρ)
        w .= max.(w, 1e-11*maximum(w))
        SC[] = maximum(w)
        H[] = setup(laplacian_clean(B*Diagonal(w./SC[])*B'); options=LAMGOptions())
        JF[] = frozenJ(B, G, c, dρ); ST[] = false; setups[] += 1
    end
    trial_nr = (Φt) -> begin
        Gt = B'*Φt; Ft = zeros(m,K); ct = zeros(m); dt = zeros(m)
        mc_law!(Ft, ct, dt, Gt, L); norm(B*Ft .- α.*D)
    end
    for it in 1:nmax
        steps = it
        G = B'*Φ; mc_law!(F, c, dρ, G, L); R = B*F .- α.*D; nr = norm(R)
        if fresh_prev && isfinite(nr_prev) && nr_prev > 0
            thr = clamp(1.3*nr/nr_prev, refresh, 0.9); fresh_prev = false
        end
        nr < tol*nd && break
        (time()-t0 > tlim) && break
        Δ = zeros(n,K)
        if inner === :direct                 # exact frozen-J solve: the validation reference
            Jb = frozenJ(B, G, c, dρ); A = assembleJ(Jb, K)
            idx = vec([(k-1)*n + i for i in keepix, k in 1:K])
            x = A[idx,idx] \ -vec(R)[idx]
            for (j,ii) in enumerate(idx); Δ[ii] = x[j]; end
            zm2!(Δ)
        else
            if H[] === nothing || (ST[] && nr > thr*nr_prev)
                rebuild!(G); fresh_prev = true
            end
            if inner === :mgcg
                _, its = mc_fcg!(Δ, JF[], R, H, SC, GG; eta=eta); cgits[] += its
            else
                mc_mg!(Δ, R, H, SC, GG; eta=eta)
            end
        end
        nr_prev = nr
        τ = 1.0; ok = false
        for _ in 1:50
            Φt = zm2!(Φ .+ τ.*Δ)
            if trial_nr(Φt) <= nr; Φ .= Φt; ok = true; break; end; τ *= 0.5
        end
        if !ok && inner !== :direct && ST[]  # stale stall: rebuild + retry once
            rebuild!(G); fresh_prev = true
            Δ .= 0.0
            inner === :mgcg ? (mc_fcg!(Δ, JF[], R, H, SC, GG; eta=eta)) : mc_mg!(Δ, R, H, SC, GG; eta=eta)
            τ = 1.0
            for _ in 1:50
                Φt = zm2!(Φ .+ τ.*Δ)
                if trial_nr(Φt) <= nr; Φ .= Φt; ok = true; break; end; τ *= 0.5
            end
        end
        ok || break
        ST[] = true
    end
    steps
end

# ---------- the K-commodity NLF solver: load continuation, shared lazy hierarchy ----------
function nlf_mc(B, L, D, α; inner=:mg, loads=(0.25,0.5,1.0), tol=1e-9, tlim=Inf,
                 refresh=0.25, hweight=:geo, setups=Ref(0), cgits=Ref(0))
    n = size(B,1); m = size(B,2); K = size(D,2)
    deg = vec(sum(abs.(B), dims=2)); pin = argmax(deg); keepix = setdiff(1:n, pin)
    Φ = zeros(n,K); tot = 0; t0 = time()
    H = Ref{Any}(nothing); SC = Ref(1.0); ST = Ref(false); GG = Ref(1.0); JF = Ref{Any}(nothing)
    for ℓ in loads
        tot += mc_newton_fixed!(Φ, B, L, D, ℓ*α, keepix; inner=inner, tol=tol, t0=t0, tlim=tlim,
                                refresh=refresh, hweight=hweight, H=H, SC=SC, ST=ST,
                                setups=setups, GG=GG, JF=JF, cgits=cgits)
    end
    G = B'*Φ; F = zeros(m,K); c = zeros(m); dρ = zeros(m); mc_law!(F, c, dρ, G, L)
    zm2!(Φ), F, tot, norm(B*F .- α.*D)
end

# ---------- K well-separated demand dipoles: greedy k-center on BFS distance ----------
function k_dipoles(B, K)
    n = size(B,1); adj = adjacency(B)
    bfs(s) = (dist = fill(typemax(Int)÷2, n); dist[s] = 0; q = [s];
        while !isempty(q); u = popfirst!(q);
            for v in adj[u]; dist[v] > dist[u]+1 && (dist[v] = dist[u]+1; push!(q,v)); end; end; dist)
    picks = Int[argmax(bfs(1))]
    mind = bfs(picks[1])
    while length(picks) < min(2K, n)
        nxt = argmax(mind)
        nxt in picks && (nxt = first(v for v in 1:n if !(v in picks)))
        push!(picks, nxt); mind = min.(mind, bfs(nxt))
    end
    # cyclic pairing: with P = length(picks) ≥ 2, consecutive mod-P indices are always distinct,
    # so tiny components (n < 2K) get duplicate-but-valid dipoles rather than out-of-bounds picks
    P = length(picks)
    D = zeros(n, K)
    for k in 1:K
        s = picks[mod1(2k-1, P)]; t = picks[mod1(2k, P)]
        s == t && (t = picks[mod1(2k+1, P)])
        D[s,k] = -1.0; D[t,k] = +1.0
    end
    D
end
