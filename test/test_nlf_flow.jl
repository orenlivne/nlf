# Unit tests for the generic source-form nonlinear-flow solver (nlf_flow.jl):
# newton_flow!, flow_continuation!, and the swappable build_solver hook.
using Test, NLF, SparseArrays, LinearAlgebra, Random

# small connected weighted graph: path backbone + random chords
function rand_graph(n; seed=1, extra=2n)
    rng = MersenneTwister(seed)
    I = Int[]; J = Int[]; V = Float64[]; e = 0
    add(i, j) = (e += 1; push!(I, i); push!(J, e); push!(V, -1.0); push!(I, j); push!(J, e); push!(V, +1.0))
    for i in 1:n-1; add(i, i+1); end
    for _ in 1:extra; a = rand(rng, 1:n); b = rand(rng, 1:n); a != b && add(a, b); end
    B = sparse(I, J, V, n, e); w = 0.5 .+ rand(rng, size(B, 2))
    B, w
end
zm(x) = x .- sum(x) / length(x)

@testset "nlf_flow" begin
    n = 40; B, w = rand_graph(n)
    b = zm(randn(MersenneTwister(3), n))                 # zero-mean source

    @testset "linear law == dense Laplacian solve" begin
        lin! = (f, dρ, g) -> (f .= w .* g; dρ .= w; nothing)
        Lw = Matrix(B * Diagonal(w) * B')
        xref = zm(pinv(Lw) * b)                          # min-norm zero-mean solution
        for inner in (:multigrid, :direct)
            x = zeros(n)
            res = newton_flow!(x, B, lin!, b; inner=inner, tol=1e-11)
            @test res.converged
            @test norm(zm(res.x) .- xref) / norm(xref) < 1e-6
        end
    end

    @testset "nonlinear monotone law converges & satisfies the equation" begin
        p = 3.0
        law! = (f, dρ, g) -> begin
            @inbounds for e in eachindex(g)
                s = g[e]; a = abs(s)
                f[e]  = w[e] * a^(p-1) * sign(s)
                dρ[e] = max(w[e] * (p-1) * a^(p-2), 1e-9 * maximum(w))
            end; nothing
        end
        x = zeros(n)
        res = newton_flow!(x, B, law!, b; inner=:multigrid, tol=1e-9)
        @test res.converged
        me = size(B, 2); f = zeros(me); dρ = zeros(me); g = B' * res.x; law!(f, dρ, g)
        @test norm(B * f .- b) < 1e-6                    # genuine stationarity
    end

    @testset "flow_continuation! warm-starts through a p-schedule" begin
        laws = map((2.0, 2.5, 3.0)) do p
            (f, dρ, g) -> begin
                @inbounds for e in eachindex(g)
                    s = g[e]; a = abs(s)
                    f[e]  = w[e] * a^(p-1) * sign(s)
                    dρ[e] = max(w[e] * (p-1) * a^(p-2), 1e-9 * maximum(w))
                end; nothing
            end
        end |> collect
        x = zeros(n)
        stages = flow_continuation!(x, B, laws, b; inner=:multigrid, tol=1e-8)
        @test length(stages) == 3
        @test stages[end].converged
    end

    @testset "build_solver hook (injected dense inner) matches LAMG+" begin
        lin! = (f, dρ, g) -> (f .= w .* g; dρ .= w; nothing)
        # inject a dense pinned solver as the inner engine
        dense_builder = L -> (rhs -> begin
            A = Matrix(L); A[1, :] .= 0; A[:, 1] .= 0; A[1, 1] = 1
            y = A \ (rhs .- rhs[1] * 0); y[1] = 0; y .- sum(y)/length(y)
        end)
        x1 = zeros(n); r1 = newton_flow!(x1, B, lin!, b; inner=:multigrid, tol=1e-11)
        x2 = zeros(n); r2 = newton_flow!(x2, B, lin!, b; build_solver=dense_builder, tol=1e-11)
        @test r2.converged
        @test norm(zm(r1.x) .- zm(r2.x)) / norm(zm(r1.x)) < 1e-6
    end
end
