using Test
using LinearAlgebra
using SparseArrays
using Random
using NLF

@testset "max-flow elimination (degree 0/1)" begin
    # ─────────────────────────────────────────── identifier
    @testset "low_degree_nlf_nodes finds 0/1-degree, respects protect" begin
        #     1 — 2 — 3 — 4        plus isolated node 5.
        edges = [(2, 1), (3, 2), (4, 3)]
        c⁺ = [1.0, 1.0, 1.0]; c⁻ = [1.0, 1.0, 1.0]
        mfp = make_problem(5, edges, c⁺, c⁻, 1, 4; name = "path5")
        z, f1 = low_degree_nlf_nodes(mfp; protect = [mfp.s, mfp.t])
        @test 5 ∈ z                 # isolated
        @test 1 ∉ f1                # protected (source)
        @test 4 ∉ f1                # protected (sink)
        # Nodes 1, 4 are degree-1 but protected; 2, 3 are degree-2.
        @test isempty(f1)
    end

    @testset "low_degree_nlf_nodes finds a non-protected degree-1 leaf" begin
        # Path 1—2—3 with a degree-1 leaf 4 attached to 2. Source=1, sink=3.
        # Node 4 has degree 1 and is unprotected → should be in f1.
        edges = [(2, 1), (3, 2), (2, 4)]
        c⁺ = [1.0, 1.0, 1.0]; c⁻ = [1.0, 1.0, 1.0]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 3; name = "Y")
        z, f1 = low_degree_nlf_nodes(mfp; protect = [mfp.s, mfp.t])
        @test 4 ∈ f1
        @test isempty(z)
    end

    # ─────────────────────────────────────────── reduced problem builds
    @testset "eliminate_low_degree produces a well-formed reduced problem" begin
        # Path 1—2—3 with leaf 4 off node 2.
        edges = [(2, 1), (3, 2), (2, 4)]
        c⁺ = [1.0, 1.0, 1.0]; c⁻ = [1.0, 1.0, 1.0]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 3; name = "Y")
        mfp_r, stage = eliminate_low_degree(mfp)
        @test size(mfp_r.A, 1) == 3        # node 4 gone
        @test length(mfp_r.head) == 2      # edge (2,4) gone
        @test mfp_r.A == mfp_r.B * mfp_r.B'    # still a Laplacian
        # Demand: leaf 4 has d[4] = 0 (not s, not t), so d^red == d_c.
        @test mfp_r.d == mfp.d[stage.c]
    end

    # ─────────────────────────────────────────── interpolation round-trips
    @testset "interpolate_eliminated! reconstructs φ_u = φ_v + α·d_u" begin
        # Y: 1—2—3, leaf 4 off node 2.
        edges = [(2, 1), (3, 2), (2, 4)]
        c⁺ = [1.0, 1.0, 1.0]; c⁻ = [1.0, 1.0, 1.0]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 3; name = "Y")
        # Leaf node 4 has d[4] = 0 (not source, not sink) so φ_4 = φ_2 regardless of α.
        _, stage = eliminate_low_degree(mfp)
        φ_c = [10.0, 20.0, 30.0]          # arbitrary values on c = [1,2,3]
        α = 0.5
        φ_full = zeros(4)
        interpolate_eliminated!(φ_full, φ_c, α, stage)
        @test φ_full[1] == 10.0
        @test φ_full[2] == 20.0
        @test φ_full[3] == 30.0
        @test φ_full[4] == 20.0           # = φ_2 + α·0  (leaf has d_4 = 0)
    end

    @testset "α-interval from a leaf with non-zero demand" begin
        # Construct a degree-1 leaf at node 4 with d[4] ≠ 0.
        # Trick: build a graph where 4 IS the sink (so d_4 = +1) but only
        # one edge connects to it. Then eliminating sink would normally be
        # disallowed; flip protect = [s] only.
        edges = [(2, 1), (3, 2), (4, 3)]
        c⁺ = [2.0, 2.0, 1.5]            # last cap is the binding one
        c⁻ = [2.0, 2.0, 1.5]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 4; name = "path4")
        # Skip the s/t protection: pass protect=[1] so node 4 can be eliminated.
        mfp_r, stage = eliminate_low_degree(mfp; protect = [1])
        # Node 4 is f. Edge e=(4,3) ⇒ σ_4 = +1, d_4 = +1. So box on e:
        # 1·α·1 ∈ [-1.5, +1.5] → α ∈ [-1.5, +1.5].
        @test stage.alpha_lo ≈ -1.5 atol = 1e-12
        @test stage.alpha_hi ≈ +1.5 atol = 1e-12
        @test 4 ∈ stage.f
    end

    # ─────────────────────────────────────────── Schur complement correctness
    @testset "Schur complement matches direct reduction on path-with-leaf" begin
        # 1—2—3, leaf 4 off node 2. Verify A^red equals the Laplacian of
        # the path 1—2—3 directly.
        edges = [(2, 1), (3, 2), (2, 4)]
        c⁺ = [1.0, 1.0, 1.0]; c⁻ = [1.0, 1.0, 1.0]
        mfp = make_problem(4, edges, c⁺, c⁻, 1, 3; name = "Y")
        mfp_r, _ = eliminate_low_degree(mfp)
        # The "direct" reduced problem: path 1—2—3 with edges (2,1), (3,2).
        # Its Laplacian is [[1,-1,0],[-1,2,-1],[0,-1,1]] — but our reduced
        # graph's node 2 has lost only one edge to leaf 4. Original A[2,2]=3
        # (degree 3), leaf adjusted -1·1·-1 = -1 ⇒ A_red[2,2] = 2. That
        # matches the direct path Laplacian.
        @test Matrix(mfp_r.A) ≈ [1.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]
    end

    # ─────────────────────────────────────────── operator-substitution at σ=-1
    @testset "interpolate_eliminated! satisfies φ_u = φ_v + α·d_u (σ-independent)" begin
        # Construct a graph where the sink (d=+1) is a degree-1 leaf at the
        # TAIL of its incident edge (σ = -1). The substitution φ_u = φ_v + α·d_u
        # must hold regardless of head/tail orientation. (The σ appears only
        # in the edge-gradient identity.)
        edges = [(2, 1), (3, 2), (3, 5)]      # last edge: head=3, tail=5 → σ_5=-1
        c⁺ = [2.0, 2.0, 2.0]; c⁻ = [0.5, 0.5, 0.5]
        mfp = make_problem(5, edges, c⁺, c⁻, 1, 5; name = "tail-leaf")
        # Skip s-protection (default protects [1, 5] so node 5 would survive).
        mfp_r, stage = eliminate_low_degree(mfp; protect = [1])
        @test 5 ∈ stage.f
        @test stage.f_sigma[findfirst(==(5), stage.f)] == -1
        @test stage.f_d[findfirst(==(5), stage.f)] == +1.0   # d_5 = +1 (sink)
        # α-interval: σ·α·d_u = (-1)·α·(+1) = -α ∈ [-0.5, 2.0]
        #          ⇒ α ∈ [-2.0, 0.5]
        @test stage.alpha_lo ≈ -2.0 atol = 1e-12
        @test stage.alpha_hi ≈ +0.5 atol = 1e-12
        # Interpolation: φ_5 = φ_3 + α · d_5 = φ_3 + α.
        # If we set φ_c = [10, 20, 30] on c = [1, 2, 3] and α = 0.1, the
        # operator equation at node 5 dictates φ_5 = 30 + 0.1 = 30.1.
        # (Under the OLD buggy σ-in-substitution code, this would have been
        # 30 - 0.1 = 29.9 — would have failed.)
        φ_c = [10.0, 20.0, 30.0]
        α = 0.1
        φ_full = zeros(5)
        interpolate_eliminated!(φ_full, φ_c, α, stage)
        @test φ_full[5] ≈ 30.1 atol = 1e-12
        # Sanity: the operator row at node 5 is exactly satisfied.
        @test (mfp.A * φ_full)[5] ≈ α * mfp.d[5] atol = 1e-10
    end

    # ─────────────────────────────────────────── FAS-elim cycle correctness
    @testset "FAS-elim 2-level cycle is stationary at interior-feasible exact φ*" begin
        # Path 1—2—3—4—5 with a leaf 6 off node 3 (leaf has d_6 = 0).
        # Capacities loose so the LAMG-solve gives an interior-feasible φ*.
        edges = [(2, 1), (3, 2), (4, 3), (5, 4), (3, 6)]
        c⁺ = fill(3.0, 5); c⁻ = fill(3.0, 5)
        mfp = make_problem(6, edges, c⁺, c⁻, 1, 5; name = "leaf-path")
        n = 6
        α = 0.1
        x, _info = solve(mfp.A, α .* mfp.d; options = LAMGOptions(tol = 1e-12))
        x .-= sum(x) / n
        grad = mfp.B' * x
        @test all(mfp.low .+ 1e-6 .<= grad .<= mfp.high .- 1e-6)
        φ = copy(x)
        r_before = norm(mfp.A * φ .- α .* mfp.d)
        fas_2level_cycle_elimination!(mfp, φ; α = α, ν_pre = 2, ν_post = 2,
                                       ν_coarsest = 20)
        r_after = norm(mfp.A * φ .- α .* mfp.d)
        # Cycle is exact at the exact solution because Schur elimination is exact.
        @test r_after < 1e-6 * norm(α .* mfp.d) + 1e-8
    end

    @testset "FAS-elim 2-level cycle reduces residual on Aφ = α d (cold start)" begin
        edges = [(2, 1), (3, 2), (4, 3), (5, 4), (3, 6)]
        c⁺ = fill(3.0, 5); c⁻ = fill(3.0, 5)
        mfp = make_problem(6, edges, c⁺, c⁻, 1, 5; name = "leaf-path")
        α = 0.1
        φ = zeros(6)
        r0 = norm(mfp.A * φ .- α .* mfp.d)
        ratio = fas_2level_cycle_elimination!(mfp, φ; α = α, ν_pre = 2,
                                               ν_post = 2, ν_coarsest = 30)
        @test ratio < 1.0
    end
end
