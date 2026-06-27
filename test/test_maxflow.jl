using Test
using LinearAlgebra
using SparseArrays
using NLF

@testset "maxflow (Phase 1)" begin
    @testset "edge_resistance is asymmetric barrier" begin
        @test edge_resistance(1.0, 1.0, 0.0) ≈ 2.0
        @test edge_resistance(1.0, 1.0, 0.999) > 1e3
        @test edge_resistance(1.0, 1.0, -0.999) > 1e3
        @test edge_resistance(2.0, 2.0, 0.0) ≈ 0.5
        @test edge_resistance(2.0, 2.0, 0.5) ≈ edge_resistance(2.0, 2.0, -0.5)
    end

    @testset "edge_conductance = 1 / R" begin
        @test edge_conductance(1.0, 1.0, 0.0) ≈ 0.5
        @test edge_conductance(1.0, 1.0, 0.999) < 1e-3
    end

    @testset "solve_implicit_flow: Δφ = 0 ⇒ f = 0" begin
        f, R, Rp = NLF.solve_implicit_flow(1.0, 1.0, 0.0)
        @test abs(f) < 1e-12
    end

    @testset "solve_implicit_flow: linear regime recovers Ohm's law" begin
        # For Δφ very small relative to capacities, f ≈ Δφ / R(0).
        c⁺ = c⁻ = 1.0
        R0 = edge_resistance(c⁺, c⁻, 0.0)
        Δφ = 0.01
        f, R, _ = NLF.solve_implicit_flow(c⁺, c⁻, Δφ)
        @test f ≈ Δφ / R0 rtol = 1e-3
    end

    @testset "solve_implicit_flow: satisfies the implicit equation" begin
        for (c⁺, c⁻, Δφ) in ((1.0, 1.0, 0.3), (2.0, 1.0, 0.1),
                              (1.0, 2.0, -0.2), (0.5, 0.5, 0.1))
            f, R, _ = NLF.solve_implicit_flow(c⁺, c⁻, Δφ)
            @test f * R ≈ Δφ atol = 1e-8
            @test -c⁻ < f < c⁺
        end
    end

    @testset "solve_implicit_flow: barrier prevents saturation" begin
        # Pushing Δφ enormous: f approaches the capacity but never crosses.
        f, R, _ = NLF.solve_implicit_flow(1.0, 1.0, 1e4)
        @test f < 1.0
        f, R, _ = NLF.solve_implicit_flow(1.0, 1.0, -1e4)
        @test f > -1.0
    end

    @testset "MaxFlowLaplacian constructor + incidence lists" begin
        # 3-node path: 1—e1—2—e2—3, edges oriented (1,2) and (2,3).
        # B[u, e] = +1 if u head, -1 if u tail.
        # Orientation:  e1 tail=1, head=2;  e2 tail=2, head=3.
        B = sparse([-1.0  0.0;
                     1.0 -1.0;
                     0.0  1.0])
        mf = MaxFlowLaplacian(B, [1.0, 1.0], [1.0, 1.0], [-1.0, 0.0, 1.0])
        @test mf.f == zeros(2)
        @test length(mf.incident[1]) == 1
        @test length(mf.incident[2]) == 2
        @test length(mf.incident[3]) == 1
        # Node 1 is tail of e1 ⇒ σ_u = +1 (out-of-1 aligns with reference).
        inc = mf.incident[1][1]
        @test inc.e_idx == 1
        @test inc.v == 2
        @test inc.σ == 1
        @test inc.c_fwd == 1.0 && inc.c_rev == 1.0
        # Node 3 is head of e2 ⇒ σ_u = -1.
        inc3 = mf.incident[3][1]
        @test inc3.σ == -1
    end

    @testset "NonlinearGSRelaxer: zero potential ⇒ residual = -d, one step solves" begin
        # 3-node path with unit capacities and zero d (homogeneous).
        # Starting from x = 0, all g_e = 0, r = -d = 0 ⇒ no change.
        # If we start from a non-zero x, GS reduces the residual.
        B = sparse([-1.0  0.0;
                     1.0 -1.0;
                     0.0  1.0])
        mf = MaxFlowLaplacian(B, [1.0, 1.0], [1.0, 1.0], zeros(3))
        rx = NonlinearGSRelaxer(mf)
        # Start from a "bumpy" x; check that GS reduces the L-norm.
        L = laplacian(sparse([0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0]))
        x = [1.0, -0.5, 0.3]
        r0 = norm(L * x)
        relax!(rx, x, zeros(3); sweeps = 20)
        @test norm(L * x) < r0
    end

    @testset "NonlinearGSRelaxer: linear regime matches linear solve (small d)" begin
        # With c⁺ = c⁻ = 1 and small enough d that the induced flows stay
        # well below capacity, the nonlinear sweep should reproduce the
        # linear Laplacian solve. Here d induces flow ~0.03 — barrier inactive.
        B = sparse([-1.0  0.0;
                     1.0 -1.0;
                     0.0  1.0])
        c⁺ = [1.0, 1.0]; c⁻ = [1.0, 1.0]
        L_lin = 0.5 * Matrix(B * B')        # W(0) = 0.5 ⇒ L_lin = 0.5 · BBᵀ
        d = [-0.03, 0.01, 0.02]              # small RHS, flows ≪ capacity
        mf = MaxFlowLaplacian(B, c⁺, c⁻, d)
        rx = NonlinearGSRelaxer(mf)
        x = zeros(3)
        for _ in 1:200
            relax!(rx, x, d; sweeps = 1)
        end
        L_aug = L_lin + 1e-10 * I
        x_lin = L_aug \ d
        x .-= sum(x) / 3
        x_lin .-= sum(x_lin) / 3
        @test norm(x - x_lin) < 1e-3
    end

    @testset "NonlinearGSRelaxer: nonlinear regime satisfies Bf = −d, Ohm" begin
        # With moderate d, edges saturate. Verify the converged state
        # satisfies conservation and nonlinear Ohm's law per edge.
        B = sparse([-1.0  0.0;
                     1.0 -1.0;
                     0.0  1.0])
        c⁺ = [1.0, 1.0]; c⁻ = [1.0, 1.0]
        d = [-0.6, 0.2, 0.4]                 # induces ~60% saturation
        mf = MaxFlowLaplacian(B, c⁺, c⁻, d)
        rx = NonlinearGSRelaxer(mf)
        x = zeros(3)
        for _ in 1:100
            relax!(rx, x, d; sweeps = 1)
        end
        # 1. Conservation: B · f = −d.
        @test B * mf.f ≈ -d atol = 1e-6
        # 2. Ohm: for each edge, f · R(f) = φ_tail − φ_head.
        for e in 1:length(mf.f)
            f = mf.f[e]
            R = edge_resistance(c⁺[e], c⁻[e], f)
            # Find head and tail from B.
            tail = head = 0
            for k in nzrange(B, e)
                if nonzeros(B)[k] < 0
                    tail = rowvals(B)[k]
                else
                    head = rowvals(B)[k]
                end
            end
            @test f * R ≈ x[tail] - x[head] atol = 1e-6
        end
    end
end
