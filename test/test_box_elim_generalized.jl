"""
test_box_elim_generalized.jl — hand-computed unit tests for the
exact-interpolation box coarsening at ELIM levels (LAMG §3.3 Algorithm 2).

We construct EliminationStage instances BY HAND so the math is decoupled
from the low_degree_nodes selection heuristic.
"""

using Test
using NLF
using LinearAlgebra
using SparseArrays
import NLF: low_degree_nodes, eliminate_once, coarsen_box_elim,
             GeneralizedConstraint, EliminationStage

# ── Test 1: 4-cycle 1-2-3-4-1. Manually set F = {1, 3}, C = {2, 4}. ──────
#
# Edges (in pre-elim numbering): e1=(1,2), e2=(2,3), e3=(3,4), e4=(4,1)
#
# A_ff = diag(2, 2)  (degree-2 nodes)
# A_fc = [-1 -1; -1 -1]   (each F node connected to both C nodes via cycle)
# P    = -A_ff^{-1} A_fc = [0.5 0.5; 0.5 0.5]
# q    = [0.5, 0.5]
#
# x_F[1] = x_1 = 0.5 x_C[1] + 0.5 x_C[2] + 0.5 b_F[1]    (x_C[1]=x_2, x_C[2]=x_4)
# x_F[2] = x_3 = 0.5 x_C[1] + 0.5 x_C[2] + 0.5 b_F[2]
#
# e1 = (h=1, t=2): F-C, h_F=1, t_C=1
#   x_1 - x_2 = -0.5 x_C[1] + 0.5 x_C[2] + 0.5 b_F[1]
#   ⇒ cols=[1,2], vals=[-0.5, +0.5], b_coeff=+0.5, b_idx=1
# e2 = (h=2, t=3): C-F, h_C=1, t_F=2
#   x_2 - x_3 = +0.5 x_C[1] - 0.5 x_C[2] - 0.5 b_F[2]
#   ⇒ cols=[1,2], vals=[+0.5, -0.5], b_coeff=-0.5, b_idx=2
# e3 = (h=3, t=4): F-C, h_F=2, t_C=2
#   x_3 - x_4 = +0.5 x_C[1] - 0.5 x_C[2] + 0.5 b_F[2]
#   ⇒ cols=[1,2], vals=[+0.5, -0.5], b_coeff=+0.5, b_idx=2
# e4 = (h=4, t=1): C-F, h_C=2, t_F=1
#   x_4 - x_1 = -0.5 x_C[1] + 0.5 x_C[2] - 0.5 b_F[1]
#   ⇒ cols=[1,2], vals=[-0.5, +0.5], b_coeff=-0.5, b_idx=1

@testset "coarsen_box_elim: 4-cycle hand-built stage" begin
    A = sparse([ 2.0 -1.0  0.0 -1.0;
                -1.0  2.0 -1.0  0.0;
                 0.0 -1.0  2.0 -1.0;
                -1.0  0.0 -1.0  2.0])
    # Build the stage manually so the test is independent of the
    # low_degree_nodes greedy selection.
    P_dense = [0.5 0.5;
               0.5 0.5]
    stage = EliminationStage([1, 3], [2, 4], 4,
                              sparse(P_dense),
                              sparse(P_dense'),       # R = Pᵀ since A_ff is diagonal
                              [0.5, 0.5])

    head_f = [1, 2, 3, 4]
    tail_f = [2, 3, 4, 1]
    low_f  = [-1.0, -1.0, -1.0, -1.0]
    high_f = [+1.0, +1.0, +1.0, +1.0]

    low_c, high_c, head_c, tail_c, gen =
        coarsen_box_elim(low_f, high_f, head_f, tail_f, stage, A)

    @test isempty(head_c)
    @test length(gen) == 4

    # Edge e1 = (1,2): F-C
    g = gen[1]
    @test g.cols == [1, 2]
    @test g.vals ≈ [-0.5, +0.5]
    @test g.b_correction_coeff ≈ +0.5
    @test g.b_correction_index == 1
    @test g.low_static  == -1.0
    @test g.high_static == +1.0
    @test (g.fine_head, g.fine_tail) == (1, 2)

    # Edge e2 = (2,3): C-F
    g = gen[2]
    @test g.cols == [1, 2]
    @test g.vals ≈ [+0.5, -0.5]
    @test g.b_correction_coeff ≈ -0.5
    @test g.b_correction_index == 2
    @test (g.fine_head, g.fine_tail) == (2, 3)

    # Edge e3 = (3,4): F-C
    g = gen[3]
    @test g.cols == [1, 2]
    @test g.vals ≈ [+0.5, -0.5]
    @test g.b_correction_coeff ≈ +0.5
    @test g.b_correction_index == 2
    @test (g.fine_head, g.fine_tail) == (3, 4)

    # Edge e4 = (4,1): C-F
    g = gen[4]
    @test g.cols == [1, 2]
    @test g.vals ≈ [-0.5, +0.5]
    @test g.b_correction_coeff ≈ -0.5
    @test g.b_correction_index == 1
    @test (g.fine_head, g.fine_tail) == (4, 1)
end

# ── Test 2: degree-3 F node, C-C survivor edge ─────────────────────────
#
# Graph:    1
#           |
#       2 - 3 - 4
#           |
#           5
# Edges: (1,3), (2,3), (3,4), (3,5), (2,4) — last one is C-C survivor
# F = {3}, C = {1, 2, 4, 5}
#
# A_ff = [4] (degree 4), q = [0.25]
# A_fc[1, :] = entries A[3, 1], A[3, 2], A[3, 4], A[3, 5] = -1, -1, -1, -1
# P = -A_ff^{-1} A_fc = -(0.25) * [-1, -1, -1, -1] = [0.25, 0.25, 0.25, 0.25]
#
# x_3 = 0.25(x_C[1] + x_C[2] + x_C[3] + x_C[4]) + 0.25 b_F[1]
#       where C = [1, 2, 4, 5], so C-local indices map to nodes 1, 2, 4, 5.
#
# Edge (1,3): C-F. h=1 ∈ C (C-local 1), t=3 ∈ F (F-local 1)
#   x_1 - x_3 = x_C[1] - 0.25(x_C[1]+x_C[2]+x_C[3]+x_C[4]) - 0.25 b_F[1]
#             = 0.75 x_C[1] - 0.25 x_C[2] - 0.25 x_C[3] - 0.25 x_C[4] - 0.25 b_F[1]
#   ⇒ cols=[1,2,3,4], vals=[0.75, -0.25, -0.25, -0.25], b_coeff=-0.25, b_idx=1
#
# Edge (2,4): C-C. Both ∈ C, re-indexed: 2→2, 4→3 (since C=[1,2,4,5])

@testset "coarsen_box_elim: degree-3 F + C-C survivor" begin
    # Build a Laplacian where node 3 has degree 4 (edges to 1,2,4,5)
    # and there is a C-C edge (2,4).
    A = spzeros(5, 5)
    for (i, j) in [(1,3), (2,3), (3,4), (3,5), (2,4)]
        A[i, j] -= 1.0; A[j, i] -= 1.0
        A[i, i] += 1.0; A[j, j] += 1.0
    end

    stage = EliminationStage([3], [1, 2, 4, 5], 5,
                              sparse([0.25 0.25 0.25 0.25]),    # P (1×4)
                              sparse([0.25; 0.25; 0.25; 0.25;;]),# R = Pᵀ (4×1)
                              [0.25])

    # 5 edges total; (2,4) is C-C, others are F-incident.
    head_f = [1, 2, 3, 3, 2]
    tail_f = [3, 3, 4, 5, 4]
    low_f  = [-1.0, -1.0, -1.0, -1.0, -2.5]
    high_f = [+1.0, +1.0, +1.0, +1.0, +3.5]

    low_c, high_c, head_c, tail_c, gen =
        coarsen_box_elim(low_f, high_f, head_f, tail_f, stage, A)

    # C-C edge (2,4) re-indexed: node 2 → C-local 2, node 4 → C-local 3
    @test head_c == [2]
    @test tail_c == [3]
    @test low_c  == [-2.5]
    @test high_c == [+3.5]
    @test length(gen) == 4

    # Edge (1,3): C-F, h_C=1
    g = gen[1]
    @test g.cols == [1, 2, 3, 4]
    @test g.vals ≈ [+0.75, -0.25, -0.25, -0.25]
    @test g.b_correction_coeff ≈ -0.25
    @test g.b_correction_index == 1

    # Edge (2,3): C-F, h_C=2
    g = gen[2]
    @test g.cols == [1, 2, 3, 4]
    @test g.vals ≈ [-0.25, +0.75, -0.25, -0.25]
    @test g.b_correction_coeff ≈ -0.25

    # Edge (3,4): F-C, t_C=3
    g = gen[3]
    @test g.cols == [1, 2, 3, 4]
    @test g.vals ≈ [+0.25, +0.25, -0.75, +0.25]
    @test g.b_correction_coeff ≈ +0.25

    # Edge (3,5): F-C, t_C=4
    g = gen[4]
    @test g.cols == [1, 2, 3, 4]
    @test g.vals ≈ [+0.25, +0.25, +0.25, -0.75]
    @test g.b_correction_coeff ≈ +0.25
end
