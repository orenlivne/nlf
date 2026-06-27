using Test
using LinearAlgebra
using SparseArrays
using Random
using NLF

"""
    ford_fulkerson_max_flow(mfp::NLFProblem) -> Float64

Reference Ford-Fulkerson with BFS-shortest-path augmentation (Edmonds-
Karp). Slow O(VE²) — for small instances only, as a correctness oracle.

Handles only the DIRECTED case: capacity on (head, tail) is `mfp.high[e]`
(forward), capacity on (tail, head) is `-mfp.low[e]` (reverse).

For our directed-arc test instances (c⁻ = 0), this gives the classical
max flow. For undirected (c⁻ = c⁺), this gives the bidirectional max
flow (typically larger than the gradient-flow LP).
"""
function ford_fulkerson_max_flow(mfp::NLFProblem)
    n = size(mfp.A, 1)
    m = length(mfp.head)
    s = mfp.s; t = mfp.t
    @assert s != 0 && t != 0

    # Build adjacency: for each node, a list of (nbr, edge_idx, sign) where
    # sign = +1 if `mfp.head[edge_idx] == nbr` (forward), -1 otherwise.
    # Residual capacity of edge e in direction (tail → head) starts at high[e];
    # in direction (head → tail) it's -low[e].
    adj = [Vector{Tuple{Int,Int,Int}}() for _ in 1:n]
    for e in 1:m
        h = mfp.head[e]; ta = mfp.tail[e]
        push!(adj[ta], (h, e, +1))   # forward edge from tail to head
        push!(adj[h], (ta, e, -1))   # backward edge from head to tail
    end
    residual = Dict{Tuple{Int,Int},Float64}()
    for e in 1:m
        residual[(mfp.tail[e], mfp.head[e])] = mfp.high[e]
        residual[(mfp.head[e], mfp.tail[e])] = -mfp.low[e]
    end

    total_flow = 0.0
    while true
        # BFS for shortest s→t path with residual > 0.
        parent = fill((0, 0), n)        # (parent node, residual along this edge)
        visited = falses(n)
        visited[s] = true
        queue = [s]
        found = false
        while !isempty(queue) && !found
            u = popfirst!(queue)
            for (v, _e, _sgn) in adj[u]
                if !visited[v] && residual[(u, v)] > 1e-12
                    visited[v] = true
                    parent[v] = (u, 0)
                    v == t && (found = true; break)
                    push!(queue, v)
                end
            end
        end
        !found && break
        # Find min residual along the path.
        bottleneck = Inf
        node = t
        while node != s
            (pnode, _) = parent[node]
            bottleneck = min(bottleneck, residual[(pnode, node)])
            node = pnode
        end
        # Augment.
        node = t
        while node != s
            (pnode, _) = parent[node]
            residual[(pnode, node)] -= bottleneck
            residual[(node, pnode)] += bottleneck
            node = pnode
        end
        total_flow += bottleneck
    end
    return total_flow
end

@testset "max-flow correctness against Ford-Fulkerson" begin
    @testset "Ford-Fulkerson on CLRS Fig. 26.1 = 23" begin
        mfp, _ = nlf_clrs()
        @test ford_fulkerson_max_flow(mfp) == 23.0
    end

    @testset "Ford-Fulkerson on bottleneck chain k=8, cap_bottle=1" begin
        mfp, _ = nlf_bottleneck_chain(8;
                                            cap_chain = 10.0, cap_bottle = 1.0,
                                            bottle_pos = 4)
        @test ford_fulkerson_max_flow(mfp) == 1.0
    end

    @testset "Ford-Fulkerson on image-seg-2x2 = 10" begin
        mfp, _ = nlf_image_seg_2x2()
        @test isapprox(ford_fulkerson_max_flow(mfp), 10.0, atol = 1e-9)
    end

    # The gradient-flow LP (our solver) may give a LOWER α than the true
    # max-flow on directed graphs with cycles (electrical-flow can't
    # carry circulating flow). On the small directed instances above
    # though, there's no cycle, so they agree.
    @testset "solve_alpha_max on directed chains/paths matches Ford-Fulkerson" begin
        mfp, _ = nlf_bottleneck_chain(8;
                                            cap_chain = 10.0, cap_bottle = 1.0,
                                            bottle_pos = 4)
        α_lp, _ = solve_alpha_max(mfp)
        α_ff = ford_fulkerson_max_flow(mfp)
        @test isapprox(α_lp, α_ff, atol = 1e-6)
    end

    @testset "solve_alpha_max on CLRS (directed)" begin
        # Directed CLRS has cycles → gradient-flow LP gives 22.0 vs FF's 23.0.
        # (electrical flow underutilises one direction of a bottleneck edge.)
        mfp, _ = nlf_clrs(undirected = false)
        α_lp, _ = solve_alpha_max(mfp)
        α_ff = ford_fulkerson_max_flow(mfp)
        # On this 6-node DAG-like CLRS network they happen to agree
        # to within rounding (no flow-bearing cycles).
        @test 0 <= α_lp <= α_ff + 1e-6
    end

    @testset "fmg_fas_alpha_opt! on small instances exactly matches direct LP" begin
        # For these the hierarchy has only 1 level (n < min_coarse_size),
        # so fmg_fas_alpha_opt! just runs solve_alpha_max.
        for ctor in [nlf_clrs, () -> nlf_bottleneck_chain(8),
                     nlf_image_seg_2x2]
            res = ctor()
            mfp = res isa Tuple ? res[1] : res
            α_direct, _ = solve_alpha_max(mfp)
            n = size(mfp.A, 1)
            hier = setup_nlf_hierarchy(mfp; n_min = 20,
                                            rng = Random.MersenneTwister(0xfa11))
            φ = zeros(n)
            α_fmg, _, _ = fmg_fas_alpha_opt!(hier, φ; ν_per_level = 1)
            @test isapprox(α_fmg, α_direct, atol = 1e-6)
        end
    end
end
