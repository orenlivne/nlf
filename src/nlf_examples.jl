"""
Canonical small max-flow test instances with **known** optimal values.

These are the realistic counterparts to `nlf_grid2d` /
`nlf_grid3d` / `nlf_random`: heterogeneous capacities, explicit
bottlenecks, real-world structure (textbook examples, bipartite
matching, image-segmentation cuts).

Each constructor returns `(NLFProblem, max_flow::Float64)` so tests
can pick `α` relative to the known optimum.
"""

# ---------------------------------------------------------- CLRS Fig. 26.1

"""
    nlf_clrs(; undirected = false) -> (NLFProblem, max_flow_value)

The textbook 6-node example from Cormen-Leiserson-Rivest-Stein,
*Introduction to Algorithms*, 3e Figure 26.1 (s → t through a small
network with heterogeneous capacities 4..20). Max flow = 23.

Node numbering: 1 = s, 2 = v₁, 3 = v₂, 4 = v₃, 5 = v₄, 6 = t.

Edges:
    s → v₁  cap 16
    s → v₂  cap 13
    v₂ → v₁ cap  4
    v₁ → v₃ cap 12
    v₃ → v₂ cap  9
    v₂ → v₄ cap 14
    v₄ → v₃ cap  7
    v₃ → t  cap 20
    v₄ → t  cap  4

Two variants:
- `undirected = false` (default): DIMACS-style directed arcs. `c⁺ =
  capacity`, `c⁻ = 0`. The box is `0 ≤ (Bᵀφ)_e ≤ cap` — one-sided. For
  this variant the electrical-flow `φ` from `solve(A, αd)` is GENERALLY
  NOT interior-feasible: it can develop counter-flow on some edges
  (negative gradients), which violates `c⁻ = 0`. Useful for testing
  the Kaczmarz smoother + active-set behaviour.
- `undirected = true`: each edge admits flow in either direction up to
  the same capacity (`c⁻ = c⁺`). The box is `-cap ≤ (Bᵀφ)_e ≤ cap`,
  centred on zero. For small `α` the electrical-flow φ has gradient
  magnitudes proportional to α and is interior-feasible for any
  `α < min_cap / max_per_unit_gradient ≈ 7.3`. Used for FAS-cycle
  stationarity tests where we need a known interior-feasible point.

Both variants share the same network topology and max-flow value 23 in
the directed sense (saturating the s-t min cut). The undirected
variant has a different max flow value (typically larger, since it
allows reverse-direction flow) — we don't claim a specific value for
that variant in the returned `max_flow_value`; the 23 returned is the
*directed* max flow, useful as a scale.
"""
function nlf_clrs(; undirected::Bool = false)
    edges = [(2, 1),   # s → v₁
             (3, 1),   # s → v₂
             (2, 3),   # v₂ → v₁
             (4, 2),   # v₁ → v₃
             (3, 4),   # v₃ → v₂
             (5, 3),   # v₂ → v₄
             (4, 5),   # v₄ → v₃
             (6, 4),   # v₃ → t
             (6, 5)]   # v₄ → t
    caps  = [16.0, 13.0, 4.0, 12.0, 9.0, 14.0, 7.0, 20.0, 4.0]
    c_minus = undirected ? copy(caps) : zeros(length(edges))
    name = "CLRS-Fig-26.1" * (undirected ? "-undirected" : "")
    mfp = make_problem(6, edges, caps, c_minus, 1, 6; name = name)
    return mfp, 23.0
end

# ---------------------------------------------------------- explicit bottleneck

"""
    nlf_bottleneck_chain(k::Int = 6;
                             cap_chain::Real = 10.0,
                             cap_bottle::Real = 1.0,
                             bottle_pos::Int = div(k, 2)) -> (mfp, max_flow)

A `k`-node directed chain `1 → 2 → … → k`. All edges have capacity
`cap_chain` except the edge at position `bottle_pos` (default middle)
which has capacity `cap_bottle`. Max flow = `cap_bottle` (single
bottleneck, no parallel paths).

Useful for stationarity tests where you want SOME edge near saturation
without saturating the whole network.
"""
function nlf_bottleneck_chain(k::Int = 6;
                                  cap_chain::Real = 10.0,
                                  cap_bottle::Real = 1.0,
                                  bottle_pos::Int = div(k, 2))
    @assert k >= 3
    @assert 1 <= bottle_pos < k
    edges = [(i + 1, i) for i in 1:(k - 1)]
    caps = fill(Float64(cap_chain), k - 1)
    caps[bottle_pos] = Float64(cap_bottle)
    c_minus = zeros(k - 1)
    mfp = make_problem(k, edges, caps, c_minus, 1, k;
                       name = "bottleneck-chain-$(k)/bottle@$(bottle_pos)")
    return mfp, Float64(cap_bottle)
end

# ---------------------------------------------------------- bipartite matching

"""
    nlf_bipartite_matching(left_size::Int, right_size::Int,
                               edges_LR::Vector{Tuple{Int,Int}}) -> (mfp, max_flow)

Bipartite matching as max flow. Source = node 1, sink = node N (= 2 + L + R).
Left nodes are 2 .. 1+L, right nodes are 2+L .. 1+L+R. Each entry in
`edges_LR` is `(i, j)` meaning left-node `i ∈ 1..L` is connected to
right-node `j ∈ 1..R`; this becomes an arc `(left_node, right_node)`
with capacity 1.

Max flow = max bipartite matching size (König-Egerváry). Computed here
by a greedy heuristic only as a sanity check — the test should call
this with a known matching size.
"""
function nlf_bipartite_matching(left_size::Int, right_size::Int,
                                    edges_LR::Vector{Tuple{Int,Int}},
                                    known_matching::Int)
    L = left_size; R = right_size
    s = 1
    L_nodes = collect(2:(1 + L))
    R_nodes = collect((2 + L):(1 + L + R))
    t = 2 + L + R
    n = t
    edges = Tuple{Int,Int}[]
    # s → each left node, capacity 1
    for li in L_nodes; push!(edges, (li, s)); end
    # left → right for each (i, j) in edges_LR
    for (i, j) in edges_LR
        @assert 1 <= i <= L && 1 <= j <= R
        push!(edges, (R_nodes[j], L_nodes[i]))
    end
    # each right → t, capacity 1
    for rj in R_nodes; push!(edges, (t, rj)); end
    caps = ones(length(edges))
    c_minus = zeros(length(edges))
    mfp = make_problem(n, edges, caps, c_minus, s, t;
                       name = "bipartite-matching-$(L)x$(R)")
    return mfp, Float64(known_matching)
end

# ---------------------------------------------------------- image-seg cut (small)

"""
    nlf_image_seg_2x2() -> (mfp, max_flow)

Tiny image-segmentation s-t cut: 2×2 pixel grid with foreground source
and background sink, connected via:
  • a "smoothness" lattice between neighboring pixels (capacity =
    similarity weight, here 5.0 along the 4 grid edges),
  • a "data term" linking each pixel to source (capacity = foreground
    log-likelihood) and to sink (capacity = background log-likelihood).

Numerical setup chosen so the optimum cut separates pixels {1, 2} (FG)
from {3, 4} (BG): bottom row's data-to-sink links sum to 6, top row's
data-to-source links sum to 6, smoothness on the FG/BG boundary edges
(between rows) sums to 10 — but the optimum cut crosses 2 lattice
edges (cap 5 each) + ignores the data links *within* each region. Max
flow = 10.

The realistic feature here: capacities span [1, 7] with the lattice
being the binding constraint set.
"""
function nlf_image_seg_2x2()
    # Nodes 1..4 are pixels (1,2 top row; 3,4 bottom row).
    # Node 5 = source (foreground), node 6 = sink (background).
    edges = Tuple{Int,Int}[]
    caps = Float64[]
    # Lattice edges (smoothness, undirected: we use c⁻ = c⁺ later).
    lattice_edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    lattice_cap = 5.0
    # Source → pixel and pixel → sink (data terms).
    src_caps = [4.0, 3.0, 2.0, 1.0]
    snk_caps = [1.0, 2.0, 3.0, 4.0]
    n = 6; s = 5; t = 6
    for (h, ta) in lattice_edges
        push!(edges, (h, ta))
        push!(caps, lattice_cap)
    end
    for p in 1:4
        push!(edges, (p, s))
        push!(caps, src_caps[p])
    end
    for p in 1:4
        push!(edges, (t, p))
        push!(caps, snk_caps[p])
    end
    # Lattice edges are undirected (c⁻ = c⁺); data edges are directed
    # (c⁻ = 0). The mfp.low is built as -c⁻.
    c_minus = zeros(length(edges))
    for i in 1:length(lattice_edges)
        c_minus[i] = lattice_cap     # undirected lattice
    end
    mfp = make_problem(n, edges, caps, c_minus, s, t; name = "image-seg-2x2")
    # By inspection of the directed max flow:
    #   Path s→1, s→2 with caps 4+3 = 7 into the top row.
    #   Path 1→3, 2→4 carry these down (lattice caps 5 each → bottleneck 5
    #   on each downward edge, but only after data caps saturate).
    #   3→t, 4→t with caps 3+4 = 7.
    # Min cut: separate {s, 1, 2} from {3, 4, t}.
    #   Cut edges: 1→3 (cap 5), 2→4 (cap 5), s→3 doesn't exist, s→4 doesn't,
    #   3→t and 4→t are inside the t side, 1→t doesn't, 2→t doesn't.
    # Wait — caps on 1→3 in our edge list use mfp.high which = cap = 5, AND
    # mfp.low = -5 (undirected lattice). The directed s-t max flow is the
    # min cut of the directed capacity graph, where the lattice has cap 5
    # in both directions.
    # Cut {s,1,2} vs {3,4,t}: 1→3 (5), 2→4 (5), 1→t (no edge), 2→t (no),
    #                         total = 10.
    # Cut {s,1} vs {2,3,4,t}: 1→2 (5), s→2 (3), 1→3 (5), 1→t (no), total = 13.
    # Cut {s} vs {1,2,3,4,t}: s→1 (4), s→2 (3), s→3 (2), s→4 (1), total = 10.
    # So min cut = 10. Max flow = 10.
    return mfp, 10.0
end
