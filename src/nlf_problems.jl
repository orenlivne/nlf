"""
Max-flow problem data structures + generators + DIMACS loader.

We use the **constrained-linear** formulation (NOT the barrier one):

    max α
    s.t.  A φ = α d
          low_e ≤ (Bᵀφ)_e ≤ high_e   for every edge e

where:
- A     :: weighted graph Laplacian, *constant* (no f-dependence)
- B     :: signed incidence matrix (n × m); B[u,e] = +1 if u head, -1 if tail
- φ     :: node potentials (unknown)
- d     :: source/sink vector (+1 at source, -1 at sink, 0 elsewhere)
- α     :: flow value scalar (global Lagrange-style parameter)
- low_e :: -c⁻_e (reverse-direction capacity, negative)
- high_e:: +c⁺_e (forward-direction capacity, positive)

Flow on edge e (in reference orientation tail→head):
    f_e = W_e · (Bᵀφ)_e
and the constraint `low_e ≤ Bᵀφ_e ≤ high_e` becomes `|f_e| ≤ c_e` when W=I.

For unit weights, A = BᵀB.
"""

"""
    NLFProblem

A single max-flow instance.

Fields:
- `A`       :: n × n graph Laplacian (=BᵀB for unit weights).
- `B`       :: n × m signed incidence matrix.
- `low`     :: m-vector of lower bounds on (Bᵀφ)_e  (typically -c⁻_e).
- `high`    :: m-vector of upper bounds on (Bᵀφ)_e  (typically +c⁺_e).
- `d`       :: n-vector source/sink (zero-sum).
- `s, t`    :: source and sink node indices (so d = e_t − e_s).
- `name`    :: human-readable identifier.
- `endpoints` :: m×2 matrix of (head, tail) per edge — cached for fast loops.
"""
struct NLFProblem
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    low::Vector{Float64}
    high::Vector{Float64}
    d::Vector{Float64}
    s::Int
    t::Int
    name::String
    head::Vector{Int}      # head[e] = node u with B[u,e] = +1
    tail::Vector{Int}      # tail[e] = node u with B[u,e] = -1
end

"""
    incidence_from_edge_list(n, edges) -> SparseMatrixCSC

Build the n × m signed incidence matrix from a list of (head, tail) pairs.
Convention: B[head, e] = +1, B[tail, e] = -1.
"""
function incidence_from_edge_list(n::Int, edges::Vector{Tuple{Int,Int}})
    m = length(edges)
    rows = Vector{Int}(undef, 2m)
    cols = Vector{Int}(undef, 2m)
    vals = Vector{Float64}(undef, 2m)
    @inbounds for (e, (head, tail)) in enumerate(edges)
        rows[2e - 1] = head; cols[2e - 1] = e; vals[2e - 1] = +1.0
        rows[2e]     = tail; cols[2e]     = e; vals[2e]     = -1.0
    end
    return sparse(rows, cols, vals, n, m)
end

"""
    make_problem(n, edges, c_plus, c_minus, s, t; name="")

Build a NLFProblem from a node count + an oriented edge list +
per-edge forward/reverse capacities + source/sink indices.
"""
function make_problem(n::Int, edges::Vector{Tuple{Int,Int}},
                      c_plus::AbstractVector, c_minus::AbstractVector,
                      s::Int, t::Int; name::AbstractString = "")
    m = length(edges)
    @assert length(c_plus) == m
    @assert length(c_minus) == m
    @assert s != t && 1 <= s <= n && 1 <= t <= n
    B = incidence_from_edge_list(n, edges)
    head = [e[1] for e in edges]
    tail = [e[2] for e in edges]
    A = B * B'
    d = zeros(n); d[t] = 1.0; d[s] = -1.0
    low = -collect(Float64.(c_minus))
    high = collect(Float64.(c_plus))
    return NLFProblem(A, B, low, high, d, s, t, String(name), head, tail)
end

# ---------------------------------------------------------- Synthetic generators

"""
    nlf_grid2d(k::Int; cap_lo=0.5, cap_hi=2.0, rng=...) -> NLFProblem

k × k 2D grid with edges in both axis directions. Capacities random uniform
in [cap_lo, cap_hi]. Source = corner (1,1), sink = opposite corner (k,k).
"""
function nlf_grid2d(k::Int; cap_lo::Real = 0.5, cap_hi::Real = 2.0,
                       rng = Random.default_rng())
    @assert k >= 2
    n = k * k
    idx(i, j) = (j - 1) * k + i
    edges = Tuple{Int,Int}[]
    for j in 1:k, i in 1:k
        if i < k
            push!(edges, (idx(i + 1, j), idx(i, j)))    # horizontal: tail i, head i+1
        end
        if j < k
            push!(edges, (idx(i, j + 1), idx(i, j)))
        end
    end
    m = length(edges)
    c_plus  = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    c_minus = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    s, t = idx(1, 1), idx(k, k)
    return make_problem(n, edges, c_plus, c_minus, s, t;
                        name = "grid2d/$(k)x$(k)")
end

"""
    nlf_grid3d(k::Int; cap_lo, cap_hi, rng) -> NLFProblem
"""
function nlf_grid3d(k::Int; cap_lo::Real = 0.5, cap_hi::Real = 2.0,
                       rng = Random.default_rng())
    @assert k >= 2
    n = k * k * k
    idx(i, j, l) = ((l - 1) * k + (j - 1)) * k + i
    edges = Tuple{Int,Int}[]
    for l in 1:k, j in 1:k, i in 1:k
        if i < k; push!(edges, (idx(i + 1, j, l), idx(i, j, l))); end
        if j < k; push!(edges, (idx(i, j + 1, l), idx(i, j, l))); end
        if l < k; push!(edges, (idx(i, j, l + 1), idx(i, j, l))); end
    end
    m = length(edges)
    c_plus  = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    c_minus = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    s, t = idx(1, 1, 1), idx(k, k, k)
    return make_problem(n, edges, c_plus, c_minus, s, t;
                        name = "grid3d/$(k)³")
end

"""
    nlf_random(n::Int, p::Real; cap_lo, cap_hi, rng) -> NLFProblem

Erdős–Rényi connected graph + random capacities. Source/sink picked at
opposite ends of a longest-shortest-path approximation (just nodes 1 and n).
"""
function nlf_random(n::Int, p::Real; cap_lo::Real = 0.5, cap_hi::Real = 2.0,
                       rng = Random.default_rng())
    edges = Tuple{Int,Int}[]
    for i in 1:n, j in (i + 1):n
        if rand(rng) < p
            push!(edges, (j, i))   # head j, tail i (arbitrary orientation)
        end
    end
    m = length(edges)
    @assert m >= n - 1 "draw is too sparse to be connected; raise p"
    c_plus  = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    c_minus = cap_lo .+ (cap_hi - cap_lo) .* rand(rng, m)
    return make_problem(n, edges, c_plus, c_minus, 1, n;
                        name = "rand/$(n)_p$(p)")
end

# ---------------------------------------------------------- Goldberg–Cherkassky generators

"""
    nlf_genrmf(a1::Int, a2::Int, b::Int;
                   cap_high::Real = 100.0, cap_low::Real = 1.0,
                   rng = Random.default_rng()) -> NLFProblem

Cherkassky–Goldberg `genrmf` benchmark family (1st DIMACS Challenge,
1991). A 3D grid of `a1 × a2 × b` nodes:

* `b` "frames" each of size `a1 × a2`.
* Within each frame: full 2D grid edges with **high** capacity `cap_high`.
* Between consecutive frames: every node is connected to one *random*
  node in the next frame, with **low** capacity `cap_low`.
* Source = (1,1,1), sink = (a1,a2,b).

This network has max flow ≈ `cap_low × a1 × a2` (each inter-frame
slice is the bottleneck), independent of `b`. Standard hard instance:
many cycles needed but graph has clean geometric structure.

`nlf_genrmf(4, 4, 8)` = 128 nodes, ~200 edges.
`nlf_genrmf(8, 8, 16)` = 1024 nodes, ~1900 edges.
`nlf_genrmf(16, 16, 32)` = 8192 nodes, ~16000 edges.
"""
function nlf_genrmf(a1::Int, a2::Int, b::Int;
                        cap_high::Real = 100.0, cap_low::Real = 1.0,
                        rng = Random.default_rng())
    @assert a1 >= 2 && a2 >= 2 && b >= 2
    n = a1 * a2 * b
    idx(i, j, k) = ((k - 1) * a2 + (j - 1)) * a1 + i
    edges = Tuple{Int,Int}[]; caps = Float64[]
    # In-frame grid edges with cap_high.
    for k in 1:b
        for j in 1:a2, i in 1:a1
            i < a1 && (push!(edges, (idx(i + 1, j, k), idx(i, j, k))); push!(caps, cap_high))
            j < a2 && (push!(edges, (idx(i, j + 1, k), idx(i, j, k))); push!(caps, cap_high))
        end
    end
    # Inter-frame random edges with cap_low — each frame-k node maps to
    # a random node in frame k+1.
    for k in 1:(b - 1)
        # Random permutation of frame-(k+1) nodes.
        perm = randperm(rng, a1 * a2)
        for j in 1:a2, i in 1:a1
            u = idx(i, j, k)
            v_idx = perm[(j - 1) * a1 + i]
            v_j = ((v_idx - 1) ÷ a1) + 1
            v_i = ((v_idx - 1) % a1) + 1
            v = idx(v_i, v_j, k + 1)
            push!(edges, (v, u))
            push!(caps, cap_low)
        end
    end
    m = length(edges)
    c_minus = zeros(m)        # directed network
    s = idx(1, 1, 1); t = idx(a1, a2, b)
    return make_problem(n, edges, caps, c_minus, s, t;
                        name = "genrmf/$(a1)x$(a2)x$(b)")
end

"""
    nlf_washington(rows::Int, cols::Int;
                       cap_min::Real = 1.0, cap_max::Real = 1000.0,
                       rng = Random.default_rng()) -> NLFProblem

Washington random-level-graph (RLG) — another DIMACS-classical
max-flow benchmark family. Layered s → layer-1 → layer-2 → … → t,
each layer with `rows` nodes, total `cols` middle layers. Capacities
random log-uniform in `[cap_min, cap_max]`.

Within each layer: all-to-all to next layer's nodes (dense).
Source → first layer, last layer → sink.

Total: n = 2 + rows × cols nodes; m = rows + rows + rows²(cols − 1) edges.
`nlf_washington(8, 4)` = 34 nodes, ~200 edges.
"""
function nlf_washington(rows::Int, cols::Int;
                            cap_min::Real = 1.0, cap_max::Real = 1000.0,
                            rng = Random.default_rng())
    @assert rows >= 2 && cols >= 2
    s = 1; t = 2 + rows * cols
    # Layer-k node r: index 2 + (k-1)*rows + r, for k=1..cols, r=1..rows.
    layer_node(k, r) = 2 + (k - 1) * rows + r
    n = t
    edges = Tuple{Int,Int}[]; caps = Float64[]
    function rand_cap()
        # Log-uniform draw in [cap_min, cap_max].
        return exp(log(cap_min) + rand(rng) * (log(cap_max) - log(cap_min)))
    end
    # s → layer 1.
    for r in 1:rows
        push!(edges, (layer_node(1, r), s))
        push!(caps, rand_cap())
    end
    # Between layers k, k+1: all-to-all (dense).
    for k in 1:(cols - 1)
        for r1 in 1:rows, r2 in 1:rows
            push!(edges, (layer_node(k + 1, r2), layer_node(k, r1)))
            push!(caps, rand_cap())
        end
    end
    # Last layer → t.
    for r in 1:rows
        push!(edges, (t, layer_node(cols, r)))
        push!(caps, rand_cap())
    end
    m = length(edges)
    c_minus = zeros(m)
    return make_problem(n, edges, caps, c_minus, s, t;
                        name = "washington/$(rows)x$(cols)")
end

"""
    nlf_acyclic_dense(n::Int, p::Real;
                         cap_min::Real = 1.0, cap_max::Real = 100.0,
                         rng = Random.default_rng()) -> NLFProblem

Random acyclic dense graph: every pair (i, j) with i < j has an arc
with probability `p` and random capacity. Source = 1, sink = n.
"""
function nlf_acyclic_dense(n::Int, p::Real;
                               cap_min::Real = 1.0, cap_max::Real = 100.0,
                               rng = Random.default_rng())
    @assert n >= 3 && 0 < p <= 1
    edges = Tuple{Int,Int}[]; caps = Float64[]
    for i in 1:n, j in (i + 1):n
        if rand(rng) < p
            push!(edges, (j, i))   # head=j, tail=i (directed i→j)
            push!(caps, cap_min + (cap_max - cap_min) * rand(rng))
        end
    end
    # Force at least one s-t path.
    if isempty(edges) || !any(e -> e[1] == n, edges)
        for i in 1:(n - 1)
            push!(edges, (i + 1, i)); push!(caps, cap_max)
        end
    end
    m = length(edges)
    c_minus = zeros(m)
    return make_problem(n, edges, caps, c_minus, 1, n;
                        name = "acyclic-dense/$(n)_p$(p)")
end

# ---------------------------------------------------------- DIMACS .max loader

"""
    load_dimacs_max(path::AbstractString) -> NLFProblem

Parse a DIMACS max-flow (.max) format file. Lines:
  c     comment
  p max <n> <m>      problem header
  n <i> s            mark node i as source
  n <i> t            mark node i as sink
  a <u> <v> <cap>    edge from u to v with capacity cap

For our undirected formulation: each DIMACS arc (u, v, cap) sets forward
capacity c⁺ = cap on the (u → v) reference orientation, and reverse
capacity c⁻ = 0. If a reverse arc (v, u, cap2) appears, it sets c⁻ on the
same edge.
"""
function load_dimacs_max(path::AbstractString)
    n = 0; m_decl = 0; s = 0; t = 0
    fwd = Dict{Tuple{Int,Int}, Float64}()   # (head, tail) -> c⁺ on that orientation
    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            tag = line[1]
            if tag == 'c'
                continue
            elseif tag == 'p'
                parts = split(line)
                @assert length(parts) >= 4
                n = parse(Int, parts[3])
                m_decl = parse(Int, parts[4])
            elseif tag == 'n'
                parts = split(line)
                @assert length(parts) >= 3
                node = parse(Int, parts[2])
                if parts[3] == "s"
                    s = node
                elseif parts[3] == "t"
                    t = node
                end
            elseif tag == 'a'
                parts = split(line)
                @assert length(parts) >= 4
                u = parse(Int, parts[2])
                v = parse(Int, parts[3])
                cap = parse(Float64, parts[4])
                # Reference orientation: head = v, tail = u (DIMACS arc u→v).
                fwd[(v, u)] = get(fwd, (v, u), 0.0) + cap
            end
        end
    end
    # Merge oppositely-oriented arcs into a single undirected edge with
    # asymmetric c⁺ / c⁻.
    edges = Tuple{Int,Int}[]; c_plus = Float64[]; c_minus = Float64[]
    seen = Set{Tuple{Int,Int}}()
    for ((h, ta), cap) in fwd
        canon = h < ta ? (h, ta) : (ta, h)
        canon ∈ seen && continue
        push!(seen, canon)
        rev_cap = get(fwd, (ta, h), 0.0)
        # Choose head = h (the original direction's downstream).
        push!(edges, (h, ta))
        push!(c_plus, cap)
        push!(c_minus, rev_cap)
    end
    return make_problem(n, edges, c_plus, c_minus, s, t;
                        name = "dimacs/" * basename(path))
end
