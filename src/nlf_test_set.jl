"""
Comprehensive max-flow test set — analogous to the LAMG paper's
1,666-graph benchmark for the linear solver, this module assembles
a tiered collection of max-flow instances spanning canonical
textbook cases, synthetic scaling families, and real-world DIMACS
.max benchmark files (vision graph-cuts).

The four tiers:

  Tier 1: **canonical** (≤ 20 nodes, known max-flow value)
    - CLRS Fig. 26.1 (directed and undirected)
    - bottleneck chain, image-seg 2x2, bipartite matching
    - small textbook examples for unit-test sanity

  Tier 2: **synthetic generators** (parameterised scaling families)
    - grid2d at k = 8, 16, 32, 64, 128 (4-neighbor lattice)
    - grid3d at k = 4, 8, 16, 24 (7-point stencil)
    - genrmf (Goldberg–Cherkassky DIMACS-1 classical)
    - washington layered random
    - acyclic dense, random Erdős-Rényi

  Tier 3: **real DIMACS .max files** (Boykov–Kolmogorov vision)
    - BVZ-tsukuba / -venus / -sawtooth stereo
    - KZ2-tsukuba / -venus / -sawtooth stereo (Kolmogorov-Zabih)
    - BL06-camel/gargoyle 3D multiview reconstruction
    - LB07-bunny shape fitting
    - {adhead, babyface, bone, liver} 3D medical segmentation

  Tier 4: **SuiteSparse-derived** (LAMG paper's 2,031 Laplacians
            with synthetic capacities + farthest-node source/sink)
    - Reuses existing data/*.mtx; assigns log-uniform caps
    - Lets us test the max-flow algorithm on the full 2,031-graph
      mix at a marginal extra cost over the LAMG linear-solver bench.

`build_test_set(; tiers, sizes)` returns a `Vector{TestCase}` ready
for the max-flow benchmark (`scripts/maxflow_benchmark.jl`).

See `doc/maxflow_test_set.md` for the full table and provenance.
"""

# ---------------------------------------------------------- TestCase struct

"""
    MaxFlowTestCase

A single max-flow test instance with provenance.

Fields:
- `name`        : human-readable identifier (e.g., "tier2/grid2d/256")
- `tier`        : 1..4 (canonical, synthetic, real-DIMACS, SuiteSparse-derived)
- `category`    : :canonical, :grid2d, :grid3d, :genrmf, :washington,
                  :acyclic, :random, :vision, :medical, :ss_geom2d, etc.
- `loader`      : zero-arg function returning the (mfp, max_flow_value).
- `n`, `m`      : node and edge counts (≈ for lazily-loaded; 0 if unknown)
- `max_flow`    : known max-flow value (Float64 NaN if unknown)
"""
struct MaxFlowTestCase
    name::String
    tier::Int
    category::Symbol
    loader::Function
    n::Int
    m::Int
    max_flow::Float64
end

# ---------------------------------------------------------- Tier 1: canonical

function _tier1_canonical()
    tcs = MaxFlowTestCase[]
    push!(tcs, MaxFlowTestCase(
        "canonical/CLRS-directed", 1, :canonical,
        () -> nlf_clrs(undirected = false),
        6, 9, 23.0))
    push!(tcs, MaxFlowTestCase(
        "canonical/CLRS-undirected", 1, :canonical,
        () -> nlf_clrs(undirected = true),
        6, 9, 23.0))
    push!(tcs, MaxFlowTestCase(
        "canonical/bottleneck-k8-mid", 1, :canonical,
        () -> nlf_bottleneck_chain(8; cap_chain = 10.0,
                                         cap_bottle = 1.0, bottle_pos = 4),
        8, 7, 1.0))
    push!(tcs, MaxFlowTestCase(
        "canonical/bottleneck-k16-mid", 1, :canonical,
        () -> nlf_bottleneck_chain(16; cap_chain = 100.0,
                                          cap_bottle = 5.0, bottle_pos = 8),
        16, 15, 5.0))
    push!(tcs, MaxFlowTestCase(
        "canonical/image-seg-2x2", 1, :canonical,
        () -> nlf_image_seg_2x2(),
        6, 12, 10.0))
    # Bipartite K_{3,3} with full edges: max matching = 3.
    push!(tcs, MaxFlowTestCase(
        "canonical/bipartite-K33", 1, :canonical,
        () -> nlf_bipartite_matching(3, 3,
            [(i, j) for i in 1:3, j in 1:3 |> vec],
            3),
        8, 15, 3.0))
    return tcs
end

# ---------------------------------------------------------- Tier 2: synthetic generators

"""
    _tier2_synthetic(; rng_seed = 0xfa11) -> Vector{MaxFlowTestCase}

Scaling families: grid2d, grid3d, genrmf, washington, acyclic. Sizes
chosen to span ~3 decades of m (so β fitting per family is meaningful).
"""
function _tier2_synthetic(; rng_seed::UInt = UInt(0xfa11))
    tcs = MaxFlowTestCase[]
    # grid2d sequence — capacities heterogeneous (log-uniform).
    for k in (16, 32, 64, 128, 256)
        push!(tcs, MaxFlowTestCase(
            "grid2d/$(k)x$(k)", 2, :grid2d,
            let rs = rng_seed
                () -> (nlf_grid2d(k; cap_lo = 0.1, cap_hi = 10.0,
                                        rng = MersenneTwister(rs + k)),
                       NaN)
            end,
            k * k, 2 * k * (k - 1), NaN))
    end
    # grid3d sequence.
    for k in (8, 12, 16, 20, 24)
        push!(tcs, MaxFlowTestCase(
            "grid3d/$(k)^3", 2, :grid3d,
            let rs = rng_seed
                () -> (nlf_grid3d(k; cap_lo = 0.1, cap_hi = 10.0,
                                        rng = MersenneTwister(rs + k)),
                       NaN)
            end,
            k^3, 3 * k^2 * (k - 1), NaN))
    end
    # genrmf sequence.
    for (a, b) in [(4, 4), (8, 8), (16, 8), (16, 16), (32, 16)]
        push!(tcs, MaxFlowTestCase(
            "genrmf/$(a)x$(a)x$(b)", 2, :genrmf,
            let rs = rng_seed, a_ = a, b_ = b
                () -> (nlf_genrmf(a_, a_, b_;
                                        rng = MersenneTwister(rs + a_ + b_)),
                       NaN)
            end,
            a * a * b, 0, NaN))
    end
    # washington sequence.
    for (rows, cols) in [(4, 4), (8, 8), (16, 16), (32, 16)]
        push!(tcs, MaxFlowTestCase(
            "washington/$(rows)x$(cols)", 2, :washington,
            let rs = rng_seed, r_ = rows, c_ = cols
                () -> (nlf_washington(r_, c_;
                                            rng = MersenneTwister(rs + r_ + c_)),
                       NaN)
            end,
            2 + rows * cols, 0, NaN))
    end
    # acyclic dense at multiple sizes.
    for (n, p) in [(50, 0.10), (200, 0.05), (1000, 0.02), (5000, 0.005)]
        push!(tcs, MaxFlowTestCase(
            "acyclic-dense/$(n)_p$(p)", 2, :acyclic,
            let rs = rng_seed, n_ = n, p_ = p
                () -> (nlf_acyclic_dense(n_, p_;
                                               rng = MersenneTwister(rs + n_)),
                       NaN)
            end,
            n, 0, NaN))
    end
    # Random Erdős–Rényi.
    for (n, p) in [(100, 0.05), (500, 0.02), (2000, 0.005), (10_000, 0.001)]
        push!(tcs, MaxFlowTestCase(
            "random/$(n)_p$(p)", 2, :random,
            let rs = rng_seed, n_ = n, p_ = p
                () -> (nlf_random(n_, p_;
                                        rng = MersenneTwister(rs + n_)),
                       NaN)
            end,
            n, 0, NaN))
    end
    return tcs
end

# ---------------------------------------------------------- Tier 3: real DIMACS .max

"""
    _tier3_dimacs(data_dir) -> Vector{MaxFlowTestCase}

Loads `data_dir/*.max` files. Paired `.sol` files (if present) provide
the known max-flow value.
"""
function _tier3_dimacs(data_dir::String)
    tcs = MaxFlowTestCase[]
    isdir(data_dir) || return tcs
    for f in sort(readdir(data_dir))
        endswith(f, ".max") || continue
        path = joinpath(data_dir, f)
        sol_path = replace(path, ".max" => ".sol")
        mf = NaN
        if isfile(sol_path)
            for line in eachline(sol_path)
                if startswith(line, "s ")
                    mf = parse(Float64, split(line)[2])
                    break
                end
            end
        end
        category = startswith(f, "BVZ") || startswith(f, "KZ2") ? :vision_stereo :
                   startswith(f, "BL06") || startswith(f, "LB07") ? :vision_3d :
                   any(occursin(prefix, f) for prefix in
                       ["adhead", "babyface", "bone", "liver"]) ? :medical : :other_dimacs
        local_path = path
        push!(tcs, MaxFlowTestCase(
            "dimacs/" * replace(f, ".max" => ""), 3, category,
            () -> begin
                mfp = load_dimacs_max(local_path)
                return mfp, mf
            end,
            0, 0, mf))
    end
    return tcs
end

# ---------------------------------------------------------- Tier 4: SuiteSparse-derived

"""
    _tier4_suitesparse(data_dir; max_instances = 20, rng_seed = 0xfa11) -> Vector{MaxFlowTestCase}

Picks a sample of SuiteSparse Laplacians from `data_dir/*.mtx`, lazily
converts each to a NLFProblem by:
1. Extracting the largest CC.
2. Assigning log-uniform random capacities to each edge.
3. Picking the two farthest nodes by BFS as source/sink.

The known max-flow value is not computed (NaN) — the test is about
algorithm convergence + scaling, not absolute correctness.

`max_instances` caps the count (start small for the test set; raise to
all 2,031 for the full bench).
"""
function _tier4_suitesparse(data_dir::String;
                            max_instances::Int = 20,
                            rng_seed::UInt = UInt(0xfa11))
    tcs = MaxFlowTestCase[]
    isdir(data_dir) || return tcs
    files = filter(f -> endswith(f, ".mtx"), readdir(data_dir))
    if length(files) > max_instances
        # Deterministic-but-spread-out subsample.
        step = length(files) ÷ max_instances
        files = files[1:step:end][1:max_instances]
    end
    for f in files
        path = joinpath(data_dir, f)
        local_path = path
        push!(tcs, MaxFlowTestCase(
            "ss/" * replace(f, ".mtx" => ""), 4, :ss_derived,
            let lp = local_path, rs = rng_seed
                () -> _ss_to_maxflow(lp; rng = MersenneTwister(rs))
            end,
            0, 0, NaN))
    end
    return tcs
end

"""
    _ss_to_maxflow(path; rng, cap_lo, cap_hi) -> (NLFProblem, NaN)

Load a SuiteSparse `.mtx`, take the largest CC, orient each edge
arbitrarily, assign log-uniform capacities, pick (s, t) as the BFS-
extreme pair.
"""
function _ss_to_maxflow(path::AbstractString; rng = Random.default_rng(),
                        cap_lo::Real = 0.5, cap_hi::Real = 2.0)
    W = _read_mtx_adj(path)
    W, _ = _ss_largest_cc(W)
    n = size(W, 1)
    edges = Tuple{Int,Int}[]
    @inbounds for j in 1:n
        for k in W.colptr[j]:(W.colptr[j+1]-1)
            i = W.rowval[k]
            i < j && push!(edges, (i, j))
        end
    end
    m = length(edges)
    log_lo = log(cap_lo); log_hi = log(cap_hi)
    c = [exp(log_lo + (log_hi - log_lo) * rand(rng)) for _ in 1:m]
    s, t = _bfs_farthest_pair(W)
    return make_problem(n, edges, c, s, t), NaN
end

function _read_mtx_adj(path::AbstractString)
    open(path, "r") do io
        header = readline(io)
        tokens = split(lowercase(header))
        @assert tokens[1] == "%%matrixmarket" && tokens[2] == "matrix"
        @assert tokens[3] == "coordinate"
        field = tokens[4]
        symmetry = tokens[5]
        line = readline(io)
        while startswith(strip(line), "%"); line = readline(io); end
        nrows, ncols, nentries = parse.(Int, split(line))
        @assert nrows == ncols
        n = nrows
        cap = nentries * 2
        I = Vector{Int}(undef, cap); J = Vector{Int}(undef, cap)
        V = Vector{Float64}(undef, cap); kept = 0
        for _ in 1:nentries
            parts = split(readline(io))
            i = parse(Int, parts[1]); j = parse(Int, parts[2])
            v = field == "pattern" ? 1.0 :
                field == "complex" ? abs(parse(Float64, parts[3])) :
                parse(Float64, parts[3])
            i == j && continue
            kept += 1; I[kept] = i; J[kept] = j; V[kept] = v
            if symmetry in ("symmetric", "hermitian", "skew-symmetric")
                kept += 1; I[kept] = j; J[kept] = i; V[kept] = v
            end
        end
        resize!(I, kept); resize!(J, kept); resize!(V, kept)
        thr = sqrt(eps(Float64)) * maximum(abs, V; init = 0.0)
        @inbounds for k in eachindex(V)
            V[k] < -thr ? (V[k] = -V[k]) :
                abs(V[k]) < thr && (V[k] = 0.0)
        end
        W = SparseArrays.sparse(I, J, V, n, n, +)
        Wu = LinearAlgebra.triu(W, 1)
        return Wu + Wu'
    end
end

function _ss_largest_cc(W)
    n = size(W, 1); visited = falses(n); best = Int[]; q = Int[]
    for r in 1:n
        visited[r] && continue
        empty!(q); push!(q, r); visited[r] = true; comp = Int[r]
        while !isempty(q)
            u = popfirst!(q)
            for k in W.colptr[u]:(W.colptr[u+1]-1)
                v = W.rowval[k]
                if !visited[v]; visited[v] = true; push!(q, v); push!(comp, v); end
            end
        end
        length(comp) > length(best) && (best = comp)
    end
    keep = sort(best)
    return W[keep, keep], keep
end

function _bfs_farthest_pair(W)
    n = size(W, 1)
    function _bfs_far(src)
        dist = fill(-1, n); dist[src] = 0
        q = [src]; head = 1; far = src
        while head <= length(q)
            u = q[head]; head += 1
            for k in W.colptr[u]:(W.colptr[u+1]-1)
                v = W.rowval[k]
                if dist[v] < 0
                    dist[v] = dist[u] + 1
                    push!(q, v)
                    dist[v] > dist[far] && (far = v)
                end
            end
        end
        return far
    end
    a = _bfs_far(1); b = _bfs_far(a)
    return (a, b)
end

# ---------------------------------------------------------- Builder

"""
    build_test_set(; tiers = (1, 2, 3),
                    data_dir = joinpath(@__DIR__, "..", "data", "maxflow"),
                    ss_data_dir = joinpath(@__DIR__, "..", "data"),
                    ss_max::Int = 20,
                    rng_seed::UInt = 0xfa11) -> Vector{MaxFlowTestCase}

Assemble the max-flow test set. Pass `tiers = (1, 2, 3, 4)` to also
include the SuiteSparse-derived instances (heavier).
"""
function build_test_set(; tiers = (1, 2, 3),
                          data_dir = joinpath(@__DIR__, "..", "data", "maxflow"),
                          ss_data_dir = joinpath(@__DIR__, "..", "data"),
                          ss_max::Int = 20,
                          rng_seed = UInt(0xfa11))
    tcs = MaxFlowTestCase[]
    1 in tiers && append!(tcs, _tier1_canonical())
    2 in tiers && append!(tcs, _tier2_synthetic(rng_seed = UInt(rng_seed)))
    3 in tiers && append!(tcs, _tier3_dimacs(data_dir))
    4 in tiers && append!(tcs, _tier4_suitesparse(ss_data_dir;
                                                    max_instances = ss_max,
                                                    rng_seed = UInt(rng_seed)))
    return tcs
end
