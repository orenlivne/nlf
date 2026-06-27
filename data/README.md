# Benchmark data

NLF's corpus experiments run over the [SuiteSparse Matrix Collection](https://sparse.tamu.edu) (2003
graphs) and road networks from [TNTP](https://github.com/bstabler/TransportationNetworks). The full
collection is tens of GB, so we **do not** ship it.

## What is here

A small set of the graphs used by the reproducibility notebook
([`examples/nlf_demo.ipynb`](../examples/nlf_demo.ipynb)), in Matrix Market `.mtx` format:

| file | n | role in the notebook |
|---|---|---|
| `AG-Monien__grid2.mtx`        | 3,296  | fold inner-solver swap (Table "tab:innerfold", `grid2` row) |
| `AG-Monien__airfoil1.mtx`     | 4,253  | 2-D FEM fold instance |
| `Bomhof__circuit_1.mtx`       | 2,624  | small fold + no-fold congestion + competitor comparison |
| `Bomhof__circuit_3.mtx`       | 11,941 | circuit fold instance |
| `Gleich__wb-cs-stanford.mtx`  | 8,929  | poorly-separable (web) fold instance |

The synthetic max-flow instances in the exactness table (`nlf_grid2d`, `nlf_grid3d`,
`nlf_washington`, `nlf_genrmf`, `nlf_bottleneck_chain`) are generated in code by the `NLF` package — no
files needed.

## Getting the rest

To reproduce the full-corpus experiments (`scripts/run_corpus_3solver.jl`,
`scripts/run_fold_bakeoff.jl`), download the `.mtx` files into this directory. Either fetch by name from
the SuiteSparse Collection, or use Julia's loader:

```julia
using SuiteSparseMatrixCollection, MatrixMarket
ssmc = ssmc_db()
# e.g. one graph:
path = fetch_ssmc(ssmc[ssmc.name .== "grid2", :], format="MM")
```

Files are named `<group>__<name>.mtx` (the SuiteSparse group and matrix name joined by `__`).
