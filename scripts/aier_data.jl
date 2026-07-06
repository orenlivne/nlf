# aier_data.jl — resolve the consolidated aier data hub via the $AIER_DATA env var,
# with a fallback to the package's own committed `data/` so a fresh `git clone`
# (e.g. a SISC reviewer's machine, with no hub and no $AIER_DATA) still finds the
# vendored / fetched data.
#
# Canonical copy lives in ~/code/aier/data (this file); an identical copy is shipped
# in each package's scripts/ dir so scripts stay self-contained:
#   include(joinpath(@__DIR__, "aier_data.jl"))
#   L = read_mm(aier_corpus("AG-Monien__grid2.mtx"))
#
# Resolution order for aier_corpus/aier_ssl/... :
#   1. $AIER_DATA/<subdir>/<parts>            (the hub; default ~/code/aier/data)
#   2. <repo>/data/<parts>                    (in-repo committed/fetched data — reviewer path)
#   3. the hub path (so a missing file yields a clear, canonical error message)

"Root of the data hub: \$AIER_DATA or ~/code/aier/data."
aier_data_root() = get(ENV, "AIER_DATA", joinpath(homedir(), "code", "aier", "data"))

# The package's own committed data dir. This file lives in scripts/, so ../data.
_aier_inrepo() = normpath(joinpath(@__DIR__, "..", "data"))

# Resolve `parts` under hub subdir `sub`, falling back to the in-repo data/ dir.
function _aier(sub::AbstractString, parts::AbstractString...)
    hub = joinpath(aier_data_root(), sub, parts...)
    ispath(hub) && return hub
    if isempty(parts)
        return isdir(_aier_inrepo()) ? _aier_inrepo() : hub
    end
    inrepo = joinpath(_aier_inrepo(), parts...)
    ispath(inrepo) ? inrepo : hub
end

"Join `parts` under the data-hub root (no in-repo fallback)."
aier_data(parts::AbstractString...) = joinpath(aier_data_root(), parts...)

"SuiteSparse / SNAP `.mtx` graph corpus (hub `suitesparse/`, else in-repo `data/`)."
aier_corpus(parts::AbstractString...) = _aier("suitesparse", parts...)

"MNIST / FashionMNIST tsv + VAE npz for NP experiments (hub `ssl/`, else in-repo `data/`)."
aier_ssl(parts::AbstractString...) = _aier("ssl", parts...)

"SPE10 pressure-Laplacian matrix (hub `spe10/`, else in-repo `data/`)."
aier_spe10(parts::AbstractString...) = _aier("spe10", parts...)

"TNTP road networks (hub `tntp/`, else in-repo `data/`)."
aier_tntp(parts::AbstractString...) = _aier("tntp", parts...)

"Max-flow `.max` instances (hub `maxflow/`, else in-repo `data/`)."
aier_maxflow(parts::AbstractString...) = _aier("maxflow", parts...)
