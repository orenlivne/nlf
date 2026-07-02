# NLF — Follow-Up Problems (research backlog)

Draft backlog started 2026-07-01, right after the SISC/arXiv submission. Honest framing:
each item notes where NLF/LAMG+ genuinely fits, the strongest incumbent to beat, and what
a real "win" would have to look like. Items we've already tested and killed are marked so
we don't re-open them blindly.

---

## 1. Nonlinear network propagation / graph-based semi-supervised learning
**The π_ij(x_i − x_j) energy.**

Minimize a separable edge energy `Σ_ij π_ij · φ(x_i − x_j)` over node values x, with a few
labeled/seed nodes fixed. This is disease-gene prioritization / network medicine
(propagate a signal over a PPI or knowledge graph) and, more broadly, graph-based SSL.

- **Linear version** (φ quadratic → `Σ π_ij (x_i−x_j)²`): this is exactly the SDDM solve
  `(I + γL) y = x` on a scale-free, irregular graph — **LAMG+'s moat**, not NLF's. Standard
  tools (diffuStats, igraph) do dense O(n³) kernel inversion capped at ~10–20k nodes; a
  near-linear solver reaches biomedical KGs at millions of edges (Hetionet, PrimeKG, SPOKE ~50M).
  Clean scale-unlocking win — but it's a **LAMG+ (linear) play, not the NLF nonlinear story**,
  and it competes with approxChol (public co-incumbent → no unique IP from the solver alone).
- **Nonlinear version = the actual NLF lane**: replace the quadratic with a *robust / sparsifying*
  edge cost — p-Laplacian propagation, graph total variation, ℓ1 / graph trend filtering. Now the
  per-edge law is nonlinear and monotone → drops straight into `B ρ(Bᵀφ) = αd`. This is where NLF's
  chord-Newton-on-a-frozen-Laplacian buys something a linear solver can't.
- **What a win looks like:** robust/TV propagation at KG scale (10⁷–10⁸ edges) where the nonlinear
  regularizer matters and first-order / dense methods choke.
- **Already killed — do NOT re-open:** the "reuse the multigrid hierarchy to *cluster* too" idea.
  AMG/SWA aggregation loses to Louvain/Leiden/Infomap on modularity and to nested-SBM on planted
  blocks (details in the bioinformatics memory). Clustering-quality moat is dead unless the data is
  genuinely geometric/multiscale (images/manifolds), not abstract density communities.

## 2. Machine-learning / data-science lane (the "breakthrough" hunt)
From the NLF review: the moat is **irregular, poorly-separable graphs** — so hunt ML-on-graphs,
not structured physics.

- **Best-fit candidates (nonlinear → real NLF):** p-Laplacian SSL, graph total-variation denoising,
  graph trend filtering, robust label propagation. All are `Σ π_ij φ(x_i−x_j)` with nonlinear φ —
  same machinery as §1's nonlinear version, different application framing.
- **Tested and REFUTED — do not chase:** optimal transport. Plain graph-W1 is *linear* min-cost
  flow; NLF loses to exact MCF at every scale (slower + less accurate, no crossover). NLF's edge is
  only the *nonlinear* regime, which W1 isn't. (Refuted end-to-end 2026-06-27, graphot/FINDINGS.md.)
- **Near-fatally obstructed — park:** AMG on a DNN Hessian. Keep on the shelf, not the roadmap.
- **What a win looks like:** a named ML task where (a) the objective is genuinely nonlinear-separable
  on a graph, (b) the graph is irregular at 10⁷⁺ scale, (c) there's a real user and no near-linear
  incumbent. This is the single most likely path from "solid methods paper" to "breakthrough."

## 3. Directed traffic assignment (directed user equilibrium) — the big prize
NLF currently solves the **undirected relaxation**. The transportation field solves the **directed**
UE (nonnegative arc flows, one-sided cost). Winning *that* would displace the field's incumbent on
its own problem — unambiguous breakthrough, highest value, highest risk.

- **The gap is real, not cosmetic:** on SiouxFalls the undirected objective differs from directed UE
  by ~99.9%. The relaxation is *not* a shortcut to the directed answer.
- **Champion to beat:** bush methods — Algorithm B / TAPAS (tap-b). Validated reference: relgap 1e-12
  in 0.024s on SiouxFalls. Any directed-NLF must beat tuned bush methods on real metropolitan
  networks at scale, not just match them on toy graphs.
- **What's needed:** per-edge complementarity / free-boundary machinery for the one-sided cost (the
  paper only sketches this as future work). Multi-month research program.
- **Adjacent, already prototyped:** directed-NLF's real payoff may be *design* gradients (tolling / OD
  / network-design) via O(k)-cheaper adjoints, not the forward solve — see the DNLF notes.

## 4. DC optimal power flow (already seeded in the paper, §7.5)
Listed as a "structural instance, not yet solved." A fold-type problem with an FAS extension that
does **not yet work** (caliber-1 interpolation misrepresents the coarse flux near saturation). Honest
status: open research, not a near-term win. Gas/H2 pipeline networks were tested and **killed**
(lose to Newton-Krylov; separability trap) — do not add them to the roadmap.

---

### Priority read
- **Highest breakthrough potential:** #3 (directed UE) — but longest and riskiest.
- **Most tractable real NLF win:** #1/#2 nonlinear version (p-Laplacian / graph-TV propagation at KG scale).
- **Cleanest scale-unlock, but it's a LAMG+ (linear) story:** #1 linear network propagation.
- **Do not re-open:** OT (refuted), clustering-quality moat (dead), gas/reservoir (killed).
