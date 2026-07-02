# Follow-Up #1 — Nonlinear Network Propagation / Graph-Based SSL

**Formulation draft, 2026-07-02.** The flagship buyer problem, its math, incumbents, sizes, and
why FMG–FAS (not frozen-chord-Newton) is the natural engine. Honest throughout: the *quadratic*
case is already owned by fast linear solvers; the win is the *general-φ* nonlinear case, and the
fold that broke NLF's own FAS is absent here.

---

## 0. The most important buyer problem: seeded network propagation

Given a graph `G=(V,E)`, `n=|V|` nodes, `m=|E|` edges, nonnegative edge weights `π_ij` (affinity /
confidence). A small seed set `S ⊂ V` carries known signal `y_i` (disease genes, a drug's targets,
a labeled class). **Propagate** the seed signal over the graph to score every node — rank candidate
genes / drugs / labels by their diffused score. This single primitive (seed → diffuse → rank) is the
money problem in network medicine (target ID, drug repurposing) and the canonical baseline in
graph-based semi-supervised learning.

## 1. The objective (general φ)

Minimize a separable edge energy + a nodal label-fit term:

```
    min_x   E(x) = Σ_{(i,j)∈E} π_ij · φ(x_i − x_j)  +  Σ_{i∈V} ψ_i(x_i)
```

- `φ` : even, convex edge potential with monotone derivative `ρ := φ'` (the "edge law").
- `ψ_i` : nodal fit / reaction. Standard choice `ψ_i(x_i) = (μ_i/2)(x_i − y_i)²` with `μ_i>0` on
  seeds (soft clamp) — or a hard clamp `x_i = y_i` on `S`.

**Stationarity** (unlabeled `i`):

```
    Σ_j π_ij · φ'(x_i − x_j)  +  ψ_i'(x_i) = 0
```

Write `ρ_ij = φ'(x_i − x_j)`. In incidence form with `B` the node–edge incidence and diagonal
weight `Π = diag(π_e)`:

```
    B Π ρ(Bᵀ x)  +  ∇ψ(x) = 0            ⟺            B ρ̃(Bᵀ x) = s
```

**This is exactly NLF's master equation** `B ρ(Bᵀφ)=αd`, with the propagated scores `x` playing the
role of NLF's potentials, the monotone edge law `ρ=φ'`, and the label-fit term supplying a nodal
**reaction (mass) term** `∇ψ` — a source `s` on seeds plus a diagonal shift.

## 2. Special cases of φ — where linear stops and NLF/FAS starts

| φ(s) | ρ=φ'(s) | character | solver regime |
|---|---|---|---|
| `s²` | `2s` | quadratic diffusion (Zhou/Zhu label spreading, RWR) | **LINEAR** SDDM `(L+diag μ)x = μy` → LAMG+/approxChol already near-linear |
| `|s|^p`, 1<p<2 | `p·sign(s)|s|^{p-1}` | **p-Laplacian SSL** — fixes low-label degeneracy | nonlinear → NLF/FAS |
| `|s|` | `sign(s)` | **graph total variation** — sparse, edge-preserving | nonlinear, nonsmooth → FAS/primal-dual |
| Huber / Charbonnier | saturating | robust to noisy labels/edges | nonlinear → NLF/FAS |

**Key point:** with `φ=s²` the problem is a single SPD linear solve and LAMG+/approxChol/CMG already
win — there is **no differentiation there**. The opening is the *general φ*: p-Laplacian, TV, Huber.
Those are what buyers increasingly want (low-label-rate robustness, noise tolerance) and what current
tools solve *badly* (see incumbents).

**Convex, coercive, monotone, NO fold.** `φ` convex + `ψ` convex+coercive ⇒ unique minimizer,
monotone `ρ`, and crucially **no saturating fold** (unlike max-flow). This is NLF's *easy* (no-fold)
class — congestion-like. The singularity that broke NLF's caliber-1 FAS at `F*` **does not arise
here**, so FAS is unobstructed on this problem.

## 3. Why FMG–FAS, not frozen-chord-Newton (the hint, worked out)

NLF-as-published freezes **one global** linearization — the weighted Laplacian with edge weights
`φ''(x_i−x_j)` — and calls a linear solver (approxChol/LAMG+) as a black box, with continuation for
the fold. For propagation that's serviceable but leaves efficiency and generality on the table:

1. **FAS handles a *general* φ with no global refactor.** Relax the nonlinear stationarity locally
   (nodal Newton / nonlinear Gauss–Seidel: each node solves a 1-D monotone equation) and transfer the
   FAS defect + coarse representation. Change `φ'` in the local relaxation and the coarse law — one
   framework, any edge law. A linear solver can only ever do `φ=s²`.

2. **The frozen Jacobian is *bad* exactly where φ matters most.** For TV (`φ''→∞` at `s=0`) and
   p-Laplacian near `p=1`, the weights `φ''` span orders of magnitude and degenerate — the single
   frozen weighted Laplacian is ill-conditioned, and freeze-and-solve stalls. FAS never forms it:
   local relaxation sees the *current* `φ''` per edge per level, and coarsening rebuilds the operator
   level-by-level. This is the technical case for FAS over chord-Newton on strongly nonlinear φ.

3. **FMG (nested iteration) gives O(m) with a tiny constant *and* a physical warm start.** Solve on
   the coarsest aggregated graph, interpolate up. The coarse solution *is* a smoothed diffusion — a
   natural initial guess. Brandt's FMG reaches truncation-error accuracy in O(m) with ~1–2 V-cycles
   per level; for a seed→diffuse→rank query that is the whole budget.

4. **The label-fit reaction term is a gift to multigrid.** `ψ' = μ(x−y)` is a Helmholtz/mass shift
   → each level's operator is nonsingular and diagonally dominant (no null-space / compatibility
   dance that pure-Laplacian coarsening needs). Mass terms *help* FAS. This is the benign, well-posed
   setting multigrid was built for.

5. **Many right-hand sides amortize the hierarchy.** Buyers run *thousands* of queries (one per
   disease / drug / seed set) on the *same* graph. Build the FAS hierarchy once, reuse across all
   seeds — `O(m)` per query after an `O(m)` setup. This is the LAMG+ "frozen hierarchy, many solves"
   pattern, now for the nonlinear operator.

**Framing:** NLF = freeze + linear-inner + continuation (built for the *fold*). Propagation = no
fold, general φ, many RHS → the natural engine is **nonlinear FAS in an FMG wrapper**, with LAMG+/
approxChol available as the *linear* fallback for the `φ=s²` special case. The two are complementary,
not competing.

## 4. Applications

- **Disease-gene prioritization / network medicine** — seed known disease genes, diffuse over
  PPI / multi-omics / KG, rank candidates. (Cowen et al., *Network propagation: a universal amplifier
  of genetic associations*, Nat. Rev. Genet. 2017.)
- **Drug repurposing / target ID** — diffuse over drug–target–disease KGs; pharma's core use.
- **Protein function prediction** — GeneMANIA is literally label propagation on association networks.
- **Graph SSL in ML** — node classification at low label rates (the p-Laplacian's home turf).
- **Personalized PageRank / recommendation / trust / fraud propagation.**
- **Graph-TV denoising on geometric graphs** (images/manifolds/point clouds) — the regime where
  nonlinear φ *provably* beats quadratic (edge preservation).

## 5. Challenges (= the sales pitch)

- **Scale.** Real KGs are `10⁶–10⁸` edges; the standard tools invert the diffusion kernel densely
  (`O(n³)`), capping at `~10⁴` nodes. Near-linear is the unlock.
- **Low-label degeneracy.** Quadratic Laplacian SSL degenerates as `n→∞` at fixed labels
  (Nadler–Srebro–Zhou spikes); `p>d` p-Laplacian fixes it (Slepcev–Thorpe, El Alaoui et al.) — a
  *modeling* reason to go nonlinear, not just robustness.
- **Noisy labels / edges.** Quadratic is not robust; Huber/TV φ is.
- **Throughput.** Thousands of seed sets per graph → repeated-solve efficiency dominates.
- **Nonsmoothness** (TV, p→1) → needs a solver that tolerates degenerate `φ''` (FAS does; frozen
  Newton doesn't).

## 6. Incumbents + specific instances/sizes

**Solvers (quadratic case — the part already owned):**
- approxChol (Laplacians.jl, Kyng–Sachdeva), CMG (Koutis–Miller–Peng), Lean AMG / LAMG, PyAMG.
  Near-linear on `φ=s²`. **We do not differentiate here.**

**Off-the-shelf ML / bioinformatics tools (the part that's slow or capped):**
- scikit-learn `LabelPropagation`/`LabelSpreading` — dense, small graphs only.
- PyTorch-Geometric APPNP / label-prop, DGL — fixed quadratic diffusion.
- `diffuStats` (R) — **dense kernel inversion**, `~10⁴` node ceiling.
- p-Laplacian / TV SSL — research code via **IRLS / lagged-diffusivity** (an *outer* loop of 10s–100s
  of linear solves) or first-order primal-dual (Chambolle–Pock; many iterations). **This is the
  incumbent to beat:** FMG–FAS solves the nonlinear problem *directly* in ~one nested pass.

**Benchmark instances & sizes:**

| graph | nodes | edges | domain |
|---|---:|---:|---|
| ogbn-arxiv | 169 k | 1.2 M | ML node-classif (labels ✓) |
| Reddit (GraphSAGE) | 233 k | 11.6 M | ML SSL |
| Hetionet v1.0 | 47 k | 2.25 M | biomedical KG |
| PrimeKG | 129 k | ~4.1 M | biomedical KG |
| ogbn-products | 2.45 M | 61.9 M | ML SSL (labels ✓) |
| STRING (human PPI) | ~19.6 k | up to ~11 M | PPI |
| SPOKE | ~tens of M | ~tens of M | biomedical KG |
| ogbn-papers100M | 111 M | 1.6 B | ML SSL (labels ✓) |

OGB is the ideal proving ground: real labels → demonstrate **accuracy AND scale** head-to-head with
PyG/DGL; sizes span `10⁶→10⁹` edges for the O(m) curve. Hetionet/PrimeKG/SPOKE are the buyer-facing
biomedical demos.

## 7. Prior-art landmines to check before claiming novelty

Be honest — multigrid-for-nonlinear-graph-problems is not virgin territory:
- **Continuum p-Laplacian / TV multigrid** exists (image processing, PDE). The graph analog needs a
  literature check.
- **Graph p-Laplacian** work (Bühler–Hein spectral clustering; Slepcev–Thorpe consistency) — mostly
  analysis, not fast solvers. Confirm no one has shipped an FAS solver for it.
- **Algebraic multigrid on graph-SSL** — check PyAMG/CMG applications to label propagation.
- Our own **clustering-quality moat is dead** (AMG aggregation loses to Louvain/Leiden) — that's a
  *different* objective; propagation is a *solve*, not a partition, so that null does not apply here.
- The likely-honest novelty = **a general-φ FMG–FAS propagation solver, O(m), any edge law, at KG/OGB
  scale, beating IRLS/primal-dual** — a *methods+systems* contribution, not a new mathematical object.

## 8. Proposed first milestone

1. Fix the flagship: seeded propagation with `φ = Huber` (robust) and `φ = |s|^p`, `p=1.5`.
2. Build FAS V-cycle: nodal nonlinear GS smoother + aggregation coarsening (reuse LAMG+ machinery) +
   FAS defect transfer; wrap in FMG.
3. Prove-out on ogbn-arxiv (labels → accuracy) vs PyG label-prop and vs IRLS-on-approxChol (speed).
4. Scale on ogbn-products (62 M) then papers100M (1.6 B) for the O(m) curve.
5. Buyer demo on Hetionet/PrimeKG: robust target ranking vs `diffuStats`, at a scale it can't reach.

**Success = one nested FMG–FAS pass matches IRLS accuracy at a fraction of the cost, for arbitrary φ,
at a scale the dense/first-order incumbents cannot touch.**

---

Related: [[followup_problems]] · NLF paper §7 (FAS discussion, fold failure) · LAMG+ (linear inner /
quadratic-φ fallback). The Brandt agent is worth consulting on the FAS smoother + FMG schedule.
