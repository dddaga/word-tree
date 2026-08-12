# Physics of Deep Learning — Research Program

**Opened:** 2026-04-21
**Thesis:** Same computation in deep learning often doable in far fewer ops by changing mathematical frame — exactly as Lagrangian mechanics collapses Newtonian force integration, or center-of-mass frame decouples internal/external degrees of freedom.

Goal not approximate methods. Goal EXACT reformulations that cheaper.

---

## Core Insight

Switching forces to energy (Hamiltonian/Lagrangian) not change answer. Changes frame. Answer accessible without integrating forces step by step — conserved quantities (energy, momentum, action) read off directly.

Deep learning analogues:
- FFT: exact convolution O(n log n) vs O(n²). Same result, different frame.
- Softmax = Boltzmann distribution. Temperature = regularization. Known but structural implications underexploited.
- Batch normalization = center-of-mass frame for activations. Decouples scale from direction.
- Residual connections = perturbative expansion around identity. Gradient flows through identity by default.

All discovered empirically, not derived. Program here: derive them.

---

## Inventory: What Is Underutilized

### 1. Curl / Irrotationality (PRIORITY — see Experiment step964)

Gradient fields always irrotational: curl(grad L) = 0 identically.
Loss landscape gradient = conservative field with potential function L.
Gradient descent IS potential descent. Known.

What NOT known: whether ROUTING field in SGNNET irrotational.

Routing field F at node i: F(i) = normalize(W_pos[i] - W_pos[j]) for neighbors j.
If curl(F) = 0 everywhere on graph → F = grad(phi) for scalar phi.
Implication: routing = gradient flow on phi → K_iter iterations = Euler steps toward phi minimum.
Fixed point of routing = minimum of phi. Find DIRECTLY by solving grad(phi) = 0,
skip K_iter entirely. Reduces routing from O(K_iter × N × K_hh × D) to O(N × D).

If curl ≠ 0: routing has genuine circulation (information cycles). Iteration necessary.
Curl magnitude measures how much iteration buys.

Discrete curl check on graph: for each triangle (i, j, k) with i→j→k→i edges,
compute circulation = F(i,j) + F(j,k) + F(k,i). If |circulation| ≈ 0 for all triangles → irrotational.

### 2. Divergence as Information Flow

div(F) at node i: measures whether i source (div > 0) or sink (div < 0) of routing signal.

In SGNNET: dead neurons (low norm Z) are sinks. High-activation nodes are sources.
Gauss's divergence theorem: total outflow through boundary = integral of div inside.
Implication: instead of computing routing at every interior node, compute boundary flux.
For sphere of radius r in W_pos space: routing signal inside = surface integral outside.
Could reduce O(N) routing to O(N^{2/3}) for spherically organized W_pos.

Current metric measuring this: dead_frac (fraction near-zero nodes = divergence sinks).

### 3. Potential Field Routing

If routing irrotational: exists phi(W_pos) such that F = grad(phi).
phi = "routing potential" — scalar field over position embedding space.
Routing dynamics = gradient ascent on phi: Z moves toward high-phi regions.
Fixed point: Z lives at local maxima of phi.

Reframes K_iter routing as: find local maxima of phi.
Analytic solutions exist for many potential functions (harmonic, Coulomb, etc.).
If phi harmonic (Laplacian = 0): solutions well-characterized, no iteration needed.

### 4. Lie Group Routing

Rotations on S^(D-1) form Lie group SO(D). Routing update Z_t+1 = f(Z_t, W_pos)
expressible as: Z_t+1 = exp(A(Z_t, W_pos)) Z_t for skew-symmetric A.

Key: matrix exponential always stays on manifold. No normalize() needed.
Lie algebra (tangent space at identity) = skew-symmetric D×D matrices.
Parameterize routing directly in Lie algebra → guarantee manifold membership algebraically.
Removes 5 normalize() calls per forward pass (one per K_iter).

Related to hypercomplex representation (quaternions = SU(2) = double cover of SO(3)).
Quaternion multiplication IS Lie group operation for 3D rotations.
For D=16: relevant Lie group SO(16) or subgroup Sp(8) (symplectic group).

### 5. Optimal Transport Routing

Routing matrix A_ij (attention/weight between nodes i and j) = transport plan:
moves "mass" from source distribution p(j) to target distribution q(i).
Optimal transport plan minimizes Wasserstein distance W(p, q).

Sinkhorn algorithm: finds optimal transport in O(N/epsilon²) iterations.
This IS routing step, computed optimally. Current softmax routing = greedy approximation.
Sinkhorn fixed point = routing matrix minimizing information transport cost.

Implication: replace routing loop with Sinkhorn iterations. Same fixed point, principled cost.
Routing "energy" becomes Wasserstein distance between activation distributions.

### 6. Tropical Geometry

ReLU networks = tropical polynomials: f(x) = max(a·x + b, 0) in tropical arithmetic.
Tropical arithmetic: max replaces addition, addition replaces multiplication.
Tropical polynomials piecewise linear with provable sparsity structure.
Number of linear regions = number of terms in tropical polynomial.

SGNNET with LeakyReLU ≈ tropical polynomial (LeakyReLU piecewise linear).
Implications:
- Routing function has finite, enumerable set of linear regions.
- Each region = convex polytope in input space.
- Sparsity of routing = active regions / total regions.
- Pruning = removing regions with low activation mass.

Gives provable characterization of routing sparsity without empirical testing.

### 7. Symplectic / Hamiltonian Structure

Define H(Z, W_pos) = routing Hamiltonian.
Routing dynamics = Hamilton's equations: dZ/dt = ∂H/∂W_pos, dW_pos/dt = -∂H/∂Z.
Properties: energy conservation, symplectic structure, no volume contraction.

If routing Hamiltonian: W_pos and Z conjugate variables (like position and momentum).
Routing update symplectic — preserves phase space volume.
Implication: routing cannot collapse (volume-preserving) unless H explicitly dissipative.
Z collapse (PR=1) means routing NOT Hamiltonian — has dissipation.
Making routing Hamiltonian = enforcing PR preservation by construction.

### 8. Information Geometry

Softmax routing distribution p_i(j) = softmax(S_ij) lives on probability simplex.
Fisher information metric on simplex gives natural distance between routing policies.
Natural gradient descent in this metric = Newton's method but tractable.

Current routing update = Euclidean gradient (ignores Riemannian structure of simplex).
Natural gradient routing: ∇̃L = F^{-1} ∇L where F = Fisher information matrix.
For diagonal Fisher (mean-field approximation): just dividing by activation variance.
This exactly what Adam does! Adam ≈ natural gradient under diagonal Fisher approximation.

Implication: Adam IS already doing information-geometric routing updates.
But off-diagonal Fisher terms (correlations between nodes) ignored.
K-FAC or similar could exploit: O(N × K_hh) instead of O(N²).

---

### 9. Adiabatic Training (PRIORITY — see step965)

**Fundamental motivation: finite precision = discrete optimization landscape.**
fp32/bf16 arithmetic NOT continuous. Every gradient step quantized to nearest
representable value. Taking discrete jumps on quantized landscape, not flowing
along smooth manifold. "Adiabatic" framing not just analogy — only physically
correct description of gradient descent under finite precision.

Given each update already discrete jump:
- Non-adiabatic: jump all N nodes simultaneously. Each node's next gradient
  computed against landscape shifted by N-1 other discrete jumps
  it did not observe. Accumulated stale gradients → noise floor.
- Adiabatic: jump few nodes per step. Remaining nodes recompute gradients
  against updated state. Each subsequent update computed on actual
  post-jump landscape, not stale pre-jump approximation.

Quantum mechanics adiabatic theorem: change system slowly enough → stays in
ground state. Change fast → excite into higher-energy states → need more time to settle.
Discrete-precision analogy: each finite jump = "excitation." Adiabatic training
minimizes excitations per unit optimization progress.

Standard backprop: all N nodes update every batch. Non-adiabatic.
Hypothesis: updating only top-K% nodes by gradient magnitude per batch keeps network
near current attractor between updates. Other nodes adapt before next change.
"Center of mass frame" for optimization: decouple high-gradient nodes from
network bulk, update one at a time, let bulk adapt.

Concretely for SGNNET:
- Gradient mask applied to W_pos only (routing geometry — adiabatically-sensitive part)
- All other params (W_in, W_out, theta) update normally
- Selection: top-K by |∇W_pos[i]| norm per batch

Quantum step variant: ΔW rounded to nearest Δθ=0.01. Most batches: no update (rounds to 0).
Network changes only in discrete steps. Analogous to quantized charge in condensed matter.

Gradient accumulation variant: accumulate 4 batches → top-5% update → reset.
Smoother gradient estimate before committing to change.

Risk: gradient staleness. If node j updates while node i frozen, node i's next gradient
computed against already-changed node j — gradient stale.
Mitigation: small update_frac → fewer stale interactions.

step965: T0 ablation over update_frac ∈ {1%, 5%, 20%} + quantum + accum4.
Advance criterion: ≥+0.5pp OR same accuracy with ep98 (98% of best) earlier.

## Priority Experiments

| Step | Concept | Test | Expected insight |
|------|---------|------|-----------------|
| step964 | Irrotationality of routing field | Compute discrete curl of F before/after training | If curl→0: routing has potential, K_iter solving for fixed point |
| step965 | Potential routing | If curl≈0: replace K_iter with direct phi minimization | Same accuracy, 0 routing iterations |
| step966 | Lie group routing | Replace normalize() with exp(A), A skew-symmetric | Same manifold membership, fewer ops |
| step967 | Sinkhorn routing | Replace softmax aggregation with Sinkhorn fixed point | Optimal transport routing |

step964 = diagnostic. Run first. Results determine which of 965/966/967 worth pursuing.

---

## Connection to SGNNET's Current Results

PR = 2.3 after routing: only 2-3 effective dimensions out of D=16 active.
Routing PROJECTING onto 2-3D submanifold of S^15.
2D submanifold of 15-sphere has near-zero curl (locally flat).
Prediction: routing field IS approximately irrotational (curl ≈ 0).
If confirmed: 5-step routing loop solving 2D gradient flow problem with closed form.

"Center of mass frame" insight applied to SGNNET:
routing lives in 2D effective subspace. Everything else redundant computation.
Find 2D frame → routing collapses to single operation.

---

## Long-Term Program

1. Characterize SGNNET routing mathematically: irrotational? Hamiltonian? Tropical?
2. Find natural frame where routing has analytic solutions
3. Replace iterative routing with algebraic evaluation
4. Generalize: same tools apply to attention (transformer routing) and graph convolution

Claim to stake: "routing is potential flow on learned scalar field."
If true, attention/routing across all architectures collapses to field evaluation.
Would be first-principles unification of routing mechanisms in deep learning.