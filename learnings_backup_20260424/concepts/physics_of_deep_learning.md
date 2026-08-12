# Physics of Deep Learning — Research Program

**Opened:** 2026-04-21
**Thesis:** The same computation in deep learning can often be done in far fewer operations by
changing the mathematical frame — exactly as Lagrangian mechanics collapses Newtonian force
integration, or the center-of-mass frame decouples internal and external degrees of freedom.

The goal is not to find approximate methods. It is to find EXACT reformulations that are cheaper.

---

## Core Insight

In mechanics, switching from forces to energy (Hamiltonian/Lagrangian) does not change the
answer. It changes the frame. The answer becomes accessible without integrating forces step by
step, because the conserved quantities (energy, momentum, action) can be read off directly.

Analogues in deep learning:
- FFT: exact convolution in O(n log n) vs O(n²). Same result, different frame.
- Softmax = Boltzmann distribution. Temperature = regularization. Known but structural implications underexploited.
- Batch normalization = center-of-mass frame for activations. Decouples scale from direction.
- Residual connections = perturbative expansion around identity. Gradient flows through identity by default.

These were discovered empirically, not derived. The program here: derive them.

---

## Inventory: What Is Underutilized

### 1. Curl / Irrotationality (PRIORITY — see Experiment step964)

Gradient fields are always irrotational: curl(grad L) = 0 identically.
This means: the loss landscape gradient is a conservative field with a potential function L.
Gradient descent IS potential descent. This is known.

What is NOT known: whether the ROUTING field in SGNNET is irrotational.

The routing field F at node i: F(i) = normalize(W_pos[i] - W_pos[j]) for neighbors j.
If curl(F) = 0 everywhere on the graph → F = grad(phi) for some scalar phi.
Implication: routing = gradient flow on phi → K_iter iterations = Euler steps toward phi minimum.
The fixed point of routing is the minimum of phi. It can be found DIRECTLY by solving grad(phi) = 0,
skipping K_iter entirely. This would reduce routing from O(K_iter × N × K_hh × D) to O(N × D).

If curl ≠ 0: routing has genuine circulation (information cycles). Iteration is necessary.
The curl magnitude measures how much iteration buys you.

Discrete curl check on graph: for each triangle (i, j, k) with i→j→k→i edges,
compute circulation = F(i,j) + F(j,k) + F(k,i). If |circulation| ≈ 0 for all triangles → irrotational.

### 2. Divergence as Information Flow

div(F) at node i: measures whether i is a source (div > 0) or sink (div < 0) of routing signal.

In SGNNET: dead neurons (low norm Z) are sinks. High-activation nodes are sources.
Gauss's divergence theorem: total outflow through boundary = integral of div inside.
Implication: instead of computing routing at every interior node, compute boundary flux.
For a sphere of radius r in W_pos space: routing signal inside = surface integral outside.
This could reduce O(N) routing to O(N^{2/3}) for spherically organized W_pos.

Current metric that measures this: dead_frac (fraction of near-zero nodes = divergence sinks).

### 3. Potential Field Routing

If routing is irrotational: there exists phi(W_pos) such that F = grad(phi).
phi is the "routing potential" — a scalar field over the position embedding space.
Routing dynamics = gradient ascent on phi: Z moves toward high-phi regions.
Fixed point: Z lives at local maxima of phi.

This reframes K_iter routing as: find local maxima of phi.
Analytic solutions exist for many potential functions (harmonic, Coulomb, etc.).
If phi is harmonic (Laplacian = 0): solutions are well-characterized, no iteration needed.

### 4. Lie Group Routing

Rotations on S^(D-1) form a Lie group SO(D). The routing update Z_t+1 = f(Z_t, W_pos)
can be expressed as: Z_t+1 = exp(A(Z_t, W_pos)) Z_t for some skew-symmetric A.

Key property: matrix exponential always stays on the manifold. No normalize() call needed.
The Lie algebra (tangent space at identity) = skew-symmetric D×D matrices.
Parameterize routing directly in the Lie algebra → guarantee manifold membership algebraically.
Removes 5 normalize() calls per forward pass (one per K_iter).

This is related to the hypercomplex representation (quaternions = SU(2) = double cover of SO(3)).
Quaternion multiplication IS the Lie group operation for 3D rotations.
For D=16: the relevant Lie group is SO(16) or its subgroup Sp(8) (symplectic group).

### 5. Optimal Transport Routing

The routing matrix A_ij (attention / weight between nodes i and j) is a transport plan:
it moves "mass" from source distribution p(j) to target distribution q(i).
The optimal transport plan minimizes the Wasserstein distance W(p, q).

Sinkhorn algorithm: finds optimal transport in O(N/epsilon²) iterations.
This IS the routing step, computed optimally. Current softmax routing is a greedy approximation.
The Sinkhorn fixed point = the routing matrix that minimizes information transport cost.

Implication: replace routing loop with Sinkhorn iterations. Same fixed point, principled cost.
The routing "energy" becomes the Wasserstein distance between activation distributions.

### 6. Tropical Geometry

ReLU neural networks are tropical polynomials: f(x) = max(a·x + b, 0) in tropical arithmetic.
Tropical arithmetic: max replaces addition, addition replaces multiplication.
Tropical polynomials are piecewise linear with provable sparsity structure.
The number of linear regions = number of terms in the tropical polynomial.

SGNNET with LeakyReLU ≈ tropical polynomial (LeakyReLU is piecewise linear).
Implications:
- The routing function has a finite, enumerable set of linear regions.
- Each region is a convex polytope in input space.
- Sparsity of routing = number of active regions / total regions.
- Pruning = removing regions with low activation mass.

This gives a provable characterization of routing sparsity without empirical testing.

### 7. Symplectic / Hamiltonian Structure

Define H(Z, W_pos) = routing Hamiltonian.
Routing dynamics = Hamilton's equations: dZ/dt = ∂H/∂W_pos, dW_pos/dt = -∂H/∂Z.
Properties of Hamiltonian systems: energy conservation, symplectic structure, no volume contraction.

If routing has Hamiltonian structure: W_pos and Z are conjugate variables (like position and momentum).
The routing update is symplectic — it preserves the phase space volume.
Implication: routing cannot collapse (it's volume-preserving) unless H is explicitly dissipative.
Z collapse (PR=1) means routing is NOT Hamiltonian — it has dissipation.
Making routing Hamiltonian = enforcing PR preservation by construction.

### 8. Information Geometry

The softmax routing distribution p_i(j) = softmax(S_ij) lives on a probability simplex.
The Fisher information metric on this simplex gives the natural distance between routing policies.
Natural gradient descent in this metric = Newton's method but tractable.

Current routing update = Euclidean gradient (ignores the Riemannian structure of the simplex).
Natural gradient routing update = ∇̃L = F^{-1} ∇L where F is Fisher information matrix.
For diagonal Fisher (mean-field approximation): this is just dividing by activation variance.
This is exactly what Adam does! Adam ≈ natural gradient under diagonal Fisher approximation.

Implication: Adam IS already doing information-geometric routing updates.
But off-diagonal Fisher terms (correlations between nodes) are being ignored.
K-FAC or similar could exploit these: O(N × K_hh) instead of O(N²).

---

### 9. Adiabatic Training (PRIORITY — see step965)

**Fundamental motivation: finite precision = discrete optimization landscape.**
fp32/bf16 arithmetic is NOT continuous. Every gradient step is quantized to the nearest
representable value. We are taking discrete jumps on a quantized landscape, not flowing
along a smooth manifold. The "adiabatic" framing is thus not just an analogy — it is
the only physically correct description of gradient descent under finite precision.

Given each update is already a discrete jump:
- Non-adiabatic: jump all N nodes simultaneously. Each node's next gradient
  is computed against a landscape that was shifted by N-1 other discrete jumps
  it did not observe. Accumulated stale gradients → noise floor.
- Adiabatic: jump few nodes per step. Remaining nodes recompute gradients
  against the updated state. Each subsequent update is computed on the actual
  post-jump landscape, not a stale pre-jump approximation.

In quantum mechanics, the adiabatic theorem: change a system slowly enough and it stays in
its ground state. Change it fast → excite into higher-energy states → need more time to settle.
The discrete-precision analogy: each finite jump is an "excitation." Adiabatic training
minimizes excitations per unit of optimization progress.

Standard backprop: all N nodes update every batch. Non-adiabatic.
Hypothesis: updating only top-K% nodes by gradient magnitude per batch keeps the network
near its current attractor between updates. Other nodes adapt before the next change is made.
This is the "center of mass frame" for optimization: decouple high-gradient nodes from the
network bulk, update them one at a time, let the bulk adapt.

Concretely for SGNNET:
- Gradient mask applied to W_pos only (routing geometry — the adiabatically-sensitive part)
- All other params (W_in, W_out, theta) update normally
- Selection: top-K by |∇W_pos[i]| norm per batch

Quantum step variant: ΔW rounded to nearest Δθ=0.01. Most batches: no update (rounds to 0).
Network changes only in discrete steps. Analogous to quantized charge in condensed matter.

Gradient accumulation variant: accumulate 4 batches → top-5% update → reset.
Smoother gradient estimate before committing to a change.

Risk: gradient staleness. If node j updates while node i is frozen, node i's next gradient
is computed against an already-changed node j — the gradient is stale.
Mitigation: small update_frac → fewer stale interactions.

step965: T0 ablation over update_frac ∈ {1%, 5%, 20%} + quantum + accum4.
Advance criterion: ≥+0.5pp OR same accuracy with ep98 (98% of best) earlier.

## Priority Experiments

| Step | Concept | Test | Expected insight |
|------|---------|------|-----------------|
| step964 | Irrotationality of routing field | Compute discrete curl of F before/after training | If curl→0: routing has potential, K_iter is solving for fixed point |
| step965 | Potential routing | If curl≈0: replace K_iter with direct phi minimization | Same accuracy, 0 routing iterations |
| step966 | Lie group routing | Replace normalize() with exp(A), A skew-symmetric | Same manifold membership, fewer ops |
| step967 | Sinkhorn routing | Replace softmax aggregation with Sinkhorn fixed point | Optimal transport routing |

step964 is the diagnostic. Run it first. Results determine which of 965/966/967 is worth pursuing.

---

## Connection to SGNNET's Current Results

PR = 2.3 after routing: only 2-3 effective dimensions out of D=16 active.
This means routing is PROJECTING onto a 2-3D submanifold of S^15.
A 2D submanifold of a 15-sphere has near-zero curl (it's locally flat).
Prediction: routing field IS approximately irrotational (curl ≈ 0).
If confirmed: the 5-step routing loop is solving a 2D gradient flow problem that has a closed form.

This is the "center of mass frame" insight applied to SGNNET:
the routing lives in a 2D effective subspace. Everything else is redundant computation.
Find the 2D frame → routing collapses to a single operation.

---

## Long-Term Program

1. Characterize SGNNET routing mathematically: irrotational? Hamiltonian? Tropical?
2. Find the natural frame where routing has analytic solutions
3. Replace iterative routing with algebraic evaluation
4. Generalize: same tools apply to attention (transformer routing) and graph convolution

The claim to stake: "routing is potential flow on a learned scalar field."
If true, attention/routing across all architectures collapses to field evaluation.
This would be a first-principles unification of routing mechanisms in deep learning.
