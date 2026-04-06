# SGNNET Excitatory Dynamic Routing — Research Report

**Researched:** 2026-03-30
**Domain:** Dynamic graph construction, attention-weighted message passing, direction-vector routing
**Confidence:** HIGH (mechanisms 1–3), MEDIUM (mechanism 4), MEDIUM (mechanism 5)

---

## Architecture Context (from project files)

Current SGNNET state at time of research:

| Property | Value |
|---|---|
| Neurons N | 512 |
| Direction space D | 16 (S¹⁵ sphere) |
| State Z[h] | R^D, l2-normalised |
| Structural graph | Watts-Strogatz, K_hh=6 (K_local=4, K_random=2) |
| Routing iterations | K_iter=3 |
| Beam size (inhibitory) | 32 (top-32 by magnitude broadcast) |
| Current radiation | INHIBITORY only (dynamic_z_geo) |
| State-of-the-art | 29.22% (step9A, D=16 Fourier, dynamic_z_geo, 150ep) |
| W_phase | Exists but static/learned graph failed; excitatory radiation is its replacement |

Key prior result: W_phase trained as a learned static weight matrix produced no benefit
(step10b: learned −0.69% vs reference; static random W_phase competitive but no gain).
The failure mode is clear: any learned static graph is baked in at init and does not
respond to the current input. The excitatory mechanism must rebuild from current Z.

---

## Summary

Five mechanisms are reviewed as candidates for excitatory dynamic connections in SGNNET.
The central question is: given that inhibitory dynamic routing (dynamic_z_geo) works well,
what is the right dual mechanism for excitation?

The mechanisms span a spectrum from most-direct-analog (Graph Attention Networks, which
already implement what we want) to more exotic (capsule routing-by-agreement, linear
attention variants). The core insight across all five: excitation via direction similarity
is well-supported by theory and practice, and the right similarity function for unit
vectors on S^(D-1) is the dot product, which equals cosine similarity when both vectors
are l2-normalised. The softmax temperature (equivalently, a score scaling factor) is the
single most important hyperparameter — it controls whether routing is sharp (winner-take-all)
or diffuse (uniform averaging).

**Primary recommendation:** Implement GAT-style attention as the excitatory pathway.
The formula is proven, differentiable, O(N²) worst case (controllable to O(N·beam) with
top-K selection), and directly interpretable as direction-similarity-weighted aggregation.
Combine with the existing inhibitory beam to give a complete Turing-type two-pathway system.

---

## Mechanism 1: Graph Attention Networks (GAT)

**Source:** Velickovic et al. "Graph Attention Networks", ICLR 2018.
**Confidence:** HIGH — well-understood, widely replicated.

### Core formula

The GAT attention coefficient between node i and neighbour j is:

```
e(i,j)  = LeakyReLU( a^T · [W·h_i || W·h_j] )
α(i,j)  = softmax_j( e(i,j) )
         = exp(e(i,j)) / Σ_{k∈N(i)} exp(e(i,k))
h'_i    = σ( Σ_{j∈N(i)} α(i,j) · W·h_j )
```

where `||` is concatenation, `a` is a learnable attention vector, `W` is a shared
linear projection, and `N(i)` is the neighbourhood of node i.

**Adapted version for SGNNET (no W projections, use current Z directly):**

Since Z is already l2-normalised and D=16 gives well-separated directions, we can
drop the linear projection W and compute attention directly from Z:

```
e(h,j)  = dot(Z[h], Z[j])         # cosine similarity on S¹⁵
           (optionally: dot(Z[h], Z[j]) / sqrt(D))  # temperature-scaled
α(h,j)  = softmax over j∈N(h) of e(h,j)
Z'[h]   = Σ_j α(h,j) · Z[j]      # weighted average of neighbours
Z[h]    ← normalize(Z'[h])        # keep on sphere
```

This is already structurally identical to what step14 implements for excitatory radiation,
with the difference that GAT applies softmax (normalised, sums to 1) whereas step14
uses raw cosine weights divided by sum (equivalent to softmax without exp transform).

### What makes it input-specific

The attention weights α(h,j) are computed from CURRENT h_i (Z[h]) and h_j (Z[j])
at each routing step. If the input changes Z, the attention weights change. This is
exactly the "dynamic_z" property that makes inhibitory routing work in SGNNET.

### Excitation, inhibition, or both

Standard GAT: **excitation only** (weighted average of neighbours — signal averaging
moves each neuron toward its neighbours, not away). Inhibition is absent by design.

### Why attention does not collapse (all neurons attend to the same node)

Four structural defences:

1. **Neighbourhood restriction:** attention is over N(h), not all N. With K_hh=6,
   only 6 candidates. Collapse to one of six is the worst case — not full-network collapse.

2. **Softmax temperature:** low temperature (sharp) makes attention collapse to one
   neighbour; high temperature (uniform) makes all weights equal. The 1/√D scaling
   factor is the practical temperature control. At D=16, scores are in a good range
   without explicit scaling. Without scaling, large-D dot products become large,
   making softmax sharper than intended — this is the main collapse risk.

3. **Multi-head attention:** GAT paper uses multiple independent attention heads,
   each attending to different parts of the neighbourhood. Averages or concatenates
   heads at output. Not critical for our case (we're not doing full GAT); noted for
   completeness.

4. **GATv2 fix:** Original GAT has a theoretical limitation — it can only compute
   a static ordering of attention (the attention of i→j is the same for any h_i with
   the same relative direction to h_j). GATv2 (Brody et al., 2022) fixes this by
   changing the order of operations: `e(i,j) = a^T · LeakyReLU(W·h_i + W·h_j)`.
   For SGNNET, since we use dot product directly (not a learned `a`), this limitation
   does not apply — dot(Z[h], Z[j]) is already fully dynamic in both arguments.

### Adaptation to SGNNET D=16 direction vectors

The simplest valid adaptation:

```
# All-to-all (O(N²), fine at N=512)
Z_norm = F.normalize(Z, dim=-1)                     # [B, N, D]
sim    = torch.bmm(Z_norm, Z_norm.transpose(1,2))   # [B, N, N], in [-1, 1]
sim    = sim / math.sqrt(D)                          # temperature scaling
sim.diagonal(dim1=1,dim2=2).fill_(-1e9)             # no self-attention

α = F.softmax(sim, dim=-1)                          # [B, N, N]
Z_exc = torch.bmm(α, Z_norm)                        # [B, N, D]
```

For top-K variant (O(N·beam), beam=16 or 32):
```
sim     = torch.bmm(Z_norm, Z_norm.transpose(1,2))   # [B, N, N]
top_val, top_idx = sim.topk(beam_exc, dim=-1)
top_val = top_val / math.sqrt(D)
α_top   = F.softmax(top_val, dim=-1)                 # [B, N, beam_exc]
# Gather neighbours and aggregate
idx_exp = top_idx.unsqueeze(-1).expand(-1,-1,-1,D)
Z_nb    = Z_norm.unsqueeze(1).expand(-1,N,-1,-1)
Z_nb    = torch.gather(Z_nb, 2, idx_exp)             # [B, N, beam_exc, D]
Z_exc   = (α_top.unsqueeze(-1) * Z_nb).sum(2)       # [B, N, D]
```

The top-K variant is what step14 currently implements (without the softmax — raw
cosine weights). Adding softmax is the single biggest change that converts step14's
excrad into proper GAT attention.

### Critical hyperparameters

| Hyperparameter | Effect | Recommended range |
|---|---|---|
| Temperature τ = 1/√D | Low τ → sharp (one winner); high τ → diffuse | τ = 1/√16 = 0.25 is the standard; try 0.1–1.0 |
| beam_exc (top-K) | Neighbourhood size for attention | 8–32; must be < N |
| alpha_exc (mixing weight) | How much excitatory signal is added | 0.1–0.5; higher risks over-smoothing |
| Use softmax vs raw weights | Softmax normalises; raw weights proportional to cosine | Softmax is safer (bounded); raw can blow up if many strong cosines |

---

## Mechanism 2: Routing by Agreement (Capsule Networks)

**Source:** Sabour et al. "Dynamic Routing Between Capsules", NeurIPS 2017.
         Hinton et al. "Matrix Capsules with EM Routing", ICLR 2018.
**Confidence:** HIGH — well-documented, mathematically clear.

### Core formula (Sabour 2017)

Capsule j at layer L+1 receives "prediction votes" from all capsules i at layer L:

```
û_{j|i} = W_{ij} · u_i                   # prediction: where capsule i thinks j should be

b_{ij}  ← 0                              # coupling log-probabilities, initialised to 0

for r iterations:
    c_{ij}  = softmax_j(b_{ij})          # coupling coefficients (sum to 1 over j per i)
    v_j     = squash( Σ_i c_{ij} · û_{j|i} )   # weighted sum → squash activation
    b_{ij} += dot(v_j, û_{j|i})          # agreement update: if v_j agrees with vote, strengthen

squash(s) = (||s||² / (1 + ||s||²)) · (s / ||s||)   # keeps direction, squashes magnitude
```

The key loop is the AGREEMENT step: `b_{ij} += dot(v_j, û_{j|i})`. If capsule j's
current output v_j is aligned with capsule i's prediction û_{j|i}, their coupling
coefficient is increased. This is positive feedback — excitatory connections form
between lower capsules and upper capsules whose activation they agree with.

### What makes it input-specific

The votes û_{j|i} are linear projections of the current input capsules u_i.
Changing the input changes u_i, which changes û_{j|i}, which changes which upper
capsules get excited. The routing is genuinely input-dependent at every forward pass.

### Excitation, inhibition, or both

Capsule routing is **excitatory only within each iteration** — the coupling coefficients
c_{ij} increase for agreeing pairs. However, because c_{ij} is a softmax (sum=1), when
one coupling increases, others necessarily decrease — this creates implicit lateral
inhibition via the normalisation. The net effect is agreement-based winner selection.

### Translation to SGNNET direction vectors

SGNNET's Z vectors are already the capsule analog: l2-normalised direction vectors on
S^(D-1). The "agreement" is dot product similarity between the broadcast direction of
one neuron and the receiving neuron's current direction. The EM routing version
(Hinton 2018) uses Gaussian mixture models and is more complex but conceptually similar.

**Direct adaptation:**

Replace W_{ij} (learned projection matrix per capsule pair — expensive and static) with
current Z similarity:

```
û_{j|i} = Z[i]                           # prediction IS current state (no projection)
b_{ij}  ← 0                              # log-coupling, shape [N, N]

for r_agree iterations (inner loop, typically 3):
    c_{ij}  = softmax_j(b_{ij})          # [N, N], sum over j = 1 per i
    # Weighted sum (batch dimension suppressed for clarity)
    s_j     = Σ_i c_{ij} · Z[i]         # [N, D]
    v_j     = F.normalize(s_j)           # direction only (no squash magnitude needed)
    b_{ij} += dot(v_j, Z[i])             # agreement: update coupling if directions align
```

**Key difference from step14 excrad:** Step14 computes similarity once and aggregates.
Capsule routing runs the inner loop r_agree=3 times, iteratively sharpening the
coupling coefficients. This is richer but adds cost: O(r_agree · N² · D) per routing step.

**Practical concern for SGNNET:** The inner routing loop (r_agree iterations) is
computationally redundant when the outer routing loop already runs K_iter=3 steps.
Two nested loops may cause gradient issues (backprop through the inner loop's soft-max
iterations interacts with the outer loop). The safest approach: use r_agree=1, which
degenerates to a single softmax aggregation — identical to GAT with dot-product attention.

### Critical hyperparameters

| Hyperparameter | Effect |
|---|---|
| r_agree (inner iterations) | More = sharper routing; 1 = reduces to GAT; 3 = Sabour original |
| Initial b_{ij} | Zero = uniform coupling; non-zero = prior bias toward certain pairs |
| Whether to detach v_j before agreement update | Detaching breaks gradients through inner loop (sometimes done for stability) |

**Recommendation for SGNNET:** Use r_agree=1 (single-pass). Multi-iteration inner loop
adds computation without proven benefit given the outer K_iter loop. If r_agree=1 shows
promise, try r_agree=2 as ablation.

---

## Mechanism 3: Transformer Self-Attention as Dynamic Graph Construction

**Source:** Vaswani et al. "Attention Is All You Need", NeurIPS 2017.
**Confidence:** HIGH — canonical, thoroughly analysed.

### Core formula

```
Q = X · W_Q,   K = X · W_K,   V = X · W_V    # linear projections [N, d_k]

Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V   # [N, N] attention matrix × values

MultiHead: concat of H independent heads, each with d_k = D/H
```

The attention matrix A = softmax(Q·K^T / √d_k) is a dynamic N×N weighted graph built
at every layer from the current input via Q, K projections. This is exactly what we want:
a complete dynamic weighted graph that changes per input.

### Key: the 1/√d_k scale factor

Without this factor, dot products between D-dimensional vectors scale as O(√D) in
magnitude when the vectors have unit variance. At D=16:
- Average dot product magnitude ≈ 4 (√16)
- Softmax of values in {-4, +4} is much sharper than softmax of {-1, +1}
- Sharp softmax → attention collapse: almost all weight on one token

The 1/√D factor normalises dot products back to O(1), keeping softmax in a useful
information-rich regime. **This is critical for SGNNET at D=16.**

However, since SGNNET's Z is ALREADY l2-normalised, dot products are already in [-1, +1]
without any scaling. The dot product of two unit vectors in R^16 is bounded by the
Cauchy-Schwarz inequality. The 1/√D factor is most important when vectors are NOT
normalised. For unit vectors, temperature of 1/√D ≈ 0.25 still helps by sharpening
the distribution slightly — but 1.0 is also valid.

**Empirical guidance from transformer practice:**
- Temperature = 1/√D: standard, works for most cases
- Temperature < 1/√D (sharper): encourages sparse, winner-take-all routing
- Temperature > 1/√D (softer): more diffuse averaging, resists collapse

### What prevents degenerate solutions

1. **Scale factor (above):** prevents attention from collapsing to one-hot due to
   magnitude explosion in the dot products.

2. **Positional encodings:** In transformers, positional encodings prevent tokens
   from attending purely based on content and ignoring position. In SGNNET, the Fourier
   spatial encoding already bakes position into Z at seed time — the routing then has
   both position and content signals available from the start. Explicit positional
   encoding in the attention formula (as in Rotary Position Embedding / RoPE) is not
   needed since position is already encoded in Z.

3. **Dropout on attention weights:** "Attention dropout" (dropout on the N×N attention
   matrix) is regularisation against collapse. In SGNNET, the existing routing_dropout_p
   (dropout on full Z vectors) serves a similar purpose.

4. **Residual connections:** Transformer blocks include X + Attention(X), preventing
   the network from entirely overwriting the input. SGNNET's Z routing already has
   a residual character since Z_new is always normalised and the seed at each step
   contributes through Z_struct.

### Adaptation to SGNNET

Without learned Q, K, V projections (simplest):

```
# Z is already l2-normalised [B, N, D]
scores  = torch.bmm(Z, Z.transpose(1,2)) / math.sqrt(D)   # [B, N, N]
scores.diagonal(dim1=1,dim2=2).fill_(-1e9)                 # no self-attention
α       = F.softmax(scores, dim=-1)                        # [B, N, N]
Z_exc   = torch.bmm(α, Z)                                  # [B, N, D]
```

With learned projections (adds D×D parameters per head, breaks "no learned graph" requirement):
```
# W_Q, W_K, W_V: [D, D] — adds N_heads * 3 * D² learnable params
# This is the full transformer formulation; likely overkill for current phase
```

**Key decision:** To satisfy the "not a static learned graph" constraint, use the
projection-free version (dot product directly on Z). Learned Q, K, V projections are
themselves learned static transforms — they produce a static mapping from Z to attention
weights, unlike the identity (direct dot product) which is fully dynamic.

### Critical insight: softmax vs top-K softmax

Full softmax over all N=512 neurons: every neuron receives some signal from every other.
This is soft over-smoothing — the excitatory signal becomes a diffuse mean.

Top-K gated softmax (what step14 implements): attend to only the K most similar neurons.
This is closer to the capsule routing intuition: form excitatory connections only with
neurons that are already "resonating."

**For SGNNET, top-K is preferred** because:
1. Step10a showed that simpler routing beats complex routing at D=16 — clean, sharp
   selection is better than diffuse averaging
2. Full O(N²) softmax with N=512 is ~262K ops per step — fine for computation,
   but produces very diffuse gradients that may be harder to learn from
3. Top-beam=16 already performs well in the inhibitory pathway

---

## Mechanism 4: EvolveGCN and Dynamic GNNs

**Source:** Pareja et al. "EvolveGCN: Evolving Graph Convolutional Networks for
            Dynamic Graphs", AAAI 2020. Also: Xu et al. "Inductive Representation
            Learning on Temporal Graphs" (TGAT), ICLR 2020.
**Confidence:** MEDIUM — well-cited but less directly applicable to SGNNET's structure.

### Core idea

EvolveGCN treats the GNN weight matrix W^(t) as evolving over time/layers:

```
# EvolveGCN-H variant (hidden-state-based evolution):
H^(t) = GRU(H^(t-1), W^(t-1))       # GRU over the weight matrix itself
W^(t) = H^(t)                         # evolved weight matrix
```

The adjacency itself is treated as fixed (structural graph); only the convolution
weights evolve. This is NOT what SGNNET needs — SGNNET already has fixed weights
(binary C matrices) and wants the TOPOLOGY to be dynamic.

### EvolveGCN-O variant (closer to our need):

```
W^(t) = GRU(W^(t-1))                  # simpler: only uses previous weight, no node feats
```

This produces input-independent weight evolution — NOT what we want. It's temporal
adaptation across training steps, not input-adaptive per-forward-pass topology.

### TGAT / temporal graph networks

These methods use temporal attention to weight neighbours based on their recency:
```
α(i,j,t) = softmax( dot(Q_i, [K_j || Φ(t-t_j)]) / √d )
```
where Φ(t-t_j) encodes the time elapsed since interaction. This is relevant for
time-varying graphs but requires a temporal dimension — SGNNET processes static images
(no time axis in the forward pass). The routing iterations (K_iter) are not sequential
in time; they are iterative refinement of a single forward pass.

### Over-smoothing and over-squashing (relevant to SGNNET)

These are the two canonical failure modes in any iterative GNN routing:

**Over-smoothing:** After K routing iterations, all Z[h] converge to the same direction.
  - Cause: each step takes a weighted average, which contracts toward the mean.
  - In SGNNET: l2-normalisation after each step partially prevents this, but deep
    routing (K_iter=12) is at risk.
  - Symptom: accuracy peaks early then drops; val_top1 vs best_epoch shows early peak.
  - Detection in experiment logs: check top1_history — does accuracy peak at e30-60
    then decline? That is over-smoothing.
  - Countermeasure: keep K_iter small (3–5), add skip connections (Z_new += alpha·Z_old),
    use the routing_dropout_p to prevent convergence.

**Over-squashing:** Information from distant nodes fails to propagate because the
  neighbourhood aggregation "squashes" the contribution of far-away nodes exponentially
  in distance.
  - In SGNNET: With K_iter=3 and diameter≈9, only 1/3 of the graph is reachable.
    Long-range information does not propagate. The small-world K_random=2 shortcuts
    help (diameter is O(log N)=9), but 3 hops still reach at most ~1-hop^3 neighbours.
  - Countermeasure: increase K_iter (tested in step13 depth row), or use the beam
    broadcast (which already enables long-range inhibition — excitatory radiation
    can serve the same purpose).
  - **Key insight:** The dynamic_z inhibitory beam (top-32 by magnitude broadcast to
    all N) is an over-squashing countermeasure — it allows any neuron to influence
    any other in a single step if it makes the top-32. Excitatory radiation with
    top-beam (by similarity, not magnitude) is the direct parallel.

### What EvolveGCN contributes to SGNNET design

The failure-mode analysis is the main contribution. The actual EvolveGCN mechanism
(evolving weight matrices) is not applicable. What IS applicable:

1. Over-smoothing check: monitor convergence of step13 depth row. If K_iter=8 or
   K_iter=12 show accuracy regression vs K_iter=3, over-smoothing is the cause.

2. Skip connections: residual routing `Z_new = F.normalize(Z_struct + Z_old * skip_weight)`
   prevents over-smoothing. Not yet tested in SGNNET.

3. Adaptive stopping: if over-smoothing begins at K_iter=5, train with random K_iter
   drawn uniformly from {1, 2, 3, 4, 5} per batch — forces the model to produce good
   representations at all depths.

---

## Mechanism 5: FAVOR+ / Linear Attention — Softmax Temperature Analysis

**Source:** Choromanski et al. "Rethinking Attention with Performers", ICLR 2021.
           Also: Katharopoulos et al. "Transformers are RNNs", ICML 2020.
**Confidence:** MEDIUM — the FAVOR+ approximation is less relevant; the temperature
analysis and softmax properties are HIGH confidence.

### FAVOR+ core idea

FAVOR+ (Fast Attention Via Orthogonal Random features) approximates the full softmax
attention matrix without computing all N² pairs:

```
softmax(Q·K^T/√d) ≈ φ(Q) · φ(K)^T           # random feature map φ
Attention(Q,K,V) ≈ φ(Q) · (φ(K)^T · V)       # O(N·r) instead of O(N²)
```

where r is the number of random features. This is an approximation for LARGE N (N>>1000).
**For SGNNET at N=512, full O(N²) is fine — 512²=262,144 ops, trivial on MPS.**
FAVOR+ is not needed.

### What IS relevant: the softmax temperature problem

The critical question for SGNNET routing: what temperature τ in softmax(scores/τ) produces
useful routing vs noise?

**Temperature effects on a distribution of cosine similarities on S¹⁵:**

At D=16, two uniformly random unit vectors have expected dot product ≈ 0.
If a neuron h has one "resonating" neighbour with cosine=0.8 and 15 others with cosine≈0:

| τ | softmax values (approx) | Effect |
|---|---|---|
| τ = 0.05 | resonating=0.99, others≈0.0 | Near-hard routing — one winner |
| τ = 0.25 = 1/√D | resonating=0.54, others=0.03 each | Good discrimination, some diversity |
| τ = 1.0 | resonating=0.12, others=0.06 each | Diffuse — close to uniform |
| τ = 4.0 | resonating=0.07, others=0.0625 | Effectively uniform — no routing signal |

**Recommendation for SGNNET excitatory radiation:**

Since Z is l2-normalised (unit vectors) and D=16:
- Raw dot products are in [-1, +1], most pairs near 0 for unrelated neurons
- Scaling by 1/√D = 0.25 is a good starting point
- BUT: τ can also be interpreted as the "discrimination threshold" — if you want only
  the top-few similar neurons to excite, lower τ; if you want broad diffuse excitation,
  higher τ

**Step14's current approach (no softmax, raw cosine with `.clamp(min=0)`):**
```
exc_weight = exc_vals / (exc_vals.sum(-1, keepdim=True).clamp(min=1e-6))
```
This is a linear normalisation over positive cosines. It is equivalent to softmax at
τ→∞ among the positive-cosine subset. It is milder than softmax — less discrimination,
less winner-take-all. The `.clamp(min=0)` means only positive cosine similarity triggers
excitation, which is a sensible prior for "resonance."

### Sharp routing vs soft routing: when does each win?

**Sharp routing (winner-take-all) advantages:**
- Creates clear specialisation: each neuron gets signal from its "best match"
- Prevents diffuse averaging (which destroys directional information)
- Encourages sparse representations (few neurons actually influence each other)
- Known to work better when the routing space is CLEAN (well-separated directions)
- SGNNET at D=16 is the clean-space case → sharp routing is more appropriate

**Soft routing (weighted average) advantages:**
- More gradient flow (all connections have non-zero gradients)
- Better for noisy representations where one best match may be unreliable
- Prevents "dead neuron" problem (neurons that never receive signal)
- SGNNET at D=4 (S³, crowded) would benefit more from soft routing

**Empirical evidence from SGNNET step10a:** At D=16, theta-only routing (sharper) beat
full routing (turing+reflection+theta, softer). This directly supports sharp routing
for D=16.

**Recommendation:** Start with softmax at τ = 1/√D = 0.25. If results improve,
try τ = 0.1 (very sharp) and τ = 0.5 (moderate). Do NOT use τ > 1.0 at D=16 —
this creates diffuse averaging that conflicts with the D=16 direction-discrimination capacity.

---

## Core Question: Similarity Function for S¹⁵

### Which similarity function is right for direction vectors?

SGNNET's Z is l2-normalised, so every Z[h] has unit norm.

**Dot product and cosine similarity are identical for unit vectors:**
```
cosine(a, b) = dot(a,b) / (||a|| · ||b||) = dot(a,b)   # when ||a||=||b||=1
```
This means dot product IS cosine similarity for unit vectors. No need to call F.normalize
before computing similarity if Z is already normalised after each routing step.

**Negative L2 distance and cosine similarity for unit vectors:**
```
||a - b||² = ||a||² + ||b||² - 2·dot(a,b) = 2 - 2·dot(a,b)
```
So `-||a-b||²/2 = dot(a,b) - 1`. The two are linearly related — negative L2 distance
gives the same ranking as cosine similarity (and thus the same top-K selection).
They differ only in the absolute scale of values:
- Cosine: range [-1, 1]
- Negative L2: range [-2, 0] (shifted by -1)

**For routing, rankings are what matter** → dot product and cosine are equivalent.
Negative L2 distance can be useful as a score because it has a fixed upper bound of 0
(two identical vectors, -||a-a||²/2 = 0) which makes the threshold interpretation cleaner.

### Recommendation

Use dot product on l2-normalised Z:
```
sim = torch.bmm(Z_norm, Z_norm.transpose(1,2))   # [B, N, N] in [-1, 1]
```

Reasons:
1. Already used in _inhibit_dynamic_z (consistency)
2. Directly interpretable as the cosine of the angle between directions on S¹⁵
3. torch.bmm is highly optimised; no extra cdist call needed
4. The `.clamp(min=0)` in step14 is correct for excitation: only positive cosine
   similarity (angle < 90°) should create excitatory connections

---

## Direct Applicability to step14 (Excitatory Radiation)

Step14 already implements the functional core of GAT-style excitatory routing.
The current implementation:

```python
# step14 current implementation (in SGNNET_TopKCond.forward)
Z_norm    = F.normalize(Z, dim=-1)
sim_all   = torch.bmm(Z_norm, Z_norm.transpose(1, 2))   # cosine similarity
sim_all  -= 1e9 * eye                                    # remove self
exc_vals, exc_idx = sim_all.topk(beam_exc, dim=-1)       # top-K similar
exc_vals  = exc_vals.clamp(min=0)                        # positive only
exc_weight = exc_vals / exc_vals.sum(-1, keepdim=True).clamp(min=1e-6)  # linear normalise
Z_radiate  = (exc_weight.unsqueeze(-1) * Z_exc_nb).sum(dim=2)          # weighted avg
Z_new     += alpha_exc * Z_radiate
```

**What this is:** Top-K constrained, positive-cosine-only, linearly-normalised
aggregation of directionally-similar neighbours. This is GAT attention with:
- Fixed neighbourhood defined by similarity (not graph topology) — correct
- Linear normalisation instead of softmax — mild vs sharp
- Positive cosine gate (`.clamp(min=0)`) — correct prior
- alpha_exc mixing weight — correct

**What could be changed based on this research:**

1. **Add temperature scaling before the clamp:**
   `exc_vals_scaled = exc_vals / math.sqrt(D)`
   Tests whether softmax temperature matters at D=16.

2. **Replace linear normalise with softmax:**
   `exc_weight = F.softmax(exc_vals / math.sqrt(D), dim=-1)`
   This is proper GAT attention.

3. **Try full N×N softmax (not top-K):**
   Allow all neurons to contribute, not just top-beam_exc.
   Tests whether restricting to top-K is important.

4. **Add capsule agreement loop (r_agree=1 is step14 current, r_agree=2 would be novel):**
   After computing Z_exc, update sim scores based on agreement between Z_exc and
   original Z[j], then re-aggregate. This is expensive but interesting.

These are four candidate next experiments that step15/16 could include.

---

## Failure Modes Specific to SGNNET

### Failure Mode 1: Excitation–inhibition imbalance

If excitatory radiation is too strong (alpha_exc >> alpha_turing), neurons may
converge too quickly to the same direction (all neurons excite each other into agreement).
The inhibitory beam prevents this, but if alpha_exc is large, the excitatory pull
dominates.

Detection: val_top1 climbs faster but peaks lower; loss decreases but routing
collapses (all neurons become similar direction vectors after routing).

Prevention: Keep alpha_exc ≤ 0.3 initially; the inhibitory pathway already runs at
alpha_turing=0.3, so keep excitation at the same scale or lower.

### Failure Mode 2: Resonance bubble (positive feedback loop)

Excitatory radiation creates positive feedback: neurons with similar directions
excite each other, which makes their directions even more similar (averaging toward
each other), which increases their similarity, which increases excitation. This is
identical to the attention collapse problem in transformers.

The l2-normalisation at each step partially prevents this: averaging two unit vectors
and then normalising does not reduce to either of them — it finds an intermediate direction.
But with many excitation steps, all routing neurons could converge to the mean direction.

Prevention:
- Top-K selection (step14 already has this) limits excitation to the top beam_exc
- alpha_exc < 1.0 ensures partial mixing, not full replacement
- The inhibitory beam is the critical counterweight: if neurons start converging,
  the top-magnitude neurons will inhibit each other via dynamic_z_geo

### Failure Mode 3: Gradient of top-K operation

`topk` returns indices; the gradient flows only through the selected top-K values,
not through the selection decision itself. This is a well-known limitation.

For step14: gradients flow through the SCORES (cosine similarities) of the selected
neurons. The which neurons are selected is not differentiable, but the how much they
contribute (exc_weight) is. This is the standard "straight-through" approximation.

In practice this works well — it is the same mechanism as the inhibitory beam
(also uses topk) which gave +3.1% from geo mode. No evidence this is a problem.

### Failure Mode 4: W_phase entanglement

The current step14 forward pass uses W_phase for the inhibitory pathway
(`_phase_inhibit` still runs with W_ph_norm). W_phase currently functions as a
static random matrix (not learned in dynamic_z_geo mode). If excitatory radiation
is added and alpha_turing is kept at 0.3, the inhibitory pathway via W_phase remains.

This means the excitatory radiation interacts with a fixed random inhibitory structure.
This is not necessarily wrong — random inhibitory graphs are a reasonable prior —
but it is worth ablating whether setting alpha_turing=0 (pure dynamic_z_geo inhibition
without W_phase component) improves results alongside excitatory radiation.

---

## Summary Table

| Mechanism | Formula type | Input-specific | Type | D=16 adaptation complexity | Key hyperparameter |
|---|---|---|---|---|---|
| GAT (Velickovic 2018) | softmax(QK^T/√d)·V | YES — from current features | Excitation | Low — step14 almost there | Temperature τ = 1/√D |
| Capsule routing | Agreement update: b += dot(v,û) | YES — from current capsule state | Excitation + implicit inhibition | Medium — inner loop overhead | r_agree iterations (use 1) |
| Transformer self-attn | softmax(ZZ^T/√D)·Z | YES — fully dynamic | Excitation (+ skip conn) | Low — one bmm + softmax | τ = 1/√D; no learned W needed |
| EvolveGCN | W evolves via GRU | Temporal, not per-input | N/A for SGNNET | High and wrong abstraction | Not recommended |
| FAVOR+ | Random feature approx | N/A at N=512 | Approximation | Not needed | Temperature τ for base attn |

**Verdict:** Mechanisms 1 and 3 (GAT and transformer self-attention) are isomorphic
at our scale and with l2-normalised vectors. Mechanism 2 (capsule routing) adds one
inner loop — use r_agree=1 which reduces to mechanism 1. Mechanisms 4 and 5 do not
apply directly but contribute diagnostic insights (over-smoothing, temperature).

---

## Recommended Experiments (Priority Order)

These build directly on step14's current design and can be added as config variants:

### Priority 1: Softmax vs linear normalisation (step14 variant)

**Change:** Replace `exc_vals / exc_vals.sum()` with `F.softmax(exc_vals / sqrt(D), dim=-1)`
**Why:** This is the single most meaningful difference from proper GAT attention.
**Expected effect:** Sharper selection, less diffuse excitation, better agreement with
step10a finding that sharp routing wins at D=16.

### Priority 2: Temperature sweep (τ = 0.1, 0.25, 0.5, 1.0)

**Change:** Scale scores before softmax: `exc_vals_scaled = exc_vals / τ`
**Why:** Calibrate the excitation sharpness for D=16 direction space.
**Expected optimal:** τ = 0.1 (sharp) or τ = 0.25 (standard). τ > 0.5 likely too soft.

### Priority 3: Full N×N excitatory attention vs top-K

**Change:** Compute full softmax over all N neurons (no topk), then aggregate.
**Why:** Tests whether top-K restriction is helping (sparse excitation) or hurting
(missing relevant excitatory connections not in top-K).
**Compute cost:** N²=262K ops per routing step, same as current sim computation.
**Expected effect:** Worse — step10a showed that simpler (sharper) routing wins at D=16.
But it is a necessary ablation data point.

### Priority 4: Capsule agreement loop (r_agree=2, inner 2-step)

**Change:** After computing Z_exc from step14 formulation, update coupling coefficients
by computing `agreement = dot(Z_exc, Z_norm[j])` for the top-K selected j, then
re-aggregate with updated weights.
**Why:** Tests whether iterative agreement (capsule routing) adds value over single-pass.
**Compute cost:** 2× per routing step for the re-aggregation only.

### Priority 5: Skip connection in excitatory aggregation

**Change:** `Z_new = F.normalize(alpha_exc * Z_exc + (1 - alpha_exc) * Z)` instead of
`Z_new = Z_struct + alpha_exc * Z_exc + ...` (summing into the full aggregation).
**Why:** Tests whether excitation should reinforce current direction or mix with structural.

---

## Sources

| Source | Type | Confidence | Topic |
|---|---|---|---|
| Velickovic et al. 2018 (GAT paper, ICLR) | Primary — training data | HIGH | Attention coefficients, neighbourhood restriction |
| Sabour et al. 2017 (NeurIPS, Capsule Nets) | Primary — training data | HIGH | Routing by agreement, coupling coefficients |
| Vaswani et al. 2017 (NeurIPS, Transformers) | Primary — training data | HIGH | 1/√d scaling, softmax temperature analysis |
| Brody et al. 2022 (GATv2) | Training data | HIGH | Static vs dynamic attention limitation |
| Pareja et al. 2020 (EvolveGCN, AAAI) | Training data | MEDIUM | Dynamic graph failure modes |
| Choromanski et al. 2021 (FAVOR+, ICLR) | Training data | MEDIUM | Linear attention, temperature analysis |
| Project files: model_resonant.py, train_step14_topk_cond_excrad.py | Direct inspection | HIGH | Exact current implementation |
| Project files: LEARNINGS_phase5_p3_breakthrough.md, p4_d16.md | Direct inspection | HIGH | Prior experimental results |

**Note on confidence:** WebSearch was not available for this session. All mechanism
descriptions are from training data (knowledge cutoff August 2025). These are canonical,
well-established papers that are unlikely to have changed significantly. The adaptation
analysis is specific to SGNNET's current architecture and is HIGH confidence given the
direct code inspection.

---

## Metadata

**Research date:** 2026-03-30
**Valid until:** This research covers theoretical mechanisms with stable foundations.
The SGNNET-specific recommendations depend on results from step14 (currently running).
Re-evaluate after step14 results arrive.

**Key open question not resolved by this research:**
What is the current step14 result (excrad vs no-excrad)? Once the neuro_j session
completes, the empirical answer to "does alpha_exc=0.3 beam=16 help?" will be available.
The research above provides the theoretical grounding to interpret whatever that result is:
- If excrad helps: sharpen the approach with softmax temperature (Priority 1-2 above)
- If excrad hurts: investigate whether diffuse averaging (low τ equivalent) is the cause;
  try sharp τ=0.1 or reduce alpha_exc to 0.05-0.1
