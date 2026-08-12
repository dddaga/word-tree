# Hypercomplex Representations for Routing — Paper 2 Concept

**Status:** HYPOTHESIS — not implemented. Captured from 2026-04-21 session.
**Scope:** Paper 2 (dynamic routing). Paper 1 is unaffected.

---

## The Core Question

Going from Cartesian real-valued vectors to hypercomplex representations: is there a computational benefit?

**Answer:** The benefit is parameter efficiency, NOT FLOP reduction per operation.

A hypercomplex multiply encodes the same transformation as a real matrix multiply but with
fewer stored parameters, because one hypercomplex multiply couples all components simultaneously
(via Hamilton product or equivalent). Raw FLOP count is roughly equivalent; the saving is
structural compression of the weight representation.

---

## The Algebra Hierarchy

| Algebra | Dims | Associative | Commutative | Division | Parameter savings | Notes |
|---------|------|-------------|-------------|----------|-------------------|-------|
| Real | 1 | Yes | Yes | Yes | 1× | baseline |
| Complex C | 2 | Yes | Yes | Yes | 2× | standard in signal processing |
| Quaternion H | 4 | Yes | No | Yes | 4× | i,j,k with ijk=-1 |
| Octonion O | 8 | No (alternative) | No | Yes | 8× | non-associative, backprop demonstrated |
| Sedenion S | 16 | No | No | No (zero divisors) | 16× | has zero divisors |

Beyond sedenions (Cayley-Dickson construction): loses even more properties.
Hurwitz theorem: only R, C, H, O are normed division algebras.

---

## Evidence from Literature

Quaternion neural networks (Parcollet et al., Gaudet et al.):
- 4× parameter reduction vs real-valued FC layers at matched expressiveness
- Quaternion GAN: 75% parameter reduction, maintained quality
- Quaternion CNN: state-of-the-art at image classification with fewer params
- Cost: quaternion layers train ~50% slower (dense Hamilton product)
- Mitigation: EdgeLDR uses block-circulant structure + FFT for efficient quaternion evaluation

Octonion neural networks: non-associative but backpropagation has been demonstrated.
Sedenion neural networks: backprop algorithm exists, application in time series.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/ (2025 survey)
Source: https://arxiv.org/pdf/1712.04604 (Deep Quaternion Networks)

---

## The D=16 Coincidence

SGNNET currently uses D=16 real dimensions per node.

- Sedenion dimensionality: 16 real components = 1 sedenion
- Quaternion grouping: D=16 = 4 quaternion units (4 × 4 real components)
- Octonion grouping: D=16 = 2 octonion units (2 × 8 real components)

At D=16, W_pos (position embeddings on S^15) can be represented as:
- 4 unit quaternions: captures 4D rotation structure, 4× parameter savings
- 2 unit octonions: captures 8D rotation structure, 8× parameter savings
- 1 sedenion: full 16D structure with zero-divisor sparsity pattern

The sedenion zero divisors are not obviously a bug — they enforce structured sparsity
in the routing geometry automatically. Whether this is useful is unknown.

---

## Relevance to Dynamic Routing (Paper 2)

The routing coefficient in ΔW-proj is:
  c_ij = dot(Z_nb, normalize(W_pos[i] - W_pos[j]))

In quaternion form, the relative rotation from position i to j is:
  q_ij = W_pos[j] * conj(W_pos[i])  (unit quaternion, exact relative rotation)

This gives EXACT rotation representation (not just projection approximation) with the same
D=16 storage. The routing signal becomes the phase-aligned component Re(q_ij · z_i).

This is a CLEAN architectural fit: ΔW-proj already approximates rotational geometry;
quaternions make the rotation exact and parameterize it efficiently.

---

## For Complex Numbers Specifically

W_pos lives on S^(D-1). With r=1 always (unit sphere), magnitude carries no information.
Complex representation of W_pos = unit complex numbers (phases only). For D=16:
- 8 complex phases per node, encoding relative orientation
- Routing coefficient = Re(Z_i * conj(Z_j)) = cos(theta_i - theta_j) — phase alignment
- This is essentially what ΔW-proj computes but made exact

Quantized phases (n-th roots of unity): valid angles = {2πk/n : k=0,...,n-1}
- Pre-computed cosine lookup table: n×n = 64 entries at n=8
- Inference: table lookup replaces trig — negligible cost
- Training: standard fp32, quantize at inference

---

## Recommended Path for Paper 2

1. Quaternion W_pos: replace 16-real W_pos with 4-quaternion W_pos (same storage, structural constraint)
   Hypothesis: exact rotation routing improves over approximate projection
   Test: step960 (T0 scout, quaternion ΔW-proj vs real ΔW-proj)

2. If quaternion shows signal → test octonion (2 units at D=16)
3. Sedenion: test only if 1+2 show monotone improvement; zero divisors need stability check

Do NOT pursue this for Paper 1. Paper 1 uses real-valued D=16 throughout.

---

## Open Questions

1. Does quaternion W_pos improve PR (participation ratio) above current 2.3?
2. Is the zero-divisor property of sedenions actually equivalent to structured sparsity?
3. At what D does the hypercomplex benefit become significant? (D=16 may be too small)
4. Can standard PyTorch autograd handle Hamilton-product gradients, or is custom kernel needed?
   Answer: yes — hypercomplex-pytorch library implements this.
