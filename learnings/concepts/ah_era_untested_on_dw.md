# AH Era Gains — Not Retested on ΔW-proj Architecture

**Date compiled:** 2026-04-20  
**Context:** All 6 mechanisms showed positive signal in AH era (steps 16–131, D=64/N=1024 arch).  
None tested on current efficiency config: N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj + α_AH=1.0.

Current arch forward loop:  
`dw = normalize(W_h[i] − W_h[conn_hh[i]])`, `Z_agg = Σ Z_nb × |proj_coeff|`, α_reflect=0.5 buffer.

---

## 1. Z-bias per K_iter (step106 → step937)

**AH era result:** +7.42pp at N=1024 D=64 K_iter=12 (83.36% → 90.78%)  
**Mechanism:** `Z_t += emb[t]` — learned D-dim additive bias per iteration step  
**Why superseded:** Labeled "superseded by ΔW-proj" in 2026-04-14 audit, but never tested ON ΔW-proj  
**Risk:** None — additive, no gate-death  
**Param cost:** K_iter × D = 5 × 16 = 80 extra params (negligible)  
**Adapt for ΔW-proj:** Add `nn.ParameterList([nn.Parameter(torch.zeros(D)) for _ in range(K_iter)])`.
In K_iter loop at step t: `Z = Z + self.zbias[t]` before routing.  
**Hypothesis:** Per-step bias lets early iterations coarse-route, late iterations refine.
Different specialization may emerge vs AH arch — ΔW-proj already provides directional routing.

---

## 2. Refractory Neurons (step16E → step938)

**AH era result:** +4.31pp at D=16 N=512 (step16E: β=0.7, α_r=2.0)  
**Mechanism:** Neurons fired strongly previous step get suppressed: `Z_t = Z_t − α_r × max(0, β × |Z_{t-1}|)`  
**Why superseded:** Old wave/phase architecture. Never ported to ΔW-proj era.  
**Risk:** Low — additive suppression, no gate-death  
**Param cost:** 2 scalars (β, α_r) or fixed values  
**Note:** α_reflect already acts as memory buffer; refractory is complementary "forgetting" side.
May compound with AH (spatial diversity) since refractory handles temporal diversity.  
**Hypothesis:** Prevents same nodes dominating routing across all K_iter steps.

---

## 3. Signed ΔW Routing (step128/131-A → step939)

**AH era result:** +3.97pp T1 at N=1024 D=64 AH arch (weighted_neg: `Z_fwd = relu(Z-θ) + β·relu(θ-Z)`)  
**Mechanism in AH era:** Allow sub-threshold negative activations to contribute to routing  
**ΔW-proj equivalent:** Replace `|proj_coeff|` with signed `proj_coeff` in routing weight:
`Z_agg = Σ Z_nb × proj_coeff` (signed, not abs).  
Negative `proj_coeff` = "anti-aligned neighbor" — node moving opposite direction.  
**Warning:** Step23 showed signed coupling + K_iter≥8 = mode collapse, BUT that was N²-full coupling.
ΔW-proj with K_hh=2 local neighborhood has small fan-in — different stability regime.
Also: step895 killed softmax-based routing variants, but signed ΔW-proj structurally different.  
**Risk:** Medium — worth 20ep T0 first  
**Change required:** 1 line in ΔW-proj forward: `coeff = proj_coeff` instead of `coeff = proj_coeff.abs()`  
**Hypothesis:** Anti-aligned neighbors carry complementary class info. Negative-sign inclusion analogous to excitation/inhibition in biology.

---

## 4. Input-Conditioned Edge Reweighting (step36 → step940)

**AH era result:** +2.34pp at D=64 without AntiHebb (step36: `w_ij = sigmoid(x_i · x_j / tau)`)  
**Mechanism:** Edge weights modulated by raw input feature similarity (not Z state)  
**Key distinction from killed gate-death family:**  
- Gate-death kills: `g = σ(W·Z)` → `g^{K_iter} → 0` (Z-state, repeated application)  
- Step36 uses `x_input` (fixed per sample, not updated by K_iter) — no iterative compounding  
- Step898 tested edge gating with Z-based bias → neutral (−0.05pp). Input-based untested.  
**Warning:** step904 node-level input gating → −1.55pp (bottleneck, not gate-death).
That gated Z before K_iter, not edge reweighting.  
**Risk:** Low-medium  
**Adapt for ΔW-proj:** Before K_iter loop, compute `w_ij = sigmoid(x_scattered_i · x_scattered_j / tau)` for each conn_hh edge. Scale `Z_agg` by `w_ij`.  
**Hypothesis:** Input similarity makes routing data-aware at graph edge level without touching Z.

---

## 5. Sparse Beam / Active Set (step25 → step941)

**AH era result:** +2.88pp AND 53× FLOP reduction at route=64 (1/8 of N=512 nodes active per step)  
**Mechanism:** Each K_iter step, only top-K nodes by Z-activation magnitude route; others hold state  
**Why beam was "killed":** Old `STALE` audit entry refers to wave-arch beam (step13/44) where beam + signed coupling + K_iter≥8 was catastrophic. Mechanism itself never killed on ΔW-proj.  
**Relevance to paper:** *Efficiency* mechanism. If maintains accuracy at 1/4–1/8 active nodes,
reduces K_iter compute further — could push FLOPs below 0.20M.  
**Risk:** Low for accuracy (step25 showed +2.88pp — top-K nodes carry most signal). Medium for implementation.  
**Adapt for ΔW-proj:** Each K_iter step: `active_mask = topk(Z.abs().sum(-1), k=N//8)`.
Non-active nodes: Z_new = Z_prev (skip routing). Only update active nodes.  
**Hypothesis:** High-magnitude nodes already informative; routing through them sufficient.
Aligns with SGNNET's "sparse signal" motivation.

---

## 6. Positional Max-pool Readout (step150 → step942)

**AH era result:** Step150 was QUEUED in audit but never actually run  
**Mechanism:** Instead of mean-pool over all N nodes, take max-pool (or soft-max via attention)  
**Context:** Mean-pool confirmed load-bearing (step155). Max-pool untested.  
**Note:** Step907/910/911 readout gate T0→T1→T2 showed T1 artifact (+0.61pp T1, +0.15pp T2).
Max-pool different — permutation-invariant aggregation, not gating.  
**Risk:** Low  
**Hypothesis:** Max-pool gives strongest class signal; mean-pool dilutes with background nodes.

---

## Summary Table

| Step | Mechanism | AH Gain | Risk | Priority |
|------|-----------|---------|------|----------|
| 937 | Z-bias per K_iter | +7.42pp (AH arch) | None | **1st** |
| 938 | Refractory neurons | +4.31pp (AH arch) | Low | **2nd** |
| 939 | Signed ΔW routing | +3.97pp T1 (AH arch, indirect) | Medium | **3rd** |
| 940 | Input edge reweighting | +2.34pp (AH arch, no AH) | Low-Med | **4th** |
| 941 | Sparse beam active set | +2.88pp + 53× speedup | Medium | **5th** |
| 942 | Max-pool readout | not run | Low | **6th** |

All T0 configs: 20ep, 50% data, N=2048 D=16 K_hh=2 K_iter=5 ΔW-proj + α_AH=1.0.  
Advance rule: ≥+0.5pp vs canonical Ref (Imagenette) → T1.

## Important Caveat
AH arch was D=64/N=1024/K_iter=12 — VERY different operating regime than current
D=16/N=2048/K_iter=5 efficiency config. High D=64 and K_iter=12 provided rich signal amplifying
these mechanisms. At D=16, all gains may be smaller. Treat AH era gains as directional signal only —
not predictive of magnitude.