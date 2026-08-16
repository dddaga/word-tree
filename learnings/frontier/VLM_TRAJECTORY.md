# Visual-LLM Trajectory — SGNNET learnings → VLM → drone

**Created:** 2026-07-28 · **Branch C of the /goal directive** ("use a visual LLM and prune it
down with our learnings so it runs parameter-efficiently on a drone")
**Target model:** `HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3: SigLIP vision + Llama-576 text).

---

## 0. Thesis going in, and what the Amdahl gate did to it

The plan inherited from Branch B was: *find the VLM's readout slot, replace it with a compact
SGNNET-style head, ship the smaller model.* That plan is **CONFIRMED dead** by `vlm_step001`.
The plan that replaced it is on a different axis — **tokens, not weights**.

The organizing principle is unchanged and is the load-bearing result of this whole directive:

| Slot | Shape | SGNNET verdict | Evidence |
|---|---|---|---|
| **READOUT** | pooled features → output space | **transfers** | step605 (VGG FC), det_step002 (box head) |
| **TRANSFORM** | attention/MLP inside a block | **does not transfer** | step989, llm_step002 |
| **EMBEDDING** | metric/lookup space | **does not transfer** (top-k collapses it) | reid_step001 |

## 1. `vlm_step001` — where a small VLM's params actually live

`scripts/frontier/vlm_step001_role_profile.py`, 256,484,928 params, 0 weight-tied tensors.
Classification is by **role**, category rule, no per-model hardcoding. **CONFIRMED (measured):**

| Role | Params | % model | Slot verdict |
|---|---|---|---|
| TRANSFORM_TEXT | 106,168,320 | **41.39%** | KILLED role (step989 / llm_step002) |
| TRANSFORM_VISION | 85,017,600 | **33.15%** | KILLED role |
| EMBEDDING | 29,762,304 | 11.60% | lookup table → quantize, don't distil |
| READOUT_LMHEAD | 28,385,280 | 11.07% | addressable (not tied) |
| READOUT_PROJECTOR (connector) | 7,077,888 | 2.76% | addressable |
| NORM | 73,536 | 0.03% | — |

Submodules: `text_model` 134.59M (52.47%) · `vision_model` 86.43M (33.70%) ·
`lm_head` 28.39M (11.07%) · `connector` 7.08M (2.76%).

**SGNNET-addressable (readout) share = 13.83% (35.46M params).** Projecting whole-model savings
at the ratios *measured* in det_step002/003 — not assumed:

| Recipe (measured keep-ratio) | Model after | Reduction | Compression |
|---|---|---|---|
| dense_lowrank_r256 (23.6%) | 229.39M | 10.56% | **1.118×** |
| sgn_topk_r128k32 (11.7%) | 225.17M | 12.21% | **1.139×** |
| sgn_topk_r64k16 (5.8%) | 223.08M | 13.03% | **1.150×** |

> **CONFIRMED — readout-only VLM pruning is Amdahl-dead.** The recipe that addressed **72%** of a
> Faster-RCNN (det_step001) addresses **13.83%** of a VLM. Even a perfect readout (zero params)
> caps at 1.16×. 74.5% of the model sits in TRANSFORM blocks, the one role our own evidence says
> SGNNET cannot take over. Do not spend the drone budget here.

## 2. The redirect — the profile named the real lever

The same probe measured the runtime side: **1 image = 1135 tokens vs 4 text tokens** (Idefics3
image-splitting: 17 tiles × 64). A VLM's drone cost is **token-bound, not param-bound**.

That reframes the SGNNET top-k idea onto an axis where it can actually pay:

- **WEIGHT axis (det_step003, CONFIRMED on MPS / HYPOTHESIS on CUDA — MPS-only bench):** hard top-k masking does not shrink a dense
  matmul. MACs track *rank*, not *k*; the sort cost exceeded the rank-halving saving, so
  `sgn_topk_r128k32` was **slower** than `lowrank_r256` at half the params.
- **TOKEN axis (this branch):** dropping a token genuinely removes work — a shorter sequence is
  strictly less attention and less FFN. **No sparse kernel required.** Same selection idea
  (keep the top-k by magnitude), moved to where dense hardware rewards it.

## 3. `vlm_step002` — image-token budget sweep (T0)

`scripts/frontier/vlm_step002_token_budget.py`. Reference = `split17_full` (1088 image tokens,
seq 1171). Arms:

| Arm | What it changes | Why it is in the table |
|---|---|---|
| `topk_{512,256,64}` | keep top-k image tokens by post-connector L2 norm | the SGNNET selection rule, on the token axis |
| `stride_64` | even stride, same 64-token budget as `topk_64` | **CONTROL** — top-k must beat it, or only *how many* matters, not *which* |
| `nosplit_64` | `do_image_splitting=False` | same 64-token budget, but vision runs 1 tile not 17 → cuts vision cost too |

Metrics: imagenette top-1 by **constrained 10-way label likelihood** (length-normalised
teacher-forced log P over each label's own tokens, reusing the prefill KV cache) — free
generation is uninformative here, SmolVLM-256M answers *"Frog"* for a tench; plus next-token
agreement and KL vs the reference, and **vision / prefill wall-time reported separately**
(token pruning cuts only prefill; `nosplit` cuts both — the honest Pareto split).

**Validation:** the manual embed-splice used by every pruned arm reproduces the native
`model(**batch)` forward **bit-exactly** (max abs diff 0.0), so arm deltas are the pruning, not
the harness.

### Results (mini_mps, n=50 imagenette-val)

| arm | imgtok | seq | top1 | same | agree | KL | vis_ms | pre_ms | total | speed |
|---|---|---|---|---|---|---|---|---|---|---|
| split17_full (ref) | 832* | 906 | 0.660 | 1.000 | 1.000 | 0.000 | 978.2 | 147.2 | 1125.4 | 1.00× |
| topk_512 | 512 | 586 | **0.680** | 0.940 | 0.860 | 0.202 | 978.2 | 63.7 | 1041.9 | 1.08× |
| topk_256 | 256 | 330 | **0.680** | 0.760 | 0.700 | 0.588 | 978.2 | 38.6 | 1016.8 | 1.11× |
| topk_64 | 64 | 138 | 0.440 | 0.400 | 0.520 | 1.957 | 978.2 | 21.0 | 999.2 | 1.13× |
| stride_64 (CONTROL) | 64 | 138 | 0.260 | 0.240 | 0.360 | 3.111 | 978.2 | 16.7 | 994.8 | 1.13× |
| **nosplit_64** | 64 | 111 | **0.740** | 0.740 | 0.660 | 0.604 | **78.5** | 17.6 | **96.1** | **11.71×** |

\*`image_tokens` logs the last image's tile count, not the mean — Idefics3 splitting is
aspect-ratio dependent. Budget arms are exact (k fixed).

**Three CONFIRMED results:**

1. **The selection rule transfers.** magnitude top-k beats the stride control at an *identical*
   64-token budget: **0.440 vs 0.260 top-1 (+18pp), KL 1.96 vs 3.11.** *Which* tokens survive
   matters, not merely how many. This is the SGNNET top-k idea earning its keep on an axis where
   dense hardware actually pays for it — and 512/256 tokens are **free**: top-1 0.680 ≥ the
   832-token reference's 0.660.
2. **Token pruning alone is Amdahl-capped at ~1.13×.** Prefill is only **13%** of per-image
   latency (147 of 1125 ms); the vision encoder (978 ms, 87%) is untouched by dropping
   post-connector tokens. Cutting 832→64 tokens cuts prefill 7× and total latency 12%.
   *Second Amdahl gate, same lesson as §1: measure the addressable share before optimizing it.*
3. **`nosplit_64` dominates the whole table** — 11.71× total speedup **and** +8pp top-1 over the
   full reference, at KL 0.604 (better fidelity than `topk_64` at the same token budget).
   Attacking **tile count** hits both stages (vision 978→78 ms); token pruning hits one.

> HYPOTHESIS (untested): the accuracy *gain* from `nosplit` is because 13-tile splitting feeds a
> 10-way whole-object task redundant high-res crops it cannot use. Expect the sign to flip where
> fine detail matters (OCR, small-object VQA) — the drone case is closer to the latter, so this
> must be retested on a detection-flavoured benchmark before it becomes a deployment rule.

## 4. `vlm_step003` — vision-encoder DEPTH (T0)

`scripts/frontier/vlm_step003_vision_depth.py`. step002 pointed here: the vision tower is 87% of
latency and 33.15% of params, so it is the only lever with room left. SmolVLM-256M's tower =
**12 SigLIP layers × 7,087,872 params**; dropping 6 removes 42.5M = **16.58% of the whole model,
more than the entire 13.83% readout share §1 declared Amdahl-dead.**

Truncation is legitimate where SGNNET substitution is not: this does not ask a sparse head to
imitate a TRANSFORM block (step989/llm_step002 KILLED that), only whether the block is *needed*.
The connector consumes hidden_size=768, which every intermediate layer already emits, so layer
L's output feeds it unchanged. Reference = the step002 champion (`nosplit`, 64 tokens).

### Results (mini_mps, n=50 imagenette-val, chance = 0.100)

| arm | tok | params | %save | top1 | same | agree | KL | vis_ms | pre_ms | total | speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d12_full (ref) | 64 | 256,484,928 | 0.00 | **0.740** | 1.000 | 1.000 | 0.000 | 66.4 | 14.6 | 81.0 | 1.00× |
| d9 | 64 | 235,221,312 | 8.29 | 0.080 | 0.180 | 0.000 | 9.973 | 49.9 | 14.2 | 64.1 | 1.26× |
| d6 | 64 | 213,957,696 | 16.58 | 0.120 | 0.140 | 0.000 | 10.669 | 35.0 | 14.1 | 49.1 | 1.65× |
| d3 | 64 | 192,694,080 | 24.87 | 0.100 | 0.080 | 0.000 | 12.457 | 19.9 | 14.1 | 34.0 | 2.38× |
| d12_topk32 | 32 | 256,484,928 | 0.00 | 0.680 | 0.920 | 0.860 | 0.138 | 66.4 | 12.9 | 79.2 | 1.02× |
| d12_topk16 | 16 | 256,484,928 | 0.00 | 0.600 | 0.780 | 0.700 | 0.464 | 66.4 | 11.4 | 77.7 | 1.04× |
| d6_topk32 | 32 | 213,957,696 | 16.58 | 0.100 | 0.120 | 0.020 | 10.183 | 35.0 | 11.8 | 46.8 | 1.73× |

**Cross-validation:** `d12_full` 0.740 reproduces step002's `nosplit_64` 0.740 exactly on an
independent run → the harness is sound and the two tables are directly comparable.

**Two CONFIRMED results:**

1. **Zero-shot vision-tower truncation collapses to chance at every depth tested.** d9 0.080,
   d6 0.120, d3 0.100 against chance 0.100, with KL 9.97–12.46 — removing even **3 of 12** layers
   is fatal. The params are removable (16.6–24.9% of the model, at 1.26–2.38× latency); the
   *function* is not free. post_layernorm and the connector receive an unadapted distribution.
2. **Sharp axis asymmetry.** Halving tokens costs 6pp at KL 0.138; quartering costs 14pp at
   KL 0.464; removing a quarter of the tower costs **62pp at KL 10.7.** Cutting *what the tower
   already computed* is survivable. Cutting *the computing* is not.

> **HYPOTHESIS → the truncated tower needs distillation, not deletion.** Apply the recipe that
> worked twice (step605 K=1 soft-KD, det_step002 box head) to a TRANSFORM stack's **OUTPUT**
> rather than its internals — which is the exact thing step989/llm_step002 killed. → `vlm_step004`.

**Two honest caveats.**
(a) `d6_topk32` is **CONFOUNDED, not a compounding test** — d6 alone is already at chance, so the
arm cannot separate composition from d6's own collapse. The Compounding Rule question for
depth × tokens stays **OPEN** until a depth arm is above chance.
(b) On the nosplit champion the tower is 66.4/81.0 ms = **82% of latency**, so the token axis buys
only 1.02–1.04× wall-time here; its value in this config is KV-cache and memory, not speed.
**step002's 1.13× must NOT be quoted for the nosplit config** — that figure was measured against
the 17-tile reference.

---

**Continues in [`VLM_TRAJECTORY_part2.md`](VLM_TRAJECTORY_part2.md)** — §5 `vlm_step004` (tower
distillation, the answer to the HYPOTHESIS above), §6 drone recipe, §7 next steps. Split at the
200-line file limit; part 1 is the profile + the two pruning-axis sweeps, part 2 is the repair.
