# Paper Campaign — Scale + Veteran-CNN Efficiency (2-day box)

**Owner:** Dhiraj · **group_id:** dhiraj · **Opened:** 2026-08-13 · **Deadline:** 2026-08-15 (2 days)
**Goal:** conclude all remaining SGNNET + GLAM research, produce the scaled efficiency
comparison, and hand the paper a finished results section.

> **Execution note (user, 2026-08-13):** this file is a RECORDED PLAN, not a launch order.
> For the next 2 days the work stays on the /goal — iterate the algorithm (GLAM mechanisms,
> starting with mechanism 4 energy) on the datasets already on disk. The large-dataset scaling
> (ImageNet / veteran backbones, phases P1–P7 below) runs AFTER the 2-day iteration, using the
> BEST version produced. Do not launch feature extraction or a dataset download yet.

## Thesis (one line)
A structured, shared-pool FC-head replacement (GLAM locality champion / SGNNET) matches a dense
FC classifier at a fraction of the params/FLOPs/energy — and the gap does not widen with scale.

## Invariant methodology (points 6–7, fixed, do NOT touch)
1. **Conv backbone frozen.** Never trained, never modified. It is a fixed feature extractor.
2. **Backbone output = our input.** Extract features once → HDF5 store, then train only the head.
3. **Labels = targets.** Standard classification CE on the dataset's classes.
4. **Our custom layer fills the FC gap.** Replace the veteran's dense classifier head with the
   GLAM locality champion (shared-pool structured projection) / SGNNET head. Everything else held.
5. **Judged on the full Pareto row** — accuracy + params + FLOPs + wall-time + peak memory + Joules.
   An arm losing accuracy while winning params/FLOPs/energy stays alive.

## Veterans (backbones under test)
Each contributes a frozen feature tensor + its native dense FC-head as the within-tier anchor.

| Backbone | Feature dim (pre-FC) | Native FC head | Role |
|---|---|---|---|
| VGG16 | 25088 (512×7×7) | 25088→…→1000 | current baseline, have Imagenette+CIFAR stores |
| AlexNet | 9216 (256×6×6) | 9216→4096→4096→1000 | classic heavy-FC veteran (biggest FC to beat) |
| ResNet-18 | 512 (GAP) | 512→1000 | modern lean head — hard case (little FC to save) |
| ResNet-50 | 2048 (GAP) | 2048→1000 | scale rung, moderate head |

ResNet's GAP heads are deliberately lean → honest hard case for a head-replacement claim. Report
even where we do NOT win: the story is "FC-heavy nets (VGG/AlexNet) win big; GAP nets already lean."

## Datasets — phased small→large (point 5)
Rung up only after the prior rung's ranking is stable; carry the same arms each rung.

| Rung | Dataset | Classes | Why | Gate to advance |
|---|---|---|---|---|
| R0 | Imagenette | 10 | have it, ceiling saturated | DONE (97.68% champ) |
| R1 | CIFAR-10 | 10 | have it, headroom | DONE (step005 verdict) |
| R2 | ImageNet-100 | 100 | first real-scale sweep, cheap | champ ≥ dense−1pp @ ≤5% params |
| R3 | ImageNet-1k | 1000 | terminal claim, ≥1 backbone | gap does not widen vs R2 |

**Default terminal target:** ImageNet-1k on VGG16 + AlexNet (FC-heavy → where the claim is
strongest); ImageNet-100 across all four backbones for the broad sweep. Override if compute box
forces ImageNet-100 as terminal — state it, don't silently cap.

**Version-selection for scaling (CONFIRMED CIFAR-100 T2, step007 capacity sweep):** the "best version"
to carry is CLASS-COUNT-DEPENDENT, not a single fixed champion. On-disk CIFAR-100 gives an accuracy↔params
Pareto frontier for the locality head at 100 classes: d_out=8 65.35% (416K, −3.14pp vs dense) → d_out=16
66.27% (832K, −2.22) → d_out=32 66.44% (1.66M, −2.05). Returns log-diminish; M is floor-only (inert
diversity). **The gap is STRUCTURAL** (shared-pool compresses 25088→P·d_out bottleneck before the readout;
dense is full-rank), so at ImageNet-1k the readout term P·d_out·1000 dominates even more and the gap likely
widens further. **Scaling protocol: carry the FRONTIER (sweep d_out), not one point — report the
accuracy-vs-params curve at R2/R3, not a single champion. The honest claim is a tunable
compression↔accuracy tradeoff, strongest at low class count (near-free ≤10cls), costing a structural
~2pp at 100cls that capacity only partly recovers.** Do NOT over-claim scale-invariance: it holds at low
class count, breaks at high (T2, not T1).

## Two-day phase plan
**Day 1 — infrastructure + R2 sweep**
- P1. Feature stores: extract frozen features for {VGG16, AlexNet, ResNet-18, ResNet-50} ×
  {ImageNet-100}. One-time GPU pass on 5060ti when free; write `data/store_in100_<backbone>.h5`.
- P2. Head arms per backbone: `DENSE` (native FC anchor, within-tier), `GLAM-LOC` (locality
  champion), `SGNNET` (graph head). T0 (20ep/50%) rejection filter first.
- P3. Advance neutral/positive arms to T1 (75ep/50%). Full Pareto row each.

**Day 2 — R3 terminal + energy + write-up**
- P4. ImageNet-1k feature stores for VGG16 + AlexNet. T1 → T2 (150ep/100%) for the two head arms
  that survived R2.
- P5. Energy arm (GLAM mechanism 4, slope-anneal) measured on the champion head: Joules/inference
  + activation-zero fraction (meditation-005 methodology), NOT accuracy. Last open GLAM arm.
- P6. Benchmark: wall-time (bench_step608 harness) + peak memory + int8 size for each survivor.
- P7. Assemble the scaled Pareto table (accuracy vs dense at each scale, params/FLOPs/energy) →
  paper results section. Confirm the scale-invariance claim (gap flat R2→R3) or report where it breaks.

## Open forks (defaulted, user may override)
- **Terminal scale:** default ImageNet-1k on 2 FC-heavy backbones; fallback ImageNet-100 if the
  compute box (2 days, 1 GPU slot) can't fit 1k extraction + T2. Decide after P1 timing.
- **Data source:** ImageNet-100 = standard 100-class subset; needs local ImageNet tarballs or a
  torchvision/HF pull. If unavailable, substitute a comparable 100-class set (e.g. Food-101,
  Places-100) and note the swap.

## Success = paper-ready
- Scaled Pareto table across ≥3 datasets × ≥3 backbones.
- Scale-invariance claim CONFIRMED or its breaking point reported.
- Energy arm scored on real Joules.
- Every causal claim tagged CONFIRMED / HYPOTHESIS / STALE.

## Cross-refs
- GLAM verdict: `learnings/concepts/glam_grouped_multiplicative_routing.md`
- Champion ladder + energy methodology: `learnings/EXPERIMENT_QUEUE.md`, meditation-005.
