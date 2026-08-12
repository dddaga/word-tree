# Qwen Failure Modes — Do NOT Do These

## Architecture confusion

**WRONG:** "SGNNET trains on raw images"
**RIGHT:** SGNNET receives pre-extracted VGG16 conv features (25088-dim). It is a classification head, not a vision model.

**WRONG:** "FLOPs comparison is end-to-end"
**RIGHT:** All FLOPs numbers are FC-head-only. VGG16 conv cost is identical for both and not counted.

**WRONG:** "SGNNET prunes VGG16 weights"
**RIGHT:** SGNNET is a different architecture replacing only the FC head. The conv weights are frozen and never touched.

**WRONG:** "K_in=25 edges are learned"
**RIGHT:** All connectivity (K_in input edges, K_hh hidden edges) is fixed random at init. Not learned.

## Claim status confusion

**WRONG:** Treating a HYPOTHESIS as CONFIRMED
**RIGHT:** Claims are tagged CONFIRMED (clean ablation), HYPOTHESIS (post-hoc), or STALE. Only CONFIRMED claims are paper-ready.

**WRONG:** Suggesting experiments already done
**RIGHT:** Check the DONE section of baselines_needed.md before suggesting any experiment.

## Baseline confusion

**WRONG:** Comparing SGNNET accuracy to VGG16 fine-tuned end-to-end
**RIGHT:** Compare SGNNET head vs VGG16's FC head (same frozen features as input).

**WRONG:** "Linear probe" means logistic regression on raw pixels
**RIGHT:** Linear probe = linear layer on VGG16 conv features. Same features as SGNNET.

## JL theory

**WRONG:** "K_in=25 per neuron is a Johnson-Lindenstrauss embedding"
**RIGHT:** The N=2048 ensemble collectively forms a sparse random projection. JL-style guarantees (Achlioptas 2003) apply to the ensemble, not per-neuron.

## Distillation

**WRONG:** "Teacher is VGG16"
**RIGHT:** Teacher is SGNNET (ΔW K=5). Student is a small MLP. VGG16 soft labels are used as targets for both — that's not KD, that's label generation.

## Topology

**WRONG:** "Small-world topology is the Watts-Strogatz model"
**RIGHT:** SGNNET uses a custom ring-based sparse topology with K_hh=2 neighbors. Similar concept, not identical.

## Scale

**WRONG:** Assuming N=4096 is the default
**RIGHT:** Default is N=2048, D=16, K_in=25 for Imagenette.

## Scope

**WRONG:** Suggesting text/audio SGNNET as a solved problem
**RIGHT:** Text gap (SST-2/AG News) is confirmed NEGATIVE — SGNNET trails linear on text. Audio gap is structural and under investigation. Paper scope is vision-only until further results.
