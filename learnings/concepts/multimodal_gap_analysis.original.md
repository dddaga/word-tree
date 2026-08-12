# Multimodal Gap Analysis — ESC-50 and CIFAR-100

**Status:** Living document. Last updated 2026-04-20 (session 28).

## Summary

SGNNET works on Imagenette (VGG16, 10 classes) but fails on ESC-50 (Whisper, 50 classes) and CIFAR-100 (VGG16, 100 classes). Understanding why requires distinguishing three failure modes:

1. **Dense input incompatibility** — ESC-50 (Whisper 384-dim)
2. **Readout capacity bottleneck** — CIFAR-100 (100 classes, D=16)
3. **Routing = identity on current metrics** — all modalities (cosine sim ≈ 1.0)

---

## GA Experiment Results

### Step929 — ESC-50 K_hh/K_iter Scout T0

| Config | Acc | Δ vs Linear |
|--------|-----|-------------|
| Linear | 47.75% | 0.0 |
| K2I5   | 37.50% | −10.25pp |
| K4I5   | 38.50% | −9.25pp |
| K4I7   | 33.50% | −14.25pp |

Best: K4I5 = 38.5%. Gap persists across all K configs (−9 to −22pp).

### Step930 — CIFAR-100 K_hh/K_iter Scout T0

| Config | Acc | Δ vs Linear |
|--------|-----|-------------|
| Linear | 60.30% | 0.0 |
| K4I5   | 42.01% | −18.29pp |
| K4I7   | 41.74% | −18.56pp |
| K2I5   | 40.85% | −19.45pp |

Best: K4I5 = 42.01%. All configs −18 to −28pp vs linear.

### Step931 — GA ESC-50 Hyperparam Search (4 gens, 16 configs)

Best config: `D16_Kin15_Khh2_Ki7_N2048` = 41.5%  
Linear baseline: 47.75% → Gap = **−6.25pp** (best after genetic search)  
K_iter=7 sweet spot confirmed. K_hh=2−4 optimal.

### Step932 — GA CIFAR-100 (stalled — N≥4096 configs take 12min each)

Gen-0 partial: `D16_Kin15_Khh2_Ki5_N4096` = 41.81% vs linear=60.3% (−18.49pp).  
ETA if continued: 16+ hours. Recommend restart with N≤2048.

---

## EDA Critical Finding: Routing is NOT the mechanism (cosine metric)

Cosine similarity analysis (per-node Z vectors):

| Modality | After seed | After K_iter | Routing improvement |
|----------|-----------|--------------|---------------------|
| ESC-50   | within=1.000, across=1.000 | within=1.000, across=1.000 | +0.000pp |
| CIFAR-100| within=0.999, across=0.997 | within=1.000, across=0.999 | −0.001pp |
| Imagenette| within=0.995, across=0.993 | within=0.999, across=0.999 | −0.002pp |

Seed Z variance: ESC-50=0.0000, CIFAR-100=0.0005, Imagenette=0.0022

**Interpretation:** Individual node cosine similarity is NOT the right metric. This was the wrong analysis.

## Activation Postmortem (step943): Correct Metrics

Running postmortem on **untrained Imagenette model** (step943 baseline, session 28):

| Step | Fisher Ratio | Participation Ratio | Probe Acc | 1-NN Acc | L2 Sep |
|------|-------------|---------------------|-----------|----------|--------|
| seed (t=0) | 0.482 | 1.0 | 56.25% | 38.44% | +0.0083 |
| iter1 | 0.241 | 1.5 | 67.34% | 37.03% | +0.0025 |
| iter2 | 0.220 | 1.5 | 64.77% | 44.84% | +0.0030 |
| iter3 | 0.261 | 1.3 | 63.28% | 39.14% | +0.0046 |
| iter4 | 0.448 | 1.9 | 64.06% | 50.31% | +0.0046 |
| iter5 | 0.313 | 3.7 | 64.84% | 44.22% | +0.0025 |

**CKA(seed, final) = 0.686** — routing substantially transforms the representation (not identity!)

### Critical Insights

1. **Routing DOES transform Z** — CKA=0.686 (not 1.0). The cosine EDA was measuring individual node similarity, not the pooled representation structure.

2. **Seeding alone contains class signal** — probe=56.25% with untrained random weights. VGG16 features are already class-separable. Random K_in scatter projects them into N=2048 node activations that preserve linear separability.

3. **One routing step jumps probe 56% → 67%** even without training. Routing is doing useful feature mixing before any learning occurs.

4. **Participation ratio grows seed→1.0 to final→3.7** — routing EXPANDS the representation into more dimensions (not compression). This is the mechanism: routing distributes class signal across more orthogonal dimensions, making it easier for the linear readout.

5. **Fisher ratio does NOT monotonically increase** — routing reorganizes variance in complex ways, but linear separability (probe) is the right target metric.

---

## Root Causes by Modality

### ESC-50 (Whisper 384-dim → SGNNET, 50 classes)

**Primary cause: Dense embedding incompatibility**

- Whisper encoder produces dense 384-dim embeddings. Top-10 PCs = 82.9% variance (effective dim ~30-50).
- K_in=25 scatter onto N=2048 nodes: each node sees 25/384 = 6.5% of features.
- Unlike VGG features (sparse spatial), Whisper embeddings are holographic — the class information is distributed across ALL 384 dims. Dropping 93.5% destroys the hologram.
- VGG features: localized spatial patterns. K_in=25 scatter still captures meaningful spatial features.
- Only 40 samples/class in train — insufficient to learn class-conditional scatter patterns.

**Physical process insight:**  
Whisper already found the compact acoustic manifold (latent dim ~30-50). SGNNET's random scatter is destroying this manifold by projecting it onto an incompatible random basis.

**Fix hypothesis:**  
- Smaller N (N≤256) with high K_in coverage (K_in=192, 50% coverage). Fewer nodes → each node aggregates more features → hologram preserved.
- OR: learned projection (not random K_in scatter) — project 384→D×N via learned linear before scatter.

### CIFAR-100 (VGG16 25088-dim → SGNNET, 100 classes)

**Primary cause: Readout capacity + feature magnitude**

- Same VGG feature type as Imagenette but: std=0.574 vs 1.318 (2.3× lower magnitude).
- D=16, mean-pool → 16-dim vector → linear to 100 classes: 1600 params vs 2,508,900 for baseline linear.
- 100 fine-grained classes need richer representation than 10 Imagenette classes.
- Lower feature magnitude = lower initial Z separation at seed → smaller signal to amplify.

**Physical process insight:**  
CIFAR-100 superclasses (vehicles, animals, etc.) may share physical properties → denser class manifold → higher required representation dimensionality.

**Fix hypotheses:**  
1. D=32 or D=64 (increases readout capacity)
2. Feature normalization to match Imagenette std
3. Soft label KD from VGG top-k (soft_labels already in HDF5)
4. N scaling: N=4096+ (more nodes = richer mean-pooled representation)

---

## Proposed Experiments

### ESC-50 Gap Experiments

| Step | Description | Target | Slots |
|------|-------------|--------|-------|
| step944 | ESC-50 N sweep: N=64/128/256/512 with proportional K_in coverage (K_in=N//4 to N//2) | Can smaller N + higher coverage beat N=2048? | mini_cpu T0 |
| step945 | ESC-50 K_in coverage sweep at N=256: K_in=32/64/128/192 | What % coverage is needed? | mini_cpu T0 |

### CIFAR-100 Gap Experiments

| Step | Description | Target | Slots |
|------|-------------|--------|-------|
| step946 | CIFAR-100 D scaling: D=16/32/64 at N=2048 | Readout capacity bottleneck test | studio_cpu T0 |
| step947 | CIFAR-100 feature normalization (zscore to Imagenette std) | Magnitude hypothesis test | mini_cpu T0 |
| step948 | CIFAR-100 soft label KD (soft_labels already in HDF5) | Does KD from VGG teacher close gap? | studio_cpu T1 |

### Cross-modal

| Step | Description |
|------|-------------|
| step943 | Activation postmortem: Fisher ratio, participation ratio, probe acc, CKA per K_iter step — compare untrained vs trained Imagenette vs ESC-50 |

---

## What Breaks at Each Modality (Summary Table)

| Modality | Seeding problem | Routing problem | Readout problem |
|----------|----------------|-----------------|-----------------|
| Imagenette | ✓ VGG high-SNR features, good scatter | ✓ Routing expands PR, helps separability | ✓ 10 classes, D=16 sufficient |
| ESC-50 | ✗ Dense embedding, random scatter destroys hologram | unknown | moderate (50 classes, D=16) |
| CIFAR-100 | partial (VGG but 2.3× lower magnitude) | unknown | ✗ 100 classes, D=16 severe bottleneck |
