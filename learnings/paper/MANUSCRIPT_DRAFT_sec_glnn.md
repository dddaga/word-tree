## §7 Generalization: SGNNET as Teacher

We evaluate whether SGNNET's learned representations can be transferred to downstream MLP students via knowledge distillation (KD).

**Setup.** Imagenette 10-class classification. Teacher: canonical SGNNET (step887, 96.38%). Students: MLP with hidden dimension h, trained from scratch with (a) cross-entropy on hard labels, or (b) KD loss against SGNNET soft labels at temperature T=2.0, weight λ=0.5.

**Results (step621).**

| Student | h | Params | Scratch | + SGNNET KD | Δ |
|---------|---|--------|---------|-------------|---|
| MLP_large | 256 | 6.4M | 97.53% | 97.35% | −0.18pp |
| MLP_tiny | 2 | 50K | 70.50% | 9.86% | −60.64pp |

**Analysis.** SGNNET soft labels do not improve a large MLP student (neutral within noise). For a tiny student (h=2), SGNNET's soft labels cause catastrophic collapse: the student falls to near-random accuracy (9.86%). This is the inverse of the VGG16 → SGNNET distillation case, where soft labels provide a strong training signal.

**Why distillation fails in the reverse direction (HYPOTHESIS).** VGG16 is trained on ImageNet (Imagenette ⊂ ImageNet): its confidence is well-calibrated, class boundaries sharp. SGNNET's probability distributions reflect routing geometry rather than calibrated class probabilities — a soft label at T=2 amplifies SGNNET's near-uniform off-diagonal probability mass, which adds noise rather than dark knowledge.

For the tiny student (h=2), the distillation objective overpowers the sparse signal in only 2 hidden units, collapsing the network to a degenerate solution. Large students (h=256) can absorb the noise — but gain nothing.

**Verdict.** SGNNET is an effective distillation *student* (from VGG16 teacher: step605 K=1 KD at 95.95%) but a poor *teacher* (step621: neutral for large students, catastrophic for small ones). This asymmetry is consistent with the soft-label equivalence result (step977: KD T=1 ≈ hard CE for Imagenette) — SGNNET's soft labels carry no information beyond what's already in the hard labels.

**Cross-dataset note.** We do not extend GLNN distillation to CIFAR-10 or CIFAR-100: given the neutral result on Imagenette and the CIFAR-100 structural gap (§8.2), cross-dataset GLNN distillation is unlikely to yield positive results.
