# complex_weights — multiply-as-addition via complex phase

Goal: energy-efficient DL. Replace multiplies with additions using Euler:
unit-magnitude complex multiply = **phase addition**. Vehicle = foreground
segmentation (FgSegNet_v2 family). Win on params + compute, hold accuracy.

## The one idea
w = exp(iθ), x lifted to z = exp(iφ). Then w·z = exp(i(θ+φ)) — multiply → add.
Conv accumulates phasors, project real: `y = Σ cos(θ+φ) = Re(Σ exp(i(θ+φ)))`.
Cosine is not ad-hoc — it is Re() of the phasor sum.

## Science vs systems (do not conflate)
- **Science (Phase 0):** does phase-addition preserve information at iso-param?
  PyTorch reference (`phaseconv.py`) computes Re(S) EXACTLY but via real convs
  (multiplies under the hood). Faithful for ACCURACY. Says nothing about energy.
- **Systems (Phase 2):** add-not-mul + LUT phases + integer-overflow rotation,
  measured as Joules/bytes on 5060ti. The energy claim lives only here.

## Files
- `phaseconv.py` — `PhaseConv2d` (variant real|phase), `encode_phase` (wrap=overflow-rotate).
- `models.py` — `TinySegNet`, per-layer real↔phase swap.
- `data.py` — synthetic foreground task (fast proxy) + F-measure.
- `train.py` — baseline vs single-layer swap; `--sweep` scans all 6 layers.

## Roadmap
- **P0a** synthetic sanity: does a single PhaseConv layer train + match real? (iso-param)  ← RUNNING
- **P0b** which layers tolerate phase? full-phase net ceiling on synthetic.
- **P0c** real data: CDnet2014 sequence, FgSegNet_v2-style net, F-measure vs real.
- **P1** log-magnitude 2nd channel (mag product→sum); phase-native propagation.
- **P1** quantize θ to R roots-of-unity → low-bit weights (param-memory win).
- **P2** custom CUDA/JAX add-kernel; measure Joules vs real conv at iso-accuracy.

## Evidence tags (CONFIRMED / HYPOTHESIS / STALE) per neuro_graph tenet.
