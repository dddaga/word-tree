# Parallel Line: CNN Compressibility — Multi-Branch Full-Context CNNs

**Status: ACTIVE. Isolated line — own step numbers (cnnc_stepNNN), own queue, own sub-agents.**
Created 2026-06-10 per Dhiraj's directive. Third parallel line (main SGNNET / ffn_baseline / this).

## Hypothesis
CNN module itself is compressible. Convolutions are local + translation-invariant but
miss FULL image context. Pathak et al. 2016 (Context Encoders, inpainting) used a
channel-wise fully-connected layer to propagate global context cheaply. Adding a
parallel branch operating on aggressively-downsampled image gives full-context signal
at tiny cost → push parameter-efficiency limits of CNNs with standard, existing tricks.

## Design space
- **Multi-branch**: several lightweight branches over the image at different receptive
  fields / downsample rates (e.g., full-res shallow local branch + 4×-down mid branch +
  16×-down "global context" branch with channel-wise FC à la Context Encoders).
- **Aggressive downsampling** early; spend budget on DEPTH not width.
- **CReLU** (concatenated ReLU, Shang et al. 2016): conv filters in early layers come in
  negative pairs — concat(relu(x), relu(−x)) halves filters for same expressivity.
- **PVANet** (Kim et al. 2016): CReLU + Inception + deep-narrow design, 1/10 compute.
- Also in scope: dilated convs, depthwise-separable convs, channel reduction.

## Evaluation
Same 5-dim Pareto: accuracy + params + FLOPs + wall-time + memory.
Dataset: Imagenette (match main line). Baselines: VGG16 (full), existing cnn_distiller
line results (scripts/cnn_distiller/), SGNNET feature-head numbers as context.
Tier protocol identical: T0 20ep/50% → T1 75ep/50% → T2 150ep/100%.

## Compute
Jobs submitted via scheduler (`scripts/scheduler/submit.py cnn_compress <script>`),
round-robin with other lines over slots {5060ti_cuda, mini_mps, mini_cpu}.
CNN training on raw images is heavier — prefer 5060ti_cuda slot.

## Management
Designed + iterated by dedicated sub-agents. Research independent of other lines;
only compute shared.

## References
- Pathak et al. 2016, Context Encoders: Feature Learning by Inpainting (arXiv:1604.07379)
- Shang et al. 2016, Understanding and Improving CNNs via CReLU (arXiv:1603.05201)
- Kim et al. 2016, PVANET: Deep but Lightweight Neural Networks (arXiv:1608.08021)

## Queue
See [QUEUE.md](QUEUE.md).
