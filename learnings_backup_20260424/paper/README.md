# Paper Materials — SGNNET

**Working title (2026-04-14 revised):** "SGNNET: A Sparse Graph Classification Head for Pre-Trained Feature Extractors — Sub-1% FLOPs Across Vision and Language"

**Core thesis (2026-04-14):** SGNNET replaces the fully-connected classification layer of pre-trained feature extractors at a fraction of FC FLOPs/params, with matched or better accuracy. Validated across FOUR modalities using latest-available models (April 2026):

| Modality | Feature extractor | HF ID | Params | Feature dim | Dataset | HF dataset ID | Size | Classes | Status |
|---|---|---|---|---|---|---|---|---|---|
| Vision (CNN) | VGG16 conv backbone | `torchvision/vgg16` | 14.7M conv | 25088 | Imagenette | — | 9.4K/3.9K | 10 | ✅ DONE: 95.52% @ 0.98M routing MACs |
| Text (encoder) | ModernBERT-base | `answerdotai/ModernBERT-base` | 149M | 768 (CLS or mean-pool) | AG News | `fancyzhx/ag_news` | 120K/7.6K | 4 | PENDING |
| Audio | Whisper-tiny encoder | `openai/whisper-tiny` | 39M total (encoder only for feats) | 384 (encoder last-hidden, mean over time) | Speech Commands v2 | `google/speech_commands` | 85K/11K | 35 | PENDING |
| LLM (decoder) | Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 600M | 1024 (last-token hidden) | AG News | `fancyzhx/ag_news` | 120K/7.6K | 4 | PENDING |

**Model selection rationale (from research 2026-04-14):**
- **ModernBERT-base** supersedes DistilBERT as the credible small text encoder (Dec 2024, Answer.AI; 2T-token pretraining, 8192-token context, rotary embeddings). Same d=768 as BERT → drop-in.
- **Whisper-tiny** is still correct for small speech. No Whisper v4 as of 2026-04; latest is `whisper-large-v3-turbo` (Sep 2024) which is big. Speech Commands v2 matches Whisper's pretraining domain exactly (1s spoken English words, 16kHz).
- **Qwen3-0.6B** (April 2025, Apache 2.0) is the latest Qwen <1B. Qwen3.5-0.8B (Feb 2026) exists but is multimodal/hybrid architecture — Qwen3-0.6B is the clean pure-text causal transformer. Last-token hidden state is the standard feature for causal-LM classification.
- **AG News** used for BOTH ModernBERT and Qwen3 to enable apples-to-apples encoder-vs-decoder comparison on the same task.

**Fallback/ablation datasets (if primary run is fast enough to add):**
- ModernBERT: SST-2 (2-class, 67K, `stanfordnlp/sst2`)
- Whisper: ESC-50 (50-class, 2K, `ashraq/esc50`), UrbanSound8K (10-class, 8.7K, `danavery/urbansound8K`)
- Qwen3: IMDB (2-class, 25K/25K, `stanfordnlp/imdb`)

The paper shows SGNNET as a **universal classification head** across modalities, architectures (CNN vs transformer-encoder vs transformer-decoder), and domains (vision, text, audio, LLM).

## Status: SOFT-CONCLUSION PUSH (2026-04-14)

Priority: close paper 1 with current evidence. Exploration paused. Only paper-blocking experiments (cross-domain validation, missing baselines) proceed.

## Files

| File | Contents |
|------|----------|
| `MANUSCRIPT_DRAFT.md` | Current manuscript draft (645 lines, primary write-up) |
| `PAPER_OUTLINE.md` | Section outline and narrative structure |
| `claims.md` | Core claims with evidence status (CONFIRMED/NEEDS WORK) |
| `findings_log.md` | Chronological log of paper-worthy discoveries |
| `baselines_needed.md` | Missing comparisons (cross-domain LLM experiment is top) |
| `figures_planned.md` | Key figures/tables to include |
