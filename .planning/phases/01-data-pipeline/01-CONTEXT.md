# Phase 1: Data Pipeline - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Download Imagenette 320px dataset, run all images through pretrained VGG16 CNN backbone to extract 25088-dim pre-FC activations and 10-dim soft probability vectors, then persist everything to an HDF5 tensor store with a CSV manifest. No training occurs in this phase — this is pure data extraction and storage.

</domain>

<decisions>
## Implementation Decisions

### Python Environment
- **D-01:** Keep the existing `d_env/` virtual environment (Python 3.14.3). Install all required packages (torch, torchvision, h5py, scikit-learn, pandas, matplotlib, tqdm) into it. If any package fails on 3.14, try the latest pre-release wheel before escalating.

### Soft Label Storage
- **D-02:** Store raw T=1 softmax outputs from VGG16. Do NOT apply temperature scaling at extraction time. Temperature (T=4.0) is applied on-the-fly during training when computing the KL divergence loss. This preserves full flexibility to sweep T values without re-running extraction.
- **D-03:** Soft labels are computed as: forward 10 Imagenette class indices through `model.classifier` → select those 10 logits from the 1000-class output → softmax (T=1). Store as float32.

### HDF5 Configuration
- **D-04:** No compression. Use sample-level chunking: `chunks=(1, 25088)` for features, `chunks=(1, 10)` for soft labels. This maximizes random-access read speed during training (one disk I/O per sample, no decompression overhead). Expected file size: ~1.34GB.
- **D-05:** HDF5 schema is locked (from ROADMAP.md):
  ```
  store.h5
    /train/features    [9469, 25088]  float32
    /train/soft_labels [9469, 10]     float32
    /train/labels      [9469]         int64
    /val/features      [3925, 25088]  float32
    /val/soft_labels   [3925, 10]     float32
    /val/labels        [3925]         int64
  ```

### Data Location
- **D-06:** All data lives inside the project under `data/`. Imagenette images at `data/imagenette2-320/`, tensor store at `data/store.h5`, manifest at `data/manifest.csv`. A `.gitignore` entry for `data/` keeps images and tensors out of git.

### Claude's Discretion
- Extraction batch size (256 recommended for MPS — adjust down if memory issues)
- Imagenette download method (torchvision.datasets.Imagenette is the standard path)
- Whether to add a resumability check (skip re-extraction if store.h5 already exists and has expected shape)
- HDF5 write mode (create store in two passes: first train, then val; or pre-allocate and fill)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture & Experiment Design
- `sparse_geometric_network_report.md` — Full SGNNET architecture spec; background on why distillation is used and what the tensor store feeds into
- `.planning/ROADMAP.md` §Phase 1 — Locked plan structure: module names, class signatures, HDF5 schema, CSV schema, verification criteria for each sub-plan
- `.planning/REQUIREMENTS.md` §Data Pipeline — DATA-01 through DATA-06 acceptance criteria

### Project Constraints
- `.planning/PROJECT.md` §Constraints — File size limit (250 lines/file), top-down code style, MPS hardware, 3-day timeline

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `d_env/` — Python 3.14 virtual environment. No packages installed yet. All dependencies must be installed here.

### Established Patterns
- No existing code patterns — this is the first phase. Patterns established here will carry forward to all subsequent phases.

### Integration Points
- `data/store.h5` is consumed by Phase 2 (baseline eval), Phase 3 (SGNNET input), Phase 4 (training), and Phase 5 (PCA sweep). Schema must match exactly what downstream phases expect.
- `data/manifest.csv` is used for per-class analysis in Phase 6.

</code_context>

<specifics>
## Specific Ideas

- Imagenette class-to-ImageNet-index mapping is fixed (from ROADMAP.md): tench=0, english_springer=217, cassette_player=482, chain_saw=491, church=497, french_horn=566, garbage_truck=569, gas_pump=571, golf_ball=574, parachute=701
- VGG16 hook target: `model.features[-1]` (final AdaptiveAvgPool2d or MaxPool2d), output shape [batch, 512, 7, 7] → flatten → [batch, 25088]
- MPS device priority: use `torch.device("mps")` if `torch.backends.mps.is_available()`, else CPU

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-data-pipeline*
*Context gathered: 2026-03-23*
