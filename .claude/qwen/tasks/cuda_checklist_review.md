# Task: CUDA-5060ti checklist review

## Role & scope
Review a training script against the 5060ti CUDA checklist. Flag violations. Do not rewrite the script.

## Input contract
1. Full script source
2. The CUDA checklist rules (provided inline below)

## Checklist
1. Data must load from `data/store_aug.h5` — NOT torchvision `ImageFolder` (torchvision broken on 5060ti host)
2. No `num_workers>0` with CUDA tensors pre-loaded to GPU (forked workers can't access CUDA mem)
3. No `GradScaler` — fp16+GradScaler is 4.4× slower on Blackwell (step801)
4. Model `.to(DEVICE)` must be called AFTER full wrapper stack construction
5. Seed set via `torch.manual_seed(SEED)` BEFORE model construction
6. `# CUDA-5060ti-validated` marker on line 2 of docstring
7. Results path ends in `__5060ti_cuda.json`

## Output contract
Markdown table: | Rule # | Status (PASS/FAIL/N/A) | Evidence (line number or quote) |
Then 1-line overall verdict: SAFE TO LAUNCH / FIX REQUIRED.
No preamble.

## Known blind-spots
- Don't check algorithm correctness — only infra
- Don't check whether the experiment is worth running
