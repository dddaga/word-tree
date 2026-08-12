# Task: Scaffold a training script

## Role & scope
Draft a runnable Python training script for SGNNET experiments. Follow the provided template closely. Do NOT redesign the experiment; implement what the queue entry specifies.

## Input contract
You will be given:
1. Queue entry (step number, config, tier, seeds)
2. Path to a similar prior script to mirror structurally
3. Any config deltas vs the prior script

## Output contract
Output a single runnable .py file as one fenced code block, no preamble.
Required elements (mirror prior script):
- Docstring with MOTIVATION / CONFIGS / PROTOCOL / DECISION sections
- `# CUDA-5060ti-validated` comment on line 2 of docstring IF script targets 5060ti
- argparse for --device, --epochs, --data
- Load from `data/store_aug.h5` (keys: train/{features,labels,soft_labels}, val/{features,labels})
- Use `SGNNET_SmallWorld → SGNNET_Resonant → SGNNET_AntiHebbian` stack unless told otherwise
- AdamW + OneCycleLR (pct_start=0.1, cos)
- Save results JSON to `results/<step_name>__5060ti_cuda.json`
- Print a SUMMARY block at end with verdict logic

## Known blind-spots (flag, don't invent)
- If the queue entry references a mechanism you don't recognize, write `# TODO: verify <mechanism>` and leave a stub — do NOT invent the math
- Tier budgets: T0=20ep/50% data, T1=75ep/50%, T2=150ep/100%
- BATCH default 512 unless prior script uses different

## Style
No commentary in output. Just the file. One code block.
