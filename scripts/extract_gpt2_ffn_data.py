"""Extract GPT-2-small layer-6 FFN (x_ffn, y_ffn) pairs for step989.
Requires: transformers, datasets (both installed in d_env).
Output: data/gpt2_ffn_layer6.pt  — dict with keys 'x': [T,768], 'y': [T,768]
Usage: d_env/bin/python3 scripts/extract_gpt2_ffn_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

try:
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: d_env/bin/pip install transformers")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed. Run: d_env/bin/pip install datasets")
    sys.exit(1)

MAX_LENGTH  = 128
TARGET_TOKS = 50_000
BATCH_SIZE  = 4        # conservative to avoid OOM
OUT_PATH    = ROOT / "data" / "gpt2_ffn_layer6.pt"

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def main() -> None:
    print(f"Device: {DEVICE}")
    print("Loading GPT-2-small model + tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

    # ── Hook: capture MLP input (x_ffn) and output (y_ffn) at layer 6 ──
    x_buf: list[torch.Tensor] = []
    y_buf: list[torch.Tensor] = []

    def _hook(module, inp, out):
        # inp[0]: [B, seq, 768] — MLP input (after attn residual + LN2)
        # out:    [B, seq, 768] — MLP output
        x_buf.append(inp[0].detach().cpu().reshape(-1, 768))
        y_buf.append(out.detach().cpu().reshape(-1, 768))

    handle = model.h[6].mlp.register_forward_hook(_hook)

    # ── Load WikiText-2-raw-v1 validation split ──
    print("Loading WikiText-2-raw-v1 validation split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [row["text"] for row in ds if row["text"].strip()]
    print(f"  {len(texts)} non-empty lines loaded")

    # ── Tokenize into fixed-length windows ──
    print(f"Tokenizing into {MAX_LENGTH}-token sequences...")
    all_ids: list[int] = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)

    # Split into non-overlapping windows
    windows = [
        all_ids[i : i + MAX_LENGTH]
        for i in range(0, len(all_ids) - MAX_LENGTH + 1, MAX_LENGTH)
    ]
    print(f"  {len(windows)} windows × {MAX_LENGTH} tokens = {len(windows)*MAX_LENGTH:,} tokens")

    # ── Forward pass in mini-batches ──
    n_collected = 0
    print(f"Running forward pass (batch={BATCH_SIZE}), target≥{TARGET_TOKS:,} tokens...")
    with torch.no_grad():
        for start in range(0, len(windows), BATCH_SIZE):
            batch_ids = windows[start : start + BATCH_SIZE]
            ids_tensor = torch.tensor(batch_ids, dtype=torch.long).to(DEVICE)
            attn_mask  = torch.ones_like(ids_tensor)
            model(input_ids=ids_tensor, attention_mask=attn_mask)
            n_collected += len(batch_ids) * MAX_LENGTH
            if n_collected >= TARGET_TOKS:
                print(f"  Reached {n_collected:,} tokens — stopping early")
                break
            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  batch {start//BATCH_SIZE}: {n_collected:,} tokens so far")

    handle.remove()

    x_all = torch.cat(x_buf, dim=0)   # [T, 768]
    y_all = torch.cat(y_buf, dim=0)   # [T, 768]
    print(f"\nCollected x: {tuple(x_all.shape)}  y: {tuple(y_all.shape)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x_all, "y": y_all}, OUT_PATH)
    print(f"Saved → {OUT_PATH}")
    print("  Keys: 'x' [T,768], 'y' [T,768]")
    print("  x dtype:", x_all.dtype, "  y dtype:", y_all.dtype)


if __name__ == "__main__":
    main()
