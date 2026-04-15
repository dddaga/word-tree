"""Extract Qwen3-0.6B last-token features from AG News → HDF5.

HDF5 schema (matches data/store.h5 exactly):
    /{split}/features    [N, 1024]  float32   (last-token hidden state)
    /{split}/soft_labels [N, 4]     float32   (one-hot)
    /{split}/labels      [N]        int64

Last-token strategy: for causal LMs the rightmost non-padding token carries
the full sequence representation (standard for classification fine-tuning).

Usage:
    python scripts/extract_features_qwen3_agnews.py --help
    python scripts/extract_features_qwen3_agnews.py --device auto --batch 16
"""

from __future__ import annotations
import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Qwen3-0.6B last-token features from AG News and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B",
                   help="HuggingFace model ID")
    p.add_argument("--dataset_id", default="fancyzhx/ag_news",
                   help="HuggingFace dataset ID (same as ModernBERT for apples-to-apples)")
    p.add_argument("--output", default="data/store_agnews_qwen3.h5",
                   help="Output HDF5 path")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device for inference")
    p.add_argument("--batch", type=int, default=32, help="Inference batch size")
    p.add_argument("--max_length", type=int, default=128,
                   help="Token truncation length (left-truncate for causal LM)")
    p.add_argument("--num_classes", type=int, default=4,
                   help="Number of AG News classes (for one-hot soft labels)")
    return p.parse_args()


def resolve_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_split(
    split_name: str,
    hf_split: str,
    dataset,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    num_classes: int,
    h5_file,
) -> None:
    """Stream-extract features and write directly into open HDF5 file."""
    import torch

    texts = dataset[hf_split]["text"]
    labels_list = dataset[hf_split]["label"]
    N = len(texts)
    D = model.config.hidden_size  # 1024 for Qwen3-0.6B

    grp = h5_file.require_group(split_name)
    feat_ds = grp.require_dataset("features",    shape=(N, D),           dtype="float32")
    lbl_ds  = grp.require_dataset("labels",      shape=(N,),             dtype="int64")
    soft_ds = grp.require_dataset("soft_labels", shape=(N, num_classes), dtype="float32")

    model.eval()
    idx = 0
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_texts  = texts[start:end]
            batch_labels = labels_list[start:end]
            bsz = end - start

            # Left-truncate for causal LM: keep the rightmost max_length tokens
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
                padding_side="left",    # pad on left so last real token is always rightmost
                truncation_side="left",
            ).to(device)

            out = model(**enc, output_hidden_states=False)
            hidden = out.last_hidden_state  # [B, T, D]

            # Extract the last non-padding token per sequence
            attn_mask = enc["attention_mask"]  # [B, T]
            # Last non-zero position along T dimension
            last_token_idx = attn_mask.sum(dim=1) - 1  # [B]  (0-indexed)
            vecs = hidden[torch.arange(bsz, device=device), last_token_idx]  # [B, D]

            vecs_np  = vecs.cpu().float().numpy()
            lbls_np  = np.array(batch_labels, dtype=np.int64)
            soft_np  = np.eye(num_classes, dtype=np.float32)[lbls_np]

            feat_ds[idx:idx + bsz] = vecs_np
            lbl_ds[idx:idx + bsz]  = lbls_np
            soft_ds[idx:idx + bsz] = soft_np
            idx += bsz

            if (start // batch_size) % 50 == 0:
                print(f"  [{split_name}] {idx}/{N}", flush=True)

    print(f"  [{split_name}] done — {N} samples, D={D}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}  model={args.model_id}  max_len={args.max_length}")

    # Lazy imports so --help is free
    import h5py
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    print("Loading dataset …")
    dataset = load_dataset(args.dataset_id)

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Ensure left padding for causal inference (set on tokenizer level too)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model_id).to(device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as h5f:
        for split_name, hf_split in [("train", "train"), ("val", "test")]:
            extract_split(
                split_name, hf_split, dataset, tokenizer, model,
                device, args.batch, args.max_length, args.num_classes, h5f,
            )

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
