"""Extract ModernBERT-base features from AG News → HDF5.

HDF5 schema (matches data/store.h5 exactly):
    /{split}/features    [N, 768]  float32
    /{split}/soft_labels [N, 4]    float32   (one-hot)
    /{split}/labels      [N]       int64

Usage:
    python scripts/extract_features_modernbert_agnews.py --help
    python scripts/extract_features_modernbert_agnews.py --device auto --batch 32
"""

from __future__ import annotations
import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract ModernBERT-base features from AG News and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="answerdotai/ModernBERT-base",
                   help="HuggingFace model ID")
    p.add_argument("--dataset_id", default="fancyzhx/ag_news",
                   help="HuggingFace dataset ID")
    p.add_argument("--output", default="data/store_agnews_modernbert.h5",
                   help="Output HDF5 path")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device for inference")
    p.add_argument("--batch", type=int, default=32, help="Inference batch size")
    p.add_argument("--max_length", type=int, default=128,
                   help="Token truncation length")
    p.add_argument("--pooling", default="mean", choices=["cls", "mean"],
                   help="Pooling strategy over last hidden states")
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
    pooling: str,
    num_classes: int,
    h5_file,
) -> None:
    """Stream-extract features and write directly into open HDF5 file."""
    import torch

    texts = dataset[hf_split]["text"]
    labels_list = dataset[hf_split]["label"]
    N = len(texts)
    D = model.config.hidden_size

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

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)

            out = model(**enc, output_hidden_states=False)
            hidden = out.last_hidden_state  # [B, T, D]

            if pooling == "cls":
                vecs = hidden[:, 0, :]
            else:
                # Mean over non-padding tokens
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

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
    print(f"device={device}  model={args.model_id}  pooling={args.pooling}")

    # Lazy imports so --help is free
    import h5py
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    print("Loading dataset …")
    dataset = load_dataset(args.dataset_id)

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as h5f:
        for split_name, hf_split in [("train", "train"), ("val", "test")]:
            extract_split(
                split_name, hf_split, dataset, tokenizer, model,
                device, args.batch, args.max_length, args.pooling,
                args.num_classes, h5f,
            )

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
