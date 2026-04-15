"""Extract Whisper-tiny encoder features from Speech Commands v2 → HDF5.

HDF5 schema (matches data/store.h5 exactly):
    /{split}/features    [N, 384]  float32   (encoder last-hidden mean-pooled)
    /{split}/soft_labels [N, 35]   float32   (one-hot)
    /{split}/labels      [N]       int64

Usage:
    python scripts/extract_features_whisper_speech_commands.py --help
    python scripts/extract_features_whisper_speech_commands.py --device auto --batch 32
"""

from __future__ import annotations
import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Whisper-tiny encoder features from Speech Commands v2 and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_id", default="openai/whisper-tiny",
                   help="HuggingFace model ID (encoder only used)")
    p.add_argument("--dataset_id", default="google/speech_commands",
                   help="HuggingFace dataset ID")
    p.add_argument("--dataset_version", default="v0.02",
                   help="Dataset config version (v0.01 or v0.02)")
    p.add_argument("--output", default="data/store_speech_commands_whisper.h5",
                   help="Output HDF5 path")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device for inference")
    p.add_argument("--batch", type=int, default=32, help="Inference batch size")
    p.add_argument("--sample_rate", type=int, default=16000,
                   help="Target audio sample rate (Whisper expects 16kHz)")
    p.add_argument("--max_duration", type=float, default=1.0,
                   help="Max clip duration in seconds (clips are truncated/padded)")
    p.add_argument("--num_classes", type=int, default=35,
                   help="Number of Speech Commands classes")
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


def pad_or_truncate(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (zero) or truncate 1-D audio to exactly target_len samples."""
    if len(audio) >= target_len:
        return audio[:target_len].astype(np.float32)
    return np.pad(audio, (0, target_len - len(audio))).astype(np.float32)


def extract_split(
    split_name: str,
    hf_split: str,
    dataset,
    processor,
    encoder,
    device: str,
    batch_size: int,
    sample_rate: int,
    max_duration: float,
    num_classes: int,
    label2idx: dict,
    h5_file,
) -> None:
    """Stream-extract features and write directly into open HDF5 file."""
    import torch

    split_data = dataset[hf_split]
    N = len(split_data)
    target_len = int(sample_rate * max_duration)
    D = encoder.config.d_model  # 384 for whisper-tiny

    grp = h5_file.require_group(split_name)
    feat_ds = grp.require_dataset("features",    shape=(N, D),           dtype="float32")
    lbl_ds  = grp.require_dataset("labels",      shape=(N,),             dtype="int64")
    soft_ds = grp.require_dataset("soft_labels", shape=(N, num_classes), dtype="float32")

    encoder.eval()
    idx = 0
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = split_data.select(range(start, end))
            bsz = end - start

            # Resample and pad/truncate audio waveforms
            raw_audios = []
            for audio_item in batch["audio"]:
                wave = audio_item["array"].astype(np.float32)
                sr   = audio_item["sampling_rate"]
                if sr != sample_rate:
                    # Resample using librosa if needed (datasets usually provides 16kHz)
                    import librosa
                    wave = librosa.resample(wave, orig_sr=sr, target_sr=sample_rate)
                raw_audios.append(pad_or_truncate(wave, target_len))

            # Whisper feature extractor: log-mel spectrogram
            inputs = processor(
                raw_audios,
                sampling_rate=sample_rate,
                return_tensors="pt",
            ).to(device)

            out = encoder(inputs.input_features)
            hidden = out.last_hidden_state  # [B, T_enc, D]

            # Mean-pool over encoder time dimension (no padding mask needed for fixed-len)
            vecs = hidden.mean(dim=1)  # [B, D]

            # Map string labels → integer indices
            batch_labels_str = batch["label"]
            batch_lbls = np.array(
                [label2idx[l] if isinstance(l, str) else int(l) for l in batch_labels_str],
                dtype=np.int64,
            )

            vecs_np = vecs.cpu().float().numpy()
            soft_np = np.eye(num_classes, dtype=np.float32)[batch_lbls]

            feat_ds[idx:idx + bsz] = vecs_np
            lbl_ds[idx:idx + bsz]  = batch_lbls
            soft_ds[idx:idx + bsz] = soft_np
            idx += bsz

            if (start // batch_size) % 50 == 0:
                print(f"  [{split_name}] {idx}/{N}", flush=True)

    print(f"  [{split_name}] done — {N} samples, D={D}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}  model={args.model_id}  sr={args.sample_rate}  max_dur={args.max_duration}s")

    # Lazy imports so --help is free
    import h5py
    from datasets import load_dataset
    from transformers import WhisperProcessor, WhisperModel

    print("Loading dataset …")
    dataset = load_dataset(args.dataset_id, args.dataset_version)

    # Build label→int mapping (Speech Commands uses string labels)
    label_names = dataset["train"].features["label"].names
    label2idx = {name: i for i, name in enumerate(label_names)}
    num_classes = len(label_names)
    if num_classes != args.num_classes:
        print(f"  WARNING: dataset has {num_classes} classes, --num_classes={args.num_classes}; using {num_classes}")
        args.num_classes = num_classes

    print("Loading model …")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    # Encoder-only: load full WhisperModel, access .encoder
    full_model = WhisperModel.from_pretrained(args.model_id)
    encoder = full_model.encoder.to(device)
    del full_model  # free decoder weights

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Speech Commands splits: train / validation / test
    # We map validation → val for consistency; test exists but is small
    split_map = [("train", "train"), ("val", "validation")]

    with h5py.File(args.output, "w") as h5f:
        for split_name, hf_split in split_map:
            extract_split(
                split_name, hf_split, dataset, processor, encoder,
                device, args.batch, args.sample_rate, args.max_duration,
                args.num_classes, label2idx, h5f,
            )

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
