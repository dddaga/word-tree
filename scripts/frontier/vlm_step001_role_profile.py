"""vlm_step001: where do a small VLM's params live, and how many sit in SGNNET-ADDRESSABLE slots?

FINAL /goal item: prune a visual LLM so it runs parameter-efficiently on a drone. Before
spending compute we answer the Amdahl question the detection branch answered first
(det_step001 -> box head = 72% -> worth attacking). Same method, VLM target.

ROLE CLASSIFICATION is the whole point — our evidence says the SGNNET recipe transfers by
ROLE, not by shape:
  READOUT  (CONFIRMED transfers): pooled/hidden feature vector -> output space.
           = the vision->text connector/projector, and the lm_head (hidden -> vocab).
           VGG-FC (step605), FasterRCNN box head (det_step002) both live here.
  TRANSFORM (CONFIRMED does NOT transfer): attention + MLP/FFN inside encoder/decoder
           blocks. step989 + llm_step002 killed the FFN-transform role explicitly.
  EMBEDDING/NORM: neither; embeddings are lookup tables (quantize, don't distil).

So the readout share is a HARD CEILING on what a readout-only prune can win. We report it,
then project whole-model savings using the compression ratios we MEASURED (det_step003:
dense low-rank 23.6% of teacher params; top-k 11.7%) — not assumed ones. Weight-tied
tensors are de-duplicated by data_ptr so a tied lm_head is not double-counted.

Also probes the runtime lever that is unique to VLMs: how many VISION TOKENS one image
costs the decoder (sequence length is quadratic-ish in attention and is often the real
drone bottleneck, not params).

Output: results/frontier/vlm_step001_role_profile__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="cpu")  # profile is param-counting; CPU is enough
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"vlm_step001_role_profile__{SLOT}.json"

# Measured compression ratios (params kept) from the detection branch — NOT assumptions.
RATIOS = {"dense_lowrank_r256": 0.236, "sgn_topk_r128k32": 0.117, "sgn_topk_r64k16": 0.058}
READOUT = ("READOUT_PROJECTOR", "READOUT_LMHEAD")


def classify(name: str) -> str:
    """Role of a parameter, by name. Category rule — no per-model hardcoding."""
    n = name.lower()
    if any(t in n for t in ("connector", "modality_projection", "multi_modal_projector",
                            "visual_projection", "text_projection", "mm_projector")):
        return "READOUT_PROJECTOR"
    if "lm_head" in n or "output_projection" in n:
        return "READOUT_LMHEAD"
    if any(t in n for t in ("embed", "wte", "wpe")):
        return "EMBEDDING"
    if any(t in n for t in ("norm", "ln_", "ln1", "ln2")):
        return "NORM"
    vision = any(t in n for t in ("vision", "visual", "image_encoder", "vit"))
    if any(t in n for t in ("attn", "attention", "mlp", "fc1", "fc2", "proj", "dense")):
        return "TRANSFORM_VISION" if vision else "TRANSFORM_TEXT"
    return "OTHER"


def profile(model):
    """Bucket params by role, de-duplicating weight-tied tensors via data_ptr."""
    seen, buckets, tied = set(), {}, []
    for name, p in model.named_parameters():
        ptr = p.data_ptr()
        if ptr in seen:
            tied.append(name)
            continue
        seen.add(ptr)
        role = classify(name)
        b = buckets.setdefault(role, {"params": 0, "tensors": 0, "examples": []})
        b["params"] += p.numel(); b["tensors"] += 1
        if len(b["examples"]) < 3:
            b["examples"].append(f"{name}{tuple(p.shape)}")
    return buckets, tied


def token_probe(model_id):
    """How many tokens does ONE image cost the decoder? Best-effort; None on failure."""
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
        img = Image.new("RGB", (512, 512), (127, 127, 127))
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "What is here?"}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        with_img = proc(text=prompt, images=[img], return_tensors="pt")["input_ids"]
        txt_only = proc(text="What is here?", return_tensors="pt")["input_ids"]
        return {"seq_len_with_image": int(with_img.shape[1]),
                "seq_len_text_only": int(txt_only.shape[1]),
                "image_token_cost": int(with_img.shape[1] - txt_only.shape[1])}
    except Exception as e:
        print(f"  [WARN] token probe failed: {type(e).__name__}: {str(e)[:90]}")
        return None


def amdahl(buckets, total):
    """Whole-model saving if ONLY readout slots are compressed, at MEASURED ratios."""
    ro = sum(buckets.get(r, {}).get("params", 0) for r in READOUT)
    out = {"readout_params": ro, "readout_share": round(ro / total, 4), "projections": {}}
    for tag, keep in RATIOS.items():
        new_total = total - ro + ro * keep
        out["projections"][tag] = {
            "model_params_after": int(new_total),
            "whole_model_reduction_pct": round(100 * (1 - new_total / total), 3),
            "compression_x": round(total / new_total, 3)}
    return out


def main():
    if args.smoke_test:
        for n in ["model.connector.modality_projection.proj.weight", "lm_head.weight",
                  "model.text_model.layers.0.mlp.gate_proj.weight",
                  "model.vision_model.encoder.layers.0.mlp.fc1.weight",
                  "model.text_model.embed_tokens.weight", "model.text_model.norm.weight"]:
            print(f"  {classify(n):<18} {n}")
        sys.exit(0)

    import transformers
    # transformers>=5 renamed Vision2Seq -> ImageTextToText; support both.
    auto = next((getattr(transformers, c) for c in
                 ("AutoModelForImageTextToText", "AutoModelForVision2Seq")
                 if hasattr(transformers, c)), None)
    if auto is None:
        sys.exit("no image-text-to-text AutoModel class in this transformers build")
    print(f"{'='*70}\nvlm_step001 role profile — {args.model}  via {auto.__name__}")
    model = auto.from_pretrained(args.model, dtype=torch.float32).eval()
    buckets, tied = profile(model)
    total = sum(b["params"] for b in buckets.values())

    # top-level submodule split (vision tower vs decoder vs rest), tied-deduped
    top, seen = {}, set()
    for name, p in model.named_parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        key = ".".join(name.split(".")[:2]) if name.count(".") >= 1 else name
        top[key] = top.get(key, 0) + p.numel()

    tok = token_probe(args.model)
    am = amdahl(buckets, total)

    print(f"\n  total (tied-deduped) = {total:,} params   tied tensors skipped: {len(tied)}")
    print(f"\n  {'role':<20} {'params':>12} {'%model':>8}  example")
    for role, b in sorted(buckets.items(), key=lambda kv: -kv[1]["params"]):
        print(f"  {role:<20} {b['params']:>12,} {100*b['params']/total:>7.2f}%  {b['examples'][0][:44]}")
    print(f"\n  {'submodule':<32} {'params':>12} {'%model':>8}")
    for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:<32} {v:>12,} {100*v/total:>7.2f}%")

    print(f"\n  SGNNET-ADDRESSABLE (readout) share = {100*am['readout_share']:.2f}% "
          f"({am['readout_params']:,} params) <- Amdahl ceiling for readout-only prune")
    for tag, pr in am["projections"].items():
        print(f"    {tag:<20} -> {pr['model_params_after']:>12,} params  "
              f"({pr['whole_model_reduction_pct']:>5.2f}% smaller, {pr['compression_x']:.3f}x)")
    if tok:
        print(f"\n  vision-token cost: 1 image = {tok['image_token_cost']} tokens "
              f"(seq {tok['seq_len_text_only']} -> {tok['seq_len_with_image']})")

    res = {"step": "vlm_step001", "model": args.model, "total_params": total,
           "tied_skipped": len(tied), "roles": buckets, "submodules": top,
           "amdahl": am, "token_probe": tok, "measured_ratios_source": "det_step002/003"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
