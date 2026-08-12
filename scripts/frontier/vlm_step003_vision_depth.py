"""vlm_step003: cut VISION-ENCODER DEPTH — the lever vlm_step002 exposed.

vlm_step002 CONFIRMED two Amdahl facts. (a) Token pruning caps at ~1.13x: prefill is only 13% of
per-image latency (147/1125 ms). (b) nosplit_64 wins 11.71x by cutting TILE COUNT, i.e. by
attacking the vision encoder, which is 87% of latency AND 33.15% of params. vlm_step001 already
killed the readout-prune plan at a 1.15x ceiling. So the target is the vision tower.

SmolVLM-256M's tower = 12 SigLIP layers x 7,087,872 params. Dropping 6 removes 42.5M = 16.6% of
the whole model — MORE than the entire readout share (13.83%) — and 6/12 of the 87% latency
path. Truncation is legitimate where SGNNET substitution is not: we are not asking a sparse head
to imitate a TRANSFORM block (step989/llm_step002 KILLED that), only whether the block is needed
at all. The connector consumes hidden_size=768, which every intermediate layer already emits.

Arms (ref = d12_full, nosplit, 64 tok): d{9,6,3} truncate the tower; d12_topk{32,16} re-run the
step002 token axis on the nosplit champion; d6_topk32 COMPOSES both (Compounding Rule: depth and
sequence length are different signal paths, so composition is admissible — this arm tests it
rather than assuming). Metrics: top-1 via constrained 10-way label likelihood, agreement + KL vs
d12_full, vision/prefill wall-time SEPARATELY, params per arm.

Output: results/frontier/vlm_step003_vision_depth__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--n_images", type=int, default=50)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"vlm_step003_vision_depth__{SLOT}.json"
VAL = ROOT / "data" / "imagenette2-320" / "val"

WNID2LABEL = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette",
              "n03000684": "chainsaw", "n03028079": "church", "n03394916": "horn",
              "n03417042": "truck", "n03425413": "pump", "n03445777": "golf",
              "n03888257": "parachute"}
LABELS = list(WNID2LABEL.values())
QUESTION = ("Which one is in this image: " + ", ".join(LABELS) + "? Answer with one word.")
# (name, vision_depth, token_budget)
ARMS = [("d12_full", 12, 64), ("d9", 9, 64), ("d6", 6, 64), ("d3", 3, 64),
        ("d12_topk32", 12, 32), ("d12_topk16", 12, 16), ("d6_topk32", 6, 32)]


def sync():
    if DEVICE.type == "mps": torch.mps.synchronize()
    elif DEVICE.type == "cuda": torch.cuda.synchronize()


def sample_images(n):
    rng, per, out = random.Random(42), max(1, n // len(WNID2LABEL)), []
    for wnid in sorted(WNID2LABEL):
        files = sorted((VAL / wnid).glob("*.JPEG"))
        out += [(f, wnid) for f in rng.sample(files, min(per, len(files)))]
    return out[:n]


class Runner:
    def __init__(self, model, proc):
        self.m, self.p = model, proc
        self.img_id = model.config.image_token_id
        self.layers = list(model.model.vision_model.encoder.layers)   # keep full stack alive
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION}]}]
        self.prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        self.label_ids = [torch.tensor(proc.tokenizer(" " + l, add_special_tokens=False)
                                       ["input_ids"], device=DEVICE) for l in LABELS]

    def set_depth(self, d):
        """Swap in the first d SigLIP layers. post_layernorm + connector are untouched — layer d
        emits the same hidden_size the connector already consumes."""
        self.m.model.vision_model.encoder.layers = nn.ModuleList(self.layers[:d])

    def batch(self, img):
        b = self.p(text=self.prompt, images=[img], return_tensors="pt", do_image_splitting=False)
        return {k: v.to(DEVICE) for k, v in b.items()}

    def encode(self, b, depth):
        """pixel -> post-connector image embeds at truncated depth. Returns (feats[T,H], ms)."""
        self.set_depth(depth)
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            lh = self.m.model.get_image_features(
                pixel_values=b["pixel_values"],
                pixel_attention_mask=b.get("pixel_attention_mask")).last_hidden_state
            feats = self.m.model.connector(lh)
        sync()
        return feats.reshape(-1, feats.shape[-1]), (time.perf_counter() - t0) * 1000

    def merge(self, b, feats):
        ids = b["input_ids"]
        emb = self.m.get_input_embeddings()(ids).clone()
        pos = (ids[0] == self.img_id).nonzero(as_tuple=True)[0]
        assert len(pos) == len(feats), f"{len(pos)} slots vs {len(feats)} feats"
        emb[0, pos] = feats.to(emb.dtype)
        return emb, pos

    def forward(self, emb):
        """Prefill (TIMED) + constrained 10-way choice (untimed). Returns (logits[V], pred, ms)."""
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=DEVICE)
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            o = self.m.model.text_model(inputs_embeds=emb, attention_mask=mask, use_cache=True)
            logits = self.m.lm_head(o.last_hidden_state[:, -1])[0].float()
        sync(); ms = (time.perf_counter() - t0) * 1000
        return logits, self.choose(logits, o.past_key_values, emb.shape[1]), ms

    def choose(self, logits, past, plen):
        """Length-normalised log P over each label's OWN tokens, teacher-forced, reusing the
        prefill cache (cropped per label). Free generation is uninformative — SmolVLM-256M
        answers 'Frog' for a tench — but its ranking over the 10 named options is."""
        lp0, out = logits.log_softmax(-1), []
        for ids in self.label_ids:
            with torch.no_grad():
                o = self.m.model.text_model(
                    inputs_embeds=self.m.get_input_embeddings()(ids[None]), past_key_values=past,
                    attention_mask=torch.ones(1, plen + len(ids), dtype=torch.long, device=DEVICE),
                    use_cache=True)
                lg = self.m.lm_head(o.last_hidden_state)[0].float().log_softmax(-1)
            s = lp0[ids[0]] + sum(lg[i, ids[i + 1]] for i in range(len(ids) - 1))
            out.append(float(s) / len(ids))
            past.crop(plen)
        return int(torch.tensor(out).argmax())


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    n = 3 if args.smoke_test else args.n_images
    print(f"{'='*78}\nvlm_step003 vision depth  device={DEVICE}  n_images={n}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    r = Runner(m, AutoProcessor.from_pretrained(args.model))

    total_p = sum(p.numel() for p in m.parameters())
    per_layer = sum(p.numel() for p in r.layers[0].parameters())
    print(f"  total {total_p:,} params | vision layer {per_layer:,} x {len(r.layers)}")
    acc = {a: dict.fromkeys(("correct", "same", "agree", "kl", "vis", "pre", "tok"), 0)
           for a, _, _ in ARMS}

    for i, (path, wnid) in enumerate(sample_images(n)):
        gold = LABELS.index(WNID2LABEL[wnid])
        b = r.batch(Image.open(path).convert("RGB"))
        enc = {d: r.encode(b, d) for d in sorted({a[1] for a in ARMS}, reverse=True)}
        ref_lg = ref_pred = ref_lp = ref_top = None
        for name, depth, k in ARMS:
            feats, vis_ms = enc[depth]
            emb, pos = r.merge(b, feats)
            if k < len(feats):
                keep = feats.norm(dim=-1).topk(k).indices.sort().values
                drop = torch.ones(emb.shape[1], dtype=torch.bool, device=DEVICE)
                drop[pos] = False; drop[pos[keep]] = True
                emb = emb[:, drop]
            lg, pred, pre_ms = r.forward(emb)
            if ref_lg is None:
                ref_lg, ref_pred = lg, pred
                ref_lp, ref_top = F.log_softmax(lg, -1), lg.argmax().item()
            a = acc[name]
            a["correct"] += int(pred == gold); a["same"] += int(pred == ref_pred)
            a["agree"] += int(lg.argmax().item() == ref_top)
            a["kl"] += float(F.kl_div(F.log_softmax(lg, -1), ref_lp,
                                      log_target=True, reduction="sum"))
            a["vis"] += vis_ms; a["pre"] += pre_ms; a["tok"] = min(k, len(feats))
        if (i + 1) % 10 == 0 or args.smoke_test:
            print(f"  [{i+1}/{n}] {path.name} gold={LABELS[gold]} ref={LABELS[ref_pred]}", flush=True)

    print(f"\n  {'arm':<12} {'tok':>4} {'params':>12} {'%save':>6} {'top1':>6} {'same':>6} "
          f"{'agree':>6} {'KL':>7} {'vis_ms':>8} {'pre_ms':>7} {'total':>8} {'speed':>7}")
    ref_tot = (acc["d12_full"]["vis"] + acc["d12_full"]["pre"]) / n
    res = {"step": "vlm_step003", "model": args.model, "device": str(DEVICE), "n_images": n,
           "total_params": total_p, "vision_layer_params": per_layer, "arms": {}}
    for name, depth, k in ARMS:
        d = acc[name]; v, p_ = d["vis"] / n, d["pre"] / n
        prm = total_p - (len(r.layers) - depth) * per_layer
        row = {"vision_depth": depth, "image_tokens": d["tok"], "params": prm,
               "params_saved_pct": round(100 * (total_p - prm) / total_p, 2),
               "top1": d["correct"] / n, "same_class_as_ref": d["same"] / n,
               "agree_with_ref": d["agree"] / n, "kl_vs_ref": d["kl"] / n,
               "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
               "total_ms": round(v + p_, 2), "speedup": round(ref_tot / (v + p_), 3)}
        res["arms"][name] = row
        print(f"  {name:<12} {d['tok']:>4} {prm:>12,} {row['params_saved_pct']:>6.2f} "
              f"{row['top1']:>6.3f} {row['same_class_as_ref']:>6.3f} {row['agree_with_ref']:>6.3f} "
              f"{row['kl_vs_ref']:>7.3f} {v:>8.1f} {p_:>7.1f} {v+p_:>8.1f} {row['speedup']:>6.2f}x")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
