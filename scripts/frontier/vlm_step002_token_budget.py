"""vlm_step002: cut a VLM's IMAGE-TOKEN budget — the lever vlm_step001 says actually matters.

vlm_step001 CONFIRMED the naive plan is Amdahl-dead: only 13.83% of SmolVLM-256M params sit in
READOUT slots (connector 2.8% + lm_head 11.1%); 74.5% sit in TRANSFORM blocks, the role
step989/llm_step002 KILLED — readout-only prune ceiling = 1.15x. Same profile handed over the
real lever: 1 image = 1088 image tokens vs 4 text tokens (17 tiles x 64, Idefics3 splitting).
VLM cost is TOKEN-bound, not param-bound. So move the SGNNET top-k idea off the WEIGHT axis
(det_step003: a dense matmul ignores the mask, so top-k is a wall-time liability) onto the TOKEN
axis, where dropping an item genuinely removes work — no sparse kernel needed.

Arms (ref = split17_full): topk_{512,256,64} keep top-k image tokens by post-connector L2 norm;
stride_64 = CONTROL (even stride, same budget as topk_64 — top-k must BEAT it or only "how many"
matters, not "which"); nosplit_64 = do_image_splitting=False, same budget but vision runs 1 tile
not 17, cutting vision cost too. Metrics: imagenette top-1, next-token agreement + KL vs ref,
vision/prefill wall-time SEPARATELY.

Output: results/frontier/vlm_step002_token_budget__{SLOT}.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))

import torch
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
OUT = ROOT / "results" / "frontier" / f"vlm_step002_token_budget__{SLOT}.json"
VAL = ROOT / "data" / "imagenette2-320" / "val"

WNID2LABEL = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette",
              "n03000684": "chainsaw", "n03028079": "church", "n03394916": "horn",
              "n03417042": "truck", "n03425413": "pump", "n03445777": "golf",
              "n03888257": "parachute"}
LABELS = list(WNID2LABEL.values())
QUESTION = ("Which one is in this image: " + ", ".join(LABELS) + "? Answer with one word.")
BUDGETS = [("topk", 512), ("topk", 256), ("topk", 64), ("stride", 64)]


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
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION}]}]
        self.prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        self.label_ids = [torch.tensor(proc.tokenizer(" " + l, add_special_tokens=False)
                                       ["input_ids"], device=DEVICE) for l in LABELS]

    def encode(self, img, split):
        """pixel -> post-connector image embeds. Returns (batch, img_feats[T,H], vision_ms)."""
        b = self.p(text=self.prompt, images=[img], return_tensors="pt",
                   **({} if split else {"do_image_splitting": False}))
        b = {k: v.to(DEVICE) for k, v in b.items()}
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            lh = self.m.model.get_image_features(
                pixel_values=b["pixel_values"],
                pixel_attention_mask=b.get("pixel_attention_mask")).last_hidden_state
            feats = self.m.model.connector(lh)          # (tiles, 64, H)
        sync()
        return b, feats.reshape(-1, feats.shape[-1]), (time.perf_counter() - t0) * 1000

    def merge(self, b, feats):
        """Splice image embeds into the text embedding sequence; return embeds + image mask."""
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
        """Length-normalised sum log P over each label's OWN tokens, teacher-forced, reusing the
        prefill cache (cropped back per label). Free generation is uninformative — SmolVLM-256M
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


def select(feats, mode, k):
    """Which image tokens survive. Order preserved (positions stay monotonic)."""
    T, dev = len(feats), feats.device
    if k >= T: return torch.arange(T, device=dev)
    if mode == "stride": return torch.linspace(0, T - 1, k, device=dev).round().long().unique()
    return feats.norm(dim=-1).topk(k).indices.sort().values


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    n = 3 if args.smoke_test else args.n_images
    print(f"{'='*72}\nvlm_step002 token budget  device={DEVICE}  n_images={n}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    r = Runner(m, AutoProcessor.from_pretrained(args.model))

    arms = ["split17_full"] + [f"{md}_{k}" for md, k in BUDGETS] + ["nosplit_64"]
    acc = {a: dict.fromkeys(("correct", "same_class", "agree", "kl", "vis_ms", "pre_ms",
                             "tokens", "seq"), 0) for a in arms}

    for i, (path, wnid) in enumerate(sample_images(n)):
        gold = LABELS.index(WNID2LABEL[wnid])
        img = Image.open(path).convert("RGB")
        b, feats, vis_ms = r.encode(img, split=True)
        emb, pos = r.merge(b, feats)
        ref_logits, ref_pred, pre_ms = r.forward(emb)
        ref_lp = F.log_softmax(ref_logits, -1)
        ref_top = ref_logits.argmax().item()

        rows = [("split17_full", len(feats), vis_ms, pre_ms, ref_logits, ref_pred, emb.shape[1])]
        for md, k in BUDGETS:
            keep = select(feats, md, k)
            drop = torch.ones(emb.shape[1], dtype=torch.bool, device=DEVICE)
            drop[pos] = False
            drop[pos[keep]] = True
            e2 = emb[:, drop]
            lg, pr, pm = r.forward(e2)
            rows.append((f"{md}_{k}", len(keep), vis_ms, pm, lg, pr, e2.shape[1]))

        b2, f2, vis2 = r.encode(img, split=False)
        e3, _ = r.merge(b2, f2)
        lg3, pr3, pm3 = r.forward(e3)
        rows.append(("nosplit_64", len(f2), vis2, pm3, lg3, pr3, e3.shape[1]))

        for name, tok, vm, pm, lg, pred, seq in rows:
            a = acc[name]
            a["correct"] += int(pred == gold); a["same_class"] += int(pred == ref_pred)
            a["agree"] += int(lg.argmax().item() == ref_top)
            a["kl"] += float(F.kl_div(F.log_softmax(lg, -1), ref_lp,
                                      log_target=True, reduction="sum"))
            a["vis_ms"] += vm; a["pre_ms"] += pm
            a["tokens"] = tok; a["seq"] = seq
        if (i + 1) % 10 == 0 or args.smoke_test:
            print(f"  [{i+1}/{n}] {path.name} gold={LABELS[gold]} ref={LABELS[ref_pred]}", flush=True)

    print(f"\n  {'arm':<14} {'imgtok':>7} {'seq':>6} {'top1':>6} {'same':>6} {'agree':>6} "
          f"{'KL':>7} {'vis_ms':>8} {'pre_ms':>8} {'total':>8} {'speed':>7}")
    ref_tot = (acc['split17_full']['vis_ms'] + acc['split17_full']['pre_ms']) / n
    res = {"step": "vlm_step002", "model": args.model, "device": str(DEVICE),
           "n_images": n, "question": QUESTION, "arms": {}}
    for a in arms:
        d = acc[a]; v, p_ = d["vis_ms"] / n, d["pre_ms"] / n
        row = {"image_tokens": d["tokens"], "seq_len": d["seq"], "top1": d["correct"] / n,
               "same_class_as_full": d["same_class"] / n, "agree_with_full": d["agree"] / n,
               "kl_vs_full": d["kl"] / n, "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
               "total_ms": round(v + p_, 2), "speedup": round(ref_tot / (v + p_), 3)}
        res["arms"][a] = row
        print(f"  {a:<14} {d['tokens']:>7} {d['seq']:>6} {row['top1']:>6.3f} "
              f"{row['same_class_as_full']:>6.3f} {row['agree_with_full']:>6.3f} "
              f"{row['kl_vs_full']:>7.3f} {v:>8.1f} {p_:>8.1f} {v+p_:>8.1f} "
              f"{row['speedup']:>6.2f}x")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
