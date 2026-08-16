"""Shared eval harness for the vlm_step00X frontier line.

LIBRARY ONLY — no argparse, no I/O, no side effects at import, so step scripts can import it.
Extracted at vlm_step004 because the constrained-choice protocol is now used by more than one
step and the repo's 200-line file limit forbids a third copy. vlm_step002/003 keep their inline
copies deliberately: they are DONE experiments and their scripts must stay reproducible as run.

The protocol: SmolVLM-256M cannot answer a 10-way naming prompt by free generation (it replies
"Frog" for a tench), so top-1 is measured as a CONSTRAINED choice — length-normalised
teacher-forced log P over each candidate label's own tokens, reusing one prefill KV cache that
is cropped back between candidates. Free generation is uninformative; the model's ranking over
the 10 named options is not.

All evaluation runs nosplit (do_image_splitting=False, 64 image tokens), the champion config
CONFIRMED by vlm_step002 (11.71x faster and +8pp top-1 vs the 832-token split reference).
"""
from __future__ import annotations
import math
import random
import time

import torch
import torch.nn as nn
from PIL import Image

WNID2LABEL = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette",
              "n03000684": "chainsaw", "n03028079": "church", "n03394916": "horn",
              "n03417042": "truck", "n03425413": "pump", "n03445777": "golf",
              "n03888257": "parachute"}
LABELS = list(WNID2LABEL.values())
QUESTION = ("Which one is in this image: " + ", ".join(LABELS) + "? Answer with one word.")


def sync(device):
    if device.type == "mps": torch.mps.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize()


def sample_images(root, n, seed=42, wnids=None):
    """n images balanced over the wnids, deterministic. Returns [(path, wnid), ...].

    `wnids` restricts sampling to a subset (vlm_step007's seen/held class split). Omitting it
    reproduces the pre-step007 call sequence exactly, so step004/005/006 samples are unchanged.
    Note `per` floors at n // len(wnids), so the returned count is usually below n.
    """
    keys = sorted(wnids or WNID2LABEL)
    rng, per, out = random.Random(seed), max(1, n // len(keys)), []
    for wnid in keys:
        files = sorted((root / wnid).glob("*.JPEG"))
        out += [(f, wnid) for f in rng.sample(files, min(per, len(files)))]
    rng.shuffle(out)
    return out[:n]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on paired 0/1 outcomes. b = A right & B wrong, c = the reverse.
    Under H0 each discordant pair is a fair coin, so p = P(|X - n/2| >= |b - n/2|), X ~ Bin(n, 1/2).
    Concordant pairs carry no information and are correctly excluded.

    Kept as one exact int ratio: the tail sum reaches ~2^n, so converting it to float first overflows
    once n passes ~1020 discordant pairs -- which is not hypothetical, it killed vlm_step008's d3 arm
    at n=1129 after a 38-minute eval had already succeeded. int/int division is computed to a
    correctly-rounded float without building either huge value as a float, so it is exact AND safe."""
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def boot_ci(a, b, n_boot=10000, seed=0):
    """Paired percentile bootstrap on mean(a) - mean(b); a, b are 0/1 lists over the SAME items,
    resampled by a shared index so the pairing survives."""
    rng, n, out = random.Random(seed), len(a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


class VLMEval:
    """Wraps an Idefics3-family model with the constrained-choice eval + a swappable vision depth.

    set_depth(d) installs the first d SigLIP layers; post_layernorm and the connector are
    untouched because every intermediate layer emits the same hidden_size the connector consumes.
    set_layers(mods) installs an arbitrary stack (used to drop in a distilled student tower).
    """

    def __init__(self, model, proc, device):
        self.m, self.p, self.dev = model, proc, device
        self.img_id = model.config.image_token_id
        self.full = list(model.model.vision_model.encoder.layers)   # keep the full stack alive
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION}]}]
        self.prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        self.label_ids = [torch.tensor(proc.tokenizer(" " + l, add_special_tokens=False)
                                       ["input_ids"], device=device) for l in LABELS]

    # --- vision stack control -------------------------------------------------------------
    def set_depth(self, d):
        self.set_layers(self.full[:d])

    def set_layers(self, mods):
        self.m.model.vision_model.encoder.layers = nn.ModuleList(mods)

    # --- forward pieces -------------------------------------------------------------------
    def batch(self, img):
        """Full processor output (text + image) for one image, nosplit, on device."""
        b = self.p(text=self.prompt, images=[img], return_tensors="pt", do_image_splitting=False)
        return {k: v.to(self.dev) for k, v in b.items()}

    def pixels(self, images):
        """Image-only batch for the vision tower. Idefics3 packs a list as tiles of one sample,
        which is equivalent here: SigLIP has no cross-tile attention, so tiles == batch."""
        o = self.p.image_processor(images, return_tensors="pt", do_image_splitting=False)
        return {k: v.to(self.dev) for k, v in o.items()}

    def encode(self, b, grad=False):
        """pixels -> post-connector image embeds. Returns (feats[T, H], ms). T = 64 per image."""
        sync(self.dev); t0 = time.perf_counter()
        with torch.set_grad_enabled(grad):
            lh = self.m.model.get_image_features(
                pixel_values=b["pixel_values"],
                pixel_attention_mask=b.get("pixel_attention_mask")).last_hidden_state
            feats = self.m.model.connector(lh)
        sync(self.dev)
        return feats.reshape(-1, feats.shape[-1]), (time.perf_counter() - t0) * 1000

    def merge(self, b, feats):
        """Splice image embeds into the text embedding sequence. Verified bit-exact vs the native
        forward in vlm_step002 (max abs diff 0.0)."""
        ids = b["input_ids"]
        emb = self.m.get_input_embeddings()(ids).clone()
        pos = (ids[0] == self.img_id).nonzero(as_tuple=True)[0]
        assert len(pos) == len(feats), f"{len(pos)} slots vs {len(feats)} feats"
        emb[0, pos] = feats.to(emb.dtype)
        return emb, pos

    def prune(self, emb, pos, feats, k):
        """Keep the top-k image tokens by post-connector L2 norm, order preserved. vlm_step002
        CONFIRMED this beats an even-stride control at an identical budget (+18pp top-1)."""
        if k >= len(feats): return emb
        keep = feats.norm(dim=-1).topk(k).indices.sort().values
        drop = torch.ones(emb.shape[1], dtype=torch.bool, device=self.dev)
        drop[pos] = False
        drop[pos[keep]] = True
        return emb[:, drop]

    def forward(self, emb):
        """Prefill (TIMED) + constrained 10-way choice (untimed). Returns (logits[V], pred, ms)."""
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.dev)
        sync(self.dev); t0 = time.perf_counter()
        with torch.no_grad():
            o = self.m.model.text_model(inputs_embeds=emb, attention_mask=mask, use_cache=True)
            logits = self.m.lm_head(o.last_hidden_state[:, -1])[0].float()
        sync(self.dev); ms = (time.perf_counter() - t0) * 1000
        return logits, self.choose(logits, o.past_key_values, emb.shape[1]), ms

    def choose(self, logits, past, plen):
        """Length-normalised sum log P over each label's OWN tokens, teacher-forced, reusing the
        prefill cache (cropped back per label). Returns the argmax label index."""
        lp0, out = logits.log_softmax(-1), []
        for ids in self.label_ids:
            with torch.no_grad():
                o = self.m.model.text_model(
                    inputs_embeds=self.m.get_input_embeddings()(ids[None]), past_key_values=past,
                    attention_mask=torch.ones(1, plen + len(ids), dtype=torch.long,
                                              device=self.dev), use_cache=True)
                lg = self.m.lm_head(o.last_hidden_state)[0].float().log_softmax(-1)
            s = lp0[ids[0]] + sum(lg[i, ids[i + 1]] for i in range(len(ids) - 1))
            out.append(float(s) / len(ids))
            past.crop(plen)
        return int(torch.tensor(out).argmax())

    def run(self, img, k=64):
        """One image, current vision stack: returns (logits, pred, vision_ms, prefill_ms)."""
        b = self.batch(img)
        feats, vis_ms = self.encode(b)
        emb, pos = self.merge(b, feats)
        lg, pred, pre_ms = self.forward(self.prune(emb, pos, feats, k))
        return lg, pred, vis_ms, pre_ms


def evaluate(ev, val):
    """Top-1 + timings over `val`. Moved here from vlm_step023 at vlm_step031, which re-scores an
    already-written-up checkpoint and must use the SAME path -- a copy could drift from the quoted
    numbers. `correct` is the per-image hit vector in fixed sample_images order, so arms stay paired
    image-by-image; McNemar needs that pairing and cannot reconstruct it after the fact.
    """
    ok = vis = pre = 0.0
    correct = []
    for path, wnid in val:
        b = ev.batch(Image.open(path).convert("RGB"))
        feats, v = ev.encode(b)
        emb, _ = ev.merge(b, feats)
        _, pred, p = ev.forward(emb)
        hit = LABELS[pred] == WNID2LABEL[wnid]
        correct.append(int(hit))
        ok += hit; vis += v; pre += p
    n = len(val)
    return {"top1": ok / n, "vision_ms": vis / n, "prefill_ms": pre / n, "n": n, "correct": correct}
