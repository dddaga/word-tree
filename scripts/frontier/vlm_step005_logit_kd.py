"""vlm_step005: add the LOGIT-KD term vlm_step004's caveat demanded.

vlm_step004 CONFIRMED that output-distillation recovers a truncated SmolVLM vision tower (d6:
0.560 vs naive 0.120 vs teacher 0.740, 16.58% params saved, 1.62x) and that recovery is
depth-INDEPENDENT over 6..9 layers — so the ceiling is the objective, not student capacity. Its
caveat: `same` 0.660 and `agree` 0.420 sit BELOW top-1 0.560, i.e. the student reaches the right
label partly by a DIFFERENT route. Feature-MSE does not preserve the teacher's function
pointwise — invisible on a closed 10-way task, maybe not on open-ended VLM generation.

step605's actual recipe distilled soft OUTPUTS, not intermediate features. This script restores
that term: loss = alpha * relMSE(post-connector features) + beta * KL(student || teacher) on the
LM's next-token distribution at the answer position — the only place the whole pipeline's function
is observable. --beta 0 reproduces vlm_step004 exactly; --alpha 0 gives pure logit-KD, isolating
which term carries the recovery. Eval = the same 50 val images (seed 42) and constrained 10-way
choice as vlm_step002/003/004, so top-1 compares directly against 0.740 / 0.120 / 0.560. `agree`
is the target metric: argmax-identity with the teacher, which feature-MSE failed to preserve.

Output: results/frontier/vlm_step005_logit_kd_d{depth}_a{alpha}_b{beta}__{SLOT}.json"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from scripts.frontier.vlm_eval import LABELS, WNID2LABEL, VLMEval, sample_images

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--alpha", type=float, default=1.0, help="weight on feature relative-MSE")
parser.add_argument("--beta", type=float, default=1.0, help="weight on next-token logit KD")
parser.add_argument("--accum", type=int, default=8, help="grad accumulation = effective batch")
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_eval", type=int, default=50)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs = 24, 10, 1

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
TAG = f"d{args.depth}_a{args.alpha:g}_b{args.beta:g}"
OUT = ROOT / "results" / "frontier" / f"vlm_step005_logit_kd_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def load(p):
    return Image.open(p).convert("RGB")


class Path1:
    """Differentiable one-image path: pixels -> student tower -> connector -> splice into the
    (fixed) text embeddings -> text model -> next-token logits. The text ids are IDENTICAL for
    every image (fixed prompt, nosplit => always 64 image tokens), so build them once."""

    def __init__(self, ev, sample_img):
        self.ev = ev
        b = ev.batch(sample_img)
        self.ids = b["input_ids"]
        self.pos = (self.ids[0] == ev.img_id).nonzero(as_tuple=True)[0]
        self.mask = torch.ones(self.ids.shape, dtype=torch.long, device=ev.dev)

    def __call__(self, img, grad=False):
        ev = self.ev
        feats, _ = ev.encode(ev.pixels([img]), grad=grad)
        with torch.set_grad_enabled(grad):
            emb = ev.m.get_input_embeddings()(self.ids).clone()
            emb[0, self.pos] = feats.to(emb.dtype)
            o = ev.m.model.text_model(inputs_embeds=emb, attention_mask=self.mask)
            logits = ev.m.lm_head(o.last_hidden_state[:, -1])[0].float()
        return feats, logits


def cache_teacher(path1, ev, train):
    """Full-tower features AND next-token logits, once. fp16 on CPU: 74 KB + ~98 KB per image."""
    ev.set_depth(len(ev.full))
    feats, logits, t0 = [], [], time.perf_counter()
    for i, (p, _) in enumerate(train):
        f, lg = path1(load(p))
        feats.append(f.cpu().half()); logits.append(lg.cpu().half())
        if i % 400 == 0:
            print(f"    teacher cache {i}/{len(train)}", flush=True)
    print(f"    teacher cache done in {time.perf_counter() - t0:.0f}s", flush=True)
    return feats, logits


def train_student(path1, ev, train, tf, tl):
    """Only the truncated tower moves; everything else is frozen, so any accuracy change is
    attributable to it. Gradients still FLOW through the frozen text model to carry the KD term."""
    ev.m.requires_grad_(False)
    student = nn.ModuleList([copy.deepcopy(l) for l in ev.full[:args.depth]]).train()
    student.requires_grad_(True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    hist = []
    for ep in range(args.epochs):
        ev.set_layers(student)
        acc = dict.fromkeys(("mse", "kl", "cos", "top1"), 0.0)
        opt.zero_grad(set_to_none=True)
        for i, (p, _) in enumerate(train):
            t_f, t_l = tf[i].to(DEVICE).float(), tl[i].to(DEVICE).float()
            s_f, s_l = path1(load(p), grad=True)
            mse = F.mse_loss(s_f, t_f) / (t_f.pow(2).mean() + 1e-8)
            kl = F.kl_div(s_l.log_softmax(-1)[None], t_l.log_softmax(-1)[None],
                          log_target=True, reduction="batchmean")
            # accumulate: the KD path is one image per forward, but a bs=1 update is far too
            # noisy (smoke test drove cosine to -0.03). accum=8 matches vlm_step004's batch.
            ((args.alpha * mse + args.beta * kl) / args.accum).backward()
            if (i + 1) % args.accum == 0 or i + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            acc["mse"] += float(mse.detach()); acc["kl"] += float(kl.detach())
            acc["cos"] += float(F.cosine_similarity(s_f.detach().flatten()[None],
                                                    t_f.flatten()[None], dim=-1))
            acc["top1"] += float(s_l.detach().argmax() == t_l.argmax())
        n = len(train)
        hist.append({"epoch": ep + 1, **{k: v / n for k, v in acc.items()}})
        print(f"  ep {ep+1}/{args.epochs}  rel_mse {acc['mse']/n:.4f}  kl {acc['kl']/n:.4f}  "
              f"cosine {acc['cos']/n:.4f}  tok_agree {acc['top1']/n:.4f}", flush=True)
    return student.eval(), hist


def evaluate(ev, arms, val):
    """Every arm on every image; the first arm is the reference for same/agree/KL."""
    acc = {a: dict.fromkeys(("correct", "same", "agree", "kl", "vis", "pre"), 0.0) for a, _ in arms}
    for i, (path, wnid) in enumerate(val):
        img, gold, ref = load(path), LABELS.index(WNID2LABEL[wnid]), None
        for name, mods in arms:
            ev.set_layers(mods)
            lg, pred, vm, pm = ev.run(img)
            lp = F.log_softmax(lg, -1)
            if ref is None: ref = (pred, lp, lg.argmax().item())
            a = acc[name]
            a["correct"] += pred == gold; a["same"] += pred == ref[0]
            a["agree"] += lg.argmax().item() == ref[2]
            a["kl"] += float(F.kl_div(lp, ref[1], log_target=True, reduction="sum"))
            a["vis"] += vm; a["pre"] += pm
        if (i + 1) % 10 == 0:
            print(f"  eval [{i+1}/{len(val)}]", flush=True)
    return acc


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step005 logit-KD  device={DEVICE}  depth={args.depth}  "
          f"alpha={args.alpha}  beta={args.beta}  n_train={args.n_train}  epochs={args.epochs}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    total_p = sum(p.numel() for p in m.parameters())
    per_layer = sum(p.numel() for p in ev.full[0].parameters())
    print(f"  total {total_p:,} params | vision layer {per_layer:,} x {len(ev.full)}")

    train = sample_images(DATA / "train", args.n_train, seed=7)
    val = sample_images(DATA / "val", args.n_eval, seed=42)
    path1 = Path1(ev, load(train[0][0]))
    tf, tl = cache_teacher(path1, ev, train)
    student, hist = train_student(path1, ev, train, tf, tl)

    arms = [("teacher_d12", ev.full), (f"naive_d{args.depth}", ev.full[:args.depth]),
            (f"kd_d{args.depth}", list(student))]
    acc = evaluate(ev, arms, val)

    n, cut = len(val), total_p - (len(ev.full) - args.depth) * per_layer
    ref_tot = (acc["teacher_d12"]["vis"] + acc["teacher_d12"]["pre"]) / n
    res = {"step": "vlm_step005", "model": args.model, "device": str(DEVICE),
           "depth": args.depth, "alpha": args.alpha, "beta": args.beta, "n_train": len(train),
           "n_eval": n, "epochs": args.epochs, "lr": args.lr, "total_params": total_p,
           "vision_layer_params": per_layer, "train_history": hist, "arms": {}}
    print(f"\n  {'arm':<14} {'params':>12} {'%save':>6} {'top1':>6} {'same':>6} {'agree':>6} "
          f"{'KL':>7} {'vis_ms':>8} {'pre_ms':>7} {'total':>8} {'speed':>7}")
    for name, _ in arms:
        d = acc[name]; v, p_ = d["vis"] / n, d["pre"] / n
        prm = total_p if name == "teacher_d12" else cut
        row = {"params": prm, "params_saved_pct": round(100 * (total_p - prm) / total_p, 2),
               "top1": d["correct"] / n, "same_class_as_teacher": d["same"] / n,
               "agree_with_teacher": d["agree"] / n, "kl_vs_teacher": d["kl"] / n,
               "vision_ms": round(v, 2), "prefill_ms": round(p_, 2),
               "total_ms": round(v + p_, 2), "speedup": round(ref_tot / (v + p_), 3)}
        res["arms"][name] = row
        print(f"  {name:<14} {prm:>12,} {row['params_saved_pct']:>6.2f} {row['top1']:>6.3f} "
              f"{row['same_class_as_teacher']:>6.3f} {row['agree_with_teacher']:>6.3f} "
              f"{row['kl_vs_teacher']:>7.3f} {v:>8.1f} {p_:>7.1f} {v+p_:>8.1f} "
              f"{row['speedup']:>6.2f}x")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    # keep the student: a KD run is multi-hour, downstream compounding must not retrain it.
    ckpt = OUT.with_suffix(".pt"); torch.save(student.state_dict(), ckpt)
    print(f"-> {OUT}\n-> {ckpt}")


if __name__ == "__main__":
    main()
