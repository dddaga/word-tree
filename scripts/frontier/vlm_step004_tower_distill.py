"""vlm_step004: DISTIL the truncated vision tower — the fix vlm_step003 demanded.

vlm_step003 CONFIRMED that zero-shot truncation of SmolVLM-256M's 12-layer SigLIP tower collapses
to chance at every depth tried (d9 0.080 / d6 0.120 / d3 0.100 vs chance 0.100, KL 9.97-12.46),
while the TOKEN axis degrades gracefully (32 tok = -6pp, 16 tok = -14pp). Cutting what the tower
already computed is survivable; cutting the computing is not. The tower's params ARE removable
(d6 = 16.58% of the whole model, more than the entire readout share vlm_step001 declared
Amdahl-dead) but its function is not free — post_layernorm and the connector receive an
unadapted feature distribution.

So apply the recipe that worked twice: step605 (K=1 soft-KD student, 5.26x wall-time) and
det_step002 (box-head distillation). Freeze the full 12-layer tower as TEACHER, train a d-layer
STUDENT initialised from its first d layers to match the teacher's POST-CONNECTOR features. This
does not contradict step989/llm_step002 — those KILLED replacing a TRANSFORM block's internals
with a sparse head. Here the student is the same architecture, only shorter, and is asked to
reproduce the stack's OUTPUT. Different claim, so it needs its own evidence.

Arms: teacher_d12 (reference) | naive_d{depth} (untrained truncation, reproduces step003) |
distil_d{depth} (this experiment). Same 50 val images and the same constrained 10-way choice as
step002/003, so top-1 is directly comparable to 0.740 (teacher) and 0.120 (naive d6).

Output: results/frontier/vlm_step004_tower_distill__{SLOT}.json
"""
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
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_eval", type=int, default=50)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.n_eval, args.epochs, args.batch = 32, 10, 1, 4

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
OUT = ROOT / "results" / "frontier" / f"vlm_step004_tower_distill_d{args.depth}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def tower_feats(ev, paths, grad=False):
    """Post-connector features for a list of image paths -> (B, 64, H)."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    flat, _ = ev.encode(ev.pixels(ims), grad=grad)
    return flat.view(len(ims), -1, flat.shape[-1])


def cache_teacher(ev, train):
    """Run the FULL tower once over the train subset; keep targets on CPU in fp16 (64x576 per
    image = 74 KB, so 2000 images = 147 MB). Avoids re-running the teacher every epoch."""
    ev.set_depth(len(ev.full))
    out, t0 = [], time.perf_counter()
    for i, chunk in enumerate(batches(train, args.batch)):
        out.append(tower_feats(ev, [p for p, _ in chunk]).cpu().half())
        if i % 50 == 0:
            print(f"    teacher cache {i * args.batch}/{len(train)}", flush=True)
    print(f"    teacher cache done in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def train_student(ev, train, targets):
    """Train a depth-truncated copy of the tower to match the teacher's post-connector features.
    Everything else (patch embeddings, post_layernorm, connector, text model) stays frozen — the
    student is the ONLY thing that moves, so an accuracy change is attributable to it."""
    ev.m.requires_grad_(False)
    student = nn.ModuleList([copy.deepcopy(l) for l in ev.full[:args.depth]]).train()
    student.requires_grad_(True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    hist = []
    for ep in range(args.epochs):
        ev.set_layers(student)
        tot = cos = n = 0.0
        for chunk, tgt in zip(batches(train, args.batch), targets):
            t = tgt.to(DEVICE).float()
            s = tower_feats(ev, [p for p, _ in chunk], grad=True)
            # relative MSE: scale-free, so the number is comparable across depths
            loss = F.mse_loss(s, t) / (t.pow(2).mean() + 1e-8)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sd = s.detach()
            tot += float(loss.detach()) * len(chunk)
            cos += float(F.cosine_similarity(sd.flatten(1), t.flatten(1), dim=-1).mean()) * len(chunk)
            n += len(chunk)
        hist.append({"epoch": ep + 1, "rel_mse": tot / n, "cosine": cos / n})
        print(f"  ep {ep+1}/{args.epochs}  rel_mse {tot/n:.4f}  cosine {cos/n:.4f}", flush=True)
    return student.eval(), hist


def evaluate(ev, arms, val):
    """Every arm on every image; the first arm is the reference for same/agree/KL."""
    acc = {a: dict.fromkeys(("correct", "same", "agree", "kl", "vis", "pre"), 0.0) for a, _ in arms}
    for i, (path, wnid) in enumerate(val):
        img = Image.open(path).convert("RGB")
        gold = LABELS.index(WNID2LABEL[wnid])
        ref = None
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
    print(f"{'='*78}\nvlm_step004 tower distill  device={DEVICE}  depth={args.depth}  "
          f"n_train={args.n_train}  epochs={args.epochs}  bs={args.batch}  lr={args.lr}")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)
    total_p = sum(p.numel() for p in m.parameters())
    per_layer = sum(p.numel() for p in ev.full[0].parameters())
    print(f"  total {total_p:,} params | vision layer {per_layer:,} x {len(ev.full)}")

    train = sample_images(DATA / "train", args.n_train, seed=7)
    val = sample_images(DATA / "val", args.n_eval, seed=42)
    targets = cache_teacher(ev, train)
    student, hist = train_student(ev, train, targets)

    arms = [("teacher_d12", ev.full), (f"naive_d{args.depth}", ev.full[:args.depth]),
            (f"distil_d{args.depth}", list(student))]
    acc = evaluate(ev, arms, val)

    n, cut = len(val), total_p - (len(ev.full) - args.depth) * per_layer
    ref_tot = (acc["teacher_d12"]["vis"] + acc["teacher_d12"]["pre"]) / n
    res = {"step": "vlm_step004", "model": args.model, "device": str(DEVICE),
           "depth": args.depth, "n_train": len(train), "n_eval": n, "epochs": args.epochs,
           "batch": args.batch, "lr": args.lr, "total_params": total_p,
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
    # keep the student: compounding (depth x tokens) and any in-model swap must not have to
    # retrain a multi-hour distillation just to reuse it.
    ckpt = OUT.with_suffix(".pt"); torch.save(student.state_dict(), ckpt)
    print(f"-> {OUT}\n-> {ckpt}")


if __name__ == "__main__":
    main()
