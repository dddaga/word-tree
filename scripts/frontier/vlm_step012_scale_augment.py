"""vlm_step012: re-distil d6 WITH scale augmentation -- capacity or distribution? (part 5 SS18.1)

vlm_step011 CONFIRMED that d6's teacher parity does not survive small objects: I(0.5) = -12.78pp,
I(0.25) = -24.41pp, both far outside the +-2.0pp tolerance, with d6 collapsing to 0.252 top-1 where
the teacher still holds 0.494. That result has two explanations and step011 cannot separate them:

  (a) CAPACITY   -- the six deleted vision layers are the ones carrying fine spatial detail. If so,
                    no amount of retraining recovers it and the drone ships d12 + int8 only.
  (b) DISTRIBUTION -- d6 was distilled exclusively on whole-object Imagenette and has simply never
                    seen a small target. If so, scale-augmented distillation recovers most of it and
                    the -33.1%-bytes configuration is restored.

They point opposite ways on the only decision that matters, so this run separates them by changing
exactly one thing about the distillation: the train-time object-scale distribution.

WHY THIS IS A CLEAN ONE-VARIABLE RUN. The control is the EXISTING d6 checkpoint
(vlm_step004 d6_25ep9k), not a new arm, so no second training run is needed and the comparison costs
one distillation instead of two. That only holds if the train set is bit-identical, which took some
care: `sample_images` floors to per = n // 10 images per class, so the SAME nominal 10k request
yields 9469 images while step004's literal `--n_train 9469` yields 9352. The default below is
therefore step004's argument, not a round number -- passing 10000 would silently train on a
117-image-different set and quietly turn this into a two-variable run. epochs/batch/lr/seed are
likewise pinned to step004's (25 / 8 / 2e-4 / 7); nothing but the transform differs.

THE AUGMENTATION IS DETERMINISTIC PER PATH, WHICH IS A CORRECTNESS REQUIREMENT, NOT A STYLE CHOICE.
The teacher's targets are cached once while the student is re-encoded every epoch. A transform that
re-randomised per call would train the student to match teacher features computed from a DIFFERENT
view of the same image -- a silently corrupted objective. Seeding the per-image scale from the file
name makes the pairing exact and lets the teacher see precisely the view the student trains on.

Scale is drawn from a CONTINUOUS range while step011 tests at three discrete values, so a recovery
is generalisation across scale rather than memorisation of the test manipulation. Half the images
are left untouched so the whole-object competence that the original claim rests on is still trained.

PRE-REGISTERED DECISION RULE (written before the run). This script only produces the checkpoint;
the verdict comes from re-running the vlm_step011 grid UNCHANGED with --ckpt pointing here, which is
why no eval logic is duplicated below. Let I_aug(f) be step011's double difference for this student:
  * DISTRIBUTION CONFIRMED iff I_aug(f)'s 95% CI is contained in +-2.0pp at BOTH f = 0.5 and 0.25
    (parity restored; the drone claim is un-scoped).
  * CAPACITY CONFIRMED iff I_aug(0.25)'s upper bound is still < -2.0pp (no recovery).
  * PARTIAL if it recovers at f = 0.5 but not 0.25 -- report the scale at which it breaks, do not
    round it into either verdict.
  * SECOND, INDEPENDENT CRITERION, and a real way for this to fail: `d6_f100` must stay within
    2.0pp of the teacher's 0.711. Augmentation that buys small-object skill by spending the
    whole-object parity has not helped -- that parity IS the original claim.
  * ASYMMETRY, stated up front: a NULL here is confound-free and therefore decisive. The training
    distribution would have matched the test manipulation exactly (same gray canvas, same centring)
    and still failed, which no confound can rescue. A POSITIVE is weaker: it is confounded between
    genuine scale-invariance and mere adaptation to out-of-distribution gray padding, and needs a
    follow-up on real small objects before the drone claim is un-scoped.

Output: results/frontier/vlm_step012_scale_augment_{TAG}__{SLOT}.json  (+ .pt student checkpoint)
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(ROOT := Path(__file__).parent.parent.parent))
# The 5060ti's shared venv belongs to teammates and must not be modified; `vlm_libs` is a private
# --target install that shadows it for THIS process only.
if (_LIBS := ROOT / "vlm_libs").is_dir(): sys.path.insert(0, str(_LIBS))

import torch
from PIL import Image

from scripts.frontier.vlm_distill import cache_teacher, park, train_student
from scripts.frontier.vlm_eval import VLMEval, sample_images

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--depth", type=int, default=6, help="pinned to the arm step011 falsified")
parser.add_argument("--n_train", type=int, default=9469,
                    help="step004 d6_25ep9k's literal arg; sample_images floors it to its exact 9352")
parser.add_argument("--epochs", type=int, default=25, help="step004 d6_25ep9k budget, do not change")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--p_keep", type=float, default=0.5,
                    help="fraction of images left untouched, so whole-object competence is trained")
parser.add_argument("--scale_lo", type=float, default=0.2)
parser.add_argument("--scale_hi", type=float, default=0.9,
                    help="continuous; step011 tests at 0.5/0.25, so recovery is generalisation")
parser.add_argument("--seed", type=int, default=7, help="step004's train-sampling seed")
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_train, args.epochs, args.batch = 40, 1, 4

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")) \
    if args.device == "auto" else torch.device(args.device)
SLOT = os.environ.get("SGN_SLOT", "local")
GRAY = (128, 128, 128)  # identical to vlm_step011's canvas, so train and test padding match exactly
TAG = f"d{args.depth}_{args.epochs}ep_aug{int(args.scale_lo*100)}-{int(args.scale_hi*100)}"
OUT = ROOT / "results" / "frontier" / f"vlm_step012_scale_augment_{TAG}__{SLOT}.json"
DATA = ROOT / "data" / "imagenette2-320"


def scale_for(path):
    """The scale this image is ALWAYS shown at. Pure function of the file name: seeding from the
    path (not from a global RNG whose call order depends on batching) is what makes the cached
    teacher target and the per-epoch student input the same view."""
    h = hashlib.sha256(f"{args.seed}:{Path(path).name}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    if rng.random() < args.p_keep:
        return 1.0
    return rng.uniform(args.scale_lo, args.scale_hi)


def shrink(img, s):
    """Paste `img` resized to a fraction `s` of its own size, centred on a gray canvas of the
    ORIGINAL size -- byte-for-byte the manipulation vlm_step011 evaluates, so a recovery here is a
    recovery there. Canvas size is held fixed, so the processor's input resolution never varies."""
    if s >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), GRAY)
    canvas.paste(small, ((w - small.size[0]) // 2, (h - small.size[1]) // 2))
    return canvas


def augment(img, path):
    return shrink(img, scale_for(path))


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"{'='*78}\nvlm_step012 scale-augmented distill  device={DEVICE}  depth={args.depth}  "
          f"epochs={args.epochs}  lr={args.lr}  p_keep={args.p_keep}  "
          f"scale~U({args.scale_lo},{args.scale_hi})")
    m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval().to(DEVICE)
    m.requires_grad_(False)
    ev = VLMEval(m, AutoProcessor.from_pretrained(args.model), DEVICE)

    train = sample_images(DATA / "train", args.n_train, seed=args.seed)
    scales = [scale_for(p) for p, _ in train]
    kept = sum(1 for s in scales if s >= 1.0)
    print(f"  {len(train)} train images (step004 d6_25ep9k used 9352 -- these must match)")
    print(f"  augmented: {len(train)-kept} shrunk, {kept} untouched ({100*kept/len(train):.1f}%), "
          f"mean scale {sum(scales)/len(scales):.3f}", flush=True)

    t0 = time.perf_counter()
    targets = cache_teacher(ev, train, args.batch, tf=augment)  # teacher sees the augmented view
    park(ev, "cpu")  # text stack + frozen teacher layers are idle for the whole training loop
    student, hist = train_student(ev, train, targets, args.depth, args.epochs, args.batch,
                                  args.lr, DEVICE, tf=augment)
    park(ev, DEVICE)
    mins = (time.perf_counter() - t0) / 60

    res = {"step": "vlm_step012", "device": str(DEVICE), "depth": args.depth,
           "n_train": len(train), "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
           "seed": args.seed, "p_keep": args.p_keep, "scale_lo": args.scale_lo,
           "scale_hi": args.scale_hi, "frac_untouched": kept / len(train),
           "mean_scale": sum(scales) / len(scales), "train_minutes": round(mins, 1),
           "control_ckpt": "results/frontier/vlm_step004_tower_distill_d6_25ep9k__mini_mps.pt",
           "train_history": hist}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(res, indent=2))
    ckpt = OUT.with_suffix(".pt"); torch.save(student.state_dict(), ckpt)
    print(f"\n  trained in {mins:.1f} min; final rel_mse {hist[-1]['rel_mse']:.4f} "
          f"cosine {hist[-1]['cosine']:.4f}")
    print("  rel_mse is NOT comparable to step004's: the targets are features of DIFFERENT (shrunk)"
          "\n  images, so a higher loss here does not mean a worse student. The verdict is step011.")
    print(f"-> {OUT}\n-> {ckpt}\n\n  NEXT (the pre-registered verdict, grid unchanged):\n"
          f"    scripts/frontier/vlm_step011_fine_detail.py --ckpt {ckpt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
