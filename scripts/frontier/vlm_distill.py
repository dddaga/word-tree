"""Shared tower-distillation helpers for the vlm_step00X frontier line.

LIBRARY ONLY -- no argparse, no I/O at import, so step scripts can import it. Extracted at
vlm_step007 because the teacher-caching + student-training loop is now needed by a third step and
the repo's 200-line file limit forbids a third copy.

vlm_step004 and vlm_step005 keep their inline copies deliberately: both are DONE experiments whose
scripts must stay reproducible exactly as run. The code here is a parameterised transcription of
vlm_step004's -- same objective (relative MSE on post-connector features), same AdamW + weight
decay 0.01, same grad clip 1.0 -- so a step007 result stays comparable to step004's numbers.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def tower_feats(ev, paths, grad=False, tf=None):
    """Post-connector features for a list of image paths -> (B, 64, H).

    `tf` is an optional (PIL image, path) -> PIL image train-time transform. It MUST be a pure
    function of the path: the teacher's targets are cached once (cache_teacher) while the student is
    re-encoded every epoch, so a transform that re-randomises per call would train the student to
    match teacher features computed from a *different* view of the image. Deterministic-per-path
    keeps the pairing exact and costs nothing extra."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    if tf is not None:
        ims = [tf(im, p) for im, p in zip(ims, paths)]
    flat, _ = ev.encode(ev.pixels(ims), grad=grad)
    return flat.view(len(ims), -1, flat.shape[-1])


def cache_teacher(ev, train, bs, log_every=50, tf=None):
    """Run the FULL tower once over the train subset; keep targets on CPU in fp16 (64x576 per
    image = 74 KB). Avoids re-running the frozen teacher every epoch.

    `tf` is forwarded so the teacher sees exactly the view the student will be trained on -- the
    objective is 'match the teacher ON THIS INPUT', so an augmented student paired with unaugmented
    targets would be a different (and wrong) objective."""
    ev.set_depth(len(ev.full))
    out, t0 = [], time.perf_counter()
    for i, chunk in enumerate(batches(train, bs)):
        out.append(tower_feats(ev, [p for p, _ in chunk], tf=tf).cpu().half())
        if i % log_every == 0:
            print(f"    teacher cache {i * bs}/{len(train)}", flush=True)
    print(f"    teacher cache done in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def park(ev, device):
    """Move everything the distillation loop does NOT touch onto `device` (and back afterwards).

    Training only needs the vision tower + connector. The text stack (~0.68 GB fp32) and the frozen
    teacher layers (~0.34 GB) sit idle through every epoch, so on a GPU shared with someone else's
    job they are pure occupancy. Call park(ev, "cpu") after cache_teacher() and park(ev, DEVICE)
    before eval. No-op on an empty GPU other than a few transfers.
    """
    ev.m.model.text_model.to(device)
    ev.m.lm_head.to(device)
    for l in ev.full:
        l.to(device)


def new_student(ev, depth, device=None):
    """A trainable deepcopy of the teacher's first `depth` layers -- the truncated tower's init."""
    st = nn.ModuleList([copy.deepcopy(l) for l in ev.full[:depth]])
    return st.to(device) if device is not None else st


def load_student(ev, ckpt, depth, device):
    """Rebuild a saved student. Returns it in eval mode, on `device`."""
    st = new_student(ev, depth)
    st.load_state_dict(torch.load(ckpt, map_location=device))
    return st.eval().to(device)


def train_or_resume(ev, ckpt, train, targets, depth, epochs, bs, lr, device, tf=None):
    """Reuse `ckpt` if it already exists, else train a student and save it there -> (student, hist).

    A depth sweep runs for hours, and until this existed one late failure discarded every arm that
    had already finished: vlm_step008 lost d3's completed 25-epoch student AND the shared eval when
    d9 hit a CUDA OOM caused by another user's process on the same card. Reusing the saved arm costs
    nothing in validity -- the checkpoint IS the object the run would have produced. `hist` is None
    for a resumed arm so that a caller reading the loss curve (floor vs ceiling) reports 'unknown'
    instead of inventing one; the curve lives in the original run's log, not in the weights.
    """
    if Path(ckpt).exists():
        print(f"\n--- depth {depth} <- {Path(ckpt).name} (trained already, not repeated) ---", flush=True)
        return load_student(ev, ckpt, depth, device), None
    print(f"\n--- distilling depth {depth} ---", flush=True)
    st, hist = train_student(ev, train, targets, depth, epochs, bs, lr, device, tf=tf)
    torch.save(st.state_dict(), ckpt)
    return st, hist


def train_student(ev, train, targets, depth, epochs, bs, lr, device, tf=None):
    """Train a depth-truncated copy of the tower to match the teacher's post-connector features.

    Everything else (patch embeddings, post_layernorm, connector, text model) stays frozen -- the
    student is the ONLY thing that moves, so an accuracy change is attributable to it. The loss is
    relative MSE, mse(s,t) / mean(t^2), which is scale-free and therefore comparable across depths.
    """
    ev.m.requires_grad_(False)
    student = new_student(ev, depth, device).train()
    student.requires_grad_(True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for ep in range(epochs):
        ev.set_layers(student)
        tot = cos = n = 0.0
        for chunk, tgt in zip(batches(train, bs), targets):
            t = tgt.to(device).float()
            s = tower_feats(ev, [p for p, _ in chunk], grad=True, tf=tf)
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
        print(f"  ep {ep+1}/{epochs}  rel_mse {tot/n:.4f}  cosine {cos/n:.4f}", flush=True)
    return student.eval(), hist
