"""Shared teacher-loading + KD training loop for the seg line.

LIBRARY ONLY — no argparse, no I/O at import. Extracted from seg_step010 at seg_step022, which
varies the M_FPM composition under the SAME loss and must not fork the training code: a second
copy would drift from the numbers seg_step010–014 are quoted under. The move is a pure lift —
every line is the seg_step010 original — and was verified inert by a 2-epoch/200-image run whose
best_mAP and per-class AP are bit-identical before and after (0.1056; see seg_step022 docstring).

`fit()` takes the caller's argparse namespace directly rather than a re-declared config object,
so a flag added to one step script is visible here without a second definition to keep in sync.
Required attributes: arm, alpha, beta, temp, shuffle_hint, decor_lambda, epochs, save_ckpt, k.
"""
from __future__ import annotations
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.seg.seg_common import decor_penalty
from scripts.seg.seg_encoders import SegNet
from scripts.seg.seg_eval import BATCH, evaluate

ROOT = Path(__file__).parent.parent.parent


def load_teacher(args, device, slot):
    """seg_step007 already trained and cached this teacher; refuse to run without it."""
    cp = ROOT / "results" / "seg" / (f"teacher_{args.teacher_arm}_e{args.teacher_epochs}"
                                     f"_seed{args.teacher_seed}__{slot}.pt")
    if not cp.exists():
        raise FileNotFoundError(f"{cp} missing — run seg_step007 first so the teacher is shared")
    torch.manual_seed(args.teacher_seed)
    model = SegNet(args.teacher_arm, k=args.k).to(device)
    d = torch.load(cp, map_location=device)
    model.load_state_dict(d["state"])
    model.eval()
    print(f"  teacher {cp.name}  best_mAP={d['best_mAP']:.4f}", flush=True)
    return model


def fit(student, teacher, tx, ty, vx, vy, args, device, seed):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    nb = (len(tx) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * nb,
                                                pct_start=0.1, anneal_strategy="cos")
    best, best_ep, t0 = 0.0, 0, time.time()
    best_state = None
    g = torch.Generator().manual_seed(seed)
    T, hint = args.temp, args.arm != "H0"
    ch_perm = None   # set on the first batch; fixed thereafter (see hint block below).
    # NB: NOT named `perm` — that name is the per-epoch batch order a few lines down.
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(len(tx), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x = tx[b].to(device)
            with torch.no_grad():
                t_enc = teacher.enc(x)
                t_cl = teacher.class_logits(teacher.head(teacher.mfpm(t_enc)))
            s_enc = student.enc(x)
            out = student.head(student.mfpm(s_enc))
            cl = student.class_logits(out)
            loss = F.binary_cross_entropy_with_logits(cl, ty[b].to(device))
            loss = loss + args.alpha * T * T * F.binary_cross_entropy_with_logits(
                cl / T, torch.sigmoid(t_cl / T))
            if hint:
                # Sliced init => student channel j IS teacher channel j. No projection needed.
                tgt = t_enc[:, :s_enc.shape[1]]
                if args.shuffle_hint:
                    # Same channels, same statistics, alignment destroyed. Fixed permutation, seeded
                    # independently of `seed` so every seed of this arm sees the SAME misalignment.
                    if ch_perm is None:
                        ch_perm = torch.randperm(
                            tgt.shape[1], generator=torch.Generator().manual_seed(0)).to(tgt.device)
                    tgt = tgt[:, ch_perm]
                loss = loss + args.beta * F.mse_loss(s_enc, tgt)
            loss = loss + args.decor_lambda * decor_penalty(out, args.k)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(student, vx, vy, device)
        if m["mAP"] > best:
            best, best_ep = m["mAP"], ep + 1
            if args.save_ckpt:   # keep the BEST epoch, not the last — matches the reported number
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    e{ep+1:3d}/{args.epochs}  mAP={m['mAP']:.4f}  best={best:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return best, best_ep, best_state
