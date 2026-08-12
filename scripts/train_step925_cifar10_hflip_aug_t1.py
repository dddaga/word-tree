"""Step 925: CIFAR-10 hflip-aug T1 (75ep, 50% data).

MOTIVATION
==========
Follows from step924 T0 winners. On Imagenette, aug gives +0.43–0.99pp T2.
This confirms whether the mechanism holds on CIFAR-10 at calibration level.

PREREQUISITE
============
data/store_cifar10_aug.h5 must exist. Run first if missing:
    python scripts/extract_cifar10_vgg16_features_aug.py --device mps

CONFIGS (run only the T0 winners from step924)
===============================================
Default runs A_aug_n2048 + B_aug_n4096 + C_aug_n8192 vs Ref_noaug.
Override with --configs to run subset.

ADVANCE RULE (T1, 75ep, 50% data)
==================================
  +aug delta ≥ +0.5pp → advance to T2 (step926) — paper claim candidate
  +aug delta +0.2–+0.5pp → VIABLE — advance as supporting evidence
  +aug delta < 0pp → KILL — T0 gain was noise

SLOT
====
5060ti_cuda preferred (multi-N sweep, N=8192 needs RAM).
studio_mps acceptable for N=2048 / N=4096 only.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=75)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store_cifar10.h5")
parser.add_argument("--data_aug", default="data/store_cifar10_aug.h5")
parser.add_argument("--configs",  default="Ref_noaug,A_aug_n2048,B_aug_n4096,C_aug_n8192")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N_IN   = 25088
N_OUT  = 10
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step925_cifar10_hflip_aug_t1_seed{SEED}__{SLOT}.json"

# T2 anchors
STEP882_LINEAR = 0.8624
STEP882_N2048  = 0.8069
STEP909_N4096  = 0.8253
STEP914_N8192  = 0.8358

# (N, K_in, use_aug, description)
CONFIGS = {
    "Ref_noaug":   (2048, 25, False, "N=2048 K_in=25 no-aug — canonical T1 reference"),
    "A_aug_n2048": (2048, 25, True,  "N=2048 K_in=25 hflip-aug"),
    "B_aug_n4096": (4096, 15, True,  "N=4096 K_in=15 hflip-aug"),
    "C_aug_n8192": (8192, 15, True,  "N=8192 K_in=15 hflip-aug"),
}


def make_model(N: int, K_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4)
    K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class DeltaW(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = resonant

        @property
        def W_pos(self):   return self.m.W_pos
        @property
        def W_phase(self): return getattr(self.m, "W_phase", None)
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            W_h       = self.m.W_pos[:self.m.base.N_hidden]
            dw        = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(self.m.base.K_iter):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb  = Z_nb * proj_coeff.abs()
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


def make_subset_loader(data_path: str, fraction: float = 0.5) -> torch.utils.data.DataLoader:
    tr_full, _ = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    n_full  = len(tr_full.dataset)
    n_sub   = int(n_full * fraction)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_sub]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )


def main():
    clean_path = ROOT / args.data
    aug_path   = ROOT / args.data_aug
    if not clean_path.exists():
        print(f"ERROR: {clean_path} not found."); sys.exit(1)
    if not aug_path.exists():
        print(f"ERROR: {aug_path} not found.")
        print(f"  Run first: python scripts/extract_cifar10_vgg16_features_aug.py --device mps")
        sys.exit(1)

    print("Building data loaders...")
    _, va        = make_loaders(str(clean_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    tr_clean_sub = make_subset_loader(str(clean_path), fraction=0.5)
    tr_aug_sub   = make_subset_loader(str(aug_path),   fraction=0.5)

    print(f"\n{'='*70}")
    print(f"step925 — CIFAR-10 hflip-aug T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Context: step924 T0 winners advance to T1")
    print(f"  Advance: ≥+0.5pp → T2 paper claim; +0.2–0.5pp → VIABLE")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        N, K_in, use_aug, desc = CONFIGS[key]
        model  = make_model(N, K_in)
        n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tr     = tr_aug_sub if use_aug else tr_clean_sub

        print(f"\n{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  N={N}  K_in={K_in}  aug={'yes' if use_aug else 'no'}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(
            model=model, train_loader=tr, val_loader=va,
            device=DEVICE, **kw,
        ).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}  "
                f"loss={m['train_loss']:.4f}  lr={m['lr']:.2e}", flush=True,
            ),
        )
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if key == "Ref_noaug":
            ref_acc = best

        delta_vs_ref = best - ref_acc if ref_acc is not None else None
        dstr = f"{delta_vs_ref*100:+.2f}pp" if delta_vs_ref is not None else "(baseline)"
        gap_vs_linear = best - STEP882_LINEAR

        print(f"  -> best={best:.4f} @ep{best_ep}  delta_vs_ref={dstr}  "
              f"gap_vs_linear={gap_vs_linear*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "desc": desc, "N": N, "K_in": K_in, "use_aug": use_aug, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta_vs_ref, 4) if delta_vs_ref is not None else None,
            "gap_vs_linear": round(gap_vs_linear, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 925 SUMMARY — CIFAR-10 hflip-aug T1")
    print(f"  {'Config':<20} {'N':>6} {'aug':>4} {'best':>7} {'Δ_ref':>8} {'gap_lin':>9}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        gstr = f"{r['gap_vs_linear']*100:+.2f}pp"
        if d is None:
            v = "(baseline)"
        elif d >= 0.005:
            v = "ADVANCE→T2" if d >= 0.005 else "VIABLE"
            if d >= 0.005:
                v = "STRONG→T2" if d >= 0.015 else "ADVANCE→T2"
        elif d >= -0.003:
            v = "NEUTRAL"
        else:
            v = "KILL"
        print(f"  {k:<20} {r['N']:>6} {'y' if r['use_aug'] else 'n':>4} "
              f"{r['best']:>7.4f} {dstr:>8} {gstr:>9}  {v}")

    strong = [k for k, r in results.items()
              if r.get("delta_vs_ref") is not None and r["delta_vs_ref"] >= 0.005]
    if strong:
        best_w = max(strong, key=lambda k: results[k]["delta_vs_ref"])
        d = results[best_w]["delta_vs_ref"]
        N = results[best_w]["N"]
        t2_gap = {2048: STEP882_N2048, 4096: STEP909_N4096, 8192: STEP914_N8192}.get(N, 0.0)
        proj   = t2_gap + STEP882_LINEAR + d  # project T2 = T2_noaug + T1_aug_delta
        print(f"\n  → STRONG: {best_w} ({d*100:+.2f}pp) → T2 on step926")
        print(f"    Imagenette T1→T2 compression ~0.4-0.6×; project T2 gap ≈ "
              f"{(t2_gap + d)*100:+.2f}pp vs Linear (optimistic)")
    else:
        print(f"\n  → No T1 winner ≥+0.5pp. Check T0 — aug likely Imagenette-specific.")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
