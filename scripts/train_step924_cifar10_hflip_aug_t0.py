"""Step 924: CIFAR-10 hflip augmentation T0 (20ep, 50% data).

MOTIVATION
==========
On Imagenette, hflip augmentation delivers +0.43–0.99pp T2 at every N tested
(steps 269, 273, 276, 280, 287). The mechanism is scale-invariant: aug helps
most at large N (N=16384 +0.99pp) and least at N=2048 (+0.18pp).

CIFAR-10 gap vs Linear:
  N=2048: −5.55pp (step882 T2)
  N=4096: −3.71pp (step909 T2)
  N=8192: −2.66pp (step914 T2)

Open question: can hflip augmentation close the remaining CIFAR-10 gap by
another +0.5–1pp? If so, N=8192+aug could reach ~−1.7pp vs Linear.

PREREQUISITE
============
This script requires data/store_cifar10_aug.h5 (2× train: original + hflip).
If missing, run first:
    python scripts/extract_cifar10_vgg16_features_aug.py --device mps

CONFIGS
=======
  Ref_k25_noaug  : N=2048, K_in=25, no aug  (canonical baseline)
  A_aug_n2048    : N=2048, K_in=25, hflip aug
  B_aug_n4096    : N=4096, K_in=15, hflip aug  (capacity + aug compound)
  C_aug_n8192    : N=8192, K_in=15, hflip aug  (best gap config + aug)

ADVANCE RULE (T0, 20ep, 50% data)
==================================
  +aug delta ≥ +0.3pp → advance config to T1  (aug consistent with Imagenette pattern)
  +aug delta +0.1–+0.3pp → marginal, advance for cheapness (T0 noise band ~0.3pp)
  +aug delta < 0pp → KILL (aug hurts, architectural incompatibility)

Note: step920 T0 reference Ref_k25=75.32%; A_k15=76.03% (+0.71pp, run-order).
Use this run's Ref_k25_noaug as the within-experiment baseline.

NOTES
=====
- T0 scout only: 20ep, 50% of 100K aug training samples = 50K samples
- aug store has 100K train samples (50K orig + 50K flipped); 50% subset = 50K
  which is exactly what the clean 100% store has — T0 is a fair comparison
- Multi-N configs in a single T0 run to identify the best aug config quickly
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
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store_cifar10.h5")
parser.add_argument("--data_aug", default="data/store_cifar10_aug.h5")
parser.add_argument("--configs",  default="Ref_k25_noaug,A_aug_n2048,B_aug_n4096,C_aug_n8192")
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
OUT_PATH = ROOT / "results" / f"train_step924_cifar10_hflip_aug_t0_seed{SEED}__{SLOT}.json"

# Anchors from prior steps
STEP882_LINEAR = 0.8624   # Linear 86.24% (T2, 150ep)
STEP882_N2048  = 0.8069   # N=2048 T2 baseline
STEP909_N4096  = 0.8253   # N=4096 T2 baseline
STEP914_N8192  = 0.8358   # N=8192 T2 baseline

# (N, K_in, use_aug, description)
CONFIGS = {
    "Ref_k25_noaug": (2048, 25, False, "N=2048 K_in=25 no-aug — canonical T0 reference"),
    "A_aug_n2048":   (2048, 25, True,  "N=2048 K_in=25 hflip-aug — aug isolation at default N"),
    "B_aug_n4096":   (4096, 15, True,  "N=4096 K_in=15 hflip-aug — capacity+aug compound"),
    "C_aug_n8192":   (8192, 15, True,  "N=8192 K_in=15 hflip-aug — best-gap config + aug"),
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
    """50% subset of training data for T0 scout."""
    tr_full, _ = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    n_full  = len(tr_full.dataset)
    n_sub   = int(n_full * fraction)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_sub]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10, pin_memory=False,
    )


def main():
    # Check data files
    clean_path = ROOT / args.data
    aug_path   = ROOT / args.data_aug
    if not clean_path.exists():
        print(f"ERROR: {clean_path} not found."); sys.exit(1)
    if not aug_path.exists():
        print(f"ERROR: {aug_path} not found.")
        print(f"  Run first: python scripts/extract_cifar10_vgg16_features_aug.py --device mps")
        sys.exit(1)

    # Pre-build loaders (50% subsets for T0)
    print("Building data loaders...")
    _, va          = make_loaders(str(clean_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    tr_clean_sub   = make_subset_loader(str(clean_path), fraction=0.5)
    tr_aug_sub     = make_subset_loader(str(aug_path),   fraction=0.5)

    print(f"\n{'='*70}")
    print(f"step924 — CIFAR-10 hflip-aug T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Anchors: N2048={STEP882_N2048:.4f} N4096={STEP909_N4096:.4f} "
          f"N8192={STEP914_N8192:.4f} Linear={STEP882_LINEAR:.4f}")
    print(f"  Imagenette aug T2 deltas: N2048=+0.18pp N4096=+0.56pp N8192=+0.43pp N16384=+0.99pp")
    print(f"  Advance: +aug delta ≥ +0.3pp → T1; <0pp → KILL")
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
                f"loss={m['train_loss']:.4f}", flush=True,
            ) if (m['epoch'] + 1) % 5 == 0 else None,
        )
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        if key == "Ref_k25_noaug":
            ref_acc = best

        delta_vs_ref = best - ref_acc if ref_acc is not None else None
        dstr = f"{delta_vs_ref*100:+.2f}pp" if delta_vs_ref is not None else "(baseline)"

        # Compare vs T2 anchor for same N
        t2_anchor = {2048: STEP882_N2048, 4096: STEP909_N4096, 8192: STEP914_N8192}.get(N)
        gap_vs_linear = best - STEP882_LINEAR

        print(f"  -> best={best:.4f} @ep{best_ep}  delta_vs_ref={dstr}  "
              f"gap_vs_linear={gap_vs_linear*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "desc": desc, "N": N, "K_in": K_in, "use_aug": use_aug, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta_vs_ref, 4) if delta_vs_ref is not None else None,
            "gap_vs_linear": round(gap_vs_linear, 4),
            "t2_anchor": round(t2_anchor, 4) if t2_anchor else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print(f"STEP 924 SUMMARY — CIFAR-10 hflip-aug T0")
    print(f"  {'Config':<20} {'N':>6} {'aug':>4} {'best':>7} {'Δ_ref':>8} {'gap_lin':>9}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        gstr = f"{r['gap_vs_linear']*100:+.2f}pp"
        if d is None:
            v = "(baseline)"
        elif d >= 0.003:
            v = "ADVANCE→T1" if d >= 0.003 else "MARGINAL"
            if d >= 0.003:
                v = "STRONG→T1" if d >= 0.010 else "ADVANCE→T1"
        elif d >= -0.003:
            v = "NEUTRAL"
        else:
            v = "KILL"
        print(f"  {k:<20} {r['N']:>6} {'y' if r['use_aug'] else 'n':>4} "
              f"{r['best']:>7.4f} {dstr:>8} {gstr:>9}  {v}")

    winners = [k for k, r in results.items()
               if r.get("delta_vs_ref") is not None and r["delta_vs_ref"] >= 0.003]
    if winners:
        best_w = max(winners, key=lambda k: results[k]["delta_vs_ref"])
        d = results[best_w]["delta_vs_ref"]
        N = results[best_w]["N"]
        print(f"\n  → ADVANCE: {best_w} ({d*100:+.2f}pp) → T1 on step925")
        print(f"    Project T2 aug delta ~{d*100*0.6:+.2f}pp (60% T0→T2 compression typical).")
        if N == 8192:
            proj_t2 = STEP914_N8192 + d * 0.6
            print(f"    Projected N=8192+aug T2 ≈ {proj_t2:.4f}  "
                  f"gap_vs_linear ≈ {(proj_t2 - STEP882_LINEAR)*100:+.2f}pp")
    else:
        print(f"\n  → KILL: no aug config advances. CIFAR-10 gap is architectural, not data-limited.")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
