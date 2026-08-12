"""Step 982: CIFAR-10 augmentation T1 (75ep, 50% data).

MOTIVATION
==========
step929 crashed (OOM loading 2 h5 files simultaneously: clean + aug = ~16GB).
This is the fixed version: trains configs SEQUENTIALLY, releases RAM between
each config. Uses separate h5 files for Ref (clean) and Aug (augmented).

CIFAR-10 aug is a paper-scope experiment: Imagenette augmentation gave +0.18
to +0.79pp T2 across all N (step269-282). Does the pattern hold on CIFAR-10?

CONFIGS
=======
  Ref:    store_cifar10.h5     (50K train), 50% → 25K samples
  A_aug:  store_cifar10_aug.h5 (100K train = 50K orig + 50K hflip), 50% → 50K samples

ADVANCE RULE
============
  ≥+0.5pp → T2 (CIFAR-10 aug publishable)
  +0.0–0.5pp → NEUTRAL (aug less effective on CIFAR-10 than Imagenette)
  <0.0pp → KILL

Usage:
    python scripts/train_step982_cifar10_aug_t1.py --device cuda --configs Ref,A_aug
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
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
from src.training.dataset           import make_loaders, make_subset_loader

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--configs", default="Ref,A_aug")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step982_cifar10_aug_t1_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref":   str(ROOT / "data" / "store_cifar10.h5"),
    "A_aug": str(ROOT / "data" / "store_cifar10_aug.h5"),
}


def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_model() -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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
        def W_pos(self): return self.m.W_pos
        @property
        def W_phase(self): return getattr(self.m, "W_phase", None)
        def tick_epoch(self):
            if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

        def forward(self, x):
            Z         = self.m.base._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = self.m.base.conn_hh
            dw        = _dw_proj(self.m.W_pos, conn_hh)
            Z_ref     = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                Z_agg = _dw_agg(Z_nb, dw)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return self.m.base._readout(Z)

    return DeltaW()


def main():
    print(f"\n{'='*70}")
    print(f"step982 — CIFAR-10 augmentation T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_in={K_IN} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  OOM fix: trains configs SEQUENTIALLY, releases RAM between each")
    print(f"  Ref: clean 50K train → 25K @ 50%")
    print(f"  A_aug: aug 100K train → 50K @ 50%")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None

    for key in keys:
        h5_path = CONFIGS[key]
        if not Path(h5_path).exists():
            print(f"ERROR: {h5_path} not found."); sys.exit(1)

        tr = make_subset_loader(h5_path, fraction=0.5, batch_size=BATCH,
                                seed=SEED, pin_memory=False)
        _, va = make_loaders(h5_path, batch_size=BATCH, seed=SEED,
                             pin_memory=False)
        n_train = len(tr.dataset)

        model = make_model()
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: h5={Path(h5_path).name}  train_samples={n_train}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        history = Trainer(
            model=model, train_loader=tr, val_loader=va,
            device=DEVICE, **kw,
        ).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
            ),
        )

        elapsed = time.time() - t0
        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                   for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1

        delta = (best - ref_acc) if ref_acc is not None else 0.0
        if ref_acc is None:
            ref_acc = best

        results[key] = {
            "h5_file": Path(h5_path).name,
            "train_samples": n_train,
            "n_params": n_p,
            "best": round(best, 4),
            "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4),
            "elapsed_s": round(elapsed, 1),
        }

        verdict = ("REF" if key == "Ref" else
                   "ADVANCE" if delta >= 0.005 else
                   "NEUTRAL" if delta >= 0.0 else "KILL")
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta:+.4f}  [{verdict}]")
        print(f"  → elapsed={elapsed:.0f}s")

        # Release RAM before next config
        del model, tr, va, history, top1h
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {OUT_PATH}")

    if len(results) >= 2:
        ref_b  = results["Ref"]["best"]
        aug_b  = results["A_aug"]["best"]
        delta  = aug_b - ref_b
        print(f"\nSUMMARY: Ref={ref_b:.4f}  A_aug={aug_b:.4f}  Δ={delta:+.4f}")
        if delta >= 0.005:
            print(f"  → ADVANCE to T2 (+{delta:.2%})")
        elif delta >= 0.0:
            print(f"  → NEUTRAL (aug less effective on CIFAR-10)")
        else:
            print(f"  → KILL (aug hurts CIFAR-10)")


if __name__ == "__main__":
    main()
