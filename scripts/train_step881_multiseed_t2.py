"""Step 881: ΔW-proj T2 multi-seed Imagenette — paper error bars.

MOTIVATION
==========
step760 T0 showed ΔW-proj seed variance ±0.20pp (3 seeds at 20ep/50%).
For the paper, we need T2 (150ep, 100%) multi-seed error bars.
Seeds 0, 1, 42 → mean ± std for the main accuracy result.

Prior T2 single-seed results (step199/step871/step291):
  step199 base: ~94.5-95.5% (varies by aug, ΔW)
  step871 D_very ΔW T1: 0.9607 (T1 best so far at 75ep/50%)
  step291 D=16: 97.30% T2 (with aug)
  ΔW-proj T2 reference (no aug): ~95.52% expected

This run establishes mean ± std for ΔW-proj T2 (no aug, no distillation).

NOTE: This runs on studio_mps. The studio model_resonant.py has
67,744 params (vs canonical 34,976). Absolute numbers are ~1pp higher.
Use this result for variance estimation only; canonical T2 numbers
must come from 5060ti CUDA path.

CONFIGS (T2: 150ep, 100% data, seeds=[0,1,42])
  Ref_dw : K_hh=2 (K_local=1, K_random=1) — standard ΔW-proj
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seeds",   default="0,1,42")
parser.add_argument("--data",    default="data/store.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step881_multiseed_t2__{SLOT}.json"

SEEDS = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]


class SGNNET_DeltaW(nn.Module):
    """ΔW-proj routing — standard paper model."""
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW(resonant)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=42,
                          pin_memory=False)

    n_p_check = sum(p.numel() for p in make_model(42).parameters() if p.requires_grad)
    canonical = n_p_check == 34976
    param_note = "(CANONICAL)" if canonical else f"(NON-CANONICAL: {n_p_check:,} params — studio path)"

    print(f"\n{'='*70}")
    print(f"step881 — ΔW-proj T2 multi-seed (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seeds={SEEDS}")
    print(f"  params={n_p_check:,} {param_note}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    results = {}

    for seed in SEEDS:
        model = make_model(seed)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\nseed={seed}  params={n_p:,}")

        # Reseed data loader for each seed
        tr_seeded, va_seeded = make_loaders(str(data_path), batch_size=BATCH,
                                            seed=seed, pin_memory=False)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr_seeded, val_loader=va_seeded,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        print(f"  -> best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")

        results[f"seed{seed}"] = {
            "seed": seed, "n_params": n_p, "canonical": canonical,
            "best": best, "best_ep": best_ep, "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    bests = [r["best"] for r in results.values()]
    mean_acc = np.mean(bests)
    std_acc = np.std(bests)

    print(f"\n{'='*70}")
    print(f"STEP 881 SUMMARY — ΔW-proj T2 multi-seed")
    print(f"{'='*70}")
    print(f"  params: {n_p_check:,} {param_note}")
    for k, r in results.items():
        print(f"  {k}: {r['best']:.4f} @ep{r['best_ep']}")
    print(f"\n  mean={mean_acc:.4f}  std={std_acc*100:.2f}pp  min={min(bests):.4f}  max={max(bests):.4f}")
    if canonical:
        print(f"  Paper claim: {mean_acc:.4f} ± {std_acc*100:.2f}pp (CANONICAL)")
    else:
        print(f"  Variance estimate only — rerun on CUDA for paper canonical numbers")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
