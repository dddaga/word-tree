"""Step 750: K_iter=3 floor search at N=4096 K_hh=4 (Phase 1 of latency-Pareto track).

GOAL: find smallest N where K_iter=3 reaches ≥95% accuracy. Then scale to find
where K_iter=2 is enough, then K_iter=1. Connectivity scales with user's rule:
K_hh = round(N/1000) → N=4096→K_hh=4 (vs efficiency config N=2048 K_hh=2).

EXISTING CURVE:
  N=2048 K_hh=2 K_iter=3: 89.25% (step202 T1) — KILLED
  N=4096 K_hh=4 K_iter=3: THIS STEP ← gap to fill
  N=8192 K_hh=2 K_iter=3: 94.93% (step211 T1) — borderline (diff K_hh)
  N=8192 K_hh=2 K_iter=4: 95.49% (step210 T1)
  N=8192 K_hh=2 K_iter=5: 97.17% (step209 T2)

HYPOTHESIS: larger N + proportional K_hh compensates for shallower K_iter by
providing richer neighborhoods per hop (more independent paths).

Verdict rules:
  best ≥ 95%: K_iter=3 viable at N=4096 → advance Phase 2 (try K_iter=2 at N=8192)
  best ≥ 93%: borderline → run T2 before advancing
  best < 93%: try next N (step751, N=8192 K_hh=8 K_iter=3)

Inference latency benchmark deferred to end-of-track.

Tier-1 (75ep, 50% data). Device: 5060ti_cuda.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 4; K_IN = 25; K_ITER = 3   # K_iter=3 is the key variable
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step750_kiter3_n4096_khh4.json"


def _build(seed):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def main():
    print(f"\n{'='*70}")
    print(f"Step 750 — K_iter=3 floor search Phase 1")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"K_hh = round(N/1000) = {K_HH} (user rule)")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    model = _build(SEED).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_p:,}")
    t0 = time.time()
    history = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **trainer_kwargs(N, n_epochs=EPOCHS)).train(
        n_epochs=EPOCHS,
        log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                         if (m['epoch'] + 1) % 5 == 0 else None)
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    results = {"best": {"top1_best": best, "best_epoch": bep, "top1_history": top1h,
                        "elapsed_s": round(time.time()-t0, 1), "n_params": n_p,
                        "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER, "flops": FLOPS}}
    print(f"\n{'='*70}")
    print(f"best={best:.4f} @ep{bep}  params={n_p:,}  FLOPs={FLOPS/1e6:.3f}M  elapsed={time.time()-t0:.1f}s")
    if best >= 0.95:
        print(f"  → ≥95% VIABLE: K_iter=3 works at N=4096. Advance Phase 2 (K_iter=2 @ N=8192)")
    elif best >= 0.93:
        print(f"  → BORDERLINE (93-95%): run T2 before Phase 2 decision")
    else:
        print(f"  → <93%: K_iter=3 insufficient at N=4096. Try step751 (N=8192 K_hh=8)")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
