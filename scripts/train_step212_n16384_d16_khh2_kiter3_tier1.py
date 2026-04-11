"""Step 212: N=16384 D=16 K_hh=2 K_iter=3 Tier-1 — N-scaling at K_iter floor.

MOTIVATION
==========
K_iter=3 at N=8192 (step211): 94.93%@ep72 — just below 95% threshold (−0.07pp).
Compared to N=2048 K_iter=3 (step202): 89.25% → N-scaling gain = +5.68pp!

N-scaling has consistently lifted K_iter floors:
  N=2048 K_iter=4: KILLED (92.74%) → N=8192 K_iter=4: 95.49% (+2.75pp, viable!)
  N=2048 K_iter=3: KILLED (89.25%) → N=8192 K_iter=3: 94.93% (+5.68pp, borderline)
  N=16384 K_iter=3: ???             → expected ~96%+ if scaling continues

If K_iter=3 clears 95% at N=16384, FLOPs = 4.72M — same as step206 (N=8192 K_iter=6).
This would demonstrate that N-scaling can maintain accuracy while reducing K_iter.

FLOPs = 3×16384×2×16×3 = 4,718,592 ≈ 4.72M — within ≤6.18M budget.

Note: N=16384 uses BATCH=32 (reduced from 64) to manage memory.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 32; SEED = 42; DATA = "data/store.h5"
N = 16384; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 3
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 4,718,592 ≈ 4.72M
OUT_PATH = ROOT / "results" / "train_step212_n16384_d16_khh2_kiter3_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 212 — N=16384 D=16 K_hh=2 K_iter=3 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"N=8192 K_iter=3 (step211): 94.93% — just below 95% (−0.07pp)")
    print(f"N-scaling effect on K_iter=3: +5.68pp per N-doubling from N=2048→N=8192")
    print(f"N=16384 projection: 94.93% + N-scaling gain = ~96%+ if pattern holds")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    print(f"  K_local={K_l}  K_random={K_r}  n_groups={ng}  batch={BATCH}")
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB,
                                variant="wpos").to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:int(n * DATA_FRAC)]
    sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0)

    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps": torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            flag = (" *** K3 VIABLE AT N=16384! PHASE EXIT ***" if m["val_top1"] >= 0.95 else
                    " *** ABOVE N=8192 K3 FLOOR ***"             if m["val_top1"] >= 0.9493 else "")
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{flag}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    result = {"A": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                    "alpha_ahebb": ALPHA_AHEBB, "warm": False, "data_frac": DATA_FRAC,
                    "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                    "epochs_run": len(history), "top1_history": top1h,
                    "elapsed_s": round(elapsed, 1), "n_params": n_p, "flops": FLOPS,
                    "label": "A scratch N=16384 D=16 K_hh=2 K_iter=3 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n8192_k3 = 0.9493; n8192_k5 = 0.9577
    if best >= 0.95:
        verdict = "✓ K_iter=3 VIABLE AT N=16384 — phase exit, N-scaling law confirmed"
    elif best > n8192_k3:
        verdict = f"ABOVE N=8192 FLOOR (+{best-n8192_k3:.4f}pp) but below 95%"
    else:
        verdict = "FLAT/REGRESSION — N-scaling stalled for K_iter=3"
    print(f"\n{'='*70}")
    print(f"STEP 212: {best:.4f}  vs_N8192_K3={best-n8192_k3:+.4f}  vs_N8192_K5={best-n8192_k5:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  {verdict}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
