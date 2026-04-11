"""Step 210: N=8192 D=16 K_hh=2 K_iter=4 Tier-1 — K_iter scaling at N=8192.

MOTIVATION
==========
N=8192 K_iter axis results so far:
  K_iter=6 T2 (step207): 96.20%@ep54 — N-scaling BREAKS (−0.95pp vs N=4096 T2)
  K_iter=5 T1 (step208): 95.77%@ep51 — beats K_iter=6 T1 by +0.66pp
  K_iter=5 T2 (step209): ep60=95.97% (running, outpacing step207)

Pattern: optimal K_iter DECREASES as N increases.
  N=2048: K_iter=6 optimal (96.08% T2)
  N=4096: K_iter=5 ≈ K_iter=6 (97.17% vs 97.15% T2)
  N=8192: K_iter=5 > K_iter=6 by ~0.66pp T1, larger gap at T2

Hypothesis: at large N with fixed D=16, too many routing steps causes
over-smoothing (representations collapse toward mean-field). Fewer steps
preserve representational diversity needed for classification.

If this hypothesis holds: K_iter=4 may outperform K_iter=5 at N=8192.
  N=2048 K_iter=4 (step196): KILLED at 92.74% — too few steps.
  N=8192 K_iter=4: untested. 4× larger N may compensate for fewer steps.

FLOPs = 3×8192×2×16×4 = 3,145,728 ≈ 3.15M — within ≤6.18M budget.
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

EPOCHS = args.epochs; BATCH = 64; SEED = 42; DATA = "data/store.h5"
N = 8192; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 4
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 3,145,728 ≈ 3.15M
OUT_PATH = ROOT / "results" / "train_step210_n8192_d16_khh2_kiter4_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 210 — N=8192 D=16 K_hh=2 K_iter=4 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"K_iter axis at N=8192: K6→96.20%, K5→95.77% T1 (+0.66pp over K6)")
    print(f"Hypothesis: optimal K_iter decreases as N grows. K_iter=4 probe.")
    print(f"Ref: N=2048 K_iter=4 KILLED (92.74%) — N=8192 may compensate.")
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
            flag = (" *** BEATS K5 T1! K_iter=4 WINS ***" if m["val_top1"] >= 0.9578 else
                    " *** PHASE EXIT ***"                   if m["val_top1"] >= 0.95 else "")
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
                    "label": "A scratch N=8192 D=16 K_hh=2 K_iter=4 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n8192_k5_t1 = 0.9577; n8192_k6_t1 = 0.9511
    verdict = ("✓ K_iter=4 BEATS K5 — advance T2" if best >= n8192_k5_t1 else
               f"KILLED: K_iter=5 T1 still better ({best-n8192_k5_t1:+.4f}pp)" if best < 0.93 else
               f"BORDERLINE ({best-n8192_k5_t1:+.4f}pp vs K5 T1)")
    print(f"\n{'='*70}")
    print(f"STEP 210: {best:.4f}  vs_K5_T1={best-n8192_k5_t1:+.4f}  vs_K6_T1={best-n8192_k6_t1:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  {verdict}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
