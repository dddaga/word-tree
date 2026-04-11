"""Step 211: N=8192 D=16 K_hh=2 K_iter=3 Tier-1 — K_iter minimum at N=8192.

MOTIVATION
==========
K_iter axis at N=8192 T1 reveals: optimal K_iter decreases as N increases.
  N=2048: K_iter=4 KILLED (92.74%), K_iter=5→95.52%, K_iter=6→96.08%  optimal=K6
  N=4096: K_iter=5→97.17% ≈ K_iter=6→97.15%                           optimal≈K5=K6
  N=8192: K_iter=4→95.49%, K_iter=5→95.77%, K_iter=6→96.20% T2 BREAKS optimal=K5

At N=2048, the gap between K_iter=3 and K_iter=4 was:
  K_iter=4: 92.74% (KILLED) | K_iter=3: 89.25% (KILLED, −3.49pp)

The question: does N=8192 rehabilitate K_iter=3 as it did K_iter=4?
  N=2048 K_iter=4: KILLED → N=8192 K_iter=4: 95.49% (massive +2.75pp N-scaling)
  N=2048 K_iter=3: KILLED → N=8192 K_iter=3: ???

If scaling pattern holds: ~89.25% + 2.75pp = ~92% minimum; optimistically ~95%+
If K_iter=3 hits ≥95% at N=8192: new minimum viable K_iter discovered, and
  FLOPs = 2.36M would be the new minimum at N=8192 D=16.

This would be a major finding: K_iter=3 is only viable because N is large enough
to provide sufficient signal flow with minimal routing iterations.

FLOPs = 3×8192×2×16×3 = 2,359,296 ≈ 2.36M — within ≤6.18M budget.
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
D = 16; K_HH = 2; K_IN = 25; K_ITER = 3
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
DATA_FRAC = 0.5

FLOPS = 3 * N * K_HH * D * K_ITER  # 2,359,296 ≈ 2.36M
OUT_PATH = ROOT / "results" / "train_step211_n8192_d16_khh2_kiter3_tier1.json"


def main():
    print(f"\n{'='*70}")
    print(f"Step 211 — N=8192 D=16 K_hh=2 K_iter=3 Tier-1 (50% data 75ep)")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"N=2048 K_iter=3: KILLED (89.25%) | N=2048 K_iter=4: KILLED → N=8192: 95.49%!")
    print(f"Hypothesis: N=8192 rehabilitates K_iter=3. K_iter axis: K4=95.49%, K5=95.77%")
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
            flag = (" *** K_iter=3 VIABLE AT N=8192! ***" if m["val_top1"] >= 0.95 else
                    " *** ABOVE N=2048 K3 FLOOR ***"       if m["val_top1"] >= 0.90 else "")
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
                    "label": "A scratch N=8192 D=16 K_hh=2 K_iter=3 α=1.0 50% 75ep"}}
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    n2048_k3 = 0.8925; n8192_k4 = 0.9549; n8192_k5 = 0.9577
    if best >= 0.95:
        verdict = "✓ K_iter=3 VIABLE — N-scaling rehabilitates minimal routing"
    elif best >= 0.93:
        verdict = f"BORDERLINE — above N=2048 K3 floor but below ≥95%"
    else:
        verdict = f"KILLED — K_iter=3 not viable even at N=8192"
    print(f"\n{'='*70}")
    print(f"STEP 211: {best:.4f}  vs_N2048_K3={best-n2048_k3:+.4f}  vs_N8192_K4={best-n8192_k4:+.4f}  vs_N8192_K5={best-n8192_k5:+.4f}  FLOPs={FLOPS/1e6:.2f}M")
    print(f"best_ep={bep}  params={n_p:,}  {verdict}")
    print(f"→ {OUT_PATH}\n{'='*70}")

if __name__ == "__main__":
    main()
