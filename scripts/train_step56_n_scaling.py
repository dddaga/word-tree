"""Step 56: Neuron scaling sweep at D=64 with AntiHebb alpha=0.7 wpos.

QUESTION
========
Best config from step29: N=1024, D=64, K_iter=8, AntiHebb alpha=0.7 wpos = 75.24%.
Does increasing N beyond 1024 improve accuracy, or does it plateau?

Sweep: N = [512, 1024, 2048, 4096, 10000]
  - N=512:  baseline (smaller than best)
  - N=1024: step29 reference point (expect ~75%)
  - N=2048: 2x scale (does accuracy keep rising?)
  - N=4096: 4x scale (diminishing returns?)
  - N=10000: max scale (OOM test + ceiling)

Key constraints:
  - N > 5000: safety_valve_loss disabled (cdist OOM guard)
  - n_groups capped at min(128, N//8) for stable topology
  - Each N runs sequentially (large N needs full GPU memory)

Output: results/train_step56_n_scaling.json with per-N metrics
"""

import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.experiment_config import (
    trainer_kwargs, topology_kwargs, run_metadata, GA_BEST_D64,
)
from src.training.trainer            import Trainer
from src.training.dataset            import make_loaders
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian


# -- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
DEVICE = parser.parse_args().device

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 64
K_ITER = 8
ANTIHEBB_ALPHA = 0.7

N_VALUES = [512, 1024, 2048, 4096, 10000]

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# -- Model factory -----------------------------------------------------------
def make_model(N: int) -> nn.Module:
    """Build SmallWorld + Resonant + AntiHebb for given N_hidden."""
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=min(128, max(8, N // 8)),
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ANTIHEBB_ALPHA, variant="wpos")
    return model


# -- Run helper --------------------------------------------------------------
def run(N: int) -> dict:
    label = f"N={N:>5d}  D={D}  K_iter={K_ITER}  AntiHebb a={ANTIHEBB_ALPHA}"
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    model = make_model(N).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)

    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    wall_min = elapsed / 60.0

    best_top1 = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    final_loss = history[-1]["train_loss"]

    print(f"  RESULT: top1={best_top1:.4f}  best_ep={best_ep}/{EPOCHS}"
          f"  params={n_params:,}  time={wall_min:.1f}min")

    return {
        "N_hidden": N,
        "D": D,
        "K_iter": K_ITER,
        "antihebb_alpha": ANTIHEBB_ALPHA,
        "top1_best": best_top1,
        "best_epoch": best_ep,
        "final_train_loss": final_loss,
        "params": n_params,
        "wall_time_minutes": round(wall_min, 1),
        "epochs_run": len(history),
        "stopped_early": history[-1].get("stopped_early", False),
        "nan_detected": history[-1].get("nan_detected", False),
        "_meta": run_metadata(__file__, {
            "N": N, "D": D, "K_iter": K_ITER,
            "antihebb_alpha": ANTIHEBB_ALPHA, "epochs": EPOCHS,
        }),
    }


# -- Main --------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Step 56: N-scaling sweep at D={D} K_iter={K_ITER} AntiHebb={ANTIHEBB_ALPHA}")
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print(f"N values: {N_VALUES}")
    print(f"Reference: step29 Config C (N=1024) = 75.24%")

    results = []
    for N in N_VALUES:
        result = run(N)
        results.append(result)

        # Save incrementally (so partial results survive crashes)
        out_path = ROOT / "results" / "train_step56_n_scaling.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)}/{len(N_VALUES)} results -> {out_path}")

    # -- Summary table --------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"N-Scaling Summary (D={D}, AntiHebb a={ANTIHEBB_ALPHA})")
    print(f"{'='*70}")
    print(f"  {'N':>6}  {'top1':>7}  {'params':>10}  {'time(min)':>10}  {'best_ep':>8}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['N_hidden']:>6}  {r['top1_best']:>7.4f}"
              f"  {r['params']:>10,}  {r['wall_time_minutes']:>10.1f}"
              f"  {r['best_epoch']:>8}")

    ref_1024 = next((r for r in results if r["N_hidden"] == 1024), None)
    if ref_1024:
        print(f"\n  N=1024 reference: {ref_1024['top1_best']:.4f}"
              f"  (step29 was 75.24%)")
        for r in results:
            if r["N_hidden"] != 1024:
                delta = r["top1_best"] - ref_1024["top1_best"]
                print(f"  N={r['N_hidden']:>5} vs N=1024: {delta:+.4f}")

    print(f"\nAll results saved to {out_path}")
