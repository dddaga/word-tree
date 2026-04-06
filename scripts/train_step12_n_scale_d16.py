"""Step 12: Neuron count (N) scaling sweep at D=16 Fourier dynamic_z_geo.

Step 9 established D=16 Fourier N=512 = 29.22% as the new best.
With D=16 confirmed as the right dimensionality, does scaling N further help?

Previous D=8 findings: N=512→1024 gave +1.02% (marginal at D=8).
At D=16, S¹⁵ has vast capacity — many more distinct directions available.
N=1024 at D=16 may benefit more from scale since neurons don't crowd the sphere.

Classic deep learning insight: learnability scales POLYNOMIALLY with parameter count
(width). Compare: N=512→1024→2048 at D=16 — does accuracy scale polynomially or
does it plateau? The plateau point reveals when routing bottleneck overrides capacity.

Configs (D=16 Fourier dynamic_z_geo full routing, 120ep plateau):
  A. N=512   D=16  dynamic_z_geo  (step9A reference at 120ep)
  B. N=1024  D=16  dynamic_z_geo  (2× neurons — 4× routing cost per step)
  C. N=2048  D=16  dynamic_z_geo  (4× neurons — 16× routing cost, 120ep)

Note: step9A used 150ep; here we use 120ep for fair comparison across N.
      N=2048 training cost ~16× that of N=512 per epoch.

To reproduce:
    python -u scripts/train_step12_n_scale_d16.py --device mps
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.trainer        import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  encoding=fourier")
print(f"Goal: N scale sweep at D=16 — polynomial width scaling test")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(n_hidden: int) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(n_hidden)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=n_hidden, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=tk["K_iter"],
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )


def run(label: str, model: nn.Module, n_hidden: int) -> dict:
    print(f"\n{'='*65}\n{label}\n{'='*65}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(n_hidden, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
        "n_hidden":        n_hidden,
        "top1_best":       best,
        "top1_last":       history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch":      int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1,
        "epochs_run":      len(history),
        "elapsed_s":       round(elapsed, 1),
        "top1_history":    [round(h.get("val_top1", 0.0), 4) for h in history],
    }
    print(f"  top1_best={best:.4f}  best_ep={result['best_epoch']}  "
          f"task={result['final_task_loss']:.3f}  t={elapsed:.0f}s")
    return result


CONFIGS = [
    ("A. N=512   D=16  dynamic_z_geo  120ep  [step9A ref at 120ep]",  512),
    ("B. N=1024  D=16  dynamic_z_geo  120ep  [2x neurons]",          1024),
    ("C. N=2048  D=16  dynamic_z_geo  120ep  [4x neurons]",          2048),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    results = {}
    for key, (label, n_hidden) in zip("ABC", CONFIGS):
        model = make_model(n_hidden).to(DEVICE)
        results[key] = run(label, model, n_hidden)

    out = Path("results/train_step12_n_scale_d16.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = results["A"]["top1_best"]
    print("\n-- N scale at D=16 Fourier (polynomial width scaling) --------------------")
    print("  %-55s  %9s  %+8s  %6s" % ("Config", "top1_best", "vs_N512", "t(s)"))
    print("  " + "-"*55 + "  " + "-"*9 + "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - ref
        print("  %-55s  %9.4f  %+8.4f  %6.0f" % (
            k, r["top1_best"], delta, r["elapsed_s"]))

    # Polynomial scaling check: if accuracy ~ N^alpha, then
    # delta(N512→1024) / delta(N512→2048) should approach 0.5 (linear) or 1.0 (none)
    if len(results) == 3:
        d_1024 = results["B"]["top1_best"] - results["A"]["top1_best"]
        d_2048 = results["C"]["top1_best"] - results["A"]["top1_best"]
        print(f"\n  N=512→1024 gain: {d_1024:+.4f}")
        print(f"  N=512→2048 gain: {d_2048:+.4f}")
        if abs(d_2048) > 0.001:
            ratio = d_1024 / d_2048 if abs(d_2048) > 0.001 else float('inf')
            print(f"  Gain ratio (1024/2048): {ratio:.2f}  (linear=0.5, plateau>1.0)")
        print(f"\n  step9A full 150ep ref: 0.2922")
