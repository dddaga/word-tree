"""Step 13: Beam size × K_iter depth/width scaling study at D=16 Fourier.

THEORETICAL MOTIVATION
======================
Classic deep learning scaling laws:
  - Width (N neurons): accuracy gains POLYNOMIAL in parameter count
  - Depth (layers): accuracy gains EXPONENTIAL in depth (more expressive per parameter)

In SGNNET routing, the analogous quantities are:
  - beam_size: how many top-scored Z-vectors are selected per routing step
               → controls "width" of information propagation
  - K_iter:    how many routing iterations (recursive forward passes)
               → controls "depth" of propagation (graph distance reachable)

At K_iter=k: each neuron integrates signal from neurons within k hops in the graph.
With a small-world graph (diameter ~log N ≈ 9 for N=512), K_iter≥9 reaches all neurons.
Beam truncation controls HOW MANY of those neighbors contribute at each step.

ATTRIBUTE ACCUMULATION
======================
Each routing step: Z[k+1][h] = normalize(sum_{j in top-beam(h)} score[h,j] * Z[k][j])
The score is the dot-product (cosine similarity) of geometric positions W_pos[h,j].
At step 0: Z encodes the input seed (feature from VGG pool5).
At step 1: Z holds 1-hop neighbourhood average — local integration.
At step 2: Z holds 2-hop average — locality grows. Input signal dilutes.
At step K: Z has diffused K hops. Information "spreads" like heat on the graph.

HYPOTHESES TO TEST
==================
H1: K_iter=3 (default) may underfit — more depth should help (exponential gain theory).
H2: Large beam helps more when K_iter is large (more relevant neighbours to beam).
H3: Very large K_iter causes over-smoothing (all Z collapse to same vector).
H4: Beam=unlimited (all N neighbours, no truncation) removes selection bottleneck.

Experiment design: 2D sweep of (beam_size × K_iter)
  Row 1 — beam sweep at K_iter=3 (current default):
    A. beam=8    K_iter=3   (narrow selection, 3 hops)
    B. beam=16   K_iter=3
    C. beam=32   K_iter=3   (reference config)
    D. beam=64   K_iter=3
    E. beam=128  K_iter=3   (wide selection)

  Row 2 — depth sweep at beam=32 (current default):
    F. beam=32   K_iter=1   (single-hop — no propagation)
    G. beam=32   K_iter=2
    H. beam=32   K_iter=5   (deeper)
    I. beam=32   K_iter=8   (much deeper — reaches most nodes in small-world)
    J. beam=32   K_iter=12  (over-smoothing test)

  Row 3 — depth × beam interaction:
    K. beam=8    K_iter=8   (narrow but deep)
    L. beam=128  K_iter=8   (wide and deep)

All: D=16 Fourier N=512 dynamic_z_geo, 120ep, plateau, store.h5, MPS.
Reference: step9A beam=32 K_iter=3 = 29.22% (150ep), 120ep expected ~28-29%.

Note: K_iter=12 + beam=128 is computationally heavier. Timing will reveal MPS limits.

To reproduce:
    python -u scripts/train_step13_beam_iter_scaling.py --device mps
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
# Optional: run only one row to parallelise across multiple sessions
parser.add_argument("--row", type=str, default="all",
                    choices=["all", "beam", "depth", "interaction"],
                    help="Which row to run: all | beam | depth | interaction")
args = parser.parse_args()

DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 120
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16

print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  encoding=fourier")
print(f"Goal: beam_size × K_iter scaling study (depth vs width in routing)")
print(f"Row: {args.row}")

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_model(beam_size: int, k_iter: int) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(512)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=512, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=k_iter,       # <-- depth
        n_groups=tk["n_groups"],
        norm_mode="l2",
        D=D,
        encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=beam_size,  # <-- width
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo",
        resonance_threshold=0.0,
        geo_gamma=1.0,
    )


def run(label: str, model: nn.Module) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr_loader, va_loader = get_loaders()
    tk = trainer_kwargs(512, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr_loader, val_loader=va_loader,
                      device=DEVICE, **tk)
    t0 = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best  = max(h.get("val_top1", 0.0) for h in history)
    last5 = history[-5:]
    result = {
        "label":           label,
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


# Row 1: beam sweep (width) at fixed K_iter=3
BEAM_CONFIGS = [
    ("A. beam=8    K_iter=3   D=16 geo  [narrow width]",   8,  3),
    ("B. beam=16   K_iter=3   D=16 geo",                  16,  3),
    ("C. beam=32   K_iter=3   D=16 geo  [reference]",     32,  3),
    ("D. beam=64   K_iter=3   D=16 geo",                  64,  3),
    ("E. beam=128  K_iter=3   D=16 geo  [wide width]",   128,  3),
]

# Row 2: depth sweep (K_iter) at fixed beam=32
DEPTH_CONFIGS = [
    ("F. beam=32   K_iter=1   D=16 geo  [single-hop]",    32,  1),
    ("G. beam=32   K_iter=2   D=16 geo",                  32,  2),
    # C is reference (K_iter=3, run in beam configs)
    ("H. beam=32   K_iter=5   D=16 geo  [deeper]",        32,  5),
    ("I. beam=32   K_iter=8   D=16 geo  [reaches 8-hop]", 32,  8),
    ("J. beam=32   K_iter=12  D=16 geo  [over-smooth?]",  32, 12),
]

# Row 3: depth × beam interaction
INTERACTION_CONFIGS = [
    ("K. beam=8    K_iter=8   D=16 geo  [narrow+deep]",   8,  8),
    ("L. beam=128  K_iter=8   D=16 geo  [wide+deep]",   128,  8),
]

if __name__ == "__main__":
    get_loaders()
    n_train = len(_loaders[0].dataset)
    n_val   = len(_loaders[1].dataset)
    print(f"Dataset: train={n_train}  val={n_val}  (in-memory, {DATA})")

    # Select which rows to run
    if args.row == "beam":
        all_configs = list(zip("ABCDE", BEAM_CONFIGS))
    elif args.row == "depth":
        all_configs = list(zip("FGHIJ", DEPTH_CONFIGS))
    elif args.row == "interaction":
        all_configs = list(zip("KL", INTERACTION_CONFIGS))
    else:  # all
        all_configs = (list(zip("ABCDE", BEAM_CONFIGS)) +
                       list(zip("FGHIJ", DEPTH_CONFIGS)) +
                       list(zip("KL",    INTERACTION_CONFIGS)))

    results = {}
    for key, (label, beam, kiter) in all_configs:
        model = make_model(beam, kiter).to(DEVICE)
        results[key] = run(label, model)
        results[key]["beam_size"] = beam
        results[key]["k_iter"]    = kiter

    suffix = f"_{args.row}" if args.row != "all" else ""
    out = Path(f"results/train_step13_beam_iter_scaling{suffix}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref = 0.2922  # step9A 150ep
    print(f"\n-- Beam × K_iter scaling at D=16 Fourier (ref: step9A={ref:.4f}) ------")
    print("  %-55s  %5s  %6s  %9s  %+8s  %6s" %
          ("Config", "beam", "K_iter", "top1_best", "vs_ref", "t(s)"))
    print("  " + "-"*55 + "  " + "-"*5 + "  " + "-"*6 + "  " + "-"*9 +
          "  " + "-"*8 + "  " + "-"*6)
    for k, r in results.items():
        delta = r["top1_best"] - ref
        print("  %-55s  %5d  %6d  %9.4f  %+8.4f  %6.0f" % (
            k, r["beam_size"], r["k_iter"],
            r["top1_best"], delta, r["elapsed_s"]))

    # Summary analysis
    if "C" in results:
        print(f"\n  === Beam sweep (K_iter=3) ===")
        for k in "ABCDE":
            if k in results:
                r = results[k]
                print(f"  beam={r['beam_size']:3d}: {r['top1_best']:.4f}")

    if "F" in results or "H" in results:
        print(f"\n  === Depth sweep (beam=32) ===")
        # Include C as reference
        if "C" in results:
            print(f"  K_iter={results['C']['k_iter']:2d} (ref): {results['C']['top1_best']:.4f}")
        for k in "FGHIJ":
            if k in results:
                r = results[k]
                print(f"  K_iter={r['k_iter']:2d}: {r['top1_best']:.4f}")

    if "K" in results or "L" in results:
        print(f"\n  === Beam×depth interaction (K_iter=8) ===")
        for k in "KL":
            if k in results:
                r = results[k]
                print(f"  beam={r['beam_size']:3d} K_iter={r['k_iter']}: {r['top1_best']:.4f}")
