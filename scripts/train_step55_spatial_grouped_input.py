"""Step 55: Spatial grouped input — learned per-group projections with bridge neurons.

Current input mechanism
-----------------------
N_in=25088 → N=1024 via sparse K_in=50 random gather-sum per neuron.
n_groups=128 groups; each group exclusively samples from ~196-feature input slice.
Fixed weights (all 1.0); no learned projection; full random global coverage.

New mechanism
-------------
Two neuron populations:

  Exclusive (1-overlap_frac of N):
    Small learned linear per spatial group (G_in → ng scalars).
    Group g sees ONLY its input slice — local feature specialisation.
    Z = val × W_pos_norm (scalar modulates learned geometric direction).

  Bridge (overlap_frac of N):
    Random K_in=50 global sparse gather (no learned params).
    Full N_in coverage — provides cross-group information mixing.
    Same mechanism as current Ref baseline.

Hypotheses
----------
  H1: Any learned proj > sparse K_in=50 → learning which features matter
      is more valuable than random uniform aggregation.
  H2: Adding bridge neurons > pure exclusive → cross-group info improves routing.
  H3: VGG16 spatial rows > flat groups → explicit spatial structure helps.
  H4: Position-level > row-level → finer spatial resolution is better.

Configs (all: D=64, N=1024, K_iter=8, AntiHebb α=0.7 wpos — step29 best = 75.24%)
----------------------------------------------------------------------------------
  Ref   K_in=50 sparse gather  n_groups=128 flat  NO proj   [current best base]
  A     7 spatial rows  separate proj  overlap=0%   ~3.67M proj params
  B     7 spatial rows  separate proj  overlap=20%  ~3.0M proj params  + 205 bridge
  C     7 spatial rows  separate proj  overlap=40%  ~2.2M proj params  + 410 bridge
  D     49 positions    separate proj  overlap=20%  ~430K proj params   + 205 bridge
  E     8 flat groups   separate proj  overlap=20%  ~2.6M proj params   + 205 bridge

Decision rules
--------------
  A > Ref      → learned proj helps (pure exclusive is already better)
  B > A        → bridge neurons add value on top of learned proj
  B vs C       → optimal overlap fraction (20% vs 40%)
  D vs B       → position-level vs row-level spatial resolution
  E vs B       → VGG16 spatial rows vs flat equal blocks (structure matters?)
  B vs Ref     → combined (learned local + random global) vs all-random
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_spatial_grouped import SGNNET_SpatialGrouped
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import (
    trainer_kwargs, topology_kwargs, run_metadata,
)
from src.training.dataset import make_loaders

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("DEVICE", "mps")
if len(sys.argv) > 1 and sys.argv[1] == "--device":
    DEVICE = sys.argv[2]

EPOCHS       = 150
BATCH        = 128
SEED         = 42
DATA         = "data/store.h5"
N, D, K_ITER = 1024, 64, 8
ALPHA_AHEBB  = 0.7   # step29 Config C — all-time best (75.24%)

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


# ── Model factories ────────────────────────────────────────────────────────────

def wrap(base: nn.Module) -> nn.Module:
    resonant = SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_ref() -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return wrap(base)


def make_grouped(n_groups_proj: int, mode: str,
                 overlap_frac: float, shared: bool = False) -> nn.Module:
    torch.manual_seed(SEED)
    tk = topology_kwargs(N)
    base = SGNNET_SpatialGrouped(
        n_groups_proj=n_groups_proj,
        shared_proj=shared,
        spatial_mode=mode,
        overlap_frac=overlap_frac,
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return wrap(base)


def count_proj_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "W_proj" in n)


# ── Run helper ─────────────────────────────────────────────────────────────────
def run(label: str, model: nn.Module, config_meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va = get_loaders()

    pp    = count_proj_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  proj_params={pp:,}  total_params={total:,}")

    base_tk = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **base_tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0

    best    = max(h["val_top1"] for h in history)
    best_ep = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac    = best_ep / EPOCHS

    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    return {
        "label":        label,
        "top1_best":    best,
        "best_ep":      best_ep,
        "ep_frac":      frac,
        "elapsed_s":    round(elapsed),
        "proj_params":  pp,
        "total_params": total,
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**config_meta, "N": N, "D": D,
                                         "K_iter": K_ITER, "alpha_ahebb": ALPHA_AHEBB,
                                         "epochs": EPOCHS}),
    }


# ── Configs ────────────────────────────────────────────────────────────────────
CONFIGS = [
    (
        "Ref",
        "Ref   K_in=50 sparse gather  n_groups=128 flat  NO proj  [baseline 75.24%]",
        lambda: make_ref(),
        {"mechanism": "sparse_gather", "K_in": 50, "n_groups": 128, "overlap_frac": 1.0},
    ),
    (
        "A",
        "A     7 rows  separate proj  overlap=0%   [pure exclusive, no bridge]",
        lambda: make_grouped(7, "rows", 0.0),
        {"mechanism": "rows_separate", "n_groups_proj": 7, "overlap_frac": 0.0},
    ),
    (
        "B",
        "B     7 rows  separate proj  overlap=20%  [819 excl + 205 bridge]",
        lambda: make_grouped(7, "rows", 0.2),
        {"mechanism": "rows_separate", "n_groups_proj": 7, "overlap_frac": 0.2},
    ),
    (
        "C",
        "C     7 rows  separate proj  overlap=40%  [614 excl + 410 bridge]",
        lambda: make_grouped(7, "rows", 0.4),
        {"mechanism": "rows_separate", "n_groups_proj": 7, "overlap_frac": 0.4},
    ),
    (
        "D",
        "D     49 positions  separate proj  overlap=20%  [819 excl + 205 bridge]",
        lambda: make_grouped(49, "positions", 0.2),
        {"mechanism": "positions_separate", "n_groups_proj": 49, "overlap_frac": 0.2},
    ),
    (
        "E",
        "E     8 flat groups  separate proj  overlap=20%  [819 excl + 205 bridge]",
        lambda: make_grouped(8, "flat", 0.2),
        {"mechanism": "flat_separate", "n_groups_proj": 8, "overlap_frac": 0.2},
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}")
    print("Step 55: Spatial grouped input — learned per-group projections")
    print(f"Base: D={D} N={N} K_iter={K_ITER} AntiHebb α={ALPHA_AHEBB} wpos")
    print(f"Reference: ~75.24% (step29 Config C)")
    tr, va = get_loaders()
    print(f"Dataset: train={len(tr.dataset)}  val={len(va.dataset)}")

    results = {}
    ref_top1 = None

    for key, label, model_fn, meta in CONFIGS:
        model = model_fn()
        results[key] = run(label, model, meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n-- Spatial Grouped Input @ D={D} N={N} AntiHebb α={ALPHA_AHEBB} ------")
    print(f"  {'Key':<4}  {'top1':>7}  {'vs Ref':>8}  {'proj_params':>12}  Label")
    for key, label, _, _ in CONFIGS:
        r   = results[key]
        top = r["top1_best"]
        pp  = r["proj_params"]
        vs  = f"{(top - ref_top1)*100:+.2f}pp" if ref_top1 else "—"
        print(f"  {key:<4}  {top:.4f}  {vs:>8}  {pp:>12,}  {label.split('[')[0].strip()}")

    print(f"\n  Interpretation:")
    print(f"  A > Ref       → learned proj outperforms K_in=50 random gather")
    print(f"  B > A         → bridge neurons add value (overlap needed)")
    print(f"  B vs C        → optimal bridge fraction (20% vs 40%)")
    print(f"  D vs B        → position-level vs row-level resolution")
    print(f"  E vs B        → VGG16 spatial structure vs flat groups")

    import json
    out_path = "results/train_step55_spatial_grouped_input.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
