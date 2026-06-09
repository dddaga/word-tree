"""Step 991: Hebbian prune-and-grow dynamic topology — T0 scout (20ep, 50% Imagenette).

Tests brief §9: edges with high ΔW-proj alignment are kept; low-coactivation
edges are pruned and regrown at epoch boundaries. This is the one safe topology-
learning pathway: epoch-boundary update mirrors AH (CONFIRMED load-bearing since
step218). In-forward multiplicative gating causes gate-death (KILLED steps 873–916);
epoch-boundary is safe.

Edge score: |c_ij| = |dot(Z[j], normalize(W[i]-W[j]))| accumulated per forward
step, normalised per epoch, then bottom-prune_frac edges pruned and replaced.

Configs (all N=2048, D=16, K_iter=5, K_hh=2):
  Ref              — SGNNET_SmallWorld baseline (no Hebbian)
  A_hebbian_random — prune_frac=0.10, grow_mode='random'
  B_hebbian_wpos   — prune_frac=0.10, grow_mode='wpos' (geometry-guided regrowth)
  C_hebbian_fast   — prune_frac=0.20, grow_mode='random' (faster turnover)

Trainer calls tick_epoch() automatically (trainer.py:331) — no manual loop needed.
Advance criterion: any config >= Ref + 0.5pp → advance T1.

Usage:
    python scripts/train_step991_hebbian_t0.py [--device cuda] [--epochs 20]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training.trainer              import Trainer
from src.training.experiment_config    import trainer_kwargs
from src.training.dataset              import make_loaders, make_subset_loader
from src.sgnnet.model_smallworld       import SGNNET_SmallWorld
from src.sgnnet.model_hebbian          import HebbianRewirer
from src.sgnnet.model_resonant_cuda    import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA

# ── Constants ─────────────────────────────────────────────────────────────────
N        = 2048;  N_IN = 25088;  N_OUT = 10
D        = 16;    K_HH = 2;      K_IN  = 25;  K_ITER = 5
N_GROUPS = max(8, N // 8)        # 256
K_r      = max(1, K_HH // 4)    # 1
K_l      = K_HH - K_r            # 1

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--data",   default="data/store.h5")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 512
SEED   = args.seed
SLOT   = os.environ.get("SGN_SLOT", "local")
OUT    = ROOT / "results" / f"train_step991_hebbian_t0_seed{SEED}__{SLOT}.json"


# ── Model factories ───────────────────────────────────────────────────────────

ALPHA_REFLECT = 0.5
ALPHA_AHEBB   = 1.0


# Diagnostic batch for ΔW-proj edge scoring (set in main from first train batch)
_DIAG_X: torch.Tensor | None = None


def _make_full(seed: int, prune_frac: float | None = None,
               grow_mode: str = "random"):
    """Canonical SmallWorld → Resonant_CUDA → AntiHebbian_CUDA chain.
    If prune_frac set, attach a HebbianRewirer: each tick_epoch prunes/grows
    base.conn_hh from ΔW-proj scores on the diagnostic batch, then invalidates
    the AH suppression-weight cache (supp_w depends on conn_hh)."""
    torch.manual_seed(seed)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant_CUDA(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
        resonance_threshold=0.0, compile=(DEVICE.type == "cuda"),
    )
    ah = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    if prune_frac is not None:
        rewirer = HebbianRewirer(base, prune_frac=prune_frac, grow_mode=grow_mode)
        _orig_tick = ah.tick_epoch
        def _tick():
            _orig_tick()
            if _DIAG_X is not None:
                n = rewirer.rewire(_DIAG_X)
                ah._invalidate_supp_w()
                print(f"    [hebbian] rewired {n} edges "
                      f"(total {rewirer.n_rewired_total})", flush=True)
        ah.tick_epoch = _tick
    return ah


def make_configs(seed: int) -> dict:
    return {
        "Ref":              lambda: _make_full(seed),
        "A_hebbian_random": lambda: _make_full(seed, 0.10, "random"),
        "B_hebbian_wpos":   lambda: _make_full(seed, 0.10, "wpos"),
        "C_hebbian_fast":   lambda: _make_full(seed, 0.20, "random"),
    }


# ── Train one config ──────────────────────────────────────────────────────────

def train_config(name: str, builder, tr, va) -> dict:
    model = builder().to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kw    = trainer_kwargs(N, n_epochs=EPOCHS)
    print(f"\n{'─'*60}")
    print(f"Config: {name}  params={n_p:,}  device={DEVICE}", flush=True)
    t0 = time.time()
    history = Trainer(
        model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw,
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
    print(f"  DONE: best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")
    return {
        "best": round(best, 4), "best_ep": best_ep,
        "n_params": n_p, "elapsed_s": round(elapsed, 1),
        "history": [round(v, 4) for v in top1h],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    CONFIGS = make_configs(SEED)

    print(f"\n{'='*70}")
    print(f"step991 — Hebbian prune-and-grow topology T0 (20ep, 50% Imagenette)")
    print(f"  Mechanism: epoch-boundary prune+grow on ΔW-proj edge scores")
    print(f"  Safe pattern: epoch-boundary (AH precedent); NOT in-forward gating")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  Configs: {', '.join(CONFIGS)}")
    print(f"  Advance: any config >= Ref + 0.5pp")
    print(f"{'='*70}\n")

    _pin = (DEVICE.type == "cuda")
    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=_pin)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                         pin_memory=_pin)

    # Fixed diagnostic batch for ΔW-proj edge scoring at epoch boundaries
    global _DIAG_X
    _DIAG_X = next(iter(tr))[0][:256].to(DEVICE)

    results = {"step": "step991", "seed": SEED, "configs": {}}
    for name, builder in CONFIGS.items():
        results["configs"][name] = train_config(name, builder, tr, va)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"]["Ref"]["best"]
    print(f"\n{'='*70}")
    print(f"step991 SUMMARY  (Ref={ref_best:.4f})")
    for name, r in results["configs"].items():
        delta   = r["best"] - ref_best
        verdict = ("Ref" if name == "Ref"
                   else "ADVANCE" if delta >= 0.005
                   else "NEUTRAL/KILL")
        print(f"  {name:<30}  {r['best']:.4f}  ({delta:+.4f})  {verdict}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
