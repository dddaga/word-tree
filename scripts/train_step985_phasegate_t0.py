"""Step 985: Per-node PhaseGate routing — T0 scout (20ep, 50% data, Imagenette).

MECHANISM
=========
Replace ΔW-proj routing weights with a per-node learned weight vector w_n[i]
that gates aggregation via phase alignment + stochastic leaky ReLU.

COMPOUNDING RULE: PhaseGate REPLACES ΔW-proj on the same signal path.
Comparison: PhaseGate vs Ref (ΔW-proj). NOT stacked on top.

CONFIGS (see train_step985_configs.py)
  Ref                   — canonical ΔW-proj (step199 baseline)
  A_phasegate           — PhaseGate, stochastic alpha per fwd (random)
  B_phasegate_norandom  — PhaseGate, fixed alpha=1.0 (isolates stochasticity)
  C_phasegate_learned   — PhaseGate, single global learned scalar alpha

ADVANCE CRITERION: A/B/C >= Ref +0.5pp → advance T1.

v2 fix: asymmetric [relu(s), alpha*relu(-s)] + sum-divide norm.
v1 KILLED: [s,-s]+softmax was always 50/50 (symmetric, zero routing signal).

CUDA AUDIT (5060ti checklist):
  Model: SGNNET_PhaseGate is a new routing variant; cannot wrap SGNNET_Resonant_CUDA
  or SGNNET_AntiHebbian_CUDA (PhaseGate replaces the routing kernel entirely).
  pin_memory=True applied to DataLoaders. Batch transfers use non_blocking=True
  via Trainer (see src/training/trainer.py). AMP disabled (fp32 only).

Usage:
    python scripts/train_step985_phasegate_t0.py [--device cuda] [--smoke_test]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders, make_subset_loader
from scripts.train_step985_configs  import make_configs, N, N_OUT

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--data",       default="data/store.h5")
parser.add_argument("--smoke_test", action="store_true",
                    help="One forward pass per config, exit 0 on success")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS   = args.epochs
BATCH    = 512
SEED     = args.seed
N_IN     = 25088

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step985_phasegate_t0_seed{SEED}__{SLOT}.json"


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test():
    print("=== SMOKE TEST ===")
    CONFIGS = make_configs(SEED)
    dummy   = torch.randn(4, N_IN)
    all_ok  = True
    for name, builder in CONFIGS.items():
        m = builder().train()  # train mode so alpha is stochastic for A_phasegate
        with torch.no_grad():
            # Use debug=True for PhaseGate configs to verify asymmetry
            has_debug = hasattr(m, "_phase_gate_agg")
            if has_debug:
                out = m(dummy, debug=True)
            else:
                out = m(dummy)
        ok = (out.shape == (4, N_OUT)) and (not torch.isnan(out).any())
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:<35}  params={n_p:,}  shape={tuple(out.shape)}  {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    if not all_ok:
        print("=== SMOKE TEST FAILED ==="); sys.exit(1)
    print("=== SMOKE TEST PASSED ==="); sys.exit(0)


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
    top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
    best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
    print(f"  DONE: best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")
    return {"best": round(best, 4), "best_ep": best_ep, "n_params": n_p,
            "elapsed_s": round(elapsed, 1), "history": [round(v, 4) for v in top1h]}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if args.smoke_test:
        smoke_test()

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    CONFIGS = make_configs(SEED)

    print(f"\n{'='*70}")
    print(f"step985 v2 — PhaseGate routing T0 (20ep, 50% data, Imagenette)")
    print(f"  v2: asymmetric [relu(s), alpha*relu(-s)] + sum-divide norm")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  Configs: {', '.join(CONFIGS)}")
    print(f"  Advance: A/B/C >= Ref +0.5pp")
    print(f"{'='*70}\n")

    _pin = (DEVICE.type == "cuda")
    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=_pin)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=_pin)

    results = {"step": "step985", "seed": SEED, "configs": {}}
    for name, builder in CONFIGS.items():
        results["configs"][name] = train_config(name, builder, tr, va)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"]["Ref"]["best"]
    print(f"\n{'='*70}")
    print(f"step985 SUMMARY  (Ref={ref_best:.4f})")
    for name, r in results["configs"].items():
        delta   = r["best"] - ref_best
        verdict = ("Ref" if name == "Ref"
                   else "ADVANCE" if delta >= 0.005
                   else "NEUTRAL/KILL")
        print(f"  {name:<35}  {r['best']:.4f}  ({delta:+.4f})  {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
