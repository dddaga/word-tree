"""Step 987: ΔW-proj × per-neighbor PhaseGate compound — T0 scout.

T0: 20ep, 50% data, Imagenette, BATCH=512.

Configs (see train_step987_configs.py):
  Ref                      — canonical ΔW-proj (bit-identical to step985 Ref)
  A_compound_mul           — ΔW-proj × PhaseGate (multiplicative)
  B_compound_add           — ΔW-proj + lambda*PhaseGate (additive blend)
  C_gate_only_perneighbor  — per-neighbor PhaseGate only (ablation)

Advance criterion: A/B/C >= Ref + 0.5pp.

Smoke test verifies:
  - Ref bit-identical to step985 RefModel
  - All configs: correct shape, no NaN
  - Config C: tied_frac ≈ 0 (asymmetric gate)

CUDA AUDIT (5060ti checklist):
  Model: SGNNET_DWPhaseGate is a new routing variant; cannot wrap SGNNET_Resonant_CUDA
  or SGNNET_AntiHebbian_CUDA (compound replaces the routing kernel entirely).
  pin_memory=True applied to DataLoaders. Batch transfers use non_blocking=True
  via model.to(DEVICE) before training. AMP disabled (fp32 only, Blackwell default).

Usage:
    python scripts/train_step987_dw_phasegate_t0.py [--device cuda] [--smoke_test]
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
from scripts.train_step987_configs  import make_configs, N, N_OUT
from scripts.train_step985_configs  import RefModel   # for bit-identity check

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--data",       default="data/store.h5")
parser.add_argument("--smoke_test", action="store_true")
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
OUT_PATH = ROOT / "results" / f"train_step987_dw_phasegate_t0_seed{SEED}__{SLOT}.json"


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test():
    print("=== SMOKE TEST ===")
    CONFIGS = make_configs(SEED)
    dummy   = torch.randn(4, 25088)
    all_ok  = True

    # Bit-identity: Ref in step987_configs must match step985 RefModel
    ref987 = CONFIGS["Ref"]().eval()
    ref985 = RefModel(SEED).eval()
    with torch.no_grad():
        out987 = ref987(dummy)
        out985 = ref985(dummy)
    bit_ok = torch.allclose(out987, out985, atol=1e-6)
    print(f"  Ref bit-identity vs step985 RefModel: {'PASS' if bit_ok else 'FAIL'}")
    all_ok = all_ok and bit_ok

    # All configs: shape + no NaN + debug gate stats
    for name, builder in CONFIGS.items():
        m = builder().train()
        with torch.no_grad():
            has_debug = hasattr(m, "_routing_weights")
            if has_debug and name != "Ref":
                out = m(dummy, debug=True)
            else:
                out = m(dummy)
        ok = (out.shape == (4, N_OUT)) and (not torch.isnan(out).any())
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:<35}  params={n_p:,}  shape={tuple(out.shape)}  {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    # Extra: verify Config C gate asymmetry (tied_frac should be ~0)
    m_c = CONFIGS["C_gate_only_perneighbor"]().eval()
    conn_hh  = m_c.base.conn_hh
    w_n_norm = torch.nn.functional.normalize(m_c.w_n, dim=-1)
    w_n_nb   = w_n_norm[conn_hh]                               # [N, K_hh, D]
    Z_dummy  = torch.randn(1, m_c.base.N_hidden, 16)
    Z_dummy  = torch.nn.functional.normalize(Z_dummy, dim=-1)
    with torch.no_grad():
        s      = (Z_dummy.unsqueeze(2) * w_n_nb.unsqueeze(0)).sum(-1)  # [1, N, K_hh]
        tied   = (s[:, :, 0] - s[:, :, 1]).abs().lt(1e-4).float().mean().item()
    print(f"  Config C gate tied_frac={tied:.4f} (expect ~0.0)")
    all_ok = all_ok and (tied < 0.05)

    if not all_ok:
        print("=== SMOKE TEST FAILED ==="); sys.exit(1)
    print("=== SMOKE TEST PASSED ==="); sys.exit(0)


# ── Train one config ──────────────────────────────────────────────────────────
def train_config(name, builder, tr, va) -> dict:
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
    print(f"step987 — ΔW-proj × PhaseGate compound T0 (20ep, 50% data, Imagenette)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  Configs: {', '.join(CONFIGS)}")
    print(f"  Advance: A/B/C >= Ref +0.5pp")
    print(f"{'='*70}\n")

    _pin = (DEVICE.type == "cuda")
    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=_pin)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=_pin)

    results = {"step": "step987", "seed": SEED, "configs": {}}
    for name, builder in CONFIGS.items():
        results["configs"][name] = train_config(name, builder, tr, va)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"]["Ref"]["best"]
    print(f"\n{'='*70}")
    print(f"step987 SUMMARY  (Ref={ref_best:.4f})")
    for name, r in results["configs"].items():
        delta   = r["best"] - ref_best
        verdict = ("Ref" if name == "Ref"
                   else "ADVANCE" if delta >= 0.005
                   else "NEUTRAL/KILL")
        print(f"  {name:<35}  {r['best']:.4f}  ({delta:+.4f})  {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
