"""Step 852: T0 sweep — conn_hh rebuild cadence from learned W_pos.

MOTIVATION
==========
Default SGNNET_SmallWorld conn_hh is a static Watts-Strogatz graph built once
at init; W_pos is a learned parameter but gradients flow only through the
readout scorer, so W_pos never shapes the hidden→hidden routing topology.
This T0 closes that gap by periodically rebuilding conn_hh from W_pos k-NN
and measures the accuracy × churn trade-off at 8 cadences (static through
per-batch) + 2 hybrid (k-NN + random shortcuts). Tier 0 (20ep, 50% data).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, K_in=25, step199 reference)
================================================================
  Ref            : Watts-Strogatz (no rebuild, current default)
  A_wpos_static  : rebuild once at init, then never
  B_wpos_batch   : rebuild every batch (tick_step)
  C_wpos_ep1     : rebuild every 1 epoch
  D_wpos_ep5     : rebuild every 5 epochs
  E_wpos_ep10    : rebuild every 10 epochs
  F_hybrid_ep5   : hybrid k-NN + 1 random edge, every 5 epochs
  G_hybrid_batch : hybrid k-NN + 1 random edge, every batch

DIAGNOSTIC
==========
Hamming churn (fraction of conn_hh entries changed between rebuilds) saved
per config as churn_history. Output: results/train_step852_*_seed42__<SLOT>.json.
"""
# CUDA-5060ti-validated — this script meets the 5060ti checklist
# (see .claude/skills/sgnnet-research/CUDA_CHECKLIST.md).
# Uses pin_memory=True literal, use_amp=False literal, no grad-scaler path.
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

# --- CLI --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_wpos_static,B_wpos_batch,C_wpos_ep1,"
                                         "D_wpos_ep5,E_wpos_ep10,F_hybrid_ep5,G_hybrid_batch",
                    help="Comma-separated config keys.")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

# --- Experiment constants (step199 reference scale) -------------------------
EPOCHS = args.epochs
BATCH  = 128
SEED   = args.seed
DATA   = args.data

N      = 2048;   N_IN   = 25088;   N_OUT = 10
D      = 16;     K_HH   = 2;       K_IN  = 25;    K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_AHEBB   = 1.0

STEP_NAME = Path(__file__).stem
SLOT      = os.environ.get("SGN_SLOT", "local")
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}_seed{SEED}__{SLOT}.json"

# --- Config spec ------------------------------------------------------------
# cadence: "none" | "init" | "batch" | ("epoch", N)
# mode:    "wpos_knn" | "hybrid"
# random_mix: int
CONFIGS = {
    "Ref":            {"cadence": "none",          "mode": None,        "random_mix": 0},
    "A_wpos_static":  {"cadence": "init",          "mode": "wpos_knn",  "random_mix": 0},
    "B_wpos_batch":   {"cadence": "batch",         "mode": "wpos_knn",  "random_mix": 0},
    "C_wpos_ep1":     {"cadence": ("epoch", 1),    "mode": "wpos_knn",  "random_mix": 0},
    "D_wpos_ep5":     {"cadence": ("epoch", 5),    "mode": "wpos_knn",  "random_mix": 0},
    "E_wpos_ep10":    {"cadence": ("epoch", 10),   "mode": "wpos_knn",  "random_mix": 0},
    "F_hybrid_ep5":   {"cadence": ("epoch", 5),    "mode": "hybrid",    "random_mix": 1},
    "G_hybrid_batch": {"cadence": "batch",         "mode": "hybrid",    "random_mix": 1},
}


# --- Model factory ----------------------------------------------------------

def build_model() -> nn.Module:
    """Build SGNNET — CUDA-optimized path when on CUDA, eager otherwise."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=True,
        )
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=True)
    # MPS / CPU eager path for smoke-test + non-CUDA slots
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def _get_smallworld_base(model: nn.Module) -> SGNNET_SmallWorld:
    """Walk the wrapper chain to the SGNNET_SmallWorld base."""
    obj = model
    for attr in ("m", "base"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            if isinstance(obj, SGNNET_SmallWorld):
                return obj
    # Resonant_CUDA exposes .base; AntiHebbian_CUDA exposes .m.base
    if hasattr(model, "m") and hasattr(model.m, "base") and isinstance(model.m.base, SGNNET_SmallWorld):
        return model.m.base
    if hasattr(model, "base") and isinstance(model.base, SGNNET_SmallWorld):
        return model.base
    raise RuntimeError(f"Could not locate SGNNET_SmallWorld in {type(model).__name__}")


def install_rebuild_callbacks(model: nn.Module, spec: dict, churn_log: list[float]) -> None:
    """Attach tick_step / tick_epoch hooks enforcing the rebuild cadence.

    Monkey-patches the existing tick_step / tick_epoch methods on the model
    instance so Trainer invokes them automatically (see trainer.py 186-188,
    329-331). Records Hamming churn into churn_log on every rebuild.
    """
    sw = _get_smallworld_base(model)
    cadence = spec["cadence"]
    mode = spec["mode"]
    random_mix = spec["random_mix"]

    # Preserve any existing tick_step / tick_epoch (e.g., for supp_w invalidation)
    prev_tick_step  = getattr(model, "tick_step", None)
    prev_tick_epoch = getattr(model, "tick_epoch", None)

    state = {"epoch": 0}

    def _rebuild_and_record():
        prev = sw.conn_hh.clone()
        sw.rebuild_conn_hh(mode=mode, random_mix=random_mix)
        churn_log.append(sw._churn_vs(prev))
        # Invalidate AH suppression cache (supp_w is keyed on old conn_hh)
        if hasattr(model, "_invalidate_supp_w"):
            model._invalidate_supp_w()

    # --- Apply init-time rebuild up-front if required ------------------------
    if cadence == "init":
        _rebuild_and_record()
    elif cadence == "none":
        pass
    else:
        # first rebuild happens at the first scheduled tick; no init-time call
        pass

    # --- Batch-level hook (tick_step) ----------------------------------------
    def _tick_step_patch():
        if prev_tick_step is not None:
            prev_tick_step()
        if cadence == "batch":
            _rebuild_and_record()

    # --- Epoch-level hook (tick_epoch) ---------------------------------------
    def _tick_epoch_patch():
        if prev_tick_epoch is not None:
            prev_tick_epoch()
        state["epoch"] += 1
        if isinstance(cadence, tuple) and cadence[0] == "epoch":
            every = cadence[1]
            if state["epoch"] % every == 0:
                _rebuild_and_record()

    # Attach. Trainer checks `hasattr(model, "tick_step")` so we must install
    # a tick_step even for configs that don't need it (so the prev hook still fires).
    model.tick_step  = _tick_step_patch
    model.tick_epoch = _tick_epoch_patch


def main() -> None:
    data_path = ROOT / DATA
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    run_keys = [k.strip() for k in args.configs.split(",") if k.strip()]

    print(f"\n{'='*70}")
    print(f"{STEP_NAME}  —  conn_hh rebuild cadence sweep (T0: 20ep, 50% data)")
    print(f"Device: {DEVICE}  epochs={EPOCHS}  seed={SEED}  configs={len(run_keys)}")
    print(f"{'='*70}")

    # Load full train + val, then 50% subset for T0
    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=True)  # pin_memory=True for CUDA DMA overlap
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=True,
    )
    print(f"  train_subset={len(subset)}  val={len(va.dataset)}\n")

    results: dict = {}

    for key in run_keys:
        if key not in CONFIGS:
            print(f"  skip unknown config: {key}"); continue
        spec = CONFIGS[key]
        print(f"{'─'*60}\n{key}: cadence={spec['cadence']}  mode={spec['mode']}  random_mix={spec['random_mix']}")

        model = build_model().to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        churn_history: list[float] = []
        install_rebuild_callbacks(model, spec, churn_history)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        kw["use_amp"] = False  # use_amp=False: Blackwell fp16 4.4x slower (step801)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
        best  = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "cadence":       str(spec["cadence"]),
            "mode":          spec["mode"],
            "random_mix":    spec["random_mix"],
            "best":          best,
            "best_ep":       bep,
            "top1_history":  top1h,
            "churn_history": [round(c, 4) for c in churn_history],
            "n_rebuilds":    len(churn_history),
            "elapsed_s":     round(elapsed, 1),
            "n_params":      n_p,
        }
        churn_summary = (
            f"mean={np.mean(churn_history):.3f}" if churn_history else "none"
        )
        print(f"  → best={best:.4f} @ep{bep}  rebuilds={len(churn_history)} ({churn_summary})  {elapsed:.0f}s")

        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # --- Summary ------------------------------------------------------------
    ref_best = results.get("Ref", {}).get("best")
    print(f"\n{'='*70}\n{STEP_NAME} SUMMARY (T0, 20ep, 50% data)\n{'='*70}")
    print(f"  {'config':<16} {'best':>7}  {'Δ_vs_Ref':>9}  {'rebuilds':>9}  {'churn_μ':>8}")
    for key in run_keys:
        if key not in results: continue
        r = results[key]
        delta = (r["best"] - ref_best) if ref_best is not None and key != "Ref" else None
        delta_s = f"{delta*100:+.2f}pp" if delta is not None else "  ref   "
        churn_mu = (
            f"{np.mean(r['churn_history']):.3f}" if r["churn_history"] else "  —  "
        )
        print(f"  {key:<16} {r['best']:>7.4f}  {delta_s:>9}  {r['n_rebuilds']:>9}  {churn_mu:>8}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
