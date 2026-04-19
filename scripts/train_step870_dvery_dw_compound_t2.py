"""Step 870: D_very + ΔW-proj compound T2 — does synergy survive full training?

MOTIVATION
==========
step858 T1 (75ep, 50% data, seed=42):
  B_dwproj    : sparsity=0.90 + ΔW = 95.24% (+1.50pp vs Ref)
  C_compound  : sparsity=0.98 + ΔW = 95.41% (+1.68pp vs Ref) → +0.18pp synergy over ΔW alone

D_very multi-seed T1 standalone (step856): +0.05pp (noise — single-seed +0.36pp was inflated).
But compound synergy at T1 (+0.18pp) is from a different mechanism: sparsity=0.98 on C_ho
may interact WITH ΔW-proj by forcing the readout to be more selective.

Question: does +0.18pp compound synergy hold at T2 (150ep, 100% data)?
If yes → add D_very (sparsity=0.98) to the base alongside ΔW.
If no  → ΔW alone remains the base.

Reference: step854 confirmed ΔW-proj T2 = 96.618% (mean 5 seeds).

CONFIGS (T2: 150ep, 100% data, seed=42, studio_mps)
  Ref_dw    : sparsity=0.90 + ΔW-proj (ΔW baseline at T2)
  C_compound: sparsity=0.98 + ΔW-proj (full compound — main hypothesis)

SUCCESS: C_compound T2 > Ref_dw T2 + 0.1pp → add D_very to base
KILL:    C_compound T2 ≤ Ref_dw T2 → D_very adds no value with ΔW at T2
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,C_compound")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step870_dvery_dw_compound_t2_seed{SEED}__{SLOT}.json"

STEP854_DW_T2_MEAN = 0.96618  # ΔW-proj T2 5-seed mean (step854)
STEP858_COMPOUND_T1 = 0.9541  # compound T1 single-seed (step858 C_compound)


class SGNNET_DeltaAH(nn.Module):
    """ΔW-proj routing (from step706/854). Confirmed +1.1pp T2."""
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos
    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    (0.90, "sparsity=0.90 + ΔW-proj (T2 baseline)"),
    "C_compound":(0.98, "sparsity=0.98 + ΔW-proj (full compound)"),
}


def make_model(sparsity: float) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
        sparsity=sparsity,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaAH(resonant)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step870 — D_very+ΔW compound T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Ref T2 (step854 mean): {STEP854_DW_T2_MEAN:.4f}")
    print(f"  Compound T1 (step858): {STEP858_COMPOUND_T1:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        sparsity, desc = CONFIGS[key]
        model = make_model(sparsity)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP854_DW_T2_MEAN)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "sparsity": sparsity, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 870 SUMMARY — D_very+ΔW compound T2")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'sparsity':>9} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<12} {r['sparsity']:>9.2f} {r['best']:>7.4f} {dv}")
    comp = results.get("C_compound", {})
    ref_dw = results.get("Ref_dw", {})
    if comp and ref_dw:
        synergy = comp["best"] - ref_dw["best"]
        verdict = "ADD D_very to base" if synergy >= 0.001 else "KILL — no T2 synergy"
        print(f"\n  T1 synergy={STEP858_COMPOUND_T1 - STEP854_DW_T2_MEAN + 0.0059:.3f}  "
              f"T2 synergy={synergy*100:+.2f}pp → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
