"""Step 875: Hub aggregation T2 — paper-bound validation at full data.

MOTIVATION
==========
step872 T1 (75ep, 50% data, 5060ti_cuda):
  Ref_dw  : 95.36%
  A_hub005: 95.54% (+0.18pp VIABLE)

alpha=0.05 global mean adds just enough background context.
T1 verdict: VIABLE — advances to T2 for paper claim.

CONFIGS (T2: 150ep, 100% data, seed=42)
  Ref_dw  : standard ΔW-proj (K_hh=2)
  A_hub005: alpha_hub=0.05 — light global mean

SUCCESS: A_hub005 ≥ Ref + 0.2pp → hub becomes paper finding
VIABLE:  A_hub005 ≥ Ref + 0.0pp → note in ablation table
KILL:    A_hub005 < Ref - 0.1pp → T1 was noise
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
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_hub005")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step875_hub_t2_seed{SEED}__{SLOT}.json"

STEP872_HUB_T1 = 0.9554   # A_hub005 T1 (step872)
STEP199_REF_T2 = 0.9552   # ΔW-proj T2 baseline (step199 equivalent)


class SGNNET_Hub(nn.Module):
    """ΔW-proj routing + global hub aggregation.

    alpha_hub=0 → identical to standard ΔW-proj.
    """
    def __init__(self, resonant, alpha_hub: float = 0.0):
        super().__init__()
        self.m = resonant
        self.alpha_hub = alpha_hub

    @property
    def W_pos(self):   return self.m.W_pos
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
            if self.alpha_hub > 0:
                Z_hub = Z_fwd.mean(dim=1, keepdim=True)
                hub_signal = self.alpha_hub * Z_hub.expand(-1, Z_fwd.size(1), -1)
            else:
                hub_signal = 0.0
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize(
                (Z_nb.sum(2) + Z_ref + hub_signal).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":   (0.0,  "alpha_hub=0.0 — standard ΔW-proj (T2 baseline)"),
    "A_hub005": (0.05, "alpha_hub=0.05 — light global mean (T1 VIABLE)"),
}


def make_model(alpha_hub: float) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=False)
    else:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_Hub(resonant, alpha_hub=alpha_hub)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step875 — Hub aggregation T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  T1 result (step872): A_hub005={STEP872_HUB_T1:.4f} (+0.18pp VIABLE)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        alpha_hub, desc = CONFIGS[key]
        model = make_model(alpha_hub)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False
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
        delta = best - (ref_acc if ref_acc is not None else STEP199_REF_T2)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "alpha_hub": alpha_hub, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 875 SUMMARY — Hub T2")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'alpha':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<12} {r['alpha_hub']:>6.3f} {r['best']:>7.4f} {dv}")
    hub = results.get("A_hub005", {})
    ref = results.get("Ref_dw", {})
    if hub and ref:
        gap = hub["best"] - ref["best"]
        if gap >= 0.002:
            verdict = "STRONG — hub is a paper finding"
        elif gap >= 0.0:
            verdict = "VIABLE — ablation table entry"
        else:
            verdict = "KILL — T1 was noise"
        print(f"\n  Hub T2 verdict: {gap*100:+.2f}pp → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
