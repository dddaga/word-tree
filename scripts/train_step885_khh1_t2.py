"""Step 885: K_hh=1 T2 — paper efficiency claim validation.

MOTIVATION
==========
step878 T1: K_hh=1 = -0.41pp vs Ref_dw (STRONG — within 0.5pp threshold)
step865 T0: K_hh=1 = -0.48pp vs Ref_dw (VIABLE)

K_hh=1 cuts routing MACs by 50% (K_hh=2 → K_hh=1).
T1 confirms this is not a scout artifact. T2 validates for paper.

EXPECTED
  Ref_dw  : ~95.5%  (step878 T1 Ref=0.9536)
  A_khh1  : ~95.0-95.2%  (Δ ≈ -0.3 to -0.5pp at T2)

SUCCESS: A_khh1 within -0.5pp of Ref → paper efficiency claim:
  "K_hh=1 achieves 50% routing MAC reduction at <0.5pp accuracy cost"
MARGINAL: -0.5 to -1.0pp → mention with caveat
KILL:     >-1.0pp → efficiency claim dropped

CONFIGS (T2: 150ep, 100% data, seed=42)
  Ref_dw : K_hh=2 (K_local=1, K_random=1) — standard ΔW-proj
  A_khh1 : K_hh=1 (K_local=0, K_random=1) — 50% fewer routing ops

# CUDA-5060ti-validated
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step885_khh1_t2_seed{SEED}__{SLOT}.json"

STEP878_REF = 0.9536   # K_hh=2 T1 reference
STEP878_A   = 0.9496   # K_hh=1 T1 result (-0.41pp)


class SGNNET_DeltaW(nn.Module):
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

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
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


# (k_hh, k_local, k_random, desc)
CONFIGS = {
    "Ref_dw": (2, 1, 1, "K_hh=2 (K_local=1, K_random=1) — standard ΔW-proj"),
    "A_khh1": (1, 0, 1, "K_hh=1 (K_local=0, K_random=1) — 50% fewer routing MACs"),
}


def make_model(k_local: int, k_random: int) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=k_local, K_random=k_random,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW(resonant)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=False)

    print(f"\n{'='*70}")
    print(f"step885 — K_hh=1 T2 validation (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  step878 T1 context: K_hh=1={STEP878_A:.4f}(-0.41pp), Ref={STEP878_REF:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    results = {}
    ref_acc = None

    for key, (k_hh, k_local, k_random, desc) in CONFIGS.items():
        model = make_model(k_local, k_random)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        route_m = N * K_ITER * k_hh * D * 2 / 1e6
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")
        print(f"  routing MACs: {route_m:.3f}M  (K_hh={k_hh})")

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
        delta = best - (ref_acc if ref_acc is not None else STEP878_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "k_hh": k_hh, "label": desc, "n_params": n_p,
            "route_flops_M": round(route_m, 4),
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 885 SUMMARY — K_hh=1 T2 paper efficiency validation")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'K_hh':>5} {'FLOPs':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "    (ref)"
        print(f"  {k:<10} {r['k_hh']:>5} {r['route_flops_M']:>7.3f}M {r['best']:>7.4f} {dv}")
    khh1 = results.get("A_khh1", {})
    ref = results.get("Ref_dw", {})
    if khh1 and ref:
        gap = khh1["best"] - ref["best"]
        flop_ratio = khh1["route_flops_M"] / ref["route_flops_M"]
        if gap >= -0.005:
            verdict = f"STRONG — {flop_ratio:.2f}× routing MACs at <0.5pp: paper claim confirmed"
        elif gap >= -0.010:
            verdict = f"MARGINAL — {flop_ratio:.2f}× routing MACs at {gap*100:.2f}pp: mention with caveat"
        else:
            verdict = f"KILLED — {gap*100:.2f}pp at {flop_ratio:.2f}× FLOPs: efficiency claim dropped"
        print(f"\n  K_hh=1: {gap*100:+.2f}pp @ {flop_ratio:.2f}× routing MACs → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
