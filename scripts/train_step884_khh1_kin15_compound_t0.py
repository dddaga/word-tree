"""Step 884: K_hh=1 + K_in=15 compound efficiency probe T0.

MOTIVATION
==========
Two independent efficiency mechanisms:
  K_hh=1: 50% routing MAC reduction (step865 T0: -0.48pp VIABLE)
  K_in=15: 40% seed computation reduction (step632 T2: -0.33pp CONFIRMED)

Both operate on different signal paths (routing vs seed gather) →
compound is safe per compounding rule.

If K_hh=1 T1 (step878) is VIABLE, the compound K_hh=1 + K_in=15 would be
the paper's "ultra-efficient" config:
  - Routing MACs: 50% fewer (K_hh=1 vs K_hh=2)
  - Seed MACs: 40% fewer (K_in=15 vs K_in=25)
  - Combined: ~45% total FLOPs reduction vs standard ΔW-proj
  - Paper claim: 0.55× FLOPs at <1.5pp accuracy cost

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw    : K_hh=2, K_in=25 — standard (0.98M routing MACs)
  A_khh1    : K_hh=1, K_in=25 — routing only (-50% routing MACs)
  B_kin15   : K_hh=2, K_in=15 — seed only (-40% seed MACs)
  C_compound: K_hh=1, K_in=15 — both reductions combined

SUCCESS: C_compound within -1.5pp of Ref → ultra-efficient config is viable
KILL:    C_compound > -1.5pp below Ref → compound exceeds capacity limit
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
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_khh1,B_kin15,C_compound")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10; D = 16; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step884_khh1_kin15_compound_t0_seed{SEED}__{SLOT}.json"

STEP865_REF = 0.9373   # K_hh=2, K_in=25 T0 reference
STEP865_A   = 0.9325   # K_hh=1 T0 result (-0.48pp)
STEP632_B   = 0.9513   # K_in=15 T2 result (T0 estimate ~-0.3pp)


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


# (k_hh, k_in, desc)
CONFIGS = {
    "Ref_dw":    (2, 25, "K_hh=2, K_in=25 — standard ΔW-proj (0.98M routing MACs)"),
    "A_khh1":    (1, 25, "K_hh=1, K_in=25 — 50% fewer routing MACs"),
    "B_kin15":   (2, 15, "K_hh=2, K_in=15 — 40% fewer seed MACs"),
    "C_compound":(1, 15, "K_hh=1, K_in=15 — both: ~45% total FLOPs reduction"),
}


def make_model(k_hh: int, k_in: int) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, k_hh // 4) if k_hh > 1 else 1
    K_l = k_hh - K_r if k_hh > 1 else 0
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=k_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
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

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=False)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step884 — K_hh=1 + K_in=15 compound efficiency probe T0")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  step865 context: K_hh=1={STEP865_A:.4f}(-0.48pp), Ref={STEP865_REF:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        k_hh, k_in, desc = CONFIGS[key]
        model = make_model(k_hh, k_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # FLOPs estimate: seed + routing
        seed_m = N * k_in * D * 2 / 1e6
        route_m = N * K_ITER * k_hh * D * 2 / 1e6
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")
        print(f"  FLOPs: seed={seed_m:.3f}M  routing={route_m:.3f}M  total≈{seed_m+route_m:.3f}M")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 10 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP865_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "k_hh": k_hh, "k_in": k_in, "label": desc, "n_params": n_p,
            "seed_flops_M": round(seed_m, 4), "route_flops_M": round(route_m, 4),
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 884 SUMMARY — K_hh=1 + K_in=15 compound efficiency probe")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'K_hh':>5} {'K_in':>5} {'FLOPs':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        total_f = r['seed_flops_M'] + r['route_flops_M']
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "    (ref)"
        print(f"  {k:<12} {r['k_hh']:>5} {r['k_in']:>5} {total_f:>7.3f}M {r['best']:>7.4f} {dv}")
    compound = results.get("C_compound", {})
    ref = results.get("Ref_dw", {})
    if compound and ref:
        gap = compound["best"] - ref["best"]
        flop_ratio = (compound["seed_flops_M"] + compound["route_flops_M"]) / (ref["seed_flops_M"] + ref["route_flops_M"])
        if gap >= -0.015:
            verdict = f"VIABLE — compound {flop_ratio:.2f}× FLOPs at <1.5pp: advance to T1"
        else:
            verdict = f"KILLED — {gap*100:.2f}pp at {flop_ratio:.2f}× FLOPs: compound exceeds capacity"
        print(f"\n  compound: {gap*100:+.2f}pp @ {flop_ratio:.2f}× FLOPs → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
