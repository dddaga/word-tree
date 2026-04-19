"""Step 869: Hub aggregation / global teleportation T0.

MOTIVATION
==========
Current SGNNET: each node only reaches K_hh=2 neighbors per K_iter step.
Over K_iter=5 steps, a signal can travel at most 5 hops.

Hub aggregation provides GLOBAL connectivity in O(N·D):
  Z_hub = mean(Z_fwd, dim=nodes)  — global mean of all active states
  Z[i]  = normalize(Z_nb[i] + Z_ref[i] + alpha_hub * Z_hub)

Any activated node instantly "radiates" to the hub; any node receives it.
Effective path length: 1 hop (regardless of graph distance).
This is "activation teleportation" — the global mean acts as a broadcast medium.

Orthogonal to ΔW-proj (different aggregation channel, same W_pos).
CUDA checklist: pin_memory, non_blocking, SGNNET_Resonant_CUDA imported.

CONFIGS (T0: 20ep, 50% data, seed=42, 5060ti_cuda)
  Ref_dw      : alpha_hub=0   — standard ΔW-proj (control)
  A_hub005    : alpha_hub=0.05 — very light global signal
  B_hub03     : alpha_hub=0.3  — medium hub influence
  C_hub10     : alpha_hub=1.0  — full hub (balance hub vs local routing)
  D_hub_beam  : alpha_hub=0.3, only top-16 most active nodes → hub

SUCCESS: any alpha_hub > Ref_dw +0.1pp → global teleportation useful
"""
# CUDA-5060ti-validated
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
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_hub005,B_hub03,C_hub10,D_hub_beam")
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
OUT_PATH = ROOT / "results" / f"train_step869_hub_aggregation_t0_seed{SEED}__{SLOT}.json"

STEP858_DW_T1 = 0.9524  # ΔW-proj T1 reference (step858 B_dwproj)


class SGNNET_Hub(nn.Module):
    """ΔW-proj routing + global hub aggregation (activation teleportation).

    At each K_iter step:
      Z_hub = mean(Z_fwd, dim=1)  [or top-M beam if hub_m > 0]
      Z_out = normalize(Z_nb + Z_ref + alpha_hub * Z_hub)
    alpha_hub=0 → identical to standard ΔW-proj.
    """
    def __init__(self, resonant, alpha_hub: float = 0.0, hub_m: int = 0):
        super().__init__()
        self.m = resonant
        self.alpha_hub = alpha_hub
        self.hub_m = hub_m  # 0 = global mean; >0 = top-M beam hub

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
            if self.alpha_hub > 0:
                if self.hub_m > 0:
                    norms = Z_fwd.norm(dim=-1)  # [B, N]
                    M = min(self.hub_m, Z_fwd.size(1))
                    top_idx = norms.topk(M, dim=1).indices  # [B, M]
                    Z_top = Z_fwd.gather(
                        1, top_idx.unsqueeze(-1).expand(-1, -1, Z_fwd.size(-1)))
                    Z_hub = Z_top.mean(dim=1, keepdim=True)  # [B, 1, D]
                else:
                    Z_hub = Z_fwd.mean(dim=1, keepdim=True)  # [B, 1, D]
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
    "Ref_dw":    (0.0,  0,  "alpha_hub=0 — standard ΔW-proj (control)"),
    "A_hub005":  (0.05, 0,  "alpha_hub=0.05 — very light global signal"),
    "B_hub03":   (0.3,  0,  "alpha_hub=0.3 — medium hub (global mean)"),
    "C_hub10":   (1.0,  0,  "alpha_hub=1.0 — full hub balance"),
    "D_hub_beam":(0.3,  16, "alpha_hub=0.3, top-16 beam hub (1.6% of nodes)"),
}


def make_model(alpha_hub: float, hub_m: int) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_Hub(resonant, alpha_hub=alpha_hub, hub_m=hub_m)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=True)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True,
    )

    print(f"\n{'='*70}")
    print(f"step869 — Hub aggregation / global teleportation T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Ref: ΔW-proj T1={STEP858_DW_T1:.4f} (step858 B_dwproj)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        alpha_hub, hub_m, desc = CONFIGS[key]
        model = make_model(alpha_hub, hub_m)
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
                                   flush=True) if (m['epoch']+1) % 5 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "alpha_hub": alpha_hub, "hub_m": hub_m, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 869 SUMMARY — Hub aggregation T0")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'alpha':>6} {'hub_m':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<12} {r['alpha_hub']:>6.3f} {r['hub_m']:>6} {r['best']:>7.4f} {dv}")
    best_v = max((k for k in results if k != "Ref_dw"),
                 key=lambda k: results[k]["best"], default=None)
    if best_v:
        delta = results[best_v]["best"] - (ref_acc or 0)
        verdict = "ADVANCES to T1" if delta > 0.001 else "KILLED — no benefit"
        print(f"\n  Best: {best_v} ({delta*100:+.2f}pp) → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
