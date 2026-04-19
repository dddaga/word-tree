"""Step 868: Z-memory activation retention T0 — temporal spreading.

MOTIVATION
==========
Standard ΔW-proj routing: each K_iter step reads only the *current* Z state.
Z-memory retains a running average of past active states:
  Z_route = Z_fwd + gamma * Z_mem  (memory amplifies previously-active nodes)
  Z_mem   = gamma * Z_mem + (1-gamma) * Z_fwd  (EMA update)

At gamma=0: identical to ΔW-proj Ref. At gamma>0: activations from earlier
K_iter steps persist and continue to broadcast signal — "temporal teleportation".
Orthogonal to alpha_reflect (which retains suppressed signal).

CONFIGS (T0: 20ep, 50% data, seed=42, mini_cpu)
  Ref_dw    : gamma=0.0 — standard ΔW-proj (control)
  A_g03     : gamma=0.3 — light temporal accumulation
  B_g05     : gamma=0.5 — medium accumulation
  C_g08     : gamma=0.8 — heavy accumulation
  D_g09     : gamma=0.9 — very heavy accumulation

SUCCESS: any gamma variant > Ref_dw +0.1pp → advance to T1
KILL:    all gamma variants ≤ Ref_dw → temporal memory adds no value
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
parser.add_argument("--configs", default="Ref_dw,A_g03,B_g05,C_g08,D_g09")
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
OUT_PATH = ROOT / "results" / f"train_step868_zmem_retention_t0_seed{SEED}__{SLOT}.json"

STEP858_DW_T1 = 0.9524  # ΔW-proj T1 reference (step858 B_dwproj)


class SGNNET_ZMem(nn.Module):
    """ΔW-proj routing with temporal Z-memory accumulation.

    Z_route = Z_fwd + gamma * Z_mem   (augmented routing signal)
    Z_mem   = gamma * Z_mem + (1-gamma) * Z_fwd  (EMA of past activations)
    gamma=0 → identical to standard ΔW-proj.
    """
    def __init__(self, resonant, gamma: float = 0.0):
        super().__init__()
        self.m = resonant
        self.gamma = gamma

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
        Z_mem = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_route = Z_fwd + self.gamma * Z_mem
            Z_mem = self.gamma * Z_mem + (1.0 - self.gamma) * Z_fwd
            Z_nb = Z_route[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw": (0.0,  "gamma=0.0 — standard ΔW-proj (control)"),
    "A_g03":  (0.3,  "gamma=0.3 — light temporal accumulation"),
    "B_g05":  (0.5,  "gamma=0.5 — medium accumulation"),
    "C_g08":  (0.8,  "gamma=0.8 — heavy accumulation"),
    "D_g09":  (0.9,  "gamma=0.9 — very heavy accumulation"),
}


def make_model(gamma: float) -> nn.Module:
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
    return SGNNET_ZMem(resonant, gamma=gamma)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step868 — Z-memory activation retention T0 (20ep, 50% data)")
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
        gamma, desc = CONFIGS[key]
        model = make_model(gamma)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
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
            "gamma": gamma, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 868 SUMMARY — Z-memory retention T0")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'gamma':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<10} {r['gamma']:>6.2f} {r['best']:>7.4f} {dv}")
    best_variant = max((k for k in results if k != "Ref_dw"),
                       key=lambda k: results[k]["best"], default=None)
    if best_variant:
        delta = results[best_variant]["best"] - (ref_acc or 0)
        verdict = "ADVANCES to T1" if delta > 0.001 else "KILLED — no benefit"
        print(f"\n  Best: {best_variant} ({delta*100:+.2f}pp) → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
