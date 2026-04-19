"""Step 879: Z-memory gamma fine-scan T0 — nail sweet spot before T2.

MOTIVATION
==========
step868 T0 found gamma=0.8 is the sweet spot (+0.33pp). gamma=0.9 KILLED (-0.33pp).
step874 T1 running with gamma=0.8.

Before planning Z-mem T2, confirm: is 0.8 truly optimal or does 0.75/0.85 beat it?
The gap between 0.8 (+0.33pp) and 0.9 (-0.33pp) is large — a fine scan around 0.8
may reveal a sharper peak or a plateau.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw    : gamma=0.0 — standard ΔW-proj (control)
  A_g070    : gamma=0.70 — faster decay (less memory)
  B_g075    : gamma=0.75 — between step868 neutral and sweet spot
  C_g080    : gamma=0.80 — confirmed sweet spot from step868
  D_g085    : gamma=0.85 — between sweet spot and killed

SUCCESS: C_g080 or adjacent remains top → gamma=0.8 confirmed, proceed to T2
SURPRISE: B or D beats C by >0.1pp → shift T2 to that gamma
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
parser.add_argument("--configs", default="Ref_dw,A_g070,B_g075,C_g080,D_g085")
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
OUT_PATH = ROOT / "results" / f"train_step879_zmem_gamma_scan_t0_seed{SEED}__{SLOT}.json"

STEP868_DW_T0 = 0.9396   # Ref_dw T0 baseline
STEP868_G08   = 0.9429   # gamma=0.8 T0 result (+0.33pp)


class SGNNET_ZMem(nn.Module):
    """ΔW-proj with Z-memory temporal EMA (gamma=0 → standard ΔW-proj)."""
    def __init__(self, resonant, gamma: float = 0.0):
        super().__init__()
        self.m = resonant
        self.gamma = gamma

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
    "Ref_dw": (0.00, "gamma=0.0 — standard ΔW-proj (control)"),
    "A_g070": (0.70, "gamma=0.70 — faster decay"),
    "B_g075": (0.75, "gamma=0.75 — below sweet spot"),
    "C_g080": (0.80, "gamma=0.80 — confirmed sweet spot (step868 +0.33pp)"),
    "D_g085": (0.85, "gamma=0.85 — above sweet spot"),
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
                               pin_memory=False)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step879 — Z-mem gamma fine-scan T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  step868 baseline: Ref={STEP868_DW_T0:.4f}, g08={STEP868_G08:.4f} (+0.33pp)")
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
                                   flush=True) if (m['epoch']+1) % 10 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP868_DW_T0)
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
    print(f"STEP 879 SUMMARY — Z-mem gamma fine-scan")
    print(f"{'='*70}")
    print(f"  {'config':<10} {'gamma':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<10} {r['gamma']:>6.2f} {r['best']:>7.4f} {dv}")
    best_config = max(
        ((k, r) for k, r in results.items() if k != "Ref_dw"),
        key=lambda x: x[1]["best"], default=(None, {})
    )
    if best_config[0]:
        k, r = best_config
        print(f"\n  Best gamma: {r['gamma']:.2f} ({k}) → {r['best']:.4f}")
        if r["delta_vs_ref"] is not None and r["delta_vs_ref"] > 0:
            print(f"  → Use gamma={r['gamma']:.2f} for Z-mem T2 planning")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
