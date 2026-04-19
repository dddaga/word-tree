"""Step 883: ΔW-proj component ablation T0 — paper ablation table.

MOTIVATION
==========
ΔW-proj routing has 4 distinct components:
  1. Projection direction: dw = normalize(W_h[i] - W_h[j]) — geometric direction
  2. Projection weighting: Z_nb * |proj_coeff| — magnitude gating
  3. Reflection signal: alpha_reflect=0.5 — momentum from prior activation gap
  4. Theta gating: Z_fwd = relu(Z - theta_pos) — per-neuron threshold

Each component needs a justification row in the paper ablation table.
Which components are load-bearing? Which can be removed with minimal loss?

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw     : standard ΔW-proj (all components) — canonical baseline
  A_sign     : proj_coeff.clamp(0) instead of .abs() — directional-only routing
  B_no_ref   : alpha_reflect=0 — remove reflection signal
  C_no_theta : theta forced to 0 — remove per-neuron threshold
  D_rand_dir : random unit vectors replace W_pos differences — no geometry

SUCCESS: D_rand_dir << Ref → geometric direction is essential (paper claim)
SUCCESS: B_no_ref < Ref − 0.3pp → reflection is load-bearing
SUCCESS: A_sign ≈ Ref → directional routing equally good (efficiency gain possible)
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
parser.add_argument("--configs", default="Ref_dw,A_sign,B_no_ref,C_no_theta,D_rand_dir")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step883_dwproj_ablation_t0_seed{SEED}__{SLOT}.json"


class SGNNET_DeltaW_Ablation(nn.Module):
    """ΔW-proj with configurable ablation of each component."""
    def __init__(self, resonant, mode: str = "standard"):
        super().__init__()
        self.m = resonant
        self.mode = mode

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]

        # Component 1: projection direction
        if self.mode == "D_rand_dir":
            dw = F.normalize(torch.randn(1, W_h.shape[0], conn_hh.shape[1], W_h.shape[1],
                                         device=W_h.device), dim=-1)
        else:
            dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

        # Component 3: theta gating
        if self.mode == "C_no_theta":
            theta_pos = torch.zeros(1, 1, 1, device=Z.device)
        else:
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Component 4: reflection coefficient
        alpha_ref = 0.0 if self.mode == "B_no_ref" else ALPHA_REFLECT

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)

            # Component 2: projection weighting
            if self.mode == "A_sign":
                Z_nb = Z_nb * proj_coeff.clamp(0)    # directional-only
            else:
                Z_nb = Z_nb * proj_coeff.abs()         # standard magnitude gating

            Z_ref = alpha_ref * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    ("standard",  "standard ΔW-proj — all components active"),
    "A_sign":    ("A_sign",    "proj_coeff.clamp(0) — directional-only routing"),
    "B_no_ref":  ("B_no_ref",  "alpha_reflect=0 — no reflection signal"),
    "C_no_theta":("C_no_theta","theta=0 — no per-neuron threshold"),
    "D_rand_dir":("D_rand_dir","random unit vectors — no W_pos geometry"),
}


def make_model(mode: str) -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW_Ablation(resonant, mode=mode)


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
    print(f"step883 — ΔW-proj component ablation T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        mode, desc = CONFIGS[key]
        model = make_model(mode)
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
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "mode": mode, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 883 SUMMARY — ΔW-proj ablation table")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'best':>7} {'Δ_vs_Ref':>10}  component removed")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "    (ref)"
        print(f"  {k:<12} {r['best']:>7.4f} {dv}  {r['label']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
