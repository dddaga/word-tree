"""Step 704: ΔW projection N-scaling test at N=4096 (CUDA).

MOTIVATION
==========
step403 confirmed ΔW projection (no AH) = 95.75% at N=2048 (+1.61pp Tier-1).
Question: does the mechanism scale to N=4096?

Known reference (N=4096 AH=1.0): 97.17% @ 1.97M FLOPs (step205/209).
If ΔW proj gives +1-2pp at N=4096, this would become the new efficiency record.
If neutral, it confirms ΔW proj is N=2048-specific.

MECHANISM
=========
ΔW = W_pos[receiver] - W_pos[sender] (relational axis between neurons).
Each neighbour signal Z_nb is weighted by |Z_nb · ΔW_unit| — selects signals
that are moving in the relational direction. No AH suppression (redundant).

CONFIGS (N=4096, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0 scout)
  Ref    : AH α=1.0 (standard step199/step203 baseline)
  A_proj : ΔW projection, no AH (step403 Tier-1 winner at N=2048)
  B_ah   : AH only, α=1.0 (sanity check — should match Ref closely)
"""
from __future__ import annotations
import argparse, json, sys, time
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import H5Dataset

parser = argparse.ArgumentParser(description="Step 704: ΔW projection at N=4096")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,A_proj). Empty = all.")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS    = args.epochs
BATCH     = 128; SEED = 42; FRAC_DATA = 0.5
N = 4096; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step704_delta_n4096.json"


# ── ΔW projection model (copied from step403) ──────────────────────────────

class SGNNET_DeltaAH(nn.Module):
    """ΔW projection with optional AH suppression.

    Projects Z_nb onto ΔW = W_pos[receiver] - W_pos[sender] direction.
    alpha_ahebb=0: pure ΔW projection (step403 winner).
    """
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb: float = 0.0):
        super().__init__()
        self.m           = base
        self.alpha_ahebb = alpha_ahebb

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden
        W_h = self.m.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)

        supp_w = None
        if self.alpha_ahebb > 0:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                       ).unsqueeze(0).unsqueeze(-1)

        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]        # [N, K_hh, D]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                       # [B, N, K_hh, D]
            if supp_w is not None:
                Z_nb = Z_nb * supp_w
            proj_coeff  = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb        = Z_nb * proj_coeff.abs()
            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


# ── Model builder ────────────────────────────────────────────────────────────

def make_model(device, alpha_ahebb: float, use_delta: bool):
    torch.manual_seed(SEED)
    n_groups = max(8, N // 8)
    K_local  = max(1, K_HH - max(1, K_HH // 4))
    K_random = K_HH - K_local
    sw = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_local=K_local, K_random=K_random,
        n_groups=n_groups, K_iter=K_ITER,
        norm_mode="l2", encoding_mode="fourier",
    ).to(device)
    res = SGNNET_Resonant(
        base=sw, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    ).to(device)
    if use_delta:
        return SGNNET_DeltaAH(res, alpha_ahebb=alpha_ahebb).to(device)
    else:
        return SGNNET_AntiHebbian(base=res, alpha_ahebb=alpha_ahebb, variant="wpos").to(device)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A_proj"]
    run_keys = ([k.strip() for k in args.configs.split(",")]
                if args.configs else all_keys)

    labels = {
        "Ref":    "Ref: AH α=1.0 (step199/step203 baseline)",
        "A_proj": "A_proj: ΔW projection, no AH (step403 Tier-1 winner at N=2048)",
    }
    configs = {
        "Ref":    dict(alpha_ahebb=1.0, use_delta=False),
        "A_proj": dict(alpha_ahebb=0.0, use_delta=True),
    }

    print(f"\n{'='*70}")
    print(f"Step 704 — ΔW projection N-scaling test at N=4096 (Tier-0 scout)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)  Device={DEVICE}")
    print(f"Question: does ΔW proj (+1.61pp at N=2048) scale to N=4096?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    train_ds = H5Dataset(str(ROOT / "data/store.h5"), split="train")
    val_ds   = H5Dataset(str(ROOT / "data/store.h5"), split="val")
    n_train  = int(len(train_ds) * FRAC_DATA)
    g   = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    tr  = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, idx),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    va = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")
        cfg = configs[key]
        model = make_model(DEVICE, **cfg)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 5 == 0 else None
        ))
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1],
            "best_epoch": bep, "epochs_run": len(history),
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "N": N, "D": D,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    # Summary
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 704 SUMMARY — ΔW projection N-scaling")
    print(f"{'='*70}")
    print(f"  N=2048 known: Ref=93.96% (step203 T1), A_proj=95.75% (+1.79pp) (step403 T1)")
    print(f"  N=4096 this run:")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            d = r['top1_best'] - ref_best
            if d >= 0.010: verdict = " → SCALES — ΔW proj helps at N=4096"
            elif d >= -0.005: verdict = " → NEUTRAL at N=4096"
            else: verdict = " → KILLS at N=4096 (N-scale dependent)"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
