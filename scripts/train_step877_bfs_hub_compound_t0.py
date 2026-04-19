"""Step 877: BFS + Hub compound T0 — same signal-path interaction test.

MOTIVATION
==========
Both BFS (step873 T1 running) and Hub (step872 VIABLE +0.18pp) modify
the aggregation step. They share the signal path:
  Hub:  adds alpha * Z.mean() to Z_nb.sum()  (global broadcast)
  BFS:  replaces Z_nb with top-M broadcaster projections (sparse global)

COMPOUNDING RULE: mechanisms on the same signal path may cancel.
Hub injects global mean → "soft" broadcast from all nodes
BFS injects hard top-M broadcast → "hard" sparse broadcast

HYPOTHESIS: They encode the same intent (global context) via different
mechanisms → likely redundant at best, cancelling at worst.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw    : standard ΔW-proj (control)
  A_bfs32   : BFS M=32 only
  B_hub005  : hub alpha=0.05 only
  C_compound: BFS M=32 + hub alpha=0.05

SUCCESS: C_compound > max(A, B) + 0.1pp → genuinely orthogonal
CANCEL:  C_compound < min(A, B) → same path confirmed, don't compound
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
parser.add_argument("--configs", default="Ref_dw,A_bfs32,B_hub005,C_compound")
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
OUT_PATH = ROOT / "results" / f"train_step877_bfs_hub_compound_t0_seed{SEED}__{SLOT}.json"

STEP868_DW_T0 = 0.9396   # Ref T0 reference


class SGNNET_BFSHub(nn.Module):
    """ΔW-proj with optional BFS broadcasting and Hub global mean.

    bfs_m=0, alpha_hub=0 → standard ΔW-proj.
    """
    def __init__(self, resonant, bfs_m: int = 0, alpha_hub: float = 0.0):
        super().__init__()
        self.m = resonant
        self.bfs_m = bfs_m
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
            if self.bfs_m > 0:
                norms = Z_fwd.norm(dim=-1)
                M = min(self.bfs_m, Z_fwd.size(1))
                top_idx = norms.topk(M, dim=1).indices
                Z_bc = Z_fwd.gather(
                    1, top_idx.unsqueeze(-1).expand(-1, -1, Z_fwd.size(-1)))
                W_bc = W_h[top_idx]
                W_recv = W_h.unsqueeze(0).unsqueeze(0)
                dw_bc = F.normalize(W_recv - W_bc.unsqueeze(2), dim=-1)
                Z_bc_exp = Z_bc.unsqueeze(2)
                proj = (Z_bc_exp * dw_bc).sum(-1, keepdim=True) * dw_bc
                Z_agg = proj.sum(1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb_agg = proj.sum(1) / Z_agg
            else:
                Z_nb = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb_agg = (Z_nb * proj_coeff.abs()).sum(2)

            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            hub_signal = (self.alpha_hub * Z_fwd.mean(dim=1, keepdim=True).expand_as(Z_fwd)
                          if self.alpha_hub > 0 else 0.0)
            Z = F.normalize(
                (Z_nb_agg + Z_ref + hub_signal).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    (0,  0.0,  "ΔW-proj only (control)"),
    "A_bfs32":   (32, 0.0,  "BFS M=32 only"),
    "B_hub005":  (0,  0.05, "hub alpha=0.05 only"),
    "C_compound":(32, 0.05, "BFS M=32 + hub alpha=0.05"),
}


def make_model(bfs_m: int, alpha_hub: float) -> nn.Module:
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
    return SGNNET_BFSHub(resonant, bfs_m=bfs_m, alpha_hub=alpha_hub)


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
    print(f"step877 — BFS + Hub compound T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  T0 dw ref (step868): {STEP868_DW_T0:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        bfs_m, alpha_hub, desc = CONFIGS[key]
        model = make_model(bfs_m, alpha_hub)
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
                                   flush=True) if (m['epoch']+1) % 10 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP868_DW_T0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "bfs_m": bfs_m, "alpha_hub": alpha_hub, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 877 SUMMARY — BFS + Hub compound T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'M':>4} {'hub':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<14} {r['bfs_m']:>4} {r['alpha_hub']:>6.3f} {r['best']:>7.4f} {dv}")
    cmp = results.get("C_compound", {})
    a = results.get("A_bfs32", {})
    b = results.get("B_hub005", {})
    if cmp and a and b:
        best_solo = max(a.get("best", 0), b.get("best", 0))
        gap = cmp["best"] - best_solo
        if gap >= 0.001:
            verdict = f"SYNERGY — compound advances ({gap*100:+.2f}pp over solo)"
        elif gap >= -0.005:
            verdict = f"NEUTRAL — no clear benefit to compounding"
        else:
            verdict = f"CANCEL — BFS+Hub on same path, DO NOT compound"
        print(f"\n  BFS+Hub: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
