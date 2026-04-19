"""Step 873: BFS M=32 routing T1 — does +0.15pp T0 hold at calibration?

MOTIVATION
==========
step864 T0 (20ep, 5060ti_cpu):
  Ref       : 91.80% (K_hh=2 standard routing)
  F_bfs_M32 : 91.95% (+0.15pp ADVANCES)  top-32 broadcasters
  H_bfs_M16 : 91.77% (-0.03pp neutral)
  G_bfs_M64 : 91.82% (+0.02pp neutral)

BFS M=32 broadcasts from the 32 most active nodes per K_iter step.
1.6% of N=2048 nodes act as hubs. Potentially significant FLOP reduction
if broadcasting replaces all-to-K connections (32 vs 2 per node, but
only 32 sources — different routing geometry).

Paper angle: if BFS advances, it validates sparse broadcasting as an
efficiency mechanism orthogonal to ΔW-proj.

CONFIGS (T1: 75ep, 50% data, seed=42, 5060ti_cuda)
  Ref_dw     : standard ΔW-proj (K_hh=2)
  A_bfs_m32  : BFS top-32 + ΔW-proj

SUCCESS: A_bfs_m32 ≥ Ref + 0.2pp → BFS advances to T2
VIABLE:  A_bfs_m32 ≥ Ref + 0.0pp → advances to T2
KILL:    A_bfs_m32 < Ref - 0.1pp → T0 was noise
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
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_bfs_m32")
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
OUT_PATH = ROOT / "results" / f"train_step873_bfs_m32_t1_seed{SEED}__{SLOT}.json"

STEP858_DW_T1 = 0.9524   # ΔW-proj T1 reference (step858 B_dwproj)
STEP864_BFS32_T0 = 0.9195  # F_bfs_M32 T0 (step864)


class SGNNET_BFS(nn.Module):
    """ΔW-proj routing with BFS top-M broadcaster selection.

    At each K_iter step, only the top-M most active nodes broadcast.
    Each receiving node gathers from all M broadcasters (not just K_hh neighbors).
    M=0 → identical to standard ΔW-proj (no BFS).
    """
    def __init__(self, resonant, bfs_m: int = 0):
        super().__init__()
        self.m = resonant
        self.bfs_m = bfs_m

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
                # Select top-M active broadcasters
                norms = Z_fwd.norm(dim=-1)          # [B, N]
                M = min(self.bfs_m, Z_fwd.size(1))
                top_idx = norms.topk(M, dim=1).indices  # [B, M]
                Z_bc = Z_fwd.gather(
                    1, top_idx.unsqueeze(-1).expand(-1, -1, Z_fwd.size(-1)))  # [B, M, D]
                # ΔW projection in broadcaster space
                W_bc = W_h[top_idx]     # [B, M, D]
                W_recv = W_h.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
                dw_bc = F.normalize(
                    W_recv - W_bc.unsqueeze(2), dim=-1)  # [B, M, N, D]
                Z_bc_exp = Z_bc.unsqueeze(2)             # [B, M, 1, D]
                proj = (Z_bc_exp * dw_bc).sum(-1, keepdim=True) * dw_bc  # [B, M, N, D]
                Z_agg = proj.sum(1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb_bfs = proj.sum(1) / Z_agg          # [B, N, D] normalized
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z = F.normalize((Z_nb_bfs + Z_ref).clamp(-10, 10), dim=-1)
            else:
                Z_nb = Z_fwd[:, conn_hh, :]
                proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_nb = Z_nb * proj_coeff.abs()
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    (0,  "K_hh=2 standard ΔW-proj (T1 baseline)"),
    "A_bfs_m32": (32, "BFS top-32 broadcasters + ΔW-proj (T0 winner)"),
}


def make_model(bfs_m: int) -> nn.Module:
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
    return SGNNET_BFS(resonant, bfs_m=bfs_m)


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
    print(f"step873 — BFS M=32 routing T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  T0 ref (step864): F_bfs_M32={STEP864_BFS32_T0:.4f} (+0.15pp)")
    print(f"  T1 dw ref (step858): {STEP858_DW_T1:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        bfs_m, desc = CONFIGS[key]
        model = make_model(bfs_m)
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
                                   flush=True) if (m['epoch']+1) % 15 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP858_DW_T1)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "bfs_m": bfs_m, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 873 SUMMARY — BFS M=32 T1")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'bfs_m':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<12} {r['bfs_m']:>6} {r['best']:>7.4f} {dv}")
    bfs = results.get("A_bfs_m32", {})
    ref = results.get("Ref_dw", {})
    if bfs and ref:
        gap = bfs["best"] - ref["best"]
        if gap >= 0.002:
            verdict = "STRONG — BFS advances to T2"
        elif gap >= 0.0:
            verdict = "VIABLE — BFS advances to T2"
        else:
            verdict = "KILL — T0 was noise"
        print(f"\n  BFS M=32 T1: {gap*100:+.2f}pp → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
