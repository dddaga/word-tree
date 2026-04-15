"""Step 730: ΔW projection × K_hh=4 compound at N=2048 Tier-1.

MOTIVATION:
  ΔW proj (K_hh=2): +1.61pp T1, +1.56pp T2 @ 0.98M FLOPs (step403/706)
  K_hh=4 scratch Tier-2: 95.87% (+0.35pp vs K_hh=2 T2) @ 3.15M FLOPs (step185)

QUESTION: Are ΔW proj and K_hh=4 orthogonal? If so, compound gives ~+2pp.
  - K_hh=4 adds graph diversity (more edges per neuron)
  - ΔW proj modifies signal routing per edge (alignment-based scaling)
  - Mechanism axes: topology (K_hh) vs signal-routing (ΔW proj)
  → HYPOTHESIS: orthogonal → compound gain ≈ sum of individual gains

Prediction:
  Ref (K_hh=4): ~95.87% (T2 result; T1 ≈ 93.96% step182)
  A_proj (K_hh=4 + ΔW proj): ~96.5-97.0%?
  → If ≥+1pp over K_hh=4 Ref → compound works, push T2

WARNING: step704 showed ΔW proj hurts at N=4096 where neurons are densely packed on S^{D-1}.
  At N=2048 K_hh=4, the graph is denser than K_hh=2. Does this push ΔW proj toward its
  breakdown regime? If K_hh=4 reduces per-edge ΔW gain significantly → compound fails.

Controls vs step403 (ΔW proj K_hh=2):
  K_hh: 2 → 4
  FLOPs: 0.98M → 3.15M (not efficiency config, but tests compounding)

Tier-1 (75ep, 50% data).
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
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)
EPOCHS = args.epochs
BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 4; K_IN = 25; K_ITER = 5   # K_hh=4 vs K_hh=2 in step403
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step730_proj_khh4_compound.json"


class SGNNET_DeltaAH(nn.Module):
    """ΔW projection: Z_nb scaled by |alignment with relational axis|."""
    def __init__(self, base):
        super().__init__()
        self.m = base
    @property
    def W_pos(self): return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()
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
            Z_nb = Z_nb * (Z_nb * dw).sum(-1, keepdim=True).abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _build(seed, delta=False):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    sw = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                           K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                           n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(sw, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_DeltaAH(res) if delta else SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


def main():
    import math
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else ["Ref", "A_proj"]
    print(f"\n{'='*70}")
    print(f"Step 730 — ΔW proj × K_hh=4 compound at N=2048 Tier-1")
    print(f"N={N} D={D} K_hh={K_HH} FLOPs={FLOPS/1e6:.3f}M  Device={DEVICE}")
    print(f"ΔW proj K_hh=2 Ref: 95.75% T1 (step403). K_hh=4 T1 Ref: 93.96% (step182).")
    print(f"QUESTION: does ΔW proj gain (~+1.6pp) compound with K_hh=4 diversity?")
    print(f"{'='*70}")
    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    tr = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
                                     batch_size=BATCH, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    results = {}
    for key in run_keys:
        print(f"\n{'─'*50}\n{key}\n{'─'*50}")
        model = _build(SEED, delta=(key != "Ref")).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **trainer_kwargs(N, n_epochs=EPOCHS)).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
                             if (m['epoch'] + 1) % 10 == 0 else None)
        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {"top1_best": best, "best_epoch": bep, "top1_history": top1h,
                        "elapsed_s": round(time.time()-t0, 1), "n_params": n_p,
                        "N": N, "D": D, "K_hh": K_HH, "flops": FLOPS}
        print(f"  → best={best:.4f} @ ep{bep}")
    ref = results.get("Ref", {}).get("top1_best", 0.)
    proj = results.get("A_proj", {}).get("top1_best", 0.)
    delta = proj - ref if all(k in results for k in ["Ref", "A_proj"]) else float("nan")
    ref_khh2_t2 = 0.9552   # step199 efficiency baseline (K_hh=2)
    proj_khh2_t1 = 0.9575  # step403 ΔW proj K_hh=2 T1
    print(f"\n{'='*70}")
    if not math.isnan(delta):
        print(f"K_hh=4: Ref={ref:.4f}  A_proj={proj:.4f}  Δ={delta:+.4f} ({delta*100:+.2f}pp)")
        print(f"K_hh=2: Ref={ref_khh2_t2:.4f} A_proj={proj_khh2_t1:.4f} Δ=+1.61pp (step403 T1)")
        if proj > proj_khh2_t1 + 0.005:
            print(f"  → COMPOUND WORKS: K_hh=4 + ΔW proj = {proj:.4f} > K_hh=2 + ΔW proj = {proj_khh2_t1:.4f}")
            print(f"  → Advance to T2 (step731)")
        elif delta > 0.01:
            print(f"  → ΔW proj still helps K_hh=4 (+{delta*100:.2f}pp) but compound < K_hh=2 additive sum")
            print(f"  → Partial compounding (topology × signal routing partially orthogonal)")
        else:
            print(f"  → COMPOUND FAILS: ΔW proj gain vanishes at K_hh=4. Dense graph breaks proj selectivity.")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")

if __name__ == "__main__":
    main()
