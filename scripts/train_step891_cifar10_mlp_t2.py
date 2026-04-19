"""Step 891: CIFAR-10 cross-dataset MLP matched-params comparison T2.

MOTIVATION
==========
step882 T2 showed SGNNET=80.69% vs Linear=86.24% (Δ=-5.55pp) on CIFAR-10.
Missing: does a matched-params MLP do better or worse than SGNNET?

At N_in=25088, N_out=10:
  MLP_h1:  h=1  → 25,099*1 + 10 = 25,109 params  (fewer than SGNNET)
  MLP_h2:  h=2  → 25,099*2 + 10 = 50,208 params  (more than SGNNET)

Paper claim (if SGNNET > MLP_h1 or MLP_h2):
  "SGNNET (34,976 params) outperforms comparably-sized MLP on CIFAR-10,
   demonstrating graph structure adds value at matched parameter budget."

Paper claim (if MLP_h1 > SGNNET):
  SGNNET -5.55pp gap is NOT a params-budget problem — architecture specific.
  Honest negative to report.

CONFIGS (T2: 150ep, 100% data, seed=42)
  Ref_linear : Linear 25088→10  (250,890 params — accuracy ceiling)
  MLP_h1     : Linear→ReLU→Linear h=1 (25,109 params — below SGNNET)
  MLP_h2     : Linear→ReLU→Linear h=2 (50,208 params — above SGNNET)
  Ref_SGNNET : ΔW-proj N=2048 D=16 K_in=25 K_hh=2 (34,976 params)

Context from step882 T2: Linear=86.24%, SGNNET=80.69%.
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
parser.add_argument("--data",    default="data/store_cifar10.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step891_cifar10_mlp_t2_seed{SEED}__{SLOT}.json"

STEP882_LINEAR = 0.8624
STEP882_SGNNET = 0.8069


class LinearProbe(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.W_pos = self.fc.weight
    @property
    def W_phase(self): return None
    def tick_epoch(self): pass
    def forward(self, x): return self.fc(x)


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.W_pos = self.fc1.weight
    @property
    def W_phase(self): return None
    def tick_epoch(self): pass
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


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


def make_sgnnet() -> nn.Module:
    torch.manual_seed(SEED)
    K_r = 1; K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW(resonant)


CONFIGS = [
    ("Ref_linear", "linear", None),
    ("MLP_h1",     "mlp",    1),
    ("MLP_h2",     "mlp",    2),
    ("Ref_SGNNET", "sgnnet", None),
]


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    n_p_check = sum(p.numel() for p in make_sgnnet().parameters() if p.requires_grad)
    if n_p_check != 34976:
        print(f"ERROR: non-canonical params={n_p_check}, expected 34976. Abort."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED, pin_memory=False)

    print(f"\n{'='*70}")
    print(f"step891 — CIFAR-10 MLP matched-params T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  seed={SEED}")
    print(f"  step882 context: Linear=86.24%, SGNNET=80.69% (Δ=-5.55pp)")
    print(f"  Q: does matched-params MLP beat SGNNET on CIFAR-10?")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    results = {}
    ref_acc = None

    for key, kind, h in CONFIGS:
        torch.manual_seed(SEED)
        if kind == "linear":
            model = LinearProbe(N_IN, N_OUT)
        elif kind == "mlp":
            model = MLP(N_IN, h, N_OUT)
        else:
            model = make_sgnnet()

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: params={n_p:,}")

        kw = trainer_kwargs(N if kind == "sgnnet" else N_IN, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h_.get("val_top1", 0.0) if isinstance(h_, dict) else float(h_) for h_ in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_linear": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP882_LINEAR)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "kind": kind, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    sgnnet_r = results.get("Ref_SGNNET", {})
    mlp_h1   = results.get("MLP_h1", {})
    mlp_h2   = results.get("MLP_h2", {})
    print(f"\n{'='*70}")
    print(f"STEP 891 SUMMARY — CIFAR-10 MLP matched-params T2")
    print(f"{'='*70}")
    for k, r in results.items():
        dv = f"{r['delta_vs_linear']*100:>+8.2f}pp"
        print(f"  {k:<12} {r['n_params']:>8,}  {r['best']:>7.4f}  {dv}")
    if sgnnet_r and mlp_h1:
        gap_h1 = sgnnet_r["best"] - mlp_h1["best"]
        gap_h2 = sgnnet_r["best"] - mlp_h2["best"] if mlp_h2 else None
        if gap_h1 >= 0:
            verdict = f"SGNNET > MLP_h1 (+{gap_h1*100:.2f}pp) — graph structure adds value at matched params"
        else:
            verdict = f"MLP_h1 > SGNNET ({gap_h1*100:.2f}pp) — SGNNET architecture-specific gap, not params-budget"
        print(f"\n  SGNNET vs MLP_h1: {gap_h1*100:+.2f}pp → {verdict}")
        if gap_h2 is not None:
            print(f"  SGNNET vs MLP_h2: {gap_h2*100:+.2f}pp")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
