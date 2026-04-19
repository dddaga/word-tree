"""Step 880: K_hh=3 probe T0 — upper bound of routing density curve.

MOTIVATION
==========
We now have two data points on the K_hh efficiency curve:
  K_hh=1: -0.48pp (step865 T0 VIABLE) — 50% fewer routing MACs
  K_hh=2: 0pp (baseline) — standard
  K_hh=4: already killed in step285 (K_iter=4 routing test, different context)

K_hh=3 (K_local=2, K_random=1) adds 50% MORE routing vs K_hh=2:
  - More local structure (K_local=2 vs 1)
  - Same random shortcuts (K_random=1)
  - 1.5× routing MACs

HYPOTHESIS: K_hh=3 provides diminishing returns — the curve is:
  K_hh=1 (-0.5pp) < K_hh=2 (0) ≈ K_hh=3 (<+0.3pp)
  If confirmed, K_hh=2 is the efficient sweet spot. If K_hh=3 >> K_hh=2,
  the paper must justify using K_hh=2.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw   : K_hh=2 (K_local=1, K_random=1) — standard
  A_khh3   : K_hh=3 (K_local=2, K_random=1) — 50% more routing ops

SUCCESS (diminishing): A_khh3 ≤ Ref + 0.3pp → K_hh=2 is Pareto-efficient sweet spot
SUCCESS (more is better): A_khh3 ≥ Ref + 0.5pp → need to re-examine K_hh choice
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
parser.add_argument("--configs", default="Ref_dw,A_khh3")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step880_khh3_t0_seed{SEED}__{SLOT}.json"

STEP868_DW_T0 = 0.9396   # Ref_dw T0 baseline (K_hh=2)
STEP865_A_T0  = 0.9325   # K_hh=1 T0 result (-0.48pp)


class SGNNET_DeltaW(nn.Module):
    """Standard ΔW-proj routing for arbitrary K_hh."""
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


CONFIGS = {
    "Ref_dw": (2, 1, 1, "K_hh=2 (K_local=1, K_random=1) — standard"),
    "A_khh3": (3, 2, 1, "K_hh=3 (K_local=2, K_random=1) — 50% more routing ops"),
}


def make_model(k_hh: int, k_local: int, k_random: int) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=k_local, K_random=k_random,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
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
    print(f"step880 — K_hh=3 probe T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  K_hh curve: K_hh=1={STEP865_A_T0:.4f}(-0.48pp), K_hh=2={STEP868_DW_T0:.4f}(ref)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        k_hh, k_local, k_random, desc = CONFIGS[key]
        model = make_model(k_hh, k_local, k_random)
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
            "k_hh": k_hh, "k_local": k_local, "k_random": k_random,
            "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 880 SUMMARY — K_hh=3 probe T0")
    print(f"{'='*70}")
    print(f"  K_hh routing curve (T0 20ep/50%):")
    print(f"    K_hh=1: {STEP865_A_T0:.4f} (-0.48pp, 50% fewer MACs)")
    print(f"    K_hh=2: (ref, standard)")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"    K_hh={r['k_hh']}: {r['best']:>7.4f} {dv}")
    khh3 = results.get("A_khh3", {})
    ref = results.get("Ref_dw", {})
    if khh3 and ref:
        gap = khh3["best"] - ref["best"]
        if gap >= 0.005:
            verdict = "ADVANCES — K_hh=3 significantly better, re-examine default (needs T1)"
        elif gap >= -0.005:
            verdict = "NEUTRAL — K_hh=2 is Pareto sweet spot (diminishing returns confirmed)"
        else:
            verdict = "WORSE — more local edges hurt, K_hh=2 confirmed optimal"
        print(f"\n  K_hh=3 T0: {gap*100:+.2f}pp → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
