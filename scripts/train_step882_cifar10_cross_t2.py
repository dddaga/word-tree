"""Step 882: CIFAR-10 cross-dataset T2 — generalization test.

MOTIVATION
==========
All SGNNET experiments to date use Imagenette (9,296 train samples).
CIFAR-10 (50,000 train, VGG16 pool5 features 25088-dim) provides a
cross-dataset generalization test at 5× scale.

Baselines:
  Ref_linear : nn.Linear(25088→10) — 250,890 params — simplest probe
  A_sgnnet   : SGNNET ΔW-proj K_hh=2, N=2048, D=16 — 34,976 params (7.2× fewer)

CLAIM: SGNNET achieves competitive accuracy with 7× fewer params than Linear.
SUCCESS: A_sgnnet within 2pp of Ref_linear → cross-dataset generalization holds
CONCERN: gap > 2pp → model may be over-fit to Imagenette inductive structure

CONFIGS (T2: 150ep, 100% data, seed=42)
Device: mini_mps — canonical params (34,976)
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
parser.add_argument("--configs", default="Ref_linear,A_sgnnet")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 10
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step882_cifar10_cross_t2_seed{SEED}__{SLOT}.json"


class LinearProbe(nn.Module):
    """Plain linear probe — compatible with Trainer's W_pos param group."""
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.W_pos = self.fc.weight

    @property
    def W_phase(self): return None

    def tick_epoch(self): pass

    def forward(self, x):
        return self.fc(x)


class SGNNET_DeltaW(nn.Module):
    """ΔW-proj routing — standard paper model."""
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
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaW(resonant)


CONFIGS = {
    "Ref_linear": ("linear", "nn.Linear(25088→10) — 250,890 params (upper bound)"),
    "A_sgnnet":   ("sgnnet", "SGNNET ΔW-proj K_hh=2, N=2048, D=16 — 34,976 params"),
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                          pin_memory=False)

    print(f"\n{'='*70}")
    print(f"step882 — CIFAR-10 cross-dataset T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  CIFAR-10: train={len(tr.dataset)}  val={len(va.dataset)}")
    print(f"  (vs Imagenette: train=9296 — {len(tr.dataset)/9296:.1f}× scale)")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        kind, desc = CONFIGS[key]

        if kind == "linear":
            torch.manual_seed(SEED)
            model = LinearProbe(N_IN, N_OUT)
        else:
            model = make_sgnnet()

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N if kind == "sgnnet" else N_IN, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 25 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_linear": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "kind": kind, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 882 SUMMARY — CIFAR-10 cross-dataset T2")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'params':>10} {'best':>7} {'Δ_vs_linear':>12}")
    for k, r in results.items():
        dv = f"{r['delta_vs_linear']*100:>+11.2f}pp" if r['delta_vs_linear'] is not None else "      (ref)"
        print(f"  {k:<12} {r['n_params']:>10,} {r['best']:>7.4f} {dv}")
    sgnnet = results.get("A_sgnnet", {})
    linear = results.get("Ref_linear", {})
    if sgnnet and linear:
        gap = sgnnet["best"] - linear["best"]
        param_ratio = linear["n_params"] / sgnnet["n_params"]
        if gap >= -0.02:
            verdict = f"STRONG — SGNNET within 2pp at {param_ratio:.1f}× fewer params (cross-dataset CONFIRMED)"
        elif gap >= -0.05:
            verdict = f"VIABLE — SGNNET within 5pp at {param_ratio:.1f}× fewer params (cross-dataset VIABLE)"
        else:
            verdict = f"WEAK — gap {gap*100:.1f}pp — SGNNET may be over-fit to Imagenette structure"
        print(f"\n  {gap*100:+.2f}pp @ {param_ratio:.1f}× param reduction → {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
