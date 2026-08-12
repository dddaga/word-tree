"""Step 905: CIFAR-100 Rigor Baseline T1 (75ep, 50% data).

MOTIVATION
==========
All prior SGNNET accuracy claims are on Imagenette (10-class, VGG16 pool5 features).
step614 ran SGNNET on CIFAR-100 at N=4096/8192 but WITHOUT Linear/MLP baselines —
making those numbers uninterpretable. step862 showed CIFAR-10 gap: −5.55pp vs Linear.

This script establishes rigorous multi-class baselines on CIFAR-100 (100-class,
same VGG16 pool5 features, 50K/10K train/val) to answer:

  1. Does SGNNET beat or trail Linear on 100-class classification?
  2. Does the gap widen at 100 classes vs 10 (Imagenette)?
  3. What is the parameter-accuracy Pareto on this harder benchmark?

CONFIGS (T1, 75ep, 50% data, seed=42)
  Linear          25088 → 100, no hidden. The floor.
  MLP_256         25088 → 256 → 100. 1-hidden, ~6.4M params.
  MLP_512         25088 → 512 → 100. 1-hidden, ~12.8M params.
  SGNNET_can      N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj. Standard baseline.
  SGNNET_N4096    N=4096, D=16. Bigger graph — matches step614 scale.

ADVANCE RULE
============
  SGNNET_can within ±2pp of Linear → multi-dataset generalization holds.
  SGNNET_can > Linear → strong paper claim; advance CIFAR-100 to T2 (step908).
  SGNNET_can < Linear by >5pp → gap widens at 100 classes; paper caveat required.

KEY READS
=========
  SGNNET_can vs Linear: primary gap measurement.
  SGNNET_N4096 vs SGNNET_can: N scaling on harder benchmark.
  params/FLOPs vs accuracy Pareto: does efficiency case hold at 100 classes?
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=75)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store_cifar100.h5")
parser.add_argument("--configs",  default="Linear,MLP_256,MLP_512,SGNNET_can,SGNNET_N4096")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 100
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step905_cifar100_rigor_t1_seed{SEED}__{SLOT}.json"

# Config spec: (N_hidden, D, K_hh, K_iter, model_type)
# model_type: "linear", "mlp", "sgnnet"
CONFIG_SPEC = {
    "Linear":       (0,    0,  0, 0, "linear", 0),
    "MLP_256":      (256,  0,  0, 0, "mlp",    0),
    "MLP_512":      (512,  0,  0, 0, "mlp",    0),
    "SGNNET_can":   (2048, 16, 2, 5, "sgnnet", 25),
    "SGNNET_N4096": (4096, 16, 2, 5, "sgnnet", 15),
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor, n: int) -> torch.Tensor:
    W_h = W_pos[:n]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_IN, N_OUT)
        # Expose W_pos stub for Trainer compatibility
        self.W_pos = nn.Parameter(torch.zeros(1))

    def tick_epoch(self): pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float().view(x.size(0), -1))


class MLPModel(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT),
        )
        self.W_pos = nn.Parameter(torch.zeros(1))

    def tick_epoch(self): pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float().view(x.size(0), -1))


class SGNNET_DeltaW(nn.Module):
    """Standard SGNNET with ΔW-proj (step898 Ref_dw architecture)."""

    def __init__(self, resonant: SGNNET_Resonant, n: int):
        super().__init__()
        self.m = resonant
        self.n = n

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw        = _dw_proj(self.m.W_pos, conn_hh, self.n)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_dw  = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_dw + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_model(key: str) -> nn.Module:
    n, d, k_hh, k_iter, mtype, k_in = CONFIG_SPEC[key]
    torch.manual_seed(SEED)
    if mtype == "linear":
        return LinearModel()
    if mtype == "mlp":
        return MLPModel(n)
    # sgnnet
    K_r  = max(1, k_hh // 4); K_l = k_hh - K_r
    ng   = max(8, n // 8)
    base = SGNNET_SmallWorld(
        N_hidden=n, N_out=N_OUT, D=d, N_in=N_IN,
        K_in=k_in, K_iter=k_iter, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_DeltaW(resonant, n)


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
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step905 — CIFAR-100 Rigor Baseline T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N_IN={N_IN}  N_OUT={N_OUT}  (100-class, VGG16 pool5 features)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    linear_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        n, d, k_hh, k_iter, mtype, k_in = CONFIG_SPEC[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if mtype == "sgnnet":
            routing_macs = n * k_iter * k_hh * d * 2
            mac_str = f"  routing_MACs={routing_macs/1e6:.2f}M"
        else:
            routing_macs = 0
            mac_str = ""
        print(f"{'─'*60}")
        print(f"{key}: type={mtype}  params={n_p:,}{mac_str}")

        # Non-SGNNET models: use standard cross-entropy with AdamW on all params
        if mtype in ("linear", "mlp"):
            opt    = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad and p.numel() > 1],
                lr=1e-3, weight_decay=1e-4,
            )
            sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            model  = model.to(DEVICE)
            history = []
            t0 = time.time()
            for ep in range(EPOCHS):
                model.train()
                total_loss = 0.0; n_batches = 0
                for xb, _, yb in tr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward(); opt.step()
                    total_loss += loss.item(); n_batches += 1
                sched.step()
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for xb, _, yb in va:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total   += yb.size(0)
                top1 = correct / total
                history.append({"val_top1": top1, "train_loss": total_loss / n_batches,
                                 "lr": sched.get_last_lr()[0]})
                print(f"  e{ep+1:3d}  loss={total_loss/n_batches:.4f}  "
                      f"top1={top1:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
        else:
            kw = trainer_kwargs(n, n_epochs=EPOCHS)
            t0 = time.time()
            history = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw).train(
                n_epochs=EPOCHS,
                log_fn=lambda m: print(
                    f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                    f"task={m.get('task_loss', m['train_loss']):.4f}  "
                    f"safety={m.get('safety_loss', 0.0):.4f}  "
                    f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}",
                    flush=True,
                ))

        elapsed = time.time() - t0
        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Linear":
            linear_acc = best
        delta = best - (linear_acc if linear_acc is not None else 0.0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "model_type": mtype, "n_params": n_p,
            "routing_macs": routing_macs,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_linear": round(delta, 4) if linear_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 905 SUMMARY — CIFAR-100 Rigor Baseline T1")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'type':>6} {'params':>10} {'MACs(M)':>8} "
          f"{'best':>7} {'Δ_vs_Linear':>13}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_linear"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (baseline)"
        if k == "Linear":
            v = "(baseline)"
        elif d is None:
            v = "—"
        else:
            v = ("PAPER_CLAIM" if d >= 0.000
                 else "WITHIN_2pp"  if d >= -0.020
                 else "GAP_5pp"     if d >= -0.050
                 else "WIDE_GAP")
        print(f"  {k:<14} {r['model_type']:>6} {r['n_params']:>10,} "
              f"{r['routing_macs']/1e6:>8.2f} {r['best']:>7.4f} {dstr:>13}  {v}")

    if linear_acc is not None and "SGNNET_can" in results:
        sgnnet_best = results["SGNNET_can"]["best"]
        gap = (sgnnet_best - linear_acc) * 100
        print(f"\n  KEY READ: SGNNET_can vs Linear gap = {gap:+.2f}pp")
        if gap >= 0:
            print(f"  → SGNNET generalises to 100-class: strong paper claim.")
            print(f"  → Advance to T2 (step908): 150ep/100% data.")
        elif gap >= -2.0:
            print(f"  → Gap ≤2pp: competitive with far fewer params. Paper caveat needed.")
        elif gap >= -5.0:
            print(f"  → Gap 2–5pp: widened at 100 classes. Rigor caveat in paper.")
        else:
            print(f"  → Gap >5pp: SGNNET struggles on hard benchmarks. "
                  f"Narrow paper scope to Imagenette.")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
