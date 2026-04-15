"""Step 601: 10x complexity test — does MLP_37 still beat SGNNET at 100 classes?

MOTIVATION
==========
step403b found MLP_37 = 97.71% on Imagenette (10 classes), beating SGNNET step199
(95.52%) at matched FLOPs (1.86M). User hypothesis: Imagenette is too easy a
problem; MLP_37's win disappears when class complexity scales.

PREDICTION (testable)
- MLP_37 has only 37 hidden units. At 100 classes, it must encode 100 class
  decision boundaries through a 37-dim bottleneck. Information-theoretic floor:
  37 dims encode log2(2^37) directions, but discriminating 100 classes needs
  ~log2(100) = 6.64 bits, well within capacity in principle. However, in
  practice MLP_37 trained on VGG features hits a hard ceiling because the
  hidden layer cannot represent class-specific feature combinations.
- SGNNET N=2048 D=16 has effective latent of 32,768 dims, accessed through
  iterative routing. At 100 classes it should retain its representational
  power because routing acts as a soft mixture-of-experts.

CONFIGS (CIFAR-100 VGG16 features, N_IN=25088, N_OUT=100, T1 75ep, full data)
  MLP_37   : Linear(25088→37→100)        — matches step403b's hidden size
  MLP_64   : Linear(25088→64→100)
  MLP_128  : Linear(25088→128→100)
  MLP_256  : Linear(25088→256→100)
  SGNNET_AH : N=2048 D=16 K_hh=2 K_iter=5 + AH (step199 config)
  SGNNET_DeltaProj : N=2048 D=16 K_hh=2 K_iter=5 + ΔW projection (step235-style)

Acceptance for "complexity rescue" hypothesis:
  (MLP_37 top1 on CIFAR-100) - (SGNNET_DeltaProj top1 on CIFAR-100) < 0
  AND |MLP_37 ImageNette - MLP_37 CIFAR-100| > |SGNNET_DP ImageNette - SGNNET_DP CIFAR-100|
  i.e. MLP_37 degrades MORE than SGNNET when class count goes 10×.

To run:
    # Phase 1: extract (one-time, ~30min)
    python -u scripts/extract_cifar100_vgg16_features.py --device mps
    # Phase 2: train all configs
    python -u scripts/train_step601_complexity_test.py
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--data",   default="data/store_cifar100.h5")
parser.add_argument("--configs", default="MLP_37,MLP_64,MLP_128,MLP_256,SGNNET_AH,SGNNET_DeltaProj")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 25088; N_OUT = 100         # CIFAR-100 has 100 classes (vs Imagenette's 10)
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step601_complexity_test_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_IN, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT))
    def forward(self, x): return self.net(x)


def build_sgnnet_ah():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
                          mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


class SGNNET_DeltaProj(nn.Module):
    """ΔW projection routing — step235-style, no AH."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = ALPHA_REFLECT

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh; N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:N_h]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_reflected = self.alpha_reflect * Z_reflected + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


# ─────────────────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(model, tr, va, epochs):
    model = model.to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        v = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": v})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={v:.4f}", flush=True)
    return history


def train_sgnnet(model, tr, va, epochs):
    model = model.to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
    return trainer.train(n_epochs=epochs, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch']+1) % 10 == 0 else None))


def main():
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: dataset {data_path} not found.")
        print(f"  Run first: python scripts/extract_cifar100_vgg16_features.py")
        sys.exit(1)

    print(f"Step 601 — Complexity test (CIFAR-100, 100 classes vs Imagenette's 10)")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  Hypothesis: MLP_37 collapses at 100 classes; SGNNET retains its routing advantage.")

    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}
    for key in keys:
        print(f"\n{'─'*60}\nConfig: {key}\n{'─'*60}")
        torch.manual_seed(SEED)
        if   key == "MLP_37":  model = MLPBaseline(37)
        elif key == "MLP_64":  model = MLPBaseline(64)
        elif key == "MLP_128": model = MLPBaseline(128)
        elif key == "MLP_256": model = MLPBaseline(256)
        elif key == "SGNNET_AH":        model = build_sgnnet_ah()
        elif key == "SGNNET_DeltaProj": model = SGNNET_DeltaProj()
        else: print(f"  skip unknown {key}"); continue

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Compute FLOPs estimate (per-sample, fp32)
        if key.startswith("MLP_"):
            h = int(key.split("_")[1])
            flops = 2 * h * (N_IN + N_OUT)
        else:
            # SGNNET routing MACs + readout
            flops = 3 * N * K_HH * D * K_ITER + 2 * N * N_OUT
        print(f"  params={n_p:,}  flops≈{flops/1e6:.2f}M")

        t0 = time.time()
        history = train_mlp(model, tr, va, EPOCHS) if key.startswith("MLP_") else train_sgnnet(model, tr, va, EPOCHS)
        top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        elapsed = time.time() - t0
        print(f"  → best={best:.4f} @ep{bep}  elapsed={elapsed:.0f}s")

        results[key] = {
            "config": key, "n_params": n_p, "flops": flops,
            "top1_best": best, "best_epoch": bep,
            "elapsed_s": round(elapsed, 1), "epochs": EPOCHS,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n========== STEP 601 SUMMARY (CIFAR-100 100-class) ==========")
    print(f"  {'Config':<20} {'params':>10} {'FLOPs':>10} {'top1':>8}")
    for k, r in results.items():
        print(f"  {k:<20} {r['n_params']:>10,} {r['flops']/1e6:>9.2f}M {r['top1_best']:>8.4f}")

    # Verdict
    if "MLP_37" in results and "SGNNET_DeltaProj" in results:
        d = results["SGNNET_DeltaProj"]["top1_best"] - results["MLP_37"]["top1_best"]
        print(f"\nΔ(SGNNET_DeltaProj − MLP_37) = {d*100:+.2f}pp on CIFAR-100")
        print(f"  Reference: on Imagenette (step403b vs step235), Δ ≈ −0.41pp (SGNNET 97.30% − MLP 97.71%)")
        if d > 0:
            print(f"  → COMPLEXITY-RESCUE HYPOTHESIS CONFIRMED. SGNNET wins at higher class complexity.")
        else:
            print(f"  → MLP still wins. Either rescue hypothesis is wrong, or CIFAR-100 not hard enough.")


if __name__ == "__main__":
    main()
