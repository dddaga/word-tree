"""Step 400: CIFAR-10 generalization test — raw pixel SGNNET (cross-dataset).

MOTIVATION
==========
All SGNNET results to date are on Imagenette (VGG16 features, 10 classes).
This experiment tests whether the efficiency config generalises to a different
dataset using raw pixel input (no pretrained backbone).

INPUT: 32×32 RGB flattened → N_IN = 3072 (raw pixels, no VGG features)
CIFAR-10: 50K train / 10K val, 10 classes

CONFIGS (Tier-0: 30 epochs, full CIFAR-10 data)
  Linear  : nn.Linear(3072, 10) — sanity baseline
  N512    : SGNNET N=512,  D=16, K_hh=2, K_iter=5, AH α=1.0
  N2048   : SGNNET N=2048, D=16, K_hh=2, K_iter=5, AH α=1.0 (efficiency config)
  N4096   : SGNNET N=4096, D=16, K_hh=2, K_iter=5, AH α=1.0 (max tested scale)

Expected: N512 < N2048 < N4096 — tests N-scaling on a harder raw-pixel task.
CIFAR-10 SotA with simple linear: ~40%. SGNNET may show meaningful N-scaling
even from raw pixels, establishing architecture generality beyond Imagenette.
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
import torchvision
import torchvision.transforms as transforms

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=30)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128
SEED   = 42
N_IN   = 3072   # 32×32×3 flattened
N_OUT  = 10
D      = 16
K_HH   = 2
K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0
ALPHA_AHEBB   = 1.0

OUT_PATH = ROOT / "results" / "train_step400_cifar10_generalization.json"


# ── CIFAR-10 dataset wrapper ─────────────────────────────────────────────────
# Trainer expects (features, soft_labels, labels) tuples.
# CIFAR-10 gives (image_tensor, label_int).
# We flatten the image and convert hard labels to one-hot soft labels.

class CIFAR10FlatDataset(torch.utils.data.Dataset):
    """Wraps torchvision CIFAR-10 to emit (flat_pixels, one_hot, label)."""

    def __init__(self, root: Path, train: bool):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.ds = torchvision.datasets.CIFAR10(
            root=str(root / "data"), download=True, train=train, transform=tfm
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, label = self.ds[idx]
        flat = img.view(-1)                              # [3072]
        soft = F.one_hot(torch.tensor(label), N_OUT).float()  # [10]
        return flat, soft, torch.tensor(label, dtype=torch.long)


def make_cifar_loaders(batch_size: int = 128, seed: int = 42):
    train_ds = CIFAR10FlatDataset(ROOT, train=True)
    val_ds   = CIFAR10FlatDataset(ROOT, train=False)
    print(f"CIFAR-10 loaded: train={len(train_ds)}  val={len(val_ds)}  N_IN={N_IN}")
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


# ── Model wrappers ────────────────────────────────────────────────────────────

class FlattenWrapper(nn.Module):
    """Proxy wrapper that forwards flat input to an SGNNET model.

    SGNNET already accepts [B, N_IN] flat input when N_in is set.
    This wrapper exists only to expose W_pos / W_phase / tick_* at the
    top level so Trainer can access them directly.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    # Proxy attributes that Trainer accesses directly on model
    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.base.W_phase

    def tick_epoch(self):
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def tick_step(self):
        if hasattr(self.base, "tick_step"):
            self.base.tick_step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x already flat [B, 3072] from CIFAR10FlatDataset
        return self.base(x)


class LinearBaseline(nn.Module):
    """nn.Linear(3072, 10) with dummy W_pos / W_phase so Trainer is happy.

    W_pos is a 1-row dummy (no safety loss, no clamping effect — lambda_safety=0).
    W_phase is None.
    """
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        # Trainer accesses model.W_pos as an nn.Parameter for the optimizer param group.
        # Provide a dummy 1×1 parameter — it won't affect training (lr will be non-zero
        # but grad is always 0 since it's not used in forward). lambda_safety=0 so the
        # safety_valve_loss branch is never entered.
        self.W_pos   = nn.Parameter(torch.zeros(1, 1))
        self.W_phase = None

    def tick_epoch(self): pass
    def tick_step(self):  pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── Build SGNNET config ───────────────────────────────────────────────────────

def build_sgnnet(n_hidden: int) -> nn.Module:
    k_r  = max(1, K_HH // 4)
    k_l  = K_HH - k_r
    ng   = max(8, n_hidden // 8)
    k_in = max(10, N_IN // 128)   # ~24 for 3072 — proportional input fan-in
    base = SGNNET_SmallWorld(
        N_hidden=n_hidden, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=k_in, K_iter=K_ITER, K_local=k_l, K_random=k_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    ah_model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return FlattenWrapper(ah_model)


# ── Training helper ───────────────────────────────────────────────────────────

def run_config(label: str, model: nn.Module, tr, va, n: int) -> dict:
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{label}]  params={n_params:,}")

    t0  = time.time()
    kw  = trainer_kwargs(n, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    def _log(m):
        if str(DEVICE) == "mps":
            torch.mps.empty_cache()
        ep = m["epoch"] + 1
        if ep % 10 == 0:
            print(f"    ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0
    top1h   = [round(h.get("val_top1", 0.0), 4) for h in history]
    best    = max(top1h)
    bep     = int(np.argmax(top1h)) + 1

    flops = 3 * n * K_HH * D * K_ITER if n > 1 else 0  # 0 for linear baseline
    return {
        "label":        label,
        "N":            n,
        "D":            D,
        "K_hh":         K_HH,
        "K_iter":       K_ITER,
        "alpha_ahebb":  ALPHA_AHEBB,
        "n_params":     n_params,
        "flops":        flops,
        "top1_best":    best,
        "top1_last":    top1h[-1],
        "best_epoch":   bep,
        "epochs_run":   len(history),
        "elapsed_s":    round(elapsed, 1),
        "top1_history": top1h,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"Step 400 — CIFAR-10 generalization (raw pixels, Tier-0 {EPOCHS}ep)")
    print(f"N_IN={N_IN}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  AH α={ALPHA_AHEBB}")
    print(f"Configs: Linear / N=512 / N=2048 (efficiency) / N=4096")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    tr, va = make_cifar_loaders(batch_size=BATCH, seed=SEED)

    configs = [
        ("Linear",  LinearBaseline(N_IN, N_OUT), 1),
        ("N512",    build_sgnnet(512),            512),
        ("N2048",   build_sgnnet(2048),           2048),
        ("N4096",   build_sgnnet(4096),           4096),
    ]

    results = {}
    for label, model, n in configs:
        results[label] = run_config(label, model, tr, va, n)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Config':<10}  {'N':>5}  {'params':>9}  {'best_top1':>10}  {'best_ep':>8}  {'elapsed':>8}")
    print(f"{'-'*70}")
    for label, r in results.items():
        flag = " ← efficiency config" if label == "N2048" else ""
        print(f"{label:<10}  {r['N']:>5}  {r['n_params']:>9,}  "
              f"{r['top1_best']:>10.4f}  {r['best_epoch']:>8}  "
              f"{r['elapsed_s']:>7.0f}s{flag}")
    print(f"{'='*70}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}\n")


if __name__ == "__main__":
    main()
