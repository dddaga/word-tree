"""Step 862: Cross-dataset validation on CIFAR-10 T0.

MOTIVATION
==========
Paper requires generalization beyond Imagenette. CIFAR-10 is the canonical
small-image benchmark. SGNNET uses VGG16 features → same feature extractor,
different downstream distribution.

Hypothesis: SGNNET efficiency advantage holds on CIFAR-10.
Compare SGNNET K=5 vs Linear and MLP baselines at matched params.

DATA: data/store_cifar10.h5 — VGG16 pool5 features for CIFAR-10 train/val.
Features are 512-dim (VGG pool5 after GAP, not 25088 — confirm shape on load).

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_SGNNET : SGNNET K=5 N=512 D=8 (scaled for 512-dim features)
  Linear     : 512 → 10 linear baseline
  MLP_37     : 512 → h=37 → 10 MLP (matched params)
  MLP_256    : 512 → 256 → 10 MLP (stronger baseline)

Note: N_in=512 for CIFAR-10 features; reduce N to 512 or 1024 accordingly.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

parser = argparse.ArgumentParser(description="Step 862: CIFAR-10 cross-dataset T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar10.h5")
parser.add_argument("--configs", default="Linear,MLP_37,MLP_256,Ref_SGNNET")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_OUT = 10

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step862_cifar10_crossdataset_seed{SEED}__{SLOT}.json"
DATA_PATH = ROOT / args.data


class H5FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            keys = list(f[split].keys())
            feat_key = "features" if "features" in keys else keys[0]
            lbl_key = "labels" if "labels" in keys else keys[1]
            self.X = torch.from_numpy(f[f"{split}/{feat_key}"][:]).float()
            self.y = torch.from_numpy(f[f"{split}/{lbl_key}"][:]).long()
            # Flatten if needed
            if self.X.dim() > 2:
                self.X = self.X.view(len(self.X), -1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_eval(model, tr, va, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        val_top1 = correct / max(total, 1)
        history.append(val_top1)
        if (epoch + 1) % 5 == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found"); sys.exit(1)

    tr_full = H5FeatureDataset(str(DATA_PATH), "train")
    va_ds   = H5FeatureDataset(str(DATA_PATH), "val")
    N_IN = tr_full.X.shape[-1]
    print(f"CIFAR-10 features: N_in={N_IN}  train={len(tr_full)}  val={len(va_ds)}")

    n_full = len(tr_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    # Dynamically set N based on feature dim
    N_sgnnet = min(1024, N_IN)
    D_sgnnet = 8  # smaller D for smaller feature space
    K_iter = 5; K_in = max(10, N_sgnnet // 64); K_hh = 2; K_in_local = K_hh - max(1, K_hh // 4)
    K_in_random = K_hh - K_in_local

    # Late import to avoid import errors if sgnnet not needed
    from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
    from src.sgnnet.model_resonant      import SGNNET_Resonant
    from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

    CONFIGS = {}

    # Linear baseline
    class LinearModel(nn.Module):
        def __init__(self): super().__init__(); self.fc = nn.Linear(N_IN, N_OUT)
        def forward(self, x): return self.fc(x)
    CONFIGS["Linear"] = (lambda: LinearModel(), "Linear 512→10 baseline")

    # MLP baselines
    class MLP(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(N_IN, h), nn.ReLU(), nn.Linear(h, N_OUT))
        def forward(self, x): return self.net(x)
    h37 = max(1, int((-N_IN - N_OUT + (N_IN**2 + 6*N_IN*N_OUT + N_OUT**2)**0.5) / 2))
    CONFIGS["MLP_37"]  = (lambda: MLP(37),  f"MLP h=37 (~matched param MLP)")
    CONFIGS["MLP_256"] = (lambda: MLP(256), "MLP h=256 strong baseline")

    def make_sgnnet():
        torch.manual_seed(SEED)
        base = SGNNET_SmallWorld(
            N_hidden=N_sgnnet, N_out=N_OUT, D=D_sgnnet, N_in=N_IN,
            K_in=K_in, K_iter=K_iter, K_local=K_in_local, K_random=K_in_random,
            n_groups=max(8, N_sgnnet // 8), norm_mode="l2", encoding_mode="fourier",
            sparsity=0.90)
        res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
                               beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")
    CONFIGS["Ref_SGNNET"] = (make_sgnnet, f"SGNNET K=5 N={N_sgnnet} D={D_sgnnet} K_in={K_in}")

    print(f"\n{'='*70}")
    print(f"step862 — CIFAR-10 cross-dataset T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  N_in={N_IN}  SGNNET N={N_sgnnet} D={D_sgnnet}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; linear_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        make_fn, desc = CONFIGS[key]
        model = make_fn()
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        t0 = time.time()
        history = train_eval(model, tr, va, EPOCHS)
        elapsed = time.time() - t0
        best, best_ep = max(history), int(np.argmax(history)) + 1
        if key == "Linear": linear_acc = best
        delta = best - (linear_acc or 0.90)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "n_params": n_p, "best": best, "best_ep": best_ep,
                        "delta_vs_linear": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 862 SUMMARY — CIFAR-10 cross-dataset T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'params':>8} {'best':>7} {'Δ_vs_Linear':>14}")
    for k, r in results.items():
        print(f"  {k:<14} {r['n_params']:>8,} {r['best']:>7.4f} {r['delta_vs_linear']*100:>+13.2f}pp")
    sgnnet = results.get("Ref_SGNNET", {})
    if sgnnet:
        mLP256 = results.get("MLP_256", {}).get("best", 0)
        gap = (sgnnet["best"] - mLP256) * 100
        print(f"\n  SGNNET vs MLP_256: {gap:+.2f}pp  (negative = gap to close at T1)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
