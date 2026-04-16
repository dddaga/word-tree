"""Step 410: SST-2 config sweep — close the gap between SGNNET and Linear/MLP_64.

MOTIVATION (user directive 2026-04-16)
=======================================
step405 baseline on SST-2 (DistilBERT 768-dim features):
  Linear = 84.63%
  MLP_64 = 84.52%
  SGNNET (N=2048, K_in=25, K_iter=5) = 83.60% (-1.03pp vs Linear)

Default SGNNET was tuned for VGG16 features (N_in=25088). At N_in=768:
  - N=2048 may be overparameterized (too many hidden neurons for 768-dim input)
  - K_in=25 at N_in=768 means each neuron sees 3.3% of features (vs 0.1% at N_in=25088)
  - K_iter=5 may be excessive for low-dim features

Close-the-gap hypothesis: smaller N + smaller K_in + fewer K_iter will match or beat Linear.

CONFIGS (one variable changed vs Ref step405 SGNNET config)
===========================================================
  Ref_orig:    N=2048 K_in=25 K_iter=5 (replicates step405 SGNNET = 83.60%)
  A_small_n:   N=512  K_in=25 K_iter=5 (scale N down 4x)
  B_low_kin:   N=2048 K_in=10 K_iter=5 (reduce fan-in)
  C_low_kiter: N=2048 K_in=25 K_iter=2 (fewer routing iters)
  D_combined:  N=512  K_in=10 K_iter=3 (all compressions)

Tier: T1 (100ep on full SST-2 ~67K train, CUDA fast so single run)
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--configs", default="Ref_orig,A_small_n,B_low_kin,C_low_kiter,D_combined")
parser.add_argument("--data",   default="data/store_sst2_distilbert.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN = 768; N_OUT = 2
D = 16; K_HH = 2
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step410_sst2_config_sweep_seed{SEED}__{SLOT}.json"

# (N, K_in, K_iter, label)
CONFIGS = {
    "Ref_orig":    (2048, 25, 5, "baseline (replicates step405)"),
    "A_small_n":   (512,  25, 5, "scale N down 4x"),
    "B_low_kin":   (2048, 10, 5, "reduce fan-in to K_in=10"),
    "C_low_kiter": (2048, 25, 2, "fewer routing iterations"),
    "D_combined":  (512,  10, 3, "all compressions combined"),
}


class H5SST2Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            self.x = torch.from_numpy(f[f"{split}_features"][:]).float()
            self.y = torch.from_numpy(f[f"{split}_labels"][:]).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.x[i], torch.zeros(1), self.y[i]


def build_sgnnet(N_h, K_in, K_iter):
    """Build SGNNET for SST-2 — 1D features, use flat Fourier encoding."""
    from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
    from src.sgnnet.model_resonant      import SGNNET_Resonant
    from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
    from src.sgnnet.encoding import compute_fourier_encoding
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N_h, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N_h // 8), norm_mode="l2", encoding_mode="fourier")
    # Override Fourier encoding to use 1D (flat 768-dim, not spatial)
    flat_enc = compute_fourier_encoding(N_IN, D=D, h=N_IN, w=1, c=1)
    base.spatial_coords = flat_enc
    base.spatial_sum = flat_enc[base.conn_in].sum(dim=1)
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def train_one(model, tr, va, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _s, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _s, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        val_top1 = correct / max(total, 1)
        history.append(val_top1)
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def main():
    torch.manual_seed(SEED)
    if not (ROOT / args.data).exists():
        print(f"ERROR: {ROOT / args.data} not found."); sys.exit(1)

    tr_ds = H5SST2Dataset(str(ROOT / args.data), "train")
    va_ds = H5SST2Dataset(str(ROOT / args.data), "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    print(f"Step 410 — SST-2 config sweep (gap-close)  {EPOCHS}ep  device={DEVICE}")
    print(f"  Baselines: Linear=84.63%, MLP_64=84.52%, SGNNET default=83.60%\n")

    LINEAR_REF = 0.8463
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except Exception: pass

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})"); continue

        N_h, K_in, K_iter, desc = CONFIGS[key]
        model = build_sgnnet(N_h, K_in, K_iter)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}\n{key}: {desc}\n  N={N_h} K_in={K_in} K_iter={K_iter}  params={n_p:,}")

        t0 = time.time()
        history = train_one(model, tr, va, EPOCHS)
        elapsed = time.time() - t0
        best, bep = max(history), int(np.argmax(history)) + 1
        delta = best - LINEAR_REF
        print(f"  → best={best:.4f} @ep{bep}  Δ_Linear={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "N": N_h, "K_in": K_in, "K_iter": K_iter,
                        "n_params": n_p, "top1_best": best, "best_epoch": bep,
                        "delta_vs_linear": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 410 — SST-2 gap-close summary")
    print(f"  Linear=84.63%  MLP_64=84.52%  SGNNET-default=83.60%")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k:<12}  N={r['N']} K_in={r['K_in']} K_iter={r['K_iter']}  best={r['top1_best']:.4f}  Δ_Linear={r['delta_vs_linear']*100:+.2f}pp")


if __name__ == "__main__":
    main()
