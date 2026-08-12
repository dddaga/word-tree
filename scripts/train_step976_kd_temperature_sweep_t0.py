"""Step 976: KD temperature sweep T0 — does T>1 help SGNNET?

# CUDA-5060ti-validated

HYPOTHESIS
==========
VGG16 soft_labels in store_aug.h5 are at T=1 (peak ≈0.9998 — essentially one-hot).
Trainer already uses KL divergence — but with peaked targets, KL ≈ CE.
Temperature-scaling T>1 should extract signal from VGG's tail probabilities.

Temperature-scaled labels recovered exactly from T=1 soft labels:
    softmax(logits/T) = softmax(log(soft)/T)   [softmax is shift-invariant]

CONFIGS (full efficiency stack: SmallWorld + Resonant + AH, N=2048 D=16 K_hh=2 K_iter=5)
  T=1   Ref    (baseline — equivalent to existing step199)
  T=2   Warm
  T=4   Standard KD temperature
  T=8   Aggressive smoothing

T0 PROTOCOL: 20ep, 50% data, BATCH=512, seed=42, 5060ti CUDA.
DECISION: if T>1 best beats T=1 by ≥+0.5pp → advance T1; else KD story stands.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--data", default="data/store_aug.h5")
args = parser.parse_args()

DEVICE = torch.device(args.device)
EPOCHS = args.epochs
BATCH  = 512
SEED   = 42
DATA_FRAC = 0.5

# step199 efficiency champion config
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
FLOPS = 3 * N * K_HH * D * K_ITER  # 0.98M

TEMPERATURES = [1.0, 2.0, 4.0, 8.0]

OUT_PATH = ROOT / "results" / f"train_step976_kd_temperature_sweep_t0_seed{SEED}__5060ti_cuda.json"


def load_data():
    """Pre-load to GPU. num_workers=0 (CUDA tensors)."""
    print(f"Loading {args.data}...")
    with h5py.File(args.data, "r") as f:
        tr_x  = torch.tensor(f["train/features"][:],    dtype=torch.float32).to(DEVICE)
        tr_y  = torch.tensor(f["train/labels"][:],      dtype=torch.long).to(DEVICE)
        tr_sf = torch.tensor(f["train/soft_labels"][:], dtype=torch.float32).to(DEVICE)
        va_x  = torch.tensor(f["val/features"][:],      dtype=torch.float32).to(DEVICE)
        va_y  = torch.tensor(f["val/labels"][:],        dtype=torch.long).to(DEVICE)

    n = tr_x.shape[0]
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g)[:int(n * DATA_FRAC)]
    tr_x, tr_y, tr_sf = tr_x[idx], tr_y[idx], tr_sf[idx]
    print(f"  train: {tr_x.shape[0]:,}  val: {va_x.shape[0]:,}")

    # Data pre-loaded to GPU — pin_memory=True irrelevant
    # num_workers=0: forked workers cannot access CUDA tensors
    return tr_x, tr_y, tr_sf, va_x, va_y


def temperature_scale(soft_T1: torch.Tensor, T: float) -> torch.Tensor:
    """softmax(log(soft)/T) == softmax(logits/T) via shift invariance."""
    if T == 1.0:
        return soft_T1
    log_soft = torch.log(soft_T1.clamp_min(1e-9))
    return F.softmax(log_soft / T, dim=1)


def build_model():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(DEVICE)
    return model


def train_one(T: float, tr_x, tr_y, tr_sf_T, va_x, va_y, lr=3e-3):
    torch.manual_seed(SEED)
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ds = TensorDataset(tr_x, tr_sf_T, tr_y)
    tr = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_steps = EPOCHS * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.1, anneal_strategy="cos"
    )
    # GradScaler disabled — fp16+GradScaler 4.4x slower on Blackwell (step801)

    history = []
    best = 0.0; best_ep = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        loss_sum = 0.0; nb = 0
        for bx, bsf, _by in tr:
            opt.zero_grad()
            scores = model(bx)
            # KL with T^2 scaling (Hinton) preserves gradient magnitude
            loss = F.kl_div(F.log_softmax(scores, dim=1), bsf, reduction="batchmean") * (T * T)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item(); nb += 1

        # eval
        model.eval()
        with torch.no_grad():
            correct = 0; total = 0
            for i in range(0, va_x.shape[0], 512):
                s = model(va_x[i:i+512])
                correct += (s.argmax(1) == va_y[i:i+512]).sum().item()
                total += s.shape[0]
            acc = correct / total
        history.append(round(acc, 4))
        if acc > best:
            best = acc; best_ep = ep + 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    T={T:.1f} ep{ep+1:2d}  val={acc:.4f}  best={best:.4f}  loss={loss_sum/nb:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    return {
        "T": T, "lr": lr, "top1_best": best, "top1_last": history[-1],
        "best_epoch": best_ep, "epochs_run": len(history), "top1_history": history,
        "elapsed_s": round(time.time() - t0, 1), "n_params": n_params, "flops": FLOPS,
    }


def main():
    print(f"\n{'='*70}")
    print(f"Step 976 — KD Temperature Sweep T0 (5060ti CUDA)")
    print(f"Config: step199 full stack  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"Protocol: {EPOCHS}ep, {int(DATA_FRAC*100)}% data, BATCH={BATCH}, seed={SEED}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"{'='*70}")

    tr_x, tr_y, tr_sf, va_x, va_y = load_data()

    # Diagnostic: show softness of labels
    print("\nSoft-label peakedness (sample 0):")
    for T in TEMPERATURES:
        sf_T = temperature_scale(tr_sf[:1], T)
        peak = sf_T.max().item()
        entropy = -(sf_T * sf_T.clamp_min(1e-9).log()).sum(1).mean().item()
        print(f"  T={T:.1f}  peak={peak:.4f}  entropy={entropy:.4f}")

    results = {}
    for T in TEMPERATURES:
        print(f"\n── T={T:.1f} ──")
        tr_sf_T = temperature_scale(tr_sf, T)
        r = train_one(T, tr_x, tr_y, tr_sf_T, va_x, va_y)
        results[f"T{int(T)}"] = r
        print(f"  DONE T={T:.1f}: best={r['top1_best']:.4f} @ep{r['best_epoch']}  elapsed={r['elapsed_s']}s")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 976 SUMMARY")
    print(f"{'='*70}")
    ref_best = results["T1"]["top1_best"]
    print(f"  Ref T=1: {ref_best:.4f}")
    for T in TEMPERATURES[1:]:
        r = results[f"T{int(T)}"]
        delta = (r["top1_best"] - ref_best) * 100
        verdict = "WIN" if delta >= 0.5 else ("NEUTRAL" if delta >= -0.5 else "HURT")
        print(f"  T={T:.1f}:   {r['top1_best']:.4f}  Δref={delta:+.2f}pp  {verdict}")
    print(f"\n→ {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
