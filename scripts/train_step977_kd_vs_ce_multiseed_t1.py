"""Step 977: Multi-seed hard-CE vs soft-KD T=1 (paper-grade confirmation).

# CUDA-5060ti-validated

MOTIVATION
==========
step976 + step972 suggested soft-KD at T=1 contributes nothing measurable vs hard CE,
but those were single-seed T0 (20ep/50%) at different BATCH (512 vs 128) — cross-run
deltas are confounded by seed noise (our ±0.2pp) and hyperparameter divergence.

Peer review (qwen 2026-04-22) correctly flagged:
  (a) −0.13pp is within single-seed noise
  (b) step976/step972 Refs differ by 3pp due to BATCH mismatch
  (c) need multi-seed at fixed config to CONFIRM or refute

This script: 5 seeds × 2 configs, same BATCH=512, same T1 budget.

CONFIGS (step199 full stack — SmallWorld + Resonant + AH, N=2048 D=16 K_hh=2 K_iter=5)
  KD   alpha=0, T=1  — pure soft-KD (current default, what paper describes)
  CE   alpha=1       — pure hard cross-entropy

PROTOCOL: 75ep T1, 50% data, BATCH=512, seeds={0,1,42,123,2024}, 5060ti CUDA.
DECISION:
  paired mean |Δ(KD-CE)| < 0.3pp  AND  σ_Δ < 0.5pp  → NEUTRAL CONFIRMED (paper can simplify)
  mean Δ > +0.3pp                                    → soft-KD genuinely helps
  mean Δ < −0.3pp                                    → hard-CE genuinely helps
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
parser.add_argument("--epochs", type=int, default=75)
parser.add_argument("--data", default="data/store_aug.h5")
args = parser.parse_args()

DEVICE = torch.device(args.device)
EPOCHS = args.epochs
BATCH  = 512
DATA_FRAC = 0.5
SEEDS = [0, 1, 42, 123, 2024]
CONFIGS = [("KD", 0.0), ("CE", 1.0)]  # (name, alpha)

# step199 efficiency champion config
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
FLOPS = 3 * N * K_HH * D * K_ITER

OUT_PATH = ROOT / "results" / "train_step977_kd_vs_ce_multiseed_t1__5060ti_cuda.json"


def load_split(seed: int):
    with h5py.File(args.data, "r") as f:
        tr_x  = torch.tensor(f["train/features"][:],    dtype=torch.float32).to(DEVICE)
        tr_y  = torch.tensor(f["train/labels"][:],      dtype=torch.long).to(DEVICE)
        tr_sf = torch.tensor(f["train/soft_labels"][:], dtype=torch.float32).to(DEVICE)
        va_x  = torch.tensor(f["val/features"][:],      dtype=torch.float32).to(DEVICE)
        va_y  = torch.tensor(f["val/labels"][:],        dtype=torch.long).to(DEVICE)
    n = tr_x.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:int(n * DATA_FRAC)]
    return tr_x[idx], tr_y[idx], tr_sf[idx], va_x, va_y


def build_model(seed: int):
    torch.manual_seed(seed)
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
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(DEVICE)


def train_one(seed: int, alpha: float, lr=3e-3):
    tr_x, tr_y, tr_sf, va_x, va_y = load_split(seed)
    model = build_model(seed)
    ds = TensorDataset(tr_x, tr_y, tr_sf)
    tr = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=EPOCHS * len(tr),
        pct_start=0.1, anneal_strategy="cos"
    )

    best = 0.0; best_ep = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for bx, by, bsf in tr:
            opt.zero_grad()
            scores = model(bx)
            logp = F.log_softmax(scores, dim=1)
            if alpha == 1.0:
                loss = F.cross_entropy(scores, by)
            elif alpha == 0.0:
                loss = F.kl_div(logp, bsf, reduction="batchmean")
            else:
                loss = alpha * F.cross_entropy(scores, by) \
                       + (1.0 - alpha) * F.kl_div(logp, bsf, reduction="batchmean")
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for i in range(0, va_x.shape[0], 512):
                s = model(va_x[i:i+512])
                correct += (s.argmax(1) == va_y[i:i+512]).sum().item()
                total += s.shape[0]
            acc = correct / total
        if acc > best:
            best = acc; best_ep = ep + 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    seed={seed} α={alpha:.1f} ep{ep+1:2d} val={acc:.4f} best={best:.4f} [{time.time()-t0:.0f}s]", flush=True)

    return {"seed": seed, "alpha": alpha, "top1_best": best,
            "best_epoch": best_ep, "elapsed_s": round(time.time()-t0, 1)}


def main():
    print(f"\n{'='*70}")
    print(f"Step 977 — KD vs CE multi-seed T1 (5060ti CUDA)")
    print(f"Config: step199 full stack  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"Protocol: {EPOCHS}ep T1, {int(DATA_FRAC*100)}% data, BATCH={BATCH}")
    print(f"Seeds: {SEEDS}  Configs: {[c[0] for c in CONFIGS]}")
    print(f"{'='*70}")

    results = []
    for seed in SEEDS:
        for name, alpha in CONFIGS:
            print(f"\n── seed={seed}  {name} (α={alpha}) ──")
            r = train_one(seed, alpha)
            r["config"] = name
            results.append(r)
            print(f"  DONE {name} seed={seed}: best={r['top1_best']:.4f} @ep{r['best_epoch']}  elapsed={r['elapsed_s']}s")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 977 SUMMARY\n{'='*70}")
    print(f"{'seed':>6} | {'KD':>8} | {'CE':>8} | {'Δ(KD-CE)':>10}")
    deltas = []
    for seed in SEEDS:
        kd = next(r["top1_best"] for r in results if r["seed"]==seed and r["config"]=="KD")
        ce = next(r["top1_best"] for r in results if r["seed"]==seed and r["config"]=="CE")
        d  = (kd - ce) * 100
        deltas.append(d)
        print(f"{seed:>6} | {kd:>8.4f} | {ce:>8.4f} | {d:>+9.2f}pp")
    mean_d = float(np.mean(deltas)); std_d = float(np.std(deltas))
    print(f"\n  mean Δ(KD-CE) = {mean_d:+.2f}pp   σ = {std_d:.2f}pp")
    if abs(mean_d) < 0.3 and std_d < 0.5:
        verdict = "NEUTRAL CONFIRMED — paper can simplify training story"
    elif mean_d > 0.3:
        verdict = "SOFT-KD HELPS"
    else:
        verdict = "HARD-CE HELPS"
    print(f"  VERDICT: {verdict}")
    print(f"\n→ {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
