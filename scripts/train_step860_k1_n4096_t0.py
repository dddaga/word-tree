"""Step 860: K=1 student @ N=4096 T0 — better teacher → better student?

MOTIVATION
==========
step607 K=1 N=2048 = 95.95% @ 0.20M FLOPs. Teacher (step604) = 96.69%.
step293 K=5 N=4096 T1 = 96.18% → N=4096 teacher would be ~97.30% at T2.
Hypothesis: K=1 student trained at N=4096 with same step604 teacher logits
achieves better accuracy than K=1 N=2048, at same params (34,976) and FLOPs
(0.20M — only K_iter=1 routing matters for FLOP count, not N_hidden).

WHY CROSS-N KD WORKS: teacher and student share the same input (VGG 25088-dim)
and same output space (10 classes). Teacher logits are input-indexed, N-agnostic.
A K=1 N=4096 student receives the same soft targets as K=1 N=2048.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_k5     : K=5 N=4096 from scratch (baseline — what T1 N=4096 looks like at 20ep)
  A_k1_scratch : K=1 N=4096 CE only (no KD) — does K=1 converge at N=4096?
  B_k1_kd    : K=1 N=4096 + KD from step604 teacher — main hypothesis

SUCCESS CRITERIA
  B_k1_kd >= step607 K=1 N=2048 (95.95%) → N=4096 gives better student
  A_k1_scratch >= 90% → K=1 routing sufficient at N=4096 without KD
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.sgnnet.model_resonant_cuda import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.dataset           import H5Dataset, make_loaders

parser = argparse.ArgumentParser(description="Step 860: K=1 @ N=4096 T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--cache",   default="data/teacher_trajectory_step604.h5")
parser.add_argument("--configs", default="Ref_k5,A_k1_scratch,B_k1_kd")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N4 = 4096; N2 = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN_N4 = 15  # K_in=15 is default at N>=4096
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0
T_KD = 4.0

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step860_k1_n4096_t0_seed{SEED}__{SLOT}.json"
DATA_PATH = ROOT / args.data
CACHE_PATH = ROOT / args.cache

STEP607_K1_N2048 = 0.9595  # paper champion baseline


class IndexedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            self.features = torch.from_numpy(f[f"{split}/features"][:])
            self.labels = torch.from_numpy(f[f"{split}/labels"][:]).long()
            sl = f[f"{split}"].get("soft_labels")
            self.soft_labels = torch.from_numpy(sl[:]) if sl else None
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        sl = self.soft_labels[idx] if self.soft_labels is not None else torch.zeros(10)
        return self.features[idx], sl, self.labels[idx], idx


def make_model_k5(N):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN_N4, K_iter=5, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    if DEVICE.type == "cuda":
        res = SGNNET_Resonant_CUDA(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                                    alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
                                    mode="dynamic_z_geo", resonance_threshold=0.0, compile=False)
        return SGNNET_AntiHebbian_CUDA(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos", compile=False)
    else:
        res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                               beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_model_k1(N):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN_N4, K_iter=1, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return res


def train_one(model, tr, va, logits_cache, use_kd, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in tr:
            x, _, y, idx = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_s = model(x)
            if use_kd and logits_cache is not None:
                logits_t = logits_cache[idx].to(DEVICE)
                L_kd = F.kl_div(F.log_softmax(logits_s / T_KD, dim=-1),
                                 F.softmax(logits_t / T_KD, dim=-1),
                                 reduction="batchmean") * (T_KD ** 2)
                loss = 0.5 * L_kd + 0.5 * F.cross_entropy(logits_s, y)
            else:
                loss = F.cross_entropy(logits_s, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(DEVICE)
                y = (batch[2] if len(batch) > 2 else batch[1]).to(DEVICE)
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

    tr_full = IndexedH5Dataset(str(DATA_PATH), "train")
    va_ds = H5Dataset(str(DATA_PATH), "val")
    n_full = len(tr_full)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    logits_cache = None
    teacher_top1 = 0.0
    if CACHE_PATH.exists():
        with h5py.File(str(CACHE_PATH), "r") as f:
            logits_cache = torch.from_numpy(f["train/logits"][:])
            teacher_top1 = float(f.attrs.get("teacher_top1", 0.0))
    else:
        print(f"WARNING: teacher cache not found at {CACHE_PATH}. B_k1_kd will use CE only.")

    print(f"\n{'='*70}")
    print(f"step860 — K=1 student @ N=4096 T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Teacher: {teacher_top1:.4f}  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"  Baseline: step607 K=1 N=2048 = {STEP607_K1_N2048:.4f}")
    print(f"{'='*70}\n")

    CONFIGS = {
        "Ref_k5":     (make_model_k5, N4, False, "K=5 N=4096 scratch — T0 baseline"),
        "A_k1_scratch": (make_model_k1, N4, False, "K=1 N=4096 CE only — no KD"),
        "B_k1_kd":    (make_model_k1, N4, True,  "K=1 N=4096 + KD from step604 teacher"),
    }

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        make_fn, N_model, use_kd, desc = CONFIGS[key]
        model = make_fn(N_model)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  N={N_model}  params={n_p:,}")

        t0 = time.time()
        history = train_one(model, tr, va, logits_cache, use_kd, EPOCHS)
        elapsed = time.time() - t0
        best, best_ep = max(history), int(np.argmax(history)) + 1
        if key == "Ref_k5": ref_acc = best
        delta_ref = best - (ref_acc or 0.94)
        delta_champion = best - STEP607_K1_N2048
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta_ref*100:+.2f}pp  "
              f"Δ_vs_champion={delta_champion*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "N": N_model, "K_iter": 1 if "k1" in key else 5, "use_kd": use_kd,
            "label": desc, "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta_ref, 4),
            "delta_vs_champion": round(delta_champion, 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 860 SUMMARY — K=1 @ N=4096 T0")
    print(f"{'='*70}")
    print(f"  {'config':<16} {'N':>5} {'best':>7} {'Δ_champion':>12}")
    for k, r in results.items():
        print(f"  {k:<16} {r['N']:>5} {r['best']:>7.4f} {r['delta_vs_champion']*100:>+11.2f}pp")
    kd = results.get("B_k1_kd", {})
    if kd:
        v = "ADVANCES to T1" if kd["best"] >= STEP607_K1_N2048 else f"trails by {(STEP607_K1_N2048 - kd['best'])*100:.2f}pp"
        print(f"\n  K=1 N=4096 KD verdict: {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
