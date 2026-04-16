"""Step 607: K=1 pure-KD student T2 validation (150ep, paper-quality).

MOTIVATION
==========
step605 Config_3 (K=1, K_in=25, pure KD) T1 = 96.33% (-0.36pp vs teacher 96.69%).
step606 Ref_k25 T1 = 95.69% (0.64pp lower than step605 — unexplained discrepancy).

T2 validation: does 150ep close the gap? Does K=1 student match or exceed teacher at T2?
Paper claim needs T2 confirmation: K=1 routing (5× reduction) at <0.5pp cost.

CONFIGS (all K=1, K_in=25, pure-KD with T=4)
  A_pure_kd : pure KD (β=1.0, α=0) — step605 Config_3 equivalent at T2
  B_balanced: balanced KD+CE (β=0.5) — test if adding CE helps at T2

Teacher: step604 cache (1.7GB, K=5 teacher at 96.69%).
Tier: T2 (150ep, 100% data)
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
from src.training.dataset           import H5Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=150)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--configs", default="A_pure_kd,B_balanced")
parser.add_argument("--cache",   default="data/teacher_trajectory_step604.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER_STUDENT = 1; K_IN = 25
ALPHA_REFLECT = 0.5
T_KD = 4.0

DATA_PATH  = ROOT / "data" / "store.h5"
CACHE_PATH = ROOT / args.cache
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step607_k1_kd_t2_seed{SEED}__{SLOT}.json"

# (alpha_CE_weight, label)  — beta_KD = 1 - alpha_CE
CONFIGS = {
    "A_pure_kd":  (0.0, "K=1 pure KD (β_KD=1.0, α_CE=0)"),
    "B_balanced": (0.5, "K=1 balanced (β_KD=0.5, α_CE=0.5)"),
}


class SGNNET_K1(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER_STUDENT, K_local=K_l, K_random=K_r,
            n_groups=ng, norm_mode="l2", encoding_mode="fourier")
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
        conn_hh = self.base.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.base.W_pos[:self.base.N_hidden]
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


class IndexedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            self.features = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.labels = torch.from_numpy(f[f"{split}/labels"][:]).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx], self.labels[idx], idx


def train_one(model, tr, va, logits_cache, alpha_ce, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _sl, y, idx in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_s = model(x)
            logits_t = logits_cache[idx].to(DEVICE)
            L_kd = F.kl_div(
                F.log_softmax(logits_s / T_KD, dim=-1),
                F.softmax(logits_t / T_KD, dim=-1),
                reduction="batchmean") * (T_KD ** 2)
            if alpha_ce > 0:
                L_ce = F.cross_entropy(logits_s, y)
                loss = (1 - alpha_ce) * L_kd + alpha_ce * L_ce
            else:
                loss = L_kd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(DEVICE)
                y = batch[2].to(DEVICE) if len(batch) > 2 else batch[1].to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        val_top1 = correct / max(total, 1)
        history.append(val_top1)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def main():
    torch.manual_seed(SEED)
    if not DATA_PATH.exists() or not CACHE_PATH.exists():
        print(f"ERROR: missing {DATA_PATH} or {CACHE_PATH}"); sys.exit(1)

    tr_ds = IndexedH5Dataset(str(DATA_PATH), "train")
    va_ds = H5Dataset(str(DATA_PATH), "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    with h5py.File(CACHE_PATH, "r") as f:
        logits_cache = torch.from_numpy(f["train/logits"][:])
        teacher_top1 = float(f.attrs.get("teacher_top1", 0.0))
    print(f"Step 607 — K=1 pure-KD T2 ({EPOCHS}ep, 100% data)")
    print(f"  device={DEVICE}  Teacher={teacher_top1:.4f}  Cache: {logits_cache.shape}\n")

    REF_TEACHER = teacher_top1  # step604 cache
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

        alpha_ce, desc = CONFIGS[key]
        model = SGNNET_K1()
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}\n{key}: {desc}  params={n_p:,}")

        t0 = time.time()
        history = train_one(model, tr, va, logits_cache, alpha_ce, EPOCHS)
        elapsed = time.time() - t0
        best, bep = max(history), int(np.argmax(history)) + 1
        delta = best - REF_TEACHER
        print(f"  → best={best:.4f} @ep{bep}  Δ_teacher={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "alpha_ce": alpha_ce,
                        "n_params": n_p, "top1_best": best, "best_epoch": bep,
                        "delta_vs_teacher": round(delta, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 607 — K=1 pure-KD T2 @ N=2048")
    print(f"  Teacher (K=5): {REF_TEACHER:.4f}  |  step605 T1 pure-KD: 0.9633")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k:<12}  best={r['top1_best']:.4f}  Δ_teacher={r['delta_vs_teacher']*100:+.2f}pp")


if __name__ == "__main__":
    main()
