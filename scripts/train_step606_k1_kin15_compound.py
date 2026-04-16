"""Step 606: K=1 + K_in=15 compound efficiency — pure KD distillation.

MOTIVATION
==========
step605: K=1 pure-KD student (K_in=25) = 96.33% (-0.36pp vs K=5 teacher 96.69%).
step632: K_in=15 at K=5 = 95.13% (-0.33pp vs K_in=25).
step293: K_in=15 helps at N>=4096 (+0.33pp).

Does K_in=15 stack with K=1 distillation? If yes:
  Compound = 26.7× seed FLOP reduction + 5× routing reduction
  Total routing FLOPs: ~0.16M (vs original ~2.3M = 14× total)
  At <1pp cost from K=5 teacher.

PROTOCOL
========
Pure KD only (α=0, β=1.0, T=4.0 — step605 Config_3 was best non-trajectory config).
Teacher cache from step604 (K=5, K_in=25, 96.69%).
Student: K=1, 3 configs:
  Ref_k25:    K_in=25 K=1 (replicates step605 Config_3 baseline)
  A_k15:      K_in=15 K=1 (compound test)
  B_scratch:  K_in=15 K=1 no-distillation (scratch baseline)

Tier: T1 (75ep, 100% data — uses teacher cache, full dataset required).
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
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--configs", default="Ref_k25,A_k15,B_scratch")
parser.add_argument("--cache",   default="data/teacher_trajectory_step604.h5")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER_STUDENT = 1
ALPHA_REFLECT = 0.5
T_KD = 4.0

DATA_PATH  = ROOT / "data" / "store.h5"
CACHE_PATH = ROOT / args.cache
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step606_k1_kin15_compound_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref_k25":   (25, True,  "K=1 K_in=25 pure-KD (replicates step605 Config_3)"),
    "A_k15":     (15, True,  "K=1 K_in=15 pure-KD (compound efficiency test)"),
    "B_scratch": (15, False, "K=1 K_in=15 scratch (no distillation baseline)"),
}


class SGNNET_K1_DeltaProj(nn.Module):
    def __init__(self, K_in):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_in, K_iter=K_ITER_STUDENT, K_local=K_l, K_random=K_r,
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


class TeacherCache:
    def __init__(self, h5_path, split):
        with h5py.File(h5_path, "r") as f:
            self.logits = torch.from_numpy(f[f"{split}/logits"][:])
            self.teacher_top1 = float(f.attrs.get("teacher_top1", 0.0))
        print(f"  Teacher cache [{split}]: {list(self.logits.shape)}  teacher={self.teacher_top1:.4f}")

    def get_logits(self, indices, device):
        return self.logits[indices].to(device)


class IndexedH5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        with h5py.File(path, "r") as f:
            self.features = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.labels = torch.from_numpy(f[f"{split}/labels"][:]).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx], self.labels[idx], idx


def train_one(model, tr, va, cache_train, use_kd, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _sl, y, idx in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_s = model(x)
            L_ce = F.cross_entropy(logits_s, y)
            if use_kd and cache_train is not None:
                logits_t = cache_train.get_logits(idx, DEVICE)
                L_kd = F.kl_div(
                    F.log_softmax(logits_s / T_KD, dim=-1),
                    F.softmax(logits_t / T_KD, dim=-1),
                    reduction="batchmean") * (T_KD ** 2)
                loss = L_kd
            else:
                loss = L_ce
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
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def main():
    torch.manual_seed(SEED)
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found."); sys.exit(1)

    tr_ds = IndexedH5Dataset(str(DATA_PATH), "train")
    va_ds = H5Dataset(str(DATA_PATH), "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    cache_train = None
    if CACHE_PATH.exists():
        cache_train = TeacherCache(CACHE_PATH, "train")
    else:
        print(f"WARNING: Teacher cache not found at {CACHE_PATH}. KD configs will use CE only.")

    print(f"Step 606 — K=1+K_in=15 compound ({EPOCHS}ep)")
    print(f"  device={DEVICE}  Teacher=96.69% (step604)  K=1 KD=96.33% (step605)")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    if OUT_PATH.exists():
        try: results = json.loads(OUT_PATH.read_text())
        except Exception: pass

    REF_TEACHER = 0.9669
    REF_K1_K25 = 0.9633

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        if key in results:
            r = results[key]
            print(f"  skip {key} (done: {r['top1_best']:.4f})"); continue

        K_in, use_kd, desc = CONFIGS[key]
        model = SGNNET_K1_DeltaProj(K_in)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─'*60}\n{key}: {desc}\n  K_in={K_in} K=1 KD={'yes' if use_kd else 'no'} params={n_p:,}")

        t0 = time.time()
        history = train_one(model, tr, va, cache_train, use_kd, EPOCHS)
        elapsed = time.time() - t0
        best, bep = max(history), int(np.argmax(history)) + 1
        delta_teacher = best - REF_TEACHER
        delta_k1k25 = best - REF_K1_K25
        print(f"  → best={best:.4f} @ep{bep}  Δ_teacher={delta_teacher*100:+.2f}pp  Δ_k1k25={delta_k1k25*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"label": desc, "K_in": K_in, "K_iter": K_ITER_STUDENT,
                        "use_kd": use_kd, "n_params": n_p,
                        "top1_best": best, "best_epoch": bep,
                        "delta_vs_teacher": round(delta_teacher, 4),
                        "delta_vs_k1_k25": round(delta_k1k25, 4),
                        "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 606 — K=1+K_in=15 compound summary")
    print(f"  Teacher (K=5, K_in=25): 96.69%  |  K=1 KD K_in=25: 96.33%")
    for k in keys:
        r = results.get(k, {})
        if r: print(f"  {k:<12}  K_in={r['K_in']}  KD={'y' if r['use_kd'] else 'n'}  best={r['top1_best']:.4f}  Δ_teacher={r['delta_vs_teacher']*100:+.2f}pp")


if __name__ == "__main__":
    main()
