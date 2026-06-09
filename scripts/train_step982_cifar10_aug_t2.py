"""Step 982: CIFAR-10 augmentation T2 (150ep, 100% data).
# CUDA-5060ti-validated
T1 ADVANCED: Ref=78.60%, A_aug=79.63%, Δ=+1.03pp. T2 = paper claim.
Configs trained SEQUENTIALLY (OOM fix). Advance: A_aug >= Ref +0.5pp.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py

import argparse
from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
# SGNNET_Resonant_CUDA: DeltaW loop accelerated via torch.compile below

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = 150
BATCH  = 256 if DEVICE.type == "mps" else 512
SEED   = 42

N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "5060ti_cuda")
OUT_PATH = ROOT / "results" / f"train_step982_cifar10_aug_t2_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref":   str(ROOT / "data" / "store_cifar10.h5"),
    "A_aug": str(ROOT / "data" / "store_cifar10_aug.h5"),
}

T1_REF  = 0.7860   # step982 T1 Ref
T1_AUG  = 0.7963   # step982 T1 A_aug (+1.03pp)


# ── ΔW-proj model (exact match to step980/step882) ───────────────────────────

def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class DeltaW(nn.Module):
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        dw        = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref     = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    model = DeltaW(resonant).to(DEVICE)
    return torch.compile(model) if DEVICE.type == "cuda" else model


class H5Dataset(Dataset):
    """Lazy h5 reader — keeps file open, reads per-item. Avoids 10 GB full load."""
    def __init__(self, h5_path: str, split: str):
        # 1 GB chunk cache reduces random-access decompress overhead
        self._path = h5_path
        self._split = split
        self._f = h5py.File(h5_path, "r", rdcc_nbytes=1 * 1024**3,
                            rdcc_nslots=10007, rdcc_w0=0.75)
        self._n = len(self._f[f"{split}/labels"])

    def __len__(self): return self._n

    def __getitem__(self, i):
        x = torch.from_numpy(self._f[f"{self._split}/features"][i].astype(np.float32))
        y = int(self._f[f"{self._split}/labels"][i])
        return x, y

    def close(self):
        self._f.close()


def load_val(h5_path: str):
    # Val is 10K × 25088 × float32 = ~1 GB — keep on CPU (shared GPU is tight),
    # transfer per eval batch in train_config.
    with h5py.File(h5_path, "r") as f:
        va_x = torch.tensor(f["val/features"][:], dtype=torch.float32)
        va_y = torch.tensor(f["val/labels"][:], dtype=torch.long)
    return va_x, va_y


def train_config(key: str, h5_path: str) -> dict:
    print(f"\n{'─'*60}")
    if not Path(h5_path).exists():
        print(f"ERROR: {h5_path} not found."); sys.exit(1)

    ds   = H5Dataset(h5_path, "train")
    n_train = len(ds)
    va_x, va_y = load_val(h5_path)
    _pin = (DEVICE.type == "cuda")
    tr   = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                      pin_memory=_pin, generator=torch.Generator().manual_seed(SEED))
    model = build_model()
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * len(tr),
        pct_start=0.1, anneal_strategy="cos",
    )
    print(f"{key}: h5={Path(h5_path).name}  train={n_train}  params={n_p:,}  device={DEVICE}")

    best = 0.0; best_ep = 0; t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for bx, by in tr:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            F.cross_entropy(model(bx), by).backward()
            opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for i in range(0, va_x.shape[0], BATCH):
                vb = va_x[i:i+BATCH].to(DEVICE, non_blocking=True)
                s  = model(vb)
                correct += (s.argmax(1).cpu() == va_y[i:i+BATCH]).sum().item()
                total   += s.shape[0]
        acc = correct / total
        if acc > best:
            best = acc; best_ep = ep + 1
        if (ep + 1) % 30 == 0 or ep == 0:
            print(f"  e{ep+1:3d}/{EPOCHS}  val={acc:.4f}  best={best:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  DONE: best={best:.4f} @ep{best_ep}  {elapsed:.0f}s")

    ds.close()
    del model, tr, va_x, va_y, ds
    gc.collect(); torch.cuda.empty_cache()

    return {"h5_file": Path(h5_path).name, "train_samples": n_train,
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "elapsed_s": round(elapsed, 1)}


def main():
    print(f"\n{'='*70}")
    print(f"step982 — CIFAR-10 augmentation T2 (150ep, 100% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}  batch={BATCH}")
    print(f"  T1 result: Ref={T1_REF:.4f}  A_aug={T1_AUG:.4f}  Δ=+1.03pp [ADVANCED]")
    print(f"  Advance T2: A_aug >= Ref + 0.5pp → paper claim")
    print(f"{'='*70}\n")

    results = {}
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
        print(f"Resuming: {list(results)} already in {OUT_PATH.name}")
    for key, h5_path in CONFIGS.items():
        if key in results:
            print(f"Skip {key}: done, best={results[key]['best']:.4f}")
            continue
        results[key] = train_config(key, h5_path)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_b = results["Ref"]["best"]
    aug_b = results["A_aug"]["best"]
    delta = aug_b - ref_b
    verdict = "PAPER CLAIM" if delta >= 0.005 else ("NEUTRAL" if delta >= 0.0 else "KILL")
    print(f"\n{'='*70}")
    print(f"step982 T2 SUMMARY")
    print(f"  Ref:   {ref_b:.4f}")
    print(f"  A_aug: {aug_b:.4f}  Δ={delta:+.4f}  [{verdict}]")
    print(f"\n→ {OUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
