"""Step 407: Cross-modal SGNNET on AG News 4-class text classification.

MOTIVATION
==========
step405 SST-2: SGNNET=83.60% (CPU) vs Linear=84.63%, MLP_64=84.52%.
SGNNET is competitive on binary text classification (-1pp).

AG News extends the cross-modal story to:
  - Multi-class (4 topics: World/Sports/Business/Sci-Tech)
  - Larger dataset (120K train / 7.6K test vs SST-2 67K/872)
  - More challenging than binary sentiment

Hypothesis: SGNNET topology routing generalizes to 4-class text at similar
parity with Linear/MLP baselines. Confirms cross-modal claim.

CONFIGS
=======
  Linear : 768→4 linear probe
  MLP_64 : 768→64→4 MLP (matched params)
  SGNNET : N=2048, D=16, K_hh=2, K_iter=5 (same as step405)

Tier: T2 (150ep, 100% data) — same as step405
DEPS: pip install transformers datasets h5py
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--phase",  choices=("extract", "train", "both"), default="both")
parser.add_argument("--model",  default="distilbert-base-uncased",
                    help="HF model name for feature extraction")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--configs", default="Linear,MLP_64,SGNNET")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN_TEXT = 768; N_OUT_TEXT = 4            # DistilBERT CLS dim, AG News 4-class
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

H5_PATH  = ROOT / "data" / "store_agnews_distilbert.h5"
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step407_agnews_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features():
    print(f"Phase 1: extracting AG News features from {args.model}")
    try:
        from transformers import AutoTokenizer, AutoModel
        from datasets import load_dataset
        import h5py
    except ImportError as e:
        print(f"ERROR: missing deps. Install: pip install transformers datasets h5py")
        print(f"  ({e})")
        sys.exit(1)

    ds = load_dataset("ag_news")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).to(DEVICE).eval()

    def encode(split, text_col="text"):
        data = ds[split]
        n = len(data)
        feats = np.zeros((n, N_IN_TEXT), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, n, 32):
                batch = data[i:i+32]
                toks = tok(batch[text_col], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
                out = mdl(**toks).last_hidden_state[:, 0, :]  # CLS embedding
                feats[i:i+len(out)] = out.cpu().numpy()
                labels[i:i+len(out)] = batch["label"]
                if i % 12800 == 0:
                    print(f"    {split} {i}/{n}", flush=True)
        return feats, labels

    tr_x, tr_y = encode("train")
    va_x, va_y = encode("test")  # AG News uses "test" split

    H5_PATH.parent.mkdir(exist_ok=True)
    import h5py
    with h5py.File(H5_PATH, "w") as f:
        f.create_dataset("train_features", data=tr_x)
        f.create_dataset("train_labels",   data=tr_y)
        f.create_dataset("val_features",   data=va_x)
        f.create_dataset("val_labels",     data=va_y)
        f.attrs["model"] = args.model
        f.attrs["dim"]   = N_IN_TEXT
    print(f"  Saved → {H5_PATH}  train={len(tr_x)}  test={len(va_x)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: training
# ─────────────────────────────────────────────────────────────────────────────

class H5TextDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split):
        import h5py
        with h5py.File(h5_path, "r") as f:
            self.x = torch.from_numpy(f[f"{split}_features"][:]).float()
            self.y = torch.from_numpy(f[f"{split}_labels"][:]).long()

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.x[i], torch.zeros(1), self.y[i]


def build_linear():
    return nn.Linear(N_IN_TEXT, N_OUT_TEXT)


def build_mlp(hidden=64):
    return nn.Sequential(nn.Linear(N_IN_TEXT, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_OUT_TEXT))


def build_sgnnet():
    from src.sgnnet.model_smallworld  import SGNNET_SmallWorld
    from src.sgnnet.model_resonant    import SGNNET_Resonant

    class SGNNET_DeltaProj(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(SEED)
            K_r = max(1, K_HH // 4); K_l = K_HH - K_r
            self.base = SGNNET_SmallWorld(
                N_hidden=N, N_out=N_OUT_TEXT, D=D, N_in=N_IN_TEXT,
                K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
            # 1D encoding for flat NLP embeddings (not spatial 7×7 VGG layout)
            from src.sgnnet.encoding import compute_fourier_encoding
            flat_enc = compute_fourier_encoding(N_IN_TEXT, D=D, h=N_IN_TEXT, w=1, c=1)
            self.base.register_buffer("spatial_coords", flat_enc)
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

    return SGNNET_DeltaProj()


def train_loop(model, tr, va, epochs):
    model = model.to(DEVICE)
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit  = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _s, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, _s, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.numel()
        val_top1 = correct / max(total, 1)
        history.append({"epoch": epoch, "val_top1": val_top1})
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def train_all():
    if not H5_PATH.exists():
        print(f"ERROR: features not extracted. Run --phase extract first.")
        sys.exit(1)

    tr_ds = H5TextDataset(H5_PATH, "train")
    va_ds = H5TextDataset(H5_PATH, "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"Phase 2: training  Train={len(tr_ds)}  Test={len(va_ds)}  dim={N_IN_TEXT}")

    results = {}
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text())
            print(f"  Resuming: {list(results.keys())} already done")
        except Exception:
            pass

    for key in [k.strip() for k in args.configs.split(",") if k.strip()]:
        if key in results:
            print(f"  skip {key} (done: {results[key]['top1_best']:.4f})")
            continue
        print(f"\n{'─'*60}\nConfig: {key}\n{'─'*60}")
        torch.manual_seed(SEED)
        if   key == "Linear": model = build_linear()
        elif key == "MLP_64": model = build_mlp(64)
        elif key == "SGNNET": model = build_sgnnet()
        else: print(f"  skip unknown {key}"); continue

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  device={DEVICE}")
        t0 = time.time()
        history = train_loop(model, tr, va, epochs=EPOCHS)
        top1h = [h.get("val_top1", 0.0) for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        elapsed = time.time() - t0
        print(f"  → {key}  best={best:.4f} @ep{bep}  elapsed={elapsed:.0f}s")
        results[key] = {"n_params": n_p, "top1_best": best, "best_epoch": bep, "elapsed_s": elapsed}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}\nSTEP 407 SUMMARY — AG News 4-class (DistilBERT CLS)")
    for k, r in results.items():
        print(f"  {k:<8}  params={r['n_params']:>8,}  best={r['top1_best']:.4f}")


def main():
    if args.phase in ("extract", "both"):
        extract_features()
    if args.phase in ("train", "both"):
        train_all()


if __name__ == "__main__":
    main()
