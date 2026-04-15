"""Step 405: Cross-modal SGNNET on SST-2 text (DistilBERT CLS embeddings).

MOTIVATION (V3 Gap 2.8, paper-blocker)
======================================
Paper claims "universal FC replacement." One dataset on one modality is
insufficient. SST-2 binary sentiment via DistilBERT CLS features is the
smallest credible cross-modal experiment (~1 hour total including extraction).

PIPELINE
  Phase 1 (extraction):  HuggingFace datasets.load_dataset('glue', 'sst2')
                         → DistilBERT (distilbert-base-uncased) CLS embedding
                         → [N_samples, 768] float32 tensor
                         → stored at data/store_sst2_distilbert.h5
  Phase 2 (training):    SGNNET efficiency config with N_IN=768, N_OUT=2
                         Same ΔW proj mechanism as step235. Tier-2 (150ep).
                         Baselines: Linear probe (768→2), MLP_64 at matched params.

EXPECTED
  SST-2 Linear probe with DistilBERT CLS: ~85-88% (SOTA DistilBERT fine-tune: 92%).
  SGNNET should be competitive with MLP_64 at fewer params → paper claim stands.

Note (2026-04-14): README references switching to ModernBERT + Qwen3-0.6B.
This script keeps DistilBERT for the baseline claim; a ModernBERT variant is
a trivial --model argument swap once transformers>=4.48 is installed.

DEPS: pip install transformers datasets h5py

To run:
    # Phase 1: extract features (one-time, ~30min CPU or ~5min GPU)
    python -u scripts/train_step405_sgnnet_sst2.py --phase extract

    # Phase 2: train SGNNET + baselines
    python -u scripts/train_step405_sgnnet_sst2.py --phase train
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
                    help="HF model name. Alternatives: 'answerdotai/ModernBERT-base'")
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--configs", default="Linear,MLP_64,SGNNET")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N_IN_TEXT = 768; N_OUT_TEXT = 2            # DistilBERT CLS dim, SST-2 is binary
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

H5_PATH  = ROOT / "data" / "store_sst2_distilbert.h5"
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step405_sst2_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features():
    """Extract CLS embeddings for SST-2 train+val using the configured HF model."""
    print(f"Phase 1: extracting features from {args.model}")
    try:
        from transformers import AutoTokenizer, AutoModel
        from datasets import load_dataset
        import h5py
    except ImportError as e:
        print(f"ERROR: missing deps. Install: pip install transformers datasets h5py")
        print(f"  ({e})")
        sys.exit(1)

    ds = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).to(DEVICE).eval()

    def encode(split):
        n = len(ds[split])
        feats = np.zeros((n, N_IN_TEXT), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, n, 32):
                batch = ds[split][i:i+32]
                toks = tok(batch["sentence"], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(DEVICE)
                out = mdl(**toks).last_hidden_state[:, 0, :]   # CLS embedding
                feats[i:i+len(out)] = out.cpu().numpy()
                labels[i:i+len(out)] = batch["label"]
                if i % 3200 == 0:
                    print(f"    {split} {i}/{n}", flush=True)
        return feats, labels

    tr_x, tr_y = encode("train")
    va_x, va_y = encode("validation")

    H5_PATH.parent.mkdir(exist_ok=True)
    with h5py.File(H5_PATH, "w") as f:
        f.create_dataset("train_features", data=tr_x)
        f.create_dataset("train_labels",   data=tr_y)
        f.create_dataset("val_features",   data=va_x)
        f.create_dataset("val_labels",     data=va_y)
        f.attrs["model"] = args.model
        f.attrs["dim"]   = N_IN_TEXT
    print(f"  Saved → {H5_PATH}  train={len(tr_x)}  val={len(va_x)}")


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
        # Matches Imagenette loader: (feats, soft_labels_placeholder, labels)
        return self.x[i], torch.zeros(1), self.y[i]


def build_linear():
    return nn.Linear(N_IN_TEXT, N_OUT_TEXT)


def build_mlp(hidden=64):
    return nn.Sequential(nn.Linear(N_IN_TEXT, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_OUT_TEXT))


def build_sgnnet():
    # Lazy imports because Phase 1 doesn't need these
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
            # Fix: default fourier encoding assumes VGG h=7,w=7,c=512 spatial layout —
            # meaningless for flat 768-dim NLP embeddings. Override with 1D encoding.
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


def train_loop(model, tr, va, epochs, use_trainer=False):
    model = model.to(DEVICE)
    if use_trainer:
        from src.training.trainer           import Trainer
        from src.training.experiment_config import trainer_kwargs
        kw = trainer_kwargs(N, n_epochs=epochs)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
        history = trainer.train(n_epochs=epochs, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch']+1) % 10 == 0 else None))
        return history

    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt  = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    crit = nn.CrossEntropyLoss()
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}", flush=True)
    return history


def train_all():
    if not H5_PATH.exists():
        print(f"ERROR: features not extracted yet. Run --phase extract first.")
        sys.exit(1)

    tr_ds = H5TextDataset(H5_PATH, "train")
    va_ds = H5TextDataset(H5_PATH, "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"Phase 2: training  Train={len(tr_ds)}  Val={len(va_ds)}  dim={N_IN_TEXT}")

    results = {}
    for key in [k.strip() for k in args.configs.split(",") if k.strip()]:
        print(f"\n{'─'*60}\nConfig: {key}\n{'─'*60}")
        torch.manual_seed(SEED)
        if   key == "Linear": model = build_linear()
        elif key == "MLP_64": model = build_mlp(64)
        elif key == "SGNNET": model = build_sgnnet()
        else: print(f"  skip unknown {key}"); continue

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")
        t0 = time.time()
        history = train_loop(model, tr, va, epochs=EPOCHS, use_trainer=False)
        top1h = [h.get("val_top1", 0.0) for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        elapsed = time.time() - t0
        print(f"  → best={best:.4f} @ep{bep}  elapsed={elapsed:.0f}s")
        results[key] = {"n_params": n_p, "top1_best": best, "best_epoch": bep, "elapsed_s": elapsed}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n========== STEP 405 SUMMARY (SST-2, DistilBERT CLS) ==========")
    for k, r in results.items():
        print(f"  {k:<8}  params={r['n_params']:>8,}  best={r['top1_best']:.4f}")


def main():
    if args.phase in ("extract", "both"):
        extract_features()
    if args.phase in ("train", "both"):
        train_all()


if __name__ == "__main__":
    main()
