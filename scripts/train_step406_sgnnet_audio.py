"""Step 406: Cross-modal SGNNET on audio (ESC-50 + Whisper-tiny encoder features).

MOTIVATION
==========
step405 SST-2 CPU: SGNNET=83.60% vs Linear=84.63%, MLP_64=84.52%. Competitive.
Audio is the third modality to validate cross-modal claim.

ESC-50: 50 environmental sound classes, 2000 clips, 5-fold cross-validation.
Whisper-tiny encoder: 384-dim mean-pooled log-mel encoder output.
Use fold 5 as test, folds 1-4 as train (standard split).

PIPELINE
  Phase 1 (extraction): Download ESC-50, load audio at 16kHz,
                         run Whisper-tiny encoder, mean-pool → [N, 384] features.
  Phase 2 (training):   SGNNET + Linear + MLP_64 on 384-dim features.

EXPECTED
  Linear probe: ~65-70% (typical for simple probe on ESC-50 audio features)
  SGNNET: competitive within 2pp, confirming cross-modal claim on audio.

DEPS: pip install openai-whisper soundfile datasets h5py
      OR: pip install transformers[torch] h5py soundfile
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
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--seed",   type=int, default=42)
parser.add_argument("--configs", default="Linear,MLP_64,SGNNET")
parser.add_argument("--data_dir", default="data/esc50",
                    help="Directory to cache ESC-50 audio and HDF5 features")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 64; SEED = args.seed
N_IN_AUDIO = 384; N_OUT_AUDIO = 50   # Whisper-tiny encoder dim, ESC-50 classes
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

DATA_DIR = ROOT / args.data_dir
H5_PATH  = DATA_DIR / "store_esc50_whisper.h5"
SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step406_audio_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features():
    """Download ESC-50, extract Whisper-tiny encoder features."""
    print("Phase 1: extracting ESC-50 audio features with Whisper-tiny encoder")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download ESC-50 if not present
    esc50_dir = DATA_DIR / "ESC-50-master"
    if not esc50_dir.exists():
        import urllib.request, zipfile
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        zip_path = DATA_DIR / "esc50.zip"
        print(f"  Downloading ESC-50 from {url}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"  Extracted to {esc50_dir}")

    # Load ESC-50 metadata
    import csv
    meta_path = esc50_dir / "meta" / "esc50.csv"
    samples = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            samples.append({
                "filename": esc50_dir / "audio" / row["filename"],
                "label": int(row["target"]),
                "fold": int(row["fold"]),
            })
    print(f"  ESC-50: {len(samples)} clips, 50 classes, 5 folds")

    # Load Whisper-tiny via transformers or openai-whisper
    try:
        from transformers import WhisperModel, WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        model = WhisperModel.from_pretrained("openai/whisper-tiny").encoder.to(DEVICE).eval()
        def get_feat(audio_np, sr):
            inputs = fe(audio_np, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                enc = model(inputs["input_features"].to(DEVICE)).last_hidden_state
            return enc.mean(1).squeeze(0).cpu().numpy()  # mean-pool → [384]
    except ImportError:
        try:
            import whisper
            wmodel = whisper.load_model("tiny").to(DEVICE)
            def get_feat(audio_np, sr):
                import whisper.audio as wa
                mel = wa.log_mel_spectrogram(torch.from_numpy(audio_np).float()).to(DEVICE)
                mel = mel.unsqueeze(0)
                with torch.no_grad():
                    enc = wmodel.encoder(mel)  # [1, T, 384]
                return enc.mean(1).squeeze(0).cpu().numpy()
        except ImportError:
            print("ERROR: Install transformers or openai-whisper:")
            print("  pip install transformers[torch] h5py soundfile")
            print("  OR: pip install openai-whisper h5py soundfile")
            sys.exit(1)

    # Extract features
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: pip install soundfile"); sys.exit(1)

    tr_feats, tr_labels = [], []
    va_feats, va_labels = [], []
    for i, s in enumerate(samples):
        audio, sr = sf.read(str(s["filename"]), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(1)   # stereo → mono
        # resample to 16kHz if needed
        if sr != 16000:
            try:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
            except ImportError:
                audio = audio  # skip resampling if not available
        feat = get_feat(audio, 16000)
        if s["fold"] == 5:   # fold 5 = test
            va_feats.append(feat); va_labels.append(s["label"])
        else:
            tr_feats.append(feat); tr_labels.append(s["label"])
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)}", flush=True)

    tr_x = np.stack(tr_feats).astype(np.float32)
    va_x = np.stack(va_feats).astype(np.float32)
    tr_y = np.array(tr_labels, dtype=np.int64)
    va_y = np.array(va_labels, dtype=np.int64)

    import h5py
    with h5py.File(H5_PATH, "w") as f:
        f.create_dataset("train_features", data=tr_x)
        f.create_dataset("train_labels",   data=tr_y)
        f.create_dataset("val_features",   data=va_x)
        f.create_dataset("val_labels",     data=va_y)
        f.attrs["model"] = "whisper-tiny"
        f.attrs["dim"]   = N_IN_AUDIO
    print(f"  Saved → {H5_PATH}  train={len(tr_x)}  val={len(va_x)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: training
# ─────────────────────────────────────────────────────────────────────────────

class H5AudioDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, split):
        import h5py
        with h5py.File(h5_path, "r") as f:
            self.x = torch.from_numpy(f[f"{split}_features"][:]).float()
            self.y = torch.from_numpy(f[f"{split}_labels"][:]).long()

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.x[i], torch.zeros(1), self.y[i]


def build_linear():
    return nn.Linear(N_IN_AUDIO, N_OUT_AUDIO)


def build_mlp(hidden=64):
    return nn.Sequential(nn.Linear(N_IN_AUDIO, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_OUT_AUDIO))


def build_sgnnet():
    from src.sgnnet.model_smallworld  import SGNNET_SmallWorld
    from src.sgnnet.model_resonant    import SGNNET_Resonant

    class SGNNET_DeltaProj(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(SEED)
            K_r = max(1, K_HH // 4); K_l = K_HH - K_r
            self.base = SGNNET_SmallWorld(
                N_hidden=N, N_out=N_OUT_AUDIO, D=D, N_in=N_IN_AUDIO,
                K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
            # 1D encoding for flat audio embeddings
            from src.sgnnet.encoding import compute_fourier_encoding
            flat_enc = compute_fourier_encoding(N_IN_AUDIO, D=D, h=N_IN_AUDIO, w=1, c=1)
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

    tr_ds = H5AudioDataset(H5_PATH, "train")
    va_ds = H5AudioDataset(H5_PATH, "val")
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va = torch.utils.data.DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"Phase 2: training  Train={len(tr_ds)}  Val={len(va_ds)}  dim={N_IN_AUDIO}")

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

    print(f"\n{'='*60}\nSTEP 406 SUMMARY — ESC-50 audio (Whisper-tiny encoder)")
    for k, r in results.items():
        print(f"  {k:<8}  params={r['n_params']:>8,}  best={r['top1_best']:.4f}")


def main():
    if args.phase in ("extract", "both"):
        extract_features()
    if args.phase in ("train", "both"):
        train_all()


if __name__ == "__main__":
    main()
