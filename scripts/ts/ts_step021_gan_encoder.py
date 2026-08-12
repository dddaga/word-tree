"""ts_step021 — GAN-based context encoder for next-day return prediction.

Architecture (Pathak et al. 2016, adapted for time-series):
  Encoder:       3× Conv2d (n_features→16→32→64) + ChannelwiseFC
  Generator:     Encoder + 3× ConvTranspose2d (64→32→16→n_features)
  Discriminator: 3× Conv2d (n_features→32→64→1) + AdaptiveAvgPool2d(1) + Sigmoid

Masking (same as ts_step020):
  - Future T_pred=5 days fully masked (rightmost T-axis columns)
  - Random 25% of remaining past days masked
  - Masked positions replaced with learnable mask token

Loss:
  Generator:     L = 0.999 * L_recon + 0.001 * L_adv
  Discriminator: L_D = 0.5*BCE(real,1) + 0.5*BCE(fake.detach(),0)
  Warmup 10 epochs: reconstruction only (no adversarial)

Downstream: frozen encoder + Linear(64*N_stocks → N_stocks) head.

Usage:
    python scripts/ts/ts_step021_gan_encoder.py [--device cpu] [--epochs 100]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "ts" / "raw"
CKPT_DIR    = ROOT / "data" / "ts" / "checkpoints"
RESULTS_DIR = ROOT / "results" / "ts"
for _d in (CKPT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ret_1d",
    "ma_ratio_5", "ma_ratio_21", "ma_ratio_55", "ma_ratio_89",
    "ma_cross_5_21", "ma_cross_13_55", "ma_cross_21_89",
    "vol_20", "vol_60",
    "spread_z", "resid_z", "delta_corr", "lag1_ret",
    "high_low_ratio", "volume_z",
]
assert len(FEATURE_COLS) == 16, "F must equal 16"

T_LOOKBACK = 60
T_PRED     = 5      # future days fully masked
MASK_RATIO = 0.25   # fraction of past days randomly masked
TC         = 0.001  # transaction cost

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2023-12-31"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",          default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--epochs",          type=int,   default=100,
                   help="downstream fine-tune epochs")
    p.add_argument("--pretrain_epochs", type=int,   default=50,
                   help="GAN pretraining epochs")
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--data_dir",        type=str,   default=str(RAW_DIR))
    p.add_argument("--results_dir",     type=str,   default=str(RESULTS_DIR))
    return p.parse_args()

# ---------------------------------------------------------------------------
# Feature engineering (same as ts_step010, no lookahead)
# ---------------------------------------------------------------------------

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close  = df["close"].shift(1)
    volume = df["volume"].shift(1)
    high   = df["high"].shift(1)
    low    = df["low"].shift(1)

    out["ret_1d"] = close / close.shift(1) - 1

    for w in [5, 21, 55, 89]:
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean()

    for f, s in [(5, 21), (13, 55), (21, 89)]:
        out[f"ma_cross_{f}_{s}"] = close.rolling(f).mean() / close.rolling(s).mean() - 1

    ret = close / close.shift(1) - 1
    out["vol_20"] = ret.rolling(20).std()
    out["vol_60"] = ret.rolling(60).std()

    mu60  = close.rolling(60).mean()
    std60 = close.rolling(60).std()
    out["spread_z"] = (close - mu60) / (std60 + 1e-8)
    out["resid_z"]  = (ret - ret.rolling(60).mean()) / (ret.rolling(60).std() + 1e-8)
    out["delta_corr"] = out["vol_20"] / (out["vol_60"] + 1e-8) - 1.0
    out["lag1_ret"]   = ret.shift(1)

    out["high_low_ratio"] = (high - low) / (close + 1e-8)
    vol_mu  = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mu) / (vol_std + 1e-8)

    return out


def _compute_target(df: pd.DataFrame) -> pd.Series:
    """Next-day % return: no lookahead."""
    return df["close"].pct_change().shift(-1)


def load_universe(data_dir: Path) -> dict:
    """Load all parquets, compute features + targets; return dict keyed by ticker."""
    parquets = sorted(f for f in data_dir.glob("*.parquet") if not f.stem.startswith("."))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    universe = {}
    for pq in parquets:
        ticker = pq.stem
        df = pd.read_parquet(pq)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()

        feats  = _compute_features(df)
        target = _compute_target(df)

        combined = feats.copy()
        combined["target"] = target
        combined = combined.dropna()
        universe[ticker] = combined

    return universe

# ---------------------------------------------------------------------------
# Dataset (same as ts_step010: sliding windows, common-date alignment)
# ---------------------------------------------------------------------------

class StockDataset(Dataset):
    """
    Returns x=[T, N, F] and y=[N] (next-day return for each stock).
    Tickers sorted alphabetically. Only common dates used.
    """
    def __init__(self, universe: dict, start_date: str, end_date: str,
                 T: int = T_LOOKBACK):
        self.T = T

        filtered = {
            t: df.loc[start_date:end_date]
            for t, df in universe.items()
            if len(df.loc[start_date:end_date]) >= T + 50
        }
        if not filtered:
            self.D = 0
            self.N = 0
            return

        tickers = sorted(filtered.keys())
        self.N  = len(tickers)
        frames  = [filtered[t] for t in tickers]

        common_idx = frames[0].index
        for fr in frames[1:]:
            common_idx = common_idx.intersection(fr.index)
        common_idx = common_idx.sort_values()
        print(f"  [{start_date}:{end_date}] {len(tickers)} tickers, "
              f"{len(common_idx)} common dates")

        feat_arr = np.stack(
            [fr.loc[common_idx, FEATURE_COLS].values for fr in frames], axis=1
        ).astype(np.float32)   # [D, N, F]
        tgt_arr = np.stack(
            [fr.loc[common_idx, "target"].values for fr in frames], axis=1
        ).astype(np.float32)   # [D, N]

        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_arr  = np.nan_to_num(tgt_arr,  nan=0.0, posinf=0.0, neginf=0.0)

        self.feat_arr = feat_arr
        self.tgt_arr  = tgt_arr
        self.D        = len(common_idx)

    def __len__(self):
        return max(0, self.D - self.T - 1)

    def __getitem__(self, idx):
        x = self.feat_arr[idx: idx + self.T]   # [T, N, F]
        y = self.tgt_arr[idx + self.T]          # [N]
        return torch.tensor(x), torch.tensor(y)

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_mask(T: int, T_pred: int, mask_ratio: float, N: int, F: int,
              device: torch.device) -> torch.Tensor:
    """
    Returns bool mask [T, N, F]. True = masked.
    Future T_pred cols (rightmost time steps) fully masked.
    Random 25% of past masked per (stock, feature) independently.
    """
    mask = torch.zeros(T, N, F, dtype=torch.bool, device=device)
    mask[-T_pred:] = True  # future: fully masked

    n_past_masked = int((T - T_pred) * mask_ratio)
    for n in range(N):
        for f_i in range(F):
            idx = torch.randperm(T - T_pred, device=device)[:n_past_masked]
            mask[idx, n, f_i] = True
    return mask

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ChannelwiseFC(nn.Module):
    """Per-channel linear across N_stocks spatial axis (Pathak key component).

    Allows global cross-stock information flow per feature channel.
    Without this, local CNN receptive field limits cross-stock context.
    """
    def __init__(self, spatial_size: int, n_channels: int):
        super().__init__()
        # shared linear applied independently per (batch, channel, T) position
        self.fc = nn.Linear(spatial_size, spatial_size, bias=False)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        B, C, T, N = x.shape
        x = x.reshape(B * C * T, N)
        x = self.fc(x)
        return x.reshape(B, C, T, N)


class GANEncoder(nn.Module):
    """Encoder: 3× Conv2d (F→16→32→64) + ChannelwiseFC on N axis."""
    def __init__(self, n_features: int, n_stocks: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_features, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.channel_fc = ChannelwiseFC(spatial_size=n_stocks, n_channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T, N] → z: [B, 64, T, N]
        return self.channel_fc(self.conv(x))


class GANDecoder(nn.Module):
    """Generator decoder: 3× ConvTranspose2d (64→32→16→n_features)."""
    def __init__(self, n_features: int):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, n_features, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, 64, T, N] → [B, F, T, N]
        return self.deconv(z)


class Generator(nn.Module):
    """Full generator: encoder + decoder + learnable mask token."""
    def __init__(self, n_features: int, n_stocks: int):
        super().__init__()
        self.encoder = GANEncoder(n_features, n_stocks)
        self.decoder = GANDecoder(n_features)
        # Learnable mask token, one value per feature channel
        self.mask_token = nn.Parameter(torch.zeros(n_features))

    def _apply_mask(self, x: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        """
        Replace masked positions with the learnable mask token.
        x:      [B, F, T, N]
        mask_b: [1, F, T, N] bool, broadcast over B
        """
        x_masked = x.clone()
        # mask_b expanded: [B, F, T, N]
        expanded = mask_b.expand_as(x_masked)
        # mask_token: [F] → broadcast to masked positions
        token = self.mask_token.view(1, -1, 1, 1).expand_as(x_masked)
        x_masked = torch.where(expanded, token, x_masked)
        return x_masked

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without masking — used for downstream features."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, F, T, N]
        mask_b: [1, 1, T, N]
        Returns reconstruction: [B, F, T, N]
        """
        x_masked = self._apply_mask(x, mask_b)
        z = self.encoder(x_masked)
        return self.decoder(z)


class Discriminator(nn.Module):
    """
    Small CNN discriminator.
    Input: [B, F, T, N] — a full reconstruction or real window.
    Output: [B, 1] — real/fake probability (sigmoid).
    3× Conv2d (F→32→64→1) + AdaptiveAvgPool2d(1).
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_features, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),   # → [B, 1, 1, 1]
            nn.Flatten(),              # → [B, 1]
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T, N] → [B, 1]
        return self.net(x)

# ---------------------------------------------------------------------------
# Downstream prediction head (frozen encoder + linear)
# ---------------------------------------------------------------------------

class ReturnHead(nn.Module):
    """
    Freeze encoder, train Linear(64*N_stocks → N_stocks).
    Encoder bottleneck [B, 64, T, N] → mean-pool over T → [B, 64*N] → [B, N].
    Design decision: flatten 64*N rather than per-stock Linear(64→1)
    to allow cross-stock interaction at the head level (more expressive).
    """
    def __init__(self, generator: Generator, n_stocks: int):
        super().__init__()
        for p in generator.parameters():
            p.requires_grad = False
        self.encoder = generator.encoder
        self.head = nn.Linear(64 * n_stocks, n_stocks)
        self.n_stocks = n_stocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T, N]
        z = self.encoder(x)              # [B, 64, T, N]
        z = z.mean(dim=2)               # [B, 64, N] — pool over T
        z = z.reshape(z.size(0), -1)    # [B, 64*N]
        return self.head(z)             # [B, N]

# ---------------------------------------------------------------------------
# Metrics (same as ts_step010/020)
# ---------------------------------------------------------------------------

def compute_sharpe(portfolio_returns: np.ndarray) -> float:
    if len(portfolio_returns) < 2:
        return 0.0
    mu    = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)
    if sigma < 1e-8:
        return 0.0
    return float(mu / sigma * math.sqrt(252))


def compute_metrics(preds: np.ndarray, actuals: np.ndarray):
    """preds, actuals: [n_samples, N_stocks]. Returns dir_acc, sharpe, mae."""
    pred_dir = np.sign(preds)
    act_dir  = np.sign(actuals)
    dir_acc  = float((pred_dir == act_dir).mean())

    positions = np.tanh(preds)
    port_ret  = (positions * actuals - TC).mean(axis=1)  # [n_samples]
    sharpe    = compute_sharpe(port_ret)

    mae = float(np.abs(preds - actuals).mean())
    return dir_acc, sharpe, mae


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_batch(loader_iter, loader, device):
    """Pull one batch from a persistent iterator; restart on exhaustion."""
    try:
        x, y = next(loader_iter[0])
    except StopIteration:
        loader_iter[0] = iter(loader)
        x, y = next(loader_iter[0])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_downstream(head: ReturnHead, loader: DataLoader, device: torch.device):
    """Evaluate frozen-encoder + head on val set."""
    head.eval()
    all_preds, all_actuals = [], []
    for x_raw, y in loader:
        # x_raw: [B, T, N, F] → permute to [B, F, T, N]
        x = x_raw.permute(0, 3, 1, 2).to(device)
        y = y.to(device)
        pred = head(x)
        all_preds.append(pred.cpu().numpy())
        all_actuals.append(y.cpu().numpy())
    preds   = np.concatenate(all_preds,   axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    dir_acc, sharpe, mae = compute_metrics(preds, actuals)
    return mae, dir_acc, sharpe

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device      = torch.device(args.device)
    data_dir    = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("ts_step021 — GAN Context Encoder")
    print(f"  device={args.device}  pretrain={args.pretrain_epochs}ep  "
          f"finetune={args.epochs}ep  batch={args.batch_size}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\nLoading universe from {data_dir} ...")
    universe = load_universe(data_dir)
    print(f"Tickers loaded: {len(universe)}")

    SPLITS = [(TRAIN_START, TRAIN_END), (VAL_START, VAL_END)]
    valid_tickers = {
        t for t, df in universe.items()
        if all(len(df.loc[s:e]) >= T_LOOKBACK + 50 for s, e in SPLITS)
    }
    universe = {t: df for t, df in universe.items() if t in valid_tickers}
    print(f"Tickers after cross-split filter: {len(universe)}")

    train_ds = StockDataset(universe, TRAIN_START, TRAIN_END)
    val_ds   = StockDataset(universe, VAL_START,   VAL_END)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    N = train_ds.N
    F_feat = len(FEATURE_COLS)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    G = Generator(n_features=F_feat, n_stocks=N).to(device)
    D = Discriminator(n_features=F_feat).to(device)

    n_params_G = count_params(G)
    n_params_D = count_params(D)
    print(f"\nGenerator params:     {n_params_G:,}")
    print(f"Discriminator params: {n_params_D:,}")

    # Standard GAN betas (0.5, 0.999)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCELoss()

    # Persistent loader iterators (avoid recreating DataLoader each step)
    train_iter = [iter(train_loader)]

    # ------------------------------------------------------------------
    # GAN Pretraining
    # ------------------------------------------------------------------
    WARMUP_EPOCHS = 10   # reconstruction only, no adversarial
    LAMBDA_RECON  = 0.999
    LAMBDA_ADV    = 0.001

    print(f"\n{'epoch':>6} | {'d_loss':>8} | {'g_loss':>8} | {'recon_loss':>10} | "
          f"{'val_mae':>8} | {'val_dir_acc':>11} | {'val_sharpe':>10}")
    print("-" * 78)

    pretrain_history = []
    best_pretrain_recon = float("inf")
    best_G_state = None

    for epoch in range(1, args.pretrain_epochs + 1):
        G.train()
        D.train()
        t0 = time.time()

        x_raw, _ = get_batch(train_iter, train_loader, device)
        # [B, T, N, F] → [B, F, T, N]
        x_real = x_raw.permute(0, 3, 1, 2)
        B_cur  = x_real.size(0)

        # Build mask: [T, N, F] → permute to [1, F, T, N] for x=[B, F, T, N]
        mask_tnf = make_mask(T_LOOKBACK, T_PRED, MASK_RATIO, N, F_feat, device)
        mask_b   = mask_tnf.permute(2, 0, 1).unsqueeze(0)  # [1, F, T, N]

        # ---------- Discriminator step ----------
        if epoch > WARMUP_EPOCHS:
            opt_D.zero_grad()
            real_labels = torch.ones(B_cur,  1, device=device)
            fake_labels = torch.zeros(B_cur, 1, device=device)

            with torch.no_grad():
                x_fake = G(x_real, mask_b)

            d_real = D(x_real)
            d_fake = D(x_fake)
            loss_D = 0.5 * bce(d_real, real_labels) + 0.5 * bce(d_fake, fake_labels)
            loss_D.backward()
            opt_D.step()
            d_loss_val = loss_D.item()
        else:
            d_loss_val = float("nan")

        # ---------- Generator step ----------
        opt_G.zero_grad()
        x_fake = G(x_real, mask_b)

        # Reconstruction loss on masked positions only
        # Permute to [B, T, N, F] for masked indexing
        x_real_p = x_real.permute(0, 2, 3, 1)   # [B, T, N, F]
        x_fake_p = x_fake.permute(0, 2, 3, 1)   # [B, T, N, F]
        mask_exp  = mask_tnf.unsqueeze(0).expand_as(x_real_p)  # [B, T, N, F]
        loss_recon = F.mse_loss(x_fake_p[mask_exp], x_real_p[mask_exp])

        if epoch > WARMUP_EPOCHS:
            real_labels_g = torch.ones(B_cur, 1, device=device)
            d_fake_g      = D(x_fake)
            loss_adv      = bce(d_fake_g, real_labels_g)
            loss_G        = LAMBDA_RECON * loss_recon + LAMBDA_ADV * loss_adv
        else:
            loss_adv = torch.tensor(0.0, device=device)
            loss_G   = loss_recon

        loss_G.backward()
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_G.step()

        g_loss_val     = loss_G.item()
        recon_loss_val = loss_recon.item()

        # Save best generator by reconstruction loss
        if recon_loss_val < best_pretrain_recon:
            best_pretrain_recon = recon_loss_val
            best_G_state = {k: v.cpu().clone() for k, v in G.state_dict().items()}

        pretrain_history.append({
            "epoch":      epoch,
            "d_loss":     round(d_loss_val, 6) if not math.isnan(d_loss_val) else None,
            "g_loss":     round(g_loss_val, 6),
            "recon_loss": round(recon_loss_val, 6),
        })

        # Print every epoch (same cadence requested in spec)
        print(f"{epoch:>6} | {d_loss_val:>8.5f} | {g_loss_val:>8.5f} | "
              f"{recon_loss_val:>10.6f} | {'—':>8} | {'—':>11} | {'—':>10}")

    # Restore best generator
    if best_G_state is not None:
        G.load_state_dict(best_G_state)
    ckpt_path = CKPT_DIR / "ts_step021_gan_best_G.pt"
    torch.save(G.state_dict(), ckpt_path)
    print(f"\nBest generator saved → {ckpt_path}")

    # ------------------------------------------------------------------
    # Downstream fine-tuning (frozen encoder + linear head)
    # ------------------------------------------------------------------
    print(f"\nDownstream fine-tuning ({args.epochs} epochs, frozen encoder)...")

    head     = ReturnHead(G, n_stocks=N).to(device)
    n_head   = count_params(head)
    opt_head = torch.optim.Adam(head.head.parameters(), lr=1e-4)

    print(f"Head trainable params: {n_head:,}")

    finetune_history = []
    best_val_sharpe  = -1e9
    best_head_state  = None

    train_iter_ft = [iter(train_loader)]

    print(f"\n{'epoch':>6} | {'d_loss':>8} | {'g_loss':>8} | {'recon_loss':>10} | "
          f"{'val_mae':>8} | {'val_dir_acc':>11} | {'val_sharpe':>10}")
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        head.train()
        # re-enable grad for head only (encoder frozen inside ReturnHead)
        x_raw, y = get_batch(train_iter_ft, train_loader, device)
        x = x_raw.permute(0, 3, 1, 2).to(device)   # [B, F, T, N]

        opt_head.zero_grad()
        pred   = head(x)                             # [B, N]
        loss_h = F.mse_loss(pred, y)
        loss_h.backward()
        opt_head.step()

        # Eval every epoch
        val_mae, val_dir_acc, val_sharpe = eval_downstream(head, val_loader, device)
        finetune_history.append({
            "epoch":       epoch,
            "train_mse":   round(loss_h.item(), 6),
            "val_mae":     round(val_mae, 6),
            "val_dir_acc": round(val_dir_acc, 4),
            "val_sharpe":  round(val_sharpe,  4),
        })

        # Print in unified table format (d_loss/g_loss/recon_loss are N/A here)
        print(f"{epoch:>6} | {'—':>8} | {'—':>8} | {'—':>10} | "
              f"{val_mae:>8.5f} | {val_dir_acc:>11.4f} | {val_sharpe:>10.4f}")

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    # Final evaluation
    val_mae, val_dir_acc, val_sharpe = eval_downstream(head, val_loader, device)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "step":                  "ts_step021_gan_encoder",
        "model":                 "GAN_ContextEncoder",
        "n_params_generator":    n_params_G,
        "n_params_discriminator":n_params_D,
        "n_params_head":         n_head,
        "n_stocks":              N,
        "n_features":            F_feat,
        "T_lookback":            T_LOOKBACK,
        "T_pred_masked":         T_PRED,
        "mask_ratio_past":       MASK_RATIO,
        "pretrain_epochs":       args.pretrain_epochs,
        "finetune_epochs":       args.epochs,
        "warmup_epochs":         WARMUP_EPOCHS,
        "lambda_recon":          LAMBDA_RECON,
        "lambda_adv":            LAMBDA_ADV,
        "lr":                    args.lr,
        "batch_size":            args.batch_size,
        "seed":                  args.seed,
        "device":                args.device,
        "best_pretrain_recon_loss": round(best_pretrain_recon, 6),
        "best_val_sharpe":       round(best_val_sharpe, 4),
        "final_val_dir_acc":     round(val_dir_acc, 4),
        "final_val_mae":         round(val_mae, 6),
        "final_val_sharpe":      round(val_sharpe, 4),
        "pretrain_history":      pretrain_history,
        "finetune_history":      finetune_history,
    }

    out_path = results_dir / "ts_step021_gan_encoder.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n=== ts_step021 Results ===")
    print(f"  Generator params:     {n_params_G:,}")
    print(f"  Discriminator params: {n_params_D:,}")
    print(f"  Best pretrain recon:  {best_pretrain_recon:.6f}")
    print(f"  Best val Sharpe:      {best_val_sharpe:.4f}")
    print(f"  Final val dir_acc:    {val_dir_acc:.4f}")
    print(f"  Final val MAE:        {val_mae:.6f}")
    print(f"  Results:              {out_path}")


if __name__ == "__main__":
    main()
