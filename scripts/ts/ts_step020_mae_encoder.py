"""ts_step020 — MAE-style masked context encoder on 2D stock×time matrix.

Architecture:
  Encoder: CNN (3 conv layers, 16→32→64 channels) + channel-wise FC
  Decoder: transposed conv (mirror of encoder)
  Mask:    future T_pred days = fully masked; random 25% of past = masked
  Loss:    MSE on masked positions only (MAE-style reconstruction)

After pretraining, encoder features used downstream (frozen encoder + linear head)
for next-day return prediction.

Usage:
    python scripts/ts/ts_step020_mae_encoder.py [--device mps] [--epochs 100]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = ROOT / "data" / "ts" / "raw"
PROC_DIR  = ROOT / "data" / "ts" / "processed"
CKPT_DIR  = ROOT / "data" / "ts" / "checkpoints"
RESULTS_DIR = ROOT / "results" / "ts"
for d in (PROC_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

T_LOOKBACK = 60    # input window (trading days)
T_PRED     = 5     # future days fully masked (predict these)
MASK_RATIO = 0.25  # fraction of past/present to randomly mask
F_FEATURES = 11    # feature channels (MA ratios only for now; expand after ts_step003)
TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"


# -------------------------------------------------------------------
# Feature engineering (self-contained, no lookahead)
# -------------------------------------------------------------------

MA_WINDOWS = [3, 5, 8, 13, 21, 34, 55, 89, 144]  # 9 windows

def build_features(close: pd.Series) -> np.ndarray:
    """Return [T, F] feature array for one stock."""
    ret = close.pct_change().shift(1)  # no lookahead
    feats = [ret.values]
    for w in MA_WINDOWS:
        ma = close.shift(1).rolling(w).mean()
        feats.append((close.shift(1) / ma - 1).values)
    arr = np.stack(feats, axis=1)  # [T, F]
    return arr


def load_dataset(device: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Returns train tensor, val tensor, ticker list. Shape: [T_total, N_stocks, F]."""
    close_dict = {}
    for fpath in sorted(f for f in RAW_DIR.glob("*.parquet") if not f.stem.startswith(".")):
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        close_dict[fpath.stem] = df["close"].sort_index()

    # Pre-filter: only tickers with data in BOTH train and val splits (avoids post-IPO empties)
    min_train = 200
    min_val   = 100
    valid = {
        t: s for t, s in close_dict.items()
        if len(s.loc[:TRAIN_END].dropna()) >= min_train
        and len(s.loc[VAL_START:VAL_END].dropna()) >= min_val
    }
    close_dict = valid

    # Align on common dates
    closes = pd.DataFrame(close_dict).dropna(how="any")
    tickers = list(closes.columns)
    dates = closes.index

    # Build feature tensor [T, N, F]
    arrays = []
    for t in tickers:
        arr = build_features(closes[t])
        arrays.append(arr)
    data = np.stack(arrays, axis=1)  # [T, N, F]

    # Walk-forward split
    train_mask = dates <= TRAIN_END
    val_mask   = (dates >= VAL_START) & (dates <= VAL_END)

    def to_tensor(mask):
        sub = data[mask]
        # Replace NaN with 0 (NaN from rolling windows at start)
        sub = np.nan_to_num(sub, nan=0.0)
        return torch.tensor(sub, dtype=torch.float32)

    return to_tensor(train_mask), to_tensor(val_mask), tickers


# -------------------------------------------------------------------
# Masking
# -------------------------------------------------------------------

def make_mask(T: int, T_pred: int, mask_ratio: float, N: int,
              device: str) -> torch.Tensor:
    """Returns bool mask [T, N]. True = masked. Applied uniformly across F."""
    mask = torch.zeros(T, N, dtype=torch.bool, device=device)
    mask[-T_pred:] = True
    n_past_masked = int((T - T_pred) * mask_ratio)
    for n in range(N):
        idx = torch.randperm(T - T_pred, device=device)[:n_past_masked]
        mask[idx, n] = True
    return mask


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class ChannelwiseFC(nn.Module):
    """Per-channel linear across spatial positions (Pathak's key component).

    Allows information to flow between all stock positions per channel —
    equivalent to the channel-wise FC in Context Encoders. Without this,
    CNN can't propagate information across the N_stocks axis globally.
    """
    def __init__(self, spatial_size: int, n_channels: int):
        super().__init__()
        # One FC per channel: spatial_size → spatial_size
        self.fc = nn.Linear(spatial_size, spatial_size, bias=False)
        self.n_channels = n_channels

    def forward(self, x):
        # x: [B, C, H, W] — apply FC independently per channel across W
        B, C, H, W = x.shape
        x = x.contiguous().view(B * C * H, W)
        x = self.fc(x)
        return x.view(B, C, H, W)


class MAEEncoder(nn.Module):
    def __init__(self, n_stocks: int, n_features: int = F_FEATURES):
        super().__init__()
        # Input: [B, F, T, N] — treat (T, N) as 2D "image"
        self.encoder = nn.Sequential(
            nn.Conv2d(n_features, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.channel_fc = ChannelwiseFC(spatial_size=n_stocks, n_channels=64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, n_features, kernel_size=3, padding=1),
        )
        # Learnable mask token (not zeros — avoids spurious signal from empty regions)
        self.mask_token = nn.Parameter(torch.zeros(n_features))

    def encode(self, x):
        return self.channel_fc(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, F, T, N], mask: [1, 1, T, N] bool. Returns reconstruction [B, F, T, N]."""
        token = self.mask_token.view(1, -1, 1, 1).expand_as(x)
        x_masked = torch.where(mask.expand_as(x), token, x)
        z = self.encode(x_masked)
        return self.decode(z)


# -------------------------------------------------------------------
# Downstream head (frozen encoder + linear)
# -------------------------------------------------------------------

class ReturnHead(nn.Module):
    def __init__(self, encoder: MAEEncoder, n_stocks: int, T: int = T_LOOKBACK):
        super().__init__()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder
        # Pool over time, then linear per stock
        self.head = nn.Linear(64, 1)  # per spatial position (stock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x)   # [B, 64, T, N]
        z = z.mean(dim=2)            # [B, 64, N] — pool over time
        z = z.permute(0, 2, 1)       # [B, N, 64]
        return self.head(z).squeeze(-1)  # [B, N]


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def sliding_windows(data: torch.Tensor, T_win: int) -> torch.Tensor:
    """data: [T, N, F]. Returns [n_windows, T_win, N, F]."""
    T = len(data)
    windows = torch.stack([data[i:i+T_win] for i in range(T - T_win + 1)])
    return windows


def compute_sharpe(portfolio_returns: np.ndarray) -> float:
    if len(portfolio_returns) < 2:
        return 0.0
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns) + 1e-8
    return float(mu / sigma * np.sqrt(252))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epochs-finetune", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device

    print("ts_step020 — MAE Context Encoder")
    print(f"  device={dev}  epochs={args.epochs}  batch={args.batch_size}\n")

    train_data, val_data, tickers = load_dataset(dev)
    N = train_data.shape[1]
    F_feat = train_data.shape[2]
    print(f"  Loaded: train={train_data.shape}  val={val_data.shape}  N_stocks={N}  F={F_feat}")

    train_wins = sliding_windows(train_data, T_LOOKBACK)  # [W, T, N, F]
    val_wins   = sliding_windows(val_data,   T_LOOKBACK)
    print(f"  Windows: train={len(train_wins)}  val={len(val_wins)}")

    model = MAEEncoder(n_stocks=N, n_features=F_feat).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def get_batch(windows, batch_size):
        idx = torch.randperm(len(windows))[:batch_size]
        return windows[idx].to(dev)  # [B, T, N, F]

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch = get_batch(train_wins, args.batch_size)
        # [B, T, N, F] → [B, F, T, N]
        x = batch.permute(0, 3, 1, 2)
        mask = make_mask(T_LOOKBACK, T_PRED, MASK_RATIO, N, dev)
        mask_b = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, N]

        opt.zero_grad()
        recon = model(x, mask_b)  # [B, F, T, N]
        x_true = x.permute(0, 2, 3, 1)        # [B, T, N, F]
        recon_p = recon.permute(0, 2, 3, 1)   # [B, T, N, F]
        mask_b_exp = mask.unsqueeze(0).unsqueeze(-1).expand_as(x_true)  # [B, T, N, F]

        loss = F.mse_loss(recon_p[mask_b_exp], x_true[mask_b_exp])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = get_batch(val_wins, args.batch_size)
                xv = val_batch.permute(0, 3, 1, 2)
                rv = model(xv, mask_b)
                xv_p = xv.permute(0, 2, 3, 1)
                rv_p = rv.permute(0, 2, 3, 1)
                val_loss = F.mse_loss(rv_p[mask_b_exp], xv_p[mask_b_exp]).item()
            print(f"  Epoch {epoch:4d} | train_loss={loss.item():.5f} | val_loss={val_loss:.5f}")
            history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CKPT_DIR / "ts_step020_mae_best.pt")

    # Fine-tune downstream head
    print("\nDownstream fine-tuning (frozen encoder + return head)...")
    ckpt = CKPT_DIR / "ts_step020_mae_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    head = ReturnHead(model, N).to(dev)
    head_opt = torch.optim.Adam(head.head.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs_finetune + 1):
        head.train()
        # Use windows excluding the last T_PRED days as target
        batch = get_batch(train_wins[:-T_PRED], args.batch_size)
        x = batch[:, :T_LOOKBACK-T_PRED].permute(0, 3, 1, 2)
        # Target: return over next day (last row of window)
        target = batch[:, -1, :, 0]  # [B, N] — first feature is ret_1d at t+1

        head_opt.zero_grad()
        pred = head(x)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        head_opt.step()

    # Val evaluation
    head.eval()
    all_dir_acc, all_rets = [], []
    with torch.no_grad():
        for i in range(0, min(len(val_wins)-T_PRED, 200), args.batch_size):
            batch = val_wins[i:i+args.batch_size].to(dev)
            x = batch[:, :T_LOOKBACK-T_PRED].permute(0, 3, 1, 2)
            target = batch[:, -1, :, 0]
            pred = head(x)
            dir_acc = (torch.sign(pred) == torch.sign(target)).float().mean().item()
            all_dir_acc.append(dir_acc)
            port_ret = (torch.tanh(pred) * target - 0.001).mean(dim=-1)
            all_rets.extend(port_ret.cpu().numpy().tolist())

    mean_dir_acc = float(np.mean(all_dir_acc))
    sharpe = compute_sharpe(np.array(all_rets))

    results = {
        "step": "ts_step020",
        "model": "MAE_ContextEncoder",
        "n_params": n_params,
        "n_stocks": N,
        "T_lookback": T_LOOKBACK,
        "T_pred_masked": T_PRED,
        "mask_ratio_past": MASK_RATIO,
        "pretrain_epochs": args.epochs,
        "finetune_epochs": args.epochs_finetune,
        "best_val_recon_loss": best_val_loss,
        "val_directional_acc": round(mean_dir_acc, 4),
        "val_sharpe": round(sharpe, 4),
        "history": history,
    }
    out_path = RESULTS_DIR / "ts_step020_mae_encoder.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== ts_step020 Results ===")
    print(f"  Params:            {n_params:,}")
    print(f"  Val recon loss:    {best_val_loss:.5f}")
    print(f"  Val dir accuracy:  {mean_dir_acc:.3f}  (baseline: 0.500)")
    print(f"  Val Sharpe:        {sharpe:.3f}")
    print(f"  Checkpoint:        {CKPT_DIR / 'ts_step020_mae_best.pt'}")
    print(f"  Results:           {out_path}")


if __name__ == "__main__":
    main()
