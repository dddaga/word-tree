"""Step 989: SGNNET as GPT-2-small FFN replacement — T0 scout.

MOTIVATION
==========
Original brief §1: "We are targeting the feed-forward (FFN) sub-layers of transformers
first." This is the FOUNDING VISION — never tested in 988 experiments. This T0 answers:
can SGNNET functionally replicate a transformer FFN at all?

Task: MSE regression x_ffn → y_ffn where x,y are 768-dim GPT-2-small layer-6 activations.
Not text classification — pure functional FFN replacement (as the brief proposed).

CONFIGS
=======
  Ref_mlp    : 2-layer MLP baseline (d=768→256→768, GELU, ~200K params)
  A_sgnnet   : SGNNET_SmallWorld N_in=768, N=512, N_out=768, D=16, K_iter=5
  B_sgnnet_d32: SGNNET_SmallWorld N_in=768, N=512, N_out=768, D=32, K_iter=5

Advance criterion: A/B val_mse <= Ref_mlp val_mse * 2.0  AND  cos_sim >= 0.5
(Generous threshold for T0 — any signal of learning advances to T1.)

Usage: d_env/bin/python3 scripts/train_step989_ffn_distil_t0.py [--device cuda]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.sgnnet.model_smallworld import SGNNET_SmallWorld

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch",  type=int, default=256)
parser.add_argument("--seed",   type=int, default=42)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS    = args.epochs
BATCH     = args.batch
SEED      = args.seed
DATA_PATH = ROOT / "data" / "gpt2_ffn_layer6.pt"
OUT_PATH  = ROOT / "results" / f"train_step989_ffn_distil_t0_seed{SEED}.json"

torch.manual_seed(SEED)


# ── Data ──────────────────────────────────────────────────────────────────────
def make_loaders(path: Path, batch: int, seed: int):
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run first: d_env/bin/python3 scripts/extract_gpt2_ffn_data.py")
        sys.exit(1)
    d    = torch.load(path, map_location="cpu", weights_only=True)
    x, y = d["x"].float(), d["y"].float()
    T    = x.shape[0]
    rng  = torch.Generator().manual_seed(seed)
    idx  = torch.randperm(T, generator=rng)
    split = int(T * 0.8)
    tr_ds = TensorDataset(x[idx[:split]], y[idx[:split]])
    va_ds = TensorDataset(x[idx[split:]], y[idx[split:]])
    pin   = (DEVICE.type == "cuda")
    tr_l  = DataLoader(tr_ds, batch_size=batch, shuffle=True,  pin_memory=pin)
    va_l  = DataLoader(va_ds, batch_size=batch, shuffle=False, pin_memory=pin)
    print(f"Data: {T:,} pairs  train={split:,}  val={T-split:,}  dim=768")
    return tr_l, va_l


# ── Models ────────────────────────────────────────────────────────────────────
def make_ref_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 768))


def make_sgnnet(D: int = 16) -> SGNNET_SmallWorld:
    # N_in=768 → 768 "input neurons" (scalar activations of the FFN hidden state)
    # C_ho_mask is [N_hidden=512, N_out=768] — dense enough at this scale
    return SGNNET_SmallWorld(
        N_hidden=512, N_out=768, D=D, N_in=768,
        K_in=8, K_local=2, K_random=2, n_groups=64,
        K_iter=5, sparsity=0.0, norm_mode="l2", encoding_mode="fourier",
    )


CONFIGS = {
    "Ref_mlp":      make_ref_mlp,
    "A_sgnnet_d16": lambda: make_sgnnet(D=16),
    "B_sgnnet_d32": lambda: make_sgnnet(D=32),
}


# ── Training loop ─────────────────────────────────────────────────────────────
def train_loop(model, opt, tr_loader, va_loader, epochs, device):
    history = []
    for ep in range(epochs):
        model.train()
        total_mse = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_mse += loss.item()
        # val
        model.eval()
        with torch.no_grad():
            val_mse  = sum(F.mse_loss(model(xb.to(device)), yb.to(device)).item()
                           for xb, yb in va_loader) / len(va_loader)
            all_pred, all_y = [], []
            for xb, yb in va_loader:
                p = model(xb.to(device))
                all_pred.append(F.normalize(p, dim=-1))
                all_y.append(F.normalize(yb.to(device), dim=-1))
            cos_sim = (torch.cat(all_pred) * torch.cat(all_y)).sum(-1).mean().item()
        history.append({
            "epoch": ep + 1,
            "train_mse": round(total_mse / len(tr_loader), 5),
            "val_mse":   round(val_mse, 5),
            "cos_sim":   round(cos_sim, 4),
        })
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1:3d}  val_mse={val_mse:.4f}  cos_sim={cos_sim:.4f}", flush=True)
    return history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tr_l, va_l = make_loaders(DATA_PATH, BATCH, SEED)

    print(f"\n{'='*65}")
    print(f"step989 — SGNNET as GPT-2 FFN replacement (founding vision T0)")
    print(f"  Task:    MSE regression x_ffn(768) → y_ffn(768)")
    print(f"  Device:  {DEVICE}  epochs={EPOCHS}  batch={BATCH}  seed={SEED}")
    print(f"  Advance: val_mse <= Ref*2.0  AND  cos_sim >= 0.5")
    print(f"{'='*65}\n")

    results = {"step": "step989", "seed": SEED, "device": str(DEVICE), "configs": {}}

    for name, builder in CONFIGS.items():
        model = builder().to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"\n{'─'*55}")
        print(f"Config: {name}  params={n_p:,}", flush=True)
        t0 = time.time()
        hist = train_loop(model, opt, tr_l, va_l, EPOCHS, DEVICE)
        elapsed = time.time() - t0
        best_ep   = min(range(len(hist)), key=lambda i: hist[i]["val_mse"])
        best_mse  = hist[best_ep]["val_mse"]
        best_cos  = hist[best_ep]["cos_sim"]
        print(f"  DONE: best_val_mse={best_mse:.5f}  cos_sim@best={best_cos:.4f}"
              f"  ep{best_ep+1}  {elapsed:.0f}s")
        results["configs"][name] = {
            "n_params": n_p, "best_val_mse": best_mse,
            "best_cos_sim": best_cos, "best_ep": best_ep + 1,
            "elapsed_s": round(elapsed, 1), "history": hist,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── Summary table ─────────────────────────────────────────────────────────
    ref_mse = results["configs"]["Ref_mlp"]["best_val_mse"]
    print(f"\n{'='*65}")
    print(f"step989 SUMMARY  (Ref_mlp val_mse={ref_mse:.5f})")
    print(f"  {'Config':<22}  {'params':>8}  {'val_mse':>9}  {'cos_sim':>8}  verdict")
    print(f"  {'─'*60}")
    for name, r in results["configs"].items():
        adv = ("Ref" if name == "Ref_mlp"
               else "ADVANCE" if r["best_val_mse"] <= ref_mse * 2.0 and r["best_cos_sim"] >= 0.5
               else "NEUTRAL/KILL")
        print(f"  {name:<22}  {r['n_params']:>8,}  {r['best_val_mse']:>9.5f}"
              f"  {r['best_cos_sim']:>8.4f}  {adv}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
