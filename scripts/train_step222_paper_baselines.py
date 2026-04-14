"""Step 222: Paper baselines — MLP, Random Projection, Linear Probe vs SGNNET.

MOTIVATION
==========
Publication requires clean ablation baselines that isolate SGNNET's contribution.
Three control conditions:
  Ref : step199 SGNNET (N=2048 D=16 K_hh=2 K_iter=5 AH=1.0, 0.98M FLOPs, ~67K params)
  A   : MLP ~67K params — 2-hidden-layer (ReLU+BN) matching step199 param count
  B   : Random projection + linear — fixed random proj to D_rp dims, learned linear head
        Same ~67K param count as step199. Tests whether gather-sum-normalize adds value.
  C   : Linear probe — single learned 25088→10 layer. Lower bound on VGG16 feature quality.

Key question: is SGNNET's iterative routing doing anything a random projection doesn't?

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0 scouts)
"""
from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,A). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 983,040 ≈ 0.98M
OUT_PATH = ROOT / "results" / "train_step222_paper_baselines.json"

# Target param count ~67K (matching step199 SGNNET)
# step199 SGNNET params: N_IN*K_IN (embed) + N*D (W_pos) + N_OUT*D (fc_out) + misc
# Empirically ~67K. We'll target this for MLP and random proj.
TARGET_PARAMS = 67_000


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# MLP baseline: 25088 → h1 → h2 → 10, ~67K params (ReLU + BatchNorm)
# Solve for h1 = h2 = h such that param count ≈ TARGET_PARAMS:
#   25088*h + h + h*h + h + h*10 + 10 + 2*h (BN) ≈ TARGET_PARAMS
# Since N_IN=25088 >> h, dominated by first layer: h ≈ TARGET_PARAMS / N_IN ≈ 2.67
# That's too small. With symmetric layers: 25088*h + h*h + h*10 ≈ 67K → h ≈ 2.
# Instead use h1 small and h2 = remainder.
# With h1=2: 25088*2 + 2*10 = 50196 first+last → 67000-50196 = 16804 for h1→h2: h2 = 16804/2 ≈ 8402
# With h1=2, h2=8402: total ≈ 25088*2 + 2*8402 + 8402*10 = 50176 + 16804 + 84020 = 151K — too big.
# Better: single bottleneck. 25088→h→10. Params: 25088*h + h*10. Target: 25088*h ≈ 67K → h≈2.6.
# Not viable. MLP at 67K is essentially a linear probe with tiny hidden layer.
# Design: two layers with h chosen so total ≈ 67K.
#   Layer 1: N_IN → h (+ BN: 2h) → ReLU
#   Layer 2: h → h (+ BN: 2h) → ReLU
#   Layer 3: h → N_OUT
#   Params: N_IN*h + h + 2*h + h*h + h + 2*h + h*N_OUT + N_OUT
#         = N_IN*h + h*h + h*N_OUT + overhead(~8h + N_OUT)
# For N_IN=25088, N_OUT=10: dominated by N_IN*h.
# h=2: 25088*2 + 4 + 20 + ~24 = 50192 + overhead ≈ 50K. Slightly under 67K.
# h=2 is the only integer that keeps first-layer params < 67K.
# We'll use h=2 hidden dim and accept the actual param count (document in output).
# Note: this is honest — at 67K params MLP is fundamentally bottlenecked by N_IN.
# ---------------------------------------------------------------------------

def _mlp_hidden_dim(target=TARGET_PARAMS, n_in=N_IN, n_out=N_OUT):
    """Find h (two equal hidden layers) such that params ≈ target.
    Equation: n_in*h + h*h + h*n_out + 8*h + n_out ≈ target
    Solve numerically.
    """
    for h in range(1, 500):
        p = n_in * h + h * h + h * n_out + 8 * h + n_out
        if p >= target:
            # pick whichever of h-1, h is closer
            if h > 1:
                p_prev = n_in * (h-1) + (h-1)**2 + (h-1)*n_out + 8*(h-1) + n_out
                return (h-1) if abs(p_prev - target) < abs(p - target) else h
            return h
    return 2  # fallback


class MLPBaseline(nn.Module):
    """Standard 2-hidden-layer MLP with ReLU + BatchNorm. ~67K params target."""

    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _mlp_hidden_dim()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(n_in, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, n_out, bias=True),
        )
        # Trainer expects these properties
        self._W_pos   = nn.Parameter(torch.zeros(1, 1))   # dummy — satisfies Trainer check
        self._W_phase = nn.Parameter(torch.zeros(1, 1))   # dummy

    @property
    def W_pos(self):   return self._W_pos
    @property
    def W_phase(self): return self._W_phase

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Random Projection + Linear baseline
# Fixed random projection from N_IN → D_rp, then learned linear → N_OUT.
# Param count: D_rp * N_OUT (+ N_OUT bias) — the projection is fixed/frozen.
# To match 67K params: D_rp = (67K - N_OUT) / N_OUT ≈ 6690.
# Note: 67K params comes from the random-proj-then-linear head being 6690→10 = 66900 params.
# The random projection is NOT counted (frozen).
# ---------------------------------------------------------------------------

def _rp_dim(target=TARGET_PARAMS, n_out=N_OUT):
    """D_rp such that D_rp * N_OUT + N_OUT ≈ target."""
    return max(1, (target - n_out) // n_out)


class RandomProjectionLinear(nn.Module):
    """Fixed random projection + learned linear classifier.

    Tests: does gather-sum-normalize add value over a single random projection?
    The random matrix is sampled once and frozen — only the linear head learns.
    Param count equals the linear head only (frozen proj not counted).
    """

    def __init__(self, n_in=N_IN, n_out=N_OUT, d_rp=None, seed=SEED):
        super().__init__()
        if d_rp is None:
            d_rp = _rp_dim()
        self.d_rp = d_rp
        # Fixed random projection (no grad)
        torch.manual_seed(seed)
        rp = torch.randn(n_in, d_rp) / math.sqrt(n_in)
        self.register_buffer("proj", rp)
        # Learned linear head
        self.head = nn.Linear(d_rp, n_out, bias=True)
        # Trainer compatibility
        self._W_pos   = nn.Parameter(torch.zeros(1, 1))
        self._W_phase = nn.Parameter(torch.zeros(1, 1))

    @property
    def W_pos(self):   return self._W_pos
    @property
    def W_phase(self): return self._W_phase

    def forward(self, x):
        z = F.relu(x @ self.proj)          # [B, d_rp] — one nonlinear step
        return self.head(z)                # [B, 10]


# ---------------------------------------------------------------------------
# Linear probe: single 25088 → 10 layer. No hidden layers. Lower bound.
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single learned linear layer. Lower bound on VGG16 feature quality."""

    def __init__(self, n_in=N_IN, n_out=N_OUT):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out, bias=True)
        self._W_pos   = nn.Parameter(torch.zeros(1, 1))
        self._W_phase = nn.Parameter(torch.zeros(1, 1))

    @property
    def W_pos(self):   return self._W_pos
    @property
    def W_phase(self): return self._W_phase

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_sgnnet():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_model(config_key):
    torch.manual_seed(SEED)
    if config_key == "Ref":
        return build_sgnnet()
    elif config_key == "A":
        h = _mlp_hidden_dim()
        return MLPBaseline(hidden_dim=h)
    elif config_key == "B":
        d = _rp_dim()
        return RandomProjectionLinear(d_rp=d)
    elif config_key == "C":
        return LinearProbe()
    else:
        raise ValueError(f"Unknown config: {config_key}")


def main():
    all_keys = ["Ref", "A", "B", "C"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    h_dim = _mlp_hidden_dim()
    rp_dim = _rp_dim()

    labels = {
        "Ref": f"Ref: SGNNET step199 (N={N} D={D} K_hh={K_HH} K_iter={K_ITER} AH=1.0)",
        "A":   f"A: MLP 2-hidden (h={h_dim}, ReLU+BN, ~{N_IN*h_dim + h_dim**2 + h_dim*N_OUT:,}p)",
        "B":   f"B: RandProj (fixed {N_IN}→{rp_dim}) + linear head (~{rp_dim*N_OUT:,}p learned)",
        "C":   f"C: Linear probe ({N_IN}→{N_OUT}, {N_IN*N_OUT + N_OUT:,}p)",
    }

    print(f"\n{'='*70}")
    print(f"Step 222 — Paper baselines (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Question: does SGNNET iterative routing beat random projection?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        model = build_model(key).to(DEVICE)
        n_p = _count_params(model)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 222 SUMMARY — Paper baselines")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key}: {r['top1_best']:.4f}  params={r['n_params']:,}{delta}")

    if "Ref" in results and "B" in results:
        gap = results["Ref"]["top1_best"] - results["B"]["top1_best"]
        print(f"\n  SGNNET vs RandProj gap: {gap:+.4f}pp")
        print(f"  (positive = SGNNET routing adds value beyond random projection)")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
