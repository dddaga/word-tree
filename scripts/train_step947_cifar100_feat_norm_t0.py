"""Step 947: CIFAR-100 feature normalization T0 (20ep, 50% data).

Failure mode targeted: CIFAR-100 VGG16 features have 2.3× lower magnitude
than Imagenette features (std=0.574 vs 1.318). SGNNET's _seed initialises Z
by gathering input features and combining with spatial encodings. If feature
scale is too low relative to the spatial encoding scale, the spatial component
dominates and Z is under-informed by the actual image content, degrading the
quality of the initial hidden state.

Hypothesis: Scaling CIFAR-100 features to match the Imagenette magnitude
range (×2.296) improves seed Z quality and closes part of the ~18pp gap.

IMPORTANT NOTE on A_scale vs B_zscore:
  With feat_mean=0.0 (VGG features are mean-centered), the z-score
  normalization formula reduces to:
      x_norm = (x - 0) / 0.574 * 1.318 = x * (1.318/0.574) = x * 2.296
  This is mathematically identical to A_scale. The two configs are therefore
  intentional sanity duplicates — if they diverge, it indicates numerical
  noise or data-loading non-determinism. We run both to confirm.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, K_in=25)
  Ref      no normalization (standard)
  A_scale  scale x by FEAT_SCALE = 1.318 / 0.574 ≈ 2.296
  B_zscore zscore: (x - 0.0) / 0.574 * 1.318  (= A_scale; sanity duplicate)

ADVANCE: ≥+0.5pp vs Ref → T1.
Ref baseline: step930 K4I5 T0 best = 0.4201.
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store_cifar100.h5")
parser.add_argument("--configs", default="Ref,A_scale,B_zscore")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 100
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step947_cifar100_feat_norm_t0_seed{SEED}__{SLOT}.json"
STEP_REF = 0.4201   # step930 K4I5 T0 best

# Feature normalization constants (pre-computed from VGG16 CIFAR-100 pool5 activations)
FEAT_MEAN = 0.0      # VGG features are mean-centered (ReLU outputs, mean≈0 after centering)
FEAT_STD  = 0.574    # CIFAR-100 VGG pool5 activation std
IMAGENETTE_STD = 1.318  # Imagenette VGG pool5 activation std
FEAT_SCALE = IMAGENETTE_STD / FEAT_STD   # ≈ 2.296


# ---------------------------------------------------------------------------
# Config table: maps key → (norm_mode_str, scale_factor, subtract_mean)

CONFIG_SPEC = {
    "Ref":      ("none",   1.0,        False),
    "A_scale":  ("scale",  FEAT_SCALE, False),
    "B_zscore": ("zscore", FEAT_SCALE, False),  # identical to A_scale (see docstring)
}


# ---------------------------------------------------------------------------
# ΔW-proj helpers

def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


class SGNNET_FeatNorm(nn.Module):
    """SGNNET_Resonant wrapper with configurable input feature normalization.

    The scale factor is applied before _seed() so the initial Z is computed
    from re-scaled features. All downstream operations (K_iter loop, readout)
    are identical to the standard model.
    """

    def __init__(self, resonant, scale: float = 1.0):
        super().__init__()
        self.m = resonant
        self.scale = scale

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        # Scale features to Imagenette-like magnitude before seeding
        x_scaled = x * self.scale if self.scale != 1.0 else x
        Z = self.m.base._seed(x_scaled)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            Z_agg = _dw_agg(Z_nb, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    _, scale, _ = CONFIG_SPEC[key]
    r = make_base()
    return SGNNET_FeatNorm(r, scale=scale)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step947 — CIFAR-100 feature normalization T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}/K_in={K_IN}  N_OUT={N_OUT}")
    print(f"  CIFAR-100 feat std={FEAT_STD}  Imagenette feat std={IMAGENETTE_STD}")
    print(f"  FEAT_SCALE = {FEAT_SCALE:.4f}  (A_scale == B_zscore: sanity check)")
    print(f"  STEP_REF (step930 K4I5 T0): {STEP_REF:.4f}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        norm_mode, scale, _ = CONFIG_SPEC[key]
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: norm={norm_mode}  scale={scale:.4f}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        def log_fn(m):
            print(f"  e{m['epoch']+1:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"norm_mode": norm_mode, "scale": round(scale, 4), "n_params": n_p,
                        "best": round(best, 4), "best_ep": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 947 SUMMARY — CIFAR-100 feature normalization T0")
    print(f"{'='*70}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = "(baseline)" if k == "Ref" else ("ADVANCE→T1" if d >= 0.005 else ("NEUTRAL" if d >= -0.005 else "KILL"))
        print(f"  {k:<12} norm={r['norm_mode']:<8}  scale={r['scale']:.3f}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
