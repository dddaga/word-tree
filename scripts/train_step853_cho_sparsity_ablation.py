"""Step 853: C_ho readout sparsity ablation — first-ever sweep of sparsity param.

MOTIVATION
==========
SGNNET_SmallWorld's readout uses C_ho_mask [N_hidden, N_out], a binary
hidden→class mask built via _make_binary_c() with sparsity=0.90 as the
hard-coded default. That value has NEVER been ablated — it was inherited
from SGNNET_Wave and propagated through model_smallworld.py untouched.

The readout einsum Z @ C_ho currently costs O(B·N_hidden·N_out·D) regardless
of mask density (mask just zeros terms), so this sweep measures accuracy vs
sparsity only. A true-sparse implementation is a follow-up.

CONFIGS (T0: 20ep, 50% data, N=2048, D=16, K_hh=2, K_iter=5, K_in=25)
  Ref       : sparsity=0.90          — current default (~205/class)
  A_dense   : sparsity=0.00          — fully dense (2048/class)
  B_mid     : sparsity=0.50          — ~1024/class
  C_sparse  : sparsity=0.95          — ~102/class
  D_very    : sparsity=0.98          — ~41/class
  E_extreme : sparsity=0.99          — ~20/class
  F_tiny    : random K_ho=10/class   — minimum, random
  G_geometric: W_pos-nearest K_ho=10 — data-driven, geometric prior

Expected insights: identify knee of accuracy-vs-density curve; contrast
random F_tiny vs geometric G_geometric — does W_pos proximity beat random
at the same connection budget? Informs paper section on readout efficiency
and motivates true-sparse kernel follow-up.
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_dense,B_mid,C_sparse,D_very,E_extreme,F_tiny,G_geometric")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5                # step199 reference scale
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

STEP_NAME = Path(__file__).stem
SLOT      = os.environ.get("SGN_SLOT", "local")
OUT_PATH  = ROOT / "results" / f"train_step853_cho_sparsity_ablation_seed{SEED}__{SLOT}.json"

# (sparsity_arg, custom_builder_name_or_None, label)
CONFIGS = {
    "Ref":         (0.90,  None,        "sparsity=0.90 current default (~205/class)"),
    "A_dense":     (0.00,  None,        "sparsity=0.00 fully dense (2048/class)"),
    "B_mid":       (0.50,  None,        "sparsity=0.50 half dense (~1024/class)"),
    "C_sparse":    (0.95,  None,        "sparsity=0.95 2x sparser (~102/class)"),
    "D_very":      (0.98,  None,        "sparsity=0.98 5x sparser (~41/class)"),
    "E_extreme":   (0.99,  None,        "sparsity=0.99 10x sparser (~20/class)"),
    "F_tiny":      (0.90,  "random_k",  "random K_ho=10/class (min budget, random)"),
    "G_geometric": (0.90,  "geometric", "W_pos-nearest K_ho=10/class (data-driven)"),
}

K_HO_FIXED = 10   # for F_tiny and G_geometric


# ---------------------------------------------------------------
# Custom C_ho mask builders (applied after model init)
# ---------------------------------------------------------------

def _mask_random_k(N_hidden: int, N_out: int, k: int, seed: int) -> torch.Tensor:
    """Each class c picks k distinct random hidden nodes."""
    g = torch.Generator().manual_seed(seed + 1_000_000)
    mask = torch.zeros(N_hidden, N_out, dtype=torch.bool)
    for c in range(N_out):
        idx = torch.randperm(N_hidden, generator=g)[:k]
        mask[idx, c] = True
    return mask


def _mask_geometric_k(W_pos: torch.Tensor, N_hidden: int, N_out: int, k: int) -> torch.Tensor:
    """For each class c, pick k hidden nodes whose W_pos is closest to W_pos[N_hidden+c]."""
    W_hidden = W_pos[:N_hidden]                 # [N_hidden, D]
    W_out    = W_pos[N_hidden:N_hidden + N_out] # [N_out,    D]
    # cdist: [N_out, N_hidden]
    d = torch.cdist(W_out, W_hidden)            # Euclidean in D-dim W_pos space
    # topk smallest k per class
    _, topk = torch.topk(d, k, dim=1, largest=False)  # [N_out, k]
    mask = torch.zeros(N_hidden, N_out, dtype=torch.bool)
    for c in range(N_out):
        mask[topk[c], c] = True
    return mask


# ---------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------

def make_model(sparsity: float, custom: str | None) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=sparsity,
    )

    # Custom C_ho overwrites (applied AFTER init, BEFORE wrapping in Resonant/AH)
    if custom == "random_k":
        new_mask = _mask_random_k(N, N_OUT, K_HO_FIXED, SEED)
        base.C_ho_mask = new_mask.to(base.C_ho_mask.device)
    elif custom == "geometric":
        with torch.no_grad():
            new_mask = _mask_geometric_k(base.W_pos.detach().cpu(), N, N_OUT, K_HO_FIXED)
        base.C_ho_mask = new_mask.to(base.C_ho_mask.device)

    if DEVICE.type == "cuda":
        # CUDA path: use_amp=False, non_blocking=True in Trainer .to(device) calls
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=True)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=True)
    else:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ---------------------------------------------------------------
# Mask statistics
# ---------------------------------------------------------------

def mask_stats(base_model: SGNNET_SmallWorld) -> dict:
    m = base_model.C_ho_mask.bool().cpu()                 # [N_hidden, N_out]
    k_per_class   = m.sum(dim=0).float()                  # [N_out]
    coverage_mask = m.any(dim=1)                          # [N_hidden]
    # Cross-class overlap: mean Jaccard across all class pairs
    n_out = m.shape[1]
    jacc = []
    for a in range(n_out):
        for b in range(a + 1, n_out):
            inter = (m[:, a] & m[:, b]).sum().item()
            union = (m[:, a] | m[:, b]).sum().item()
            jacc.append(inter / union if union > 0 else 0.0)
    return {
        "k_ho_mean":        float(k_per_class.mean().item()),
        "k_ho_std":         float(k_per_class.std().item()),
        "k_ho_min":         int(k_per_class.min().item()),
        "k_ho_max":         int(k_per_class.max().item()),
        "coverage_pct":     float(coverage_mask.float().mean().item() * 100.0),
        "pair_jaccard_mean":float(np.mean(jacc)) if jacc else 0.0,
    }


def _base_of(model: nn.Module) -> SGNNET_SmallWorld:
    """Unwrap AntiHebbian(Resonant(SmallWorld)) — follows .m then .base."""
    m = model
    while not isinstance(m, SGNNET_SmallWorld):
        if hasattr(m, "m"):      m = m.m       # AntiHebbian → Resonant
        elif hasattr(m, "base"): m = m.base    # Resonant → SmallWorld
        else: break
    return m  # type: ignore[return-value]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    # T0: 50% data. Load full, then subset train indices.
    tr_full, va = make_loaders(
        str(data_path), batch_size=BATCH, seed=SEED,
        pin_memory=True,                                  # pin_memory=True literal
    )
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(
        n_full, generator=torch.Generator().manual_seed(SEED)
    )[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=True,                                  # pin_memory=True literal
    )

    print(f"\n{'='*70}")
    print(f"step853 — C_ho sparsity ablation (T0: 20ep, 50% data, N={N})")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  K_in={K_IN}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  (step199 reference)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown config: {key}"); continue
        sparsity, custom, desc = CONFIGS[key]
        model = make_model(sparsity, custom)

        base = _base_of(model)
        mstats = mask_stats(base)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{'-'*60}\n{key}: {desc}")
        print(f"  params={n_p:,}  k_ho={mstats['k_ho_mean']:.1f}±{mstats['k_ho_std']:.1f} "
              f"(min={mstats['k_ho_min']} max={mstats['k_ho_max']}) "
              f"coverage={mstats['coverage_pct']:.1f}% "
              f"jaccard={mstats['pair_jaccard_mean']:.3f}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False                         # use_amp=False literal
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.0) if isinstance(h, dict) else float(h), 4)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta_ref = best - (ref_acc if ref_acc is not None else best)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta_ref*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "sparsity_arg":    sparsity,
            "custom":          custom,
            "label":           desc,
            "n_params":        n_p,
            "k_ho_mean":       mstats["k_ho_mean"],
            "k_ho_std":        mstats["k_ho_std"],
            "k_ho_min":        mstats["k_ho_min"],
            "k_ho_max":        mstats["k_ho_max"],
            "coverage_pct":    mstats["coverage_pct"],
            "pair_jaccard_mean": mstats["pair_jaccard_mean"],
            "best":            best,
            "best_ep":         best_ep,
            "top1_history":    top1h,
            "delta_vs_ref":    round(delta_ref, 4),
            "elapsed_s":       round(elapsed, 1),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 853 SUMMARY — C_ho sparsity ablation")
    print(f"{'='*70}")
    print(f"  {'key':<12} {'k_ho':>8} {'cov%':>6} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<12} {r['k_ho_mean']:>8.1f} {r['coverage_pct']:>5.1f} "
              f"{r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
