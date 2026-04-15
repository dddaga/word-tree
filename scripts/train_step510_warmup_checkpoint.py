"""Step 510: Warmup checkpoint generator for dynamic-connectivity experiments.

MOTIVATION
==========
User directive (2026-04-14): try dynamic connectivity ONLY after activations stabilize (>20ep).
Prior dynamic-connectivity failures (step153 pruning, step740-742 ConnGA, step230 Gumbel, step224 scored-dynamic)
all applied changes DURING initial co-adaptation, creating a stability crisis.

This script generates the warmup checkpoint that future step511-514 dynamic mechanisms will load.

PROTOCOL
========
Phase 1 (this script):
  - Train N=512 SGNNET with AH-only for 30 epochs (well past stabilization)
  - Save checkpoint: model weights, conn_hh static topology, activation statistics:
      μ_i = mean |Z_i|  over val set
      σ_i = std  |Z_i|
      corr_ij = correlation matrix over sparse k=4 candidate set per neuron
  - Save to results/warmup_ckpt_n512_ep30.pt

Phase 2 (step511-514, separate scripts to be written after this finishes):
  - Load checkpoint
  - Engage dynamic connectivity mechanism (A co_act_low / B co_act_high / C grad_cond / D var_gated)
  - Continue training ~60-100 epochs with edge rewiring every 5 epochs
  - Measure Δ vs static baseline at same total-epoch count

N=512 scale (user-specified) is chosen for fast iteration across 4+ mechanisms.
"""
from __future__ import annotations
import argparse, json, sys, time
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
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--N", type=int, default=512)
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = args.N; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

CKPT_PATH = ROOT / "results" / f"warmup_ckpt_n{N}_ep{EPOCHS}.pt"
STATS_PATH = ROOT / "results" / f"warmup_stats_n{N}_ep{EPOCHS}.json"


def build_model():
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


@torch.no_grad()
def collect_activation_stats(model, loader, device):
    """Collect stats usable for dynamic topology decisions.

    Captures PRE-normalization |Z_fwd|_i (the activation that flows into aggregation)
    and per-neuron activation frequency (fraction of inputs for which |Z_fwd|_i > 0
    after ReLU(Z - θ)). Both are real signals that a topology-rewiring rule can use.
    """
    model.eval()
    z_sum = None; z_sq_sum = None; count = 0
    active_count = None
    for batch in loader:
        x = batch[0].to(device)
        # Replay the forward with instrumentation on the last K_iter step (pre-normalize)
        Z = model.m.base._seed(x)
        conn_hh = model.m.base.conn_hh
        theta_pos = model.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = model.m.W_pos[:N]
        W_n = F.normalize(W_h, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)
        Z_reflected = torch.zeros_like(Z)
        Z_fwd_final = None
        for k in range(K_ITER):
            Z_fwd = F.relu(Z - theta_pos)
            if k == K_ITER - 1:
                Z_fwd_final = Z_fwd   # capture final pre-aggregation signal
            Z_nb  = Z_fwd[:, conn_hh, :] * supp_w
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        # Use PRE-normalize Z_fwd magnitude → real variable per input
        Zn = Z_fwd_final.norm(dim=-1)  # [B, N]  ||Z_fwd_i|| ≥ 0
        active_mask = (Zn > 1e-6).float()  # [B, N]
        if z_sum is None:
            z_sum = Zn.sum(dim=0); z_sq_sum = (Zn**2).sum(dim=0)
            active_count = active_mask.sum(dim=0)
        else:
            z_sum += Zn.sum(dim=0); z_sq_sum += (Zn**2).sum(dim=0)
            active_count += active_mask.sum(dim=0)
        count += Zn.shape[0]
    mu    = (z_sum / count).cpu().numpy().tolist()
    var   = (z_sq_sum / count - (z_sum / count)**2).clamp(min=0).cpu().numpy()
    sigma = np.sqrt(var).tolist()
    act_freq = (active_count / count).cpu().numpy().tolist()
    return {
        "mu_per_neuron":    mu,
        "sigma_per_neuron": sigma,
        "active_freq":      act_freq,
        "n_val":            count,
    }


def main():
    print(f"Step 510 — Warmup Checkpoint N={N} epochs={EPOCHS}")
    print(f"  Output: {CKPT_PATH}")

    tr, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params:,}  N={N}  K_hh={K_HH}  K_iter={K_ITER}")

    t0 = time.time()
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch'] + 1) % 5 == 0 else None
    ))
    top1h = [h.get("val_top1", 0.0) for h in history]
    top1_best = max(top1h) if top1h else 0.0
    print(f"\nTraining done. Best val top1 = {top1_best:.4f}  elapsed={time.time()-t0:.0f}s")

    # Save checkpoint
    print(f"\nSaving checkpoint → {CKPT_PATH}")
    torch.save({
        "model_state": model.state_dict(),
        "conn_hh":     model.m.base.conn_hh.cpu(),
        "N": N, "D": D, "K_HH": K_HH, "K_ITER": K_ITER, "K_IN": K_IN,
        "epochs_trained": EPOCHS,
        "top1_best": top1_best,
        "top1_history": top1h,
    }, CKPT_PATH)

    # Collect activation statistics
    print(f"\nCollecting activation statistics...")
    stats = collect_activation_stats(model, va, DEVICE)
    stats["N"] = N; stats["top1_best"] = top1_best; stats["epochs"] = EPOCHS
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved activation stats → {STATS_PATH}")
    print(f"  mu_per_neuron range: [{min(stats['mu_per_neuron']):.4f}, {max(stats['mu_per_neuron']):.4f}]")
    print(f"  sigma_per_neuron range: [{min(stats['sigma_per_neuron']):.4f}, {max(stats['sigma_per_neuron']):.4f}]")
    print(f"\nReady for step511-514 (dynamic connectivity mechanisms).")


if __name__ == "__main__":
    main()
