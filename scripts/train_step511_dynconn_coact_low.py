"""Step 511: Dynamic connectivity mechanism A — co-activation LOW (diversification).

MOTIVATION
==========
User directive (2026-04-14): test dynamic connectivity AFTER activations stabilize.
Prior dynamic attempts failed because applied during initial co-adaptation.
Theory: extra DoF of topology should add expressive capacity, enabling higher compression.

This is the FIRST experiment in the dynamic-connectivity line (step511-514 family).
Uses step510 warmup checkpoint as starting point.

PROTOCOL
========
Phase 1 (from step510): Static N=512 AH-only for 30ep. Best=0.7745.
Phase 2 (this script): Continue training for REWIRE_EPOCHS epochs.
  Every REWIRE_EVERY epochs:
    Compute sparse activation correlations corr_ij over k_candidates={current K_hh} + {k_new random}
    For each neuron i, rank candidates by |corr_ij|:
      Mech A (co_act_low)  : KEEP the 2 lowest-|corr| candidates  (diversify)
  Update conn_hh in-place.
  Continue training (W_pos + θ + readout all trainable, just conn_hh rewires).

CONFIGS (N=512 D=16 K_hh=2 K_iter=5, 50% data)
  Ref_static : continue training with FROZEN conn_hh (static baseline)
  A_coact_low: rewire every 5 epochs toward low-correlation neighbors

Success criterion: A > Ref_static by +0.5pp at same total-epoch count (30+REWIRE_EPOCHS).
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
parser.add_argument("--ckpt", default="results/warmup_ckpt_n512_ep30.pt")
parser.add_argument("--rewire_epochs", type=int, default=60,
                    help="epochs to train after loading checkpoint")
parser.add_argument("--rewire_every", type=int, default=5,
                    help="rewire conn_hh every N epochs")
parser.add_argument("--k_candidates", type=int, default=6,
                    help="candidate pool size per neuron for rewiring")
parser.add_argument("--configs", default="", help="Ref_static,A_coact_low. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step511_dynconn_coact_low.json"


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


def load_warmup_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # conn_hh must match too (restored via state_dict buffer)
    print(f"  loaded checkpoint: ep {ckpt.get('epochs_trained', '?')}, "
          f"best top1 {ckpt.get('top1_best', 0):.4f}")
    return model


@torch.no_grad()
def compute_activation_correlation(model, loader, device, n_batches=5):
    """Compute sparse correlation matrix between neurons on a sample of data.

    Returns a per-neuron correlation vector by batching cross-product of Z_fwd.
    This IS memory-quadratic (N × N) but N=512 → 512² × 4 bytes ≈ 1 MB — trivial.
    """
    model.eval()
    feats = []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0].to(device)
        # Recompute forward to get Z_fwd (pre-normalize) at last iter
        Z = model.m.base._seed(x)
        conn_hh = model.m.base.conn_hh
        theta_pos = model.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = model.m.W_pos[:N]; W_n = F.normalize(W_h, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)
        Z_reflected = torch.zeros_like(Z)
        Z_fwd_last = None
        for k in range(K_ITER):
            Z_fwd = F.relu(Z - theta_pos)
            if k == K_ITER - 1: Z_fwd_last = Z_fwd
            Z_nb = Z_fwd[:, conn_hh, :] * supp_w
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        # Use magnitude of pre-norm Z_fwd as "activation" signal, per neuron
        Zn = Z_fwd_last.norm(dim=-1)  # [B, N]
        feats.append(Zn.cpu())
    A = torch.cat(feats, dim=0)  # [batch_total, N]
    # Pearson correlation across samples
    A_c = A - A.mean(dim=0, keepdim=True)
    std = A_c.std(dim=0, keepdim=True).clamp(min=1e-6)
    A_n = A_c / std
    n_samp = A_n.shape[0]
    corr = (A_n.T @ A_n) / (n_samp - 1)   # [N, N]
    return corr


def rewire_coact_low(model, corr, k_candidates=6):
    """For each neuron i, select K_hh neighbors minimising |corr_ij|.
    Candidate pool = current K_hh neighbors + (k_candidates - K_hh) random new ones.
    """
    conn_hh = model.m.base.conn_hh  # [N, K_hh]  (buffer, on model.device)
    dev = conn_hh.device
    N_h, K_h = conn_hh.shape
    new_conn = torch.empty_like(conn_hh)
    corr_abs = corr.abs().to(dev)

    for i in range(N_h):
        current = set(conn_hh[i].tolist())
        # Draw additional random candidates not in current, not self
        extras = []
        while len(extras) < (k_candidates - K_h):
            j = int(torch.randint(0, N_h, (1,)).item())
            if j == i or j in current or j in extras: continue
            extras.append(j)
        pool = list(current) + extras
        pool_scores = corr_abs[i, pool]                    # [k_candidates]
        # Keep K_hh with LOWEST |corr| (diversification)
        top_idx = torch.topk(pool_scores, K_h, largest=False).indices.tolist()
        new_conn[i] = torch.tensor([pool[k] for k in top_idx], device=dev, dtype=conn_hh.dtype)
    # In-place update
    model.m.base.conn_hh.data.copy_(new_conn)


def main():
    t_start = time.time()
    ckpt_path = ROOT / args.ckpt
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    print(f"Step 511 — Dynamic connectivity (A=co_act_low diversification)")
    print(f"  N={N} rewire_every={args.rewire_every} rewire_epochs={args.rewire_epochs}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    all_keys = ["Ref_static", "A_coact_low"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")
        model = build_model().to(DEVICE)
        model = load_warmup_checkpoint(model, ckpt_path)
        initial_conn = model.m.base.conn_hh.clone()

        kw = trainer_kwargs(N, n_epochs=args.rewire_epochs)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)

        history = []
        rewire_count = 0
        t0 = time.time()

        # Custom training loop: track epochs, call rewire mid-train
        for ep in range(args.rewire_epochs):
            if (key == "A_coact_low"
                and ep > 0
                and ep % args.rewire_every == 0):
                corr = compute_activation_correlation(model, va, DEVICE, n_batches=5)
                rewire_coact_low(model, corr, k_candidates=args.k_candidates)
                rewire_count += 1
                # measure fraction of edges changed vs initial
                diff = (model.m.base.conn_hh != initial_conn).float().mean().item()
                print(f"  [rewire #{rewire_count} at ep{ep}] diff from initial: {diff*100:.1f}%")

            ep_stats = trainer.train_epoch()
            # Validation
            val_top1 = trainer.validate()["val_top1"] if hasattr(trainer, "validate") else \
                       trainer._eval()["val_top1"] if hasattr(trainer, "_eval") else 0.0
            # Fallback: compute val_top1 directly
            if not val_top1:
                model.eval()
                correct = 0; total = 0
                with torch.no_grad():
                    for batch in va:
                        x = batch[0].to(DEVICE)
                        y = batch[2].to(DEVICE) if len(batch) > 2 else batch[1].to(DEVICE)
                        out = model(x)
                        correct += (out.argmax(dim=-1) == y).sum().item()
                        total += y.size(0)
                val_top1 = correct / max(total, 1)
                model.train()

            history.append({"epoch": ep, "val_top1": val_top1})
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1:3d}  val={val_top1:.4f}", flush=True)

        top1h = [h["val_top1"] for h in history]
        top1_best = max(top1h) if top1h else 0.0
        elapsed = time.time() - t0
        results[key] = {
            "top1_best": top1_best,
            "top1_last": top1h[-1] if top1h else 0.0,
            "rewire_count": rewire_count,
            "elapsed_s": elapsed,
            "history": top1h,
        }
        print(f"  → best={top1_best:.4f}  rewires={rewire_count}  elapsed={elapsed:.0f}s")
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n\n========== STEP 511 SUMMARY ==========")
    for k, r in results.items():
        print(f"  {k:<14}  best={r['top1_best']:.4f}  rewires={r['rewire_count']}")
    if "Ref_static" in results and "A_coact_low" in results:
        delta = (results["A_coact_low"]["top1_best"] - results["Ref_static"]["top1_best"]) * 100
        print(f"\nΔ(A_coact_low − Ref_static): {delta:+.2f}pp")
    print(f"\nTotal: {(time.time() - t_start):.0f}s")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
