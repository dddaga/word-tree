"""Step 703: Per-sample adaptive K_iter at inference.

MOTIVATION
==========
K_iter=5 sequential passes are the dominant latency cost. step700/701 confirmed
sequential passes cannot be parallelized (multi-hop and parallel branches both KILLED).

NEW DIRECTION: Keep training at K=5. At inference, measure logit stability after
each routing step. If logits have converged (||Δlogits|| / ||logits|| < τ), exit
early. No training change — pure inference optimization.

Hypothesis: some samples converge in 1-2 steps; hard samples need all 5.
Expected: mean effective K_iter ~3 at ≤0.5pp accuracy loss.

MECHANISM
=========
For each sample, after each routing step k (1..K_iter):
  stability = cosine_sim(logits[k], logits[k-1])
  if stability > τ: exit early, use logits[k] for prediction

CONFIGS
=======
  Ref    : K_iter=5 fixed (no early exit, τ=1.0 sentinel)
  tau_99 : exit if cosine_sim > 0.99
  tau_97 : exit if cosine_sim > 0.97
  tau_95 : exit if cosine_sim > 0.95
  tau_90 : exit if cosine_sim > 0.90

Train once (20ep Tier-0), then profile all thresholds on validation set.
Also plot K_iter distribution per threshold.

Scale: N=2048, D=16, K_hh=2, K_iter=5, AH=1.0 (efficiency config, step199)
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
from src.training.dataset             import H5Dataset

parser = argparse.ArgumentParser(description="Step 703: Adaptive K_iter inference")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
args   = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128; SEED = 42
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
FRAC_DATA = 0.5

OUT_PATH = ROOT / "results" / "train_step703_adaptive_kiter.json"

THRESHOLDS = {
    "Ref":    1.1,   # always run all K_iter (sentinel > 1.0)
    "tau_99": 0.99,
    "tau_97": 0.97,
    "tau_95": 0.95,
    "tau_90": 0.90,
}


# ---------------------------------------------------------------------------
# Model with per-step logit extraction
# ---------------------------------------------------------------------------

class SGNNETAdaptive(nn.Module):
    """SGNNET with per-step logit access for adaptive early exit."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        n_groups = max(8, N // 8)
        K_local  = max(1, K_HH - max(1, K_HH // 4))
        K_random = K_HH - K_local
        sw = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_local=K_local, K_random=K_random,
            n_groups=n_groups, K_iter=K_ITER,
            norm_mode="l2", encoding_mode="fourier",
        )
        res = SGNNET_Resonant(
            base=sw, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
        )
        self.inner = SGNNET_AntiHebbian(base=res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    def forward(self, x):
        """Standard forward for training."""
        return self.inner(x)

    @torch.no_grad()
    def forward_adaptive(self, x, tau: float):
        """Inference with early exit.

        Returns (logits, k_used_per_sample).
        k_used: [B] tensor with number of routing steps taken per sample.
        """
        model = self.inner
        # Access internal components
        base    = model.m.base           # SGNNET_SmallWorld
        res     = model.m               # SGNNET_Resonant (has theta, W_phase)
        conn_hh = base.conn_hh          # [N, K_hh]
        N_h     = base.N_hidden

        # AH suppression weights (precomputed)
        W_n = F.normalize(base.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)  # [1, N, K_hh, 1]

        B = x.shape[0]
        device = x.device
        supp_w = supp_w.to(device)

        # Seed hidden state
        Z = base._seed(x)  # [B, N, D]

        theta_pos = res.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1, 1]

        prev_logits = None
        Z_reflected = torch.zeros_like(Z)

        # Track per-sample early exit
        k_used        = torch.full((B,), K_ITER, dtype=torch.long, device=device)
        active_mask   = torch.ones(B, dtype=torch.bool, device=device)
        final_logits  = torch.zeros(B, N_OUT, device=device)

        for k in range(1, K_ITER + 1):
            # One routing step
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]
            Z_nb  = Z_nb * supp_w
            Z_struct   = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

            # Readout
            logits = base._readout(Z)  # [B, N_out]

            if prev_logits is None:
                prev_logits = logits
                continue  # need at least 2 steps to compute stability

            # Cosine similarity between consecutive logits per sample
            cos_sim = F.cosine_similarity(logits, prev_logits, dim=-1)  # [B]

            if tau <= 1.0:
                # Samples that have converged and are still active
                just_converged = active_mask & (cos_sim >= tau)
                if just_converged.any():
                    final_logits[just_converged] = logits[just_converged]
                    k_used[just_converged]        = k
                    active_mask[just_converged]   = False

            prev_logits = logits

            if not active_mask.any():
                break

        # Any still-active samples get the last logits
        if active_mask.any():
            final_logits[active_mask] = logits[active_mask]

        return final_logits, k_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print("Step 703 — Per-sample adaptive K_iter at inference")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | train {EPOCHS}ep 50% data")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}")

    # Data
    train_ds = H5Dataset(str(ROOT / "data/store.h5"), split="train")
    val_ds   = H5Dataset(str(ROOT / "data/store.h5"), split="val")
    n_train  = int(len(train_ds) * FRAC_DATA)
    g   = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    tr  = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, idx),
        batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    va = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # Train — Trainer expects model.W_pos; train model.inner, profile via wrapper
    print(f"\nTraining base model ({EPOCHS}ep 50% data)...")
    model = SGNNETAdaptive().to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model.inner, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
        print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
        if (m['epoch'] + 1) % 5 == 0 else None
    ))
    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    train_best = max(top1h)
    print(f"  Train best: {train_best:.4f} @ ep{int(np.argmax(top1h))+1}")

    # Profile adaptive K_iter on val set
    print(f"\nProfiling adaptive K_iter on val set ({len(val_ds)} samples)...")
    model.eval()

    results = {}
    for tag, tau in THRESHOLDS.items():
        correct = 0; total = 0
        k_dist  = torch.zeros(K_ITER + 1, dtype=torch.long)

        for feats, _, labels in va:
            feats  = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            logits, k_used = model.forward_adaptive(feats, tau=tau)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
            for k_val in k_used.cpu().tolist():
                k_dist[k_val] += 1

        acc      = correct / total
        mean_k   = (sum(k * k_dist[k].item() for k in range(K_ITER + 1))
                    / max(total, 1))
        speedup  = round(K_ITER / max(mean_k, 1e-6), 3)

        results[tag] = {
            "tau":       tau,
            "val_top1":  round(acc, 4),
            "mean_k":    round(mean_k, 3),
            "speedup":   speedup,
            "k_distribution": {str(k): int(k_dist[k].item())
                               for k in range(K_ITER + 1)},
        }
        print(f"  {tag:<10} τ={tau:.2f}  acc={acc:.4f}  mean_k={mean_k:.2f}  "
              f"speedup={speedup:.2f}×  k_dist={k_dist[1:].tolist()}", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print("STEP 703 SUMMARY — Adaptive K_iter at inference")
    print(f"{'='*70}")
    ref_acc = results["Ref"]["val_top1"]
    print(f"{'Config':<12} {'tau':>6} {'acc':>7} {'Δacc':>7} {'mean_k':>7} {'speedup':>8}")
    print(f"{'-'*55}")
    for tag, r in results.items():
        delta = f"{r['val_top1'] - ref_acc:+.4f}" if tag != "Ref" else "    —"
        print(f"  {tag:<12} {r['tau']:>5.2f} {r['val_top1']:>7.4f} {delta:>7}  "
              f"{r['mean_k']:>6.2f}  {r['speedup']:>7.2f}×")

    verdict = "no early exit gain"
    for tag, r in results.items():
        if tag == "Ref": continue
        if r["val_top1"] >= ref_acc - 0.005 and r["speedup"] >= 1.3:
            verdict = f"EARLY EXIT VIABLE: {tag} {r['speedup']:.2f}× at {r['val_top1']:.4f}"
            break

    print(f"\n  Verdict: {verdict}")

    out = {
        "config": {"N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
                   "epochs": EPOCHS, "device": str(DEVICE)},
        "train_best": train_best,
        "thresholds": results,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
