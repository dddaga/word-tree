"""Step 218: Random projection ablation — how much do learned hidden params matter?

MOTIVATION
==========
SGNNET's routing is static and input-blind. AH modulates edge suppression via
W_pos similarity, but the topology is fixed. This raises a fundamental question:

If routing is essentially a fixed nonlinear feature extractor (random projection
→ gather-sum-normalize × K_iter), how much does LEARNING the hidden W_pos matter?

This experiment strips learnable parameters progressively:

  Ref : Full step199 (67K params) — AH + learned W_pos + learned theta
  A   : Frozen hidden W_pos, NO AH — only output W_pos + theta learned (~2.2K params)
  B   : Frozen hidden W_pos + frozen theta — ONLY output W_pos learned (160 params!)
  C   : Frozen hidden W_pos, WITH AH (frozen supp_w) — random diversity enforced
  D   : No AH, learned hidden W_pos — isolate AH contribution vs learned positions
  E   : Random readout directions too — ZERO learned params (pure random baseline)

If A or B still achieves ~90%+, the paper story becomes:
  "SGNNET is a structured random feature extractor where 99.8% of the representational
   work is done by the fixed random graph topology + normalize; only 160 params
   (output class directions) need learning."

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0)
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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step218_random_projection_ablation.json"


class SGNNET_RandomProjection(nn.Module):
    """SGNNET with progressively frozen parameters.

    Modes:
      "full"          — standard step199 (all params learned, with AH)
      "frozen_no_ah"  — hidden W_pos frozen random, no AH, theta learned
      "frozen_all"    — hidden W_pos AND theta frozen, only output W_pos learned (160 params)
      "frozen_with_ah"— hidden W_pos frozen, AH computed from frozen positions
      "no_ah_learned" — no AH, but hidden W_pos IS learned (isolate AH vs learning)
      "pure_random"   — everything frozen including output positions (0 learned params)
    """

    def __init__(self, mode="full"):
        super().__init__()
        self.mode = mode

        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=ng, norm_mode="l2", encoding_mode="fourier")

        # Resonant layer (for theta)
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        self.use_ah = mode in ("full", "frozen_with_ah")
        self.alpha_ahebb = ALPHA_AHEBB if self.use_ah else 0.0

        # Freeze parameters based on mode
        if mode in ("frozen_no_ah", "frozen_all", "frozen_with_ah"):
            # Freeze hidden W_pos (keep output W_pos learnable except in pure_random)
            # W_pos is [N+N_out, D] — we need to handle this carefully
            # We'll split it: frozen hidden part as buffer, learnable output part
            with torch.no_grad():
                self._frozen_wpos_hidden = self.base.W_pos[:N].clone()
            # Keep W_pos as parameter but we'll zero gradients for hidden part
            self._freeze_hidden_wpos = True
        elif mode == "pure_random":
            self._freeze_hidden_wpos = True
            self._freeze_output_wpos = True
        else:
            self._freeze_hidden_wpos = False

        if mode == "frozen_all":
            # Also freeze theta
            self.resonant.theta.requires_grad_(False)

        if mode == "pure_random":
            # Freeze everything
            for p in self.parameters():
                p.requires_grad_(False)

        # Freeze W_phase always (alpha_turing=0.0)
        self.resonant.W_phase.requires_grad_(False)

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    def tick_epoch(self):
        self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Compute AH suppression if needed
        if self.use_ah:
            W_n = F.normalize(self.base.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)
        else:
            supp_w = None

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]

            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


class ZeroGradHook:
    """Hook to zero out gradients for hidden W_pos after backward."""
    def __init__(self, model, N_hidden):
        self.model = model
        self.N_hidden = N_hidden

    def __call__(self):
        if self.model.base.W_pos.grad is not None:
            self.model.base.W_pos.grad[:self.N_hidden] = 0.0


def build_model(config_key):
    modes = {
        "Ref": "full",
        "A":   "frozen_no_ah",
        "B":   "frozen_all",
        "C":   "frozen_with_ah",
        "D":   "no_ah_learned",
        "E":   "pure_random",
    }
    return SGNNET_RandomProjection(mode=modes[config_key])


def main():
    all_keys = ["Ref", "A", "B", "C", "D", "E"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: full step199 (67K params, AH + learned W_pos + theta)",
        "A":   "A: frozen hidden W_pos, no AH, theta learned (~2.2K params)",
        "B":   "B: frozen W_pos + theta, only output W_pos learned (160 params!)",
        "C":   "C: frozen hidden W_pos, WITH AH (frozen suppression)",
        "D":   "D: no AH, hidden W_pos learned (isolate AH contribution)",
        "E":   "E: pure random — zero learned params (lower bound)",
    }

    print(f"\n{'='*70}")
    print(f"Step 218 — Random Projection Ablation (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"Question: how much does learning hidden params matter?")
    print(f"If 160-param config works: SGNNET is a structured random feature extractor")
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

        # Count learnable params
        n_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())

        # For frozen_* modes, hidden W_pos gradients are zeroed via hook
        grad_hook = None
        if hasattr(model, '_freeze_hidden_wpos') and model._freeze_hidden_wpos:
            grad_hook = ZeroGradHook(model, N)

        print(f"  learnable={n_learnable:,}  total={n_total:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        # Register hook to zero hidden W_pos gradients
        if grad_hook is not None:
            for g in trainer.optimizer.param_groups:
                for p in g["params"]:
                    if p is model.base.W_pos:
                        p.register_hook(lambda grad, gh=grad_hook: _zero_hidden(grad, N))

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1),
            "n_learnable": n_learnable, "n_total": n_total,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        }
        print(f"  → best={best:.4f} @ ep{bep}  learnable={n_learnable:,}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 218 SUMMARY — Random Projection Ablation")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key}: {r['top1_best']:.4f}  params={r['n_learnable']:,}{delta}")

    print(f"\nIf B ≥ 90%: SGNNET is fundamentally a 160-param random feature extractor.")
    print(f"If B < 80% but A ≥ 90%: theta matters, W_pos hidden doesn't.")
    print(f"If A < 80%: learning hidden representations is essential.")
    print(f"\n→ {OUT_PATH}")


def _zero_hidden(grad, N_hidden):
    """Zero out gradient for hidden neuron positions, keep output gradient."""
    grad_new = grad.clone()
    grad_new[:N_hidden] = 0.0
    return grad_new


if __name__ == "__main__":
    main()
