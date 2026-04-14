"""Step 226: ResNet-style skip connections across K_iter routing steps.

MOTIVATION
==========
SGNNET routing: Z is normalized after every gather-sum step (K_iter=5 times).
After 5 iterations of gather-sum-normalize, the seed signal from the input
may be "washed out" — early information lost through repeated normalization.

ResNet insight: identity shortcuts preserve gradient flow and early features.
Applied to SGNNET: add skip connections across routing iterations.

Variants:
  Ref : standard routing (no skip connections) — step199 baseline
  A   : dense skip — Z_new = normalize(route(Z) + Z_input) at every iteration
        where Z_input is the INITIAL seeded state (constant throughout routing)
  B   : every-2 skip — Z_0 added to Z_2 output, Z_2 added to Z_4 output
        (only add once every 2 iterations, preserving intermediate states)
  C   : gated skip — Z_new = normalize(route(Z) + α*Z_input), α learned scalar
        Allows the model to decide how much initial signal to preserve

Key hypothesis: K_iter=5 may over-route the signal. Skip connections could
reduce effective routing depth while preserving gradient paths.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0 scouts)
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
                    help="Comma-separated config keys (e.g. Ref,A). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step226_skip_connections.json"


class SGNNET_SkipRouting(nn.Module):
    """SGNNET routing with ResNet-style skip connections across K_iter steps.

    skip_mode:
      "none"    — standard routing, no skip (Ref)
      "dense"   — add Z_seed to every iteration output before normalize
      "every2"  — add Z_{t-2} to Z_t output every 2 iterations
      "gated"   — add α * Z_seed to every iteration, α is learned scalar
    """

    def __init__(self, base_sw, resonant, alpha_ahebb=1.0, skip_mode="none"):
        super().__init__()
        self.base_sw     = base_sw
        self.resonant    = resonant
        self.alpha_ahebb = alpha_ahebb
        self.skip_mode   = skip_mode

        if skip_mode == "gated":
            # Learned gate scalar — init at 0 (no skip initially)
            self.skip_gate = nn.Parameter(torch.tensor(0.0))

    @property
    def W_pos(self):   return self.base_sw.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base_sw._seed(x)
        Z_seed = Z.clone()              # preserve initial state for skip connections
        conn_hh = self.base_sw.conn_hh
        N_h = self.base_sw.N_hidden

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression weights (static)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        # For every-2 skip: track the state 2 steps ago
        Z_prev2 = Z_seed.clone()   # Z at step 0

        for step in range(self.base_sw.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]   # [B, N, K_hh, D]
            Z_nb  = Z_nb * supp_w

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            if self.resonant.alpha_turing != 0.0:
                W_ph_norm = F.normalize(self.resonant.W_phase, dim=-1)
                Z_inh = self.resonant._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.resonant.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            # Apply skip connections
            if self.skip_mode == "dense":
                # Add initial seed state at every step
                Z_new = Z_new + Z_seed

            elif self.skip_mode == "every2":
                # Add Z from 2 steps ago at even steps (step 2, 4)
                if step > 0 and step % 2 == 0:
                    Z_new = Z_new + Z_prev2
                # Update Z_prev2: save current Z before normalize (2 steps ago for next even)
                if step % 2 == 0:
                    Z_prev2 = Z.clone()

            elif self.skip_mode == "gated":
                # Add α * Z_seed (α learned)
                Z_new = Z_new + self.skip_gate * Z_seed

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base_sw._readout(Z)


def build_model(config_key):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    mode_map = {"Ref": "none", "A": "dense", "B": "every2", "C": "gated"}
    if config_key not in mode_map:
        raise ValueError(f"Unknown config: {config_key}")

    return SGNNET_SkipRouting(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                               skip_mode=mode_map[config_key])


def main():
    all_keys = ["Ref", "A", "B", "C"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: standard routing, no skip connections",
        "A":   "A: dense skip — Z_seed added at every K_iter step",
        "B":   "B: every-2 skip — Z_{t-2} added at steps 2,4",
        "C":   "C: gated skip — α*Z_seed, α learned (init=0)",
    }

    print(f"\n{'='*70}")
    print(f"Step 226 — Skip connections across K_iter (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Question: do ResNet-style skips preserve early features through routing?")
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
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        # For gated skip: report final gate value
        gate_val = None
        if key == "C" and hasattr(model, "skip_gate"):
            gate_val = round(model.skip_gate.item(), 4)

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "skip_mode": key,
        }
        if gate_val is not None:
            results[key]["skip_gate_final"] = gate_val
            print(f"  skip_gate (α) final: {gate_val}")

        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 226 SUMMARY — Skip connections across K_iter")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if r["top1_best"] - ref_best > 0.005:   verdict = " → ADVANCE"
            elif r["top1_best"] - ref_best > -0.01:  verdict = " → NEUTRAL"
            else:                                     verdict = " → KILL"
        extra = ""
        if key == "C" and "skip_gate_final" in r:
            extra = f"  α={r['skip_gate_final']}"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}{extra}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
