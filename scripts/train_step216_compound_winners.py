"""Step 216: Compound all small-N winners on step199 final config (Tier-0 scouts).

MOTIVATION
==========
5 zero-cost mechanisms showed large gains at N=1024 but were NEVER tested on
the step199 efficiency config (N=2048, D=16, K_hh=2, K_iter=5). We don't know
if they're capacity crutches (disappear at larger N) or real improvements.

Mechanisms tested (all zero extra FLOPs):
  Ref : step199 baseline (homogeneous AH α=1.0)
  A   : α_ahebb=1.05 (+0.79pp confirmed at N=4096, never tested on step199)
  B   : twopop_weight — relay(×2) + specialist(×0.5) neuron populations (+6.42pp at N=1024)
  C   : twopop_theta — aggregator(θ=0.05) + filter(θ=0.20) populations (+5.27pp at N=1024)
  D   : curriculum K_iter — ramp from 2→3→4→5 during training (+2.85pp at N=1024)

Tier-0 protocol: 20 epochs, 50% data. Kill clearly bad, advance positive/neutral.
KEY QUESTION: do N=1024 gains survive at N=2048 D=16 K_hh=2?

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
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
                    help="Comma-separated config keys (e.g. A,D). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER  # 983,040 ≈ 0.98M
OUT_PATH = ROOT / "results" / "train_step216_compound_winners.json"


# ---------------------------------------------------------------------------
# Curriculum K_iter wrapper
# ---------------------------------------------------------------------------
class CurriculumKiterWrapper(nn.Module):
    """Ramp K_iter from k_start to k_end over the training schedule.

    During training, K_iter increases linearly with epoch.
    At eval, always uses full K_iter.
    """
    def __init__(self, model, k_start=2, k_end=5, total_epochs=20):
        super().__init__()
        self.model = model
        self.k_start = k_start
        self.k_end = k_end
        self.total_epochs = total_epochs
        self._current_epoch = 0

    @property
    def W_pos(self):   return self.model.W_pos
    @property
    def W_phase(self): return self.model.W_phase

    def tick_epoch(self):
        self._current_epoch += 1
        if hasattr(self.model, "tick_epoch"):
            self.model.tick_epoch()

    def forward(self, x):
        if self.training:
            frac = min(self._current_epoch / max(1, self.total_epochs - 1), 1.0)
            k = self.k_start + int(frac * (self.k_end - self.k_start))
            # Temporarily override K_iter
            base = self.model
            while hasattr(base, 'm') or hasattr(base, 'model'):
                base = getattr(base, 'm', None) or getattr(base, 'model', None)
            if hasattr(base, 'base'):
                old_k = base.base.K_iter
                base.base.K_iter = k
                out = self.model(x)
                base.base.K_iter = old_k
            else:
                old_k = base.K_iter
                base.K_iter = k
                out = self.model(x)
                base.K_iter = old_k
            return out
        else:
            return self.model(x)


# ---------------------------------------------------------------------------
# Heterogeneous neuron populations (from step143)
# ---------------------------------------------------------------------------
class SGNNET_HeteroAH(nn.Module):
    """Anti-Hebbian with heterogeneous neuron populations.

    Adapted from step143 for step199 config (K_hh=2).
    """
    def __init__(self, resonant, alpha_ahebb=1.0,
                 het_mode="homogeneous",
                 relay_scale=2.0, spec_scale=0.5,
                 theta_agg=0.05, theta_filt=0.20):
        super().__init__()
        self.m = resonant
        self.alpha_ahebb = alpha_ahebb
        self.het_mode = het_mode

        N_h = resonant.base.N_hidden
        half = N_h // 2

        if het_mode == "twopop_weight":
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1
            self.register_buffer("pop_mask", pop_mask)
            self.edge_scale = nn.Parameter(torch.tensor([relay_scale, spec_scale]))

        elif het_mode == "twopop_theta":
            pop_mask = torch.zeros(N_h, dtype=torch.long)
            pop_mask[half:] = 1
            self.register_buffer("pop_mask", pop_mask)
            self.theta_pop = nn.Parameter(torch.tensor([theta_agg, theta_filt]))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        N_h = self.m.base.N_hidden

        # Get theta
        if self.het_mode == "twopop_theta":
            theta_pos = self.theta_pop.abs()[self.pop_mask].unsqueeze(0).unsqueeze(-1)
        else:
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Get edge scale
        edge_scale = None
        if self.het_mode == "twopop_weight":
            edge_scale = self.edge_scale.abs()[self.pop_mask]  # [N_h]
            edge_scale = edge_scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N_h, 1, 1]

        # AH suppression weights (same as SGNNET_AntiHebbian.forward)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # Apply AH suppression
            Z_nb = Z_nb * supp_w

            if edge_scale is not None:
                Z_nb = Z_nb * edge_scale

            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def build_model(config_key):
    """Build model for given config key."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    if config_key == "Ref":
        model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")

    elif config_key == "A":
        # α_ahebb=1.05 instead of 1.0
        model = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.05, variant="wpos")

    elif config_key == "B":
        # twopop_weight: relay(×2) + specialist(×0.5)
        model = SGNNET_HeteroAH(resonant, alpha_ahebb=ALPHA_AHEBB,
                                 het_mode="twopop_weight")

    elif config_key == "C":
        # twopop_theta: aggregator(θ=0.05) + filter(θ=0.20)
        model = SGNNET_HeteroAH(resonant, alpha_ahebb=ALPHA_AHEBB,
                                 het_mode="twopop_theta")

    elif config_key == "D":
        # Curriculum K_iter: ramp from 2→5 during training
        ah_model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
        model = CurriculumKiterWrapper(ah_model, k_start=2, k_end=K_ITER,
                                        total_epochs=EPOCHS)

    else:
        raise ValueError(f"Unknown config: {config_key}")

    return model


def main():
    all_keys = ["Ref", "A", "B", "C", "D"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: homogeneous AH α=1.0 (step199 baseline)",
        "A":   "A: α_ahebb=1.05 (+0.79pp @ N=4096)",
        "B":   "B: twopop_weight relay×2/spec×0.5 (+6.42pp @ N=1024)",
        "C":   "C: twopop_theta agg/filt θ=0.05/0.20 (+5.27pp @ N=1024)",
        "D":   "D: curriculum K_iter 2→5 (+2.85pp @ N=1024)",
    }

    print(f"\n{'='*70}")
    print(f"Step 216 — Compound winners on step199 config (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Question: do N=1024 gains survive at N=2048 D=16 K_hh=2?")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # 50% data for Tier-0
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

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        }

        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    # Summary
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 216 SUMMARY — Compound winners on step199 config")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if r["top1_best"] - ref_best > 0.005: verdict = " → ADVANCE to Tier-1"
            elif r["top1_best"] - ref_best > -0.01: verdict = " → NEUTRAL (advance)"
            else: verdict = " → KILL"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
