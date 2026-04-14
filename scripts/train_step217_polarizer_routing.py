"""Step 217: Polarizer routing — W_pos as directional filter on incoming activations.

MOTIVATION
==========
Current routing: Z_new[i] = Σ Z[neighbors] — blind sum, ignores geometry.
Every input passes through identically regardless of content or neuron position.

Proposal (Dhiraj): Each neuron acts as a POLARIZER. Its W_pos defines the
polarization axis. Incoming activations are filtered by alignment with the
receiving neuron's polarizer:

    Z_filtered[j→i] = project(Z[j], W_pos[i]) + (1-α) × Z[j]
    Z_new[i] = normalize(Σ_j Z_filtered[j→i])

Key insight: F.normalize after aggregation recovers magnitude, so this is NOT
gate-death (no signal attenuation over iterations). It's directional filtering
with magnitude recovery — each neuron selectively attends to the component of
incoming signal aligned with its polarization axis.

Three polarization strengths tested:
  Ref : Standard gather-sum (no polarization)
  A   : Full polarizer (project onto W_pos, α=1.0) — strongest filtering
  B   : Partial polarizer (α=0.5) — 50% projected + 50% original
  C   : Soft polarizer (α=0.3) — 30% projected + 70% original (gentle)
  D   : Rotation in (Z, W_pos) plane by learned angle — Dhiraj's rotation variant

This makes routing INPUT-DEPENDENT: a cat and dog produce different projections
onto W_pos, so different effective signals flow through the same edges.

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

FLOPS = 3 * N * K_HH * D * K_ITER  # 983,040 ≈ 0.98M (routing only; polarizer adds ~2× per step)
OUT_PATH = ROOT / "results" / "train_step217_polarizer_routing.json"


class SGNNET_Polarizer(nn.Module):
    """SGNNET with polarizer-based routing.

    Each neuron's W_pos acts as a polarization axis. Incoming signals are
    partially projected onto this axis before aggregation.

    polarizer_mode:
      "none"     — standard gather-sum (Ref)
      "full"     — full projection onto W_pos (α=1.0)
      "partial"  — blended: α × project + (1-α) × original
      "rotation" — rotate Z in the (Z, W_pos) plane by learned angle
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 polarizer_mode="none", polarizer_alpha=1.0):
        super().__init__()
        self.base = base
        self.resonant = resonant
        self.alpha_ahebb = alpha_ahebb
        self.polarizer_mode = polarizer_mode
        self.polarizer_alpha = polarizer_alpha

        if polarizer_mode == "rotation":
            # Learnable temperature for rotation angle
            self.rotation_temp = nn.Parameter(torch.tensor(0.1))

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _polarize(self, Z_nb, W_pos_receivers):
        """Apply polarizer filtering to neighbor activations.

        Args:
            Z_nb: [B, N, K_hh, D] — neighbor activations
            W_pos_receivers: [N, D] — receiving neuron W_pos (polarizer axis)

        Returns:
            Z_filtered: [B, N, K_hh, D] — filtered activations
        """
        if self.polarizer_mode == "none":
            return Z_nb

        # Polarizer axis (normalized): [N, D] → [1, N, 1, D] for broadcast
        w = F.normalize(W_pos_receivers, dim=-1).unsqueeze(0).unsqueeze(2)  # [1, N, 1, D]

        if self.polarizer_mode in ("full", "partial"):
            # Project Z_nb onto W_pos direction
            # dot product: [B, N, K_hh, 1]
            proj_coeff = (Z_nb * w).sum(dim=-1, keepdim=True)
            Z_projected = proj_coeff * w  # [B, N, K_hh, D]

            if self.polarizer_mode == "full":
                return Z_projected
            else:
                # Blend: α × projected + (1-α) × original
                α = self.polarizer_alpha
                return α * Z_projected + (1 - α) * Z_nb

        elif self.polarizer_mode == "rotation":
            # Rotate Z_nb in the plane defined by (Z_nb, W_pos)
            # Decompose Z_nb into parallel and perpendicular components
            proj_coeff = (Z_nb * w).sum(dim=-1, keepdim=True)  # [B, N, K_hh, 1]
            z_parallel = proj_coeff * w                         # [B, N, K_hh, D]
            z_perp = Z_nb - z_parallel                         # [B, N, K_hh, D]
            z_perp_norm = F.normalize(z_perp, dim=-1)          # unit perpendicular

            # Rotation angle: proportional to alignment (input-dependent!)
            theta = self.rotation_temp * proj_coeff             # [B, N, K_hh, 1]

            # Rotate: preserve magnitude
            z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Z_rotated = (torch.cos(theta) * Z_nb +
                         torch.sin(theta) * z_perp_norm * z_mag)

            return Z_rotated

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden

        # W_pos for hidden neurons (polarizer axes)
        W_pos_hidden = self.W_pos[:N_h]  # [N, D]

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression weights (same as SGNNET_AntiHebbian)
        W_n = F.normalize(W_pos_hidden, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)

            # Gather neighbors
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # AH suppression
            Z_nb = Z_nb * supp_w

            # POLARIZER: filter incoming signals through receiving neuron's W_pos
            Z_nb = self._polarize(Z_nb, W_pos_hidden)

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_model(config_key):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    configs = {
        "Ref": ("none", 1.0),
        "A":   ("full", 1.0),
        "B":   ("partial", 0.5),
        "C":   ("partial", 0.3),
        "D":   ("rotation", 1.0),
    }
    mode, alpha = configs[config_key]
    return SGNNET_Polarizer(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                            polarizer_mode=mode, polarizer_alpha=alpha)


def main():
    all_keys = ["Ref", "A", "B", "C", "D"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: standard gather-sum (no polarizer)",
        "A":   "A: full polarizer (project onto W_pos, α=1.0)",
        "B":   "B: partial polarizer (α=0.5: 50% project + 50% original)",
        "C":   "C: soft polarizer (α=0.3: 30% project + 70% original)",
        "D":   "D: rotation in (Z, W_pos) plane by learned angle",
    }

    print(f"\n{'='*70}")
    print(f"Step 217 — Polarizer Routing (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M) — polarizer adds ~2× to routing step")
    print(f"Question: does input-dependent directional filtering improve routing?")
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

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "polarizer_mode": key,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 217 SUMMARY — Polarizer routing on step199 config")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if r["top1_best"] - ref_best > 0.005: verdict = " → ADVANCE"
            elif r["top1_best"] - ref_best > -0.01: verdict = " → NEUTRAL"
            else: verdict = " → KILL"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
