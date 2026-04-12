"""Step 234: Delta-vector polarization — project/rotate along ΔW_pos.

MOTIVATION
==========
Current polarizer (step217b) projects Z_nb onto W_pos[receiver]. This uses
the receiver's IDENTITY as the filter axis. But the relationship between
two neurons is encoded in ΔW = W_pos[receiver] - W_pos[sender].

Proposal: polarize along the delta vector instead. This encodes the
edge relationship, not just the receiver identity.

Variants:
  A: ΔW projection (no AH) — pure delta-vector, replaces AH entirely
  B: ΔW projection + AH — delta-vector on top of AH suppression
  C: ΔW rotation + AH — rotate Z_nb in the (Z_nb, ΔW) plane
  D: W_pos projection + AH (step217b reference, over-polarizer α=1.5)
  Ref: Standard AH (no polarizer)

KEY QUESTIONS:
- Does the delta vector carry more information than receiver W_pos alone?
- Can ΔW projection REPLACE AH (config A)?
- Does ΔW rotation (in-plane, preserving magnitude) beat projection?

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
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

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

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

OUT_PATH = ROOT / "results" / "train_step234_delta_vector_polarizer.json"


class SGNNET_DeltaPolarizer(nn.Module):
    """Polarizer using ΔW_pos = W_pos[receiver] - W_pos[sender] as axis.

    Uses EXACT same forward structure as SGNNET_AntiHebbian to avoid step232-style bugs.
    Only the Z_nb modulation changes.

    Modes:
      "delta_proj"      — project Z_nb onto normalized ΔW direction
      "delta_proj_ah"   — AH suppression + ΔW projection
      "delta_rot_ah"    — AH suppression + rotation in (Z_nb, ΔW) plane
      "wpos_proj_ah"    — W_pos[receiver] projection + AH (step217b reference)
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 alpha_ahebb=1.0, mode="delta_proj_ah",
                 polarizer_alpha=1.5, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")

        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        self.alpha_ahebb = alpha_ahebb
        self.alpha_reflect = alpha_reflect
        self.mode = mode
        self.polarizer_alpha = polarizer_alpha

        # Learnable rotation temperature for rotation modes
        if "rot" in mode:
            self.rotation_temp = nn.Parameter(torch.tensor(0.5))

    @property
    def W_pos(self):
        return self.base.W_pos

    @property
    def W_phase(self):
        return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh       # [N, K_hh]
        N_h = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_h = self.base.W_pos[:N_h]       # [N, D]
        W_n = F.normalize(W_h, dim=-1)    # normalized positions

        # Pre-compute AH suppression if needed
        use_ah = "ah" in self.mode or self.mode == "wpos_proj_ah"
        supp_w = None
        if use_ah:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N, K_hh]
            supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                 # [1, N, K_hh, 1]

        # Pre-compute delta vectors: ΔW = W_pos[i] - W_pos[j] per edge
        # W_h[conn_hh] = [N, K_hh, D] (sender positions)
        # W_h.unsqueeze(1) = [N, 1, D] (receiver positions, broadcast)
        if "delta" in self.mode:
            delta_w = W_h.unsqueeze(1) - W_h[conn_hh]            # [N, K_hh, D]
            delta_w_norm = F.normalize(delta_w, dim=-1)            # normalized ΔW
            delta_w_norm = delta_w_norm.unsqueeze(0)               # [1, N, K_hh, D]

        # Pre-compute W_pos receiver axis for wpos_proj mode
        if self.mode == "wpos_proj_ah":
            w_recv = W_n.unsqueeze(0).unsqueeze(2)                # [1, N, 1, D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # Apply AH suppression first (if enabled)
            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            # Apply polarizer
            if self.mode == "delta_proj" or self.mode == "delta_proj_ah":
                # Project Z_nb onto ΔW direction
                proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)  # [B,N,K_hh,1]
                Z_projected = proj_coeff * delta_w_norm                        # [B,N,K_hh,D]
                # Over-project: α * projected + (1-α) * original
                α = self.polarizer_alpha
                if α == 1.0:
                    Z_nb = Z_projected
                else:
                    Z_nb = α * Z_projected + (1 - α) * Z_nb

            elif self.mode == "delta_rot_ah":
                # Rotate Z_nb in the plane defined by (Z_nb, ΔW)
                # Decompose Z_nb into parallel and perpendicular to ΔW
                proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
                z_parallel = proj_coeff * delta_w_norm
                z_perp = Z_nb - z_parallel
                z_perp_unit = F.normalize(z_perp, dim=-1)

                # Rotation angle proportional to alignment * temperature
                theta_rot = self.rotation_temp * proj_coeff        # [B,N,K_hh,1]

                # Preserve magnitude
                z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb = (torch.cos(theta_rot) * Z_nb +
                        torch.sin(theta_rot) * z_perp_unit * z_mag)

            elif self.mode == "wpos_proj_ah":
                # Step217b reference: project onto W_pos[receiver]
                proj_coeff = (Z_nb * w_recv).sum(dim=-1, keepdim=True)
                Z_projected = proj_coeff * w_recv
                α = self.polarizer_alpha
                Z_nb = α * Z_projected + (1 - α) * Z_nb

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_ref():
    """Standard SGNNET_AntiHebbian reference."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_delta(mode, polarizer_alpha=1.5):
    """Delta-vector polarizer model."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaPolarizer(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        alpha_ahebb=ALPHA_AHEBB, mode=mode,
        polarizer_alpha=polarizer_alpha, seed=SEED)


def main():
    configs = {
        "Ref": ("ah", None, None),
        "A":   ("delta", "delta_proj", 1.5),          # ΔW projection, NO AH
        "B":   ("delta", "delta_proj_ah", 1.5),       # ΔW projection + AH
        "C":   ("delta", "delta_rot_ah", None),        # ΔW rotation + AH
        "D":   ("delta", "wpos_proj_ah", 1.5),        # W_pos projection + AH (step217b ref)
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 234 — Delta-Vector Polarizer: ΔW_pos as projection/rotation axis")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        kind, mode, p_alpha = configs[key]
        print(f"\n{'─'*60}")
        if kind == "ah":
            print(f"Config {key}: Standard AH (reference)")
            model = build_ref()
        else:
            print(f"Config {key}: {mode}, polarizer_alpha={p_alpha}")
            model = build_delta(mode, polarizer_alpha=p_alpha or 1.0)
        print(f"{'─'*60}")

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  total params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        # CRITICAL: Trainer only optimizes W_pos + W_phase. Inject rotation_temp
        # into the optimizer so it actually receives gradient updates.
        if hasattr(model, 'rotation_temp'):
            trainer.optimizer.add_param_group({
                "params": [model.rotation_temp],
                "lr": kw.get("lr_wpos", 2.36e-3),
                "weight_decay": 0.0,
            })
            print(f"  [injected rotation_temp into optimizer]")

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "mode": mode or "ah",
            "polarizer_alpha": p_alpha,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 234 SUMMARY — Delta-Vector Polarizer")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key} ({r['mode']}): {r['top1_best']:.4f}{delta}")

    print(f"\nInterpretation:")
    print(f"  A > Ref  → ΔW projection alone can REPLACE AH (no cosine suppression needed)")
    print(f"  B > D    → ΔW axis carries MORE info than W_pos[receiver] axis")
    print(f"  B > Ref  → ΔW + AH compound works (unlike most compounds)")
    print(f"  C > B    → rotation > projection for delta-vector filtering")
    print(f"  D ≈ 217b → W_pos projection reference matches prior result")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
