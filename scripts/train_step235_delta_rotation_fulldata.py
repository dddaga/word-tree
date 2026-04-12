"""Step 235: Delta-vector rotation — full data ± augmentation, ± AH.

MOTIVATION
==========
User-directed: Test delta-vector rotation (rotation in the plane of
ΔW = W_pos[receiver] - W_pos[sender]) with and without AH, at full
data and with horizontal flip augmentation.

If augmentation helps, we expand to more augmentations.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5)
  Ref_100     : Standard AH, 100% data, 150ep
  Ref_aug     : Standard AH, augmented data (2× hflip), 150ep
  A_100       : ΔW rotation (no AH), 100% data, 150ep
  A_aug       : ΔW rotation (no AH), augmented data, 150ep
  B_100       : ΔW rotation + AH, 100% data, 150ep
  B_aug       : ΔW rotation + AH, augmented data, 150ep

Delta-vector rotation: decompose Z_nb into components parallel and
perpendicular to ΔW = W_pos[i] - W_pos[j], then rotate by a learned
temperature × alignment angle. Preserves magnitude (no gate-death risk).
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
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42
DATA_100 = "data/store.h5"
DATA_AUG = "data/store_aug.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step235_delta_rotation_fulldata.json"


class SGNNET_DeltaRotation(nn.Module):
    """Delta-vector rotation: rotate Z_nb in the (Z_nb, ΔW) plane.

    ΔW = W_pos[receiver] - W_pos[sender] for each edge.
    Rotation angle = temperature × cos(Z_nb, ΔW).
    Preserves activation magnitude — no gate-death.

    use_ah: if True, apply AH suppression before rotation.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect,
                 alpha_ahebb=1.0, use_ah=True, seed=42):
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
        self.use_ah = use_ah

        # Learnable rotation temperature — init 0.5 (moderate rotation)
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
        W_n = F.normalize(W_h, dim=-1)

        # AH suppression weights (if enabled)
        supp_w = None
        if self.use_ah:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)  # [N, K_hh]
            supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)                 # [1, N, K_hh, 1]

        # Delta vectors: ΔW = W_pos[receiver] - W_pos[sender]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]                # [N, K_hh, D]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]  # [B, N, K_hh, D]

            # AH suppression first (if enabled)
            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            # Delta-vector rotation
            # Decompose Z_nb into parallel and perpendicular to ΔW
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)  # [B,N,K_hh,1]
            z_parallel = proj_coeff * delta_w_norm                         # [B,N,K_hh,D]
            z_perp = Z_nb - z_parallel
            z_perp_unit = F.normalize(z_perp, dim=-1)

            # Rotation angle: temperature × alignment coefficient
            theta_rot = self.rotation_temp * proj_coeff

            # Rotate, preserving magnitude
            z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Z_nb = (torch.cos(theta_rot) * Z_nb +
                    torch.sin(theta_rot) * z_perp_unit * z_mag)

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


def build_delta_rot(use_ah):
    """Delta-rotation model."""
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaRotation(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT,
        alpha_ahebb=ALPHA_AHEBB, use_ah=use_ah, seed=SEED)


def main():
    configs = {
        "Ref_100": ("ah",    True,  DATA_100),
        "Ref_aug": ("ah",    True,  DATA_AUG),
        "A_100":   ("delta", False, DATA_100),   # ΔW rotation, no AH, 100%
        "A_aug":   ("delta", False, DATA_AUG),   # ΔW rotation, no AH, augmented
        "B_100":   ("delta", True,  DATA_100),   # ΔW rotation + AH, 100%
        "B_aug":   ("delta", True,  DATA_AUG),   # ΔW rotation + AH, augmented
    }

    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 235 — Delta Rotation: full data ± augmentation, ± AH")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    # Pre-load both datasets
    data_cache = {}

    results = {}
    for key in run_keys:
        kind, use_ah, data_path = configs[key]
        data_full = str(ROOT / data_path)

        print(f"\n{'─'*60}")
        ah_str = "+AH" if use_ah else "no AH"
        aug_str = "augmented" if "aug" in data_path else "100% original"
        if kind == "ah":
            print(f"Config {key}: Standard AH, {aug_str}")
            model = build_ref()
        else:
            print(f"Config {key}: ΔW rotation {ah_str}, {aug_str}")
            model = build_delta_rot(use_ah)
        print(f"{'─'*60}")

        # Load data (cache to avoid re-reading)
        if data_full not in data_cache:
            data_cache[data_full] = make_loaders(data_full, batch_size=BATCH, seed=SEED)
        tr, va = data_cache[data_full]

        model = model.to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}  data={len(tr.dataset):,} train / {len(va.dataset):,} val")

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
            if ep % 10 == 0 or ep == 1 or ep == EPOCHS:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        # Log rotation temperature
        rot_temp = None
        if hasattr(model, 'rotation_temp'):
            rot_temp = round(model.rotation_temp.item(), 4)
            print(f"  final rotation_temp={rot_temp}")

        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "train_samples": len(tr.dataset),
            "use_ah": use_ah, "data": data_path,
            "rotation_temp": rot_temp,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 235 SUMMARY — Delta Rotation ± AH ± Augmentation")
    print(f"{'='*70}")

    # Group by mechanism
    ref_100 = results.get("Ref_100", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_100:+.4f}" if key != "Ref_100" else ""
        aug = "  (augmented)" if "aug" in r.get("data", "") else ""
        ah = " +AH" if r.get("use_ah") else " no-AH"
        kind = "AH-ref" if "Ref" in key else f"ΔW-rot{ah}"
        print(f"  {key:10s} ({kind}): {r['top1_best']:.4f}{delta}{aug}")

    # Augmentation delta analysis
    print(f"\nAugmentation effect:")
    for prefix in ["Ref", "A", "B"]:
        k100 = f"{prefix}_100"
        kaug = f"{prefix}_aug"
        if k100 in results and kaug in results:
            d = results[kaug]["top1_best"] - results[k100]["top1_best"]
            print(f"  {prefix}: aug vs 100% = {d:+.4f}pp "
                  f"({'helps' if d > 0.003 else 'neutral' if d > -0.003 else 'hurts'})")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
