"""Step 602 (B2-Teacher): GLNN-style knowledge distillation — train SGNNET teacher
and cache soft logits for MLP student distillation (step603).

MOTIVATION
==========
Candidate B2 from Opus mediation (walltime_breakthrough_2026-04-15.md):
GLNN (arXiv 2110.08727) shows GNN teachers can be distilled into MLP students
at 146-273× faster inference. If SGNNET's routing discovers structure that a
37-hidden-unit MLP can express, the deployed artifact becomes an MLP (0.088ms)
and the paper reframes from "SGNNET inference" to "SGNNET training-time teacher
discovers FC can be replaced by hidden-37 MLP."

This script:
1. Trains a fresh SGNNET ΔW-rotation teacher (step235 A_100 config: no AH, full data)
2. After training, runs inference over ALL train+val samples
3. Saves softmax logits at temperature T=4.0 to data/teacher_logits_step602.h5

Downstream: step603 trains MLP_37 student with combined KL + CE loss.

CONFIG: N=2048, D=16, K_hh=2, K_iter=5, alpha_reflect=0.5 (efficiency config).
Tier: Tier-2 (75ep full data — teacher needs to be strong).

Teacher reference: step235 A_aug best = 97.12%; target teacher ≥ 95.5%.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders, H5Dataset

parser = argparse.ArgumentParser(description="Step 602: Train SGNNET teacher + cache logits.")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--temp",    type=float, default=4.0,
                    help="Softmax temperature for cached logits (default 4.0)")
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--out_logits", default="data/teacher_logits_step602.h5",
                    help="Output path for cached teacher logits")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS         = args.epochs
BATCH          = 128
SEED           = args.seed
TEMP           = args.temp
DATA_PATH      = ROOT / args.data
OUT_LOGITS     = ROOT / args.out_logits

# Efficiency config (step199 baseline)
N    = 2048;  N_IN  = 25088;  N_OUT = 10
D    = 16;    K_HH  = 2;      K_IN  = 25;  K_ITER = 5
ALPHA_REFLECT = 0.5
ALPHA_TURING  = 0.0

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_JSON = ROOT / "results" / f"train_step602_b2_glnn_teacher_seed{SEED}__{SLOT}.json"


# ---------------------------------------------------------------------------
# Model — ΔW-rotation (no AH), same as step235 config A_100
# ---------------------------------------------------------------------------

class SGNNET_DeltaRotation(nn.Module):
    """Delta-vector rotation without AH (step235 A_100 winner config).

    Copied from step235 for self-containment.
    """

    def __init__(self, N_hidden, N_out, D, N_in, K_in, K_iter,
                 K_local, K_random, n_groups, alpha_reflect, seed=42):
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

        self.alpha_reflect = alpha_reflect
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
        conn_hh = self.base.conn_hh
        N_h     = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        W_h = self.base.W_pos[:N_h]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]            # [N, K_hh, D]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                     # [B, N, K_hh, D]

            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            z_parallel = proj_coeff * delta_w_norm
            z_perp     = Z_nb - z_parallel
            z_perp_unit = F.normalize(z_perp, dim=-1)

            theta_rot = self.rotation_temp * proj_coeff
            z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Z_nb  = (torch.cos(theta_rot) * Z_nb +
                     torch.sin(theta_rot) * z_perp_unit * z_mag)

            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_teacher():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaRotation(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT, seed=SEED)


# ---------------------------------------------------------------------------
# Logit caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_logits(model: nn.Module, split_loaders: dict[str, object], T: float, out_path: Path):
    """Run inference on each split, save softmax(logits/T) to HDF5.

    Args:
        split_loaders: {"train": DataLoader, "val": DataLoader}
        T:             softmax temperature
        out_path:      output .h5 path
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    key_name = f"logits_T{int(T)}"
    split_results = {}

    for split_name, loader in split_loaders.items():
        all_logits = []
        for feats, _soft, _labels in loader:
            feats = feats.to(DEVICE)
            logits = model(feats)                                    # [B, N_OUT]
            soft_T = F.softmax(logits / T, dim=-1).cpu().float()    # [B, N_OUT]
            all_logits.append(soft_T)
        logits_cat = torch.cat(all_logits, dim=0).numpy()           # [N_split, N_OUT]
        split_results[split_name] = logits_cat
        print(f"  [{split_name}] cached {len(logits_cat)} samples  shape={logits_cat.shape}")

    with h5py.File(out_path, "w") as f:
        for split_name, logits_arr in split_results.items():
            grp = f.require_group(split_name)
            grp.create_dataset(key_name, data=logits_arr, compression="gzip", compression_opts=4)
        # metadata
        f.attrs["temperature"]   = T
        f.attrs["step"]          = 602
        f.attrs["teacher_model"] = "SGNNET_DeltaRotation_A100"
        f.attrs["N"]             = N
        f.attrs["D"]             = D

    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 602 — GLNN Teacher: SGNNET ΔW-rotation (no AH), full data")
    print(f"  N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  alpha_reflect={ALPHA_REFLECT}")
    print(f"  epochs={EPOCHS}  seed={SEED}  T={TEMP}  device={DEVICE}")
    print(f"{'='*70}")

    tr, va = make_loaders(DATA_PATH, batch_size=BATCH, seed=SEED)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    model    = build_teacher().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  teacher params={n_params:,}")

    kw      = trainer_kwargs(N, n_epochs=EPOCHS)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    # rotation_temp needs to be in the optimizer
    if hasattr(model, "rotation_temp"):
        trainer.optimizer.add_param_group({
            "params": [model.rotation_temp],
            "lr": kw.get("lr_wpos", 2.36e-3),
            "weight_decay": 0.0,
        })
        print("  [injected rotation_temp into optimizer]")

    t0 = time.time()

    def _log(m):
        ep = m["epoch"] + 1
        if ep % 10 == 0 or ep == 1 or ep == EPOCHS:
            rot = round(model.rotation_temp.item(), 4) if hasattr(model, "rotation_temp") else None
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}"
                  + (f"  rot_temp={rot}" if rot is not None else ""), flush=True)

    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best  = max(top1h); bep = int(np.argmax(top1h)) + 1
    print(f"\n  Teacher training done: best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    # ------------------------------------------------------------------
    # Cache logits for ALL splits (full-batch inference)
    # ------------------------------------------------------------------
    print(f"\n  Caching logits at T={TEMP} ...")
    # Use full (non-shuffled) loaders so index order is deterministic
    tr_inf = torch.utils.data.DataLoader(
        tr.dataset, batch_size=256, shuffle=False, num_workers=0)
    va_inf = torch.utils.data.DataLoader(
        va.dataset, batch_size=256, shuffle=False, num_workers=0)

    cache_logits(model, {"train": tr_inf, "val": va_inf}, T=TEMP, out_path=OUT_LOGITS)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    rot_temp = round(model.rotation_temp.item(), 4) if hasattr(model, "rotation_temp") else None
    result = {
        "top1_best":        best,
        "top1_last":        top1h[-1],
        "best_epoch":       bep,
        "top1_history":     top1h,
        "elapsed_s":        round(elapsed, 1),
        "n_params":         n_params,
        "epochs":           EPOCHS,
        "seed":             SEED,
        "N":                N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
        "alpha_reflect":    ALPHA_REFLECT,
        "rotation_temp":    rot_temp,
        "teacher_logits":   str(OUT_LOGITS),
        "temperature_T":    TEMP,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 602 SUMMARY")
    print(f"  Teacher top1={best:.4f} @ ep{bep}")
    print(f"  Logits cached → {OUT_LOGITS}")
    print(f"  Results JSON  → {OUT_JSON}")
    print(f"  Next: run scripts/train_step603_b2_glnn_student.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
