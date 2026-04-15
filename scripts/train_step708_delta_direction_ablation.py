"""Step 708: ΔW projection — is the relational axis specifically needed? (Tier-0 ablation)

MOTIVATION
==========
step706 confirmed ΔW proj +1.56pp at N=2048 Tier-2 (96.87% vs 95.31%).
BUT: is the specific ΔW = W_pos[receiver] - W_pos[sender] direction load-bearing,
or does any directional filter provide the same benefit?

ABLATION CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, Tier-0, 20ep, 50% data):
  Ref        : AH α=1.0 (efficiency config baseline, step199)
  A_proj     : ΔW projection — proj onto W_pos[recv]-W_pos[send] (step706 winner)
  B_random   : Random fixed unit vector per edge — NOT ΔW; tests direction specificity
  C_wpos_recv: W_pos[receiver] direction only — no delta; tests "receiver position alone"
  D_sender   : -W_pos[sender] direction — does sender matter more than delta?

Predictions:
  A_proj ≈ B_random >> Ref → ANY directional filter helps; ΔW not specifically needed
  A_proj >> B_random ≈ Ref → ΔW relational axis IS load-bearing; mechanism specific
  A_proj ≈ C_wpos_recv     → receiver position alone sufficient; delta is noise
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

parser = argparse.ArgumentParser(description="Step 708: ΔW direction ablation")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys. Empty = all.")
args   = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

FLOPS    = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step708_delta_direction_ablation.json"


# ── Directional projection model ─────────────────────────────────────────────────

class SGNNET_DirectionProj(nn.Module):
    """Generalised directional projection for ablation.

    direction_mode:
      'delta'      — ΔW = W_pos[recv] - W_pos[send]  (step706 winner)
      'random'     — random fixed unit vector per edge (frozen at init)
      'recv_only'  — W_pos[receiver] unit vector (no delta)
      'sender_neg' — -W_pos[sender] unit vector (sender direction only)
    """
    def __init__(self, base: SGNNET_Resonant, direction_mode: str = "delta"):
        super().__init__()
        assert direction_mode in ("delta", "random", "recv_only", "sender_neg")
        self.m              = base
        self.direction_mode = direction_mode
        self._rand_dir      = None  # precomputed for 'random' mode

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _get_proj_dir(self, W_h: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
        """Return [1, N, K_hh, D] unit projection direction."""
        N_h = W_h.shape[0]
        if self.direction_mode == "delta":
            delta_w = W_h.unsqueeze(1) - W_h[conn_hh]   # [N, K_hh, D]
            return F.normalize(delta_w, dim=-1).unsqueeze(0)

        elif self.direction_mode == "random":
            if self._rand_dir is None or self._rand_dir.device != W_h.device:
                g = torch.Generator().manual_seed(999)   # CPU generator
                r = torch.randn(N_h, K_HH, D, generator=g)  # generate on CPU
                self._rand_dir = F.normalize(r, dim=-1).to(W_h.device)  # move to device
            return self._rand_dir.unsqueeze(0)

        elif self.direction_mode == "recv_only":
            recv_dir = F.normalize(W_h, dim=-1)                         # [N, D]
            return recv_dir.unsqueeze(1).unsqueeze(0).expand(1, N_h, K_HH, D)

        else:  # sender_neg
            send_dir = -F.normalize(W_h[conn_hh], dim=-1)               # [N, K_hh, D]
            return send_dir.unsqueeze(0)

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        W_h      = self.m.W_pos[:N_h]
        proj_dir = self._get_proj_dir(W_h, conn_hh)   # [1, N, K_hh, D] — computed each fwd

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]               # [B, N, K_hh, D]

            proj_coeff = (Z_nb * proj_dir).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()

            Z_struct    = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# ── Builders ─────────────────────────────────────────────────────────────────────

def _base_resonant():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, mode="dynamic_z_geo",
    )


def build_ref():
    torch.manual_seed(SEED)
    return SGNNET_AntiHebbian(_base_resonant(), alpha_ahebb=1.0, variant="wpos")


def build_dir_proj(direction_mode: str):
    torch.manual_seed(SEED)
    return SGNNET_DirectionProj(_base_resonant(), direction_mode=direction_mode)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A_proj", "B_random", "C_recv", "D_sender"]
    run_keys = ([k.strip() for k in args.configs.split(",")]
                if args.configs else all_keys)

    labels = {
        "Ref":      "Ref: AH α=1.0 (efficiency baseline)",
        "A_proj":   "A_proj: ΔW = recv-send direction (step706 winner)",
        "B_random": "B_random: random fixed unit vector per edge",
        "C_recv":   "C_recv: W_pos[receiver] direction only",
        "D_sender": "D_sender: -W_pos[sender] direction only",
    }
    modes = {
        "A_proj": "delta",
        "B_random": "random",
        "C_recv": "recv_only",
        "D_sender": "sender_neg",
    }

    print(f"\n{'='*70}")
    print(f"Step 708 — ΔW direction ablation (is relational axis specifically needed?)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data Tier-0")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)  Device={DEVICE}")
    print(f"step706: A_proj=96.87% vs Ref=95.31% (+1.56pp) at Tier-2")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_total = len(tr_full.dataset)
    idx = torch.randperm(n_total,
                         generator=torch.Generator().manual_seed(SEED))[:n_total // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True,
                                     num_workers=0,
                                     generator=torch.Generator().manual_seed(SEED))
    print(f"Train={len(subset)}  Val={len(va.dataset)}")

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        model = build_ref() if key == "Ref" else build_dir_proj(modes[key])
        model = model.to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: (
            print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
            if (m['epoch'] + 1) % 5 == 0 else None
        ))
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best  = max(top1h)
        bep   = int(np.argmax(top1h)) + 1
        results[key] = {
            "label":        labels.get(key, key),
            "top1_best":    best,
            "top1_last":    top1h[-1],
            "best_epoch":   bep,
            "epochs_run":   len(history),
            "top1_history": top1h,
            "elapsed_s":    round(elapsed, 1),
            "n_params":     n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "flops": FLOPS,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    ref_best = results.get("Ref", {}).get("top1_best", 0.)

    print(f"\n{'='*70}")
    print("STEP 708 SUMMARY — ΔW direction ablation")
    print(f"{'='*70}")
    print(f"  step706 Tier-2 reference: Ref=95.31% A_proj=96.87% (+1.56pp)")
    print(f"  This Tier-0 run:")
    print(f"  {'Config':<14} {'best':>7} {'Δref':>7} {'interpretation'}")
    print(f"  {'-'*65}")
    for key in run_keys:
        r  = results[key]
        d  = f"{r['top1_best']-ref_best:+.4f}" if key != "Ref" else "     —"
        print(f"  {key:<14} {r['top1_best']:>7.4f} {d:>7}")

    # Interpretation
    a_best = results.get("A_proj", {}).get("top1_best", ref_best)
    b_best = results.get("B_random", {}).get("top1_best", ref_best)
    if "A_proj" in results and "B_random" in results:
        ab_gap = a_best - b_best
        if abs(ab_gap) < 0.005:
            interp = "ANY directional filter helps — ΔW not specifically needed"
        elif ab_gap > 0.010:
            interp = "ΔW relational axis IS specifically load-bearing"
        else:
            interp = f"Ambiguous (A_proj vs B_random gap={ab_gap:+.4f}) → Tier-1 needed"
        print(f"\n  Interpretation: {interp}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
