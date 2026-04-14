"""Step 217b: Polarizer routing Tier-1 — full polarizer confirmed +1.27pp at Tier-0.

MOTIVATION
==========
step217-A (full polarizer): 92.89% vs 91.62% Ref = +1.27pp at Tier-0 (20ep/50%).
step217-B (partial α=0.5): 92.51% vs Ref = +0.89pp.

Advancing full polarizer (A) to Tier-1 (75ep/50% data) for reliable calibration.
Also testing α=1.5 (over-project) since the Tier-0 pattern was monotonic: more = better.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 75ep — Tier-1)
  Ref : Standard gather-sum (no polarizer)
  A   : Full polarizer (α=1.0) — Tier-0 winner +1.27pp
  B   : Over-polarizer (α=1.5) — extrapolate the monotonic trend
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step217b_polarizer_tier1.json"


class SGNNET_Polarizer(nn.Module):
    """SGNNET with polarizer routing — project Z onto W_pos before aggregation."""

    def __init__(self, base, resonant, alpha_ahebb=1.0, polarizer_alpha=1.0):
        super().__init__()
        self.base = base
        self.resonant = resonant
        self.alpha_ahebb = alpha_ahebb
        self.polarizer_alpha = polarizer_alpha

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h = self.base.N_hidden
        W_pos_hidden = self.W_pos[:N_h]

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression
        W_n = F.normalize(W_pos_hidden, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        # Polarizer axis
        w_pol = F.normalize(W_pos_hidden, dim=-1).unsqueeze(0).unsqueeze(2)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_nb = Z_nb * supp_w

            # Polarizer: project onto receiving neuron's W_pos
            if self.polarizer_alpha > 0:
                proj_coeff = (Z_nb * w_pol).sum(dim=-1, keepdim=True)
                Z_projected = proj_coeff * w_pol
                if self.polarizer_alpha == 1.0:
                    Z_nb = Z_projected
                else:
                    Z_nb = self.polarizer_alpha * Z_projected + (1 - self.polarizer_alpha) * Z_nb

            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


def build_model(polarizer_alpha):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_Polarizer(base, resonant, alpha_ahebb=ALPHA_AHEBB,
                            polarizer_alpha=polarizer_alpha)


def main():
    configs = {"Ref": 0.0, "A": 1.0, "B": 1.5}

    print(f"\n{'='*70}")
    print(f"Step 217b — Polarizer Tier-1 (75ep 50% data)")
    print(f"Tier-0 winner: full polarizer +1.27pp. Testing α=1.0 and α=1.5")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key, alpha in configs.items():
        print(f"\n{'─'*60}")
        print(f"Config {key}: polarizer_alpha={alpha}")
        print(f"{'─'*60}")

        model = build_model(alpha).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 10 == 0:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {"top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
                        "top1_history": top1h, "elapsed_s": round(elapsed, 1),
                        "polarizer_alpha": alpha, "n_params": n_p}
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["Ref"]["top1_best"]
    print(f"\n{'='*70}")
    print("STEP 217b SUMMARY — Polarizer Tier-1")
    for key in configs:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key} (α={configs[key]}): {r['top1_best']:.4f}{delta}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
