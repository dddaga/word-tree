"""Step 320: Critical ablation — F.normalize removal at N=2048 D=16.

CLAIM UNDER TEST
================
"F.normalize after every routing step is load-bearing" — prior evidence
(step129 at N=1024) showed −50 to −71pp without it.

This rerun validates the claim at the current efficiency-frontier base
(N=2048, D=16, step199 config). If the claim holds, the result should
collapse to <50%. If accuracy remains high without F.normalize, the
claim is refuted — a foundational paper correction.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep Tier-0)
  Ref  : Standard AH + F.normalize (step199 architecture)
  A    : Standard AH + clamp only (no F.normalize)
  B    : Standard AH + RMSNorm (norm without unit-sphere projection)
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
parser.add_argument("--configs", default="")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step320_fnormalize_ablation.json"


class SGNNET_AHNoNorm(nn.Module):
    """SGNNET_AntiHebbian variant with configurable normalization mode.

    norm_mode:
      'l2'      : F.normalize(dim=-1)  — default (step199 architecture)
      'clamp'   : just clamp(-10, 10)  — no normalization at all
      'rms'     : RMSNorm (divide by RMS) — norm without sphere projection
    """
    def __init__(self, base: SGNNET_Resonant, alpha_ahebb=1.0, norm_mode="l2"):
        super().__init__()
        self.m = base
        self.alpha_ahebb = alpha_ahebb
        self.norm_mode = norm_mode

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _normalize(self, Z):
        if self.norm_mode == "l2":
            return F.normalize(Z.clamp(-10, 10), dim=-1)
        elif self.norm_mode == "rms":
            rms = Z.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            return (Z / rms).clamp(-10, 10)
        else:  # clamp only
            return Z.clamp(-10, 10)

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh

        N_h = self.m.base.N_hidden
        W_n = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = self._normalize(Z_new)
        return self.m.base._readout(Z)


def build(norm_mode):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    if norm_mode == "l2":
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AHNoNorm(resonant, alpha_ahebb=ALPHA_AHEBB, norm_mode=norm_mode)


def main():
    configs = {"Ref": "l2", "A": "clamp", "B": "rms"}
    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 320 — F.normalize ablation at N=2048 D=16")
    print(f"Testing claim: 'F.normalize is load-bearing' (step129)")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        norm_mode = configs[key]
        print(f"\n{'─'*60}")
        print(f"Config {key}: norm_mode={norm_mode}")
        print(f"{'─'*60}")

        model = build(norm_mode).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)

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
            "n_params": n_p, "norm_mode": norm_mode,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"\n{'='*70}")
    print("STEP 320 SUMMARY — F.normalize ablation")
    print(f"{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        print(f"  {key} ({r['norm_mode']}): {r['top1_best']:.4f}{delta}")

    print(f"\nInterpretation:")
    print(f"  A close to Ref → F.normalize NOT load-bearing (refutes step129 claim)")
    print(f"  A collapses    → F.normalize IS load-bearing (confirms claim)")
    print(f"  B (RMS) close  → L2 sphere projection specific, but some norm needed")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
