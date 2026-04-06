"""Step 96: DropMessage during K_iter routing.

MOTIVATION
==========
DropMessage (AAAI 2023) randomly drops messages during propagation — more
fine-grained than DropEdge. Proven anti-over-smoothing at depth.

SGNNET uses K_iter=8-12 routing steps (equivalent to 8-12 GNN layers).
Over-smoothing risk is real at this depth — step71 showed K_iter=16
outperforms K_iter=12 only slightly, and K_iter=24 crashes (−3.57pp).
DropMessage could extend the usable K_iter range.

Implementation: at each K_iter step during training, randomly zero out
a fraction of the gathered neighbor messages Z_nb before aggregation.
At eval time, no dropping (standard practice).

This is ~5 lines of code. Zero new parameters. Orthogonal to AH.

CONFIGS (N=1024, D=64, K_iter=8, AH=1.0, turing=0.0, 50%/75ep)
=================================================================
  Ref : no drop (step69 Ref baseline)
  A   : drop=0.1 (10% messages dropped per step)
  B   : drop=0.2 (20%)
  C   : drop=0.3 (30%)
  D   : drop=0.2, K_iter=16 (test: does drop enable deeper routing?)

If D > B at K_iter=16: DropMessage unlocks depth otherwise blocked.

To reproduce:
    python -u scripts/train_step96_dropmessage.py --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 64
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336


class SGNNET_AH_DropMessage(nn.Module):
    """SGNNET_AntiHebbian with DropMessage during K_iter routing."""

    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float,
                 drop_rate: float):
        super().__init__()
        self.m     = resonant
        self.alpha = alpha_ahebb
        self.drop  = drop_rate

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden

        # AH wpos suppression (static)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w  = (1.0 - self.alpha * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd  = F.relu(Z - theta_pos)
            Z_nb   = Z_fwd[:, conn_hh, :]                         # [B,N,K_hh,D]

            # DropMessage: zero out random messages during training
            if self.training and self.drop > 0:
                mask = torch.bernoulli(
                    torch.full(Z_nb.shape[:3], 1.0 - self.drop,
                               device=Z_nb.device)
                ).unsqueeze(-1)                                    # [B,N,K_hh,1]
                Z_nb = Z_nb * mask / (1.0 - self.drop)            # scale to preserve mean

            Z_struct = (Z_nb * supp_w).sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


@dataclass
class Config:
    key: str; label: str; drop_rate: float; K_iter: int


CONFIGS = [
    Config("Ref", "Ref  no drop, K_iter=8",            0.0, 8),
    Config("A",   "A    drop=0.1, K_iter=8",            0.1, 8),
    Config("B",   "B    drop=0.2, K_iter=8",            0.2, 8),
    Config("C",   "C    drop=0.3, K_iter=8",            0.3, 8),
    Config("D",   "D    drop=0.2, K_iter=16 (depth)",   0.2, 16),
]

_loaders = None

def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    tk = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=N_IN, N_hidden=N, N_out=N_OUT,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=cfg.K_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if cfg.drop_rate == 0.0:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_AH_DropMessage(resonant, ALPHA_AHEBB, cfg.drop_rate)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(cfg, model, meta):
    print(f"\n{'='*70}\n{cfg.label}\n{'='*70}")
    tr, va = get_loaders()
    tk = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0 = time.time(); history = trainer.train(n_epochs=EPOCHS); elapsed = time.time() - t0
    top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
    best = max(top1_hist); best_ep = int(np.argmax(top1_hist)) + 1
    frac = best_ep / len(history)
    result = {
        "label": cfg.label, "drop_rate": cfg.drop_rate, "K_iter": cfg.K_iter,
        "top1_best": best, "top1_last": history[-1].get("val_top1", 0.0),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1), "best_epoch_frac": round(frac, 3),
        "step69_ref": STEP69_REF, "delta_vs_ref": round(best - STEP69_REF, 4),
        "params": count_params(model), "top1_history": top1_hist,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1={best:.4f}  ep={best_ep}/{len(history)}  "
          f"vs_ref={best-STEP69_REF:+.4f}  t={elapsed:.0f}s")
    return result


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  N={N}  Data: 50%")
    print(f"Step 96: DropMessage during K_iter routing")
    print(f"Baseline: step69 Ref = {STEP69_REF:.4f}\n")
    for c in CONFIGS:
        print(f"  {c.key:4s}  drop={c.drop_rate}  K_iter={c.K_iter}")
    print()
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")
    results = {}
    out_path = ROOT / "results" / "train_step96_dropmessage.json"
    for i, cfg in enumerate(CONFIGS):
        model = make_model(cfg, seed_offset=i).to(DEVICE)
        meta = {"N": N, "D": D, "K_iter": cfg.K_iter, "drop_rate": cfg.drop_rate,
                "alpha_ahebb": ALPHA_AHEBB, "data_frac": 0.5}
        results[cfg.key] = run(cfg, model, meta)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  [saved {out_path.name}]")
    print(f"\n{'='*70}\nSTEP 96 COMPLETE\n")
    print(f"  {'Key':4s}  {'drop':>5s}  {'K':>3s}  {'top1':>8s}  {'vs_ref':>8s}")
    for c in CONFIGS:
        if c.key in results:
            r = results[c.key]
            print(f"  {c.key:4s}  {c.drop_rate:>5.1f}  {c.K_iter:>3d}  "
                  f"{r['top1_best']:.4f}  {r['delta_vs_ref']:+.4f}")
    w = max(results, key=lambda k: results[k]["top1_best"])
    print(f"\n  Winner: {w}")
