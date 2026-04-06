"""Step 129: Markov Routing Revival — unlock activation-dependent routing.

MOTIVATION
==========
The original SGNNET vision: activations propagate through the graph like a Markov
chain where (direction, magnitude) encodes routing history. Three architectural
constraints currently prevent this:

1. F.normalize() destroys magnitude every step → history lost
2. AH wpos is STATIC (computed from W_pos, not Z) → routing is topology-dependent
3. F.relu(Z-theta) is content-blind → same threshold for all neurons

50 experiments confirmed that mechanisms working WITHIN these constraints succeed
(Z-bias +7.42pp, redistribution +3.98pp), while mechanisms trying to ADD gating
on top die (gate-death theorem). The fix: relax the constraints themselves.

CHANGES TESTED
==============
1. Magnitude preservation: replace F.normalize() with soft-norm (only normalize
   if ||Z|| > 1). Magnitude carries history — strong paths stay strong.
2. Dynamic AH (zact): compute suppression from Z (activations) not W_pos (positions).
   Already coded as variant="zact" but only tested at D=16/N=512 (step16, lost).
   Never tested at D=32+/N=1024/patched arch/K_iter=12.
3. Hybrid AH (wpos + zact): static scaffold + dynamic modulation. Both must agree
   to suppress → less aggressive, more nuanced routing.

D=32 RATIONALE: 31/32 dimensions are Fourier spatial encoding, only 1 is data.
Mechanism experiments don't need full D=64. D=32 is 2x faster, step86-I showed
D=32 gives 95.95% at N=4096 — viable. Winners scale to D=64 in follow-up.

CONFIGS (N=1024, D=32, K_hh=4, K_iter=12, AH=1.0, turing=0.0, 50%/75ep)
=========================================================================
  Ref : F.normalize + wpos (static AH) — current baseline at D=32
  A   : soft-norm + wpos               — does magnitude preservation alone help?
  B   : F.normalize + zact (dynamic AH) — does zact work at D=32/N=1024?
  C   : soft-norm + zact               — magnitude + dynamic AH compound
  D   : soft-norm + hybrid (wpos×zact) — static scaffold + dynamic modulation
  E   : soft-norm + hybrid + redistribution (tau=0.3) — full Markov chain

To reproduce:
    python -u scripts/train_step129_markov_routing.py --device mps
    python -u scripts/train_step129_markov_routing.py --device mps --epochs 20  # scout
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=75,
                    help="Training epochs (default 75; use 20 for Tier-0 scout)")
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys to run (e.g. A,D). Empty = run all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 1024; N_IN = 25088; N_OUT = 10; D = 32; K_ITER = 12; K_IN = 50
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
STEP69_REF = 0.8336   # D=64 reference; D=32 Ref will be lower


# ---------------------------------------------------------------------------
# Model: Markov routing with configurable normalization and AH variant
# ---------------------------------------------------------------------------

class SGNNET_MarkovRouting(nn.Module):
    """SGNNET with configurable normalization, AH variant, and routing mode.

    Normalization modes:
      "unit"  — F.normalize (project to unit sphere, destroys magnitude)
      "soft"  — divide by max(||Z||, 1.0) — preserve magnitude if ||Z|| <= 1

    AH variants:
      "wpos"   — static, from W_pos cosine similarity (standard)
      "zact"   — dynamic, from Z cosine similarity each step
      "hybrid" — product of wpos and zact suppression

    Routing modes:
      "sum"    — standard AH-weighted sum of neighbors
      "redist" — softmax redistribution with AH logit + Z-dot score
    """

    def __init__(self, resonant, alpha_ahebb: float,
                 norm_mode: str = "unit",
                 ah_variant: str = "wpos",
                 routing_mode: str = "sum",
                 tau_0: float = 0.3,
                 alpha_dynamic: float = 0.5):
        super().__init__()
        self.m             = resonant
        self.alpha         = alpha_ahebb
        self.norm_mode     = norm_mode
        self.ah_variant    = ah_variant
        self.routing_mode  = routing_mode
        self.tau_0         = tau_0
        self.alpha_dynamic = alpha_dynamic

        # For redistribution routing, learned per-neuron temperature
        if routing_mode == "redist":
            N_h = resonant.base.N_hidden
            D_  = resonant.base.W_pos.shape[1]
            self.W_temp = nn.Parameter(torch.zeros(N_h, D_))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def _normalize(self, Z: torch.Tensor) -> torch.Tensor:
        """Normalize Z based on configured mode."""
        Z = Z.clamp(-10, 10)
        if self.norm_mode == "unit":
            return F.normalize(Z, dim=-1)
        elif self.norm_mode == "soft":
            # Only normalize if magnitude > 1. Preserves magnitude <= 1.
            norms = Z.norm(dim=-1, keepdim=True).clamp(min=1.0)
            return Z / norms
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base      = self.m.base
        Z         = base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = base.conn_hh
        N_h       = base.N_hidden

        # Static AH suppression (precomputed for wpos and hybrid)
        W_n     = F.normalize(self.m.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)   # [N_h, K_hh]

        if self.routing_mode == "redist":
            ah_logit_static = -self.alpha * pos_sim.clamp(min=0)  # [N_h, K_hh]

        if self.ah_variant == "wpos":
            supp_w_static = (1.0 - self.alpha * pos_sim.clamp(min=0)
                            ).unsqueeze(0).unsqueeze(-1)          # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]                         # [B, N, K_hh, D]

            # --- Compute aggregation weights based on AH variant + routing mode ---

            if self.routing_mode == "sum":
                # Sum aggregation with AH suppression weights
                if self.ah_variant == "wpos":
                    Z_struct = (Z_nb * supp_w_static).sum(dim=2)

                elif self.ah_variant == "zact":
                    # Dynamic: suppress neighbors with similar ACTIVATIONS
                    Z_n    = F.normalize(Z_fwd, dim=-1)
                    Z_nb_n = F.normalize(Z_nb, dim=-1)
                    z_sim  = (Z_n.unsqueeze(2) * Z_nb_n).sum(-1)        # [B, N, K_hh]
                    supp_z = (1.0 - self.alpha * z_sim.clamp(min=0)
                             ).unsqueeze(-1)                              # [B, N, K_hh, 1]
                    Z_struct = (Z_nb * supp_z).sum(dim=2)

                elif self.ah_variant == "hybrid":
                    # Product of static (wpos) and dynamic (zact) suppression
                    supp_static = (1.0 - self.alpha * pos_sim.clamp(min=0))  # [N, K_hh]
                    Z_n    = F.normalize(Z_fwd, dim=-1)
                    Z_nb_n = F.normalize(Z_nb, dim=-1)
                    z_sim  = (Z_n.unsqueeze(2) * Z_nb_n).sum(-1)        # [B, N, K_hh]
                    supp_dynamic = (1.0 - self.alpha_dynamic * z_sim.clamp(min=0))  # [B, N, K_hh]
                    supp_combined = (supp_static.unsqueeze(0) * supp_dynamic
                                    ).unsqueeze(-1)                       # [B, N, K_hh, 1]
                    Z_struct = (Z_nb * supp_combined).sum(dim=2)

            elif self.routing_mode == "redist":
                # Softmax redistribution with Z-dot score + AH logit
                z_score = (Z_fwd.unsqueeze(2) * Z_nb).sum(-1)           # [B, N, K_hh]

                # Per-neuron learned temperature
                temp_score = (self.W_temp * Z_fwd).sum(-1)              # [B, N]
                temp = self.tau_0 * (1.0 + torch.sigmoid(temp_score))   # [B, N]

                # AH logit depends on variant
                if self.ah_variant in ("wpos", "hybrid"):
                    ah_logit = ah_logit_static.unsqueeze(0)             # [1, N, K_hh]
                else:  # zact
                    Z_n    = F.normalize(Z_fwd, dim=-1)
                    Z_nb_n = F.normalize(Z_nb, dim=-1)
                    z_sim  = (Z_n.unsqueeze(2) * Z_nb_n).sum(-1)
                    ah_logit = (-self.alpha * z_sim.clamp(min=0))        # [B, N, K_hh]

                if self.ah_variant == "hybrid":
                    # Add dynamic component to static logit
                    Z_n    = F.normalize(Z_fwd, dim=-1)
                    Z_nb_n = F.normalize(Z_nb, dim=-1)
                    z_sim  = (Z_n.unsqueeze(2) * Z_nb_n).sum(-1)
                    ah_logit = ah_logit + (-self.alpha_dynamic * z_sim.clamp(min=0))

                logit = z_score / temp.unsqueeze(-1) + ah_logit
                w     = F.softmax(logit, dim=2)                         # [B, N, K_hh]
                Z_struct = (w.unsqueeze(-1) * Z_nb).sum(dim=2)

            # Reflection accumulator
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z = self._normalize(Z_new)

        return base._readout(Z)


# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------

@dataclass
class Config:
    key: str
    label: str
    norm_mode: str       # "unit" or "soft"
    ah_variant: str      # "wpos", "zact", "hybrid"
    routing_mode: str    # "sum" or "redist"
    use_markov: bool = True


CONFIGS = [
    Config("Ref", "Ref  F.normalize + wpos + sum (D=32 baseline)",
           "unit", "wpos", "sum", use_markov=False),
    Config("A",   "A    soft-norm + wpos + sum (magnitude only)",
           "soft", "wpos", "sum"),
    Config("B",   "B    F.normalize + zact + sum (dynamic AH only)",
           "unit", "zact", "sum"),
    Config("C",   "C    soft-norm + zact + sum (magnitude + dynamic)",
           "soft", "zact", "sum"),
    Config("D",   "D    soft-norm + hybrid + sum (static scaffold + dynamic)",
           "soft", "hybrid", "sum"),
    Config("E",   "E    soft-norm + hybrid + redistribution (full Markov)",
           "soft", "hybrid", "redist"),
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: Config, seed_offset: int = 0) -> nn.Module:
    torch.manual_seed(SEED + seed_offset)
    topo = topology_kwargs(N)
    topo.pop("K_in", None)
    topo.pop("K_iter", None)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, encoding_mode="fourier",
        **topo,
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
        alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
        mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    if not cfg.use_markov:
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return SGNNET_MarkovRouting(
        resonant, alpha_ahebb=ALPHA_AHEBB,
        norm_mode=cfg.norm_mode, ah_variant=cfg.ah_variant,
        routing_mode=cfg.routing_mode, tau_0=0.3, alpha_dynamic=0.5,
    )


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data loaders (cached, 50% data)
# ---------------------------------------------------------------------------

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 129 — Markov Routing Revival")
    print(f"N={N}  D={D}  K_iter={K_ITER}  K_hh=4  AH={ALPHA_AHEBB}")
    print(f"Epochs={EPOCHS}  Device={DEVICE}  Data=50%")
    print(f"{'='*70}\n")

    print("Configs:")
    for c in CONFIGS:
        print(f"  {c.key:4s}  norm={c.norm_mode:4s}  ah={c.ah_variant:6s}  "
              f"route={c.routing_mode:6s}  {c.label}")
    print()

    get_loaders()
    results  = {}
    out_path = ROOT / "results" / "train_step129_markov_routing.json"

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active_configs = [(i, cfg) for i, cfg in enumerate(CONFIGS)
                      if not cfg_filter or cfg.key in cfg_filter]

    for i, cfg in active_configs:
        model    = make_model(cfg, seed_offset=i).to(DEVICE)
        n_params = count_params(model)

        print(f"\n{'─'*60}")
        print(f"Config {cfg.key}: {cfg.label}")
        print(f"  params={n_params:,}  norm={cfg.norm_mode}  ah={cfg.ah_variant}  route={cfg.routing_mode}")
        print(f"{'─'*60}")

        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        trainer = Trainer(
            model=model,
            train_loader=get_loaders()[0],
            val_loader=get_loaders()[1],
            device=DEVICE,
            **kw,
        )

        history   = trainer.train(n_epochs=EPOCHS)
        elapsed   = time.time() - t0

        top1_hist = [round(h.get("val_top1", 0.0), 4) for h in history]
        top1_best = max(top1_hist)
        best_ep   = int(np.argmax(top1_hist)) + 1

        results[cfg.key] = {
            "N": N, "D": D, "K_iter": K_ITER,
            "norm_mode": cfg.norm_mode,
            "ah_variant": cfg.ah_variant,
            "routing_mode": cfg.routing_mode,
            "alpha_ahebb": ALPHA_AHEBB,
            "data_frac": 0.5,
            "top1_best": top1_best, "top1_last": top1_hist[-1],
            "best_epoch": best_ep, "epochs_run": len(history),
            "top1_history": top1_hist,
            "elapsed_s": round(elapsed, 1),
            "n_params": n_params,
            "label": cfg.label,
        }

        # vs Ref (compute once Ref is available)
        ref_best = results.get("Ref", {}).get("top1_best", 0)
        vs_ref = top1_best - ref_best if ref_best > 0 else 0

        print(f"\n  top1_best={top1_best:.4f}  vs_Ref={vs_ref:+.4f}  "
              f"elapsed={elapsed/60:.1f}min")

        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table
    print(f"\n{'='*70}")
    print(f"STEP 129 SUMMARY (D={D})")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    print(f"{'Config':<6}  {'norm':>4}  {'ah':>6}  {'route':>6}  "
          f"{'params':>8}  {'top1':>7}  {'vs_Ref':>8}")
    print(f"{'─'*70}")
    for key, r in results.items():
        vs = r["top1_best"] - ref_best if ref_best > 0 else 0
        print(f"{key:<6}  {r['norm_mode']:>4}  {r['ah_variant']:>6}  "
              f"{r['routing_mode']:>6}  {r['n_params']:>8,}  "
              f"{r['top1_best']:.4f}  {vs:>+.4f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
