"""Step 612: Group-level ΔW — routing granularity probe.

MOTIVATION
==========
SGNNET's routing uses per-neuron ΔW direction vectors (N×D params).
This experiment replaces them with per-GROUP directions (g×D params),
shared across all neurons in each group (8 neurons per group at N=2048, g=256).
Result directly answers: does routing signal live at neuron or group granularity?

If STRONG (within 0.5pp of step235): group routing suffices → 6.8× ΔW param reduction,
  same accuracy. Paper: "group-level routing is sufficient; per-neuron MoE is over-parameterized."
If ABANDON (>3pp below step235): per-neuron routing is real →
  Paper: "neuron-level specialization confirmed; standard MoE granularity is too coarse."

From moe_hybrid_meditation_2026-04-15.md: step612 is the decisive experiment.
Structurally safest routing experiment so far: no new gate, no softmax, no auxiliary loss.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, T0 20ep 50% data)
  Ref     : step235-style per-neuron ΔW (W_pos[i]−W_pos[j]), baseline
  GroupDW : per-group learned direction vector ΔW_g ∈ R^D, shared within group

Acceptance:
  STRONG  : GroupDW ≥ Ref − 0.5pp  (group routing suffices, 6.8× ΔW savings)
  MEDIUM  : GroupDW ≥ Ref − 2pp    (partial granularity dependence, T1 warranted)
  WEAK    : GroupDW ≥ Ref − 3pp    (still worth T1 — mechanism question, not sweep)
  ABANDON : GroupDW <  Ref − 3pp   (neuron-level routing is doing real work)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld  import SGNNET_SmallWorld
from src.sgnnet.model_resonant    import SGNNET_Resonant
from src.training.trainer         import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset         import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,GroupDW")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; N_GROUPS = max(8, N // 8)  # 256

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step612_group_dw_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class SGNNET_PerNeuronDW(nn.Module):
    """Ref: step235-style ΔW projection routing — direction = W_pos[i] - W_pos[j]."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh   = self.base.conn_hh
        N_h       = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h       = self.base.W_pos[:N_h]                        # [N, D]
        delta_w   = W_h.unsqueeze(1) - W_h[conn_hh]             # [N, K_hh, D]
        dw_norm   = F.normalize(delta_w, dim=-1).unsqueeze(0)   # [1, N, K_hh, D]
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd   = F.relu(Z - theta_pos)
            Z_nb    = Z_fwd[:, conn_hh, :]
            proj    = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb    = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_reflected = ALPHA_REFLECT * Z_reflected + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


class SGNNET_GroupDW(nn.Module):
    """Group-level ΔW: one learned direction per group, shared across 8 neurons.

    ΔW param count: g × D = 256 × 16 = 4,096  (vs per-neuron N × D = 32,768)
    Total params: ~6,304  (vs ~34,976 for Ref) — 6.8× reduction in ΔW, 5.5× total.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        K_r = max(1, K_HH // 4); K_l = K_HH - K_r
        self.base = SGNNET_SmallWorld(
            N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
            K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
            n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)

        # Group-level ΔW: one direction per group, NOT derived from W_pos difference
        self.dw_group = nn.Parameter(
            torch.randn(N_GROUPS, D) * (1.0 / D**0.5))  # init on sphere

        # Precompute group assignment for each of the N hidden neurons
        neurons_per_group = N // N_GROUPS  # = 8
        group_of = torch.arange(N) // neurons_per_group  # [N]
        self.register_buffer("group_of", group_of)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase
    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"): self.resonant.tick_epoch()

    def forward(self, x):
        Z = self.base._seed(x)
        conn_hh   = self.base.conn_hh
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Per-neuron direction = its group's shared ΔW vector
        dw_per_neuron = self.dw_group[self.group_of]           # [N, D]
        dw_norm = F.normalize(dw_per_neuron, dim=-1)           # [N, D]
        # Expand to match [1, N, K_hh, D] shape used in routing
        dw_norm = dw_norm.unsqueeze(1).expand(-1, K_HH, -1).unsqueeze(0)  # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.base.K_iter):
            Z_fwd   = F.relu(Z - theta_pos)
            Z_nb    = Z_fwd[:, conn_hh, :]
            proj    = (Z_nb * dw_norm).sum(-1, keepdim=True)
            Z_nb    = Z_nb * proj.abs()
            Z_struct = Z_nb.sum(dim=2)
            Z_reflected = ALPHA_REFLECT * Z_reflected + (Z_fwd - Z)
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        return self.base._readout(Z)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    tr_full, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g)[:n // 2]
    tr  = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0)

    print(f"\nStep 612 — Group-level ΔW granularity probe (T0 {EPOCHS}ep 50% data)")
    print(f"  device={DEVICE}  seed={SEED}  N_groups={N_GROUPS}")
    print(f"  Ref: 34,976 params (per-neuron ΔW, N×D=32,768)")
    print(f"  GroupDW: ~6,304 params (per-group ΔW, g×D=4,096)")

    results: dict = {}
    for key in keys:
        print(f"\n{'─'*60}\nConfig: {key}\n{'─'*60}")
        torch.manual_seed(SEED)
        model = SGNNET_PerNeuronDW() if key == "Ref" else SGNNET_GroupDW()
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw      = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.0), 4) for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        print(f"  → best={best:.4f} @ep{bep}  ({elapsed:.0f}s)")

        results[key] = {
            "top1_best": best, "best_epoch": bep, "top1_last": top1h[-1],
            "top1_history": top1h, "n_params": n_p, "elapsed_s": round(elapsed, 1),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref = results.get("Ref", {}).get("top1_best", 0.0)
    grp = results.get("GroupDW", {}).get("top1_best")
    print(f"\n{'='*70}\nSTEP 612 SUMMARY\n{'='*70}")
    for k, r in results.items():
        delta = f"  Δ={r['top1_best']-ref:+.4f}" if k != "Ref" else ""
        print(f"  {k:<10}: {r['top1_best']:.4f}{delta}  params={r['n_params']:,}")

    if grp is not None and ref > 0:
        d = grp - ref
        print(f"\nVerdict: GroupDW Δ = {d:+.4f} vs per-neuron Ref")
        if d >= -0.005:
            print("  → STRONG: group routing suffices. ΔW granularity = group, not neuron.")
            print("     Paper: 'group-level routing sufficient at 8-neuron granularity.'")
        elif d >= -0.02:
            print("  → MEDIUM: partial dependence. Run T1 to confirm.")
        elif d >= -0.03:
            print("  → WEAK: routing granularity matters somewhat. T1 warranted (mechanism Q).")
        else:
            print("  → ABANDON: per-neuron routing is real. Neuron-level MoE framing CONFIRMED.")


if __name__ == "__main__":
    main()
