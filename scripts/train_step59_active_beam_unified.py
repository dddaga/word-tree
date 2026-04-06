"""Step 59: Active beam unified routing ablation.

Tests whether restricting broadcasts to top-K active neurons (beam gating)
improves over the standard AntiHebb baseline.

SGNNET_BeamUnified is defined inline — a one-off routing variant, no separate module.

Beam gate logic:
  - Top-M neurons by activation magnitude are designated as "senders"
  - Non-beam neurons only receive; they do not contribute to Z_nb gather
  - Option C adds active-only normalization: normalize only over neurons where
    Z.norm > theta (active set), leaving inactive neurons at zero

Configs:
  Ref : AntiHebbian alpha=1.0 wpos (baseline)
  A   : beam gates structural channel only (top-beam neurons send via conn_hh)
  B   : beam gates both structural AND phase channels
  C   : active-only normalization + beam gates both channels (full design)

Questions:
  A vs Ref : does restricting senders improve routing signal quality?
  B vs A   : does gating phase channel too help or hurt?
  C vs B   : does active-only normalization add on top of full beam gating?

To reproduce:
    python -u scripts/train_step59_active_beam_unified.py --device mps
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
from src.training.experiment_config   import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_resonant(N=1024, D=64, K_iter=8) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=16,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=0.5,
    )


# ---------------------------------------------------------------------------
# SGNNET_BeamUnified — inline one-off routing variant
# ---------------------------------------------------------------------------

class SGNNET_BeamUnified(nn.Module):
    """Active beam unified routing.

    Parameters
    ----------
    base_model      : SGNNET_Resonant backbone
    gate_phase      : if True, beam gate is also applied to the phase inhibition channel
    active_norm     : if True, normalize only over neurons with Z.norm > theta (active set)
    beam_size       : number of top-magnitude neurons that are allowed to send
    """

    def __init__(self, base_model: SGNNET_Resonant, gate_phase: bool = False,
                 active_norm: bool = False, beam_size: int = 16):
        super().__init__()
        self.m           = base_model
        self.gate_phase  = gate_phase
        self.active_norm = active_norm
        self.beam_size   = beam_size

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                                       # [B, N, D]
        B, N, D = Z.shape

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)      # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        M         = min(self.beam_size, N)

        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                              # [B, N, D]

            # Identify beam (top-M senders by activation magnitude)
            activity  = Z.norm(dim=-1)                                 # [B, N]
            top_idx   = activity.topk(M, dim=-1).indices               # [B, M]

            # Build a beam mask: 1 for sender neurons, 0 for receivers-only
            beam_mask = torch.zeros(B, N, device=Z.device)            # [B, N]
            beam_mask.scatter_(1, top_idx, 1.0)                        # [B, N]

            # Gate Z_fwd so non-beam neurons don't broadcast
            Z_send = Z_fwd * beam_mask.unsqueeze(-1)                   # [B, N, D]

            # Structural routing: gather from (gated) senders via conn_hh
            Z_struct = Z_send[:, conn_hh, :].sum(dim=2)               # [B, N, D]

            # Phase inhibition — optionally gated to beam-only sources
            if self.gate_phase:
                # Temporarily replace the model's Z for _phase_inhibit with Z_send
                # so only beam neurons contribute inhibitory signal
                Z_inh = self.m._phase_inhibit(Z_send, W_ph_norm, theta_pos)
            else:
                Z_inh = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)

            Z_new = Z_struct + self.m.alpha_turing * Z_inh             # [B, N, D]

            # Normalization
            if self.active_norm:
                # Active-only: normalize only neurons where ||Z|| > theta_scalar
                theta_scalar = self.m.theta.abs()                      # [N]
                active_mask  = (Z.norm(dim=-1) > theta_scalar.unsqueeze(0)).float()
                # [B, N]
                Z_norm = F.normalize(Z_new.clamp(-10, 10), dim=-1)    # [B, N, D]
                Z = Z_norm * active_mask.unsqueeze(-1)
            else:
                Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


N = 1024

# Load REF_BASELINE from step57
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.8008)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.8008
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")

# (key, label, kind, kwargs)
CONFIGS = [
    ("Ref", "Ref   AntiHebb alpha=1.0 wpos",
     "antihebb", {}),
    ("A",   "A     BeamUnified gate_struct_only beam=16",
     "beam", dict(gate_phase=False, active_norm=False, beam_size=16)),
    ("B",   "B     BeamUnified gate_struct+phase beam=16",
     "beam", dict(gate_phase=True,  active_norm=False, beam_size=16)),
    ("C",   "C     BeamUnified gate_both + active_norm beam=16",
     "beam", dict(gate_phase=True,  active_norm=True,  beam_size=16)),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 59: Active beam unified routing  |  REF_BASELINE={REF_BASELINE:.4f} (step57)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, kind, kwargs in CONFIGS:
        resonant = make_resonant(N=N, D=64, K_iter=8).to(DEVICE)
        if kind == "antihebb":
            model = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE)
            meta  = {"N": N, "D": 64, "K_iter": 8, "mechanism": "antihebb",
                     "alpha_ahebb": 1.0, "data_frac": 0.5}
        else:
            model = SGNNET_BeamUnified(resonant, **kwargs).to(DEVICE)
            meta  = {"N": N, "D": 64, "K_iter": 8, "mechanism": "beam_unified",
                     "data_frac": 0.5, **kwargs}

        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step59_active_beam_unified.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 59: Active beam unified routing (ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A vs Ref  → does restricting senders sharpen the routing signal?")
    print("  B vs A    → does gating phase channel too help or hurt?")
    print("  C vs B    → does active-only normalization add on top of full beam gating?")
