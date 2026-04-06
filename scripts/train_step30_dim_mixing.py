"""Step 30: Cross-dimensional mixing in SGNNET routing.

PROBLEM
=======
Current routing is:
    Z_new = Σ_j Z_j   (vector sum — dimension-independent)
    Z_h   = normalize(Z_new)

Output dim d = Σ_j Z_j[d].  Dim 0 only ever receives from dim 0 across neighbors.
The l2_norm rescales by a scalar — not cross-dim mixing.

Contrast with FFN: y = Wx mixes ALL input dims into EACH output dim.

QUESTION
========
Does adding a learned D×D linear map after aggregation (before normalize) help?

    Z_new_mixed = W_mix @ Z_new   (or W_mix[n] @ Z_new[n] per-neuron)
    Z_h         = normalize(Z_new_mixed)

Configs
=======
Ref   : baseline  D=16  N=512  K_iter=3          [step9 ref ~29.22%]
A     : + shared W_mix [D,D] per-step (before norm)    [D²=256 params]
B     : + shared W_mix [D,D] per-step (after norm)     [rotation on sphere]
C     : + per-neuron W_mix [N,D,D] per-step            [N×D²=131k params]
D     : + shared W_mix [D,D] once after all K_iter     [applied once only]
E     : D=64 N=1024 K_iter=3 + shared W_mix per-step  [test at current scale]

All D=16 configs: 150ep.  Config E (D=64) also 150ep.
"""

import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.trainer            import Trainer
from src.training.dataset            import make_loaders
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.model_smallworld     import SGNNET_SmallWorld


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
DEVICE = parser.parse_args().device

EPOCHS = 150
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"

_loaders = None
def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders

# ── helpers ───────────────────────────────────────────────────────────────────
def make_resonant(N: int = 512, D: int = 16, K_iter: int = 3) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


# ── Cross-dimensional mixing wrappers ─────────────────────────────────────────

class SGNNET_DimMixShared(nn.Module):
    """Shared D×D matrix applied to the aggregated state after each routing step."""
    def __init__(self, base: SGNNET_Resonant, after_norm: bool = False):
        super().__init__()
        self.m          = base
        self.after_norm = after_norm   # if True: mix after l2_norm (rotation on sphere)
        D               = base.base.D
        # Identity init → pure no-op at step 0
        self.W_mix = nn.Parameter(torch.eye(D))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)              # [B, N, D]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_rem    = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_rem
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh

            if not self.after_norm:
                # Mix before normalize — W_mix acts on raw aggregated vector
                Z_new = torch.einsum('dc, bnc -> bnd', self.W_mix, Z_new)
                Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            else:
                # Mix after normalize — W_mix rotates unit vector
                Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
                Z = F.normalize(
                    torch.einsum('dc, bnc -> bnd', self.W_mix, Z), dim=-1
                )

        return self.m.base._readout(Z)


class SGNNET_DimMixPerNeuron(nn.Module):
    """Per-neuron D×D matrix — each neuron transforms its aggregated state differently."""
    def __init__(self, base: SGNNET_Resonant):
        super().__init__()
        self.m = base
        D = base.base.D
        N = base.base.N_hidden
        # Identity init for every neuron
        self.W_mix = nn.Parameter(
            torch.eye(D).unsqueeze(0).expand(N, -1, -1).clone()  # [N, D, D]
        )

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_rem    = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_rem
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh

            # Per-neuron mixing: W_mix[n] @ Z_new[b,n,:]
            Z_new = torch.einsum('ndc, bnc -> bnd', self.W_mix, Z_new)
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_DimMixOnce(nn.Module):
    """Shared D×D applied ONCE after all K_iter routing steps — not per-step."""
    def __init__(self, base: SGNNET_Resonant):
        super().__init__()
        self.m = base
        D = base.base.D
        self.W_mix = nn.Parameter(torch.eye(D))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)
        conn_hh   = self.m.base.conn_hh
        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.m.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_struct = Z_fwd[:, conn_hh, :].sum(2)
            Z_rem    = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_rem
            Z_inh    = self.m._phase_inhibit(Z, W_ph_norm, theta_pos)
            Z_new    = Z_struct + Z_reflected + self.m.alpha_turing * Z_inh
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # Single application of W_mix after all routing
        Z = F.normalize(
            torch.einsum('dc, bnc -> bnd', self.W_mix, Z).clamp(-10, 10), dim=-1
        )
        return self.m.base._readout(Z)


# ── Run helper ────────────────────────────────────────────────────────────────
def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va   = get_loaders()
    tk       = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer  = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0       = time.time()
    history  = trainer.train(n_epochs=EPOCHS)
    elapsed  = time.time() - t0
    best     = max(h["val_top1"] for h in history)
    best_ep  = max(range(len(history)), key=lambda i: history[i]["val_top1"]) + 1
    frac     = best_ep / EPOCHS
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{EPOCHS} ({frac:.0%})"
          f"  t={elapsed:.0f}s")
    return {
        "label": label, "top1_best": best, "best_ep": best_ep,
        "ep_frac": frac, "t": elapsed,
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
        **meta,
    }


# ── Configs ───────────────────────────────────────────────────────────────────
# (label, N, D, K_iter, model_factory)
CONFIGS = [
    ("Ref. baseline  D=16  N=512  K_iter=3  [step9 ref]",
        512,  16, 3, lambda r: r),
    ("A.  shared W_mix [D×D] per-step  before-norm  [D²=256 params]",
        512,  16, 3, lambda r: SGNNET_DimMixShared(r, after_norm=False)),
    ("B.  shared W_mix [D×D] per-step  after-norm   [rotation on sphere]",
        512,  16, 3, lambda r: SGNNET_DimMixShared(r, after_norm=True)),
    ("C.  per-neuron W_mix [N,D,D] per-step  [N×D²=131k params]",
        512,  16, 3, lambda r: SGNNET_DimMixPerNeuron(r)),
    ("D.  shared W_mix [D×D] once after all K_iter  [single application]",
        512,  16, 3, lambda r: SGNNET_DimMixOnce(r)),
    ("E.  D=64  N=1024  K_iter=3  + shared W_mix per-step  [at scale]",
        1024, 64, 3, lambda r: SGNNET_DimMixShared(r, after_norm=False)),
]

keys = ["Ref", "A", "B", "C", "D", "E"]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}")
    print("Goal: does learned D×D cross-dimensional mixing help routing?")
    print("Ref ~29.22% (D=16).  Config E at D=64 (baseline ~56.28% without signed).")

    results = {}
    ref_top1 = None

    for key, (label, N, D, K_iter, factory) in zip(keys, CONFIGS):
        resonant = make_resonant(N=N, D=D, K_iter=K_iter).to(DEVICE)
        model    = factory(resonant).to(DEVICE)
        meta     = {"N": N, "D": D, "K_iter": K_iter,
                    "mechanism": "dim_mix"}
        results[key] = run(label, model, meta)
        if key == "Ref":
            ref_top1 = results[key]["top1_best"]

    # ── Summary ───────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "train_step30_dim_mixing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    ref16 = results["Ref"]["top1_best"]
    ref64 = results["E"]["top1_best"]   # baseline at D=64 with W_mix

    print(f"\n-- Cross-dimensional mixing  (ref_D16={ref16:.4f}) ---------------------")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_ref16':>8}  {'ep_frac':>8}  {'t(s)':>6}")
    print("  " + "-"*85)
    for key, (label, N, D, K_iter, _) in zip(keys, CONFIGS):
        r   = results[key]
        ref = ref16 if D == 16 else None
        vs  = f"{r['top1_best'] - ref:+.4f}" if ref is not None else "   N/A"
        print(f"  {label:<55}  {r['top1_best']:.4f}  {vs:>8}  "
              f"{r['ep_frac']:>7.1%}  {r['t']:>6.0f}")

    print(f"\n  Config E (D=64) baseline: {ref64:.4f}  "
          f"[compare to step22 D=64 no-mix = 0.5628]")

    print("\n  Interpretation guide:")
    print("  If A > Ref: cross-dim mixing helps (before-norm is the key)")
    print("  If B > A:   rotation on sphere is sufficient (scale doesn't matter)")
    print("  If C > A:   per-neuron specialization adds value over shared")
    print("  If D > A:   mixing once post-routing is enough (not per-step)")
    print("  If E > step22(0.5628): mixing helps even at D=64 without signed coupling")
