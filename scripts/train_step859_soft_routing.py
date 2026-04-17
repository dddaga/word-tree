"""Step 859: Soft distance-based routing T0 — with ΔW-proj and AH variants.

MOTIVATION
==========
Current routing: hard gather from static conn_hh (topology independent of learned W_pos).
Soft routing replaces hard gather with distance-weighted softmax over the same static
K_hh neighbors. Key properties:
  - Gradient flows to W_pos through distance term → positions shape routing
  - β annealing (soft → sharp) → W_pos develops routing-relevant geometry
  - At β→∞: recovers hard routing (hard-0-1 selection by nearest neighbor)

Tractable at T0: soft weights over static conn_hh (O(B·N·K_hh·D) — same as dense).
Full all-to-all distance (O(B·N²·D)) deferred to T1 if T0 validates the concept.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref            : hard gather (standard dense routing + AH baseline)
  A_soft_β1      : soft weights β=1 fixed, no AH, no ΔW
  B_soft_anneal  : soft weights β: 0.5→3.0 over epochs, no AH, no ΔW
  C_soft_ah      : B_soft_anneal + AntiHebbian (AH on W_pos diversity)
  D_soft_dwproj  : B_soft_anneal + ΔW-proj (projection-weighted routing)

SUCCESS CRITERIA
  A_soft_β1 within -1.0pp of Ref → soft weighting doesn't catastrophically break routing
  B_soft_anneal >= A_soft_β1 → annealing helps
  C_soft_ah >= B_soft_anneal → AH + soft routing compose
  D_soft_dwproj >= B_soft_anneal → ΔW-proj composes with soft routing
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 859: Soft routing with ΔW-proj/AH T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_soft_β1,B_soft_anneal,C_soft_ah,D_soft_dwproj")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step859_soft_routing_seed{SEED}__{SLOT}.json"


class SGNNET_SoftRoute(nn.Module):
    """SGNNET_SmallWorld with distance-weighted soft routing over static conn_hh.

    Routing formula (per K_iter step):
        W_nb = W_pos[conn_hh]              [N, K_hh, D]
        dist[b,n,k] = ||Z[b,n] - W_nb[n,k]||²    [B, N, K_hh]
        w = softmax(-β * dist, dim=2)              [B, N, K_hh]
        Z = (Z[:, conn_hh, :] * w.unsqueeze(-1)).sum(2)  [B, N, D]

    Gradients flow through W_pos via the distance term — W_pos learns to
    position neurons so that neighbors in W_pos space carry correlated signals.

    β is set externally via set_beta(). AH and ΔW-proj are applied by the
    outer wrappers (SGNNET_Resonant / SGNNET_AntiHebbian / SGNNET_DeltaAH).
    """

    def __init__(self, base: SGNNET_SmallWorld, beta: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("_beta", torch.tensor(beta, dtype=torch.float32))

    def set_beta(self, beta: float):
        self._beta.fill_(beta)

    @property
    def W_pos(self):            return self.base.W_pos
    @property
    def W_phase(self):          return self.base.W_phase
    @property
    def conn_hh(self):          return self.base.conn_hh
    @property
    def N_hidden(self):         return self.base.N_hidden
    @property
    def C_ho_mask(self):        return self.base.C_ho_mask
    @property
    def K_iter(self):           return self.base.K_iter

    def _normalise(self, Z):    return self.base._normalise(Z)
    def _readout(self, Z):      return self.base._readout(Z)
    def _seed(self, x):         return self.base._seed(x)

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        conn = self.base.conn_hh                   # [N, K_hh]
        W_nb = self.base.W_pos[:self.base.N_hidden][conn]  # [N, K_hh, D]
        beta = self._beta.item()

        for _ in range(self.base.K_iter):
            Z_nb = Z[:, conn, :]                   # [B, N, K_hh, D]
            # Distance from Z[b,n] to W_pos of its K_hh neighbors
            diff = Z.unsqueeze(2) - W_nb.unsqueeze(0)   # [B, N, K_hh, D]
            dist = diff.pow(2).sum(dim=-1)              # [B, N, K_hh]
            w = F.softmax(-beta * dist, dim=2)          # [B, N, K_hh]
            Z = (Z_nb * w.unsqueeze(-1)).sum(2)         # [B, N, D]
            Z = self._normalise(Z)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._seed(x)
        Z = self._route(Z)
        return self._readout(Z)


class BetaScheduler:
    """Linear β anneal from beta_start to beta_end over total_epochs."""
    def __init__(self, model, beta_start: float, beta_end: float, total_epochs: int):
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total = total_epochs
        self._epoch = 0

    def step(self):
        frac = min(1.0, self._epoch / max(1, self.total - 1))
        beta = self.beta_start + frac * (self.beta_end - self.beta_start)
        # Find SoftRoute module in model hierarchy
        for m in self.model.modules():
            if isinstance(m, SGNNET_SoftRoute):
                m.set_beta(beta)
                break
        self._epoch += 1
        return beta


# ΔW-proj wrapper (from step858)
class SGNNET_DeltaAH(nn.Module):
    def __init__(self, base, mode="proj"):
        super().__init__()
        self.m = base; self.mode = mode
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        # Route through underlying SoftRoute model
        inner = self.m
        # Get to the SoftRoute module
        soft_route = None
        for mod in inner.modules():
            if isinstance(mod, SGNNET_SoftRoute):
                soft_route = mod; break

        if soft_route is None:
            return self.m(x)

        Z = soft_route._seed(x)
        conn_hh = soft_route.base.conn_hh
        W_h = soft_route.base.W_pos[:soft_route.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        theta_pos = self.rotation_temp.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)

        for _ in range(soft_route.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)

        return soft_route._readout(Z)


CONFIGS = {
    "Ref":           ("hard",   None,         False,  "hard gather + AH (standard baseline)"),
    "A_soft_β1":     ("soft",   (1.0, 1.0),   False,  "soft β=1 fixed, no AH, no ΔW"),
    "B_soft_anneal": ("soft",   (0.5, 3.0),   False,  "soft β: 0.5→3.0 anneal, no AH, no ΔW"),
    "C_soft_ah":     ("soft",   (0.5, 3.0),   "ah",   "soft anneal + AntiHebbian"),
    "D_soft_dwproj": ("soft",   (0.5, 3.0),   "dw",   "soft anneal + ΔW-proj"),
}


def make_model(routing: str, beta_range, wrap: str | bool):
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=0.90,
    )

    if routing == "soft":
        beta0 = beta_range[0] if beta_range else 1.0
        core = SGNNET_SoftRoute(base, beta=beta0)
    else:
        core = base  # hard routing, standard

    if wrap == "dw":
        # ΔW-proj without outer AH
        if routing == "soft":
            resonant = SGNNET_Resonant(
                core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            model = SGNNET_DeltaAH(resonant, mode="proj")
        else:
            resonant = SGNNET_Resonant(
                core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            model = SGNNET_DeltaAH(resonant, mode="proj")
    elif wrap == "ah":
        if DEVICE.type == "cuda":
            resonant = SGNNET_Resonant_CUDA(
                core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                resonance_threshold=0.0, compile=False)
            model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                            variant="wpos", compile=False)
        else:
            resonant = SGNNET_Resonant(
                core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    else:
        # Ref or bare soft (no AH, no ΔW) — still wrap in Resonant for compatibility
        if routing == "hard":
            if DEVICE.type == "cuda":
                resonant = SGNNET_Resonant_CUDA(
                    core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                    beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                    resonance_threshold=0.0, compile=False)
                model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                                variant="wpos", compile=False)
            else:
                resonant = SGNNET_Resonant(
                    core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                    beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
                model = SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
        else:
            # Bare soft routing (A_soft_β1, B_soft_anneal) — no AH/ΔW to isolate effect
            model = core

    return model


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(
        n_full, generator=torch.Generator().manual_seed(SEED)
    )[: n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step859 — Soft routing + ΔW-proj/AH T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Soft routing: softmax over static conn_hh neighbors (tractable T0)")
    print(f"  β annealing: 0.5→3.0 for anneal configs (soft→sharper over training)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        routing, beta_range, wrap, desc = CONFIGS[key]
        model = make_model(routing, beta_range, wrap)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        # Build beta scheduler if needed
        scheduler = None
        if routing == "soft" and beta_range and beta_range[0] != beta_range[1]:
            scheduler = BetaScheduler(model, beta_range[0], beta_range[1], EPOCHS)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        # Custom epoch callback for β scheduling
        epoch_results = []

        def log_fn(m):
            ep = m['epoch'] + 1
            if scheduler:
                beta = scheduler.step()
                if ep % 5 == 0:
                    print(f"  ep{ep:3d}  val={m['val_top1']:.4f}  β={beta:.2f}", flush=True)
            elif ep % 5 == 0:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)

        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0.94)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "routing": routing, "beta_range": list(beta_range) if beta_range else None,
            "wrap": str(wrap), "label": desc,
            "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(best - (ref_acc or 0.94), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 859 SUMMARY — Soft routing T0")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<18} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")

    soft_base = results.get("A_soft_β1", {}).get("delta_vs_ref", None)
    if soft_base is not None:
        v = "VIABLE (<-1pp)" if soft_base >= -0.01 else "FAILS (-1pp threshold)"
        print(f"\n  Soft routing viability: {v} ({soft_base*100:+.2f}pp baseline)")

    ah_d = results.get("C_soft_ah", {}).get("delta_vs_ref", None)
    dw_d = results.get("D_soft_dwproj", {}).get("delta_vs_ref", None)
    if ah_d is not None:
        print(f"  AH on soft routing: {ah_d*100:+.2f}pp vs Ref")
    if dw_d is not None:
        print(f"  ΔW-proj on soft routing: {dw_d*100:+.2f}pp vs Ref")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
