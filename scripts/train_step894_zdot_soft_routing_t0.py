"""Step 894: Z-dot + AH softmax routing revival — T0 scout on current arch.

MOTIVATION
==========
step73 Config D (OLD arch: N=1024, D=64, K_iter=8) found +1.78pp with
softmax(Z_h·Z_j/τ=0.3 + AH_logit). step75 D added input-modulated temperature
for +3.98pp. Both use Z-state dot product + AH logit as routing score.

CRITICAL GAP: this mechanism was NEVER tested on current arch (N=2048, D=16,
K_hh=2, K_iter=5). step859 (KILLED) used W_pos distance — a static proxy —
NOT dynamic Z-dot scoring. Fundamentally different mechanism.

Risk at D=16: Z-dot noise std=1/√16=0.25 (vs 0.125 at D=64). AH logit may
compensate by providing structural anchor. Must test empirically.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw        : standard ΔW-proj baseline (~93.96%)
  A_zdot_only   : softmax(Z_h·Z_j/τ=1.0) over K_hh. No AH. Predicted: collapse.
  B_ah_softmax  : softmax(AH_logit_j/τ=1.0) over K_hh. Static AH redistribution.
  C_zdot_ah_t10 : softmax(Z_h·Z_j/1.0 + AH_logit_j). step73 D equivalent.
  D_zdot_ah_t03 : softmax(Z_h·Z_j/0.3 + AH_logit_j). Sharper temperature (step73 winner).

SUCCESS: C or D ≥+0.5pp over Ref_dw → advance to T1 (step897)
FAILURE: A collapses AND C/D neutral → D=16 noise makes Z-dot routing unviable
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_zdot_only,B_ah_softmax,C_zdot_ah_t10,D_zdot_ah_t03")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step894_zdot_soft_routing_t0_seed{SEED}__{SLOT}.json"

# ΔW-proj baseline from step883/886 T0 context
STEP_REF = 0.9396


class SGNNET_SoftRouting(nn.Module):
    """Softmax-weighted routing over K_hh neighbors.

    score(h, j) = score_fn(Z_h, Z_j, AH_logit_j) / tau
    weight(h, j) = softmax(score, dim=K_hh)
    Z_new[h] = sum_j weight(h,j) * Z_j + reflect

    AH_logit: anti-Hebbian suppression logit — suppresses neighbors with
    similar activation history. Provides structural stability as anchor.
    Static at inference (from SGNNET_Resonant phase graph).
    """

    def __init__(self, resonant: SGNNET_Resonant, mode: str, tau: float = 1.0):
        super().__init__()
        self.m    = resonant
        self.mode = mode
        self.tau  = tau

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def _get_ah_logit(self, Z: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
        """Compute AH suppression logit from phase graph.

        Returns scalar logit per (neuron, neighbor) pair: [B, N, K_hh].
        Uses SGNNET_Resonant phase-based AH if available, else position cosine.
        """
        # Use W_pos cosine similarity as structural AH proxy (static, matches step859 spirit
        # but combined here with dynamic Z-dot — different composite than step859).
        W_h = self.m.W_pos[:self.m.base.N_hidden]            # [N, D]
        W_nb = W_h[conn_hh]                                   # [N, K_hh, D]
        W_h_n = F.normalize(W_h, dim=-1)
        W_nb_n = F.normalize(W_nb, dim=-1)
        # Higher cosine = similar position = higher AH suppression
        logit = (W_h_n.unsqueeze(1) * W_nb_n).sum(-1)        # [N, K_hh]
        return logit.unsqueeze(0)                              # [1, N, K_hh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)                              # [B, N, D]
        conn_hh = self.m.base.conn_hh                         # [N, K_hh]
        B = Z.shape[0]

        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        ah_logit  = self._get_ah_logit(Z, conn_hh)            # [1, N, K_hh]

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                      # [B, N, D]
            Z_nb  = Z_fwd[:, conn_hh, :]                       # [B, N, K_hh, D]

            if self.mode == "A_zdot_only":
                # Pure Z-dot: score = Z_h · Z_j / tau
                Z_h_exp = Z_fwd.unsqueeze(2)                   # [B, N, 1, D]
                score = (Z_h_exp * Z_nb).sum(-1) / self.tau    # [B, N, K_hh]

            elif self.mode == "B_ah_softmax":
                # AH-only: score = AH_logit (static, no Z-dot)
                score = -ah_logit.expand(B, -1, -1) / self.tau  # negative: suppress similar

            elif self.mode in ("C_zdot_ah_t10", "D_zdot_ah_t03"):
                # Z-dot + AH logit composite (step73 Config D equivalent)
                Z_h_exp = Z_fwd.unsqueeze(2)
                zdot = (Z_h_exp * Z_nb).sum(-1) / self.tau     # [B, N, K_hh]
                ah   = -ah_logit.expand(B, -1, -1)             # suppress similar neighbors
                score = zdot + ah

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            weight = torch.softmax(score, dim=-1).unsqueeze(-1)  # [B, N, K_hh, 1]
            Z_agg  = (weight * Z_nb).sum(dim=2)                  # [B, N, D]

            Z_ref  = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z      = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


class SGNNET_DeltaW_Ref(nn.Module):
    """Standard ΔW-proj reference (copied from step886 for consistent Ref_dw)."""

    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb  = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":       ("ref",         1.0,  "ΔW-proj baseline"),
    "A_zdot_only":  ("A_zdot_only", 1.0,  "softmax(Z_h·Z_j/τ=1.0) — no AH"),
    "B_ah_softmax": ("B_ah_softmax",1.0,  "softmax(−AH_logit/τ=1.0) — static AH only"),
    "C_zdot_ah_t10":("C_zdot_ah_t10",1.0,"softmax(Z_h·Z_j/1.0 − AH_logit)"),
    "D_zdot_ah_t03":("D_zdot_ah_t03",0.3,"softmax(Z_h·Z_j/0.3 − AH_logit) — sharp τ"),
}


def make_base() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    ng  = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key: str) -> nn.Module:
    mode, tau, _ = CONFIGS[key]
    resonant = make_base()
    if key == "Ref_dw":
        return SGNNET_DeltaW_Ref(resonant)
    return SGNNET_SoftRouting(resonant, mode=mode, tau=tau)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step894 — Z-dot + AH softmax routing revival T0 scout")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Old-arch step73 Config D: +1.78pp (N=1024/D=64/K_iter=8)")
    print(f"  Current arch: N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        mode, tau, desc = CONFIGS[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  τ={tau}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 5 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP_REF)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "mode": mode, "tau": tau, "label": desc, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 894 SUMMARY — Z-dot + AH softmax routing T0")
    print(f"{'='*70}")
    print(f"  {'config':<16} {'tau':>5} {'best':>7} {'Δ_vs_Ref':>10}  verdict")
    for k, r in results.items():
        d  = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        v = "ADVANCE→T1" if (d is not None and d >= 0.005) else \
            "NEUTRAL" if (d is not None and d >= -0.005) else \
            "KILL" if (d is not None and d < -0.005) else "(ref)"
        print(f"  {k:<16} {r['tau']:>5.1f} {r['best']:>7.4f} {dstr:>10}  {v}")
    print(f"\n  Ref T0 context (step883): {STEP_REF:.4f}")
    print(f"  ADVANCE criterion: C or D ≥+0.5pp over Ref_dw → T1 (step897)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
