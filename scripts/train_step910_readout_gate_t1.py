"""Step 910: Input-Conditioned Readout Gate T1 (75ep, 50% data).

MOTIVATION
==========
step907 T0 showed TWO advancing configs at readout-level gating:
  E_ro_geo:    +1.12pp T0 — W_pos geometric gate: σ(cosine(W_pos[h], Z_mean[b]))
  C_ro_tau_lrn: +0.54pp T0 — x_sum z-score gate with learnable τ

All prior input-conditioned gating (steps 903/904/906) FAILED because gating
BEFORE K_iter broke collective message-passing (FM10 bottleneck). Readout-level
gating preserves MP integrity and can selectively weight node contributions.

Also including A_ro_tau1 (+0.36pp T0, below threshold) as a tie-breaker: if T1
recovers it, the z-score mechanism is broadly valid across τ values.

CONFIGS (T1, 75ep, 50% data, seed=42)
  Ref_dw          Standard ΔW-proj baseline (no gate).
  A_ro_tau1       x_sum z-score gate, τ=1.0 (fixed).
  C_ro_tau_lrn    x_sum z-score gate, learnable τ.
  E_ro_geo        W_pos cosine gate: σ(W_pos[h] · Z_mean[b]).

ADVANCE RULE
============
  E_ro_geo   ≥+0.5pp → strong paper claim: geometric readout attention.
  C_ro_tau_lrn ≥+0.5pp → secondary paper claim: learnable input gate.
  Both < +0.5pp → T0 artifact; direction CLOSED.
  Winner ≥+0.5pp → T2 (step911, 150ep/100% data).
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",   default="auto")
parser.add_argument("--epochs",   type=int, default=75)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--configs",  default="Ref_dw,A_ro_tau1,C_ro_tau_lrn,E_ro_geo")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step910_readout_gate_t1_seed{SEED}__{SLOT}.json"

STEP_REF_T1 = 0.9391  # step706 Ref_dw T1 baseline (75ep/50%)

CONFIG_SPEC = {
    "Ref_dw":       "ref",
    "A_ro_tau1":    "ro_zscore_1.0",
    "C_ro_tau_lrn": "ro_zscore_lrn",
    "E_ro_geo":     "ro_geo",
}


def _dw_proj(W_pos: torch.Tensor, conn_hh: torch.Tensor) -> torch.Tensor:
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _dw_agg(Z_nb: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def _gate_entropy(gate: torch.Tensor) -> float:
    p = gate.detach().clamp(1e-6, 1 - 1e-6)
    return (-(p * p.log() + (1 - p) * (1 - p).log())).mean().item()


def _zscore_gate(signal: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    mu  = signal.mean(dim=1, keepdim=True)
    std = signal.std(dim=1, keepdim=True) + 1e-6
    return torch.sigmoid(tau * (signal - mu) / std)  # [B, N]


def _run_mp(m: SGNNET_Resonant, Z: torch.Tensor) -> torch.Tensor:
    """Standard ΔW-proj message-passing, unmodified."""
    conn_hh   = m.base.conn_hh
    theta_pos = m.theta.abs().unsqueeze(0).unsqueeze(-1)
    dw        = _dw_proj(m.W_pos, conn_hh)
    Z_ref     = torch.zeros_like(Z)
    for _ in range(K_ITER):
        Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
        Z_nb  = Z_fwd[:, conn_hh, :]
        Z_agg = _dw_agg(Z_nb, dw)
        Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
        Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
    return Z


def _gated_readout(m_base, W_pos: torch.Tensor, Z: torch.Tensor,
                   gate: torch.Tensor) -> torch.Tensor:
    C_ho   = m_base.C_ho_mask.float()                     # [N_hidden, N_out]
    Z_g    = Z * gate.unsqueeze(-1)                       # [B, N_hidden, D]
    A_out  = torch.einsum("bhd,ho->bod", Z_g, C_ho)      # [B, N_out, D]
    W_out  = F.normalize(W_pos[N:], dim=-1)               # [N_out, D]
    return (A_out * W_out.unsqueeze(0)).sum(dim=-1)       # [B, N_out]


class SGNNET_Ref(nn.Module):
    def __init__(self, resonant: SGNNET_Resonant):
        super().__init__(); self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        Z = _run_mp(self.m, Z)
        return self.m.base._readout(Z)


class SGNNET_ReadoutGate(nn.Module):
    """Input-conditioned gate applied at readout (after K_iter steps)."""

    def __init__(self, resonant: SGNNET_Resonant, gate_spec: str):
        super().__init__()
        self.m         = resonant
        self.gate_spec = gate_spec
        self.last_gate_entropy = 0.0

        if gate_spec == "ro_zscore_lrn":
            self.tau = nn.Parameter(torch.tensor(1.0))
        else:
            self.tau = None

    @property
    def W_pos(self): return self.m.W_pos

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.m.base._seed(x)
        Z = _run_mp(self.m, Z)

        spec = self.gate_spec

        if spec.startswith("ro_zscore"):
            x_sum = x[:, self.m.base.conn_in].sum(dim=2)          # [B, N]
            tau   = self.tau if self.tau is not None else float(spec.split("_")[-1])
            gate  = _zscore_gate(x_sum, tau)                       # [B, N]

        elif spec == "ro_geo":
            z_mean = F.normalize(Z.mean(dim=1), dim=-1)            # [B, D]
            W_h    = F.normalize(self.m.W_pos[:N], dim=-1)        # [N, D]
            gate   = torch.sigmoid((W_h @ z_mean.T).T)            # [B, N]

        else:
            raise ValueError(f"Unknown gate_spec: {spec}")

        self.last_gate_entropy = _gate_entropy(gate)
        return _gated_readout(self.m.base, self.m.W_pos, Z, gate)


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
    spec = CONFIG_SPEC[key]
    resonant = make_base()
    if spec == "ref":
        return SGNNET_Ref(resonant)
    return SGNNET_ReadoutGate(resonant, spec)


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
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*70}")
    print(f"step910 — Readout Gate T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N} D={D} K_hh={K_HH} K_iter={K_ITER}")
    print(f"  T0 winners: E_ro_geo=+1.12pp, C_ro_tau_lrn=+0.54pp")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        spec = CONFIG_SPEC[key]
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: spec={spec}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        gate_entropies = []

        def _log(m):
            ge = model.last_gate_entropy if hasattr(model, "last_gate_entropy") else None
            ge_str = f"{ge:.3f}" if ge is not None and ge > 0 else "n/a"
            tau_str = ""
            if hasattr(model, "tau") and model.tau is not None:
                tau_str = f"  tau={model.tau.item():.3f}"
            if ge is not None and ge > 0:
                gate_entropies.append(ge)
            print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}  gH={ge_str}{tau_str}",
                flush=True,
            )

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=_log)

        elapsed = time.time() - t0
        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta    = best - (ref_acc if ref_acc is not None else 0.0)
        mean_ge  = float(np.mean(gate_entropies)) if gate_entropies else None
        ge_str   = f"{mean_ge:.3f}" if mean_ge is not None else "n/a"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  "
              f"gate_H={ge_str}  {elapsed:.0f}s")

        results[key] = {
            "spec": spec, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "gate_entropy_mean": round(mean_ge, 4) if mean_ge is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 910 SUMMARY — Readout Gate T1")
    print(f"{'='*70}")
    print(f"  {'config':<16} {'spec':<18} {'params':>8} {'best':>7} {'Δ_vs_Ref':>10}  "
          f"{'gH':>6}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "(baseline)"
        ge   = r["gate_entropy_mean"]
        ge_str = f"{ge:.3f}" if ge is not None else "  n/a"
        if k == "Ref_dw":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "ADVANCE→T2"
        elif d >= -0.005:
            v = "NEUTRAL"
        elif d >= -0.010:
            v = "MARGINAL"
        else:
            v = "KILL"
        print(f"  {k:<16} {r['spec']:<18} {r['n_params']:>8,} {r['best']:>7.4f} "
              f"{dstr:>10}  {ge_str:>6}  {v}")

    if ref_acc is not None and "E_ro_geo" in results:
        geo_best = results["E_ro_geo"]["best"]
        gap = (geo_best - ref_acc) * 100
        print(f"\n  KEY READ: E_ro_geo vs Ref gap = {gap:+.2f}pp")
        if gap >= 0.5:
            print(f"  → Readout geo-gate CONFIRMED. Advance to T2 (step911: 150ep/100%).")
            print(f"  → Paper claim: geometric readout attention improves accuracy at no extra params.")
        elif gap >= -0.5:
            print(f"  → T0 artifact. Readout gate direction CLOSES at T1.")
        else:
            print(f"  → Direction CLOSED. Paper note: gating at readout also fails.")

    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
