"""Step 907: Input-Conditioned Readout Gate T0 (20ep, 50% data).

MOTIVATION
==========
Steps 904 and 906 showed that input-conditioned gating BEFORE or DURING
message-passing consistently hurts accuracy (−0.28pp to −15.92pp). The
fundamental constraint: SGNNET's K_iter message-passing requires all nodes
to participate for the collective representation to form.

The only remaining unexplored location for input-conditioning is AT READOUT
— after all K_iter steps are complete. This preserves message-passing integrity.

MECHANISM
=========
Standard readout: A_out = einsum("bhd,ho->bod", Z, C_ho)  [B, N_out, D]
                  logit = (A_out * W_out_norm).sum(-1)     [B, N_out]
  where C_ho[h, o] ∈ {0,1} is the sparse static class-node mask.

Readout-gated: gate[b,h] = f(x_sum[b,h])                  [B, N_hidden]
               A_out = einsum("bhd,ho->bod", Z*gate, C_ho) [B, N_out, D]

The gate weights each hidden node's contribution to ALL output classes
based on how strongly that node was activated by the input. Nodes with
strong x_sum → contribute more; weak x_sum → contribute less.

HYPOTHESIS
==========
After K_iter steps, nodes with high initial activation (x_sum) encode
input-specific information more reliably → should contribute more to readout.
Unlike gating before MP, this does NOT disrupt the message-passing graph.

CONFIGS (T0, 20ep, 50% data, seed=42)
  Ref_dw          No gate (standard ΔW-proj baseline)
  A_ro_tau1       Readout gate, z-score τ=1.0 on x_sum
  B_ro_tau3       Readout gate, z-score τ=3.0 on x_sum
  C_ro_tau_lrn    Readout gate, learnable τ (nn.Parameter)
  D_ro_topk25     Hard top-25% gate at readout (25% of nodes active per class)
  E_ro_geo        Readout gate via W_pos geometry: σ(cosine(W_pos[h], Z_mean[b]))
  F_ro_norm_gate  Readout gate via Z norm: gate[b,h] = sigmoid(τ·||Z[b,h]||_∞)

ADVANCE RULE
============
  Any config ≥−0.5pp vs Ref_dw → NEUTRAL (safe to gate at readout level).
  Any config ≥+0.5pp → T1 (step910, 75ep/50%).
  All configs < −0.5pp → input-conditioned gating definitively closes; document
    in paper as "SGNNET message-passing graph requires all-node participation."
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
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--data",     default="data/store.h5")
parser.add_argument("--configs",  default="Ref_dw,A_ro_tau1,B_ro_tau3,C_ro_tau_lrn,D_ro_topk25,E_ro_geo,F_ro_norm_gate")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step907_readout_gate_t0_seed{SEED}__{SLOT}.json"

STEP_REF = 0.9396  # step898 Ref_dw T0 (D=16+ΔW, 20ep/50%)

CONFIG_SPEC = {
    "Ref_dw":       "ref",
    "A_ro_tau1":    "ro_zscore_1.0",
    "B_ro_tau3":    "ro_zscore_3.0",
    "C_ro_tau_lrn": "ro_zscore_lrn",
    "D_ro_topk25":  "ro_topk_0.25",
    "E_ro_geo":     "ro_geo",
    "F_ro_norm_gate": "ro_norm",
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


def _zscore_gate(signal: torch.Tensor, tau: float) -> torch.Tensor:
    mu  = signal.mean(dim=1, keepdim=True)
    std = signal.std(dim=1, keepdim=True) + 1e-6
    return torch.sigmoid(tau * (signal - mu) / std)  # [B, N]


def _run_mp(m: SGNNET_Resonant, Z: torch.Tensor) -> torch.Tensor:
    """Standard ΔW-proj message-passing, returns Z after K_ITER steps."""
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
    """Apply per-node gate at readout: A_out = einsum(Z*gate, C_ho).

    gate: [B, N_hidden] in (0, 1).
    """
    C_ho   = m_base.C_ho_mask.float()            # [N_hidden, N_out]
    Z_g    = Z * gate.unsqueeze(-1)              # [B, N_hidden, D]
    A_out  = torch.einsum("bhd,ho->bod", Z_g, C_ho)  # [B, N_out, D]
    W_out  = F.normalize(W_pos[N:], dim=-1)      # [N_out, D]
    return (A_out * W_out.unsqueeze(0)).sum(dim=-1)   # [B, N_out]


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
    """Input-conditioned gate applied at readout (after K_iter message-passing)."""

    def __init__(self, resonant: SGNNET_Resonant, gate_spec: str):
        super().__init__()
        self.m         = resonant
        self.gate_spec = gate_spec
        self.last_gate_entropy = 0.0

        # Learnable tau if needed
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
        Z = _run_mp(self.m, Z)  # message-passing UNMODIFIED

        spec = self.gate_spec

        if spec.startswith("ro_zscore"):
            # Gate from x_sum (original input signal per node)
            x_sum    = x[:, self.m.base.conn_in].sum(dim=2)  # [B, N]
            tau      = self.tau if self.tau is not None else float(spec.split("_")[-1])
            gate     = _zscore_gate(x_sum, tau)               # [B, N]

        elif spec == "ro_topk_0.25":
            x_sum    = x[:, self.m.base.conn_in].sum(dim=2).abs()  # [B, N]
            k        = max(1, int(N * 0.25))
            threshold = x_sum.topk(k, dim=1).values[:, -1:]
            gate     = (x_sum >= threshold).float()            # [B, N] {0,1}

        elif spec == "ro_geo":
            # Gate from cosine alignment of W_pos[h] with mean Z
            z_mean = F.normalize(Z.mean(dim=1), dim=-1)          # [B, D]
            W_h    = F.normalize(self.m.W_pos[:N], dim=-1)       # [N, D]
            gate   = torch.sigmoid((W_h @ z_mean.T).T)           # [B, N]

        elif spec == "ro_norm":
            # Gate from L-inf norm of Z after message-passing
            z_norm   = Z.abs().max(dim=-1).values                 # [B, N]
            gate     = _zscore_gate(z_norm, 2.0)                  # [B, N]

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
    r = make_base()
    spec = CONFIG_SPEC[key]
    if spec == "ref":
        return SGNNET_Ref(r)
    return SGNNET_ReadoutGate(r, gate_spec=spec)


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
    print(f"step907 — Input-Conditioned Readout Gate T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Ref context (step898 Ref_dw T0): {STEP_REF:.4f}")
    print(f"  Gate location: AT READOUT (after K_iter steps, before logits)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: spec={CONFIG_SPEC[key]}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        gate_entropies = []

        def log_fn(m: dict) -> None:
            if hasattr(model, "last_gate_entropy"):
                gate_entropies.append(model.last_gate_entropy)
            ep_str = (f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}  "
                      f"top1={m['val_top1']:.4f}  lr={m['lr']:.2e}")
            if hasattr(model, "last_gate_entropy"):
                ep_str += f"  gH={model.last_gate_entropy:.3f}"
            if hasattr(model, "tau") and model.tau is not None:
                ep_str += f"  tau={model.tau.item():.3f}"
            print(ep_str, flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                 for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw":
            ref_acc = best
        delta  = best - (ref_acc if ref_acc is not None else STEP_REF)
        mean_ge = float(np.mean(gate_entropies)) if gate_entropies else None
        ge_str  = f"{mean_ge:.3f}" if mean_ge is not None else "n/a"
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  "
              f"gate_H={ge_str}  {elapsed:.0f}s")

        results[key] = {
            "spec": CONFIG_SPEC[key], "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "gate_entropy_mean": round(mean_ge, 4) if mean_ge is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 907 SUMMARY — Input-Conditioned Readout Gate T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'spec':<20} {'params':>8} {'best':>7} "
          f"{'Δ_vs_Ref':>10} {'gH':>6}  verdict")
    for k, r in results.items():
        d    = r["delta_vs_ref"]
        dstr = f"{d*100:+.2f}pp" if d is not None else "  (ref)"
        ge   = r["gate_entropy_mean"]
        ge_s = f"{ge:.3f}" if ge is not None else "  n/a"
        if k == "Ref_dw":
            v = "(baseline)"
        elif d is None:
            v = "—"
        elif d >= 0.005:
            v = "ADVANCE→T1"
        elif d >= -0.005:
            v = "NEUTRAL"
        elif d >= -0.020:
            v = "MARGINAL"
        else:
            v = "KILL"
        print(f"  {k:<14} {r['spec']:<20} {r['n_params']:>8,} "
              f"{r['best']:>7.4f} {dstr:>10} {ge_s:>6}  {v}")

    print(f"\n  Ref T0 context (step898/Ref_dw): {STEP_REF}")
    print(f"  ADVANCE: ≥+0.5pp → T1 (step910, 75ep/50%).")
    print(f"  CLOSE: all < −0.5pp → input-conditioned gating direction CLOSED.")
    print(f"    Paper note: 'SGNNET requires all-node participation in MP.'")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
