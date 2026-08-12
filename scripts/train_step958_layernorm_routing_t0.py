"""Step 958: LayerNorm / RMSNorm / ScaledL2 routing normalization T0 (20ep, 50% data).

HYPOTHESIS: L2-normalize forces Z onto S^{D-1}, collapsing angular diversity.
PR=2.3 means only ~2.3 effective dimensions out of D=16 are used.
Replacing F.normalize with learnable or magnitude-preserving norms may activate
more dimensions, improving expressivity without losing geometric structure.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:             standard L2-normalize (F.normalize) — same as step887
  A_layernorm:     LayerNorm(D) with elementwise affine (learned scale+bias)
  B_rmsnorm:       RMSNorm — z / (||z|| / sqrt(D) + 1e-6), no learned params
  C_rmsnorm_learned: RMSNorm + per-dim learned scale gamma (ones init)
  D_scaledl2:      L2-normalize + single global learned scalar scale (init=1.0)

CRITICAL: ΔW-proj direction dw = normalize(W_pos[i] - W_pos[j]) still uses
F.normalize — that is a direction vector, not Z normalization. Only the routing
state update normalization changes.

Track PR (Participation Ratio) at each epoch via log_fn.
PR = (sum(ev))^2 / sum(ev^2), measures effective dimensionality of Z.
Baseline PR ≈ 2.3 (D=16).

ADVANCE: ≥+0.5pp vs Ref → T1.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
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
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_layernorm,B_rmsnorm,C_rmsnorm_learned,D_scaledl2")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step958_layernorm_routing_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


@torch.no_grad()
def compute_pr(Z):
    """Participation Ratio of Z: [B, N, D] → scalar.

    PR = (sum ev)^2 / sum(ev^2). Measures effective dimensionality.
    Baseline (L2-normalize) ≈ 2.3.
    NOTE: linalg.eigvalsh is unsupported on MPS — computation forced to CPU.
    """
    z = Z.mean(1).cpu().float()  # [B, D] — move to CPU for eigvalsh compatibility
    zc = z - z.mean(0)           # center
    cov = zc.T @ zc / max(len(zc) - 1, 1)  # [D, D]
    ev = torch.linalg.eigvalsh(cov).abs()
    return (ev.sum() ** 2 / (ev ** 2).sum().add(1e-8)).item()


def _dw_proj_vec(W_pos, conn_hh):
    """Direction vector for ΔW-proj. Uses F.normalize — this is a geometric direction, NOT Z norm."""
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]


# ──────────────────────────────────────────────────────────────────────────────
# Model variants
# ──────────────────────────────────────────────────────────────────────────────

class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj with L2-normalize routing — step887 canonical."""

    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x, return_Z=False):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                      # [B,N,K_hh,D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        if return_Z: return self.m.base._readout(Z), Z
        return self.m.base._readout(Z)


class SGNNET_LayerNorm(nn.Module):
    """Routing normalization replaced with LayerNorm(D).

    LayerNorm centers + scales per-sample. Allows different magnitudes,
    representation occupies R^D rather than S^{D-1}.
    """

    def __init__(self, resonant):
        super().__init__()
        self.m    = resonant
        self.norm = nn.LayerNorm(D)  # elementwise_affine=True by default (learned scale+bias)

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x, return_Z=False):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = self.norm((Z_agg + Z_ref).clamp(-10, 10))
        if return_Z: return self.m.base._readout(Z), Z
        return self.m.base._readout(Z)


class SGNNET_RMSNorm(nn.Module):
    """RMSNorm variant: z / (||z|| / sqrt(D) + eps).

    No learned params. Preserves direction on unit sphere but normalizes
    by RMS rather than L2-norm directly — equivalent to L2 up to sqrt(D) scaling.
    """

    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x, return_Z=False):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            z_raw = (Z_agg + Z_ref).clamp(-10, 10)
            rms   = z_raw.norm(dim=-1, keepdim=True) / math.sqrt(D) + 1e-6
            Z     = z_raw / rms
        if return_Z: return self.m.base._readout(Z), Z
        return self.m.base._readout(Z)


class SGNNET_RMSNormLearned(nn.Module):
    """RMSNorm + per-dim learned scale gamma ∈ R^D (ones init).

    Allows different per-dimension magnitudes while keeping RMS normalization.
    """

    def __init__(self, resonant):
        super().__init__()
        self.m     = resonant
        self.gamma = nn.Parameter(torch.ones(D))  # per-dim learned scale

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x, return_Z=False):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            z_raw = (Z_agg + Z_ref).clamp(-10, 10)
            rms   = z_raw.norm(dim=-1, keepdim=True) / math.sqrt(D) + 1e-6
            Z     = (z_raw / rms) * self.gamma
        if return_Z: return self.m.base._readout(Z), Z
        return self.m.base._readout(Z)


class SGNNET_ScaledL2(nn.Module):
    """L2-normalize + single global learned scalar scale (init=1.0).

    Keeps geometric structure of S^{D-1} but allows learned magnitude.
    Minimal perturbation from Ref — tests whether magnitude alone matters.
    """

    def __init__(self, resonant):
        super().__init__()
        self.m     = resonant
        self.scale = nn.Parameter(torch.tensor(1.0))  # global learned magnitude

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x, return_Z=False):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1) * self.scale
        if return_Z: return self.m.base._readout(Z), Z
        return self.m.base._readout(Z)


# ──────────────────────────────────────────────────────────────────────────────

CONFIG_SPEC = {
    "Ref":               "l2_normalize",
    "A_layernorm":       "layernorm",
    "B_rmsnorm":         "rmsnorm",
    "C_rmsnorm_learned": "rmsnorm_learned",
    "D_scaledl2":        "scaled_l2",
}

CONFIG_EXTRA_PARAMS = {
    "l2_normalize":   0,
    "layernorm":      D * 2,       # scale + bias per dim
    "rmsnorm":        0,
    "rmsnorm_learned": D,          # gamma per dim
    "scaled_l2":      1,           # single scalar
}


def _param_count(key):
    mode = CONFIG_SPEC[key]
    base  = (N + N_OUT) * D + N
    extra = CONFIG_EXTRA_PARAMS[mode]
    return base, extra, base + extra


def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    r = make_base()
    mode = CONFIG_SPEC[key]
    if mode == "l2_normalize":       return SGNNET_Ref(r)
    if mode == "layernorm":          return SGNNET_LayerNorm(r)
    if mode == "rmsnorm":            return SGNNET_RMSNorm(r)
    if mode == "rmsnorm_learned":    return SGNNET_RMSNormLearned(r)
    if mode == "scaled_l2":          return SGNNET_ScaledL2(r)
    raise ValueError(f"Unknown mode: {mode}")


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

    print(f"\n{'='*72}")
    print(f"step958 — LayerNorm/RMSNorm/ScaledL2 routing normalization T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Hypothesis: L2-norm constrains Z to S^{{D-1}}, PR=2.3 (low diversity).")
    print(f"  LayerNorm/RMSNorm allow R^D, potentially activating more dimensions.")
    print(f"  Tracking PR (Participation Ratio) per epoch — baseline ≈ 2.3.")
    print(f"  NOTE: dw = F.normalize(W_pos[i]-W_pos[j]) unchanged — geometric direction only.")
    print(f"  step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<20} {'mode':<16} {'extra_params':>12}  {'total':>8}")
    for k in CONFIG_SPEC:
        b, ex, tot = _param_count(k)
        print(f"  {k:<20} {CONFIG_SPEC[k]:<16} {ex:>12,}  {tot:>8,}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _, _, exp_total = _param_count(key)
        print(f"{'─'*60}")
        print(f"{key}: mode={CONFIG_SPEC[key]}  params={n_p:,}  (expected={exp_total:,})")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        pr_log = {}

        def log_fn(m, _model=model, _pr_log=pr_log):
            ep = m["epoch"]
            # Compute PR on a single val batch
            pr_val = float("nan")
            try:
                with torch.no_grad():
                    batch = next(iter(va))
                    xb = batch[0].to(DEVICE)
                    if hasattr(_model, "forward") and callable(_model.forward):
                        out = _model.forward(xb, return_Z=True)
                        if isinstance(out, tuple):
                            _, Z_sample = out
                            pr_val = compute_pr(Z_sample)
            except Exception:
                pass
            _pr_log[ep + 1] = round(pr_val, 4) if not math.isnan(pr_val) else None
            pr_str = f"{pr_val:.2f}" if not math.isnan(pr_val) else "n/a"
            print(f"  e{ep+1:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}  PR={pr_str}",
                  flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        # PR summary
        pr_vals = [v for v in pr_log.values() if v is not None]
        pr_first = pr_vals[0] if pr_vals else None
        pr_last  = pr_vals[-1] if pr_vals else None
        if pr_first is not None:
            print(f"  -> PR: ep1={pr_first:.2f}  ep{EPOCHS}={pr_last:.2f}")

        results[key] = {
            "mode": CONFIG_SPEC[key], "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
            "pr_log": pr_log,
            "pr_ep1": pr_first, "pr_final": pr_last,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 958 SUMMARY — LayerNorm routing normalization T0")
    print(f"{'='*72}")
    print(f"  {'Config':<20} {'mode':<16} {'params':>8}  {'best':>6}  {'Δ':>7}  {'PR_ep1':>7}  {'verdict'}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        pr1 = f"{r['pr_ep1']:.2f}" if r.get("pr_ep1") is not None else "n/a"
        print(f"  {k:<20} {r['mode']:<16} {r['n_params']:>8,}  "
              f"{r['best']:.4f}  {d*100:+.2f}pp  {pr1:>7}  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
