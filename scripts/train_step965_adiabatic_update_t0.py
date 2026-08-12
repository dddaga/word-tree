"""Step 965: Adiabatic Update T0 — selective gradient sparsity training.

HYPOTHESIS (Physics of Deep Learning — 2026-04-21):
Standard backprop updates all N=2048 nodes simultaneously every batch.
This is non-adiabatic: the network is excited away from its current attractor
before it has time to settle. Analogous to heating a crystal too fast — internal
stress, slow relaxation.

Adiabatic training: update only the highest-gradient nodes per batch.
Rest of network freezes for that step, adapts to the change over subsequent batches.
Prediction: slower parameter change per unit time → more stable optimization →
better generalization at same epoch count, or same generalization at fewer epochs.

FUNDAMENTAL MOTIVATION (finite-precision arithmetic):
  fp32/bf16 is discrete: every gradient step is quantized to the nearest
  representable value. We are NOT doing continuous gradient flow — we are
  taking discrete jumps on a quantized optimization landscape.
  Given that each update is already a finite discrete step, adiabatic training
  asks: after each discrete jump, let the network observe and digest the change
  before the next jump is made. The "breathing space" closes the gap between
  the mathematically intended continuous update and the actual discrete one.
  Nodes that didn't change get to recompute their gradients against the new
  state, so the next update reflects the updated landscape rather than a stale one.

MECHANISM:
  Per batch:
    1. Forward + backward (compute ALL gradients as normal)
    2. Rank nodes by gradient norm: |∇W_pos[i]| for i in 0..N-1
    3. Zero gradients for all nodes EXCEPT top-K% by gradient norm
    4. Optimizer.step() (only top-K nodes actually change)
    5. Zero_grad()

  This is NOT weight freezing — all nodes participate in forward/backward.
  It is selective parameter update — only the most "ready" nodes change.

QUANTUM STEP variant (D_quantum):
  Instead of continuous updates, round ΔW to nearest multiple of Δθ=0.01.
  Nodes only move in discrete steps. Most batches: Δ rounds to zero (no update).
  This is QAT-style quantization of the LEARNING PROCESS, not the weights.
  Makes the inherent fp32 discreteness explicit and controllable.

GRADIENT ACCUMULATION variant (E_accum):
  Accumulate gradients over 4 batches. After 4 batches, apply top-5% selective
  update using the accumulated signal. Reset gradients. 4-batch window gives
  a more stable gradient estimate before committing to a change.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%):
  Ref:       standard Adam, all nodes update every batch
  A_top01:   top 1% of nodes by |∇W_pos| update per batch (≈20 nodes)
  B_top05:   top 5% of nodes update per batch (≈100 nodes)
  C_top20:   top 20% of nodes update per batch (≈400 nodes)
  D_quantum: all nodes, but ΔW rounded to ±0.01 quantum (most steps = no change)
  E_accum4:  accumulate 4 batches, then top-5% update, reset

NOTE: gradient masking is applied ONLY to W_pos (the geometric routing parameter).
All other parameters (W_in, W_out, theta) update normally. W_pos is the routing
geometry — the hypothesis is that routing geometry should change adiabatically.

ADVANCE: ≥+0.5pp vs Ref → T1.
INTERESTING: comparable accuracy but faster convergence (lower ep to plateau) → T1.
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_top01,B_top05,C_top20,D_quantum,E_accum4")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5
QUANTUM_STEP = 0.01
ACCUM_BATCHES = 4

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step965_adiabatic_update_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical

CONFIG_SPEC = {
    "Ref":      {"update_frac": 1.0,  "quantum": False, "accum": 1},
    "A_top01":  {"update_frac": 0.01, "quantum": False, "accum": 1},
    "B_top05":  {"update_frac": 0.05, "quantum": False, "accum": 1},
    "C_top20":  {"update_frac": 0.20, "quantum": False, "accum": 1},
    "D_quantum":{"update_frac": 1.0,  "quantum": True,  "accum": 1},
    "E_accum4": {"update_frac": 0.05, "quantum": False, "accum": 4},
}


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


class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj reference (identical to step887 base)."""
    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)
    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = self.m.W_pos[:N]
        dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def _apply_gradient_mask(model, update_frac):
    """Zero gradients for all W_pos rows EXCEPT top-K by gradient norm."""
    W_pos = model.W_pos  # [N+N_OUT, D] or similar
    if W_pos.grad is None:
        return 0
    # Compute per-row gradient norm for hidden nodes only
    g = W_pos.grad[:N]                                  # [N, D]
    g_norms = g.norm(dim=-1)                            # [N]
    k = max(1, int(N * update_frac))
    _, top_idx = torch.topk(g_norms, k)
    # Mask: zero all rows except top-k
    mask = torch.zeros(N, device=W_pos.device)
    mask[top_idx] = 1.0
    # Apply mask (preserve readout rows W_pos[N:] unchanged)
    W_pos.grad[:N] *= mask.unsqueeze(-1)
    return int(k)


def _apply_quantum_step(model, quantum_step):
    """Round W_pos gradients so that effective ΔW is a multiple of quantum_step.
    Gradients below lr*quantum_step threshold round to zero."""
    W_pos = model.W_pos
    if W_pos.grad is None:
        return
    # Round gradient to nearest quantum (quantizes the update direction)
    W_pos.grad.data = torch.round(W_pos.grad.data / quantum_step) * quantum_step


def train_adiabatic(model, tr, va, spec, n_epochs):
    """Custom training loop with selective gradient update."""
    kw = trainer_kwargs(N, n_epochs=n_epochs)
    lr = kw.get("lr", 1e-3)
    wd = kw.get("weight_decay", 1e-4)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=n_epochs * len(tr), eta_min=lr * 0.01)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    update_frac = spec["update_frac"]
    use_quantum = spec["quantum"]
    accum       = spec["accum"]

    history = []
    accum_step = 0

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        for i, batch in enumerate(tr):
            x, _, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y) / accum
            loss.backward()
            accum_step += 1

            if accum_step % accum == 0:
                # Apply gradient modification before optimizer step
                if update_frac < 1.0:
                    _apply_gradient_mask(model, update_frac)
                if use_quantum:
                    _apply_quantum_step(model, QUANTUM_STEP)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va:
                x, _, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        top1 = correct / total
        history.append({"val_top1": top1, "epoch": ep,
                        "lr": sched.get_last_lr()[0]})
        print(f"  e{ep+1:3d}  top1={top1:.4f}  lr={sched.get_last_lr()[0]:.2e}",
              flush=True)

    return history


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full  = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*72}")
    print(f"step965 — Adiabatic Update T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Gradient masking applied to W_pos only (routing geometry).")
    print(f"  All other params (W_in, W_out, theta) update normally.")
    print(f"  Canonical ref (step887): {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'update_frac':>12}  {'quantum':>8}  {'accum':>6}")
    for k, s in CONFIG_SPEC.items():
        print(f"  {k:<12} {s['update_frac']:>12.0%}  {str(s['quantum']):>8}  {s['accum']:>6}")
    print(f"{'='*72}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        spec  = CONFIG_SPEC[key]
        model = SGNNET_Ref(make_base()).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{'─'*60}")
        print(f"{key}: update_frac={spec['update_frac']:.0%}  "
              f"quantum={spec['quantum']}  accum={spec['accum']}  params={n_p:,}")

        t0      = time.time()
        history = train_adiabatic(model, tr, va, spec, EPOCHS)
        elapsed = time.time() - t0

        top1h   = [h["val_top1"] for h in history]
        best    = max(top1h)
        best_ep = int(np.argmax(top1h)) + 1
        # Early convergence: ep at which 98% of final best is first reached
        threshold98 = best * 0.98
        ep98 = next((i+1 for i, v in enumerate(top1h) if v >= threshold98), EPOCHS)

        if key == "Ref": ref_acc = best
        delta   = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else "INTERESTING" if (delta >= -0.005 and ep98 < (EPOCHS * 0.7))
                   else "NEUTRAL"     if delta >= -0.005
                   else "KILL")
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  "
              f"ep98={ep98}  {verdict}")

        results[key] = {
            "update_frac": spec["update_frac"],
            "quantum":     spec["quantum"],
            "accum":       spec["accum"],
            "n_params":    n_p,
            "best":        round(best, 4),
            "best_ep":     best_ep,
            "ep98":        ep98,
            "delta_vs_ref": round(delta, 4),
            "elapsed_s":   round(elapsed),
            "history_top1": [round(v, 4) for v in top1h],
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 965 SUMMARY — Adiabatic Update T0")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else "ADV→T1"  if d >= 0.005
             else "INTER"   if (d >= -0.005 and r['ep98'] < (EPOCHS * 0.7))
             else "NEU"      if d >= -0.005
             else "KILL")
        print(f"  {k:<12} frac={r['update_frac']:.0%}  q={r['quantum']}  "
              f"acc={r['accum']}  best={r['best']:.4f}  Δ={d*100:+.2f}pp  "
              f"ep98={r['ep98']}  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
