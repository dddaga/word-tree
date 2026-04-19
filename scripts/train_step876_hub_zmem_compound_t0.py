"""Step 876: Hub + Z-memory compound T0 — orthogonality test.

MOTIVATION
==========
Two mechanisms advancing to T2 independently:
  Hub (step872 T1):  alpha=0.05 → +0.18pp VIABLE
  Z-mem (step868 T0): gamma=0.8 → +0.33pp ADVANCES

Compounding rule: safe only if mechanisms are ORTHOGONAL.
  Hub:   adds global mean signal AFTER Z_fwd threshold
  Z-mem: injects temporal EMA memory INTO Z_route (before gather)
  → Different insertion points: Z-mem is pre-gather, Hub is post-gather

HYPOTHESIS: Hub provides global context; Z-mem provides temporal context.
Both act on routing at different stages → likely orthogonal.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref_dw    : standard ΔW-proj (control)
  A_hub005  : alpha_hub=0.05 only
  B_zmem_g08: gamma=0.8 only
  C_compound: alpha_hub=0.05 + gamma=0.8 combined

SUCCESS: C_compound ≥ max(A, B) + 0.1pp → genuine orthogonal gain
NEUTRAL: C_compound ≈ max(A, B) → additive at best
CANCEL:  C_compound < min(A, B) → mechanisms interact negatively
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
parser.add_argument("--configs", default="Ref_dw,A_hub005,B_zmem_g08,C_compound")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step876_hub_zmem_compound_t0_seed{SEED}__{SLOT}.json"

STEP868_DW_T0 = 0.9396   # Ref T0 reference


class SGNNET_HubZMem(nn.Module):
    """ΔW-proj with optional Hub (global mean) and Z-memory (temporal EMA).

    Z-mem injects before gather (pre-routing memory).
    Hub injects after gather (post-routing global context).
    """
    def __init__(self, resonant, alpha_hub: float = 0.0, gamma: float = 0.0):
        super().__init__()
        self.m = resonant
        self.alpha_hub = alpha_hub
        self.gamma = gamma

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"):
            self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        Z_mem = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            # Z-mem: inject temporal memory pre-gather
            Z_route = Z_fwd + self.gamma * Z_mem
            Z_mem = self.gamma * Z_mem + (1.0 - self.gamma) * Z_fwd
            # ΔW-proj gather
            Z_nb = Z_route[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            # Hub: inject global mean post-gather
            hub_signal = (self.alpha_hub * Z_fwd.mean(dim=1, keepdim=True).expand_as(Z_fwd)
                          if self.alpha_hub > 0 else 0.0)
            Z = F.normalize(
                (Z_nb.sum(2) + Z_ref + hub_signal).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    (0.0, 0.0, "ΔW-proj only (control)"),
    "A_hub005":  (0.05, 0.0, "hub alpha=0.05 only"),
    "B_zmem_g08":(0.0, 0.8, "Z-mem gamma=0.8 only"),
    "C_compound":(0.05, 0.8, "hub 0.05 + Z-mem 0.8 combined"),
}


def make_model(alpha_hub: float, gamma: float) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_HubZMem(resonant, alpha_hub=alpha_hub, gamma=gamma)


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=False)
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False,
    )

    print(f"\n{'='*70}")
    print(f"step876 — Hub + Z-mem compound T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Ref T0 baseline (step868): {STEP868_DW_T0:.4f}")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        alpha_hub, gamma, desc = CONFIGS[key]
        model = make_model(alpha_hub, gamma)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 10 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP868_DW_T0)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "alpha_hub": alpha_hub, "gamma": gamma, "label": desc, "n_params": n_p,
            "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 876 SUMMARY — Hub + Z-mem compound T0")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'hub':>6} {'gmm':>5} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<14} {r['alpha_hub']:>6.3f} {r['gamma']:>5.1f} {r['best']:>7.4f} {dv}")
    cmp = results.get("C_compound", {})
    a = results.get("A_hub005", {})
    b = results.get("B_zmem_g08", {})
    if cmp and a and b:
        best_solo = max(a.get("best", 0), b.get("best", 0))
        gap = cmp["best"] - best_solo
        if gap >= 0.001:
            verdict = f"SYNERGY ({gap*100:+.2f}pp over best solo) — compound advances"
        elif gap >= -0.005:
            verdict = f"ADDITIVE ({gap*100:+.2f}pp vs best solo) — acceptable"
        else:
            verdict = f"CANCEL ({gap*100:+.2f}pp vs best solo) — mechanisms conflict"
        print(f"\n  Compound vs best-solo: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
