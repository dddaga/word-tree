"""Step 858: D_very + ΔW-proj compound T1 — additive or cancels?

MOTIVATION
==========
step853 T1: sparsity=0.98 (D_very) = +0.36pp vs Ref
step760 T1: ΔW-proj = +1.49pp vs Ref (step199 baseline mean)

These mechanisms are on orthogonal signal paths:
  - D_very: readout head (C_ho_mask density, hidden→class)
  - ΔW-proj: message-passing (hidden→hidden routing)

Compounding orthogonal mechanisms is safe per project compounding rule.
Expected compound if additive: ~+1.85pp. If sub-additive, ΔW dominates.
A cancellation would be surprising and informative.

CONFIGS (T1: 75ep, 50% data, seed=42)
  Ref         : sparsity=0.90 + standard AH (step199 baseline)
  A_dvery     : sparsity=0.98 only (control for D_very alone)
  B_dwproj    : sparsity=0.90 + ΔW-proj (control for ΔW alone)
  C_compound  : sparsity=0.98 + ΔW-proj (full compound)
"""
# CUDA-5060ti-validated
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_dvery,B_dwproj,C_compound")
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
OUT_PATH = ROOT / "results" / f"train_step858_dvery_dwproj_compound_seed{SEED}__{SLOT}.json"

STEP199_T1 = 0.9401  # approximate step199 T1 reference
STEP853_DVERY_T1 = 0.9427   # +0.36pp
STEP760_DWPROJ_T1 = 0.9537  # +1.49pp (mean over 5 seeds)


# ΔW-proj mechanism (from step706/step854)
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
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        W_h = self.m.W_pos[:self.m.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        Z_ref = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref":        (0.90, False, "sparsity=0.90 + standard AH"),
    "A_dvery":    (0.98, False, "sparsity=0.98 only"),
    "B_dwproj":   (0.90, True,  "sparsity=0.90 + ΔW-proj"),
    "C_compound": (0.98, True,  "sparsity=0.98 + ΔW-proj (full compound)"),
}


def make_model(sparsity: float, use_dwproj: bool) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=sparsity,
    )
    if use_dwproj:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_DeltaAH(resonant, mode="proj")
    else:
        if DEVICE.type == "cuda":
            resonant = SGNNET_Resonant_CUDA(
                base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                resonance_threshold=0.0, compile=True)
            return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                           variant="wpos", compile=True)
        else:
            resonant = SGNNET_Resonant(
                base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


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
    print(f"step858 — D_very + ΔW-proj compound T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Refs: D_very T1=+0.36pp, ΔW-proj T1=+1.49pp. Compound expected ~+1.85pp")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        sparsity, use_dwproj, desc = CONFIGS[key]
        model = make_model(sparsity, use_dwproj)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False  # use_amp=False, non_blocking=True in Trainer .to(device) calls
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 15 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or STEP199_T1)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "sparsity": sparsity, "use_dwproj": use_dwproj, "label": desc,
            "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(best - (ref_acc or STEP199_T1), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 858 SUMMARY — D_very + ΔW compound T1")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<12} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")
    ref_d = results.get("A_dvery", {}).get("delta_vs_ref", 0.0)
    ref_dw = results.get("B_dwproj", {}).get("delta_vs_ref", 0.0)
    comp = results.get("C_compound", {}).get("delta_vs_ref", 0.0)
    if ref_d and ref_dw and comp:
        additive = (ref_d + ref_dw)
        print(f"\n  Additive prediction: {additive*100:+.2f}pp  Actual compound: {comp*100:+.2f}pp")
        print(f"  Interaction: {(comp - additive)*100:+.2f}pp ({'synergistic' if comp > additive else 'sub-additive'})")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
