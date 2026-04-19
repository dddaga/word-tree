"""Step 871: D=12 + ΔW-proj T1 — can we reduce params 24% at same accuracy?

MOTIVATION
==========
step863 T0 (D=8/D=12 probe):
  Ref_D16: 91.75% (34,976 params)
  C_D12  : 91.46% (26,744 params) → only -0.28pp at T0, promising

D=12 uses 24% fewer params (26,744 vs 34,976). If ΔW-proj compensates
the -0.28pp T0 gap at T1, D=12+ΔW could match D=16+ΔW — a strict Pareto win
(same accuracy, 24% fewer params, 24% fewer FLOPs in routing).

Paper claim: "SGNNET achieves paper-level accuracy at 26,744 params — 4.3× smaller
than the minimum competitive MLP at same accuracy."

CONFIGS (T1: 75ep, 50% data, seed=42, studio_cpu)
  Ref_dw     : D=16, sparsity=0.90, ΔW-proj (T1 reference ~95.24% from step858)
  A_d12      : D=12, sparsity=0.90, standard AH (control — does ΔW matter?)
  B_d12_dw   : D=12, sparsity=0.90, ΔW-proj (main hypothesis)
  C_d12_comp : D=12, sparsity=0.98, ΔW-proj (kitchen sink)

SUCCESS: B_d12_dw ≥ Ref_dw - 0.3pp → D=12 viable with 24% param reduction
STRONG:  B_d12_dw ≥ Ref_dw - 0.1pp → D=12 is the new default
KILL:    B_d12_dw < Ref_dw - 0.5pp → D=16 hard floor
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

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref_dw,A_d12,B_d12_dw,C_d12_comp")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step871_d12_dw_t1_seed{SEED}__{SLOT}.json"

STEP858_DW_T1 = 0.9524  # ΔW-proj T1 (step858 B_dwproj, seed42, mini_cpu)


class SGNNET_DeltaAH(nn.Module):
    """ΔW-proj routing — projection onto ΔW direction in W_pos space."""
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_pos(self): return self.m.W_pos
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
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_nb = Z_nb * proj_coeff.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIGS = {
    "Ref_dw":    (16, 0.90, True,  "D=16, sparsity=0.90, ΔW-proj (T1 baseline)"),
    "A_d12":     (12, 0.90, False, "D=12, sparsity=0.90, standard AH (D probe)"),
    "B_d12_dw":  (12, 0.90, True,  "D=12, sparsity=0.90, ΔW-proj (main hypothesis)"),
    "C_d12_comp":(12, 0.98, True,  "D=12, sparsity=0.98, ΔW-proj (kitchen sink)"),
}


def make_model(D: int, sparsity: float, use_dwproj: bool) -> nn.Module:
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
        sparsity=sparsity,
    )
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=False)
        if use_dwproj:
            return SGNNET_DeltaAH(resonant)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=False)
    else:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        if use_dwproj:
            return SGNNET_DeltaAH(resonant)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


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
    print(f"step871 — D=12 + ΔW-proj T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Ref: ΔW-proj T1={STEP858_DW_T1:.4f} (step858, seed42)")
    print(f"  D=12 T0 delta: -0.28pp (step863 C_D12 = 91.46%)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        D_val, sparsity, use_dwproj, desc = CONFIGS[key]
        model = make_model(D_val, sparsity, use_dwproj)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        if DEVICE.type == "cuda":
            kw["use_amp"] = False
        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}",
                                   flush=True) if (m['epoch']+1) % 15 == 0 else None)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref_dw": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else STEP858_DW_T1)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "D": D_val, "sparsity": sparsity, "use_dwproj": use_dwproj,
            "label": desc, "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4) if ref_acc is not None else None,
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 871 SUMMARY — D=12 + ΔW T1")
    print(f"{'='*70}")
    print(f"  {'config':<12} {'D':>3} {'params':>7} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        dv = f"{r['delta_vs_ref']*100:>+9.2f}pp" if r['delta_vs_ref'] is not None else "  (ref)"
        print(f"  {k:<12} {r['D']:>3} {r['n_params']:>7,} {r['best']:>7.4f} {dv}")
    b_dw = results.get("B_d12_dw", {})
    ref_dw = results.get("Ref_dw", {})
    if b_dw and ref_dw:
        gap = b_dw["best"] - ref_dw["best"]
        if gap >= -0.001:
            verdict = "STRONG — D=12 matches D=16 with ΔW: new default candidate"
        elif gap >= -0.003:
            verdict = "VIABLE — D=12 within 0.3pp: advances to T2"
        else:
            verdict = "KILL — D=16 hard floor with ΔW"
        print(f"\n  D=12+ΔW: {b_dw['n_params']:,} params ({b_dw['n_params']/ref_dw['n_params']*100:.0f}% of D=16)")
        print(f"  Verdict: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
