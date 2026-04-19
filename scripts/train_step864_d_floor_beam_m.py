"""Step 864: D floor + beam M ablation T0.

MOTIVATION
==========
step851 MLP crossover (Imagenette / VGG16 features):
  MLP_4 (h=4): 95.01%  — fails vs SGNNET 95.52%  (Δ = -0.51pp)
  MLP_6 (h=6): 96.31%  — beats SGNNET            (Δ = +0.79pp)
  Crossover: h ≈ 5–6 → intrinsic rank of task ≈ 5–6 dimensions.

Implication for D:
  D is the ambient manifold of W_pos. If task intrinsic rank = 5-6,
  D=8 (step863) tests 4 complex Fourier components; D=6 tests 3; D=4 tests 2.
  D floor experiment: how low can D go before routing diversity collapses?

Implication for beam M:
  Beam M = broadcasters per iter in BFS routing (step855) or beams in Resonant.
  If intrinsic_dim ≈ 6, optimal M ≈ 2-3× = 12-18 → current M=16 may already be
  near-optimal. This ablation checks M ∈ {4, 8, 16, 32} at D=16 baseline.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref          : D=16 K=5 (control)
  A_D6         : D=6  K=5  (~13K params total, 2.5× smaller W_pos)
  B_D4         : D=4  K=5  (~9K params total, 4× smaller W_pos)
  C_D4_K10     : D=4 + K_iter=10 (more iterations to compensate lower capacity)
  D_beam_M8    : D=16, beam_size=8  in Resonant (half default)
  E_beam_M32   : D=16, beam_size=32 in Resonant (double default)

PARAM ESTIMATES
  D=16 (Ref): W_pos = [2058, 16] = 32,928 → total ≈ 34,976
  D=12:       W_pos = [2058, 12] = 24,696 → total ≈ 26,744  (step863)
  D=8:        W_pos = [2058,  8] = 16,464 → total ≈ 18,512  (step863)
  D=6:        W_pos = [2058,  6] = 12,348 → total ≈ 14,396
  D=4:        W_pos = [2058,  4] =  8,232 → total ≈ 10,280

SUCCESS CRITERIA
  B_D4 within -1pp of Ref → 10K-param SGNNET headline viable
  A_D6 within -0.5pp of Ref → D=6 default candidate
  D_beam_M8 within -0.5pp → M=8 viable (2× routing reduction)
  E_beam_M32 >= Ref → bigger beam helps (suggests current M=16 is bottleneck)
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

parser = argparse.ArgumentParser(description="Step 864: D floor + beam M ablation T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_D6,B_D4,C_D4_K10,D_Rbeam_M8,E_Rbeam_M32,H_bfs_M16,F_bfs_M32,G_bfs_M64")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
K_HH = 2; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step864_d_floor_beam_m_seed{SEED}__{SLOT}.json"


class SGNNET_SparseBFS(nn.Module):
    """Beam-gated BFS: only top-M active nodes (by norm) broadcast per K_iter step."""

    def __init__(self, base: SGNNET_SmallWorld, beam_m: int):
        super().__init__()
        self.base = base
        self.beam_m = beam_m

    # Delegated properties — required by SGNNET_Resonant_CUDA and SGNNET_AntiHebbian
    @property
    def W_pos(self):     return self.base.W_pos
    @property
    def W_phase(self):   return self.base.W_phase
    @property
    def conn_hh(self):   return self.base.conn_hh
    @conn_hh.setter
    def conn_hh(self, v): self.base.conn_hh = v
    @property
    def N_hidden(self):  return self.base.N_hidden
    @property
    def K_iter(self):    return self.base.K_iter
    @property
    def C_ho_mask(self): return self.base.C_ho_mask

    def _normalise(self, Z): return self.base._normalise(Z)
    def _readout(self, Z):   return self.base._readout(Z)
    def _seed(self, x):      return self.base._seed(x)

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        B, N, D = Z.shape
        conn_hh = self.base.conn_hh
        K_hh = conn_hh.shape[1]
        M = min(self.beam_m, N)

        for _ in range(self.base.K_iter):
            mean_norms = Z.norm(dim=-1).mean(0)          # [N]
            _, top_idx = mean_norms.topk(M, largest=True)
            nb_flat = conn_hh[top_idx].reshape(-1)       # [M*K_hh]
            Z_senders_exp = (Z[:, top_idx, :]
                             .unsqueeze(2).expand(-1, -1, K_hh, -1)
                             .reshape(B, -1, D))
            Z_update = torch.zeros_like(Z)
            Z_update.scatter_add_(1, nb_flat.view(1, -1, 1).expand(B, -1, D), Z_senders_exp)
            recv_mask = torch.zeros(N, dtype=torch.bool, device=Z.device)
            recv_mask[nb_flat] = True
            Z_new = Z.clone()
            Z_new[:, recv_mask, :] = Z_update[:, recv_mask, :]
            Z = self.base._normalise(Z_new)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)
        Z = self._route(Z)
        return self.base._readout(Z)


def make_model(D: int, K_iter: int, beam_size: int = 16) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_iter, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=0.90,
    )
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=beam_size, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=False)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=False)
    else:
        resonant = SGNNET_Resonant(
            base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=beam_size, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_model_bfs(bfs_m: int) -> nn.Module:
    """Standard D=16 K=5 stack but with BFS broadcaster limit = bfs_m."""
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=16, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=0.90,
    )
    bfs = SGNNET_SparseBFS(base, beam_m=bfs_m)
    # Wrap Resonant + AH on top of BFS (CUDA path uses _CUDA variants)
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            bfs, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=False)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=False)
    else:
        resonant = SGNNET_Resonant(
            bfs, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


K_ITER = 5  # default for standard configs

# Resonant-beam configs: (D, K_iter, resonant_beam_size)
CONFIGS_RES = {
    "Ref":        (16, 5, 16, "D=16 K=5 Rbeam=16 (control)"),
    "A_D6":       (6,  5, 16, "D=6  K=5 Rbeam=16  ~14K params"),
    "B_D4":       (4,  5, 16, "D=4  K=5 Rbeam=16  ~10K params"),
    "C_D4_K10":   (4, 10, 16, "D=4  K=10 Rbeam=16 (more iters)"),
    "D_Rbeam_M8": (16, 5,  8, "D=16 K=5 Rbeam=8   (half Resonant beam)"),
    "E_Rbeam_M32":(16, 5, 32, "D=16 K=5 Rbeam=32  (double Resonant beam)"),
}

# BFS broadcaster configs: (bfs_m, description)
CONFIGS_BFS = {
    "F_bfs_M32": (32,  "BFS top-32 broadcasters / 2048 nodes (1.6%)"),
    "G_bfs_M64": (64,  "BFS top-64 broadcasters / 2048 nodes (3.1%)"),
    "H_bfs_M16": (16,  "BFS top-16 broadcasters / 2048 nodes (0.8%) — step855 ref"),
}


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
        pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step864 — D floor + beam M ablation T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  MLP insight: task intrinsic rank ≈ 5-6 → D floor and M probes")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        is_bfs = key in CONFIGS_BFS
        is_res = key in CONFIGS_RES
        if not (is_bfs or is_res):
            print(f"  skip {key}"); continue

        if is_bfs:
            bfs_m, desc = CONFIGS_BFS[key]
            model = make_model_bfs(bfs_m)
            mode_tag = f"BFS-M{bfs_m}"
        else:
            D, K_iter, beam_size, desc = CONFIGS_RES[key]
            model = make_model(D, K_iter, beam_size)
            mode_tag = f"D{D}-K{K_iter}-Rb{beam_size}"

        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}  [{mode_tag}]")

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
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0.94)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        rec = {"label": desc, "n_params": n_p, "best": best, "best_ep": best_ep,
               "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        if is_bfs:
            rec.update({"mode": "bfs", "bfs_m": bfs_m})
        else:
            rec.update({"mode": "res", "D": D, "K_iter": K_iter, "beam_size": beam_size})
        results[key] = rec
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 864 SUMMARY — D floor + BFS-M + Resonant beam ablation")
    print(f"{'='*70}")
    print(f"  {'config':<16} {'mode':<10} {'params':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        tag = f"BFS-M{r['bfs_m']}" if r['mode'] == 'bfs' else f"D{r['D']}-K{r['K_iter']}"
        print(f"  {k:<16} {tag:<10} {r['n_params']:>8,} {r['best']:>7.4f} "
              f"{r['delta_vs_ref']*100:>+9.2f}pp")
    d4 = results.get("B_D4", {}).get("delta_vs_ref", None)
    d6 = results.get("A_D6", {}).get("delta_vs_ref", None)
    if d4 is not None:
        print(f"\n  D=4: {'10K HEADLINE VIABLE' if d4 >= -0.01 else f'fails ({d4*100:+.2f}pp)'}")
    if d6 is not None:
        print(f"  D=6: {'DEFAULT CANDIDATE' if d6 >= -0.005 else f'marginal ({d6*100:+.2f}pp)'}")
    for bfs_k in ["H_bfs_M16", "F_bfs_M32", "G_bfs_M64"]:
        if bfs_k in results:
            r = results[bfs_k]
            print(f"  {bfs_k}: {r['best']:.4f}  {r['delta_vs_ref']*100:+.2f}pp  "
                  f"{r['elapsed_s']}s")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
