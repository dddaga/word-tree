"""Step 855: Sparse BFS routing ablation T0 — beam-gated broadcaster selection.

MOTIVATION
==========
Current routing: O(B·N·K_hh·D) per iter — ALL N=2048 nodes broadcast every step.
Many nodes are near-quiescent per input; broadcasting from them wastes compute.

Beam-gated BFS: only top-M active nodes (by activation norm) broadcast per iter.
Claimed routing FLOP reduction: 128× at M=16 (fixed), 218× with cascading schedule.

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref          : dense routing (all N broadcast)
  A_fixed_M16  : top-16 broadcasters every iter, beam=[16,16,16,16,16]
  B_cascade    : narrowing schedule beam=[16,16,8,4,4]
  C_quiet_zero : cascade + quiet (non-frontier) nodes zeroed before routing
  D_readout_active : cascade + readout only over ever-active nodes

SUCCESS CRITERIA
  A_fixed_M16 within -0.5pp of Ref → beam gating viable
  B_cascade >= A_fixed_M16 → cascading useful
  C_quiet_zero >= B_cascade → quiet seeds don't matter (aggressive sparsification OK)
  D_readout_active within -1pp of Ref → sparse readout viable
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

parser = argparse.ArgumentParser(description="Step 855: Sparse BFS routing T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_fixed_M16,B_cascade,C_quiet_zero,D_readout_active")
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
OUT_PATH = ROOT / "results" / f"train_step855_sparse_bfs_seed{SEED}__{SLOT}.json"

# Beam schedules: list of M values per K_iter round
BEAM_SCHEDULES = {
    "Ref":              None,                  # dense — all N broadcast
    "A_fixed_M16":      [16, 16, 16, 16, 16],
    "B_cascade":        [16, 16, 8, 4, 4],
    "C_quiet_zero":     [16, 16, 8, 4, 4],    # + zero quiet nodes
    "D_readout_active": [16, 16, 8, 4, 4],    # + sparse readout
}


class SGNNET_SparseBFS(nn.Module):
    """Wraps SGNNET_SmallWorld base with beam-gated BFS routing.

    beam_schedule : list of M values per K_iter step, or None for dense.
    quiet_zero    : if True, nodes outside the ever-active frontier are zeroed
                    before each routing pass (tests whether quiet seed matters).
    sparse_readout: if True, readout uses only ever-active nodes.
    """

    def __init__(self, base: SGNNET_SmallWorld,
                 beam_schedule=None,
                 quiet_zero: bool = False,
                 sparse_readout: bool = False):
        super().__init__()
        self.base = base
        self.beam_schedule = beam_schedule
        self.quiet_zero = quiet_zero
        self.sparse_readout = sparse_readout

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.base.W_phase

    def _route_bfs(self, Z: torch.Tensor) -> torch.Tensor:
        """Beam-gated BFS routing. Z: [B, N, D]."""
        B, N, D = Z.shape
        conn_hh = self.base.conn_hh   # [N, K_hh]
        schedule = self.beam_schedule

        # Track ever-active set for quiet_zero / sparse_readout
        ever_active = torch.zeros(N, dtype=torch.bool, device=Z.device)

        for t, M in enumerate(schedule):
            # Select top-M nodes by L2 norm across D
            norms = Z.norm(dim=-1)            # [B, N]
            mean_norms = norms.mean(dim=0)    # [N] — mean over batch for stable selection

            M_clamped = min(M, N)
            _, top_idx = mean_norms.topk(M_clamped, largest=True)  # [M]
            ever_active[top_idx] = True

            if self.quiet_zero and t == 0:
                # Zero non-frontier nodes — test if quiet seed signal matters
                mask = torch.zeros(N, dtype=Z.dtype, device=Z.device)
                mask[top_idx] = 1.0
                Z = Z * mask.view(1, N, 1)

            # Gather: top-M nodes broadcast to their K_hh neighbors
            # neighbors of selected nodes: [M, K_hh]
            nb_idx = conn_hh[top_idx]              # [M, K_hh]
            nb_flat = nb_idx.reshape(-1)            # [M*K_hh]

            # Messages from top_idx nodes: [B, M, D]
            Z_senders = Z[:, top_idx, :]            # [B, M, D]

            # Broadcast each sender to its K_hh neighbors (scatter add)
            # Build update: [B, N, D] zero buffer, scatter-add sender signal
            # Each sender m contributes to conn_hh[m, k] for k in K_hh
            Z_update = torch.zeros_like(Z)
            # Expand senders for each of K_hh targets: [B, M*K_hh, D]
            Z_senders_exp = Z_senders.unsqueeze(2).expand(-1, -1, K_HH, -1).reshape(B, -1, D)
            # Scatter-add into Z_update
            Z_update.scatter_add_(
                1,
                nb_flat.view(1, -1, 1).expand(B, -1, D),
                Z_senders_exp,
            )

            # Only update positions that received messages
            recv_mask = torch.zeros(N, dtype=torch.bool, device=Z.device)
            recv_mask[nb_flat] = True

            # Blend: receivers get updated, non-receivers keep Z
            Z_new = Z.clone()
            Z_new[:, recv_mask, :] = Z_update[:, recv_mask, :]
            Z = self.base._normalise(Z_new)

        return Z, ever_active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.base._seed(x)

        if self.beam_schedule is None:
            # Dense path — use parent routing unchanged
            Z = self.base._route(Z)
            return self.base._readout(Z)

        Z, ever_active = self._route_bfs(Z)

        if self.sparse_readout:
            return self._readout_sparse(Z, ever_active)
        return self.base._readout(Z)

    def _readout_sparse(self, Z: torch.Tensor, ever_active: torch.Tensor) -> torch.Tensor:
        """Readout only over ever-active hidden nodes."""
        C_ho = self.base.C_ho_mask.float()            # [N_hidden, N_out]
        W_out = self.base.W_pos[self.base.N_hidden:]  # [N_out, D]
        W_out_norm = F.normalize(W_out, dim=-1)

        # Mask C_ho to active rows only
        C_ho_masked = C_ho * ever_active.float().unsqueeze(1)  # [N, N_out]

        A_out = torch.einsum("bhd,ho->bod", Z, C_ho_masked)    # [B, N_out, D]
        return (A_out * W_out_norm.unsqueeze(0)).sum(dim=-1)    # [B, N_out]


CONFIGS = {
    "Ref":              (None,               False, False, "dense routing (baseline)"),
    "A_fixed_M16":      ([16]*5,             False, False, "beam=[16,16,16,16,16]"),
    "B_cascade":        ([16,16,8,4,4],      False, False, "beam=[16,16,8,4,4]"),
    "C_quiet_zero":     ([16,16,8,4,4],      True,  False, "cascade + zero quiet nodes"),
    "D_readout_active": ([16,16,8,4,4],      False, True,  "cascade + sparse readout"),
}


def make_model(schedule, quiet_zero: bool, sparse_readout: bool) -> nn.Module:
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
        sparsity=0.90,
    )
    bfs = SGNNET_SparseBFS(base, beam_schedule=schedule,
                           quiet_zero=quiet_zero, sparse_readout=sparse_readout)
    # Wrap with AH (standard stack)
    if DEVICE.type == "cuda":
        resonant = SGNNET_Resonant_CUDA(
            base if schedule is None else bfs,
            K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
            beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
            resonance_threshold=0.0, compile=False)
        return SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB,
                                       variant="wpos", compile=False)
    else:
        resonant = SGNNET_Resonant(
            base if schedule is None else bfs,
            K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
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
    print(f"step855 — Sparse BFS routing T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Hypothesis: beam-gated BFS within -0.5pp of Ref at M=16")
    print(f"  FLOP target: 128× routing reduction at M=16 (vs dense N=2048)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip unknown: {key}"); continue
        schedule, quiet_zero, sparse_readout, desc = CONFIGS[key]
        model = make_model(schedule, quiet_zero, sparse_readout)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        beam_str = str(schedule) if schedule else "all-N"
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}  beam={beam_str}")

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

        results[key] = {
            "beam_schedule": schedule, "quiet_zero": quiet_zero,
            "sparse_readout": sparse_readout, "label": desc,
            "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(best - (ref_acc or 0.94), 4),
            "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 855 SUMMARY — Sparse BFS routing T0")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<18} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")
    a16 = results.get("A_fixed_M16", {}).get("delta_vs_ref", None)
    if a16 is not None:
        verdict = "VIABLE" if a16 >= -0.005 else "FAILS THRESHOLD (> -0.5pp)"
        print(f"\n  Beam-gating verdict: {verdict} ({a16*100:+.2f}pp at M=16)")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
