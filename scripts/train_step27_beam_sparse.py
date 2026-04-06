"""Step 27: Soft-beam routing with source-only Z_fwd (algebraic sparsity tricks).

MOTIVATION
==========
Step 25 (hard beam-gated routing) confirmed the P7 finding from literature:
hard top-K gating kills gradient signal to non-beam neurons → all configs fail
(-9.86pp best case). This step fixes both the gradient and the memory inefficiency.

TWO ALGEBRAIC IMPROVEMENTS
===========================

Trick 1 — Soft beam weighting (fix P7 gradient kill):
  Replace: hard topk mask (binary, non-differentiable)
  With:    soft attention over beam activations via temperature-scaled softmax

  The beam gate becomes a differentiable weight vector rather than a binary mask.
  Every neuron receives some gradient, proportional to its activation magnitude.
  This is structurally analogous to sparse attention (Reformer/BigBird) which uses
  soft routing scores rather than hard argmax.

  Implementation:
    norms = Z.norm(dim=-1)                         # [B, N]
    attn  = F.softmax(norms / tau, dim=-1)         # [B, N], differentiable soft-beam
    # Top-K selected for structural + radiation, but weighted by attn, not binary
    topk_idx = norms.topk(K_route, dim=-1).indices # [B, K_route]
    attn_b = attn.gather(1, topk_idx)              # [B, K_route]
    attn_b = attn_b / attn_b.sum(dim=-1, keepdim=True)  # renormalize over beam

Trick 2 — Source-only Z_fwd allocation (memory bandwidth):
  Replace: relu(Z - theta) for ALL N neurons → [B, N, D] allocation
  With:    gather K_route×K_hh source indices FIRST, apply relu only to sources

  Implementation:
    b_conn   = conn_hh[b_idx]                               # [B, K, K_hh]
    src_idx  = b_conn.reshape(B, -1)                        # [B, K*K_hh]
    Z_src    = Z.gather(1, src_idx.unsqueeze(-1).expand(-1,-1,D_))
    theta_src = theta_pos.expand(B,N,D_).gather(1, src_idx.unsqueeze(-1).expand(-1,-1,D_))
    Z_src_fwd = F.relu(Z_src - theta_src)                   # [B, K*K_hh, D] — NOT [B,N,D]
    Z_struct_b = Z_src_fwd.view(B, K, K_hh, D_).sum(2)     # [B, K, D]

  Memory: [B, K×K_hh, D] = [B, 256, D] instead of [B, 512, D] at K=32, K_hh=8.
  At N=4096: [B, 256, D] vs [B, 4096, D] = 16× reduction.

CONFIGS
=======
  Ref  : full routing (step25 Ref baseline — all neurons, no beam)
  A    : soft-beam K=32 τ=1.0  [soft attention over top-32, no hard gate]
  B    : soft-beam K=32 τ=0.5  [sharper attention]
  C    : soft-beam K=32 τ=2.0  [flatter/more uniform attention]
  D    : soft-beam K=64 τ=1.0  [wider beam]
  E    : soft-beam K=32 τ=1.0  K_iter=8  [deeper with soft beam]
  F    : threshold-aligned beam [beam = supra-threshold only, exact zero from non-beam]
  G    : soft-beam K=32 + source-only Z_fwd [Trick 1 + Trick 2 combined]

All: D=16 N=512 Fourier dynamic_z_geo 90ep plateau store.h5.

To reproduce:
    python -u scripts/train_step27_beam_sparse.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 90
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
D      = 16
N      = 512

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        _loaders = make_loaders(DATA, batch_size=BATCH, seed=SEED)
    return _loaders


def make_resonant(K_iter: int = 3) -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_iter,
        n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=32,
        theta_init=0.1, alpha_reflect=0.3, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=1.0,
    )


class SGNNET_SoftBeam(nn.Module):
    """Soft-beam routing: differentiable beam weighting + optional source-only Z_fwd.

    Fixes P7 (hard gating kills gradient) by replacing binary beam mask with
    temperature-scaled softmax attention over beam activations. Every neuron
    receives gradient signal weighted by its relative activation.
    """

    def __init__(
        self,
        base: SGNNET_Resonant,
        K_route: int = 32,
        tau: float = 1.0,
        source_only: bool = False,
        threshold_aligned: bool = False,
    ):
        super().__init__()
        self.m                 = base
        self.K_route           = K_route
        self.tau               = tau
        self.source_only       = source_only    # Trick 2: allocate [B,K*K_hh,D] not [B,N,D]
        self.threshold_aligned = threshold_aligned  # Trick 1a: beam = above-threshold neurons

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)                              # [B, N, D]
        B, N_, D_ = Z.shape
        K_hh      = self.m.base.conn_hh.shape[-1]
        conn_hh   = self.m.base.conn_hh                               # [N, K_hh]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)     # [1, N, 1]
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)               # [N, D]

        for _ in range(self.m.base.K_iter):

            norms = Z.norm(dim=-1)    # [B, N]

            if self.threshold_aligned:
                # Trick 1a: beam = supra-threshold neurons (exact non-beam contribution = 0)
                above = (norms > self.m.theta.abs().unsqueeze(0)).float()  # [B, N]
                # Cap at K_route to avoid variable-size beams
                _, b_idx = norms.topk(min(self.K_route, N_), dim=-1)      # [B, K]
                above_b  = above.gather(1, b_idx)                         # [B, K]
                attn_b   = above_b / above_b.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                # Soft beam: temperature-scaled softmax, differentiable
                _, b_idx = norms.topk(self.K_route, dim=-1)               # [B, K]
                norms_b  = norms.gather(1, b_idx)                         # [B, K]
                attn_b   = F.softmax(norms_b / self.tau, dim=-1)          # [B, K]

            K = b_idx.shape[1]

            if self.source_only:
                # Trick 2: compute Z_fwd only for the K*K_hh structural sources
                b_conn    = conn_hh[b_idx.reshape(-1)].view(B, K, K_hh)  # [B,K,K_hh]
                src_idx   = b_conn.reshape(B, -1)                         # [B, K*K_hh]
                Z_src     = Z.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, D_))
                th_src    = theta_pos.expand(B, N_, D_).gather(
                    1, src_idx.unsqueeze(-1).expand(-1, -1, D_))
                Z_src_fwd = F.relu(Z_src - th_src)                        # [B,K*K_hh,D]
                Z_struct_b = Z_src_fwd.view(B, K, K_hh, D_).sum(2)       # [B, K, D]
            else:
                # Full Z_fwd [B, N, D] then gather beam sources
                Z_fwd    = F.relu(Z - theta_pos)                          # [B, N, D]
                b_conn   = conn_hh[b_idx.reshape(-1)].view(B, K, K_hh)   # [B,K,K_hh]
                flat_nb  = b_conn.reshape(B, -1)
                Z_nb     = Z_fwd.gather(1, flat_nb.unsqueeze(-1).expand(-1, -1, D_))
                Z_struct_b = Z_nb.view(B, K, K_hh, D_).sum(2)            # [B, K, D]

            # Beam-to-beam radiation (O(K²·D))
            W_ph_b = W_ph_norm[b_idx.reshape(-1)].view(B, K, D_)         # [B,K,D]
            Z_b    = Z.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, D_)) # [B,K,D]
            cos_bm = torch.bmm(W_ph_b, Z_b.transpose(1, 2))              # [B,K,K]
            Z_inh_b = -self.m.alpha_turing * torch.bmm(
                cos_bm.clamp(min=0), Z_b)                                 # [B,K,D]

            # Update beam neurons with soft-attention weighting
            Z_beam_new = F.normalize(
                (Z_struct_b + Z_inh_b).clamp(-10, 10), dim=-1)           # [B,K,D]

            # Soft-weighted scatter back: beam neurons updated, non-beam unchanged
            # weight by attn_b so high-activation neurons update more
            Z_updated = Z_beam_new * attn_b.unsqueeze(-1)                # [B,K,D]

            # Scatter add back to full Z
            Z_out = Z.clone()
            Z_out.scatter_add_(
                1,
                b_idx.unsqueeze(-1).expand(-1, -1, D_),
                Z_updated - Z.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, D_)) * attn_b.unsqueeze(-1)
            )
            Z = F.normalize(Z_out, dim=-1)

        return self.m.base._readout(Z)


class SGNNET_ThresholdBeam(nn.Module):
    """Threshold-aligned beam: beam = above-threshold neurons, non-beam sources are exact zeros.

    This makes the non-beam structural contribution analytically zero (Trick 1 exact form).
    Cost: O(K_active × K_hh × D) where K_active = count of above-threshold neurons.
    """

    def __init__(self, base: SGNNET_Resonant, K_route: int = 32):
        super().__init__()
        self.m       = base
        self.K_route = K_route

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z         = self.m.base._seed(x)
        B, N_, D_ = Z.shape
        conn_hh   = self.m.base.conn_hh
        K_hh      = conn_hh.shape[-1]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_ph_norm = F.normalize(self.m.W_phase, dim=-1)

        for _ in range(self.m.base.K_iter):
            # Threshold-aligned beam: Z_fwd is exactly zero for non-beam
            Z_fwd = F.relu(Z - theta_pos)                               # [B, N, D]

            # Beam = neurons with nonzero Z_fwd (activation above threshold)
            # Cap at K_route to maintain fixed compute budget
            above_norm = Z_fwd.norm(dim=-1)                             # [B, N]
            _, b_idx   = above_norm.topk(self.K_route, dim=-1)         # [B, K]

            # Since we use Z_fwd (already thresholded), non-beam sources contribute 0
            b_conn   = conn_hh[b_idx.reshape(-1)].view(B, self.K_route, K_hh)
            flat_nb  = b_conn.reshape(B, -1)
            Z_nb     = Z_fwd.gather(1, flat_nb.unsqueeze(-1).expand(-1, -1, D_))
            Z_struct_b = Z_nb.view(B, self.K_route, K_hh, D_).sum(2)  # [B,K,D]

            # Beam-to-beam radiation
            W_ph_b = W_ph_norm[b_idx.reshape(-1)].view(B, self.K_route, D_)
            Z_b    = Z_fwd.gather(1, b_idx.unsqueeze(-1).expand(-1, -1, D_))
            cos_bm = torch.bmm(W_ph_b, Z_b.transpose(1, 2))
            Z_inh_b = -self.m.alpha_turing * torch.bmm(cos_bm.clamp(min=0), Z_b)

            Z_beam_new = F.normalize((Z_struct_b + Z_inh_b).clamp(-10, 10), dim=-1)

            # Hard update beam, zero non-beam delta (threshold-aligned: exact)
            Z_out = Z.clone()
            Z_out.scatter_(1, b_idx.unsqueeze(-1).expand(-1, -1, D_), Z_beam_new)
            Z = F.normalize(Z_out, dim=-1)

        return self.m.base._readout(Z)


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(N, n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})  "
          f"diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# (label, K_route, tau, K_iter, source_only, threshold_aligned, use_threshold_beam)
CONFIGS = [
    ("Ref. full routing  no beam  K_iter=3  [step25 Ref]",
     None, None, 3, False, False, False),
    ("A.  soft-beam K=32 τ=1.0  K_iter=3  [differentiable beam]",
     32, 1.0, 3, False, False, False),
    ("B.  soft-beam K=32 τ=0.5  K_iter=3  [sharper beam]",
     32, 0.5, 3, False, False, False),
    ("C.  soft-beam K=32 τ=2.0  K_iter=3  [flatter beam]",
     32, 2.0, 3, False, False, False),
    ("D.  soft-beam K=64 τ=1.0  K_iter=3  [wider beam]",
     64, 1.0, 3, False, False, False),
    ("E.  soft-beam K=32 τ=1.0  K_iter=8  [deep soft-beam]",
     32, 1.0, 8, False, False, False),
    ("F.  threshold-aligned beam  K=32  [Trick1: non-beam = exact zero]",
     32, None, 3, False, False, True),
    ("G.  soft-beam K=32 τ=1.0  +source-only  [Trick1+2 combined]",
     32, 1.0, 3, True, False, False),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  D={D}  N={N}")
    print("Fixing P7: replacing hard beam gate with differentiable soft-beam attention")
    print("Trick 2: source-only Z_fwd avoids full [B,N,D] allocation")
    print(f"Step25 hard-beam best: ~19-23%  |  Full routing ref: ~28.56%")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    keys    = ["Ref", "A", "B", "C", "D", "E", "F", "G"]
    for key, (label, K_route, tau, K_iter, src_only, thresh_align, use_thr) in zip(keys, CONFIGS):
        resonant = make_resonant(K_iter=K_iter).to(DEVICE)
        if K_route is None:
            model = resonant
        elif use_thr:
            model = SGNNET_ThresholdBeam(resonant, K_route=K_route).to(DEVICE)
        else:
            model = SGNNET_SoftBeam(
                resonant, K_route=K_route, tau=tau,
                source_only=src_only, threshold_aligned=thresh_align,
            ).to(DEVICE)
        meta  = {
            "K_route": K_route, "tau": tau, "K_iter": K_iter,
            "source_only": src_only, "threshold_aligned": thresh_align or use_thr,
            "D": D, "N": N,
        }
        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = Path("results/train_step27_beam_sparse.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")

    ref_full = results.get("Ref", {}).get("top1_best", 0.2856)
    ref_29   = 0.2922
    print(f"\n-- Soft-beam routing sweep  (full_routing={ref_full:.4f}  step9_ref={ref_29:.4f}) ---")
    print("  %-52s  %9s  %9s  %9s  %8s  %6s" % (
        "Config", "top1", "vs_full", "vs_step9", "ep_frac", "t(s)"))
    print("  " + "-"*100)
    for k, r in results.items():
        d_full = r["top1_best"] - ref_full
        d_ref  = r["top1_best"] - ref_29
        print("  %-52s  %9.4f  %+9.4f  %+9.4f  %7.1f%%  %6.0f" % (
            r["label"][:52], r["top1_best"], d_full, d_ref,
            r.get("best_epoch_frac", 0) * 100, r["elapsed_s"]))

    print("\n  If A/B/C recover to ~28%, soft beam weighting is the fix (P7 confirmed).")
    print("  If G matches A, Trick2 is free — implement as default in SGNNET_BeamGated.")
    print("  If F matches or beats A, threshold-alignment is the cleanest implementation.")
