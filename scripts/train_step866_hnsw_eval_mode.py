"""Step 866: Soft routing → HNSW eval mode T0.

MOTIVATION
==========
step859 B_soft_anneal: +1.22pp T0 routing over static K_hh=2 neighbors.
The concept (learnings/concepts/soft_routing_hnsw.md) proposes a two-mode design:
  Training : dense all-pairs soft routing from beam nodes to all N W_pos positions
             → smooth gradient through distance field, W_pos geometry learned
  Inference: hard top-K lookup (exact or HNSW) → O(log N) per query

This experiment tests the full pipeline:
  1. Does all-pairs (beam-gated) training beat static-K_hh soft routing (step859)?
  2. Does switching to exact top-K at inference preserve accuracy?
  3. (Optional) Does hnswlib approximate k-NN preserve accuracy?

KEY MATH
  dist[b, m, n] = ||Z_beam[b,m] - W_pos[n]||²  computed memory-efficiently via:
                  Z_norm + W_norm - 2 * Z_beam @ W.T  → [B, M, N], no [B,M,N,D] tensor
  w[b, m, n]    = softmax_n(-β * dist[b, m, n])
  signal[b, n]  = Σ_m w[b, m, n] * Z_beam[b, m]   (= w.T @ Z_beam)

  At inference: top-K masking before softmax → hard routing to K nearest W_pos

CONFIGS (T0: 20ep, 50% data, seed=42)
  Ref           : step859 control — static K_hh=2 hard gather + AH
  A_soft_static : step859 winner — soft over K_hh=2 (β annealed 0.5→3.0)
  B_beam_topk   : beam M=32, train all-pairs soft, eval exact top-K (K=8)
  C_beam_wider  : beam M=32, train all-pairs soft, eval exact top-K (K=16)
  D_beam_train  : beam M=32, train all-pairs soft, eval = same (dense at test too)

Measurements:
  B_train_acc vs B_eval_acc : soft-to-hard transition cost
  B vs A       : benefit of all-pairs training over K_hh=2 neighborhood
  C vs B       : wider top-K at inference vs narrow
  D vs B       : eval-mode transition cost (D trains + evals dense; B evals sparse)
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
import torch.utils.data

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 866: Soft routing → HNSW eval mode T0")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_soft_static,B_beam_topk,C_beam_wider,D_beam_train")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step866_hnsw_eval_mode_seed{SEED}__{SLOT}.json"


class SGNNET_SoftRoute(nn.Module):
    """Soft routing over static K_hh neighborhood (step859 style)."""

    def __init__(self, base: SGNNET_SmallWorld, beta: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("_beta", torch.tensor(beta, dtype=torch.float32))

    def set_beta(self, beta: float): self._beta.fill_(beta)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.base.W_phase
    @property
    def conn_hh(self): return self.base.conn_hh
    @conn_hh.setter
    def conn_hh(self, v): self.base.conn_hh = v
    @property
    def N_hidden(self): return self.base.N_hidden
    @property
    def C_ho_mask(self): return self.base.C_ho_mask
    @property
    def K_iter(self):    return self.base.K_iter

    def _normalise(self, Z): return self.base._normalise(Z)
    def _readout(self, Z):   return self.base._readout(Z)
    def _seed(self, x):      return self.base._seed(x)

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        conn = self.base.conn_hh
        W_nb = self.base.W_pos[:self.base.N_hidden][conn]  # [N, K_hh, D]
        beta = self._beta.item()
        for _ in range(self.base.K_iter):
            Z_nb = Z[:, conn, :]
            diff = Z.unsqueeze(2) - W_nb.unsqueeze(0)
            dist = diff.pow(2).sum(dim=-1)             # [B, N, K_hh]
            w = F.softmax(-beta * dist, dim=2)
            Z = (Z_nb * w.unsqueeze(-1)).sum(2)
            Z = self._normalise(Z)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._seed(x); Z = self._route(Z); return self._readout(Z)


class SGNNET_BeamAllPairs(nn.Module):
    """Beam-gated all-pairs soft routing.

    Training: top-M active nodes query all N W_pos positions via distance softmax.
    Eval mode: optionally restrict routing to exact top-K nearest W_pos per beam node.

    Memory: [B, M, N] distance matrix — at B=128, M=32, N=2048 = 32MB (float32). Tractable.
    Compute: 2 * B * M * N * D MACs for distance (via efficient ||a-b||² = ||a||²+||b||²-2a·b).
    """

    def __init__(self, base: SGNNET_SmallWorld, beam_m: int = 32,
                 beta_start: float = 0.5, eval_topk: int = 0):
        """
        beam_m    : number of broadcaster nodes selected per K_iter step
        beta_start: initial β (annealed externally via set_beta)
        eval_topk : at inference, restrict to top-K nearest W_pos per beam node
                    (0 = keep dense softmax at inference too)
        """
        super().__init__()
        self.base = base
        self.beam_m = beam_m
        self.eval_topk = eval_topk
        self.register_buffer("_beta", torch.tensor(beta_start, dtype=torch.float32))

    def set_beta(self, beta: float): self._beta.fill_(beta)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.base.W_phase
    @property
    def conn_hh(self): return self.base.conn_hh
    @conn_hh.setter
    def conn_hh(self, v): self.base.conn_hh = v
    @property
    def N_hidden(self): return self.base.N_hidden
    @property
    def C_ho_mask(self): return self.base.C_ho_mask
    @property
    def K_iter(self): return self.base.K_iter

    def _normalise(self, Z): return self.base._normalise(Z)
    def _readout(self, Z):   return self.base._readout(Z)
    def _seed(self, x):      return self.base._seed(x)

    def _all_pairs_dist(self, Z_beam: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """||Z_beam[b,m] - W[n]||² computed without O(B*M*N*D) tensor.

        Z_beam : [B, M, D]
        W      : [N, D]
        returns: [B, M, N]
        """
        Z_norm = Z_beam.pow(2).sum(-1, keepdim=True)    # [B, M, 1]
        W_norm = W.pow(2).sum(-1)                        # [N]
        cross  = Z_beam @ W.T                            # [B, M, N]
        return Z_norm + W_norm[None, None, :] - 2 * cross

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        W = self.base.W_pos[:self.base.N_hidden]  # [N, D]
        beta = self._beta.item()
        B, N_h, D = Z.shape
        M = min(self.beam_m, N_h)
        use_topk = (not self.training) and (self.eval_topk > 0)
        K = self.eval_topk if use_topk else 0

        for _ in range(self.base.K_iter):
            # Select beam broadcasters by mean activation norm
            mean_norms = Z.norm(dim=-1).mean(0)           # [N]
            _, top_idx = mean_norms.topk(M, largest=True) # [M]
            Z_beam = Z[:, top_idx, :]                      # [B, M, D]

            dist = self._all_pairs_dist(Z_beam, W)         # [B, M, N]

            if use_topk and K < N_h:
                # Mask all but top-K nearest W_pos per beam node
                _, keep_idx = dist.topk(K, dim=2, largest=False)  # [B, M, K]
                mask = torch.full_like(dist, float('inf'))
                mask.scatter_(2, keep_idx, 0.0)
                dist = dist + mask

            w = F.softmax(-beta * dist, dim=2)             # [B, M, N]
            # signal[b, n, d] = Σ_m w[b, m, n] * Z_beam[b, m, d]
            signal = torch.bmm(w.transpose(1, 2), Z_beam)  # [B, N, D]
            Z = self._normalise(signal)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._seed(x); Z = self._route(Z); return self._readout(Z)


class BetaScheduler:
    def __init__(self, model, beta_start, beta_end, total_epochs):
        self.model = model; self.beta_start = beta_start
        self.beta_end = beta_end; self.total = total_epochs; self._epoch = 0

    def step(self):
        frac = min(1.0, self._epoch / max(1, self.total - 1))
        beta = self.beta_start + frac * (self.beta_end - self.beta_start)
        for m in self.model.modules():
            if hasattr(m, 'set_beta') and callable(m.set_beta):
                m.set_beta(beta); break
        self._epoch += 1
        return beta


def make_base():
    torch.manual_seed(SEED)
    return SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier", sparsity=0.90)


def make_ref():
    base = make_base()
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_soft_static():
    base = make_base()
    core = SGNNET_SoftRoute(base, beta=0.5)
    res = SGNNET_Resonant(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def make_beam(eval_topk: int):
    base = make_base()
    core = SGNNET_BeamAllPairs(base, beam_m=32, beta_start=0.5, eval_topk=eval_topk)
    res = SGNNET_Resonant(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                           beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


CONFIGS = {
    "Ref":           (make_ref,                  None,         "hard gather + AH — control"),
    "A_soft_static": (make_soft_static,           (0.5, 3.0),   "step859 soft over K_hh=2, β 0.5→3.0"),
    "B_beam_topk":   (lambda: make_beam(8),       (0.5, 5.0),   "beam M=32 all-pairs train, top-K=8 eval"),
    "C_beam_wider":  (lambda: make_beam(16),      (0.5, 5.0),   "beam M=32 all-pairs train, top-K=16 eval"),
    "D_beam_train":  (lambda: make_beam(0),       (0.5, 5.0),   "beam M=32 all-pairs train, dense eval"),
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
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step866 — Soft routing → HNSW eval mode T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  Beam M=32 all-pairs: [B={BATCH}, M=32, N={N}] dist matrix = "
          f"{BATCH * 32 * N * 4 / 1e6:.1f}MB per forward")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        make_fn, beta_range, desc = CONFIGS[key]
        model = make_fn()
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        scheduler = None
        if beta_range is not None:
            scheduler = BetaScheduler(model, beta_range[0], beta_range[1], EPOCHS)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        def make_log_fn(sched):
            def log_fn(m):
                ep = m['epoch'] + 1
                beta_str = ""
                if sched:
                    beta = sched.step()
                    if ep % 5 == 0:
                        beta_str = f"  β={beta:.2f}"
                if ep % 5 == 0:
                    print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{beta_str}", flush=True)
            return log_fn

        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS,
                                                     log_fn=make_log_fn(scheduler))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0.91)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {
            "label": desc, "n_params": n_p, "best": best, "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 866 SUMMARY — Soft routing → HNSW eval mode T0")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'params':>8} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<18} {r['n_params']:>8,} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")

    b = results.get("B_beam_topk", {}); d = results.get("D_beam_train", {})
    if b and d:
        transition_cost = (b.get("best", 0) - d.get("best", 0)) * 100
        print(f"\n  Soft→hard transition cost (B vs D): {transition_cost:+.2f}pp")
        print(f"  (≈0pp = HNSW approximation is lossless at this β)")
    a_soft = results.get("A_soft_static", {})
    if b and a_soft:
        beam_gain = (b.get("best", 0) - a_soft.get("best", 0)) * 100
        print(f"  All-pairs gain over K_hh=2 soft (B vs A): {beam_gain:+.2f}pp")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
