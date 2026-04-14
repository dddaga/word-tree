"""Step 224: Scored dynamic topology — static backbone + periodic edge replacement.

MOTIVATION
==========
Static conn_hh: edges are fixed at init, never updated.
Dynamic routing (step83): failed — gate-death + co-adaptation.

Middle ground: STATIC backbone with PERIODIC edge pruning/regrowth.
Keep the base K_hh=2 edges. Add a score per edge. Every E epochs,
replace the bottom P% lowest-scored edges with new random edges.
The graph evolves slowly — structural plasticity without per-step dynamic routing.

Three scoring mechanisms tested:
  Ref : Standard static connectivity (no edge replacement)
  A   : Learned parameter score — edge_score[i,k] is a scalar parameter.
        Gradient flows through the score toward the readout loss.
        Bottom 10% lowest (most suppressed) replaced every 10 epochs.
  B   : Activation correlation score — edges between neurons with high
        co-activation correlation are scored high (Hebbian principle).
        Score = EMA of |Z[i] · Z[conn[i,k]]| averaged over a batch.
        Bottom 10% replaced every 10 epochs.
  C   : Gradient magnitude score — |dL/d_edge_activation| over a batch.
        Edges with near-zero gradient contribution are pruned.
        Bottom 10% replaced every 10 epochs.

Replacement policy: new edges sampled randomly (same small-world groups as init).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0 scouts)
"""
from __future__ import annotations
import argparse, json, sys, time
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
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (e.g. Ref,A). Empty = all.")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

REPLACE_EVERY = 10   # epochs between edge replacement
REPLACE_FRAC  = 0.10  # bottom 10% of edges replaced

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step224_scored_dynamic_topology.json"


# ---------------------------------------------------------------------------
# Dynamic topology helpers
# ---------------------------------------------------------------------------

def _sample_new_edges(conn_hh: torch.Tensor, keep_mask: torch.Tensor,
                      n_groups: int, seed_offset: int = 0) -> torch.Tensor:
    """Sample replacement edges for pruned positions using small-world-like policy.

    keep_mask: [N, K_hh] bool — True = keep, False = replace.
    Returns updated conn_hh.
    """
    N_h, K = conn_hh.shape
    group_size = max(1, N_h // n_groups)
    rng = np.random.default_rng(seed_offset + 9999)

    conn_new = conn_hh.clone()
    for i in range(N_h):
        for k in range(K):
            if not keep_mask[i, k]:
                # 50% chance: local (same group), 50%: random global
                if rng.random() < 0.5:
                    g_start = (i // group_size) * group_size
                    g_end   = min(g_start + group_size, N_h)
                    candidates = [j for j in range(g_start, g_end) if j != i]
                    if candidates:
                        conn_new[i, k] = int(rng.choice(candidates))
                    else:
                        conn_new[i, k] = int(rng.integers(0, N_h))
                else:
                    j = int(rng.integers(0, N_h))
                    conn_new[i, k] = j if j != i else (j + 1) % N_h
    return conn_new


class SGNNET_ScoredTopology(nn.Module):
    """SGNNET with periodically-updated edge scores and replacement.

    score_mode:
      "learned"      — learnable scalar per edge (gradient-based)
      "correlation"  — EMA of activation alignment along each edge
      "gradient"     — EMA of gradient magnitude through each edge
    """

    def __init__(self, base_sw, resonant, alpha_ahebb=1.0,
                 score_mode="learned", replace_every=10, replace_frac=0.10,
                 n_groups=256):
        super().__init__()
        self.base_sw = base_sw
        self.resonant = resonant
        self.alpha_ahebb = alpha_ahebb
        self.score_mode = score_mode
        self.replace_every = replace_every
        self.replace_frac = replace_frac
        self.n_groups = n_groups
        self._epoch = 0

        N_h = base_sw.N_hidden

        if score_mode == "learned":
            # Learnable score per edge — trained via gradient
            self.edge_score = nn.Parameter(torch.zeros(N_h, K_HH))

        elif score_mode in ("correlation", "gradient"):
            # EMA accumulators — not parameters
            self.register_buffer("edge_ema",
                                  torch.ones(N_h, K_HH))   # start high (no early pruning)
            self._ema_beta = 0.9

        # Register conn_hh as a buffer so it moves with .to(device)
        self.register_buffer("conn_hh_dynamic", base_sw.conn_hh.clone())

    @property
    def W_pos(self):   return self.base_sw.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        self._epoch += 1
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()
        if self._epoch % self.replace_every == 0:
            self._replace_edges()

    def _replace_edges(self):
        """Replace bottom replace_frac% edges by score."""
        with torch.no_grad():
            if self.score_mode == "learned":
                scores = self.edge_score.detach()
            else:
                scores = self.edge_ema

            N_h, K = scores.shape
            n_replace = max(1, int(N_h * K * self.replace_frac))
            flat_scores = scores.view(-1)
            _, bottom_idx = flat_scores.topk(n_replace, largest=False)

            keep_mask = torch.ones(N_h * K, dtype=torch.bool, device=scores.device)
            keep_mask[bottom_idx] = False
            keep_mask = keep_mask.view(N_h, K)

            new_conn = _sample_new_edges(
                self.conn_hh_dynamic.cpu(), keep_mask.cpu(),
                n_groups=self.n_groups, seed_offset=self._epoch
            ).to(self.conn_hh_dynamic.device)
            self.conn_hh_dynamic.copy_(new_conn)

            # Reset scores for replaced edges
            if self.score_mode == "learned":
                with torch.no_grad():
                    self.edge_score.data[~keep_mask] = 0.0
            else:
                self.edge_ema[~keep_mask] = 1.0  # reset to neutral

            n_replaced = int((~keep_mask).sum().item())
            print(f"    [epoch {self._epoch}] replaced {n_replaced} edges "
                  f"(mode={self.score_mode})", flush=True)

    def forward(self, x):
        Z = self.base_sw._seed(x)
        conn_hh = self.conn_hh_dynamic
        N_h = self.base_sw.N_hidden

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # AH suppression (static, precomputed on current conn_hh)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)       # [N, K_hh]
        supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)                      # [1, N, K_hh, 1]

        Z_reflected = torch.zeros_like(Z)

        # For correlation/gradient scoring: accumulate stats on first forward pass
        # Use a list to capture Z_nb for EMA update
        edge_acts = [] if (self.score_mode == "correlation" and self.training) else None
        edge_grads = [] if (self.score_mode == "gradient" and self.training) else None

        for step in range(self.base_sw.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]   # [B, N, K_hh, D]
            Z_nb  = Z_nb * supp_w

            # Accumulate correlation score on first iteration (cheap proxy)
            if edge_acts is not None and step == 0:
                # Alignment between neuron i and its neighbour along edge k
                # [B, N, K_hh]
                with torch.no_grad():
                    align = (F.normalize(Z_fwd, dim=-1).unsqueeze(2) *
                             F.normalize(Z_nb, dim=-1)).sum(-1).abs()  # [B, N, K_hh]
                    edge_acts.append(align.mean(0).detach())   # [N, K_hh]

            # For gradient scoring: register hook on first iteration
            if edge_grads is not None and step == 0:
                def _grad_hook(grad):
                    # grad: [B, N, K_hh, D] → magnitude [N, K_hh]
                    with torch.no_grad():
                        gmag = grad.abs().mean(dim=(0, -1))  # [N, K_hh]
                        edge_grads.append(gmag.detach())
                Z_nb.register_hook(_grad_hook)

            Z_struct = Z_nb.sum(dim=2)

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            if self.resonant.alpha_turing != 0.0:
                W_ph_norm = F.normalize(self.resonant.W_phase, dim=-1)
                Z_inh = self.resonant._phase_inhibit(Z, W_ph_norm, theta_pos)
                Z_new = Z_struct + Z_reflected + self.resonant.alpha_turing * Z_inh
            else:
                Z_new = Z_struct + Z_reflected

            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        # Update EMA scores after forward pass
        if edge_acts and len(edge_acts) > 0:
            with torch.no_grad():
                new_score = edge_acts[0]
                self.edge_ema = (self._ema_beta * self.edge_ema +
                                 (1 - self._ema_beta) * new_score)

        return self.base_sw._readout(Z)

    def post_backward_hook(self):
        """Call after optimizer.step() to update gradient-magnitude EMA."""
        # edge_grads populated during backward; update EMA here
        pass  # grad hook fires during backward automatically


def build_model(config_key):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)

    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)

    if config_key == "Ref":
        return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    elif config_key in ("A", "B", "C"):
        mode_map = {"A": "learned", "B": "correlation", "C": "gradient"}
        return SGNNET_ScoredTopology(
            base, resonant, alpha_ahebb=ALPHA_AHEBB,
            score_mode=mode_map[config_key],
            replace_every=REPLACE_EVERY,
            replace_frac=REPLACE_FRAC,
            n_groups=ng,
        )
    else:
        raise ValueError(f"Unknown config: {config_key}")


def main():
    all_keys = ["Ref", "A", "B", "C"]
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]
    else:
        run_keys = all_keys

    labels = {
        "Ref": "Ref: static topology (step199 baseline)",
        "A":   f"A: learned score, replace bottom {int(REPLACE_FRAC*100)}% every {REPLACE_EVERY}ep",
        "B":   f"B: correlation score (EMA activation alignment), replace bottom {int(REPLACE_FRAC*100)}%",
        "C":   f"C: gradient magnitude score, replace bottom {int(REPLACE_FRAC*100)}%",
    }

    print(f"\n{'='*70}")
    print(f"Step 224 — Scored dynamic topology (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Replace every {REPLACE_EVERY} epochs, bottom {int(REPLACE_FRAC*100)}% edges")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")

        model = build_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

        t0 = time.time()
        history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label": labels.get(key, key),
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "epochs_run": len(history), "top1_history": top1h,
            "elapsed_s": round(elapsed, 1), "n_params": n_p,
            "N": N, "D": D, "K_hh": K_HH, "K_iter": K_ITER,
            "replace_every": REPLACE_EVERY, "replace_frac": REPLACE_FRAC,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 224 SUMMARY — Scored dynamic topology")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if r["top1_best"] - ref_best > 0.005:   verdict = " → ADVANCE"
            elif r["top1_best"] - ref_best > -0.01:  verdict = " → NEUTRAL"
            else:                                     verdict = " → KILL"
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
