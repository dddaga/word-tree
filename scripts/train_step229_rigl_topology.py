"""Step 229: RigL Topology Learning — gradient-informed edge pruning + regrowth.

MOTIVATION
==========
conn_hh is fixed random, never trained. step124-B: RigL at N=1024 → +6.82pp (largest
single mechanism gain in project history). This experiments tests at D=16 efficiency config.

RigL schedule (every M epochs):
  1. Score current edges by activation correlation: score[i,k] = mean(Z[i] * Z[conn_hh[i,k]])
  2. Prune: remove 1 lowest-scoring edge per neuron (K_drop=1 of K_hh=2)
  3. Regrow: for each pruned slot, evaluate K_candidates=8 geometric W_pos neighbors
     not currently connected. Score = gradient magnitude or activation correlation.
  4. Pick highest-scoring candidate. Total edges stay K_hh=2.

Configs:
  Ref: static conn_hh (standard backprop, identical to step199 base)
  A:   RigL every 5 epochs, K_drop=1, score=activation_correlation
  B:   RigL every 2 epochs (faster adaptation)
  C:   RigL with gradient-based regrowth (true RigL — |∂L/∂edge_ij| approximated)

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep — Tier-0)
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

K_CANDIDATES = 8  # geometric neighbors to evaluate for regrowth

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step229_rigl_topology.json"


# ──────────────────────────────────────────────────────────────────────────────
# RigL topology manager
# ──────────────────────────────────────────────────────────────────────────────

class RigLTopology:
    """Manages conn_hh evolution via gradient-informed pruning + regrowth.

    Tracks:
      - co_activation: running EMA of Z[i] * Z[j] for active edges (pruning criterion)
      - candidates:    K_candidates nearest W_pos neighbors per neuron (regrowth pool)

    update() is called every M epochs by the training loop.
    """

    def __init__(self, model, device, k_candidates=K_CANDIDATES,
                 ema_decay=0.9, use_grad=False):
        """
        model:       SGNNET_AntiHebbian (wraps resonant which wraps base)
        k_candidates: candidate pool size for regrowth
        ema_decay:   smoothing for activation correlation EMA
        use_grad:    if True, use gradient magnitude for regrowth (true RigL)
                     if False, use activation correlation for regrowth (variant)
        """
        self.model        = model
        self.device       = device
        self.k_candidates = k_candidates
        self.ema_decay    = ema_decay
        self.use_grad     = use_grad

        N_h = model.m.base.N_hidden  # AntiHebbian.m = resonant; resonant.base = smallworld
        self._coact_ema   = torch.zeros(N_h, K_HH, device=device)

    def _get_base(self):
        """Navigate wrapper chain: AntiHebbian → Resonant → SmallWorld."""
        return self.model.m.base

    def accumulate(self, Z: torch.Tensor):
        """Update co-activation EMA with current routing activations.

        Z: [B, N_hidden, D] — call after each routing step.
        co_activation[i, k] = cosine_sim(Z[i], Z[conn_hh[i, k]]) per edge
        """
        base    = self._get_base()
        conn_hh = base.conn_hh         # [N, K_hh]
        N_h     = base.N_hidden

        with torch.no_grad():
            # Normalize Z for cosine similarity
            Z_n     = F.normalize(Z[:, :N_h, :], dim=-1)              # [B, N, D]
            # Gather neighbor activations
            Z_nb    = Z_n[:, conn_hh, :]                               # [B, N, K_hh, D]
            # Cosine sim: dot product (already normalized)
            coact   = (Z_n.unsqueeze(2) * Z_nb).sum(dim=-1)           # [B, N, K_hh]
            coact_m = coact.mean(dim=0)                                # [N, K_hh]

            self._coact_ema = (self.ema_decay * self._coact_ema +
                               (1.0 - self.ema_decay) * coact_m)

    def _build_candidates(self) -> torch.Tensor:
        """Build K_candidates nearest W_pos neighbors for each neuron.

        Uses cosine similarity of W_pos on S^{D-1}.
        Returns: [N, K_candidates] int64 index tensor.
        """
        base = self._get_base()
        N_h  = base.N_hidden
        with torch.no_grad():
            W = F.normalize(base.W_pos[:N_h].detach(), dim=-1)        # [N, D]
            # Cosine similarity matrix [N, N]
            sim = W @ W.T                                              # [N, N]
            sim.fill_diagonal_(-1e9)                                   # exclude self
            # Top K_candidates per neuron
            _, candidates = sim.topk(self.k_candidates, dim=-1)       # [N, K_cand]
        return candidates

    def _gradient_scores(self, loader) -> torch.Tensor:
        """Approximate |∂L/∂edge_ij| for all candidate edges.

        Runs one mini-batch forward+backward. Gradient magnitude of W_pos[i]
        dotted with normalized W_pos[j] approximates the gradient contribution
        of edge (i,j) to the loss.

        Returns: [N, K_candidates] gradient score tensor.
        """
        base      = self._get_base()
        N_h       = base.N_hidden
        candidates = self._build_candidates()                         # [N, K_cand]

        # Temporarily enable W_pos gradient tracking
        base.W_pos.requires_grad_(True)
        self.model.zero_grad()

        features, soft_labels, _ = next(iter(loader))
        features    = features.to(self.device)
        soft_labels = soft_labels.to(self.device)

        scores = self.model(features)
        loss   = F.kl_div(F.log_softmax(scores, dim=-1),
                           soft_labels, reduction="batchmean")
        loss.backward()

        grad_scores = torch.zeros(N_h, self.k_candidates, device=self.device)
        with torch.no_grad():
            if base.W_pos.grad is not None:
                grad_w   = base.W_pos.grad[:N_h]                      # [N, D]
                grad_mag = grad_w.norm(dim=-1)                         # [N]
                W_n      = F.normalize(base.W_pos[:N_h].detach(), dim=-1)  # [N, D]
                # Score candidate (i,j): |grad[i]| * |dot(W_pos[i], W_pos[j])|
                for c in range(self.k_candidates):
                    j_idx        = candidates[:, c]                   # [N]
                    dot_ij       = (W_n * W_n[j_idx]).sum(dim=-1)     # [N]
                    grad_scores[:, c] = grad_mag * dot_ij.abs()

        base.W_pos.requires_grad_(False)
        base.W_pos.grad = None
        self.model.zero_grad()
        return candidates, grad_scores

    def update(self, loader=None):
        """Run one RigL topology update step.

        1. Prune 1 edge per neuron with lowest activation correlation.
        2. Find K_candidates geometric neighbors not currently connected.
        3. Regrow 1 edge using highest-scoring candidate.
           Scoring: activation_correlation (use_grad=False) or gradient mag (use_grad=True).

        conn_hh is mutated in-place. The buffer is on CPU (built at init) — we
        need to handle device placement carefully.
        """
        base     = self._get_base()
        N_h      = base.N_hidden
        conn_hh  = base.conn_hh.clone()                               # [N, K_hh]

        # ── Step 1: build candidates (geometric neighbors) ────────────────────
        candidates = self._build_candidates()                         # [N, K_cand]

        # ── Step 2: pruning scores ─────────────────────────────────────────────
        prune_scores = self._coact_ema                                # [N, K_hh]

        # ── Step 3: regrowth scores ────────────────────────────────────────────
        if self.use_grad and loader is not None:
            _, regrow_scores = self._gradient_scores(loader)          # [N, K_cand]
        else:
            # Use activation correlation for candidates (need coact estimate)
            # Approximation: score(i, j) = |W_pos[i] · W_pos[j]| (geometric correlation)
            W_n    = F.normalize(base.W_pos[:N_h].detach(), dim=-1)   # [N, D]
            regrow_scores = torch.zeros(N_h, self.k_candidates, device=self.device)
            for c in range(self.k_candidates):
                j_idx             = candidates[:, c]                   # [N]
                dot_ij            = (W_n * W_n[j_idx]).sum(dim=-1)    # [N]
                regrow_scores[:, c] = dot_ij                          # higher = more correlated

        # ── Step 4: per-neuron update ──────────────────────────────────────────
        with torch.no_grad():
            for i in range(N_h):
                current = set(conn_hh[i].tolist())

                # Prune: remove the edge with the lowest coactivation score
                prune_k     = prune_scores[i].argmin().item()
                prune_target = conn_hh[i, prune_k].item()
                current.discard(prune_target)

                # Regrow: pick highest-scoring candidate not already connected
                cand_i   = candidates[i]                              # [K_cand]
                scores_i = regrow_scores[i]                           # [K_cand]

                # Mask out already-connected candidates
                for c_idx in range(self.k_candidates):
                    if cand_i[c_idx].item() in current or cand_i[c_idx].item() == i:
                        scores_i = scores_i.clone()
                        scores_i[c_idx] = -1e9

                best_c    = scores_i.argmax().item()
                grow_target = cand_i[best_c].item()
                current.add(grow_target)

                # Write back exactly K_hh=2 edges
                new_edges = list(current)[:K_HH]
                # Pad with any valid neighbor if we lost an edge somehow
                if len(new_edges) < K_HH:
                    for j in range(N_h):
                        if j != i and j not in current:
                            new_edges.append(j)
                            break
                conn_hh[i, :K_HH] = torch.tensor(new_edges[:K_HH],
                                                   dtype=torch.long)

        # Count rewired edges BEFORE copy (base.conn_hh = old, conn_hh = new)
        rewired = (base.conn_hh != conn_hh.to(base.conn_hh.device)).sum().item() // 2

        # Mutate buffer in-place (conn_hh is a registered buffer = not a param)
        base.conn_hh.copy_(conn_hh)

        # Reset co-activation EMA after topology change
        self._coact_ema.zero_()

        return rewired


# ──────────────────────────────────────────────────────────────────────────────
# Training loop with RigL hook
# ──────────────────────────────────────────────────────────────────────────────

class RigLTrainer:
    """Trainer that calls rigl.update() every M epochs.

    Otherwise identical to standard Trainer. Also accumulates coactivation
    statistics after each routing step via a forward hook on the model.
    """

    def __init__(self, model, train_loader, val_loader, device,
                 rigl, rigl_interval=5, n_epochs=20, lr=2.364e-3):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.rigl           = rigl
        self.rigl_interval  = rigl_interval
        self.n_epochs       = n_epochs

        self.optimizer = torch.optim.AdamW(
            [{"params": [model.W_pos], "lr": lr, "weight_decay": 0.0}]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
        )

        # Hook to accumulate Z after routing for coactivation tracking
        self._hook_handle = None
        self._register_routing_hook()

    def _register_routing_hook(self):
        """Register a hook on the AH forward to capture Z after routing."""
        # We capture the FINAL routed Z by hooking the model's forward output
        # through a wrapper. Since AntiHebbian.forward returns logits not Z,
        # we instead hook inside the loop using a module-level accumulator.
        # Simpler: re-run a quick eval pass in evaluate() for coact update.
        # We track coact directly in train_epoch() via Z reconstruction.
        pass   # handled inline in train_epoch below

    def _accumulate_coact_from_batch(self, features, soft_labels):
        """Run inference and accumulate coactivation stats for RigL.

        Uses a lightweight Z extraction: forward the model, which internally
        routes and returns logits. We need Z — so we temporarily monkey-patch
        the readout to capture Z.
        """
        base     = self.model.m.base                                   # SmallWorld
        resonant = self.model.m                                        # Resonant
        conn_hh  = base.conn_hh
        N_h      = base.N_hidden

        with torch.no_grad():
            Z = base._seed(features)
            theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
            W_n   = F.normalize(self.model.W_pos[:N_h], dim=-1)
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w  = (1.0 - self.model.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)
            Z_reflected = torch.zeros_like(Z)

            for _ in range(base.K_iter):
                Z_fwd   = F.relu(Z - theta_pos)
                Z_nb    = Z_fwd[:, conn_hh, :] * supp_w
                Z_struct = Z_nb.sum(dim=2)
                Z_remainder = Z_fwd - Z
                Z_reflected = resonant.alpha_reflect * Z_reflected + Z_remainder
                Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

            # Accumulate coactivation statistics
            self.rigl.accumulate(Z)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0; n = 0

        for features, soft_labels, _ in self.train_loader:
            features    = features.to(self.device)
            soft_labels = soft_labels.to(self.device)

            self.optimizer.zero_grad()
            scores = self.model(features)
            loss   = F.kl_div(F.log_softmax(scores, dim=-1),
                               soft_labels, reduction="batchmean")
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.W_pos.clamp_(0, 1.0)

            # Accumulate coactivation for RigL (detached, no grad)
            self._accumulate_coact_from_batch(features, soft_labels)

            total_loss += loss.item(); n += 1

        return {"train_loss": total_loss / max(n, 1)}

    def evaluate(self):
        self.model.eval()
        all_scores = []; all_labels = []
        val_loss = 0.0; n = 0

        with torch.no_grad():
            for features, soft_labels, labels in self.val_loader:
                features    = features.to(self.device)
                soft_labels = soft_labels.to(self.device)
                scores      = self.model(features)
                val_loss   += F.kl_div(F.log_softmax(scores, dim=-1),
                                        soft_labels, reduction="batchmean").item()
                all_scores.append(scores.cpu())
                all_labels.append(labels)
                n += 1

        all_s = torch.cat(all_scores, dim=0)
        all_l = torch.cat(all_labels, dim=0)
        top1  = (all_s.argmax(dim=-1) == all_l).float().mean().item()
        return {"val_loss": val_loss / max(n, 1), "val_top1": top1}

    def train(self, log_fn=None):
        history = []
        total_rewired = 0

        for epoch in range(self.n_epochs):
            train_m = self.train_epoch(epoch)
            val_m   = self.evaluate()

            if hasattr(self.model, "tick_epoch"):
                self.model.tick_epoch()

            # RigL topology update
            if (epoch + 1) % self.rigl_interval == 0:
                rewired = self.rigl.update(loader=self.train_loader)
                total_rewired += rewired
                if (epoch + 1) % 5 == 0 or rewired > 0:
                    print(f"  [RigL ep{epoch+1}] rewired={rewired} edges  "
                          f"total_rewired={total_rewired}", flush=True)

            self.scheduler.step(train_m["train_loss"])
            combined = {"epoch": epoch, **train_m, **val_m,
                        "total_rewired": total_rewired}
            history.append(combined)
            if log_fn:
                log_fn(combined)
            if (epoch + 1) % 5 == 0:
                print(f"  e{epoch+1:3d}  loss={train_m['train_loss']:.4f}  "
                      f"top1={val_m['val_top1']:.4f}", flush=True)

        return history


# ──────────────────────────────────────────────────────────────────────────────
# Build helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_model():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A", "B", "C"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    labels = {
        "Ref": "Ref: static conn_hh (standard backprop — control)",
        "A":   "A: RigL every 5 epochs, K_drop=1, score=activation_correlation",
        "B":   "B: RigL every 2 epochs (faster adaptation), score=activation_correlation",
        "C":   "C: RigL every 5 epochs, gradient-based regrowth (true RigL)",
    }

    print(f"\n{'='*70}")
    print(f"Step 229 — RigL Topology Learning (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_iter={K_ITER} | {EPOCHS}ep 50% data")
    print(f"FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"K_candidates={K_CANDIDATES} | Precedent: step124-B +6.82pp at N=1024")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n   = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}

    for key in run_keys:
        print(f"\n{'─'*60}")
        print(f"Config {labels.get(key, key)}")
        print(f"{'─'*60}")
        t0 = time.time()

        model = build_model().to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if key == "Ref":
            print(f"  params={n_p:,}  (static topology)")
            kw      = trainer_kwargs(N, n_epochs=EPOCHS)
            trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw)
            history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)

        elif key == "A":
            print(f"  params={n_p:,}  (RigL every 5ep, activation_correlation regrow)")
            rigl    = RigLTopology(model, DEVICE, k_candidates=K_CANDIDATES,
                                   use_grad=False)
            trainer = RigLTrainer(model=model, train_loader=tr, val_loader=va,
                                  device=DEVICE, rigl=rigl, rigl_interval=5,
                                  n_epochs=EPOCHS)
            history = trainer.train()

        elif key == "B":
            print(f"  params={n_p:,}  (RigL every 2ep, faster adaptation)")
            rigl    = RigLTopology(model, DEVICE, k_candidates=K_CANDIDATES,
                                   use_grad=False)
            trainer = RigLTrainer(model=model, train_loader=tr, val_loader=va,
                                  device=DEVICE, rigl=rigl, rigl_interval=2,
                                  n_epochs=EPOCHS)
            history = trainer.train()

        elif key == "C":
            print(f"  params={n_p:,}  (RigL every 5ep, gradient-based regrowth)")
            rigl    = RigLTopology(model, DEVICE, k_candidates=K_CANDIDATES,
                                   use_grad=True)
            trainer = RigLTrainer(model=model, train_loader=tr, val_loader=va,
                                  device=DEVICE, rigl=rigl, rigl_interval=5,
                                  n_epochs=EPOCHS)
            history = trainer.train()

        elapsed = time.time() - t0
        top1h   = [round(h.get("val_top1", 0.), 4) for h in history]
        best    = max(top1h); bep = int(np.argmax(top1h)) + 1

        results[key] = {
            "label":          labels.get(key, key),
            "top1_best":      best,
            "top1_last":      top1h[-1],
            "best_epoch":     bep,
            "epochs_run":     len(history),
            "top1_history":   top1h,
            "elapsed_s":      round(elapsed, 1),
            "n_params":       n_p,
            "N": N, "D": D,   "K_hh": K_HH, "K_iter": K_ITER,
            "total_rewired":  history[-1].get("total_rewired", 0),
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)  "
              f"rewired={results[key]['total_rewired']}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 229 SUMMARY — RigL Topology Learning (D=16 efficiency config)")
    print(f"{'='*70}")
    ref_best = results.get("Ref", {}).get("top1_best", 0)
    for key in run_keys:
        r     = results[key]
        delta = f"  Δ={r['top1_best']-ref_best:+.4f}" if key != "Ref" else ""
        verdict = ""
        if key != "Ref":
            if   r["top1_best"] - ref_best > 0.005:  verdict = " → ADVANCE"
            elif r["top1_best"] - ref_best > -0.010: verdict = " → NEUTRAL"
            else:                                     verdict = " → KILL"
            # RigL is worth 20ep scout if any delta > 0 based on prior +6.82pp result
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}  "
              f"(rewired={r.get('total_rewired', 0)})")

    print(f"\n  Precedent: step124-B @ N=1024 → +6.82pp. Expect positive delta here.")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
