"""Step 230: Gumbel-Softmax Differentiable Topology Learning.

MOTIVATION
==========
Learn conn_hh end-to-end via differentiable edge selection.
Gradient flows through discrete topology choices using the straight-through estimator.

Add `edge_logits = nn.Parameter(torch.zeros(N, K_candidates))` where K_candidates=8
geometric W_pos neighbors. Each neuron differentiably selects K_hh=2 edges from
K_candidates=8 options using Gumbel-Top-K.

Forward: hard discrete edges (straight-through — as-if gradient).
Backward: soft Gumbel-Softmax probabilities (gradient-smooth).

Decoupled ST-GS (2024 paper): separate temperatures for forward (τ_fwd=0.1) and
backward (τ_bwd=1.0) passes improves gradient fidelity over original ST-GS.

Configs:
  Ref: static conn_hh (standard backprop — control)
  A:   Gumbel-Softmax, τ_fwd=0.1, τ_bwd=1.0 (Decoupled ST-GS, constant temperature)
  B:   Gumbel-Softmax with annealing τ_bwd: 2.0 → 0.1 over training
  C:   Gumbel-Softmax + AH on learned topology (full config)

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

K_CANDIDATES = 8   # candidate pool size per neuron

FLOPS = 3 * N * K_HH * D * K_ITER
OUT_PATH = ROOT / "results" / "train_step230_gumbel_topology.json"


# ──────────────────────────────────────────────────────────────────────────────
# Gumbel-Top-K straight-through estimator
# ──────────────────────────────────────────────────────────────────────────────

def gumbel_topk_st(logits: torch.Tensor, k: int,
                    tau_fwd: float = 0.1, tau_bwd: float = 1.0,
                    training: bool = True) -> torch.Tensor:
    """Decoupled Straight-Through Gumbel-Top-K.

    Selects K edges per neuron from K_candidates options.
    Forward pass: hard {0, 1} selection (near-discrete at τ_fwd → 0).
    Backward pass: gradient through soft Gumbel-Softmax at τ_bwd.

    Args:
        logits:   [N, K_candidates] — learned log-probabilities for each edge
        k:        number of edges to select (K_hh=2)
        tau_fwd:  temperature for forward hard selection (lower = more discrete)
        tau_bwd:  temperature for backward gradient (higher = smoother gradient)
        training: if False, use argmax selection (no Gumbel noise)

    Returns:
        edge_weights: [N, K_candidates] soft selection weights (sum ≈ k per row)
                      Forward: hard 0/1 (nearly). Backward: differentiable.
    """
    if not training:
        # Eval: hard argmax selection, no noise
        _, top_k_idx = logits.topk(k, dim=-1)
        hard = torch.zeros_like(logits).scatter_(1, top_k_idx, 1.0)
        return hard

    # ── Forward: perturb with Gumbel noise, select top-k at low temperature ──
    # Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
    U       = torch.rand_like(logits).clamp(1e-10, 1.0 - 1e-10)
    gumbel  = -torch.log(-torch.log(U))
    perturbed_fwd = (logits + gumbel) / tau_fwd            # [N, K_cand]

    # Top-k hard selection
    top_k_idx = perturbed_fwd.topk(k, dim=-1).indices     # [N, k]
    hard_fwd  = torch.zeros_like(logits).scatter_(1, top_k_idx, 1.0)

    # ── Backward: soft Gumbel-Softmax with separate (higher) temperature ──────
    # Adding same Gumbel noise to backward for consistency (Decoupled ST-GS)
    perturbed_bwd = (logits + gumbel) / tau_bwd            # [N, K_cand]
    soft_bwd  = F.softmax(perturbed_bwd, dim=-1) * k       # scale to sum ≈ k

    # Straight-through trick: hard in forward, soft gradient in backward
    # hard_fwd + (soft_bwd - soft_bwd.detach())  →  forward=hard_fwd, grad=d/d(soft_bwd)
    edge_weights = hard_fwd + (soft_bwd - soft_bwd.detach())

    return edge_weights   # [N, K_cand]


# ──────────────────────────────────────────────────────────────────────────────
# SGNNET with Gumbel topology
# ──────────────────────────────────────────────────────────────────────────────

class SGNNET_GumbelTopology(nn.Module):
    """SGNNET with differentiable edge selection via Gumbel-Top-K.

    edge_logits: [N, K_candidates] — learned preference for each candidate edge.
    K_candidates edges are pre-selected at init as K geometric W_pos neighbors.
    Updated every UPDATE_INTERVAL epochs by rebuilding from current W_pos K-NN.

    Routing loop is like AntiHebbian but uses Gumbel-selected soft edge weights
    instead of fixed conn_hh index gather.

    use_ah: if True, apply AH suppression (Config C)
    """

    def __init__(self, base, resonant, alpha_ahebb=1.0,
                 tau_fwd=0.1, tau_bwd=1.0,
                 use_ah=True, use_annealing=False, n_epochs=20):
        super().__init__()
        self.base             = base
        self.resonant         = resonant
        self.alpha_ahebb      = alpha_ahebb
        self.tau_fwd          = tau_fwd
        self.tau_bwd_init     = tau_bwd
        self.tau_bwd          = tau_bwd
        self.use_ah           = use_ah
        self.use_annealing    = use_annealing
        self.n_epochs         = n_epochs
        self._current_epoch   = 0

        N_h = base.N_hidden

        # Learnable edge logits: [N, K_candidates]
        self.edge_logits = nn.Parameter(torch.zeros(N_h, K_CANDIDATES))

        # Candidate edges: K geometric neighbors by W_pos cosine sim
        # Register as buffer so it moves with .to(device)
        candidates = self._build_candidates()
        self.register_buffer("candidates", candidates)   # [N, K_cand] int64

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        self._current_epoch += 1
        if self.use_annealing:
            # Anneal τ_bwd: tau_bwd_init → 0.1 linearly over n_epochs
            frac          = min(1.0, self._current_epoch / max(self.n_epochs, 1))
            self.tau_bwd  = self.tau_bwd_init + frac * (0.1 - self.tau_bwd_init)

        # Rebuild candidates from current W_pos every 10 epochs
        if self._current_epoch % 10 == 0:
            new_candidates = self._build_candidates()
            self.candidates.copy_(new_candidates)

        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def _build_candidates(self) -> torch.Tensor:
        """K_candidates nearest W_pos neighbors per neuron.

        Returns [N_hidden, K_candidates] int64 on CPU.
        """
        N_h = self.base.N_hidden
        with torch.no_grad():
            W = F.normalize(self.base.W_pos[:N_h].detach().cpu().float(), dim=-1)
            sim = W @ W.T                                              # [N, N]
            sim.fill_diagonal_(-1e9)
            _, idx = sim.topk(K_CANDIDATES, dim=-1)                   # [N, K_cand]
        return idx.to(torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B     = x.shape[0]
        N_h   = self.base.N_hidden
        Z     = self.base._seed(x)                                     # [B, N, D]

        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Gumbel edge selection: [N, K_cand] soft edge weights
        # Each row sums to ~K_hh=2 (hard selection of K_hh=2 from K_cand=8)
        edge_w = gumbel_topk_st(
            self.edge_logits, k=K_HH,
            tau_fwd=self.tau_fwd, tau_bwd=self.tau_bwd,
            training=self.training
        )   # [N, K_cand]

        # Pre-compute AH suppression if needed
        if self.use_ah:
            W_n     = F.normalize(self.W_pos[:N_h], dim=-1)           # [N, D]
            # AH over candidate edges (not fixed conn_hh)
            cand    = self.candidates                                   # [N, K_cand]
            pos_sim = (W_n.unsqueeze(1) * W_n[cand]).sum(-1)           # [N, K_cand]
            ah_supp = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)) # [N, K_cand]
            # Combined: Gumbel weight * AH suppression
            combined_w = (edge_w * ah_supp).unsqueeze(0).unsqueeze(-1) # [1,N,K_cand,1]
        else:
            combined_w = edge_w.unsqueeze(0).unsqueeze(-1)             # [1,N,K_cand,1]

        Z_reflected = torch.zeros_like(Z)

        for _ in range(self.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)                              # [B, N, D]

            # Gather candidate activations: [B, N, K_cand, D]
            # candidates: [N, K_cand] → expand for batch
            cand_exp = self.candidates.unsqueeze(0).expand(
                B, -1, -1
            )                                                          # [B, N, K_cand]
            cand_exp_d = cand_exp.unsqueeze(-1).expand(
                -1, -1, -1, D
            )                                                          # [B, N, K_cand, D]
            Z_nb = torch.gather(
                Z_fwd.unsqueeze(2).expand(-1, -1, K_CANDIDATES, -1),
                1,
                cand_exp_d
            )   # [B, N, K_cand, D]
            # Note: Z_fwd has shape [B, N, D]; we want Z_fwd[:, candidates, :]
            # Easier: direct index (same as standard routing but over K_cand)
            Z_nb = Z_fwd[:, self.candidates, :]                        # [B, N, K_cand, D]

            # Weighted sum: edge_w acts as soft selection
            Z_struct = (Z_nb * combined_w).sum(dim=2)                  # [B, N, D]

            Z_remainder = Z_fwd - Z
            Z_reflected = self.resonant.alpha_reflect * Z_reflected + Z_remainder

            Z_new = Z_struct + Z_reflected
            Z     = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.base._readout(Z)


# ──────────────────────────────────────────────────────────────────────────────
# Build helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_base_parts():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return base, resonant


def build_ref():
    base, resonant = build_base_parts()
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def build_gumbel(tau_fwd=0.1, tau_bwd=1.0, use_ah=True,
                 use_annealing=False, n_epochs=EPOCHS):
    base, resonant = build_base_parts()
    return SGNNET_GumbelTopology(
        base, resonant, alpha_ahebb=ALPHA_AHEBB,
        tau_fwd=tau_fwd, tau_bwd=tau_bwd,
        use_ah=use_ah, use_annealing=use_annealing, n_epochs=n_epochs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gumbel trainer: edge_logits get their own LR group
# ──────────────────────────────────────────────────────────────────────────────

class GumbelTrainer:
    """Standard trainer extended to handle edge_logits parameter group.

    edge_logits gets a higher LR than W_pos (topology needs faster adaptation).
    Everything else uses standard AdamW.
    """

    def __init__(self, model, train_loader, val_loader, device, n_epochs=20,
                 lr_wpos=2.364e-3, lr_logits=1e-2):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.n_epochs     = n_epochs

        param_groups = [
            {"params": [model.W_pos], "lr": lr_wpos, "weight_decay": 0.0},
            {"params": [model.edge_logits], "lr": lr_logits, "weight_decay": 1e-4},
        ]
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
        )

    def train_epoch(self):
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

    def _topology_entropy(self):
        """Log entropy of edge distribution as a diagnostic.

        Low entropy → model has committed to specific edges (good for convergence).
        High entropy → model is still exploring topology.
        """
        with torch.no_grad():
            probs = F.softmax(self.model.edge_logits, dim=-1)         # [N, K_cand]
            ent   = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        return ent.item()

    def train(self, log_fn=None):
        history = []
        for epoch in range(self.n_epochs):
            train_m = self.train_epoch()
            val_m   = self.evaluate()

            if hasattr(self.model, "tick_epoch"):
                self.model.tick_epoch()

            ent = self._topology_entropy()
            self.scheduler.step(train_m["train_loss"])

            combined = {"epoch": epoch, **train_m, **val_m,
                        "tau_bwd": self.model.tau_bwd,
                        "topology_entropy": round(ent, 4)}
            history.append(combined)
            if log_fn:
                log_fn(combined)
            if (epoch + 1) % 5 == 0:
                print(f"  e{epoch+1:3d}  loss={train_m['train_loss']:.4f}  "
                      f"top1={val_m['val_top1']:.4f}  "
                      f"τ_bwd={self.model.tau_bwd:.3f}  "
                      f"topo_ent={ent:.3f}", flush=True)
        return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    all_keys = ["Ref", "A", "B", "C"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    labels = {
        "Ref": "Ref: static conn_hh (standard backprop — control)",
        "A":   "A: Gumbel-Softmax τ_fwd=0.1, τ_bwd=1.0 (Decoupled ST-GS, constant τ)",
        "B":   "B: Gumbel-Softmax with τ_bwd annealing: 2.0 → 0.1 over training",
        "C":   "C: Gumbel-Softmax + AH on learned topology (τ_fwd=0.1, τ_bwd=1.0)",
    }

    print(f"\n{'='*70}")
    print(f"Step 230 — Gumbel-Softmax Differentiable Topology (Tier-0 scouts)")
    print(f"N={N} D={D} K_hh={K_HH} K_cand={K_CANDIDATES} K_iter={K_ITER}")
    print(f"{EPOCHS}ep 50% data | FLOPs={FLOPS:,} (~{FLOPS/1e6:.2f}M)")
    print(f"Decoupled ST-GS (2024): τ_fwd decoupled from τ_bwd for better gradients")
    print(f"edge_logits adds {N * K_CANDIDATES:,} params ({N * K_CANDIDATES / 1e3:.1f}K)")
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

        if key == "Ref":
            model   = build_ref().to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (static topology — control)")
            kw      = trainer_kwargs(N, n_epochs=EPOCHS)
            trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw)
            history = trainer.train(n_epochs=EPOCHS, log_fn=lambda m: None)

        elif key == "A":
            model   = build_gumbel(tau_fwd=0.1, tau_bwd=1.0,
                                    use_ah=True, use_annealing=False).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Gumbel τ_fwd=0.1, τ_bwd=1.0, constant)")
            trainer = GumbelTrainer(model=model, train_loader=tr, val_loader=va,
                                    device=DEVICE, n_epochs=EPOCHS)
            history = trainer.train()

        elif key == "B":
            model   = build_gumbel(tau_fwd=0.1, tau_bwd=2.0,
                                    use_ah=True, use_annealing=True,
                                    n_epochs=EPOCHS).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Gumbel τ_bwd annealing: 2.0 → 0.1)")
            trainer = GumbelTrainer(model=model, train_loader=tr, val_loader=va,
                                    device=DEVICE, n_epochs=EPOCHS)
            history = trainer.train()

        elif key == "C":
            # AH on learned topology — same τ as A but explicitly confirming AH interacts
            # with the differentiable edges (already True by default, but explicit here)
            model   = build_gumbel(tau_fwd=0.1, tau_bwd=1.0,
                                    use_ah=True, use_annealing=False).to(DEVICE)
            n_p     = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  params={n_p:,}  (Gumbel + AH on learned topology, higher LR logits)")
            # Higher logit LR to allow AH to reshape topology faster
            trainer = GumbelTrainer(model=model, train_loader=tr, val_loader=va,
                                    device=DEVICE, n_epochs=EPOCHS, lr_logits=3e-2)
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
            "K_candidates":   K_CANDIDATES,
            "final_topo_entropy": history[-1].get("topology_entropy", None),
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)  "
              f"topo_ent={results[key]['final_topo_entropy']}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 230 SUMMARY — Gumbel-Softmax Differentiable Topology")
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
        ent_str = (f"  ent={r['final_topo_entropy']:.3f}"
                   if r.get("final_topo_entropy") is not None else "")
        print(f"  {key}: {r['top1_best']:.4f}{delta}{verdict}{ent_str}")

    # Topology analysis
    if "A" in results:
        ent_a = results["A"].get("final_topo_entropy", None)
        if ent_a is not None:
            max_ent = np.log(K_CANDIDATES)   # uniform = log(K_cand)
            print(f"\n  Topology entropy: {ent_a:.3f} / {max_ent:.3f} max"
                  f"  ({100*ent_a/max_ent:.0f}% of uniform)")
            if ent_a < max_ent * 0.5:
                print("  → Low entropy: model committed to specific edges (GOOD)")
            else:
                print("  → High entropy: topology still exploring (needs more epochs)")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
