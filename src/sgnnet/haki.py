"""HAKI — Hidden Activation & Knowledge Inspector.

Standalone diagnostic module for SGNNET models.
Computes mechanistic metrics that guide architecture decisions and serve
as a tooling contribution in Paper 1.

Usage:
    from src.sgnnet.haki import HAKI

    haki = HAKI(model, val_loader, device, n_classes=10)
    metrics = haki.compute()  # full metric suite
    summary = haki.summary(metrics)  # one-line printable

Metrics returned:
    # Signal purity (Fisher ratio: between/within class separation)
    seed_fisher   — class separability in Z_seed (before routing)
    final_fisher  — class separability in Z_final (after routing)
    routing_gain  — final_fisher - seed_fisher
    pr_seed       — participation ratio of Z_seed (effective dims before routing)
    pr            — participation ratio of Z_final (effective dims after routing)
    pr_gain       — pr - pr_seed (routing effect on dimensionality)

    # Routing structure
    routing_entropy  — mean entropy of |c_ij| attention weights across nodes
                        high = uniform routing; low = selective/sparse
    effective_k      — continuous effective neighbors: exp(H) where H = routing entropy
                        ranges [1, K_hh]; 1 = single-neighbor; K_hh = uniform
    routing_invariance — CKA similarity between Z_seed and Z_final pooled reps
                        high = routing preserves input structure; low = reorganizes

    # Dynamics
    convergence_deltas — list of mean |Z_t - Z_{t-1}| per routing step (length K_iter)
    convergence_ratio  — convergence_deltas[-1] / convergence_deltas[0]
                        small = fast convergence; near 1 = routing not settling

    # Capacity usage
    dead_frac        — fraction of nodes with near-zero activation (||Z||<0.05)
    node_utilization — fraction of nodes that are both alive AND high-variance
                        (||Z||≥0.05 AND std≥0.1); measures efficient capacity use

Motivation (adiabatic + finite precision):
    Routing operates in fp32/bf16 — a discrete optimization landscape.
    These metrics are early warnings: if routing_entropy is flat across training,
    the network is not differentiating neighbors despite having capacity.
    If convergence_ratio → 1.0, K_iter steps are doing nothing (already at fixed point
    before the loop begins, or never settling at all).
    If pr < 2.5 after training, routing has collapsed to a 1-2D submanifold —
    the effective routing dimensionality is far below D.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Model interface
# ─────────────────────────────────────────────────────────────────────────────

def _extract_routing_components(model):
    """Extract (base, resonant) from any known SGNNET wrapper hierarchy."""
    # Unwrap common wrappers (KinVar, Ref, etc. store model as .m)
    m = getattr(model, "m", model)
    # m is now SGNNET_Resonant (has .base)
    base     = getattr(m, "base", None)
    resonant = m
    if base is None:
        raise ValueError(
            "Model must have .base (SGNNET_SmallWorld) accessible at model.m.base or model.base"
        )
    return base, resonant


def forward_haki(model, x: torch.Tensor, K_iter: int, alpha_reflect: float = 0.5):
    """Run the ΔW-proj routing loop and return intermediate Z states.

    Works on any model with the standard SGNNET_SmallWorld .base interface.
    Returns (logits, Z_seed, Z_final, Z_per_step) where:
        Z_seed    — [B, N, D] activations before routing
        Z_final   — [B, N, D] activations after K_iter routing steps
        Z_per_step — list of K_iter tensors [B, N, D], one per routing step
    """
    base, resonant = _extract_routing_components(model)
    W_pos     = resonant.W_pos.detach()
    N         = base.N_hidden
    conn_hh   = base.conn_hh
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1).detach()
    W_h       = W_pos[:N]
    dw        = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

    with torch.no_grad():
        Z_seed = base._seed(x).detach()
        Z      = Z_seed.clone()
        Z_ref  = torch.zeros_like(Z)
        steps  = []

        for _ in range(K_iter):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = alpha_reflect * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            steps.append(Z.clone())

        Z_final = Z
        logits  = base._readout(Z_final)

    return logits, Z_seed, Z_final, steps


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions (importable independently)
# ─────────────────────────────────────────────────────────────────────────────

def fisher_ratio(Z: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    """Between-class / within-class variance ratio on mean-pooled Z [n_samples, D]."""
    grand = Z.mean(0)
    sb = sum(
        ((Z[labels == c].mean(0) - grand).pow(2).mean()
         * (labels == c).float().mean()).item()
        for c in range(n_classes)
        if (labels == c).any()
    )
    sw = sum(
        (Z[labels == c] - Z[labels == c].mean(0)).pow(2).mean().item()
        for c in range(n_classes)
        if (labels == c).any()
    ) / n_classes
    return float(sb / (sw + 1e-8))


def participation_ratio(Z: torch.Tensor) -> float:
    """Effective number of dimensions active in Z. PR=1 → collapsed; PR=D → uniform."""
    Zc  = Z - Z.mean(0)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    ev  = torch.linalg.eigvalsh(cov).abs()
    return float(ev.sum().pow(2) / (ev.pow(2).sum() + 1e-8))


def _cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two feature matrices [n, d]. Centered kernel alignment."""
    def _hsic(A, B):
        n  = A.shape[0]
        H  = torch.eye(n, device=A.device) - torch.ones(n, n, device=A.device) / n
        KA = A @ A.T
        KB = B @ B.T
        return float((H @ KA @ H * KB).sum() / ((n - 1) ** 2))
    return _hsic(X, Y) / math.sqrt(_hsic(X, X) * _hsic(Y, Y) + 1e-10)


def routing_entropy_metrics(
    model, x: torch.Tensor, K_iter: int, alpha_reflect: float = 0.5
) -> dict:
    """Compute per-step attention entropy and effective_k over a batch.

    Returns:
        per_step_entropy  — list of K_iter floats (mean entropy per routing step)
        per_step_eff_k    — list of K_iter floats (exp(H), continuous eff neighbors)
    """
    base, resonant = _extract_routing_components(model)
    N        = base.N_hidden
    conn_hh  = base.conn_hh
    K_hh     = conn_hh.shape[1]
    theta_pos = resonant.theta.abs().unsqueeze(0).unsqueeze(-1).detach()
    W_pos    = resonant.W_pos.detach()
    W_h      = W_pos[:N]
    dw       = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

    entropies = []
    eff_ks    = []

    with torch.no_grad():
        Z_seed = base._seed(x).detach()
        Z      = Z_seed.clone()
        Z_ref  = torch.zeros_like(Z)

        for _ in range(K_iter):
            Z_fwd  = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb   = Z_fwd[:, conn_hh, :]
            c_raw  = (Z_nb * dw).sum(dim=-1)              # [B, N, K_hh]
            c_abs  = c_raw.abs()                           # attention weights (unnorm)
            c_prob = c_abs / (c_abs.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

            # Entropy H = -sum p*log(p), max = log(K_hh)
            H = -(c_prob * (c_prob + 1e-9).log()).sum(dim=-1)  # [B, N]
            mean_H = float(H.mean())
            entropies.append(mean_H)
            eff_ks.append(math.exp(mean_H))

            # Advance Z
            Z_agg = (Z_nb * c_abs.unsqueeze(-1)).sum(dim=2)
            Z_ref = alpha_reflect * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

    return {"per_step_entropy": entropies, "per_step_eff_k": eff_ks}


# ─────────────────────────────────────────────────────────────────────────────
# Main interface
# ─────────────────────────────────────────────────────────────────────────────

class HAKI:
    """Hidden Activation & Knowledge Inspector.

    Args:
        model       — any SGNNET model with .m.base (SmallWorld) accessible
        val_loader  — validation DataLoader yielding (x, soft_labels, y) or (x, y)
        device      — torch.device
        n_classes   — number of output classes (default 10)
        max_batches — how many batches to collect (12 @ B=128 = 1536 samples)
        K_iter      — routing iterations (default 5)
        alpha_reflect — reflection coefficient (default 0.5)
    """

    def __init__(
        self,
        model,
        val_loader,
        device,
        n_classes:     int   = 10,
        max_batches:   int   = 12,
        K_iter:        int   = 5,
        alpha_reflect: float = 0.5,
    ):
        self.model         = model
        self.val_loader    = val_loader
        self.device        = device
        self.n_classes     = n_classes
        self.max_batches   = max_batches
        self.K_iter        = K_iter
        self.alpha_reflect = alpha_reflect

    @torch.no_grad()
    def compute(self) -> dict:
        """Full metric suite. Returns flat dict with all HAKI metrics."""
        model  = self.model
        device = self.device
        model.eval()

        base, resonant = _extract_routing_components(model)
        N = base.N_hidden

        Z_seeds_pool, Z_finals_pool = [], []
        labels_all   = []
        dead_count   = torch.zeros(N, device="cpu")
        highvar_alive = torch.zeros(N, device="cpu")
        total        = 0

        # Per-step convergence
        conv_deltas_accum = None  # will be list of floats after first batch
        n_conv_batches = 0

        # Entropy / effective_k (first batch only — expensive)
        entropy_done   = False
        entropy_metrics = {}

        for i, batch in enumerate(self.val_loader):
            if i >= self.max_batches:
                break

            # Support both (x, soft, y) and (x, y)
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(device)

            logits, Z_seed, Z_final, Z_steps = forward_haki(
                model, x, self.K_iter, self.alpha_reflect
            )

            # Pool over nodes for class-level Fisher/PR
            Z_seeds_pool.append(Z_seed.mean(dim=1).cpu())    # [B, D]
            Z_finals_pool.append(Z_final.mean(dim=1).cpu())  # [B, D]
            labels_all.append(y.cpu())

            # Dead + high-variance node tracking
            norms = Z_final.norm(dim=-1)                     # [B, N]
            stds  = Z_final.std(dim=-1)                      # [B, N]  std over D dims
            dead_count    += (norms < 0.05).float().sum(0).cpu()
            highvar_alive += ((norms >= 0.05) & (stds >= 0.1)).float().sum(0).cpu()
            total += x.shape[0]

            # Convergence deltas: mean per-node per-step displacement
            Z_prev = Z_seed
            deltas = []
            for Z_t in Z_steps:
                delta = float((Z_t - Z_prev).norm(dim=-1).mean())
                deltas.append(delta)
                Z_prev = Z_t
            if conv_deltas_accum is None:
                conv_deltas_accum = deltas
            else:
                conv_deltas_accum = [a + b for a, b in zip(conv_deltas_accum, deltas)]
            n_conv_batches += 1

            # Entropy — compute on first batch (representative, fast)
            if not entropy_done:
                entropy_metrics = routing_entropy_metrics(
                    model, x, self.K_iter, self.alpha_reflect
                )
                entropy_done = True

        Zs   = torch.cat(Z_seeds_pool,  0)
        Zf   = torch.cat(Z_finals_pool, 0)
        labs = torch.cat(labels_all,    0)

        # Fisher ratios
        sf  = fisher_ratio(Zs, labs, self.n_classes)
        ff  = fisher_ratio(Zf, labs, self.n_classes)

        # Participation ratios
        pr_s  = participation_ratio(Zs)
        pr_f  = participation_ratio(Zf)

        # CKA routing invariance
        inv = _cka_linear(Zs, Zf)

        # Convergence
        avg_deltas = [d / n_conv_batches for d in conv_deltas_accum]
        conv_ratio = avg_deltas[-1] / (avg_deltas[0] + 1e-8)

        # Node utilization
        dead_frac = float((dead_count / total).mean())
        util      = float((highvar_alive / total).mean())

        return {
            # Signal purity
            "seed_fisher":        round(sf, 4),
            "final_fisher":       round(ff, 4),
            "routing_gain":       round(ff - sf, 4),
            "pr_seed":            round(pr_s, 3),
            "pr":                 round(pr_f, 3),
            "pr_gain":            round(pr_f - pr_s, 3),
            # Routing structure
            "routing_entropy":    round(entropy_metrics["per_step_entropy"][-1], 4),
            "effective_k":        round(entropy_metrics["per_step_eff_k"][-1], 3),
            "routing_entropy_step1": round(entropy_metrics["per_step_entropy"][0], 4),
            "routing_invariance": round(inv, 4),
            # Dynamics
            "convergence_deltas": [round(d, 5) for d in avg_deltas],
            "convergence_ratio":  round(conv_ratio, 4),
            # Capacity
            "dead_frac":          round(dead_frac, 4),
            "node_utilization":   round(util, 4),
        }

    @staticmethod
    def summary(metrics: dict) -> str:
        """One-line summary of a HAKI metrics dict for inline printing."""
        return (
            f"seed_F={metrics['seed_fisher']:.3f}  "
            f"final_F={metrics['final_fisher']:.3f}  "
            f"gain={metrics['routing_gain']:+.3f}  "
            f"PR={metrics['pr_seed']:.1f}→{metrics['pr']:.1f}  "
            f"eff_k={metrics['effective_k']:.2f}  "
            f"inv={metrics['routing_invariance']:.3f}  "
            f"util={metrics['node_utilization']:.3f}  "
            f"dead={metrics['dead_frac']:.3f}  "
            f"conv={metrics['convergence_ratio']:.3f}"
        )

    @staticmethod
    def short_summary(metrics: dict) -> str:
        """Compact inline summary for per-epoch training logs."""
        return (
            f"seed_F={metrics['seed_fisher']:.3f}  "
            f"final_F={metrics['final_fisher']:.3f}  "
            f"gain={metrics['routing_gain']:+.3f}  "
            f"PR={metrics['pr']:.2f}  "
            f"eff_k={metrics['effective_k']:.2f}  "
            f"inv={metrics['routing_invariance']:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone diagnostic script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path

    ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(ROOT))

    from src.sgnnet.model_smallworld import SGNNET_SmallWorld
    from src.sgnnet.model_resonant   import SGNNET_Resonant
    from src.training.dataset        import make_loaders

    parser = argparse.ArgumentParser(description="HAKI standalone diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       default="data/store.h5")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_classes",  type=int, default=10)
    args = parser.parse_args()

    device = (
        torch.device("cuda")   if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    ) if args.device == "auto" else torch.device(args.device)

    # Canonical N=2048 D=16 architecture
    N=2048; D=16; K_HH=2; K_ITER=5; K_IN=25; N_IN=25088; N_OUT=10
    torch.manual_seed(args.seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    model = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    _, va = make_loaders(str(ROOT / args.data), batch_size=128, seed=args.seed,
                         pin_memory=(device.type == "cuda"))

    haki    = HAKI(model, va, device, n_classes=args.n_classes)
    metrics = haki.compute()

    print(json.dumps(metrics, indent=2))

    out = ROOT / "results" / f"haki_{Path(args.checkpoint).stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\n-> {out}")
