"""Training diagnostics for SGNNET — richer insight beyond loss/accuracy.

Usage:
    diag = TrainingDiagnostics(model, device, log_every=5)
    for ep in range(epochs):
        # ... training ...
        diag.log_epoch(ep, model, val_loader, optimizer)

All metrics are cheap (no O(N²) computations). Run every log_every epochs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Any


class TrainingDiagnostics:
    """Lightweight diagnostics for SGNNET training health."""

    def __init__(self, model, device, log_every: int = 5):
        self.device = device
        self.log_every = log_every
        self.history: list[dict] = []

    def should_log(self, epoch: int) -> bool:
        return epoch % self.log_every == 0 or epoch <= 2

    @torch.no_grad()
    def compute_metrics(self, model, val_loader, optimizer) -> Dict[str, Any]:
        """Compute all diagnostic metrics. Call after validation."""
        model.eval()
        metrics = {}

        # ── 1. W_pos diversity (positional collapse detection) ──
        W_pos = None
        for name, param in model.named_parameters():
            if "W_pos" in name or "W_phase" in name:
                W_pos = param.data
                break

        if W_pos is not None:
            N_h = min(W_pos.shape[0], 4096)  # cap for speed
            W_n = F.normalize(W_pos[:N_h], dim=-1)
            # Sample 500 random pairs instead of full O(N²)
            n_pairs = min(500, N_h * (N_h - 1) // 2)
            idx_a = torch.randint(0, N_h, (n_pairs,), device=W_pos.device)
            idx_b = torch.randint(0, N_h, (n_pairs,), device=W_pos.device)
            cos_sim = (W_n[idx_a] * W_n[idx_b]).sum(dim=-1)
            metrics["wpos_cos_mean"] = round(cos_sim.mean().item(), 4)
            metrics["wpos_cos_std"] = round(cos_sim.std().item(), 4)
            metrics["wpos_norm_mean"] = round(W_pos[:N_h].norm(dim=-1).mean().item(), 4)

        # ── 2. Activation statistics (from one val batch) ──
        batch = next(iter(val_loader))
        x = batch[0][:32].to(self.device)  # small batch for speed

        # Capture Z [B, N, D] before readout by hooking _readout
        Z_captured = {}

        def _hook_readout(module, input, output):
            # _readout receives Z as first arg: [B, N, D]
            if len(input) > 0 and isinstance(input[0], torch.Tensor) and input[0].dim() == 3:
                Z_captured["Z"] = input[0].detach()

        # Find the SmallWorld base that has _readout
        sw_model = _find_smallworld(model)
        hook = None
        if sw_model is not None and hasattr(sw_model, '_readout'):
            hook = sw_model._readout.__func__  # can't hook non-module methods
            # Instead, hook the forward of SmallWorld itself and grab Z
            orig_readout = sw_model._readout
            def _patched_readout(Z_arg):
                Z_captured["Z"] = Z_arg.detach()
                return orig_readout(Z_arg)
            sw_model._readout = _patched_readout
            _ = model(x)
            sw_model._readout = orig_readout  # restore
        else:
            _ = model(x)
            # Try _last_Z attributes for custom models
            for attr_path in ["_last_Z", "base._last_Z", "m._last_Z",
                              "base.m._last_Z"]:
                obj = model
                try:
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    if obj is not None and isinstance(obj, torch.Tensor) and obj.dim() == 3:
                        Z_captured["Z"] = obj.detach()
                        break
                except AttributeError:
                    continue

        if "Z" in Z_captured:
            Z = Z_captured["Z"]  # [B, N, D]
            B, N, D = Z.shape

            # Effective rank via SVD (on batch-mean, [N, D] where D is small)
            Z_mean = Z.mean(dim=0)  # [N, D]
            try:
                gram = Z_mean.T @ Z_mean  # [D, D]
                eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
                eigvals_norm = eigvals / (eigvals.sum() + 1e-8)
                # Shannon entropy → effective rank = exp(entropy)
                entropy = -(eigvals_norm * (eigvals_norm + 1e-8).log()).sum()
                eff_rank = entropy.exp().item()
                metrics["effective_rank"] = round(eff_rank, 2)
                metrics["top1_eigval_frac"] = round(eigvals[-1].item() / (eigvals.sum().item() + 1e-8), 4)
            except Exception:
                pass

            # Neuron utilization: fraction with mean |activation| > 0.01
            neuron_energy = Z.abs().mean(dim=(0, 2))  # [N]
            metrics["neuron_util_pct"] = round(
                (neuron_energy > 0.01).float().mean().item() * 100, 1)
            metrics["neuron_energy_std"] = round(neuron_energy.std().item(), 4)

            # Activation magnitude stats
            metrics["Z_abs_mean"] = round(Z.abs().mean().item(), 4)
            metrics["Z_abs_std"] = round(Z.abs().std().item(), 4)

        # ── 3. Gradient norms per parameter group ──
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Group by parameter type
                if "W_pos" in name or "W_phase" in name:
                    key = "grad_wpos"
                elif "theta" in name:
                    key = "grad_theta"
                elif "fc_out" in name or "fc" in name:
                    key = "grad_fc"
                else:
                    key = "grad_other"

                norm = param.grad.norm().item()
                if key not in grad_norms:
                    grad_norms[key] = []
                grad_norms[key].append(norm)

        for key, norms in grad_norms.items():
            metrics[key] = round(sum(norms) / len(norms), 6)

        # ── 4. Feature separability (inter vs intra class distance) ──
        feats_for_sep = None
        if "Z" in Z_captured:
            feats_for_sep = F.normalize(Z.mean(dim=1), dim=-1)
        elif "pooled" in Z_captured:
            feats_for_sep = F.normalize(Z_captured["pooled"], dim=-1)

        if feats_for_sep is not None:
            labels = batch[2][:32].to(self.device)
            feats = feats_for_sep
            unique_labels = labels.unique()

            if len(unique_labels) >= 2:
                intra_dists = []
                centroids = []
                for lab in unique_labels:
                    mask = labels == lab
                    if mask.sum() >= 2:
                        class_feats = feats[mask]
                        centroid = class_feats.mean(dim=0)
                        centroids.append(centroid)
                        dists = 1 - (class_feats @ centroid).clamp(-1, 1)
                        intra_dists.append(dists.mean().item())

                if len(centroids) >= 2:
                    centroids = torch.stack(centroids)
                    centroids = F.normalize(centroids, dim=-1)
                    inter_sim = centroids @ centroids.T
                    # Mean off-diagonal
                    n_c = centroids.shape[0]
                    mask = ~torch.eye(n_c, device=inter_sim.device, dtype=torch.bool)
                    inter_dist = (1 - inter_sim[mask]).mean().item()
                    intra_dist = sum(intra_dists) / len(intra_dists)

                    metrics["inter_class_dist"] = round(inter_dist, 4)
                    metrics["intra_class_dist"] = round(intra_dist, 4)
                    if intra_dist > 0:
                        metrics["separability_ratio"] = round(
                            inter_dist / (intra_dist + 1e-8), 2)

        # ── 5. Learning rate ──
        if optimizer is not None:
            metrics["lr"] = optimizer.param_groups[0]["lr"]

        return metrics

    def log_epoch(self, epoch: int, model, val_loader, optimizer=None,
                  extra: dict | None = None) -> dict | None:
        """Log diagnostics for this epoch. Returns metrics dict or None if skipped."""
        if not self.should_log(epoch):
            return None

        metrics = self.compute_metrics(model, val_loader, optimizer)
        if extra:
            metrics.update(extra)
        metrics["epoch"] = epoch
        self.history.append(metrics)

        # Print compact summary
        parts = [f"  [diag] ep={epoch:3d}"]
        if "effective_rank" in metrics:
            parts.append(f"eff_rank={metrics['effective_rank']:.1f}/{_get_D(model)}")
        if "neuron_util_pct" in metrics:
            parts.append(f"neuron_util={metrics['neuron_util_pct']:.0f}%")
        if "wpos_cos_mean" in metrics:
            parts.append(f"wpos_cos={metrics['wpos_cos_mean']:.3f}")
        if "separability_ratio" in metrics:
            parts.append(f"sep_ratio={metrics['separability_ratio']:.1f}")
        if "grad_wpos" in metrics:
            parts.append(f"∇wpos={metrics['grad_wpos']:.4f}")
        if "grad_theta" in metrics:
            parts.append(f"∇θ={metrics['grad_theta']:.4f}")

        print("  ".join(parts))
        return metrics


def _find_smallworld(model):
    """Find the SmallWorld base model in potentially nested wrappers."""
    for attr_path in ["", "base", "m.base", "m", "base.base",
                      "base.m.base", "base.m"]:
        obj = model
        try:
            if attr_path:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
            if hasattr(obj, '_readout') and hasattr(obj, 'N_hidden'):
                return obj
        except AttributeError:
            continue
    return None


def _find_fc_out(model) -> torch.nn.Module | None:
    """Find the fc_out / readout layer in potentially nested model."""
    for attr_path in ["fc_out", "base.fc_out", "m.fc_out",
                      "m.base.fc_out", "m.base.base.fc_out",
                      "base.m.fc_out", "base.m.base.fc_out"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.Linear):
                return obj
        except AttributeError:
            continue
    return None


def _get_D(model) -> int:
    """Extract D from model."""
    for name, param in model.named_parameters():
        if "W_pos" in name or "W_phase" in name:
            return param.shape[-1]
    return 0


def format_diagnostics_summary(history: list[dict]) -> str:
    """Format diagnostics history as a compact summary string."""
    if not history:
        return "No diagnostics recorded."

    lines = []
    first = history[0]
    last = history[-1]

    lines.append("Diagnostics Summary:")
    if "effective_rank" in first and "effective_rank" in last:
        lines.append(f"  Effective rank: {first['effective_rank']:.1f} → {last['effective_rank']:.1f}")
    if "neuron_util_pct" in first and "neuron_util_pct" in last:
        lines.append(f"  Neuron util: {first['neuron_util_pct']:.0f}% → {last['neuron_util_pct']:.0f}%")
    if "wpos_cos_mean" in first and "wpos_cos_mean" in last:
        lines.append(f"  W_pos cos sim: {first['wpos_cos_mean']:.3f} → {last['wpos_cos_mean']:.3f}")
    if "separability_ratio" in first and "separability_ratio" in last:
        lines.append(f"  Separability: {first['separability_ratio']:.1f} → {last['separability_ratio']:.1f}")
    if "grad_wpos" in first and "grad_wpos" in last:
        lines.append(f"  ∇W_pos: {first['grad_wpos']:.4f} → {last['grad_wpos']:.4f}")

    return "\n".join(lines)
