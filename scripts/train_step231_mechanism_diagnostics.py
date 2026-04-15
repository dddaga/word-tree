"""Step 231: Mechanism diagnostics — test 5 hypotheses about HOW SGNNET works.

MOTIVATION
==========
We have strong empirical results but weak mechanistic understanding.
This script trains a standard model then probes it to test:

  H1: W_pos learns class-specific directions (different classes activate different neurons)
  H2: AH forces connected neurons apart (cos(W_pos[i], W_pos[j]) stays low for connected pairs)
  H3: Iterative routing progressively refines representation (accuracy increases with K_iter)
  H4: Distributed code is input-dependent despite static topology (different images activate different neurons)
  H5: W_pos learning matters (frozen random W_pos should perform worse)

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep)
  A : Standard SGNNET (train normally, then diagnose)
  B : Frozen W_pos (random init, never updated — tests H5)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step231_mechanism_diagnostics.json"


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


# ── Diagnostic: intercept Z at each K_iter step ──────────────────────

class DiagnosticWrapper(nn.Module):
    """Wraps SGNNET_AntiHebbian to capture Z at each routing step."""

    def __init__(self, ah_model):
        super().__init__()
        self.ah = ah_model
        self.step_Zs = []          # populated during diagnostic forward
        self.capture_mode = False   # only capture when explicitly enabled

    @property
    def W_pos(self):   return self.ah.W_pos
    @property
    def W_phase(self): return self.ah.W_phase
    def tick_epoch(self): self.ah.tick_epoch()

    def forward(self, x):
        if not self.capture_mode:
            return self.ah(x)
        return self._forward_capture(x)

    def _forward_capture(self, x):
        """Full forward with Z captured at each K_iter step."""
        m = self.ah.m  # SGNNET_Resonant
        Z = m.base._seed(x)
        theta_pos = m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = m.base.conn_hh
        N_h = m.base.N_hidden

        # AH suppression (static)
        W_n = F.normalize(self.W_pos[:N_h], dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.ah.alpha_ahebb * pos_sim.clamp(min=0)
                  ).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        self.step_Zs = [Z.detach().clone()]

        for _ in range(m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            self.step_Zs.append(Z.detach().clone())

        return m.base._readout(Z)


# ── H1: Class-specific neuron activation ─────────────────────────────

def test_h1_class_specificity(model, val_loader, device, top_k=50):
    """For each class, find the top-K neurons by mean activation magnitude.
    Measure overlap between classes (low overlap = class-specific)."""
    model.eval()
    model.capture_mode = True

    class_activations = defaultdict(list)  # class -> list of [N] activation magnitudes

    with torch.no_grad():
        for x, y_soft, y_hard in val_loader:
            x = x.to(device)
            labels = y_soft.argmax(dim=-1)
            _ = model(x)
            Z_final = model.step_Zs[-1]  # [B, N, D]
            magnitudes = Z_final.norm(dim=-1)  # [B, N] — all 1.0 after normalize!

            # Use alignment with output class directions instead
            W_out = F.normalize(model.W_pos[N:], dim=-1)  # [10, D]
            alignment = torch.einsum("bnd,cd->bnc", Z_final, W_out)  # [B, N, 10]

            for c in range(N_OUT):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_activations[c].append(alignment[mask].mean(dim=0))  # [N, 10]

    # For each class, get top-K neurons by alignment with THAT class
    class_topk = {}
    for c in range(N_OUT):
        if class_activations[c]:
            avg = torch.stack(class_activations[c]).mean(dim=0)  # [N, 10]
            topk_neurons = avg[:, c].topk(top_k).indices.cpu().tolist()
            class_topk[c] = set(topk_neurons)

    # Pairwise overlap
    overlaps = []
    for c1 in range(N_OUT):
        for c2 in range(c1 + 1, N_OUT):
            if c1 in class_topk and c2 in class_topk:
                overlap = len(class_topk[c1] & class_topk[c2]) / top_k
                overlaps.append(overlap)

    model.capture_mode = False
    avg_overlap = float(np.mean(overlaps)) if overlaps else -1
    return {"avg_pairwise_overlap_top50": round(avg_overlap, 4),
            "interpretation": "low overlap = class-specific neurons (H1 supported)"}


# ── H2: AH forces connected neurons apart ────────────────────────────

def test_h2_connected_diversity(model, device):
    """Compare cos(W_pos[i], W_pos[j]) for connected pairs vs random pairs."""
    with torch.no_grad():
        W = F.normalize(model.W_pos[:N].to(device), dim=-1)
        conn_hh = model.ah.m.base.conn_hh  # [N, K_hh]

        # Connected pairs
        connected_cos = (W.unsqueeze(1) * W[conn_hh]).sum(-1)  # [N, K_hh]
        avg_connected = connected_cos.mean().item()

        # Random pairs (sample 4096 random pairs)
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, N, size=4096)
        idx_b = rng.integers(0, N, size=4096)
        random_cos = (W[idx_a] * W[idx_b]).sum(-1)
        avg_random = random_cos.mean().item()

    return {"avg_cos_connected": round(avg_connected, 4),
            "avg_cos_random": round(avg_random, 4),
            "delta": round(avg_connected - avg_random, 4),
            "interpretation": "connected < random = AH forced diversity (H2 supported)"}


# ── H3: Accuracy improves with K_iter steps ──────────────────────────

def test_h3_iterative_refinement(model, val_loader, device):
    """Measure classification accuracy using Z at each routing step."""
    model.eval()
    model.capture_mode = True

    step_correct = defaultdict(int)
    total = 0

    with torch.no_grad():
        W_out = F.normalize(model.W_pos[N:], dim=-1)  # [10, D]

        for x, y_soft, y_hard in val_loader:
            x = x.to(device)
            labels = y_soft.argmax(dim=-1)
            _ = model(x)

            B = x.shape[0]
            total += B

            for step_idx, Z_step in enumerate(model.step_Zs):
                # Quick readout using same C_ho + W_out
                C_ho = model.ah.m.base.C_ho_mask.float()
                A_out = torch.einsum("bhd,ho->bod", Z_step, C_ho)
                logits = (A_out * W_out.unsqueeze(0)).sum(dim=-1)
                preds = logits.argmax(dim=-1).cpu()
                step_correct[step_idx] += (preds == labels).sum().item()

    model.capture_mode = False
    step_acc = {f"step_{k}": round(v / total, 4) for k, v in sorted(step_correct.items())}
    step_acc["interpretation"] = "monotonic increase = routing refines representation (H3 supported)"
    return step_acc


# ── H4: Input-dependent activation patterns ──────────────────────────

def test_h4_input_dependence(model, val_loader, device, top_k=100):
    """For pairs of images from different classes, measure overlap in top-K active neurons."""
    model.eval()
    model.capture_mode = True

    class_samples = {}  # class -> Z_final for one batch

    with torch.no_grad():
        for x, y_soft, y_hard in val_loader:
            x = x.to(device)
            labels = y_soft.argmax(dim=-1)
            _ = model(x)
            Z_final = model.step_Zs[-1]  # [B, N, D]

            for c in range(N_OUT):
                mask = (labels == c)
                if mask.sum() > 0 and c not in class_samples:
                    # Take first sample of this class
                    idx = mask.nonzero(as_tuple=True)[0][0]
                    class_samples[c] = Z_final[idx]  # [N, D]

            if len(class_samples) == N_OUT:
                break

    # For each class, find top-K neurons by activation norm
    # (all norms are ~1 after normalize, so use alignment with that class's output direction)
    W_out = F.normalize(model.W_pos[N:], dim=-1)

    class_topk = {}
    for c, Z in class_samples.items():
        alignment = (Z * W_out[c].unsqueeze(0)).sum(-1)  # [N]
        class_topk[c] = set(alignment.topk(top_k).indices.cpu().tolist())

    # Cross-class overlap
    overlaps = []
    for c1 in range(N_OUT):
        for c2 in range(c1 + 1, N_OUT):
            if c1 in class_topk and c2 in class_topk:
                overlap = len(class_topk[c1] & class_topk[c2]) / top_k
                overlaps.append(overlap)

    model.capture_mode = False
    avg_overlap = float(np.mean(overlaps)) if overlaps else -1
    return {"avg_cross_class_overlap_top100": round(avg_overlap, 4),
            "interpretation": "low overlap = input-dependent activation (H4 supported)"}


# ── H5: Frozen W_pos test (via training config B) ────────────────────

def freeze_hidden_wpos(model):
    """Register hook to zero gradient for hidden W_pos rows."""
    W = model.W_pos
    # Can't use register_hook on non-leaf. Instead, mask gradient in optimizer step.
    return model


# ── Main ──────────────────────────────────────────────────────────────

def train_and_diagnose(key, model, tr, va, freeze_wpos=False):
    print(f"\n{'─'*60}")
    print(f"Config {key}: {'Frozen W_pos' if freeze_wpos else 'Standard'}")
    print(f"{'─'*60}")

    model = model.to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_p:,}")

    kw = trainer_kwargs(N, n_epochs=EPOCHS)

    if freeze_wpos:
        # Exclude hidden W_pos from optimizer by giving it lr=0
        # Find the actual parameter
        wpos_param = None
        other_params = []
        for name, p in model.named_parameters():
            if 'W_pos' in name:
                wpos_param = p
            else:
                other_params.append(p)

        if wpos_param is not None:
            # Custom optimizer: W_pos hidden rows get lr=0 via gradient masking
            # We'll use a hook to zero the hidden portion of gradient
            def zero_hidden_grad(grad):
                mask = grad.clone()
                mask[:N] = 0  # zero hidden rows, keep output rows
                return mask
            wpos_param.register_hook(zero_hidden_grad)
            frozen_params = N * D
            print(f"  Frozen {frozen_params:,} hidden W_pos params (output W_pos still learns)")

    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)

    t0 = time.time()
    def _log(m):
        ep = m["epoch"] + 1
        if ep % 5 == 0:
            print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
    history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
    elapsed = time.time() - t0

    top1h = [round(h.get("val_top1", 0.), 4) for h in history]
    best = max(top1h); bep = int(np.argmax(top1h)) + 1
    print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    return model, {"top1_best": best, "best_epoch": bep, "top1_history": top1h,
                   "elapsed_s": round(elapsed, 1), "n_params": n_p}


def main():
    print(f"\n{'='*70}")
    print(f"Step 231 — Mechanism Diagnostics: How does SGNNET actually work?")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}

    # ── Config A: Standard training + full diagnostics ────────────────
    ah_model_a = build_model()
    model_a = DiagnosticWrapper(ah_model_a)
    model_a, res_a = train_and_diagnose("A", model_a, tr, va, freeze_wpos=False)

    print(f"\n  Running diagnostics on trained model A...")

    print(f"  H1: Class-specific neuron activation...")
    h1 = test_h1_class_specificity(model_a, va, DEVICE)
    print(f"    → avg overlap = {h1['avg_pairwise_overlap_top50']}")

    print(f"  H2: Connected neuron diversity...")
    h2 = test_h2_connected_diversity(model_a, DEVICE)
    print(f"    → connected cos = {h2['avg_cos_connected']}, random cos = {h2['avg_cos_random']}")

    print(f"  H3: Per-step accuracy...")
    h3 = test_h3_iterative_refinement(model_a, va, DEVICE)
    for k, v in sorted(h3.items()):
        if k.startswith("step_"):
            print(f"    → {k}: {v}")

    print(f"  H4: Input-dependent activation patterns...")
    h4 = test_h4_input_dependence(model_a, va, DEVICE)
    print(f"    → avg cross-class overlap = {h4['avg_cross_class_overlap_top100']}")

    res_a["diagnostics"] = {"H1_class_specificity": h1, "H2_connected_diversity": h2,
                            "H3_iterative_refinement": h3, "H4_input_dependence": h4}
    results["A"] = res_a

    # ── Config B: Frozen W_pos ────────────────────────────────────────
    ah_model_b = build_model()
    model_b = DiagnosticWrapper(ah_model_b)
    model_b, res_b = train_and_diagnose("B", model_b, tr, va, freeze_wpos=True)

    print(f"\n  Running diagnostics on frozen-W_pos model B...")

    print(f"  H2 (frozen): Connected neuron diversity...")
    h2_frozen = test_h2_connected_diversity(model_b, DEVICE)
    print(f"    → connected cos = {h2_frozen['avg_cos_connected']}, random cos = {h2_frozen['avg_cos_random']}")

    print(f"  H3 (frozen): Per-step accuracy...")
    h3_frozen = test_h3_iterative_refinement(model_b, va, DEVICE)
    for k, v in sorted(h3_frozen.items()):
        if k.startswith("step_"):
            print(f"    → {k}: {v}")

    res_b["diagnostics"] = {"H2_connected_diversity_frozen": h2_frozen,
                            "H3_iterative_refinement_frozen": h3_frozen}
    results["B"] = res_b

    # ── Summary ───────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print("STEP 231 SUMMARY — Mechanism Diagnostics")
    print(f"{'='*70}")

    print(f"\n  Config A (standard):  best={res_a['top1_best']:.4f}")
    print(f"  Config B (frozen W):  best={res_b['top1_best']:.4f}")
    delta = res_a['top1_best'] - res_b['top1_best']
    print(f"  H5 delta (A-B): {delta:+.4f} — {'W_pos learning matters' if delta > 0.02 else 'W_pos learning barely matters!'}")

    print(f"\n  H1 class specificity: overlap={h1['avg_pairwise_overlap_top50']}")
    print(f"      (0.0=perfectly class-specific, 1.0=identical for all classes)")
    print(f"  H2 diversity: connected={h2['avg_cos_connected']:.4f} vs random={h2['avg_cos_random']:.4f}")
    print(f"      (connected < random = AH forces diversity)")
    print(f"  H3 refinement: step_0={h3.get('step_0','?')} → step_{K_ITER}={h3.get(f'step_{K_ITER}','?')}")
    print(f"      (increasing = routing helps)")
    print(f"  H4 input-dependence: overlap={h4['avg_cross_class_overlap_top100']}")
    print(f"      (low = different inputs activate different neurons)")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
