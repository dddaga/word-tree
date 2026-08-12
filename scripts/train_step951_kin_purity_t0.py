"""Step 951: K_in Signal Purity Sweep T0 + Observational Haki (20ep, 50% data).

SIGNAL PURITY HYPOTHESIS
========================
The seed step encodes input as:
  Z[b, i, 0]    = sum(x[b, conn_in[i, :]])      ← 1 scalar: aggregated input signal
  Z[b, i, 1:D]  = precomputed Fourier position   ← D-1 dims: spatial identity

The entire input signal lives in ONE dimension per node. K_in features are
mixed into that single scalar. Routing's job is to spread this 1D signal
across the D-dimensional sphere via W_pos geometry over K_iter steps.

SIGNAL PURITY QUESTION: does reducing K_in improve the quality of that scalar?

High K_in (25+):
  + Aggregation reduces noise (law of large numbers)
  − Each individual feature contributes only 1/K_in of the signal
  − Mixed features from spatially distant receptive field
  − Low coverage means same feature recurs across nodes → redundant

Low K_in (5-10):
  + Each node is a sharper, more specialised local detector
  + More "pure" signal (fewer mixed sources)
  − Risk of missing important features entirely
  − Noisier individual estimates

The routing loop must compensate for whatever the seed lacks. If K_in=5
gives a purer seed → routing can amplify it. If K_in=60 smears too much →
routing can't recover class boundaries.

OBSERVATIONAL HAKI
==================
Per-diagnostic-epoch metrics that tell us WHERE the information lives:

  seed_fisher   : Fisher ratio at Z^(0)  — class separation before routing
  final_fisher  : Fisher ratio at Z^(K_iter) — after routing
  routing_gain  : final_fisher - seed_fisher (what routing adds)
  pr            : Participation Ratio of Z^(final) — effective D dims used
  dead_frac     : fraction of nodes with ||Z_final||_mean < 0.05 (collapse)
  coverage_pct  : N × K_in / N_in × 100%

If seed_fisher is LOW and routing_gain is HIGH → routing compensates well.
If seed_fisher is HIGH and routing_gain is LOW → routing saturates quickly.
Optimal K_in: maximise final_fisher; routing_gain is a diagnostic, not a target.

CONFIGS (N=2048, D=16, ΔW-proj, 20ep/50%)
  Coverage = N × K_in / N_in × 100%  (N_in=25088)
  Ref:   K_in=25  coverage=204%  ← current default
  A:     K_in=5   coverage=41%   ← extreme purity, sparse
  B:     K_in=10  coverage=82%   ← near-100% boundary
  C:     K_in=15  coverage=122%  ← sub-canonical
  D:     K_in=40  coverage=327%  ← moderate mixing
  E:     K_in=60  coverage=490%  ← heavy mixing

All configs: D=16, same W_pos/theta param budget = 34,976. K_in is topology
only — no extra learnable parameters.

DIAGNOSTICS: computed at epoch 1, 5, 10, 20 (4 checkpoints per config).
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_kin5,B_kin10,C_kin15,D_kin40,E_kin60")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS     = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5

# Diagnostic epochs: check haki at these 1-indexed epochs
HAKI_EPOCHS = {1, 5, 10, 20}

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step951_kin_purity_t0_seed{SEED}__{SLOT}.json"

# Context: step887 canonical ΔW-proj K_in=25 N=2048 = 96.38%
DW_PROJ_REF = 0.9638

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_SPEC = {
    # K_in value
    "Ref":   25,
    "A_kin5":  5,
    "B_kin10": 10,
    "C_kin15": 15,
    "D_kin40": 40,
    "E_kin60": 60,
}


def coverage_pct(K_in: int) -> float:
    return N * K_in / N_IN * 100.0


# ─────────────────────────────────────────────────────────────────────────────
def _dw_proj(W_pos, conn_hh):
    """Compute normalised ΔW direction vectors. [1, N, K_hh, D]."""
    Wh = W_pos[:N]
    return F.normalize(Wh.unsqueeze(1) - Wh[conn_hh], dim=-1).unsqueeze(0)


class SGNNET_KinVar(nn.Module):
    """ΔW-proj routing with variable K_in.

    Exposes _forward_full(x) → (logits, Z_seed_pooled, Z_final)
    for observational haki diagnostics.
    """

    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def _routing_loop(self, Z):
        """Run K_iter ΔW-proj routing steps. Returns Z_final."""
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos[:N], conn_hh)            # [1, N, K_hh, D]
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]                    # [B, N, K_hh, D]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)           # [B, N, D]
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return Z

    def forward(self, x):
        Z = self.m.base._seed(x)
        Z_final = self._routing_loop(Z)
        return self.m.base._readout(Z_final)

    def _forward_full(self, x):
        """Returns (logits, Z_seed [B,N,D], Z_final [B,N,D]) for haki diagnostics."""
        Z_seed  = self.m.base._seed(x)                      # [B, N, D]
        Z_final = self._routing_loop(Z_seed.clone())
        logits  = self.m.base._readout(Z_final)
        return logits, Z_seed.detach(), Z_final.detach()


# ─────────────────────────────────────────────────────────────────────────────
def _fisher_ratio(Z: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
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


def _participation_ratio(Z: torch.Tensor) -> float:
    """Effective number of dimensions used. PR=1 → 1 dim; PR=D → uniform."""
    Zc = Z - Z.mean(0)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).abs()
    return float(ev.sum().pow(2) / (ev.pow(2).sum() + 1e-8))


@torch.no_grad()
def compute_haki(model: SGNNET_KinVar, val_loader, device, n_classes=10, max_batches=12):
    """Observational haki: compute signal purity metrics on validation set.

    max_batches=12 @ batch=128 = 1536 samples — fast, representative.
    Returns dict with seed_fisher, final_fisher, routing_gain, pr, dead_frac.
    """
    model.eval()
    Z_seeds, Z_finals, labels_all = [], [], []
    dead_count = torch.zeros(N, device="cpu")
    total = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x, _, y = batch  # (features, soft_labels, hard_labels)
        x = x.to(device)
        _, Z_seed, Z_final = model._forward_full(x)         # [B, N, D] each

        # Pool over nodes for Fisher / PR (class-level representation)
        Z_seeds.append(Z_seed.mean(dim=1).cpu())            # [B, D]
        Z_finals.append(Z_final.mean(dim=1).cpu())          # [B, D]
        labels_all.append(y)

        # Dead node count: nodes with near-zero mean activation across batch
        norms = Z_final.norm(dim=-1)                        # [B, N]
        dead_count += (norms < 0.05).float().sum(dim=0).cpu()
        total += x.shape[0]

    Zs = torch.cat(Z_seeds,  0)
    Zf = torch.cat(Z_finals, 0)
    labs = torch.cat(labels_all, 0)

    seed_f  = _fisher_ratio(Zs, labs, n_classes)
    final_f = _fisher_ratio(Zf, labs, n_classes)
    pr      = _participation_ratio(Zf)
    dead    = float((dead_count / total).mean())

    return {
        "seed_fisher":   round(seed_f,  4),
        "final_fisher":  round(final_f, 4),
        "routing_gain":  round(final_f - seed_f, 4),
        "pr":            round(pr, 3),
        "dead_frac":     round(dead, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
def make_model(K_in: int) -> SGNNET_KinVar:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_KinVar(resonant)


# ─────────────────────────────────────────────────────────────────────────────
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
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*72}")
    print(f"step951 — K_in Signal Purity Sweep T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}  (params fixed = 34,976)")
    print(f"  Seed: Z[i,0]=sum(x[conn_in[i]])  Z[i,1:D]=Fourier(pos)")
    print(f"  Signal lives in 1 scalar. Routing spreads it across {D}D.")
    print(f"  Haki checkpoints: epochs {sorted(HAKI_EPOCHS)}")
    print(f"  Context: step887 K_in=25 canonical = {DW_PROJ_REF:.4f}")
    print()
    print(f"  {'Config':<10} {'K_in':>5}  {'Coverage':>9}  {'K_in/D':>7}")
    for k, K_in in CONFIG_SPEC.items():
        cov = coverage_pct(K_in)
        print(f"  {k:<10} {K_in:>5}  {cov:>8.1f}%  {K_in/D:>7.2f}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        K_in = CONFIG_SPEC[key]
        model = make_model(K_in).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cov = coverage_pct(K_in)

        print(f"{'─'*72}")
        print(f"{key}: K_in={K_in}  coverage={cov:.1f}%  K_in/D={K_in/D:.2f}  params={n_p:,}")

        kw   = trainer_kwargs(N, n_epochs=EPOCHS)
        t0   = time.time()
        haki_log = {}

        def log_fn(m):
            ep = m["epoch"] + 1
            print(f"  e{ep:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True, end="")
            if ep in HAKI_EPOCHS:
                haki = compute_haki(model, va, DEVICE)
                haki_log[ep] = haki
                print(f"  | seed_F={haki['seed_fisher']:.3f}  final_F={haki['final_fisher']:.3f}"
                      f"  gain={haki['routing_gain']:+.3f}  PR={haki['pr']:.2f}"
                      f"  dead={haki['dead_frac']:.3f}", end="")
                model.train()
            print(flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h  = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best   = max(top1h)
        best_ep = int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta  = best - (ref_acc if ref_acc is not None else DW_PROJ_REF)

        # Final haki snapshot if not already computed
        if EPOCHS not in haki_log:
            haki_final = compute_haki(model, va, DEVICE)
            haki_log[EPOCHS] = haki_final

        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")
        print(f"     Final haki: {haki_log.get(EPOCHS, haki_log.get(max(haki_log)))}")

        results[key] = {
            "K_in": K_in, "coverage_pct": round(cov, 1),
            "ratio_kin_D": round(K_in / D, 3),
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
            "haki": haki_log,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"STEP 951 SUMMARY — K_in Signal Purity Sweep")
    print(f"  (Context: step887 K_in=25 T2 = {DW_PROJ_REF:.4f})")
    print(f"{'='*72}")
    print(f"  {'Config':<10} {'K_in':>5}  {'Cov%':>6}  {'params':>8}  "
          f"{'best':>7}  {'Δ':>8}  {'seed_F':>7}  {'finalF':>7}  {'gain':>7}  "
          f"{'PR':>5}  {'dead':>6}  verdict")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        hf = r["haki"].get(str(EPOCHS)) or r["haki"].get(EPOCHS) or {}
        if isinstance(list(r["haki"].keys())[0], int):
            last_haki = r["haki"][max(r["haki"].keys())]
        else:
            last_haki = r["haki"][max(int(k) for k in r["haki"].keys())]
        sf = last_haki.get("seed_fisher", 0)
        ff = last_haki.get("final_fisher", 0)
        gn = last_haki.get("routing_gain", 0)
        pr = last_haki.get("pr", 0)
        df = last_haki.get("dead_frac", 0)
        print(f"  {k:<10} {r['K_in']:>5}  {r['coverage_pct']:>5.1f}%  {r['n_params']:>8,}  "
              f"{r['best']:>7.4f}  {d*100:>+7.2f}pp  "
              f"{sf:>7.3f}  {ff:>7.3f}  {gn:>+7.3f}  "
              f"{pr:>5.2f}  {df:>6.4f}  {v}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
