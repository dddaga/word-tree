"""Step 957: Matryoshka Representation Learning (MRL) on D=16 T0 (20ep, 50% data).

MOTIVATION:
  step952 PR analysis: Participation Ratio ≈ 2.3 after routing — only 2-3 effective
  dims used out of 16. 13+ dims near-dead.

  Kusupati et al. (NeurIPS 2022): MRL forces ALL nesting levels to be independently
  useful by adding auxiliary readout heads at sub-dimensions. Joint training loss:

    L = L_D16 + α * L_D8 + α² * L_D4 + α³ * L_D2

  Each L_Dk = CE(W_k @ Z.mean(dim=1)[:, :k], y) — mean-pool over N nodes,
  classify from first k dims.

  Forces representation to pack class info into early dims AND use more dims → should
  raise PR from ≈2.3 toward theoretical max 16.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:        standard ΔW-proj + single readout (step887 base) — via Trainer
  A_uniform:  D_subs=[2,4,8], α=1.0 (equal weight for all sub-dim scales)
  B_decay2:   D_subs=[2,4,8], α=0.5 (geometric: weights 1.0, 0.5, 0.25, 0.125)
  C_d48only:  D_subs=[4,8] only (skip d=2 — too few dims might be noise), α=1.0

DESIGN:
  - Ref uses Trainer (KL distillation, W_pos only).
  - MRL configs use a custom training loop: main forward produces Z [B,N,D] before
    readout. aux heads = nn.Linear(d_sub, N_OUT) applied to Z.mean(1)[:, :d_sub].
  - Main loss: KL(main_logits, soft_labels) — matches Trainer's task_loss.
  - Aux losses: CE(aux_head(z_sub), hard_labels) — CE is natural for sub-dim probes.
  - val_top1 always uses main readout only (standard).

PR TRACKING (Participation Ratio):
  Logged at epochs {1, 5, 10, 20} to verify MRL forces PR up from ≈2.3 baseline.
  PR = (sum|λ|)² / sum(λ²)  where λ = eigenvalues of covariance of Z.mean(1).

ADVANCE: ≥+0.5pp vs Ref → T1.
DW_REF = 0.9638  (step887 canonical)
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
parser.add_argument("--configs", default="Ref,A_uniform,B_decay2,C_d48only")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5

PR_LOG_EPOCHS = {1, 5, 10, 20}   # epochs at which to measure PR

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step957_matryoshka_d_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical

# Config spec: (d_subs, alpha)
CONFIG_SPEC = {
    "Ref":       None,              # Trainer path — no aux heads
    "A_uniform": ([2, 4, 8], 1.0),
    "B_decay2":  ([2, 4, 8], 0.5),
    "C_d48only": ([4, 8],    1.0),
}


# ---------------------------------------------------------------------------
# Participation Ratio
# ---------------------------------------------------------------------------

def participation_ratio(Z: torch.Tensor) -> float:
    """PR of Z [B,N,D]: uses mean-pooled [B,D] representations.

    PR = (sum|λ|)² / sum(λ²)  where λ = eigenvalues of covariance matrix.
    Range: 1 (one dim dominates) to D (all dims equal).
    """
    with torch.no_grad():
        Zf = Z.mean(1).float().cpu()         # [B, D] — cpu: eigvalsh not on MPS
        Zc = Zf - Zf.mean(0, keepdim=True)  # centre
        cov = Zc.T @ Zc / max(len(Zc) - 1, 1)  # [D, D]
        ev = torch.linalg.eigvalsh(cov).abs()    # [D], real for symmetric
        pr = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-8)).item()
    return pr


# ---------------------------------------------------------------------------
# ΔW-proj vector (shared by Ref wrapper and MRL model)
# ---------------------------------------------------------------------------

def _dw_proj_vec(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K_hh,D]


# ---------------------------------------------------------------------------
# Ref model wrapper (uses Trainer)
# ---------------------------------------------------------------------------

class SGNNET_Ref(nn.Module):
    """Standard ΔW-proj model. Identical to step887/step955 Ref."""

    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# MRL model: exposes route() → Z [B,N,D] + main readout head
# ---------------------------------------------------------------------------

class SGNNET_MRL(nn.Module):
    """ΔW-proj model with Matryoshka sub-dim auxiliary heads.

    route(x) → Z [B,N,D] (before readout)
    forward(x) → main logits [B, N_OUT]  (main readout only, for eval)
    aux_heads: nn.ModuleList of nn.Linear(d_sub, N_OUT)
    """

    def __init__(self, resonant, d_subs: list[int]):
        super().__init__()
        self.m = resonant
        # Sub-dim linear classifiers (no bias — mirrors typical MRL practice)
        self.aux_heads = nn.ModuleList([
            nn.Linear(d, N_OUT, bias=False) for d in d_subs
        ])
        self.d_subs = d_subs

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """Return Z [B,N,D] after routing, before readout."""
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main readout only — used during eval."""
        Z = self.route(x)
        return self.m.base._readout(Z)


# ---------------------------------------------------------------------------
# Base model factory
# ---------------------------------------------------------------------------

def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    spec = CONFIG_SPEC[key]
    r = make_base()
    if spec is None:
        return SGNNET_Ref(r)
    d_subs, _ = spec
    return SGNNET_MRL(r, d_subs=d_subs)


# ---------------------------------------------------------------------------
# Custom MRL training loop
# ---------------------------------------------------------------------------

def mrl_train(model: SGNNET_MRL, tr, va, alpha: float, device, n_epochs: int):
    """Custom training loop for MRL configs.

    Main loss: KL(main_logits, soft_labels) — mirrors Trainer task_loss.
    Aux losses: CE(aux_head(z_sub), hard_labels) — natural for sub-dim probes.
    Combined: main_loss + alpha^1 * aux_D8 + alpha^2 * aux_D4 + alpha^3 * aux_D2
              (powers ascending from largest to smallest sub-dim)
    """
    from src.training.experiment_config import GA_BEST
    lr = GA_BEST["lr_Wpos"]

    # Optimizer: W_pos (no weight decay) + aux_heads (standard decay)
    opt = torch.optim.AdamW([
        {"params": [model.W_pos],           "lr": lr, "weight_decay": 0.0},
        {"params": list(model.aux_heads.parameters()), "lr": lr, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-7,
    )

    # AMP setup (mirrors Trainer logic)
    _dev = str(device)
    use_amp = "cpu" not in _dev
    use_scaler = use_amp and "cuda" in _dev
    scaler = torch.amp.GradScaler(device) if use_scaler else None

    history = []
    pr_log = {}

    # d_subs ordered large→small for alpha^k weighting (k=1,2,3,...)
    d_subs_desc = sorted(model.d_subs, reverse=True)
    aux_heads_desc = [
        model.aux_heads[model.d_subs.index(d)] for d in d_subs_desc
    ]

    for epoch in range(n_epochs):
        model.train()
        total_loss_sum = 0.0; n_batches = 0
        Z_sample = None  # capture last batch Z for PR

        for features, soft_labels, hard_labels in tr:
            features    = features.to(device)
            soft_labels = soft_labels.to(device)
            hard_labels = hard_labels.to(device)

            opt.zero_grad()

            _amp_ctx = (
                torch.autocast(str(device).split(":")[0], dtype=torch.float16)
                if use_amp else torch.autocast("cpu", enabled=False)
            )
            with _amp_ctx:
                Z = model.route(features)           # [B, N, D]
                main_logits = model.m.base._readout(Z)  # [B, N_OUT]

                main_loss = F.kl_div(
                    F.log_softmax(main_logits, dim=-1),
                    soft_labels,
                    reduction="batchmean",
                )

                z_mean = Z.mean(1)  # [B, D]
                aux_loss = torch.tensor(0.0, device=device)
                for k, (d, head) in enumerate(zip(d_subs_desc, aux_heads_desc), start=1):
                    w = alpha ** k
                    sub_logits = head(z_mean[:, :d])  # [B, N_OUT]
                    aux_loss = aux_loss + w * F.cross_entropy(sub_logits, hard_labels)

                loss = main_loss + aux_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # W_pos clamping (mirrors Trainer TRAIN-03)
            with torch.no_grad():
                model.W_pos.clamp_(0, 1.0)

            total_loss_sum += loss.item()
            n_batches += 1
            Z_sample = Z.detach()

        train_loss = total_loss_sum / max(n_batches, 1)

        # PR logging at selected epochs (1-indexed)
        ep1 = epoch + 1
        if ep1 in PR_LOG_EPOCHS and Z_sample is not None:
            pr_val = participation_ratio(Z_sample)
            pr_log[ep1] = round(pr_val, 3)

        # Validation (main readout only)
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for features, _, labels in va:
                features = features.to(device)
                _amp_ctx2 = (
                    torch.autocast(str(device).split(":")[0], dtype=torch.float16)
                    if use_amp else torch.autocast("cpu", enabled=False)
                )
                with _amp_ctx2:
                    scores = model(features)
                all_scores.append(scores.cpu())
                all_labels.append(labels)

        all_scores_cat = torch.cat(all_scores, 0)
        all_labels_cat = torch.cat(all_labels, 0)
        val_top1 = (all_scores_cat.argmax(1) == all_labels_cat).float().mean().item()

        scheduler.step(train_loss)
        lr_cur = opt.param_groups[0]["lr"]

        rec = {"epoch": epoch, "train_loss": train_loss,
               "val_top1": val_top1, "lr": lr_cur}
        history.append(rec)

        print(f"  e{ep1:3d}  top1={val_top1:.4f}  loss={train_loss:.4f}"
              + (f"  PR={pr_log[ep1]:.2f}" if ep1 in pr_log else "")
              + f"  lr={lr_cur:.2e}", flush=True)

    return history, pr_log


# ---------------------------------------------------------------------------
# Ref PR measurement (single eval pass, no training)
# ---------------------------------------------------------------------------

def measure_ref_pr(model: SGNNET_Ref, va, device) -> dict:
    """Measure PR on a validation batch for the Ref model (after training)."""
    model.eval()
    pr_log = {}
    with torch.no_grad():
        for features, _, _ in va:
            features = features.to(device)
            Z = model.m.base._seed(features)
            conn_hh   = model.m.base.conn_hh
            theta_pos = model.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            dw = _dw_proj_vec(model.m.W_pos, conn_hh)
            Z_ref = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            pr_val = participation_ratio(Z)
            pr_log["final"] = round(pr_val, 3)
            break  # one batch is enough
    return pr_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print(f"step957 — Matryoshka Representation Learning T0 (20ep, 50%)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}/K_in={K_IN}")
    print(f"  Motivation: PR≈2.3 (step952) → 13+ dims near-dead → MRL forces full usage")
    print(f"  PR tracked at epochs {sorted(PR_LOG_EPOCHS)}")
    print(f"  step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  {'Config':<12} {'d_subs':<14} {'alpha':>6}  {'aux_heads':>10}")
    for k, spec in CONFIG_SPEC.items():
        if spec is None:
            print(f"  {k:<12} {'(Trainer)':<14} {'—':>6}  {'0':>10}")
        else:
            d_subs, alpha = spec
            nh = len(d_subs)
            print(f"  {k:<12} {str(d_subs):<14} {alpha:>6.1f}  {nh:>10}")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue

        spec = CONFIG_SPEC[key]
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"{key}: params={n_p:,}  spec={spec}")

        t0 = time.time()

        if spec is None:
            # Ref: use Trainer (KL distillation)
            kw = trainer_kwargs(N, n_epochs=EPOCHS)

            def log_fn(m):
                print(f"  e{m['epoch']+1:3d}  top1={m['val_top1']:.4f}"
                      f"  lr={m['lr']:.2e}", flush=True)

            history = Trainer(model=model, train_loader=tr, val_loader=va,
                              device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
            pr_log = measure_ref_pr(model, va, DEVICE)
            top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                     for h in history]
        else:
            # MRL configs: custom training loop
            d_subs, alpha = spec
            history, pr_log = mrl_train(model, tr, va, alpha=alpha,
                                        device=DEVICE, n_epochs=EPOCHS)
            top1h = [h["val_top1"] for h in history]

        elapsed = time.time() - t0
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))

        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")
        print(f"  -> PR_log={pr_log}")

        results[key] = {
            "spec": str(spec),
            "n_params": n_p,
            "best": round(best, 4),
            "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4),
            "elapsed_s": round(elapsed),
            "pr_log": pr_log,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 957 SUMMARY — Matryoshka D T0")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<12} params={r['n_params']:>8,}  best={r['best']:.4f}"
              f"  Δ={d*100:+.2f}pp  {v}  pr={r['pr_log']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
