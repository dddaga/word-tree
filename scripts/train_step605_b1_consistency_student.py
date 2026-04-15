"""Step 605 (B1 student): K=1 SGNNET student via consistency-DEQ distillation.

MOTIVATION (Opus mediation candidate B1)
=========================================
step196 killed "K_iter distillation" with output-matching KD only.
Consistency-DEQ (arXiv:2602.03024, 2024) distills the SOLVER TRAJECTORY,
not just the final output. SGNNET's K_iter IS a DEQ fixed-point solver.

This script loads the K=5 teacher trajectory cache from step604, then
trains a K=1 student under a composite loss:

    L = α·L_traj + β·L_kd·T² + (1-α-β)·L_ce

  L_traj = ||student.Z_1 − teacher.Z_5||²   (trajectory consistency)
  L_kd   = KL(log_softmax(s/T), softmax(t/T)) * T²  (output KD)
  L_ce   = cross_entropy(logits_student, y_true)     (hard label)

WIN CONDITION: K=1 student ≥ K=5 teacher accuracy at 5× fewer routing iters.
VERDICT CRITERIA:
  Within 0pp : decisive win — student matches teacher, wall-time halves
  Within 1pp : viable — publish as "5× faster with <1pp drop"
  Within 3pp : keep exploring higher α or different T
  >3pp gap   : abandon B1, note for baselines_needed

SWEEP: 6 (α, β, T) configs — trajectory-dominated to pure-output-KD
  Config_0: α=0.7, β=0.2, T=4.0  (trajectory-dominated — RECOMMENDED)
  Config_1: α=0.5, β=0.4, T=4.0  (balanced)
  Config_2: α=0.3, β=0.6, T=4.0  (KD-dominated)
  Config_3: α=0.0, β=1.0, T=4.0  (pure output KD — replicates step196)
  Config_4: α=0.7, β=0.2, T=8.0  (softer KD)
  Config_5: α=1.0, β=0.0, T=4.0  (pure trajectory)

ARCH: K=1, N=2048, D=16, K_hh=2, ΔW proj — everything else identical to teacher.
TIER: Tier-1 (75ep, full data).

To run:
    python -u scripts/train_step605_b1_consistency_student.py
    python -u scripts/train_step605_b1_consistency_student.py --device cpu --epochs 3 --configs Config_0
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import H5Dataset, make_loaders

# --- CLI --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Step 605 B1 consistency-DEQ student sweep")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--configs", default="",
                    help="Comma-separated config keys (empty = all)")
parser.add_argument("--cache",   default="",
                    help="Override path to teacher cache H5 (default: data/teacher_trajectory_step604.h5)")
args = parser.parse_args()

DEVICE = (torch.device("cuda")  if torch.cuda.is_available()
    else  torch.device("mps")   if torch.backends.mps.is_available()
    else  torch.device("cpu")   ) if args.device == "auto" else torch.device(args.device)

EPOCHS     = args.epochs
BATCH      = 128
SEED       = args.seed

# Architecture — MUST match teacher exactly except K_iter
N           = 2048;  N_IN  = 25088;  N_OUT = 10
D           = 16;    K_HH  = 2;      K_IN  = 25
K_ITER_STUDENT = 1    # student uses K=1
ALPHA_REFLECT  = 0.5

DATA_PATH   = ROOT / "data" / "store.h5"
CACHE_PATH  = Path(args.cache) if args.cache else ROOT / "data" / "teacher_trajectory_step604.h5"
SLOT        = os.environ.get("SGN_SLOT", "local")
OUT_PATH    = ROOT / "results" / f"train_step605_b1_consistency_student_seed{SEED}__{SLOT}.json"

# ---------------------------------------------------------------------------
# Sweep configs: (alpha_traj, beta_kd, temperature)
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    dict(key="Config_0", alpha=0.7, beta=0.2, T=4.0, label="trajectory-dominated (recommended)"),
    dict(key="Config_1", alpha=0.5, beta=0.4, T=4.0, label="balanced"),
    dict(key="Config_2", alpha=0.3, beta=0.6, T=4.0, label="KD-dominated"),
    dict(key="Config_3", alpha=0.0, beta=1.0, T=4.0, label="pure output KD (replicates step196)"),
    dict(key="Config_4", alpha=0.7, beta=0.2, T=8.0, label="trajectory + softer KD"),
    dict(key="Config_5", alpha=1.0, beta=0.0, T=4.0, label="pure trajectory"),
]


# ---------------------------------------------------------------------------
# Model — identical to step268 / step604 teacher but K_iter=1
# ---------------------------------------------------------------------------

class SGNNET_DeltaProjStudent(nn.Module):
    """K=1 ΔW projection student for consistency-DEQ distillation."""

    def __init__(self, N_hidden, N_out, D_, N_in, K_in, K_iter, K_local, K_random,
                 n_groups, alpha_reflect, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.base = SGNNET_SmallWorld(
            N_hidden=N_hidden, N_out=N_out, D=D_, N_in=N_in,
            K_in=K_in, K_iter=K_iter, K_local=K_local, K_random=K_random,
            n_groups=n_groups, norm_mode="l2", encoding_mode="fourier")
        self.resonant = SGNNET_Resonant(
            self.base, K_phase=8, alpha_reflect=alpha_reflect,
            alpha_turing=0.0, beam_size=16, geo_gamma=0.5,
            mode="dynamic_z_geo", resonance_threshold=0.0)
        self.alpha_reflect = alpha_reflect

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.resonant.W_phase

    def tick_epoch(self):
        if hasattr(self.resonant, "tick_epoch"):
            self.resonant.tick_epoch()

    def forward_with_z(self, x):
        """Forward returning (logits, Z_1) — Z BEFORE readout.

        Returns
        -------
        logits : [B, N_out]
        Z      : [B, N_hidden, D]  — latent state after K=1 routing step
        """
        Z = self.base._seed(x)
        conn_hh = self.base.conn_hh
        N_h     = self.base.N_hidden
        theta_pos = self.resonant.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h     = self.base.W_pos[:N_h]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]            # [N, K_hh, D]
        dw_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)   # [1, N, K_hh, D]

        Z_reflected = torch.zeros_like(Z)
        # K_iter=1 — single pass
        for _ in range(self.base.K_iter):
            Z_fwd    = F.relu(Z - theta_pos)
            Z_nb     = Z_fwd[:, conn_hh, :]                   # [B, N, K_hh, D]
            proj     = (Z_nb * dw_norm).sum(-1, keepdim=True) # [B, N, K_hh, 1]
            Z_nb     = Z_nb * proj.abs()                       # sign-invariant gate
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder  = Z_fwd - Z
            Z_reflected  = self.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        logits = self.base._readout(Z)
        return logits, Z

    def forward(self, x):
        logits, _ = self.forward_with_z(x)
        return logits


def build_student():
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    return SGNNET_DeltaProjStudent(
        N_hidden=N, N_out=N_OUT, D_=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER_STUDENT, K_local=K_l, K_random=K_r,
        n_groups=ng, alpha_reflect=ALPHA_REFLECT, seed=SEED)


# ---------------------------------------------------------------------------
# Teacher trajectory cache (index-addressable)
# ---------------------------------------------------------------------------

class TeacherCache:
    """Wraps teacher H5 cache for per-index access during training.

    Loads Z_final and logits into RAM tensors for fast batch indexing.
    """

    def __init__(self, h5_path: Path, split: str):
        print(f"  Loading teacher cache [{split}] from {h5_path} ...", flush=True)
        t0 = time.time()
        with h5py.File(h5_path, "r") as f:
            self.Z_final = torch.from_numpy(f[f"{split}/Z_final"][:]).float()
            self.logits  = torch.from_numpy(f[f"{split}/logits"][:])
            self.teacher_top1 = float(f.attrs.get("teacher_top1", 0.0))
            self.K_iter_teacher = int(f.attrs.get("K_iter", 5))
        gb = self.Z_final.numel() * 4 / 1e9
        print(f"  Cache [{split}]: Z={list(self.Z_final.shape)}  logits={list(self.logits.shape)}"
              f"  ({gb:.2f} GB)  teacher_top1={self.teacher_top1:.4f}  elapsed={time.time()-t0:.1f}s")

    def __len__(self): return len(self.Z_final)

    def get(self, indices: torch.Tensor, device: torch.device):
        """Return (Z_t, logits_t) for given sample indices."""
        return (self.Z_final[indices].to(device),
                self.logits[indices].to(device))


# ---------------------------------------------------------------------------
# Dataset that tracks original indices (needed for cache lookup)
# ---------------------------------------------------------------------------

class IndexedH5Dataset(torch.utils.data.Dataset):
    """H5Dataset that also returns the sample index."""

    def __init__(self, path: str, split: str):
        import h5py as _h5
        with _h5.File(path, "r") as f:
            self.features    = torch.from_numpy(f[f"{split}/features"][:])
            self.soft_labels = torch.from_numpy(f[f"{split}/soft_labels"][:])
            self.labels      = torch.from_numpy(f[f"{split}/labels"][:]).long()

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx], self.labels[idx], idx


# ---------------------------------------------------------------------------
# Consistency-DEQ loss
# ---------------------------------------------------------------------------

def consistency_loss(logits_s, Z_s, Z_t, logits_t, y_true, alpha, beta, T):
    """Composite consistency-DEQ distillation loss.

    L = alpha * L_traj  +  beta * L_kd * T²  +  (1-alpha-beta) * L_ce

    Parameters
    ----------
    logits_s : [B, N_out]    student logits
    Z_s      : [B, N, D]     student latent state (K=1 output)
    Z_t      : [B, N, D]     teacher latent state (K=5 fixed point)
    logits_t : [B, N_out]    teacher pre-softmax logits (for KD)
    y_true   : [B]           ground-truth class indices
    alpha    : trajectory loss weight
    beta     : KD loss weight
    T        : KD temperature
    """
    # Trajectory consistency: ||Z_student_K1 - Z_teacher_K5||^2  (MSE per element)
    L_traj = F.mse_loss(Z_s, Z_t)

    # Output KD: soft labels from teacher
    L_kd   = F.kl_div(
        F.log_softmax(logits_s / T, dim=-1),
        F.softmax(logits_t / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)

    # Hard label CE
    L_ce   = F.cross_entropy(logits_s, y_true)

    gamma = 1.0 - alpha - beta
    L     = alpha * L_traj + beta * L_kd + gamma * L_ce

    return L, L_traj.item(), L_kd.item(), L_ce.item()


# ---------------------------------------------------------------------------
# Training loop (manual — bypasses Trainer to inject consistency loss)
# ---------------------------------------------------------------------------

def train_one_config(cfg, teacher_train, teacher_val, train_ds, val_ds, device):
    """Train K=1 student under one (alpha, beta, T) config. Returns history list."""
    alpha = cfg["alpha"]; beta = cfg["beta"]; T = cfg["T"]
    print(f"\n{'─'*60}\n{cfg['key']}: α={alpha} β={beta} T={T}  {cfg['label']}\n{'─'*60}")

    # Build fresh student each config
    student = build_student().to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  student params={n_params:,}  K_iter={K_ITER_STUDENT}")

    # Optimizer: AdamW, same LR as standard trainer_kwargs
    kw   = trainer_kwargs(N, n_epochs=EPOCHS)
    lr   = kw["lr_wpos"]
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)

    # LR scheduler: ReduceLROnPlateau (matching trainer defaults)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=kw["sched_patience"],
        factor=kw["sched_factor"], min_lr=kw["min_lr"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # AMP scaler (GPU/MPS; noop on CPU)
    use_amp    = (device.type in ("cuda", "mps")) and kw.get("use_amp", True)
    amp_dtype  = torch.float16 if device.type == "cuda" else torch.bfloat16
    scaler     = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history    = []
    best_top1  = 0.0
    no_improve = 0
    t0         = time.time()

    for epoch in range(EPOCHS):
        # ── train phase ──────────────────────────────────────────────
        student.train()
        if hasattr(student, "tick_epoch"):
            student.tick_epoch()

        total_loss = 0.0; n_batches = 0
        for x, _, y, indices in train_loader:
            x = x.to(device); y = y.to(device)
            # Fetch teacher cache for this batch
            Z_t, logits_t = teacher_train.get(indices, device)
            # Cast to match AMP dtype if needed
            if use_amp and device.type == "mps":
                with torch.autocast(device_type="mps", dtype=amp_dtype):
                    logits_s, Z_s = student.forward_with_z(x)
                    loss, *_ = consistency_loss(logits_s, Z_s, Z_t, logits_t, y, alpha, beta, T)
            elif use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits_s, Z_s = student.forward_with_z(x)
                    loss, *_ = consistency_loss(logits_s, Z_s, Z_t, logits_t, y, alpha, beta, T)
            else:
                logits_s, Z_s = student.forward_with_z(x)
                loss, *_ = consistency_loss(logits_s, Z_s, Z_t, logits_t, y, alpha, beta, T)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda" and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item(); n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── val phase ─────────────────────────────────────────────────
        student.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, _, y, _ in val_loader:
                x = x.to(device); y = y.to(device)
                logits_s, _ = student.forward_with_z(x)
                pred = logits_s.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total   += y.size(0)

        val_top1 = correct / max(total, 1)
        scheduler.step(val_top1)

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_top1": val_top1})

        if (epoch+1) % 10 == 0:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep{epoch+1:3d}  val={val_top1:.4f}  loss={avg_loss:.4f}"
                  f"  lr={lr_now:.2e}  t={elapsed:.0f}s", flush=True)

        # Early stopping
        if val_top1 > best_top1 + kw.get("early_stop_delta", 5e-4):
            best_top1  = val_top1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= kw.get("early_stop_patience", 50):
                print(f"  Early stop at ep{epoch+1} (no improvement in {no_improve} epochs)")
                break

    elapsed_total = time.time() - t0
    top1s  = [h["val_top1"] for h in history]
    best   = max(top1s); best_ep = int(np.argmax(top1s)) + 1
    print(f"  → {cfg['key']} best={best:.4f} @ep{best_ep}  elapsed={elapsed_total:.0f}s")

    return {
        "key": cfg["key"], "alpha": alpha, "beta": beta, "T": T,
        "label": cfg["label"],
        "K_iter_student": K_ITER_STUDENT,
        "n_params": n_params,
        "top1_best": best, "top1_last": top1s[-1], "best_epoch": best_ep,
        "top1_history": top1s,
        "elapsed_s": round(elapsed_total, 1),
    }


# ---------------------------------------------------------------------------
# Quick bench: K=1 vs K=5 wall-time projection
# ---------------------------------------------------------------------------

def bench_latency(student, device, n_warmup=20, n_trials=100):
    """Measure per-forward latency at batch=32 for quick comparison."""
    student.eval()
    dummy = torch.randn(32, N_IN, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = student.forward(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = student.forward(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_trials * 1e3
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"Step 605 B1 — Consistency-DEQ K=1 student sweep")
    print(f"  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  cache={CACHE_PATH}")
    print(f"  data={DATA_PATH}")
    print(f"{'='*70}")

    # ── 1. Verify cache exists ────────────────────────────────────────────
    if not CACHE_PATH.exists():
        print(f"\n  ERROR: Teacher cache not found at {CACHE_PATH}")
        print(f"  Run train_step604_b1_consistency_teacher.py first.")
        sys.exit(1)

    # ── 2. Load teacher cache into RAM ────────────────────────────────────
    teacher_train = TeacherCache(CACHE_PATH, split="train")
    teacher_val   = TeacherCache(CACHE_PATH, split="val")
    teacher_top1  = teacher_train.teacher_top1
    print(f"\n  Teacher K={teacher_train.K_iter_teacher}  top1={teacher_top1:.4f}")

    # ── 3. Load indexed datasets ──────────────────────────────────────────
    print(f"  Loading datasets ...")
    train_ds = IndexedH5Dataset(DATA_PATH, split="train")
    val_ds   = IndexedH5Dataset(DATA_PATH, split="val")
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # ── 4. Select sweep configs ───────────────────────────────────────────
    run_keys   = [k.strip() for k in args.configs.split(",")] if args.configs else None
    active_cfgs = [c for c in SWEEP_CONFIGS
                   if run_keys is None or c["key"] in run_keys]
    print(f"\n  Running {len(active_cfgs)} configs: {[c['key'] for c in active_cfgs]}")

    # ── 5. Train each config ──────────────────────────────────────────────
    all_results = {}
    for cfg in active_cfgs:
        r = train_one_config(cfg, teacher_train, teacher_val, train_ds, val_ds, DEVICE)
        all_results[cfg["key"]] = r
        # Save incrementally
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # ── 6. Quick bench (K=1 student wall-time) ────────────────────────────
    bench_ms = None
    if DEVICE.type in ("cuda", "mps", "cpu"):
        print(f"\n  Benchmarking K=1 student wall-time (batch=32) ...")
        try:
            # Build a fresh student just for bench
            bench_student = build_student().to(DEVICE)
            bench_ms = bench_latency(bench_student, DEVICE)
            print(f"  K=1 student: {bench_ms:.3f} ms/forward (batch=32)")
            print(f"  Projected K=5 teacher: ~0.280 ms (bench_step811)")
            if bench_ms < 0.280:
                speedup = 0.280 / bench_ms
                print(f"  Speedup vs K=5: {speedup:.2f}×")
        except Exception as e:
            print(f"  Bench failed: {e}")

    # ── 7. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Step 605 B1 SUMMARY — K=1 student vs K=5 teacher ({teacher_top1:.4f})")
    print(f"{'─'*70}")
    print(f"  {'Config':<12}  {'α':>5}  {'β':>5}  {'T':>5}  "
          f"{'K=1 top1':>10}  {'Δ vs teacher':>14}  Label")
    print(f"{'─'*70}")

    for cfg in active_cfgs:
        r = all_results.get(cfg["key"])
        if not r: continue
        delta = r["top1_best"] - teacher_top1
        verdict = (
            "DECISIVE WIN" if delta >= 0.0  else
            "VIABLE"       if delta >= -0.01 else
            "EXPLORE"      if delta >= -0.03 else
            "ABANDON"
        )
        print(f"  {cfg['key']:<12}  α={r['alpha']:.1f}  β={r['beta']:.1f}  T={r['T']:4.1f}  "
              f"{r['top1_best']:.4f}  {delta:+.4f} ({verdict:<14})  {cfg['label']}")

    if bench_ms is not None:
        print(f"\n  K=1 wall-time: {bench_ms:.3f} ms  (K=5 ref: ~0.280 ms)")

    print(f"\n  Win condition: K=1 ≥ K=5 teacher ({teacher_top1:.4f})")
    print(f"  Results → {OUT_PATH}")
    print(f"{'='*70}")

    # Final JSON patch with bench and teacher_top1
    all_results["_meta"] = {
        "teacher_top1":     teacher_top1,
        "teacher_K_iter":   teacher_train.K_iter_teacher,
        "student_K_iter":   K_ITER_STUDENT,
        "bench_k1_ms":      bench_ms,
        "bench_k5_ref_ms":  0.280,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
