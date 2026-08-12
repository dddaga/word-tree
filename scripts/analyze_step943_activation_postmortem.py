"""Step 943: Activation postmortem — information-theoretic analysis of Z representations.

Questions:
  1. Does routing improve linear separability of mean-pooled Z? (Fisher ratio per K_iter step)
  2. Is Z converging to a class-discriminative fixed point, or is it collapsing?
  3. What is the effective dimension (participation ratio) of Z at each step?
  4. Does the winner config actually encode more class information than baseline?

Metrics (all computable at D=16 without GPU):
  - Fisher ratio trace(S_B) / trace(S_W) per K_iter step
  - Linear probe accuracy per K_iter step (logistic regression, 100 iter)
  - Participation ratio (PR) = (Σλ)² / Σλ² of Z_pool covariance
  - Within-class vs across-class L2 distance (complement to cosine sim)
  - CKA between configs at final Z
  - Mutual information proxy: 1-NN accuracy in Z_pool space (class-balanced)

Configs compared:
  - trained_dw:  best Imagenette winner (step907 Ref_dw, 93.96%)
  - untrained:   same arch, random init (no training)
  - seed_only:   trained model but K_iter=0 (readout on seeded Z only)
  - esc50:       trained model on ESC-50 (step926 best config)

Usage:
  python3 scripts/analyze_step943_activation_postmortem.py --device cpu
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",      default="cpu")
parser.add_argument("--data",        default="data/store.h5",       help="Imagenette HDF5")
parser.add_argument("--data_esc50",  default="data/esc50/store_esc50_whisper.h5")
parser.add_argument("--ckpt_img",    default=None,                  help="Path to Imagenette checkpoint; if None, use random init")
parser.add_argument("--ckpt_esc50",  default=None,                  help="Path to ESC-50 checkpoint; if None, skip ESC-50")
parser.add_argument("--n_batches",   type=int, default=20,          help="Batches to sample for analysis (20 × 128 = 2560 samples)")
parser.add_argument("--seed",        type=int, default=42)
args = parser.parse_args()

DEVICE = torch.device(args.device)
SEED   = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Standard config
N = 2048; D = 16; K_HH = 2; K_IN = 25; K_ITER = 5; N_OUT_IMG = 10; N_OUT_ESC = 50
ALPHA_REFLECT = 0.5


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def _dw_proj(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)

def _dw_agg(Z_nb, dw):
    proj_coeff = (Z_nb * dw).sum(dim=-1, keepdim=True)
    return (Z_nb * proj_coeff.abs()).sum(dim=2)


def make_resonant(N_in, N_out, seed=SEED):
    torch.manual_seed(seed)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_out, D=D, N_in=N_in,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


# ---------------------------------------------------------------------------
# Z trajectory extractor — returns Z at each K_iter step
# ---------------------------------------------------------------------------

def extract_z_trajectory(model, x):
    """Run forward pass, capturing Z after each K_iter step.
    Returns: list of [B, N, D] tensors (len=K_ITER+1: seed + K_ITER steps).
    """
    m = model if not hasattr(model, 'm') else model.m
    base = m.base
    theta_pos = m.theta.abs().unsqueeze(0).unsqueeze(-1)
    conn_hh   = base.conn_hh
    dw        = _dw_proj(m.W_pos, conn_hh)

    Z = base._seed(x)
    trajectory = [Z.detach().cpu()]

    Z_ref = torch.zeros_like(Z)
    for _ in range(K_ITER):
        Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
        Z_nb  = Z_fwd[:, conn_hh, :]
        Z_agg = _dw_agg(Z_nb, dw)
        Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
        Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        trajectory.append(Z.detach().cpu())

    return trajectory  # [K_ITER+1] × [B, N, D]


# ---------------------------------------------------------------------------
# Information-theoretic metrics
# ---------------------------------------------------------------------------

def mean_pool(Z_traj):
    """Mean-pool Z_traj (list of [B,N,D]) → list of [B, D]."""
    return [z.mean(dim=1).numpy() for z in Z_traj]


def fisher_ratio(Z_pool: np.ndarray, labels: np.ndarray):
    """trace(S_B) / trace(S_W): between-class / within-class variance ratio."""
    classes = np.unique(labels)
    mu_all  = Z_pool.mean(axis=0)
    S_W = np.zeros((Z_pool.shape[1], Z_pool.shape[1]))
    S_B = np.zeros_like(S_W)
    for c in classes:
        mask = labels == c
        Z_c  = Z_pool[mask]
        mu_c = Z_c.mean(axis=0)
        diff_c = Z_c - mu_c
        S_W += diff_c.T @ diff_c
        diff_b = (mu_c - mu_all).reshape(-1, 1)
        S_B += mask.sum() * (diff_b @ diff_b.T)
    tr_W = np.trace(S_W) + 1e-10
    tr_B = np.trace(S_B)
    return tr_B / tr_W


def participation_ratio(Z_pool: np.ndarray):
    """Effective dimensionality: PR = (Σλ)² / Σλ². Max=D, min=1."""
    cov = np.cov(Z_pool.T)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.maximum(eigs, 0)
    s1 = eigs.sum()
    s2 = (eigs ** 2).sum()
    return (s1 ** 2) / (s2 + 1e-12)


def linear_probe_acc(Z_pool_tr: np.ndarray, y_tr: np.ndarray,
                     Z_pool_va: np.ndarray, y_va: np.ndarray):
    """Fit a logistic regression on Z_pool, return val accuracy."""
    sc = StandardScaler()
    Z_tr_s = sc.fit_transform(Z_pool_tr)
    Z_va_s = sc.transform(Z_pool_va)
    clf = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs',
                             random_state=SEED)
    clf.fit(Z_tr_s, y_tr)
    return accuracy_score(y_va, clf.predict(Z_va_s))


def nn1_accuracy(Z_pool_tr: np.ndarray, y_tr: np.ndarray,
                 Z_pool_va: np.ndarray, y_va: np.ndarray):
    """1-NN accuracy in Z_pool space — proxy for mutual information."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(Z_pool_tr, y_tr)
    return accuracy_score(y_va, knn.predict(Z_pool_va))


def within_across_l2(Z_pool: np.ndarray, labels: np.ndarray, n_pairs=500):
    """L2 distance within-class vs across-class (complement to cosine)."""
    rng = np.random.default_rng(SEED)
    classes = np.unique(labels)

    within_dists = []
    across_dists = []
    for _ in range(n_pairs):
        c = rng.choice(classes)
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        within_dists.append(np.linalg.norm(Z_pool[i] - Z_pool[j]))

        c2 = rng.choice(classes[classes != c])
        idx2 = np.where(labels == c2)[0]
        k = rng.choice(idx2)
        across_dists.append(np.linalg.norm(Z_pool[i] - Z_pool[k]))

    return np.mean(within_dists), np.mean(across_dists)


def linear_cka(X: np.ndarray, Y: np.ndarray):
    """Linear Centered Kernel Alignment between two representations [B, D1] and [B, D2]."""
    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    Kx = center(X @ X.T)
    Ky = center(Y @ Y.T)
    hsic_xy = np.trace(Kx @ Ky)
    hsic_xx = np.trace(Kx @ Kx)
    hsic_yy = np.trace(Ky @ Ky)
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def collect_z_and_labels(model, loader, n_batches, device):
    """Collect Z trajectories and labels from n_batches."""
    model.eval()
    all_traj = [[] for _ in range(K_ITER + 1)]
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch[0].to(device)
            y = batch[-1]  # labels are last element (features, [soft_labels,] labels)
            traj = extract_z_trajectory(model, x)   # list K_ITER+1 of [B,N,D]
            for t, z in enumerate(traj):
                all_traj[t].append(z)
            all_labels.append(y.cpu().numpy())

    traj_cat = [torch.cat(all_traj[t], dim=0) for t in range(K_ITER + 1)]
    labels    = np.concatenate(all_labels)
    return traj_cat, labels


def analyze_model(model, tr_loader, va_loader, device, name: str, n_batches=20):
    """Run full information-theoretic analysis on model."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    traj_tr, y_tr = collect_z_and_labels(model, tr_loader, n_batches, device)
    traj_va, y_va = collect_z_and_labels(model, va_loader, n_batches, device)

    pools_tr = mean_pool(traj_tr)   # list [K_ITER+1] of [B, D]
    pools_va = mean_pool(traj_va)

    results = {"name": name, "steps": []}

    for t in range(K_ITER + 1):
        Z_tr = pools_tr[t]; Z_va = pools_va[t]
        fr    = fisher_ratio(Z_tr, y_tr)
        pr    = participation_ratio(Z_tr)
        probe = linear_probe_acc(Z_tr, y_tr, Z_va, y_va)
        nn1   = nn1_accuracy(Z_tr, y_tr, Z_va, y_va)
        w_l2, a_l2 = within_across_l2(Z_tr, y_tr)

        step_res = {
            "t":            t,
            "fisher_ratio": float(round(fr, 4)),
            "part_ratio":   float(round(pr, 2)),
            "probe_acc":    float(round(probe, 4)),
            "nn1_acc":      float(round(nn1, 4)),
            "within_l2":    float(round(w_l2, 4)),
            "across_l2":    float(round(a_l2, 4)),
            "l2_sep":       float(round(a_l2 - w_l2, 4)),
        }
        label = "seed" if t == 0 else f"iter{t}"
        print(f"  t={t} ({label:6s}): FR={fr:.3f}  PR={pr:.1f}  probe={probe:.4f}  "
              f"1NN={nn1:.4f}  L2_sep={a_l2-w_l2:+.4f}")
        results["steps"].append(step_res)

    # CKA: seed vs final
    cka_seed_final = linear_cka(pools_tr[0], pools_tr[-1])
    print(f"  CKA(seed, final): {cka_seed_final:.4f}  (1.0=identical, 0.0=orthogonal)")
    results["cka_seed_final"] = round(cka_seed_final, 4)

    return results, (traj_tr[-1], traj_va[-1])  # return final Z for cross-config CKA


def main():
    img_path = ROOT / args.data
    if not img_path.exists():
        print(f"ERROR: {img_path} not found"); sys.exit(1)

    tr_loader, va_loader = make_loaders(str(img_path), batch_size=128, seed=SEED,
                                        pin_memory=False)

    print(f"\nActivation postmortem — Information-theoretic analysis of Z representations")
    print(f"  device={DEVICE}  n_batches={args.n_batches} (≈{args.n_batches*128} samples)")
    print(f"  D={D}  N={N}  K_iter={K_ITER}  K_hh={K_HH}")

    all_results = {}
    final_Zs_tr = {}
    final_Zs_va = {}

    # --- Config 1: Untrained (random init) ---
    print("\n--- Building untrained model ---")
    model_rand = make_resonant(N_in=25088, N_out=N_OUT_IMG).to(DEVICE)
    res, (Z_tr_f, Z_va_f) = analyze_model(
        model_rand, tr_loader, va_loader, DEVICE, "untrained", args.n_batches)
    all_results["untrained"] = res
    final_Zs_tr["untrained"] = mean_pool([Z_tr_f])[0]
    final_Zs_va["untrained"] = mean_pool([Z_va_f])[0]

    # --- Config 2: Trained Imagenette (if checkpoint provided) ---
    if args.ckpt_img and Path(args.ckpt_img).exists():
        print(f"\n--- Loading trained Imagenette checkpoint: {args.ckpt_img} ---")
        model_trained = make_resonant(N_in=25088, N_out=N_OUT_IMG).to(DEVICE)
        state = torch.load(args.ckpt_img, map_location=DEVICE)
        model_trained.load_state_dict(state, strict=False)
        res, (Z_tr_f, Z_va_f) = analyze_model(
            model_trained, tr_loader, va_loader, DEVICE, "trained_imagenette", args.n_batches)
        all_results["trained_imagenette"] = res
        final_Zs_tr["trained_imagenette"] = mean_pool([Z_tr_f])[0]
        final_Zs_va["trained_imagenette"] = mean_pool([Z_va_f])[0]
    else:
        print("\n--- Skipping trained Imagenette (no checkpoint) ---")
        print("    To load: --ckpt_img <path_to_checkpoint.pt>")
        print("    Save a checkpoint: add torch.save(model.state_dict(), path) to training script")

    # --- Cross-config CKA ---
    print(f"\n{'='*70}")
    print("  Cross-config CKA (final Z_pool):")
    keys = list(final_Zs_tr.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            cka = linear_cka(final_Zs_tr[k1], final_Zs_tr[k2])
            print(f"    CKA({k1}, {k2}) = {cka:.4f}")

    # --- Save results ---
    out = ROOT / "results" / "analyze_step943_activation_postmortem.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n-> {out}")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE:")
    print("  Fisher ratio: higher = more class-discriminative (between vs within variance)")
    print("  Part. ratio:  higher = richer representation (more active dimensions)")
    print("  probe_acc:    linear accuracy on mean-pooled Z — how learnable are the classes?")
    print("  1NN_acc:      kNN accuracy — how clustered are same-class samples?")
    print("  L2_sep:       across_l2 - within_l2 — positive = classes are separated in L2")
    print("  CKA(seed,final): how much does routing change the representation?")
    print("                   Near 1.0 = routing barely changes Z (identity)")
    print("                   Near 0.0 = routing completely transforms Z")


if __name__ == "__main__":
    main()
