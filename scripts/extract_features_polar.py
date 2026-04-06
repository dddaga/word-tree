"""Extract polar-compressed and PCA-compressed feature sets from store.h5.

POLAR QUANTIZATION (PolarQuant, arXiv:2502.02617)
=================================================
Recursive binary polar decomposition of each 25088-dim VGG16 pool5 vector.
Method from: "PolarQuant: Polar Quantization for Low-Bit LLMs" (KAIST, AISTATS 2026).

Algorithm:
  1. Split vector into two halves [a, b].
  2. Record polar angle ψ = arctan2(||b||, ||a||) / (π/2) ∈ [0,1].
     (This is analytically correct; energy-ratio mag_b/(mag_a+mag_b) is biased.)
  3. Recurse into normalised halves: a/||a||, b/||b||.
  4. BFS order: root angle first, then children.

The first K angles (BFS order) capture the most global structure:
  - angle[0]: global energy split (first vs second half of all 25088 dims)
  - angle[1..2]: energy splits within each half
  - angle[3..6]: quarter-level
  - angle[7..14]: eighth-level  ...  (level l has 2^l angles)

K=511 (levels 0-8): captures the ~top 9 scales of structure.
K=1023 (levels 0-9): finer detail.

WHY THIS MATTERS FOR SGNNET
============================
Current K_in=50 / N_in=25088 → each neuron sees 0.2% of input.
With compressed N_in=512 and same K_in=50 → each neuron sees 9.8% (50× better).
With K_in=256 → 50% coverage — neuron has a global receptive field.

This directly attacks the #1 bottleneck: sparse input connectivity.

PCA compression is included as a baseline to separate polar structure benefit
from the pure "reduced N_in" effect.

Outputs (saved to data/):
  store_polar512.h5    — 511 polar angles + 1 root magnitude (512 dims)
  store_polar1024.h5   — 1023 polar angles + 1 root magnitude (1024 dims)
  store_pca512.h5      — top-512 PCA components
  store_pca1024.h5     — top-1024 PCA components

To reproduce:
    python -u scripts/extract_features_polar.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py
from collections import deque


def polar_compress_batch(features: np.ndarray, K: int = 512) -> np.ndarray:
    """Compress [N_samples, N_in] → [N_samples, K] via PolarQuant decomposition.

    BFS-order: root angle first (global energy split), then finer levels.
    Uses PolarQuant (arXiv:2502.02617) arctan2 formula:
      ψ = arctan2(||b||, ||a||) / (π/2)  →  angle ∈ [0, 1]
    This is analytically correct for the recursive polar transform:
    the marginal distribution of ψ at level ℓ is sin^(2^(ℓ-1)-1)(2ψ),
    which arctan2 matches exactly (vs energy-ratio which is non-uniform).
    """
    N_samples, N_in = features.shape
    out = np.zeros((N_samples, K), dtype=np.float32)

    for i in range(N_samples):
        v = features[i].astype(np.float64)
        angles = []
        queue = deque([v])

        while queue and len(angles) < K:
            current = queue.popleft()
            n = len(current)
            if n <= 1:
                angles.append(float(current[0]) if n == 1 else 0.0)
                continue
            mid = n // 2
            a, b = current[:mid], current[mid:]
            mag_a = np.linalg.norm(a)
            mag_b = np.linalg.norm(b)
            if mag_a + mag_b < 1e-12:
                angles.append(0.5)
                continue
            # PolarQuant arctan2: ψ = arctan2(r_b, r_a) / (π/2) ∈ [0, 1]
            angle = float(np.arctan2(mag_b + 1e-12, mag_a + 1e-12) * (2.0 / np.pi))
            angles.append(angle)
            if len(angles) < K:
                queue.append(a / (mag_a + 1e-12))
            if len(angles) < K:
                queue.append(b / (mag_b + 1e-12))

        angles = angles[:K]
        angles.extend([0.0] * (K - len(angles)))
        out[i] = angles

        if (i + 1) % 500 == 0:
            print(f"  polar: {i+1}/{N_samples}", flush=True)

    return out


def pca_compress(train_feat: np.ndarray, val_feat: np.ndarray,
                 K: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Project to top-K PCA components (fit on train, transform both)."""
    print(f"  PCA: centering...", flush=True)
    mean = train_feat.mean(axis=0, keepdims=True)   # [1, N_in]
    X = train_feat - mean
    X_val = val_feat - mean

    print(f"  PCA: SVD on ({X.shape[0]}, {X.shape[1]})...", flush=True)
    # Economy SVD: U [N, N], S [N], Vt [N, N_in]
    # Use randomised SVD for speed on large N_in
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, Vt = randomized_svd(X, n_components=K, random_state=42)
        print(f"  PCA: randomized_svd done, explained var ratio ≈ {(S**2).sum() / (X**2).sum():.3f}")
    except ImportError:
        # Fallback: covariance SVD on downsampled data
        print("  PCA: sklearn not found, using numpy SVD on covariance...", flush=True)
        cov = X.T @ X / X.shape[0]   # [N_in, N_in] — large, use if RAM allows
        S2, Vt_full = np.linalg.eigh(cov)
        order = np.argsort(-S2)
        Vt = Vt_full[:, order[:K]].T   # [K, N_in]
        U = X @ Vt.T / (np.sqrt(S2[order[:K]]) + 1e-8)

    train_pca = X @ Vt.T   # [N_train, K]
    val_pca   = X_val @ Vt.T   # [N_val, K]
    return train_pca.astype(np.float32), val_pca.astype(np.float32)


def save_h5(path: Path, train_feat, train_labels, val_feat, val_labels,
            train_sl=None, val_sl=None):
    """Save features + labels to h5 in the same format as store.h5."""
    path.parent.mkdir(exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("train/features", data=train_feat, compression="gzip")
        f.create_dataset("train/labels",   data=train_labels)
        f.create_dataset("val/features",   data=val_feat, compression="gzip")
        f.create_dataset("val/labels",     data=val_labels)
        if train_sl is not None:
            f.create_dataset("train/soft_labels", data=train_sl)
            f.create_dataset("val/soft_labels",   data=val_sl)
    print(f"  Saved: {path}  train={train_feat.shape}  val={val_feat.shape}")


if __name__ == "__main__":
    SRC = Path("data/store.h5")
    print(f"Loading {SRC} ...")
    with h5py.File(SRC, "r") as f:
        tr_feat  = f["train/features"][:]    # [9469, 25088]
        tr_lab   = f["train/labels"][:]
        va_feat  = f["val/features"][:]      # [3925, 25088]
        va_lab   = f["val/labels"][:]
        tr_sl    = f["train/soft_labels"][:] if "train/soft_labels" in f else None
        va_sl    = f["val/soft_labels"][:]   if "val/soft_labels"   in f else None

    print(f"Train: {tr_feat.shape}  Val: {va_feat.shape}")

    # ── PCA compression ────────────────────────────────────────────────────────
    for K in [512, 1024]:
        print(f"\n=== PCA-{K} ===")
        t0 = time.time()
        tr_pca, va_pca = pca_compress(tr_feat, va_feat, K=K)
        print(f"  done in {time.time()-t0:.1f}s")
        save_h5(Path(f"data/store_pca{K}.h5"), tr_pca, tr_lab, va_pca, va_lab, tr_sl, va_sl)

    # ── Polar compression ──────────────────────────────────────────────────────
    for K in [512, 1024]:
        print(f"\n=== Polar-{K} ===")
        t0 = time.time()
        print(f"  compressing train ({tr_feat.shape[0]} samples)...")
        tr_pol = polar_compress_batch(tr_feat, K=K)
        print(f"  compressing val ({va_feat.shape[0]} samples)...")
        va_pol = polar_compress_batch(va_feat, K=K)
        print(f"  done in {time.time()-t0:.1f}s")
        save_h5(Path(f"data/store_polar{K}.h5"), tr_pol, tr_lab, va_pol, va_lab, tr_sl, va_sl)

    print("\n=== All done ===")
    print("Created: store_pca512.h5  store_pca1024.h5  store_polar512.h5  store_polar1024.h5")
