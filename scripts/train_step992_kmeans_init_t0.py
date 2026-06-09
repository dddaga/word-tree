"""Step 992: K-means W_pos init vs random — T0 scout. Repairs §6.1 debt.

Brief §6.1 proposed seeding W_pos from K-means cluster centers. Random uniform
was adopted WITHOUT any experiment (violates try-and-test). This tests whether
K-means init improves readout geometry (dot-product scoring hidden→output).

SCOPE NOTE [HYPOTHESIS]: conn_hh is static Watts-Strogatz here; W_pos does NOT
shape hidden-hidden routing. Broader §6.1 routing claim needs rebuild_conn_hh.

SCALE CONFOUND NOTE: default init is uniform[0,box_size]; kmeans centers are
L2-normalized. A win conflates cluster structure WITH init scale; follow-up
would add random-unit-sphere control. Loss/neutral cleanly discharges the debt.

CONFIGS
  Ref             — random uniform W_pos (current standard)
  A_kmeans_hidden — K-means centers for hidden W_pos (N=2048 clusters)
  B_kmeans_both   — K-means hidden + class-mean output W_pos (N_OUT=10)

ADVANCE CRITERION: A or B >= Ref + 0.5pp → T1.
Tier: T0 scout (20ep, 50% Imagenette, B=512). CUDA-5060ti-validated.
"""
# CUDA-5060ti-validated
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from sklearn.cluster import KMeans

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant_cuda  import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import H5Dataset, make_loaders, make_subset_loader

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--configs", default="")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs;  BATCH = 512;  SEED = 42;  DATA = "data/store.h5"
N = 2048;  N_IN = 25088;  N_OUT = 10
D = 16;    K_HH = 2;      K_IN = 25;  K_ITER = 5
ALPHA_REFLECT = 0.5;  ALPHA_AHEBB = 1.0

CONFIGS = {
    "Ref":             "random",
    "A_kmeans_hidden": "kmeans_hidden",
    "B_kmeans_both":   "kmeans_both",
}

STEP_NAME = Path(__file__).stem
SLOT      = os.environ.get("SGN_SLOT", "local")
OUT_PATH  = ROOT / "results" / f"{STEP_NAME}__{SLOT}.json"


def kmeans_init_wpos(data_h5_path, n_samples=4096, seed=42):
    """Project n_samples train features to D-sphere; cluster + class-mean.

    Returns (hidden_centers [N,D], output_centers [N_OUT,D]) — L2-normalized.
    """
    ds     = H5Dataset(str(data_h5_path), split="train")
    feats  = ds.features[:n_samples]   # [n_samples, N_IN]
    labels = ds.labels[:n_samples]
    assert feats.shape[1] == N_IN

    rng = np.random.RandomState(seed)
    P   = rng.randn(N_IN, D).astype(np.float32)
    P  /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8

    proj  = feats.numpy() @ P                                   # [n_samples, D]
    proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8  # unit-sphere

    km = KMeans(n_clusters=N, n_init=3, random_state=seed, max_iter=50)
    km.fit(proj)
    c   = km.cluster_centers_.astype(np.float32)
    c  /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-8
    hidden_c = torch.from_numpy(c)                              # [N, D]

    out_c = np.zeros((N_OUT, D), dtype=np.float32)
    for cls in range(N_OUT):
        mask = labels.numpy() == cls
        if mask.sum() > 0:
            out_c[cls] = proj[mask].mean(axis=0)
    out_c /= np.linalg.norm(out_c, axis=1, keepdims=True) + 1e-8
    output_c = torch.from_numpy(out_c)                          # [N_OUT, D]

    return hidden_c, output_c


def build_model(variant, hidden_c=None, output_c=None):
    """Standard SGNNET chain; K-means variants written into base.W_pos."""
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=ng, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant_CUDA(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
        resonance_threshold=0.0, compile=True,
    )
    model = SGNNET_AntiHebbian_CUDA(resonant, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    # W_pos lives on base (not outer wrappers) — write via base reference.
    if variant in ("kmeans_hidden", "kmeans_both") and hidden_c is not None:
        with torch.no_grad():
            base.W_pos.data[:N].copy_(hidden_c)
    if variant == "kmeans_both" and output_c is not None:
        with torch.no_grad():
            base.W_pos.data[N:].copy_(output_c)
    return model


def main():
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs \
               else list(CONFIGS.keys())

    data_path = ROOT / DATA
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    print(f"\n{'='*70}")
    print(f"{STEP_NAME}  — K-means W_pos init vs random, T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  batch={BATCH}  seed={SEED}")
    print(f"  configs={run_keys}\n{'='*70}")

    hidden_c = output_c = None
    if any(CONFIGS[k] != "random" for k in run_keys):
        print("Running K-means (n_clusters=2048, n_samples=4096)…", flush=True)
        t_km = time.time()
        hidden_c, output_c = kmeans_init_wpos(data_path)
        print(f"  done in {time.time()-t_km:.1f}s", flush=True)

    _pin = (DEVICE.type == "cuda")
    tr = make_subset_loader(str(data_path), fraction=0.5, batch_size=BATCH,
                            seed=SEED, pin_memory=_pin)
    _, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                         pin_memory=_pin)

    results = {"step": "step992", "seed": SEED, "configs": {}}
    for key in run_keys:
        variant = CONFIGS[key]
        print(f"\n{'─'*60}\n{key}  variant={variant}\n{'─'*60}")
        model = build_model(variant, hidden_c, output_c).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}", flush=True)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()
        history = Trainer(
            model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw,
        ).train(
            n_epochs=EPOCHS,
            log_fn=lambda m: print(
                f"  e{m['epoch']+1:3d}  loss={m['train_loss']:.4f}"
                f"  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True,
            ),
        )
        elapsed = time.time() - t0
        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h)
                   for h in history]
        best, bep = max(top1h), int(np.argmax(top1h)) + 1
        results["configs"][key] = {
            "best": round(best, 4), "best_ep": bep, "n_params": n_p,
            "elapsed_s": round(elapsed, 1), "variant": variant,
            "history": [round(v, 4) for v in top1h],
        }
        print(f"  DONE: best={best:.4f} @ep{bep}  {elapsed:.0f}s")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results["configs"].get("Ref", {}).get("best", 0.0)
    print(f"\n{'='*70}\n{STEP_NAME} SUMMARY  (Ref={ref_best:.4f})\n{'='*70}")
    for key, r in results["configs"].items():
        d = r["best"] - ref_best
        v = "Ref" if key == "Ref" else ("ADVANCE" if d >= 0.005 else "NEUTRAL/KILL")
        print(f"  {key:<22}  {r['best']:.4f}  ({d:+.4f})  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
