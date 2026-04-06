"""Step 63: Activation-Gated Routing (AGR) ablation.

Tests input-dependent excitatory routing where activation soft-attention over
an expanded candidate set replaces the static sum(Z[conn_hh]).

MOTIVATION
==========
Every failed wave-1 mechanism (steps 58-61) added excitatory signal additively,
triggering safety valve collapse. The key failure mode: additive excitation
oversaturates the threshold gate → routing diversity collapses.

AGR is multiplicative: soft-attention REDISTRIBUTES the same total excitation
(softmax weights sum to 1) rather than adding more. This preserves total
excitation magnitude while making WHICH neurons contribute input-specific.

ARCHITECTURE
============
  conn_expanded [N, K'] = conn_hh (spatial) ∪ conn_phase (W_phase K-NN)
  score[b,i,j]  = dot(Z[b,i], Z[b,candidates[i,j]]) / sqrt(D)
  weights        = softmax(score, dim=-1)
  Z_struct[i]   = sum_j( weights[i,j] * Z_fwd[candidates[i,j]] )

Different inputs activate different subgraphs — same weights, input-specific routing.
Long-range phase inhibition from Resonant is kept unchanged.

CONFIGS (N=1024, D=64, K_iter=8, 50% data, 75ep)
=================================================
  Ref     : AntiHebb α=1.0, static conn_hh  [step57 REF_BASELINE = 73.53%]
  A       : soft attention, mixed candidates (conn_hh + conn_phase), no AH
  B       : soft attention, mixed candidates + AH=1.0
  C       : hard top-K (K_select=K_hh) + AH=1.0
  D       : soft attention + AH=1.0 + hop_decay=0.9 (10% decay per routing step)
  E       : soft attention + AH=1.0, phase-only candidates (conn_phase only)

KEY QUESTIONS
=============
  A vs Ref   → does input-dependent routing outperform static routing?
  B vs A     → does AH compound with attention-based routing?
  C vs B     → hard top-K vs soft attention: does sparsity help?
  D vs B     → does hop-count decay improve over no decay?
  E vs B     → phase-only candidates vs spatial+phase mixed pool

To reproduce:
    python -u scripts/train_step63_act_gated_routing.py --device mps
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.sgnnet.model_smallworld         import SGNNET_SmallWorld
from src.sgnnet.model_resonant           import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory    import SGNNET_AntiHebbian
from src.sgnnet.model_act_gated_routing  import SGNNET_ActGatedRouting
from src.training.trainer                import Trainer
from src.training.experiment_config      import trainer_kwargs, topology_kwargs, run_metadata
from src.training.dataset                import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = 75
BATCH  = 128
SEED   = 42
DATA   = "data/store.h5"
N      = 1024
D      = 64
K_ITER = 8

_loaders = None


def get_loaders():
    global _loaders
    if _loaders is None:
        tr_full, va = make_loaders(DATA, batch_size=BATCH, seed=SEED)
        n   = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
        subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
        _loaders = (tr, va)
    return _loaders


def make_resonant() -> SGNNET_Resonant:
    torch.manual_seed(SEED)
    tk   = topology_kwargs(N)
    base = SGNNET_SmallWorld(
        N_in=25088, N_hidden=N, N_out=10,
        K_local=tk["K_local"], K_random=tk["K_random"],
        K_in=tk["K_in"], K_iter=K_ITER, n_groups=tk["n_groups"],
        norm_mode="l2", D=D, encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base=base, K_phase=8, beam_size=16,
        theta_init=0.1, alpha_reflect=0.5, alpha_turing=0.3,
        mode="dynamic_z_geo", resonance_threshold=0.0, geo_gamma=0.5,
    )


def make_agr(routing_mode="soft", candidate_mode="mixed",
             K_select=6, ah_alpha=0.0, hop_decay=1.0) -> SGNNET_ActGatedRouting:
    resonant = make_resonant()
    return SGNNET_ActGatedRouting(
        base=resonant,
        routing_mode=routing_mode,
        candidate_mode=candidate_mode,
        K_select=K_select,
        ah_alpha=ah_alpha,
        hop_decay=hop_decay,
    )


def run(label: str, model: nn.Module, meta: dict) -> dict:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    tr, va  = get_loaders()
    tk      = trainer_kwargs(meta["N"], n_epochs=EPOCHS, sched_type="plateau")
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **tk)
    t0      = time.time()
    history = trainer.train(n_epochs=EPOCHS)
    elapsed = time.time() - t0
    best    = max(h.get("val_top1", 0.0) for h in history)
    last5   = history[-5:]
    best_ep = int(np.argmax([h.get("val_top1", 0.0) for h in history])) + 1
    frac    = best_ep / len(history)
    result  = {
        "label": label, "top1_best": best,
        "top1_last": history[-1].get("val_top1", 0.0),
        "final_task_loss": float(np.mean([h.get("task_loss", 0.0) for h in last5])),
        "best_epoch": best_ep, "epochs_run": len(history),
        "elapsed_s": round(elapsed, 1),
        "best_epoch_frac": round(frac, 3),
        "convergence_diag": "training_too_short" if frac < 0.7 else "converged",
        "top1_history": [round(h.get("val_top1", 0.0), 4) for h in history],
        "_meta": run_metadata(__file__, {**meta, "epochs": EPOCHS}),
    }
    print(f"  top1_best={best:.4f}  best_ep={best_ep}/{len(history)} ({frac:.0%})"
          f"  diag={result['convergence_diag']}  t={elapsed:.0f}s")
    return result


# Load REF_BASELINE from step57
_ref_path = ROOT / "results" / "train_step57_benchmark_ablation.json"
try:
    REF_BASELINE = json.loads(_ref_path.read_text()).get("top1_best", 0.7353)
    print(f"Loaded REF_BASELINE from step57: {REF_BASELINE:.4f}")
except Exception:
    REF_BASELINE = 0.7353
    print(f"step57 result not found — using fallback REF_BASELINE={REF_BASELINE:.4f}")

# (key, label, kind, kwargs)
# kind="antihebb" → baseline; kind="agr" → SGNNET_ActGatedRouting
CONFIGS = [
    ("Ref", "Ref   AntiHebb α=1.0 static routing",
     "antihebb", {}),

    ("A",   "A     soft-attn mixed-candidates no-AH",
     "agr",  dict(routing_mode="soft",  candidate_mode="mixed",
                  K_select=6, ah_alpha=0.0, hop_decay=1.0)),

    ("B",   "B     soft-attn mixed-candidates AH=1.0",
     "agr",  dict(routing_mode="soft",  candidate_mode="mixed",
                  K_select=6, ah_alpha=1.0, hop_decay=1.0)),

    ("C",   "C     hard-topK mixed-candidates AH=1.0",
     "agr",  dict(routing_mode="hard",  candidate_mode="mixed",
                  K_select=6, ah_alpha=1.0, hop_decay=1.0)),

    ("D",   "D     soft-attn mixed-candidates AH=1.0 hop_decay=0.9",
     "agr",  dict(routing_mode="soft",  candidate_mode="mixed",
                  K_select=6, ah_alpha=1.0, hop_decay=0.9)),

    ("E",   "E     soft-attn phase-only-candidates AH=1.0",
     "agr",  dict(routing_mode="soft",  candidate_mode="phase",
                  K_select=6, ah_alpha=1.0, hop_decay=1.0)),
]


if __name__ == "__main__":
    print(f"Device: {DEVICE}  Epochs: {EPOCHS}  Batch: {BATCH}  Data: 50%")
    print(f"Step 63: Activation-Gated Routing  |  REF_BASELINE={REF_BASELINE:.4f}")
    print("Soft-attention over expanded candidate set (conn_hh + conn_phase)")
    get_loaders()
    print(f"Dataset: train={len(_loaders[0].dataset)}  val={len(_loaders[1].dataset)}")

    results = {}
    for key, label, kind, kwargs in CONFIGS:
        if kind == "antihebb":
            resonant = make_resonant().to(DEVICE)
            model    = SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos").to(DEVICE)
            meta     = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "antihebb",
                        "alpha_ahebb": 1.0, "data_frac": 0.5}
        else:
            model = make_agr(**kwargs).to(DEVICE)
            meta  = {"N": N, "D": D, "K_iter": K_ITER, "mechanism": "act_gated_routing",
                     "data_frac": 0.5, **kwargs}

        results[key] = run(label, model, meta)
        results[key].update(meta)

    out = ROOT / "results" / "train_step63_act_gated_routing.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    ref_val = results.get("Ref", {}).get("top1_best", REF_BASELINE)
    print(f"\n-- Step 63: AGR (ref={ref_val:.4f} / step57={REF_BASELINE:.4f}) --")
    print(f"  {'Config':<55}  {'top1':>6}  {'vs_Ref':>8}  {'best_ep':>8}  {'t(s)':>6}")
    print("  " + "-"*90)
    for k, r in results.items():
        d = r["top1_best"] - ref_val
        print(f"  {r['label'][:55]:<55}  {r['top1_best']:>6.4f}  {d:>+8.4f}  "
              f"{r.get('best_epoch', 0):>7d}    {r['elapsed_s']:>6.0f}")

    print("\n  Key questions:")
    print("  A vs Ref → does input-dependent routing beat static?")
    print("  B vs A   → does AH compound with attention routing?")
    print("  C vs B   → hard top-K vs soft attention?")
    print("  D vs B   → does hop-count decay (0.9/step) help?")
    print("  E vs B   → phase-only candidates vs mixed pool?")
