"""Step 512: Dynamic connectivity v2 — threshold-gated, rate-limited rewiring.

MOTIVATION
==========
step511 failed: co_act_low rule changed 90% of edges at first rewire → essentially random
rewiring because at N=512 most pair correlations are near 0 (noise-level discrimination).

Fixes in v2:
  1. **Rate limit:** replace at MOST 1 edge per neuron per rewire event (not all K_hh=2).
  2. **Threshold gate:** only swap if new candidate's |corr| is LOWER by ≥δ than worst current.
  3. **Lower frequency:** rewire every 15ep, not every 5ep.
  4. **Stability metric:** log fraction of edges that actually changed (vs the theoretical max).

Hypothesis: the user's theory holds if rewiring is guided + incremental, not random.

CONFIGS (N=512 D=16 K_hh=2 K_iter=5, post-warmup 60ep)
  Ref_static  : no rewiring (control)
  A_coact_low : rate-limited co_act_low (threshold δ=0.02)
  B_coact_hi  : rate-limited co_act_high (specialization — opposite mechanism)

Hypothesis test:
  - If A beats Ref_static by +0.5pp → diversification rewiring works
  - If B beats Ref_static by +0.5pp → specialization rewiring works
  - If both fail → dynamic connectivity at N=512 small-world is too noisy
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

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--ckpt", default="results/warmup_ckpt_n512_ep30.pt")
parser.add_argument("--rewire_epochs", type=int, default=60)
parser.add_argument("--rewire_every", type=int, default=15,
                    help="rewire conn_hh every N epochs (v2 default 15 vs v1 5)")
parser.add_argument("--k_candidates", type=int, default=4)
parser.add_argument("--max_swaps_per_neuron", type=int, default=1,
                    help="rate limit: at most this many edges replaced per neuron per rewire")
parser.add_argument("--swap_threshold", type=float, default=0.02,
                    help="only swap if new candidate |corr| is lower by ≥ threshold")
parser.add_argument("--configs", default="", help="Ref_static,A_coact_low,B_coact_hi. Empty = all.")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step512_dynconn_v2.json"


def build_model():
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


def load_warmup_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  loaded checkpoint: ep {ckpt.get('epochs_trained', '?')}, "
          f"best top1 {ckpt.get('top1_best', 0):.4f}")
    return model


@torch.no_grad()
def compute_activation_correlation(model, loader, device, n_batches=5):
    model.eval()
    feats = []
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0].to(device)
        Z = model.m.base._seed(x)
        conn_hh = model.m.base.conn_hh
        theta_pos = model.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        W_h = model.m.W_pos[:N]; W_n = F.normalize(W_h, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - ALPHA_AHEBB * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)
        Z_reflected = torch.zeros_like(Z)
        Z_fwd_last = None
        for k in range(K_ITER):
            Z_fwd = F.relu(Z - theta_pos)
            if k == K_ITER - 1: Z_fwd_last = Z_fwd
            Z_nb = Z_fwd[:, conn_hh, :] * supp_w
            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)
        feats.append(Z_fwd_last.norm(dim=-1).cpu())
    A = torch.cat(feats, dim=0)
    A_c = A - A.mean(dim=0, keepdim=True)
    std = A_c.std(dim=0, keepdim=True).clamp(min=1e-6)
    A_n = A_c / std
    n_samp = A_n.shape[0]
    return (A_n.T @ A_n) / (n_samp - 1)


def rewire_rate_limited(model, corr, rule: str, k_candidates: int,
                        max_swaps_per_neuron: int, threshold: float):
    """Rate-limited rewiring: at most `max_swaps_per_neuron` edges per neuron per event.
    rule: "low" → prefer lower |corr|; "high" → prefer higher |corr|.
    threshold: only swap if new edge's |corr| is `threshold` better than worst current edge.
    """
    conn_hh = model.m.base.conn_hh
    dev = conn_hh.device
    N_h, K_h = conn_hh.shape
    corr_abs = corr.abs().to(dev)
    swaps = 0
    attempts = 0

    for i in range(N_h):
        current = conn_hh[i].tolist()
        current_scores = corr_abs[i, current]

        # Sample candidates (not in current, not self)
        pool = set(current)
        pool.add(i)
        candidates = []
        while len(candidates) < (k_candidates - K_h):
            j = int(torch.randint(0, N_h, (1,)).item())
            if j in pool: continue
            candidates.append(j); pool.add(j)
        if not candidates: continue

        cand_scores = corr_abs[i, candidates]

        # For "low": we want to REPLACE highest-|corr| current edge with lowest-|corr| candidate
        # For "high": REPLACE lowest-|corr| current with highest-|corr| candidate
        for _ in range(max_swaps_per_neuron):
            attempts += 1
            if rule == "low":
                worst_idx = int(torch.argmax(current_scores).item())
                best_cand_idx = int(torch.argmin(cand_scores).item())
                worst_cur = float(current_scores[worst_idx].item())
                best_cand = float(cand_scores[best_cand_idx].item())
                if worst_cur - best_cand > threshold:
                    # Swap
                    old_j = current[worst_idx]
                    new_j = candidates[best_cand_idx]
                    current[worst_idx] = new_j
                    current_scores[worst_idx] = cand_scores[best_cand_idx]
                    # Remove used candidate
                    candidates.pop(best_cand_idx)
                    cand_scores = torch.cat([cand_scores[:best_cand_idx], cand_scores[best_cand_idx+1:]])
                    swaps += 1
                    if len(candidates) == 0: break
                else:
                    break  # no improvement worth threshold
            else:  # "high"
                best_cur_idx = int(torch.argmin(current_scores).item())
                best_cand_idx = int(torch.argmax(cand_scores).item())
                best_cur = float(current_scores[best_cur_idx].item())
                best_cand = float(cand_scores[best_cand_idx].item())
                if best_cand - best_cur > threshold:
                    current[best_cur_idx] = candidates[best_cand_idx]
                    current_scores[best_cur_idx] = cand_scores[best_cand_idx]
                    candidates.pop(best_cand_idx)
                    cand_scores = torch.cat([cand_scores[:best_cand_idx], cand_scores[best_cand_idx+1:]])
                    swaps += 1
                    if len(candidates) == 0: break
                else:
                    break

        conn_hh[i] = torch.tensor(current, device=dev, dtype=conn_hh.dtype)
    return {"swaps": swaps, "attempts": attempts, "swap_rate": swaps / max(attempts, 1)}


def validate_model(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
            out = model(x)
            correct += (out.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / max(total, 1)


def run_config(key: str, rule: str | None, tr, va):
    print(f"\n{'─'*60}\nConfig {key} (rule={rule})\n{'─'*60}")
    ckpt_path = ROOT / args.ckpt
    model = build_model().to(DEVICE)
    model = load_warmup_checkpoint(model, ckpt_path)
    initial_conn = model.m.base.conn_hh.clone()

    kw = trainer_kwargs(N, n_epochs=args.rewire_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)

    t0 = time.time()
    history = []; rewire_log = []
    for ep in range(args.rewire_epochs):
        if (rule is not None and ep > 0 and ep % args.rewire_every == 0):
            corr = compute_activation_correlation(model, va, DEVICE, n_batches=5)
            rw = rewire_rate_limited(
                model, corr, rule=rule,
                k_candidates=args.k_candidates,
                max_swaps_per_neuron=args.max_swaps_per_neuron,
                threshold=args.swap_threshold,
            )
            diff = (model.m.base.conn_hh != initial_conn).float().mean().item()
            rewire_log.append({"epoch": ep, "swaps": rw["swaps"], "diff": diff})
            print(f"  [rewire at ep{ep}] swaps={rw['swaps']} diff_from_initial={diff*100:.1f}%")

        trainer.train_epoch()
        val_top1 = validate_model(model, va, DEVICE)
        history.append({"epoch": ep, "val_top1": val_top1})
        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1:3d}  val={val_top1:.4f}", flush=True)

    top1h = [h["val_top1"] for h in history]
    return {
        "top1_best": max(top1h), "top1_last": top1h[-1],
        "rewire_log": rewire_log,
        "total_rewires": len(rewire_log),
        "total_swaps": sum(r["swaps"] for r in rewire_log),
        "elapsed_s": time.time() - t0,
        "history": top1h,
    }


def main():
    t_start = time.time()
    all_keys = ["Ref_static", "A_coact_low", "B_coact_hi"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    print(f"Step 512 — Dynamic connectivity v2 (threshold-gated, rate-limited)")
    print(f"  N={N} rewire_every={args.rewire_every} max_swaps/neuron={args.max_swaps_per_neuron}"
          f" threshold={args.swap_threshold}")
    print(f"  k_candidates={args.k_candidates}  rewire_epochs={args.rewire_epochs}")
    print(f"  Running: {run_keys}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    rules = {"Ref_static": None, "A_coact_low": "low", "B_coact_hi": "high"}
    for key in run_keys:
        results[key] = run_config(key, rules.get(key), tr, va)
        print(f"  → {key} best={results[key]['top1_best']:.4f}  "
              f"swaps={results[key]['total_swaps']}")
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n========== STEP 512 SUMMARY ==========")
    for key, r in results.items():
        print(f"  {key:<14}  best={r['top1_best']:.4f}  rewires={r['total_rewires']}"
              f" swaps={r['total_swaps']}")
    if "Ref_static" in results:
        ref_best = results["Ref_static"]["top1_best"]
        for k in ["A_coact_low", "B_coact_hi"]:
            if k in results:
                d = (results[k]["top1_best"] - ref_best) * 100
                print(f"  Δ({k} − Ref_static): {d:+.2f}pp")
    print(f"\nTotal: {(time.time() - t_start):.0f}s")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
