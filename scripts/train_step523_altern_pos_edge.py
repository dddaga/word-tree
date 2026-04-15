"""Step 520: Alternating W_pos-only / edge-only training cycle.

USER DIRECTIVE (2026-04-15)
===========================
step511-514 all failed because W_pos and conn_hh were co-updated during rewiring — the
suppression/readout parameters co-adapted to both the current topology AND the current
W_pos, so any change to either disrupts the other.

NEW PROTOCOL
============
Phase 1 (warmup):      30ep static (both W_pos + edges trainable)
Phase 2 (pos-only):    20ep — only W_pos (+ θ + C_ho) trainable. conn_hh FROZEN.
Phase 3 (edges-only):  20ep — only conn_hh rewiring. W_pos + θ + C_ho FROZEN.
Phase 4 repeat 2-3...  2-3 cycles total

EDGE CONSTRAINT (user-specified)
================================
Each rewire event changes AT MOST 1% of total edges (N * K_hh * 0.01).
At N=512 K_hh=2: total edges = 1024, so max 10 swaps per event.
Rewire every epoch during edges-only phase → 20 events × 10 swaps = 200 total edge changes over edges phase.

MECHANISMS (run in parallel slots to compare)
=============================================
  A) co_act_low    : prefer LOW |corr(Z_i, Z_j)|  (diversification)
  B) co_act_high   : prefer HIGH |corr(Z_i, Z_j)| (specialization)
  C) grad_cond     : prefer high |∂L/∂edge_weight| (signal-flow-guided)
  D) var_gated     : prefer neighbors with high σ_i (information-dense)

This script implements A/B mechanisms. C/D go in step521/522.
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
parser.add_argument("--warmup_epochs", type=int, default=30)
parser.add_argument("--pos_epochs", type=int, default=20)
parser.add_argument("--edge_epochs", type=int, default=20)
parser.add_argument("--cycles", type=int, default=2)
parser.add_argument("--edge_frac", type=float, default=0.01,
                    help="Max fraction of edges to swap per rewire event (default 0.01 = 1 percent)")
parser.add_argument("--configs", default="Ref,A_low,B_high",
                    help="Ref (no rewiring), A_low (co-act low), B_high (co-act high)")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

OUT_PATH = ROOT / "results" / "train_step520_altern_pos_edge.json"

TOTAL_EDGES = N * K_HH
MAX_SWAPS_PER_EVENT = max(1, int(TOTAL_EDGES * args.edge_frac))  # user: 1% → ~10 at N=512 K=2


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


def set_pos_trainable(model, enable: bool):
    """Freeze/unfreeze all parameters except conn_hh (which is a buffer, not trainable anyway)."""
    for p in model.parameters():
        p.requires_grad_(enable)


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


def rewire_global_topk(model, corr, rule: str, max_swaps: int):
    """Select up to `max_swaps` edges GLOBALLY (across all neurons) whose replacement
    gives the largest score improvement. Each edge swap: replace (src, current_dst) with
    (src, best_candidate_dst) from a random candidate pool.

    rule: "low"  → minimize |corr| (diversification); replace high-|corr| edges with low-|corr| candidates
          "high" → maximize |corr| (specialization);  replace low-|corr|  edges with high-|corr| candidates
    """
    conn_hh = model.m.base.conn_hh
    dev = conn_hh.device
    N_h, K_h = conn_hh.shape
    corr_abs = corr.abs().to(dev)

    # For each edge (i, k), compute best swap gain: |corr(i, current)| - |corr(i, best_candidate)|
    # (for "low" rule; reversed for "high")
    K_CAND = 4  # candidates per edge
    swap_candidates = []  # list of (gain, i, k, new_j)
    for i in range(N_h):
        current = conn_hh[i].tolist()
        pool = set(current); pool.add(i)
        cands = []
        while len(cands) < K_CAND:
            j = int(torch.randint(0, N_h, (1,)).item())
            if j in pool: continue
            cands.append(j); pool.add(j)
        if not cands: continue
        cand_scores = corr_abs[i, cands]   # [K_CAND]
        cur_scores = corr_abs[i, current]  # [K_h]
        if rule == "low":
            best_cand_idx = int(torch.argmin(cand_scores).item())
            worst_cur_idx = int(torch.argmax(cur_scores).item())
            gain = float(cur_scores[worst_cur_idx] - cand_scores[best_cand_idx])
        else:  # "high"
            best_cand_idx = int(torch.argmax(cand_scores).item())
            worst_cur_idx = int(torch.argmin(cur_scores).item())
            gain = float(cand_scores[best_cand_idx] - cur_scores[worst_cur_idx])
        if gain > 0:
            swap_candidates.append((gain, i, worst_cur_idx, cands[best_cand_idx]))

    # Sort by gain desc; take top max_swaps
    swap_candidates.sort(key=lambda x: -x[0])
    applied = 0
    for gain, i, k, new_j in swap_candidates[:max_swaps]:
        conn_hh[i, k] = new_j
        applied += 1
    return applied


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
    """Run the alternating phase cycle."""
    print(f"\n{'─'*60}\nConfig {key} (rule={rule})\n{'─'*60}")
    model = build_model().to(DEVICE)
    initial_conn = None
    history = []
    t0 = time.time()

    # Phase 1: warmup (all trainable)
    print(f"  [warmup] {args.warmup_epochs}ep static")
    set_pos_trainable(model, True)
    kw = trainer_kwargs(N, n_epochs=args.warmup_epochs)
    trainer = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw)
    for ep in range(args.warmup_epochs):
        trainer.train_epoch()
        v = validate_model(model, va, DEVICE)
        history.append({"phase": "warmup", "epoch": ep, "val_top1": v})
        if (ep + 1) % 5 == 0:
            print(f"    warmup ep{ep+1:3d}  val={v:.4f}", flush=True)
    initial_conn = model.m.base.conn_hh.clone()
    warmup_best = max(h["val_top1"] for h in history)
    print(f"    warmup best: {warmup_best:.4f}")

    # Phases 2-3 × cycles
    total_swaps = 0
    for cyc in range(args.cycles):
        # Phase 2: pos-only (conn_hh frozen automatically since it's a buffer)
        print(f"  [cycle {cyc+1}] pos-only {args.pos_epochs}ep")
        set_pos_trainable(model, True)
        kw_p = trainer_kwargs(N, n_epochs=args.pos_epochs)
        trainer_p = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw_p)
        for ep in range(args.pos_epochs):
            trainer_p.train_epoch()
            v = validate_model(model, va, DEVICE)
            history.append({"phase": f"pos_c{cyc+1}", "epoch": ep, "val_top1": v})
            if (ep + 1) % 5 == 0:
                print(f"    pos ep{ep+1:3d}  val={v:.4f}", flush=True)

        # Phase 3: edges-only (freeze all model params, rewire once per epoch)
        if rule is None:
            # Ref: no rewiring; just continue training (treat as more pos_epochs)
            print(f"  [cycle {cyc+1}] Ref_static: continuing pos training for edge_epochs")
            for ep in range(args.edge_epochs):
                trainer_p.train_epoch()
                v = validate_model(model, va, DEVICE)
                history.append({"phase": f"ref_c{cyc+1}", "epoch": ep, "val_top1": v})
        else:
            print(f"  [cycle {cyc+1}] edges-only {args.edge_epochs}ep  (max {MAX_SWAPS_PER_EVENT} swaps/ep)")
            set_pos_trainable(model, False)  # freeze everything else
            for ep in range(args.edge_epochs):
                corr = compute_activation_correlation(model, va, DEVICE, n_batches=5)
                swaps = rewire_global_topk(model, corr, rule=rule, max_swaps=MAX_SWAPS_PER_EVENT)
                total_swaps += swaps
                v = validate_model(model, va, DEVICE)
                diff = (model.m.base.conn_hh != initial_conn).float().mean().item()
                history.append({
                    "phase": f"edge_c{cyc+1}", "epoch": ep, "val_top1": v,
                    "swaps": swaps, "diff_from_initial": diff,
                })
                if (ep + 1) % 5 == 0:
                    print(f"    edge ep{ep+1:3d}  val={v:.4f}  swaps={swaps}  diff={diff*100:.1f}%", flush=True)

    # Final pos training to re-stabilize
    print(f"  [final pos] 10ep")
    set_pos_trainable(model, True)
    kw_f = trainer_kwargs(N, n_epochs=10)
    trainer_f = Trainer(model=model, train_loader=tr, val_loader=va, device=DEVICE, **kw_f)
    for ep in range(10):
        trainer_f.train_epoch()
        v = validate_model(model, va, DEVICE)
        history.append({"phase": "final_pos", "epoch": ep, "val_top1": v})

    top1h = [h["val_top1"] for h in history]
    return {
        "top1_best": max(top1h), "top1_last": top1h[-1],
        "warmup_best": warmup_best,
        "total_swaps": total_swaps,
        "elapsed_s": time.time() - t0,
        "history_phases": history,
    }


def main():
    t_start = time.time()
    print(f"Step 520 — Alternating W_pos/edge training cycle")
    print(f"  N={N} K_hh={K_HH}  edge_frac={args.edge_frac} (max {MAX_SWAPS_PER_EVENT} swaps/event)")
    print(f"  warmup={args.warmup_epochs}ep  cycles={args.cycles}  pos={args.pos_epochs}ep  edges={args.edge_epochs}ep")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    run_keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    rules = {"Ref": None, "A_low": "low", "B_high": "high"}

    results = {}
    for key in run_keys:
        results[key] = run_config(key, rules.get(key), tr, va)
        print(f"  → {key} best={results[key]['top1_best']:.4f}")
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n\n========== STEP 520 SUMMARY ==========")
    for key, r in results.items():
        print(f"  {key:<8}  best={r['top1_best']:.4f}  warmup_best={r['warmup_best']:.4f}  swaps={r['total_swaps']}")
    if "Ref" in results:
        ref = results["Ref"]["top1_best"]
        for k in ["A_low", "B_high"]:
            if k in results:
                d = (results[k]["top1_best"] - ref) * 100
                print(f"  Δ({k} − Ref): {d:+.2f}pp")
    print(f"\nTotal: {(time.time() - t_start):.0f}s")


if __name__ == "__main__":
    main()
