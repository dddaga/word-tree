"""Step 514: Incremental K_hh expansion — test user's "more DoF" hypothesis NON-destructively.

MOTIVATION
==========
All prior dynamic-connectivity experiments (step511/512/513) FAILED because they REPLACE
existing edges (destructive). The user's argument is "more DoF → higher entropy capacity"
— but replacing edges doesn't add DoF, it just exchanges them.

This step ADDS edges without removing them. Starts with K_hh=2 warmup, then grows K_hh
to 3, 4, 5 at ep {15, 30, 45} — doubling+ the per-neuron connectivity during training.

If accuracy improves beyond static baseline → user's hypothesis holds: more connectivity DOF
does add compressive capacity, but ONLY when incrementally added after warmup.
If accuracy tanks → more edges beyond K_hh=2 is too much routing for this N scale (consistent
with step140 N×K result "more K hurts at low N").

PROTOCOL
========
Phase 1 (warmup 30ep): N=512 K_hh=2 AH-only static.
Phase 2 (grow 60ep):
  ep 15: add 1 edge/neuron → K_hh=3
  ep 30: add 1 edge/neuron → K_hh=4
  ep 45: add 1 edge/neuron → K_hh=5

Selection for new edges: activation-guided (user directive "not activation-agnostic"):
  Pick the candidate with HIGHEST activation correlation (specialization — these are
  neurons that already activate together, so connecting them formalizes the functional
  group). This is the "co_act_hi" rule but in ADDITIVE form.

CONFIGS (N=512 D=16)
  Ref_static  : no expansion (K_hh=2 throughout, control)
  A_expand    : grow K_hh 2→3→4→5 at ep {15,30,45}
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
parser.add_argument("--grow_epochs", type=int, default=60)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 512; N_IN = 25088; N_OUT = 10
D = 16; K_HH_START = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

# Schedule: (epoch, add_edges)
EXPAND_SCHEDULE = [(15, 1), (30, 1), (45, 1)]  # K_hh: 2 → 3 → 4 → 5
MAX_K_HH = K_HH_START + sum(n for _, n in EXPAND_SCHEDULE)

OUT_PATH = ROOT / "results" / "train_step514_khh_expansion.json"


def build_model(K_hh):
    """Build with specified K_hh. For expansion, we'll rebuild conn_hh manually."""
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_hh // 4); K_l = K_hh - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


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


def expand_conn_hh(model, corr, n_new_edges_per_neuron, n_candidates=6):
    """Add n_new_edges_per_neuron new edges per neuron, preferring HIGH-|corr| candidates.
    Existing edges are preserved; new edges are APPENDED.
    """
    conn_hh = model.m.base.conn_hh
    dev = conn_hh.device
    N_h, K_h = conn_hh.shape
    corr_abs = corr.abs().to(dev)
    new_conn = torch.zeros((N_h, K_h + n_new_edges_per_neuron), dtype=conn_hh.dtype, device=dev)
    new_conn[:, :K_h] = conn_hh  # keep existing

    n_swaps = 0
    for i in range(N_h):
        existing = set(conn_hh[i].tolist()); existing.add(i)
        candidates = []
        while len(candidates) < n_candidates:
            j = int(torch.randint(0, N_h, (1,)).item())
            if j in existing or j in candidates: continue
            candidates.append(j)
        if not candidates: continue
        cand_scores = corr_abs[i, candidates]
        # Pick top-k highest-|corr| candidates
        topk = torch.topk(cand_scores, n_new_edges_per_neuron, largest=True).indices.tolist()
        for idx, cidx in enumerate(topk):
            new_conn[i, K_h + idx] = candidates[cidx]
            n_swaps += 1

    # Replace the module's conn_hh buffer with the new shape
    # IMPORTANT: conn_hh is a buffer inside SGNNET_SmallWorld — re-register with new shape
    del model.m.base.conn_hh
    model.m.base.conn_hh = new_conn
    # Also update model_smallworld's K_hh attribute if it tracks it
    if hasattr(model.m.base, "K_hh"):
        model.m.base.K_hh = new_conn.shape[1]
    return n_swaps


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


def main():
    t_start = time.time()
    print(f"Step 514 — Incremental K_hh expansion (additive, non-destructive)")
    print(f"  N={N} K_hh_start={K_HH_START}  expand schedule={EXPAND_SCHEDULE}")
    print(f"  Final K_hh if A_expand succeeds: {MAX_K_HH}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    # Shared warmup
    print("\n--- Shared warmup phase ---")
    warmup_model = build_model(K_HH_START).to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=args.warmup_epochs)
    trainer = Trainer(model=warmup_model, train_loader=tr, val_loader=va,
                      device=DEVICE, **kw)
    for ep in range(args.warmup_epochs):
        trainer.train_epoch()
        if (ep + 1) % 5 == 0:
            v = validate_model(warmup_model, va, DEVICE)
            print(f"  [warmup] ep{ep+1:3d}  val={v:.4f}", flush=True)
    warmup_best = validate_model(warmup_model, va, DEVICE)
    print(f"  [warmup done] top1={warmup_best:.4f}")
    warmup_state = {k: v.clone().detach() for k, v in warmup_model.state_dict().items()}

    all_keys = ["Ref_static", "A_expand"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys
    results = {"warmup_best": warmup_best, "schedule": EXPAND_SCHEDULE}

    for key in run_keys:
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")
        model = build_model(K_HH_START).to(DEVICE)
        model.load_state_dict(warmup_state)
        expand_idx = 0

        kw = trainer_kwargs(N, n_epochs=args.grow_epochs)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        history = []
        total_edges_added = 0
        t0 = time.time()
        for ep in range(args.grow_epochs):
            if (key == "A_expand"
                and expand_idx < len(EXPAND_SCHEDULE)
                and ep == EXPAND_SCHEDULE[expand_idx][0]):
                n_new = EXPAND_SCHEDULE[expand_idx][1]
                print(f"  [expand at ep{ep}] adding {n_new} edge/neuron (K_hh "
                      f"{model.m.base.conn_hh.shape[1]} → {model.m.base.conn_hh.shape[1] + n_new})")
                corr = compute_activation_correlation(model, va, DEVICE, n_batches=5)
                added = expand_conn_hh(model, corr, n_new, n_candidates=6)
                total_edges_added += added
                expand_idx += 1
            trainer.train_epoch()
            v = validate_model(model, va, DEVICE)
            history.append({"epoch": ep, "val_top1": v, "K_hh": int(model.m.base.conn_hh.shape[1])})
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1:3d}  val={v:.4f}  K_hh={model.m.base.conn_hh.shape[1]}", flush=True)

        top1h = [h["val_top1"] for h in history]
        final_khh = int(model.m.base.conn_hh.shape[1])
        results[key] = {
            "top1_best": max(top1h), "top1_last": top1h[-1],
            "final_K_hh": final_khh,
            "edges_added": total_edges_added,
            "elapsed_s": time.time() - t0,
            "history": top1h,
        }
        print(f"  → {key} best={results[key]['top1_best']:.4f} "
              f"final_K_hh={final_khh} edges_added={total_edges_added}")
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n\n========== STEP 514 SUMMARY ==========")
    print(f"  Shared warmup: {warmup_best:.4f}")
    for key in run_keys:
        r = results[key]
        print(f"  {key:<12}  best={r['top1_best']:.4f}  final_K_hh={r['final_K_hh']}  edges_added={r['edges_added']}")
    if "Ref_static" in results and "A_expand" in results:
        d = (results["A_expand"]["top1_best"] - results["Ref_static"]["top1_best"]) * 100
        print(f"\nΔ(A_expand − Ref_static): {d:+.2f}pp  (more DoF hypothesis test)")
    print(f"\nTotal: {(time.time() - t_start):.0f}s")


if __name__ == "__main__":
    main()
