"""Step 521: Multi-forward-backward K_iter — deep supervision on routing trajectory.

USER DIRECTIVE (2026-04-15)
===========================
Proposed protocol:
  iter 0: feed input
  iter 1 to k_only_forward: forward pass only (no loss)
  iter k_only_forward+1 to K_iter: forward + backward per step
  → multiple forward + multiple backward per sample

Two natural interpretations:
  A) Multiple backwards, each with opt.step() in between (weights updated mid-routing)
     — Hard to train stably. Weight drift corrupts the in-progress forward trajectory.
  B) Accumulate loss at each supervised step, single backward + single opt.step() at end
     — Equivalent to "deep supervision" (Lee et al. 2015). Well-behaved training signal.
     This is the version implemented here.

If (B) works, future step522 can try variant (A) with per-step opt.step().

CONFIGS (N=2048 D=16 K_hh=2 K_iter=5 ΔW proj, 50% data, 30ep T0 scout)
  Ref           : standard single-loss at K_iter=5 (baseline)
  A_ds_3to5     : deep supervision at iters {3, 4, 5} (k_only_forward=2)
  B_ds_2to5     : deep supervision at iters {2, 3, 4, 5} (k_only_forward=1)
  C_ds_4to5     : deep supervision only at iters {4, 5} (k_only_forward=3)
  D_ds_all      : deep supervision at every iter (k_only_forward=0, aggressive)

If A_ds_3to5 or B_ds_2to5 beats Ref by +0.5pp → deep supervision helps routing.
If all negative → routing trajectory already well-supervised by final loss (most likely).
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
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step521_multi_fwd_bwd.json"

# (key, k_only_forward, description)
CONFIGS = [
    dict(key="Ref",       k_only_forward=K_ITER, label="single loss at K_iter=5 (baseline)"),
    dict(key="A_ds_3to5", k_only_forward=2,      label="DS at iters {3,4,5}"),
    dict(key="B_ds_2to5", k_only_forward=1,      label="DS at iters {2,3,4,5}"),
    dict(key="C_ds_4to5", k_only_forward=3,      label="DS at iters {4,5}"),
    dict(key="D_ds_all",  k_only_forward=0,      label="DS at all iters {1..5}"),
]


class SGNNET_DeltaAH_DS(nn.Module):
    """ΔW proj with deep supervision — returns loss accumulated across K_iter steps ≥ k_only_forward."""
    def __init__(self, resonant: SGNNET_Resonant, k_only_forward: int = K_ITER):
        super().__init__()
        self.m = resonant
        self.k_only_forward = k_only_forward

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x, return_all_logits: bool = False):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden
        W_h       = self.m.W_pos[:N_h]
        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)

        intermediate_logits = []
        K_loop = self.m.base.K_iter
        for k in range(K_loop):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()
            Z_struct   = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)
            if k + 1 >= self.k_only_forward:
                intermediate_logits.append(self.m.base._readout(Z))

        if return_all_logits:
            return intermediate_logits
        return intermediate_logits[-1]  # for inference/validation — use final routing output


def build_model(k_only_forward):
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_DeltaAH_DS(res, k_only_forward=k_only_forward)


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0; n_batches = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits_list = model(x, return_all_logits=True)
        # Sum of CE at each supervised step, averaged
        losses = [F.cross_entropy(l, y) for l in logits_list]
        loss = sum(losses) / len(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
        out = model(x, return_all_logits=False)
        correct += (out.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main():
    keys = [c["key"] for c in CONFIGS]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else keys
    active = [c for c in CONFIGS if c["key"] in run_keys]

    print(f"Step 521 — Multi-fwd-bwd deep supervision on K_iter")
    print(f"  N={N} K_iter={K_ITER} {EPOCHS}ep 50% T0")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg['key']}: {cfg['label']}\n{'─'*60}")
        t0 = time.time()
        model = build_model(cfg["k_only_forward"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  k_only_forward={cfg['k_only_forward']}")

        param_groups = [{"params": [model.m.base.W_pos], "lr": 1e-3, "weight_decay": 0.0}]
        other = [p for n, p in model.named_parameters() if n != "m.base.W_pos" and p.requires_grad]
        if other:
            param_groups.append({"params": other, "lr": 1e-3, "weight_decay": 1e-5})
        optimizer = torch.optim.AdamW(param_groups)

        best = 0.0; hist = []
        for ep in range(EPOCHS):
            tl = train_one_epoch(model, optimizer, tr, DEVICE)
            v = validate(model, va, DEVICE)
            best = max(best, v); hist.append(v)
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1:3d}  train_loss={tl:.4f}  val={v:.4f}", flush=True)

        elapsed = time.time() - t0
        print(f"  → best={best:.4f}  elapsed={elapsed:.0f}s")
        results[cfg["key"]] = {
            "label": cfg["label"], "k_only_forward": cfg["k_only_forward"],
            "n_params": n_params, "top1_best": best, "history": hist,
            "elapsed_s": elapsed,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n========== STEP 521 SUMMARY ==========")
    for cfg in active:
        r = results.get(cfg["key"], {})
        if not r: continue
        print(f"  {cfg['key']:<12}  k_only_fwd={r['k_only_forward']}  best={r['top1_best']:.4f}")
    if "Ref" in results:
        ref = results["Ref"]["top1_best"]
        for k in ["A_ds_3to5", "B_ds_2to5", "C_ds_4to5", "D_ds_all"]:
            if k in results:
                d = (results[k]["top1_best"] - ref) * 100
                print(f"  Δ({k} − Ref): {d:+.2f}pp")


if __name__ == "__main__":
    main()
