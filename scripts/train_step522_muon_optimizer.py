"""Step 522: Muon optimizer test — measure convergence speed AND accuracy.

USER DIRECTIVE (2026-04-15)
===========================
"What happens with modern optimizers like Muon? Do not measure selection just by raw
accuracy — if we are getting same accuracy with faster convergence, that is a good win."

Muon (Kellet et al. 2024) = Momentum with orthogonalized updates via Newton-Schulz iteration.
Reported to give 2-5× speedup vs AdamW on transformers. Untested on GNN-like routing architectures.

PROTOCOL
========
Same N=2048 ΔW proj K=5 config, 50ep 50% data. Compare:
  Ref : AdamW (current baseline)
  A   : Muon (if installed via pip install muon-optimizer)

Metrics:
  - final top1_best (accuracy)
  - epochs to 93% (first-pass speed — below 20ep scout threshold)
  - epochs to 94% (mid-train speed)
  - epochs to 95% (near-ceiling speed at T1 budget)
  - wall-clock per epoch (if stable on Muon)

If Muon reaches 95% in fewer epochs at same final accuracy → paper finding: SGNNET benefits from
orthogonalized updates (unlike Adam which is rotation-equivariant only if hyperparams tuned).

DEPS
====
pip install muon-optimizer (or git clone https://github.com/KellerJordan/Muon)
Falls back to AdamW with a warning if import fails.
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

try:
    from muon import Muon           # pip install muon-optimizer
    MUON_AVAILABLE = True
    # Muon uses dist.get_world_size() internally — init process group for single-process use.
    import torch.distributed as dist
    if not dist.is_initialized():
        import os
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
except ImportError:
    MUON_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--configs", default="")
args = parser.parse_args()
DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step522_muon_optimizer.json"


class SGNNET_DeltaAH(nn.Module):
    def __init__(self, resonant: SGNNET_Resonant, alpha_ahebb: float = 0.0):
        super().__init__()
        self.m = resonant
        self.alpha_ahebb = alpha_ahebb

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase
    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z         = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh   = self.m.base.conn_hh
        N_h       = self.m.base.N_hidden
        W_h       = self.m.W_pos[:N_h]
        delta_w      = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)
        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb  = Z_fwd[:, conn_hh, :]
            proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
            Z_nb       = Z_nb * proj_coeff.abs()
            Z_struct   = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = ALPHA_REFLECT * Z_reflected + Z_remainder
            Z_new       = Z_struct + Z_reflected
            Z           = F.normalize(Z_new.clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


def build_model():
    torch.manual_seed(SEED)
    ng = max(8, N // 8)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, alpha_reflect=ALPHA_REFLECT,
                          alpha_turing=ALPHA_TURING, mode="dynamic_z_geo")
    return SGNNET_DeltaAH(res, alpha_ahebb=0.0)


def _warmup_grads(model, tr, device):
    """Run one dummy forward+backward to identify which params actually receive gradients."""
    model.train()
    batch = next(iter(tr))
    x = batch[0][:2].to(device)
    y = batch[2][:2].to(device) if len(batch) > 2 else batch[1][:2].to(device)
    model.zero_grad()
    model(x).sum().backward()
    active = {n for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return active


def build_optimizer(model, which: str, tr=None, device=None):
    """which: 'adamw' | 'muon'. Muon falls back to AdamW if not installed."""
    if which == "muon" and not MUON_AVAILABLE:
        print("  WARNING: muon-optimizer not installed — falling back to AdamW")
        which = "adamw"
    wpos_params = [model.m.base.W_pos]
    other_params = [p for n, p in model.named_parameters() if n != "m.base.W_pos" and p.requires_grad]
    if which == "adamw":
        return torch.optim.AdamW([
            {"params": wpos_params, "lr": 1e-3, "weight_decay": 0.0},
            {"params": other_params, "lr": 1e-3, "weight_decay": 1e-5},
        ])
    elif which == "muon":
        # Warmup: identify which 2D+ params actually receive gradients (avoids None-grad crash in Muon)
        if tr is not None and device is not None:
            active = _warmup_grads(model, tr, device)
            matrix_params = [p for n, p in model.named_parameters()
                             if n != "m.base.W_pos" and p.requires_grad and p.dim() >= 2 and n in active]
            vector_params = [p for n, p in model.named_parameters()
                             if n != "m.base.W_pos" and p.requires_grad and (p.dim() < 2 or n not in active)]
        else:
            # Muon handles 2D+ params; 1D use AdamW
            matrix_params = [p for p in other_params if p.dim() >= 2]
            vector_params = [p for p in other_params if p.dim() < 2]
        print(f"  Muon: matrix_params={len(matrix_params)} ({[tuple(p.shape) for p in matrix_params]}), vector_params={len(vector_params)}")
        if not matrix_params:
            print("  WARNING: no 2D+ params found for Muon — falling back to AdamW")
            return torch.optim.AdamW([
                {"params": wpos_params, "lr": 1e-3, "weight_decay": 0.0},
                {"params": other_params, "lr": 1e-3, "weight_decay": 1e-5},
            ])
        return [
            Muon(matrix_params, lr=0.02, momentum=0.95),
            torch.optim.AdamW(vector_params + wpos_params, lr=1e-3, weight_decay=1e-5),
        ]


def train_epoch(model, optimizer, loader, device):
    model.train()
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
        if isinstance(optimizer, list):
            # Muon requires grad tensors (not None) to init momentum buffer on first step.
            for opt in optimizer: opt.zero_grad(set_to_none=False)
        else:
            optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if isinstance(optimizer, list):
            for opt in optimizer: opt.step()
        else:
            optimizer.step()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
        out = model(x)
        correct += (out.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def main():
    print(f"Step 522 — Muon optimizer test")
    print(f"  Muon installed: {MUON_AVAILABLE}")

    all_keys = ["Ref_adamw", "A_muon"]
    run_keys = [k.strip() for k in args.configs.split(",")] if args.configs else all_keys

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n_half = len(tr_full.dataset) // 2
    subset = torch.utils.data.Subset(tr_full.dataset, list(range(n_half)))
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        print(f"\n{'─'*60}\nConfig {key}\n{'─'*60}")
        opt_name = "muon" if key == "A_muon" else "adamw"
        t0 = time.time()
        model = build_model().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params:,}  optimizer={opt_name}")
        optimizer = build_optimizer(model, opt_name, tr=tr, device=DEVICE)
        epochs_to = {"93": None, "94": None, "95": None}
        hist = []
        epoch_times = []
        for ep in range(EPOCHS):
            t_ep = time.time()
            train_epoch(model, optimizer, tr, DEVICE)
            v = validate(model, va, DEVICE)
            epoch_times.append(time.time() - t_ep)
            hist.append(v)
            for thr, key2 in [(0.93, "93"), (0.94, "94"), (0.95, "95")]:
                if epochs_to[key2] is None and v >= thr:
                    epochs_to[key2] = ep + 1
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1:3d}  val={v:.4f}  ep_time={epoch_times[-1]:.1f}s", flush=True)
        best = max(hist)
        elapsed = time.time() - t0
        print(f"  → best={best:.4f}  elapsed={elapsed:.0f}s")
        print(f"  convergence: ep_to_93={epochs_to['93']}  ep_to_94={epochs_to['94']}  ep_to_95={epochs_to['95']}")
        results[key] = {
            "optimizer": opt_name,
            "n_params": n_params,
            "top1_best": best,
            "epochs_to": epochs_to,
            "median_epoch_time_s": float(np.median(epoch_times)),
            "elapsed_s": elapsed,
            "history": hist,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n========== STEP 522 SUMMARY ==========")
    for key, r in results.items():
        print(f"  {key:<12}  opt={r['optimizer']:<6}  best={r['top1_best']:.4f}  "
              f"ep→93={r['epochs_to']['93']}  ep→94={r['epochs_to']['94']}  ep→95={r['epochs_to']['95']}")
    if "Ref_adamw" in results and "A_muon" in results:
        r0 = results["Ref_adamw"]; r1 = results["A_muon"]
        print(f"\nΔ(best): {(r1['top1_best']-r0['top1_best'])*100:+.2f}pp")
        if r0['epochs_to']['95'] and r1['epochs_to']['95']:
            print(f"Epochs-to-95 ratio: {r0['epochs_to']['95']/r1['epochs_to']['95']:.2f}× "
                  f"({'Muon faster' if r1['epochs_to']['95']<r0['epochs_to']['95'] else 'AdamW faster'})")


if __name__ == "__main__":
    main()
