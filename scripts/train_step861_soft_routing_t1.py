"""Step 861: Soft routing T1 — confirm step859 B_soft_anneal +1.22pp signal.

# CUDA-5060ti-validated
# pin_memory=True  non_blocking=True  (Trainer handles non_blocking internally)

MOTIVATION
==========
step859 T0 (20ep, 50% data):
  Ref        : 0.9167 (hard gather + AH)
  A_soft_β1  : 0.9090 (-0.76pp) — β=1 fixed hurts
  B_soft_anneal: 0.9289 (+1.22pp!!) — β annealing is the key mechanism

+1.22pp T0 is the strongest new signal in this session. T1 at 75ep/50% data
will determine if this holds (expected ~+0.8pp due to T0 overfit compression).

Also promotes C_soft_ah and D_soft_dwproj from step859 if they showed positive.
Runs the full compound once T1 signal is clear.

CONFIGS (T1: 75ep, 50% data, seed=42)
  Ref        : hard gather + AH (T1 reference)
  B_soft_anneal : soft β: 0.5→5.0 (wider anneal for T1, more epochs to warm)
  C_soft_ah  : B + AntiHebbian (from step859 T0)
  D_soft_dwproj : B + ΔW-proj (from step859 T0)
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
from src.sgnnet.model_resonant_cuda   import SGNNET_Resonant_CUDA, SGNNET_AntiHebbian_CUDA
from src.training.trainer             import Trainer
from src.training.experiment_config   import trainer_kwargs
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser(description="Step 861: Soft routing T1")
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,B_soft_anneal,C_soft_ah,D_soft_dwproj")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0
K_r = max(1, K_HH // 4); K_l = K_HH - K_r

SLOT = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step861_soft_routing_t1_seed{SEED}__{SLOT}.json"

STEP859_T0_SOFT_ANNEAL = 0.9289  # +1.22pp T0 target to beat


class SGNNET_SoftRoute(nn.Module):
    """Distance-weighted soft routing over static conn_hh neighbors."""

    def __init__(self, base: SGNNET_SmallWorld, beta: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("_beta", torch.tensor(beta, dtype=torch.float32))

    def set_beta(self, beta: float): self._beta.fill_(beta)

    @property
    def W_pos(self):   return self.base.W_pos
    @property
    def W_phase(self): return self.base.W_phase
    @property
    def conn_hh(self): return self.base.conn_hh
    @conn_hh.setter
    def conn_hh(self, v): self.base.conn_hh = v
    @property
    def N_hidden(self): return self.base.N_hidden
    @property
    def C_ho_mask(self): return self.base.C_ho_mask
    @property
    def K_iter(self):    return self.base.K_iter

    def _normalise(self, Z): return self.base._normalise(Z)
    def _readout(self, Z):   return self.base._readout(Z)
    def _seed(self, x):      return self.base._seed(x)

    def _route(self, Z: torch.Tensor) -> torch.Tensor:
        conn = self.base.conn_hh
        W_nb = self.base.W_pos[:self.base.N_hidden][conn]  # [N, K_hh, D]
        beta = self._beta.item()
        for _ in range(self.base.K_iter):
            Z_nb = Z[:, conn, :]
            diff = Z.unsqueeze(2) - W_nb.unsqueeze(0)      # [B, N, K_hh, D]
            dist = diff.pow(2).sum(dim=-1)                  # [B, N, K_hh]
            w = F.softmax(-beta * dist, dim=2)
            Z = (Z_nb * w.unsqueeze(-1)).sum(2)
            Z = self._normalise(Z)
        return Z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self._seed(x); Z = self._route(Z); return self._readout(Z)


class BetaScheduler:
    def __init__(self, model, beta_start, beta_end, total_epochs):
        self.model = model; self.beta_start = beta_start
        self.beta_end = beta_end; self.total = total_epochs; self._epoch = 0

    def step(self):
        frac = min(1.0, self._epoch / max(1, self.total - 1))
        beta = self.beta_start + frac * (self.beta_end - self.beta_start)
        for m in self.model.modules():
            if isinstance(m, SGNNET_SoftRoute):
                m.set_beta(beta); break
        self._epoch += 1
        return beta


class SGNNET_DeltaAH(nn.Module):
    """ΔW-proj wrapper for soft routing."""
    def __init__(self, base, mode="proj"):
        super().__init__()
        self.m = base; self.mode = mode
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        soft = None
        for mod in self.m.modules():
            if isinstance(mod, SGNNET_SoftRoute):
                soft = mod; break
        if soft is None:
            return self.m(x)
        Z = soft._seed(x)
        conn_hh = soft.base.conn_hh
        W_h = soft.base.W_pos[:soft.base.N_hidden]
        dw = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        theta_pos = self.rotation_temp.abs().unsqueeze(0).unsqueeze(-1)
        Z_ref = torch.zeros_like(Z)
        for _ in range(soft.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]
            proj = (Z_nb * dw).sum(-1, keepdim=True)
            Z_nb = Z_nb * proj.abs()
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z = F.normalize((Z_nb.sum(2) + Z_ref).clamp(-10, 10), dim=-1)
        return soft._readout(Z)


def make_model(routing: str, wrap: str):
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier", sparsity=0.90)

    if routing == "soft":
        core = SGNNET_SoftRoute(base, beta=0.5)
    else:
        core = base

    if wrap == "dw":
        res = SGNNET_Resonant(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                               beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
        return SGNNET_DeltaAH(res, mode="proj")
    elif wrap == "ah":
        if DEVICE.type == "cuda":
            res = SGNNET_Resonant_CUDA(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                                        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                                        resonance_threshold=0.0, compile=False)
            return SGNNET_AntiHebbian_CUDA(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos", compile=False)
        else:
            res = SGNNET_Resonant(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                                   beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    else:
        if DEVICE.type == "cuda":
            res = SGNNET_Resonant_CUDA(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
                                        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                                        resonance_threshold=0.0, compile=False)
            return SGNNET_AntiHebbian_CUDA(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos", compile=False)
        else:
            res = SGNNET_Resonant(core, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=ALPHA_TURING,
                                   beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0)
            return SGNNET_AntiHebbian(res, alpha_ahebb=ALPHA_AHEBB, variant="wpos")


CONFIGS = {
    "Ref":           ("hard", "ah",  "hard gather + AH (T1 reference)"),
    "B_soft_anneal": ("soft", None,  "soft β: 0.5→5.0 anneal, no AH/ΔW — T0 winner"),
    "C_soft_ah":     ("soft", "ah",  "soft anneal + AntiHebbian"),
    "D_soft_dwproj": ("soft", "dw",  "soft anneal + ΔW-proj"),
}


def main():
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    tr_full, va = make_loaders(str(data_path), batch_size=BATCH, seed=SEED,
                               pin_memory=(DEVICE.type == "cuda"))
    n_full = len(tr_full.dataset)
    sub_idx = torch.randperm(n_full, generator=torch.Generator().manual_seed(SEED))[:n_full // 2]
    tr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tr_full.dataset, sub_idx.tolist()),
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    print(f"\n{'='*70}")
    print(f"step861 — Soft routing T1 (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  T0 baseline: B_soft_anneal={STEP859_T0_SOFT_ANNEAL:.4f} (+1.22pp vs Ref)")
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")
    print(f"{'='*70}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIGS:
            print(f"  skip {key}"); continue
        routing, wrap, desc = CONFIGS[key]
        model = make_model(routing, wrap)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{'─'*60}\n{key}: {desc}  params={n_p:,}")

        scheduler = None
        if routing == "soft":
            scheduler = BetaScheduler(model, 0.5, 5.0, EPOCHS)

        kw = trainer_kwargs(N, n_epochs=EPOCHS)

        def make_log_fn(key, sched):
            def log_fn(m):
                ep = m['epoch'] + 1
                beta_str = ""
                if sched:
                    beta = sched.step()
                    beta_str = f"  β={beta:.2f}"
                if ep % 15 == 0:
                    print(f"  ep{ep:3d}  val={m['val_top1']:.4f}{beta_str}", flush=True)
            return log_fn

        t0 = time.time()
        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS,
                                                     log_fn=make_log_fn(key, scheduler))
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc or 0.94)
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ_vs_Ref={delta*100:+.2f}pp  {elapsed:.0f}s")

        results[key] = {"routing": routing, "wrap": str(wrap), "label": desc,
                        "n_params": n_p, "best": best, "best_ep": best_ep,
                        "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed)}
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 861 SUMMARY — Soft routing T1")
    print(f"{'='*70}")
    print(f"  {'config':<18} {'best':>7} {'Δ_vs_Ref':>10}")
    for k, r in results.items():
        print(f"  {k:<18} {r['best']:>7.4f} {r['delta_vs_ref']*100:>+9.2f}pp")
    soft = results.get("B_soft_anneal", {}).get("delta_vs_ref", None)
    if soft is not None:
        verdict = "NEW DEFAULT CANDIDATE" if soft >= 0.005 else f"T0 compressed — {soft*100:+.2f}pp T1"
        print(f"\n  Soft routing T1 verdict: {verdict}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
