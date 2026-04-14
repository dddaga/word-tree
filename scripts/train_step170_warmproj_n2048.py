"""Step 170: warm+W_proj at N=2048 D=16 — efficiency track scaling test.

MOTIVATION
==========
step167-B (N=1024 D=16 full data): 93.17% — ceiling, phase-exit not achievable at D=16.
step169 (N=1024 D=32 full data): ep60=93.07%, trending toward 95%.

N=2048 D=16 fills the gap:
  - step72 N-scaling: N=2048 scratch = 93.38% at full data
  - warm+W_proj added +10.32pp at N=1024 D=16 (50% data)
  - If +8pp at N=2048: 93.38% + 8pp = ~101%? (capped at 100%, realistically 95%+)
  - FLOPs at N=2048 D=16 K_iter=8 K_hh=8 ≈ 6.2M (≈ 5% of VGG FC, right at budget)

Phase-exit criterion: ≥95% @ ≤6.18M FLOPs.
N=2048 D=16 is the N-scaling complement to N=1024 D=32.

CONFIGS (N=2048, D=16, K_hh=8, K_iter=8, AH=1.0, 50% data, 75ep — Tier-1)
==========================================================================
  Ref : K=8 scratch (N=2048 D=16 baseline)
  B   : warm+W_proj (teacher K=12 N=2048 D=16)

Teacher trained fresh at N=2048 D=16 K=12, cached at results/train_step170_teacher_n2048.pt.

To reproduce:
    python -u scripts/train_step170_warmproj_n2048.py --device cpu
    python -u scripts/train_step170_warmproj_n2048.py --device mps
    python -u scripts/train_step170_warmproj_n2048.py --configs B --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--configs", default="")
parser.add_argument("--force-retrain-teacher", action="store_true")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 8; K_IN = 25; K_ITER = 8; K_ITER_TEACHER = 12
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0; ALPHA_AHEBB = 1.0

TEACHER_PATH = ROOT / "results" / "train_step170_teacher_n2048.pt"
OUT_PATH     = ROOT / "results" / "train_step170_warmproj_n2048.json"


class SGNNET_WarmProj(nn.Module):
    def __init__(self, resonant, alpha_ahebb=1.0):
        super().__init__()
        self.m = resonant; self.alpha_ahebb = alpha_ahebb
        self.proj = nn.Linear(resonant.base.D, resonant.base.D, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    @property
    def W_pos(self): return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def warm_load(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"    warm_load: {len([k for k in state if k not in unexpected])} keys, "
              f"{len(missing)} new, {len(unexpected)} teacher-only")

    def forward(self, x):
        base = self.m.base; Z = self.proj(base._seed(x))
        theta = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn  = base.conn_hh; N_h = base.N_hidden
        W_n   = F.normalize(self.m.W_pos[:N_h], dim=-1)
        supp  = (1.0 - self.alpha_ahebb * (W_n.unsqueeze(1)*W_n[conn]).sum(-1).clamp(min=0)
                ).unsqueeze(0).unsqueeze(-1)
        Zr = torch.zeros_like(Z)
        for _ in range(base.K_iter):
            Zf = F.relu(Z - theta)
            Zs = (Zf[:, conn, :] * supp).sum(2)
            Zr = self.m.alpha_reflect * Zr + (Zf - Z)
            Z  = F.normalize((Zs + Zr).clamp(-10, 10), dim=-1)
        return base._readout(Z)


@dataclass
class Config:
    key: str; label: str; warm: bool; proj: bool

CONFIGS = [
    Config("Ref", "Ref  K=8 scratch N=2048 D=16",                  False, False),
    Config("B",   "B    warm+W_proj N=2048 D=16 ← phase-exit test", True,  True),
]

_loaders_cache = None
def get_loaders():
    global _loaders_cache
    if _loaders_cache is None:
        tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
        n = len(tr_full.dataset)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n//2]
        sub = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
        _loaders_cache = (torch.utils.data.DataLoader(sub, batch_size=BATCH, shuffle=True, num_workers=0), va)
    return _loaders_cache

def _build(k_iter, seed=SEED):
    torch.manual_seed(seed)
    K_r = max(1, K_HH//4); K_l = K_HH - K_r; ng = max(8, N//8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=k_iter, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    return SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                           alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                           mode="dynamic_z_geo", resonance_threshold=0.0)

def get_teacher(force=False):
    if not force and TEACHER_PATH.exists():
        print(f"  Loading teacher from {TEACHER_PATH}")
        return torch.load(TEACHER_PATH, map_location=DEVICE, weights_only=True)
    print(f"\nTraining N=2048 D=16 teacher K={K_ITER_TEACHER} for {EPOCHS}ep...")
    t = SGNNET_AntiHebbian(_build(K_ITER_TEACHER), alpha_ahebb=ALPHA_AHEBB, variant="wpos").to(DEVICE)
    kw = trainer_kwargs(N, n_epochs=EPOCHS)
    tr = Trainer(model=t, train_loader=get_loaders()[0], val_loader=get_loaders()[1], device=DEVICE, **kw)
    def _log(m):
        if str(DEVICE)=="mps": torch.mps.empty_cache()
        if (m["epoch"]+1)%10==0: print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}", flush=True)
    h = tr.train(n_epochs=EPOCHS, log_fn=_log)
    best = max(x["val_top1"] for x in h)
    print(f"  Teacher done: {best:.4f}")
    TEACHER_PATH.parent.mkdir(exist_ok=True)
    torch.save(t.state_dict(), TEACHER_PATH)
    return t.state_dict()

def make_model(cfg, si=0):
    r = _build(K_ITER, SEED+si)
    if cfg.warm or cfg.proj:
        m = SGNNET_WarmProj(r, ALPHA_AHEBB)
    else:
        m = SGNNET_AntiHebbian(r, alpha_ahebb=ALPHA_AHEBB, variant="wpos")
    return m

def main():
    print(f"\n{'='*70}")
    print(f"Step 170 — warm+W_proj N=2048 D=16 (efficiency track phase-exit test)")
    print(f"N={N}  D={D}  K_hh={K_HH}  K_iter={K_ITER}  Epochs={EPOCHS}  Data=50%")
    print(f"Phase-exit: ≥95% @ FLOPs ~6.2M\n{'='*70}")

    cfg_filter = [k.strip() for k in args.configs.split(",") if k.strip()] if args.configs else []
    active = [(i,c) for i,c in enumerate(CONFIGS) if not cfg_filter or c.key in cfg_filter]

    need_teacher = any(c.warm for _,c in active)
    ts = get_teacher(force=args.force_retrain_teacher) if need_teacher else None
    get_loaders(); results = {}

    for i, cfg in active:
        print(f"\n{'─'*60}\nConfig {cfg.key}: {cfg.label}\n{'─'*60}")
        model = make_model(cfg, si=i).to(DEVICE)
        if cfg.warm and ts: model.warm_load(ts)
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={np_:,}")
        t0 = time.time()
        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=get_loaders()[0],
                          val_loader=get_loaders()[1], device=DEVICE, **kw)
        def _log(m):
            if str(DEVICE)=="mps": torch.mps.empty_cache()
            if (m["epoch"]+1)%10==0:
                flag = " *** PHASE EXIT! ***" if m["val_top1"]>=0.95 else ""
                print(f"  ep{m['epoch']+1:3d}  val={m['val_top1']:.4f}{flag}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time()-t0
        top1h = [round(h.get("val_top1",0.),4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h))+1
        results[cfg.key] = {"N":N,"D":D,"top1_best":best,"top1_last":top1h[-1],
                             "best_epoch":bep,"epochs_run":len(history),
                             "top1_history":top1h,"elapsed_s":round(elapsed,1),
                             "n_params":np_,"label":cfg.label}
        ref = results.get("Ref",{}).get("top1_best",0)
        pe = "✓ PHASE EXIT!" if best>=0.95 else f"({0.95-best:.3f}pp short)"
        print(f"\n  top1={best:.4f}  vs_Ref={best-ref:+.4f}  {pe}  elapsed={elapsed/60:.1f}min")
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nSTEP 170 SUMMARY — warm+W_proj N=2048 D=16\n{'='*70}")
    ref = results.get("Ref",{}).get("top1_best",0)
    for k,r in results.items():
        pe = "✓ PHASE EXIT!" if r["top1_best"]>=0.95 else f"({0.95-r['top1_best']:.3f}pp short)"
        print(f"{k}: {r['top1_best']:.4f}  vs_Ref={r['top1_best']-ref:+.4f}  {pe}")
    print(f"Results → {OUT_PATH}")

if __name__ == "__main__":
    main()
