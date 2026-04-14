"""Step 301: ΔW projection × AH interaction study at N=2048 D=16.

USER INSIGHT (2026-04-13)
=========================
ΔW proj replaces AH entirely in step234/235 (+1.68pp). But user hypothesis:
  "AH could still serve a ROLE: when two channels are very close
   together (small ΔW), AH gates that edge so the pre-existing signal
   doesn't propagate just because it wasn't polarized."

So ΔW does ROUTING QUALITY (projects onto relational axis), while AH
does EDGE MASKING (suppresses edges where ΔW is degenerate).

This sweep tests whether ΔW proj + WEAK AH outperforms ΔW proj alone.
step234 tested α=1.0 (full AH) and found it hurts ΔW. The optimum may
be α ∈ (0, 1] — weak AH as safety net without overpowering ΔW.

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 50% data, 20ep Tier-0)
  Ref     : Baseline AH α=1.0 (step199)
  A_proj  : ΔW projection, no AH (step234 winner)
  B_a025  : ΔW projection + AH α=0.25
  B_a050  : ΔW projection + AH α=0.50
  B_a075  : ΔW projection + AH α=0.75
  B_a100  : ΔW projection + AH α=1.00 (step234 showed -1.2pp)
  C_rot   : ΔW rotation, no AH (step235 alternative)
  C_rot_a050 : ΔW rotation + AH α=0.50 (parallel test)
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

from src.sgnnet.model_smallworld     import SGNNET_SmallWorld
from src.sgnnet.model_resonant       import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.trainer            import Trainer
from src.training.experiment_config  import trainer_kwargs
from src.training.dataset            import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--configs", default="")
args   = parser.parse_args()
DEVICE = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = 42; DATA = "data/store.h5"
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5; ALPHA_TURING = 0.0

OUT_PATH = ROOT / "results" / "train_step301_delta_ah_interaction.json"


class SGNNET_DeltaAH(nn.Module):
    """ΔW projection or rotation, with optional AH suppression at tunable α.

    mode: 'proj' → project Z_nb onto ΔW direction (absolute value scaling)
          'rot'  → rotate Z_nb in (Z_nb, ΔW) plane by learned angle
    alpha_ahebb: 0.0 disables AH entirely; else AH applied before proj/rot
    """
    def __init__(self, base: SGNNET_Resonant, mode="proj", alpha_ahebb=0.0):
        super().__init__()
        self.m = base
        self.mode = mode
        self.alpha_ahebb = alpha_ahebb
        # Only relevant for rotation
        self.rotation_temp = nn.Parameter(torch.tensor(0.5))

    @property
    def W_pos(self):   return self.m.W_pos
    @property
    def W_phase(self): return self.m.W_phase

    def tick_epoch(self):
        if hasattr(self.m, "tick_epoch"): self.m.tick_epoch()

    def forward(self, x):
        Z = self.m.base._seed(x)
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        conn_hh = self.m.base.conn_hh
        N_h = self.m.base.N_hidden

        W_h = self.m.W_pos[:N_h]
        W_n = F.normalize(W_h, dim=-1)

        # AH suppression (if alpha > 0)
        supp_w = None
        if self.alpha_ahebb > 0:
            pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
            supp_w = (1.0 - self.alpha_ahebb * pos_sim.clamp(min=0)
                      ).unsqueeze(0).unsqueeze(-1)

        # ΔW = W_pos[receiver] - W_pos[sender]  [N, K_hh, D]
        delta_w = W_h.unsqueeze(1) - W_h[conn_hh]
        delta_w_norm = F.normalize(delta_w, dim=-1).unsqueeze(0)  # [1, N, K, D]

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            Z_nb = Z_fwd[:, conn_hh, :]

            # Apply AH first if present
            if supp_w is not None:
                Z_nb = Z_nb * supp_w

            if self.mode == "proj":
                proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
                Z_nb = Z_nb * proj_coeff.abs()
            else:  # rot
                proj_coeff = (Z_nb * delta_w_norm).sum(dim=-1, keepdim=True)
                z_parallel = proj_coeff * delta_w_norm
                z_perp = Z_nb - z_parallel
                z_perp_unit = F.normalize(z_perp, dim=-1)
                theta_rot = self.rotation_temp * proj_coeff
                z_mag = Z_nb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                Z_nb = (torch.cos(theta_rot) * Z_nb +
                        torch.sin(theta_rot) * z_perp_unit * z_mag)

            Z_struct = Z_nb.sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z_new = Z_struct + Z_reflected
            Z = F.normalize(Z_new.clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


def build_ref():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_AntiHebbian(resonant, alpha_ahebb=1.0, variant="wpos")


def build_delta(mode, alpha_ahebb):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r; ng = max(8, N // 8)
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                              K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                              n_groups=ng, norm_mode="l2", encoding_mode="fourier")
    resonant = SGNNET_Resonant(base, K_phase=8, alpha_reflect=ALPHA_REFLECT,
                               alpha_turing=ALPHA_TURING, beam_size=16, geo_gamma=0.5,
                               mode="dynamic_z_geo", resonance_threshold=0.0)
    return SGNNET_DeltaAH(resonant, mode=mode, alpha_ahebb=alpha_ahebb)


def main():
    configs = {
        "Ref":         ("ref", None),
        "A_proj":      ("proj", 0.00),
        "B_a025":      ("proj", 0.25),
        "B_a050":      ("proj", 0.50),
        "B_a075":      ("proj", 0.75),
        "B_a100":      ("proj", 1.00),
        "C_rot":       ("rot",  0.00),
        "C_rot_a050":  ("rot",  0.50),
    }
    run_keys = list(configs.keys())
    if args.configs:
        run_keys = [k.strip() for k in args.configs.split(",")]

    print(f"\n{'='*70}")
    print(f"Step 301 — ΔW × AH interaction at N=2048 D=16 (Tier-0)")
    print(f"Hypothesis: ΔW + weak AH may outperform ΔW alone")
    print(f"Running: {run_keys}")
    print(f"{'='*70}")

    tr_full, va = make_loaders(ROOT / DATA, batch_size=BATCH, seed=SEED)
    n = len(tr_full.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr_full.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)

    results = {}
    for key in run_keys:
        mode, alpha = configs[key]
        print(f"\n{'─'*60}\nConfig {key}: mode={mode}, α_AH={alpha}\n{'─'*60}")
        model = (build_ref() if mode == "ref" else build_delta(mode, alpha)).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        trainer = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw)
        # Inject rotation_temp for rotation configs
        if hasattr(model, 'rotation_temp'):
            trainer.optimizer.add_param_group({
                "params": [model.rotation_temp],
                "lr": kw.get("lr_wpos", 2.36e-3),
                "weight_decay": 0.0,
            })

        t0 = time.time()
        def _log(m):
            ep = m["epoch"] + 1
            if ep % 5 == 0 or ep == 1:
                print(f"  ep{ep:3d}  val={m['val_top1']:.4f}", flush=True)
        history = trainer.train(n_epochs=EPOCHS, log_fn=_log)
        elapsed = time.time() - t0

        top1h = [round(h.get("val_top1", 0.), 4) for h in history]
        best = max(top1h); bep = int(np.argmax(top1h)) + 1
        results[key] = {
            "top1_best": best, "top1_last": top1h[-1], "best_epoch": bep,
            "top1_history": top1h, "elapsed_s": round(elapsed, 1),
            "n_params": n_p, "mode": mode, "alpha_ahebb": alpha,
        }
        print(f"  → best={best:.4f} @ ep{bep}  ({elapsed:.0f}s)")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    ref_best = results.get("Ref", {}).get("top1_best", 0)
    a_proj_best = results.get("A_proj", {}).get("top1_best", 0)
    print(f"\n{'='*70}\nSTEP 301 SUMMARY — ΔW × AH interaction\n{'='*70}")
    for key in run_keys:
        r = results[key]
        delta = f"  Δref={r['top1_best']-ref_best:+.4f}"
        delta_proj = (f"  Δproj={r['top1_best']-a_proj_best:+.4f}"
                      if key not in ("Ref", "A_proj") else "")
        a_str = f"α={r['alpha_ahebb']}" if r['alpha_ahebb'] is not None else ""
        print(f"  {key:12s} ({r['mode']}, {a_str}): {r['top1_best']:.4f}{delta}{delta_proj}")

    print(f"\nInterpretation:")
    print(f"  If best B_a* > A_proj: weak AH IS complementary to ΔW proj")
    print(f"  If A_proj stays best: ΔW proj alone is sufficient (step234/235 confirmed)")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
