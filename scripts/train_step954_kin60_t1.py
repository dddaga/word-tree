"""Step 954: K_in=60 T1 Calibration (75ep, 50% data).
# CUDA-5060ti-validated — training delegates to Trainer helper (handles pin_memory, non_blocking).

T0 result (step951):
  E_kin60: K_in=60, coverage=489.8%, best=95.01%, Δ=+1.15pp vs Ref → ADVANCE
  D_kin40: K_in=40, coverage=326.5%, best=94.42%, Δ=+0.56pp → also tested here

Insight from step951: coverage dominates. routing_gain is always negative
(routing = spatial smoothing), but seed_Fisher grows with K_in. More inputs
→ better class discrimination at seed → stronger readout. Signal purity
hypothesis disproved; coverage hypothesis confirmed.

Routing is a smoothing operation, not an amplifier. The seed signal quality
(set by K_in) is the dominant factor.

CONFIGS (N=2048, D=16, ΔW-proj, 75ep/50%)
  Ref:    K_in=25  coverage=204%   ← current default
  D_kin40: K_in=40  coverage=327%  ← +0.56pp T0
  E_kin60: K_in=60  coverage=490%  ← +1.15pp T0 → primary candidate

ADVANCE rule: ≥+0.5pp vs THIS run's Ref → defaults; < −0.5pp → KILL.
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

from src.sgnnet.model_smallworld    import SGNNET_SmallWorld
from src.sgnnet.model_resonant      import SGNNET_Resonant
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=75)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,D_kin40,E_kin60")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5
ALPHA_REFLECT = 0.5

HAKI_EPOCHS = {15, 30, 50, 75}

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step954_kin60_t1_seed{SEED}__{SLOT}.json"

# step887 canonical ΔW-proj K_in=25 N=2048 = 96.38%
DW_PROJ_REF = 0.9638

CONFIG_SPEC = {
    "Ref":     25,
    "D_kin40": 40,
    "E_kin60": 60,
}


def coverage_pct(K_in: int) -> float:
    return N * K_in / N_IN * 100.0


def _dw_proj(W_pos, conn_hh):
    Wh = W_pos[:N]
    return F.normalize(Wh.unsqueeze(1) - Wh[conn_hh], dim=-1).unsqueeze(0)


class SGNNET_KinVar(nn.Module):
    def __init__(self, resonant):
        super().__init__()
        self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def _routing_loop(self, Z):
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj(self.m.W_pos[:N], conn_hh)
        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return Z

    def forward(self, x):
        Z = self.m.base._seed(x)
        return self.m.base._readout(self._routing_loop(Z))

    def _forward_full(self, x):
        Z_seed  = self.m.base._seed(x)
        Z_final = self._routing_loop(Z_seed.clone())
        return self.m.base._readout(Z_final), Z_seed.detach(), Z_final.detach()


def _fisher_ratio(Z: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    grand = Z.mean(0)
    sb = sum(
        ((Z[labels == c].mean(0) - grand).pow(2).mean()
         * (labels == c).float().mean()).item()
        for c in range(n_classes)
        if (labels == c).any()
    )
    sw = sum(
        (Z[labels == c] - Z[labels == c].mean(0)).pow(2).mean().item()
        for c in range(n_classes)
        if (labels == c).any()
    ) / n_classes
    return float(sb / (sw + 1e-8))


def _participation_ratio(Z: torch.Tensor) -> float:
    Zc = Z - Z.mean(0)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).abs()
    return float(ev.sum().pow(2) / (ev.pow(2).sum() + 1e-8))


@torch.no_grad()
def compute_haki(model: SGNNET_KinVar, val_loader, device, n_classes=10, max_batches=12):
    model.eval()
    Z_seeds, Z_finals, labels_all = [], [], []
    dead_count = torch.zeros(N, device="cpu")
    total = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        x, _, y = batch
        x = x.to(device)
        _, Z_seed, Z_final = model._forward_full(x)
        Z_seeds.append(Z_seed.mean(dim=1).cpu())
        Z_finals.append(Z_final.mean(dim=1).cpu())
        labels_all.append(y)
        norms = Z_final.norm(dim=-1)
        dead_count += (norms < 0.05).float().sum(dim=0).cpu()
        total += x.shape[0]

    Zs   = torch.cat(Z_seeds,  0)
    Zf   = torch.cat(Z_finals, 0)
    labs = torch.cat(labels_all, 0)
    seed_f  = _fisher_ratio(Zs, labs, n_classes)
    final_f = _fisher_ratio(Zf, labs, n_classes)
    pr      = _participation_ratio(Zf)
    dead    = float((dead_count / total).mean())

    return {
        "seed_fisher":  round(seed_f,  4),
        "final_fisher": round(final_f, 4),
        "routing_gain": round(final_f - seed_f, 4),
        "pr":           round(pr, 3),
        "dead_frac":    round(dead, 4),
    }


def make_model(K_in: int) -> SGNNET_KinVar:
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_in, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_KinVar(resonant)


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
        batch_size=BATCH, shuffle=True, num_workers=10,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"\n{'='*72}")
    print(f"step954 — K_in=60 T1 Calibration (75ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}  params=34,976")
    print(f"  T0 context: E_kin60=+1.15pp, D_kin40=+0.56pp vs K_in=25 Ref")
    print(f"  Advance rule: ≥+0.5pp → update defaults to K_in=60")
    print()
    print(f"  {'Config':<10} {'K_in':>5}  {'Coverage':>9}")
    for k, K_in in CONFIG_SPEC.items():
        print(f"  {k:<10} {K_in:>5}  {coverage_pct(K_in):>8.1f}%")
    print(f"{'='*72}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        K_in = CONFIG_SPEC[key]
        model = make_model(K_in).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{'─'*72}")
        print(f"{key}: K_in={K_in}  coverage={coverage_pct(K_in):.1f}%  params={n_p:,}")

        kw       = trainer_kwargs(N, n_epochs=EPOCHS)
        t0       = time.time()
        haki_log = {}

        def log_fn(m):
            ep = m["epoch"] + 1
            print(f"  e{ep:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True, end="")
            if ep in HAKI_EPOCHS:
                haki = compute_haki(model, va, DEVICE)
                haki_log[ep] = haki
                print(f"  | seed_F={haki['seed_fisher']:.3f}  final_F={haki['final_fisher']:.3f}"
                      f"  gain={haki['routing_gain']:+.3f}  PR={haki['pr']:.2f}", end="")
                model.train()
            print(flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h   = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best    = max(top1h)
        best_ep = int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else DW_PROJ_REF)

        if EPOCHS not in haki_log:
            haki_log[EPOCHS] = compute_haki(model, va, DEVICE)

        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→defaults" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {
            "K_in": K_in, "coverage_pct": round(coverage_pct(K_in), 1),
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
            "haki": haki_log,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*72}")
    print(f"STEP 954 SUMMARY — K_in=60 T1")
    print(f"  Context: step887 canonical K_in=25 = {DW_PROJ_REF:.4f}")
    print(f"{'='*72}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→defaults" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        last_haki = r["haki"][max(r["haki"].keys())]
        print(f"  {k:<10} K_in={r['K_in']:>3}  cov={r['coverage_pct']:>5.1f}%  "
              f"best={r['best']:.4f}  Δ={d*100:>+6.2f}pp  "
              f"seed_F={last_haki['seed_fisher']:.3f}  final_F={last_haki['final_fisher']:.3f}  "
              f"PR={last_haki['pr']:.2f}  {v}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
