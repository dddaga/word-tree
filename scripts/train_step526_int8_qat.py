"""Step 526: INT8 quantization — inference impact + QAT training with grad-accum sweep.

MOTIVATION
==========
Two questions the paper needs to answer about int8 quantization of SGNNET:

PART A (inference impact, fast):
  Train fp32, then quantize W_pos + activations for evaluation. Measure accuracy
  drop under 3 overflow modes:
    saturate : standard (clamp to [-128, 127])
    modular  : torch.remainder-style wrap (user's idea 2026-04-15)
    crt      : CRT dual-residue (stub: falls back to saturate for this prototype)
  User's insight (validated 2026-04-15): for L2-normalized W_pos at D=16,
  components are bounded to ±1/√16 ≈ ±0.25, scale×100 ⇒ int8 range ±25, so wrap
  NEVER fires. Modular should match saturate. Test this empirically.

PART B (QAT training, gradient accumulation sweep):
  User hypothesis: int8 bins are 0.01 wide at scale=100. Typical gradient updates
  are ~1e-3, so a single-batch update does NOT cross a bin boundary. Accumulating
  gradients over N batches lets updates compound until they flip a bin — this
  may be the key to recovering fp32-level accuracy under int8 training.

  Test: QAT (fake-quant forward + STE backward) with grad_accum ∈ {1, 4, 16, 64}.
  Metric: final val top-1 + epochs-to-90%. Expected: accum=1 under-trains; larger
  accum recovers fp32 performance. If even accum=64 can't close the gap, int8
  training for this architecture needs a different fix (e.g. learned scales).

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, 20ep scout on 50% data)

  Part A (shared fp32 base model trained once, then eval per quant mode):
    A_fp32    : reference eval, no quant
    A_w_sat   : W_pos saturating-quant @ scale=100
    A_w_mod   : W_pos modular-wrap-quant @ scale=100
    A_w_crt   : W_pos CRT-stub quant  @ scale=100
    A_wz_sat  : W_pos + Z both saturating
    A_wz_mod  : W_pos + Z both modular

  Part B (separate training runs, QAT from scratch):
    B_qat_a1   : grad_accum=1  (standard QAT)
    B_qat_a4   : grad_accum=4
    B_qat_a16  : grad_accum=16
    B_qat_a64  : grad_accum=64

To run:
    python -u scripts/train_step526_int8_qat.py --device mps
    python -u scripts/train_step526_int8_qat.py --device cuda --parts A
    python -u scripts/train_step526_int8_qat.py --device cuda --parts B
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.sgnnet.model_smallworld      import SGNNET_SmallWorld
from src.sgnnet.model_resonant        import SGNNET_Resonant
from src.sgnnet.mechanisms_inhibitory import SGNNET_AntiHebbian
from src.training.dataset             import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data", default="data/store.h5")
parser.add_argument("--parts", default="AB", help="Which parts to run: A, B, or AB")
parser.add_argument("--scale_w", type=float, default=100.0)
parser.add_argument("--scale_z", type=float, default=100.0)
parser.add_argument("--k_iter", type=int, default=5,
                    help="Routing iterations. step526=5 (default), step527 uses 4 for INT8×K=4 combo.")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10; D = 16; K_HH = 2; K_IN = 25
K_ITER = args.k_iter

import os as _os
SLOT = _os.environ.get("SGN_SLOT", "local")
_step = "step527" if K_ITER == 4 else "step526"
OUT_PATH = ROOT / "results" / f"train_{_step}_int8_qat_k{K_ITER}_seed{SEED}__{SLOT}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Fake quantization with STE
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantSTE(torch.autograd.Function):
    """Fake int8 quantization with straight-through gradient.
    mode ∈ {"saturate", "modular", "crt", "abs_mod"}
      abs_mod: user proposal 2026-04-15 — modular wrap + abs → values in [0, 128].
        HYPOTHESES (untested, per "every claim needs evidence" policy we measure):
        predicted to destroy sign info and break AH / ΔW rotation. Actual impact
        must be confirmed by this experiment.
    """
    @staticmethod
    def forward(ctx, x, scale, mode):
        xq = (x * scale).round()
        wrap_events = 0
        if mode == "saturate":
            wrap_events = ((xq < -128) | (xq > 127)).sum().item()
            xq = xq.clamp(-128, 127)
        elif mode == "modular":
            wrap_events = ((xq < -128) | (xq > 127)).sum().item()
            xq = ((xq + 128) % 256) - 128
        elif mode == "crt":
            wrap_events = ((xq < -128) | (xq > 127)).sum().item()
            xq = xq.clamp(-128, 127)
        elif mode == "abs_mod":
            wrap_events = ((xq < -128) | (xq > 127)).sum().item()
            xq = ((xq + 128) % 256) - 128
            xq = xq.abs()   # now in [0, 128]
        ctx.mark_non_differentiable()
        FakeQuantSTE.last_wrap_events = wrap_events
        FakeQuantSTE.last_numel = x.numel()
        return xq / scale

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None


def fake_quant(x, scale, mode):
    return FakeQuantSTE.apply(x, scale, mode)


# ─────────────────────────────────────────────────────────────────────────────
# QAT wrapper around SGNNET_AntiHebbian
# ─────────────────────────────────────────────────────────────────────────────

class SGNNET_AH_QAT(nn.Module):
    """Int8-fake-quant wrapper around SGNNET_AntiHebbian.

    quant_w, quant_z: bool flags — enable quantization of W_pos / activations.
    mode: overflow handling. See FakeQuantSTE.
    """
    def __init__(self, ah_model: SGNNET_AntiHebbian,
                 scale_w: float = 100.0, scale_z: float = 100.0,
                 mode: str = "saturate",
                 quant_w: bool = True, quant_z: bool = True):
        super().__init__()
        self.m = ah_model.m
        self.alpha = ah_model.alpha_ahebb
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.mode = mode
        self.quant_w = quant_w
        self.quant_z = quant_z
        self.wrap_w_total = 0
        self.wrap_z_total = 0
        self.fwd_count = 0

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
        if self.quant_w:
            W_h_q = fake_quant(W_h, self.scale_w, self.mode)
            self.wrap_w_total += FakeQuantSTE.last_wrap_events
        else:
            W_h_q = W_h

        W_n = F.normalize(W_h_q, dim=-1)
        pos_sim = (W_n.unsqueeze(1) * W_n[conn_hh]).sum(-1)
        supp_w = (1.0 - self.alpha * pos_sim.clamp(min=0)).unsqueeze(0).unsqueeze(-1)

        Z_reflected = torch.zeros_like(Z)
        for _ in range(self.m.base.K_iter):
            Z_fwd = F.relu(Z - theta_pos)
            if self.quant_z:
                Z_fwd = fake_quant(Z_fwd, self.scale_z, self.mode)
                self.wrap_z_total += FakeQuantSTE.last_wrap_events
            Z_nb = Z_fwd[:, conn_hh, :]
            Z_struct = (Z_nb * supp_w).sum(dim=2)
            Z_remainder = Z_fwd - Z
            Z_reflected = self.m.alpha_reflect * Z_reflected + Z_remainder
            Z = F.normalize((Z_struct + Z_reflected).clamp(-10, 10), dim=-1)

        self.fwd_count += 1
        return self.m.base._readout(Z)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_fp32():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                             K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                             n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier")
    res = SGNNET_Resonant(base, K_phase=8, alpha_reflect=0.5, alpha_turing=0.0,
                          beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo",
                          resonance_threshold=0.0)
    return SGNNET_AntiHebbian(res, alpha_ahebb=1.0, variant="wpos")


# ─────────────────────────────────────────────────────────────────────────────
# Training with gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def train_with_accum(model, tr, va, epochs, accum_steps, device, lr=1e-3):
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    for ep in range(epochs):
        model.train()
        if hasattr(model, "tick_epoch"): model.tick_epoch()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(tr):
            x = batch[0].to(device)
            y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
            loss = F.cross_entropy(model(x), y) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
        # flush remaining grad
        opt.step(); opt.zero_grad(set_to_none=True)
        sched.step()
        # validate
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in va:
                x = batch[0].to(device)
                y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
                correct += (model(x).argmax(-1) == y).sum().item()
                total += y.size(0)
        hist.append(correct / total)
        if (ep + 1) % 5 == 0:
            print(f"    ep{ep+1:3d} val={hist[-1]:.4f}", flush=True)
    return max(hist), hist


def eval_only(model, va, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for batch in va:
            x = batch[0].to(device)
            y = batch[2].to(device) if len(batch) > 2 else batch[1].to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data_path = ROOT / args.data
    print(f"Step 526 — INT8 inference impact + QAT grad-accum sweep")
    print(f"  data={args.data}  seed={SEED}  epochs={EPOCHS}  device={DEVICE}")
    print(f"  scale_w={args.scale_w}  scale_z={args.scale_z}")
    tr, va = make_loaders(data_path, batch_size=BATCH, seed=SEED)
    # 50% subset for scout
    n = len(tr.dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n // 2]
    subset = torch.utils.data.Subset(tr.dataset, idx.tolist())
    tr = torch.utils.data.DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
    print(f"  Train={len(tr.dataset)}  Val={len(va.dataset)}")

    results = {}

    # ─── PART A ───
    if "A" in args.parts.upper():
        print(f"\n{'='*60}\nPART A — fp32 train → quant eval\n{'='*60}")
        base = build_fp32().to(DEVICE)
        print(f"  params={sum(p.numel() for p in base.parameters() if p.requires_grad):,}")
        t0 = time.time()
        best, hist = train_with_accum(base, tr, va, EPOCHS, 1, DEVICE)
        print(f"  fp32 training done: best={best:.4f} elapsed={time.time()-t0:.0f}s")
        results["A_fp32"] = {"top1_best": best, "train_hist": hist, "elapsed_s": time.time() - t0}

        for mode in ("saturate", "modular", "crt", "abs_mod"):
            mkey = "abs" if mode == "abs_mod" else mode[:3]
            for qw, qz, key in [(True, False, f"A_w_{mkey}"),
                                 (True, True,  f"A_wz_{mkey}")]:
                wrapper = SGNNET_AH_QAT(base, scale_w=args.scale_w, scale_z=args.scale_z,
                                        mode=mode, quant_w=qw, quant_z=qz).to(DEVICE)
                acc = eval_only(wrapper, va, DEVICE)
                wrap_rate_w = wrapper.wrap_w_total / max(1, wrapper.fwd_count * N * D)
                wrap_rate_z = wrapper.wrap_z_total / max(1, wrapper.fwd_count * BATCH * N * D * K_ITER)
                delta_pp = (acc - best) * 100
                print(f"  {key:<12} mode={mode:<8} qw={qw} qz={qz}  val={acc:.4f}  Δ={delta_pp:+.2f}pp  "
                      f"wrap_w={wrap_rate_w:.2e}  wrap_z={wrap_rate_z:.2e}")
                results[key] = {"mode": mode, "quant_w": qw, "quant_z": qz,
                                "val_top1": acc, "delta_pp": delta_pp,
                                "wrap_rate_w": wrap_rate_w, "wrap_rate_z": wrap_rate_z}
                OUT_PATH.parent.mkdir(exist_ok=True)
                with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2)

    # ─── PART B ───
    if "B" in args.parts.upper():
        print(f"\n{'='*60}\nPART B — QAT grad-accum sweep\n{'='*60}")
        # Saturate default (original sweep)
        b_configs = [("saturate", a) for a in (1, 4, 16, 64)]
        # abs_mod variants — only at the two larger accums (cheapest test of user's
        # hypothesis that grad-accum can rescue information-lossy quant).
        b_configs += [("abs_mod", 16), ("abs_mod", 64)]
        for mode, accum in b_configs:
            mkey = "abs" if mode == "abs_mod" else ""
            key = f"B_qat_{mkey}_a{accum}" if mkey else f"B_qat_a{accum}"
            print(f"\n── {key}  (mode={mode} accum={accum})")
            base = build_fp32()
            qat = SGNNET_AH_QAT(base, scale_w=args.scale_w, scale_z=args.scale_z,
                                 mode=mode, quant_w=True, quant_z=True).to(DEVICE)
            t0 = time.time()
            best, hist = train_with_accum(qat, tr, va, EPOCHS, accum, DEVICE)
            elapsed = time.time() - t0
            ep_90 = next((i + 1 for i, v in enumerate(hist) if v >= 0.90), None)
            print(f"  best={best:.4f}  ep→90%={ep_90}  elapsed={elapsed:.0f}s  "
                  f"wrap_w_total={qat.wrap_w_total}")
            results[key] = {
                "mode": mode, "accum_steps": accum, "top1_best": best, "train_hist": hist,
                "epochs_to_90": ep_90, "elapsed_s": elapsed,
                "wrap_w_total": qat.wrap_w_total, "wrap_z_total": qat.wrap_z_total,
            }
            OUT_PATH.parent.mkdir(exist_ok=True)
            with open(OUT_PATH, "w") as f: json.dump(results, f, indent=2)

    # ─── Summary ───
    print(f"\n{'='*60}\nSTEP 526 SUMMARY\n{'='*60}")
    if "A" in args.parts.upper() and "A_fp32" in results:
        ref = results["A_fp32"]["top1_best"]
        print(f"\nPart A  fp32 ref={ref:.4f}")
        for k, r in results.items():
            if not k.startswith("A_") or k == "A_fp32": continue
            print(f"  {k:<12} mode={r['mode']:<8} qw={r['quant_w']} qz={r['quant_z']}  "
                  f"Δ={r['delta_pp']:+.2f}pp  wrap_w_rate={r['wrap_rate_w']:.2e}")
    if "B" in args.parts.upper():
        print(f"\nPart B  QAT grad-accum sweep:")
        ref = results.get("A_fp32", {}).get("top1_best", 0.0)
        for k, r in results.items():
            if not k.startswith("B_qat"): continue
            d = (r["top1_best"] - ref) * 100 if ref else 0
            print(f"  {k:<10} accum={r['accum_steps']:>3}  best={r['top1_best']:.4f}  "
                  f"Δ_vs_fp32={d:+.2f}pp  ep→90%={r['epochs_to_90']}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
