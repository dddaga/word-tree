"""Step 952: Grouped Input Projection T0 + Observational Haki (20ep, 50% data).

CURRENT SEED STEP (no learned params)
======================================
  Z[b, i, 0]   = sum(x[b, conn_in[i, :]])    ← uniform sum, no selectivity
  Z[b, i, 1:D] = Fourier(spatial_sum[i])     ← fixed position encoding

The sum treats all K_in input features equally. The model cannot learn
"input feature j matters more than k for node i." This is the selectivity
bottleneck.

PROPOSED: LEARNED INPUT PROJECTION
====================================
Replace the scalar sum with a learned weighted combination:

  Z[b, i, 0]  = W_proj[group(i)] · x[b, conn_in[i, :]]  (or per-node W)

where W_proj is a learned weight vector of size [K_in] that selects which
of the K_in gathered features contribute most. This is NOT fully connected
(avoids N_in × N × D cost) but provides learned selectivity at reasonable
parameter cost.

VARIANTS
========
The key question: how much grouping is optimal?

  Ref:            sum(x[conn_in])  — no params, uniform weights
  A_shared:       one W∈[K_in] shared ALL nodes — global feature ordering
  B_grouped256:   G=256 groups, each W∈[K_in] — region-specific (256 groups)
  C_grouped64:    G=64  groups, each W∈[K_in] — more grouping, fewer params
  C2_grouped16:   G=16  groups, each W∈[K_in] — coarser regions
  D_node:         per-node W∈[K_in] — full selectivity, no grouping
  E_multiout:     per-node W∈[K_in, D//2] — projection to 8 dim subspace

PARAMETER COUNTS (K_in=25, D=16, N=2048)
  Ref:          34,976  (W_pos + theta only)
  A_shared:     35,001  (+25)
  B_grouped256: 41,376  (+6,400)
  C_grouped64:  36,576  (+1,600)
  C2_grouped16: 35,376  (+400)
  D_node:       86,176  (+51,200)
  E_multiout:   854,176 (+819,200) — large, but K_in=25, K_hh=2, no N² term

SPATIAL GROUP ASSIGNMENT
========================
Uses the same n_groups=max(8, N//8)=256 as the topology. Nodes within the
same spatial cluster share W_proj. This means A_shared/B_grouped/D_node
correspond to different granularities of the same spatial partition.

OBSERVATIONAL HAKI
==================
Same metrics as step951: seed_fisher, final_fisher, routing_gain, pr, dead_frac.
Key question: does learned projection raise seed_fisher? If yes → selection helps.
If routing_gain stays flat → routing still does the heavy lifting.

ADVANCE: ≥+0.5pp vs Ref at T0 → T1.
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
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_shared,C2_grouped16,C_grouped64,B_grouped256,D_node,E_multiout")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS     = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
ALPHA_REFLECT = 0.5
N_GROUPS = max(8, N // 8)     # 256 — same as topology

HAKI_EPOCHS = {1, 5, 10, 20}

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step952_grouped_input_proj_t0_seed{SEED}__{SLOT}.json"

DW_PROJ_REF = 0.9638   # step887 canonical

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_SPEC = {
    # (proj_mode, n_proj_groups_or_None)
    # proj_mode: "sum" | "shared" | "grouped" | "node" | "multiout"
    "Ref":           ("sum",     None),
    "A_shared":      ("shared",  None),   # G=1
    "C2_grouped16":  ("grouped", 16),
    "C_grouped64":   ("grouped", 64),
    "B_grouped256":  ("grouped", 256),
    "D_node":        ("node",    None),   # G=N
    "E_multiout":    ("multiout", None),  # G=N, D_out=D//2
}

MULTIOUT_D = D // 2   # 8: projection output dims for E_multiout


def proj_param_count(mode, n_groups):
    if mode == "sum":     return 0
    if mode == "shared":  return K_IN
    if mode == "grouped": return n_groups * K_IN
    if mode == "node":    return N * K_IN
    if mode == "multiout": return N * K_IN * MULTIOUT_D
    raise ValueError(mode)


def base_param_count():
    return (N + N_OUT) * D + N    # W_pos + theta = 34,976


# ─────────────────────────────────────────────────────────────────────────────
def _dw_proj(W_pos, conn_hh):
    Wh = W_pos[:N]
    return F.normalize(Wh.unsqueeze(1) - Wh[conn_hh], dim=-1).unsqueeze(0)


class SGNNET_InputProj(nn.Module):
    """ΔW-proj model with learned input projection variants.

    The projection replaces Z[b,i,0]=sum(x[conn_in[i]]) with a learned
    weighted sum (or small matrix). The Fourier position dims Z[b,i,1:D]
    are untouched.

    proj_mode options:
      'sum'     — no params, sum (Ref)
      'shared'  — W[K_in] shared across all nodes
      'grouped' — W[G, K_in], one weight per spatial group
      'node'    — W[N, K_in], per-node selectivity
      'multiout'— W[N, K_in, D//2], full projection to 8-dim subspace
    """

    def __init__(self, resonant, proj_mode: str, n_proj_groups: int | None = None):
        super().__init__()
        self.m          = resonant
        self.proj_mode  = proj_mode
        self.n_proj_groups = n_proj_groups

        if proj_mode == "shared":
            # Init = 1.0 so W_proj · x = sum(x) at epoch 0 (same as Ref)
            self.W_proj = nn.Parameter(torch.ones(K_IN))

        elif proj_mode == "grouped":
            G = n_proj_groups
            self.W_proj = nn.Parameter(torch.ones(G, K_IN))
            # Map each node → group index (same partition as topology)
            nodes_per_group = N // G
            group_ids = torch.arange(N) // nodes_per_group
            group_ids = group_ids.clamp(max=G - 1)
            self.register_buffer("group_ids", group_ids)   # [N]

        elif proj_mode == "node":
            # [N, K_in] init=1: each node starts with uniform sum
            self.W_proj = nn.Parameter(torch.ones(N, K_IN))

        elif proj_mode == "multiout":
            # W[N, K_in, D_out]: each node projects K_in → D_out
            # Init: first column = 1.0 (sum to dim 0), rest = 0 (other dims silent at start)
            W = torch.zeros(N, K_IN, MULTIOUT_D)
            W[:, :, 0] = 1.0   # warm-start: dim 0 = sum(x), dims 1..D_out-1 = 0
            self.W_proj = nn.Parameter(W)

        # 'sum': no parameters

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self):   return self.m.W_pos

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned projection to gathered inputs.

        x: [B, N_in]
        Returns: scalar signal [B, N, 1]  (or [B, N, D_out] for multiout)
        """
        gathered = x[:, self.m.base.conn_in]   # [B, N, K_in]

        if self.proj_mode == "sum":
            return gathered.sum(dim=2, keepdim=True)            # [B, N, 1]

        elif self.proj_mode == "shared":
            # W_proj: [K_in], raw linear weights, no softmax
            # At init W=ones → output = sum(x), same as Ref
            return (gathered * self.W_proj.unsqueeze(0).unsqueeze(0)).sum(dim=2, keepdim=True)

        elif self.proj_mode == "grouped":
            # W_proj: [G, K_in], separate weight vector per group — no softmax
            # group_ids: [N] → W_per_node: [N, K_in]
            w_per_node = self.W_proj[self.group_ids]           # [N, K_in]
            return (gathered * w_per_node.unsqueeze(0)).sum(dim=2, keepdim=True)

        elif self.proj_mode == "node":
            # W_proj: [N, K_in], raw linear, separate weight per node
            return (gathered * self.W_proj.unsqueeze(0)).sum(dim=2, keepdim=True)

        elif self.proj_mode == "multiout":
            # W_proj: [N, K_in, D_out]
            # gathered: [B, N, K_in] → einsum → [B, N, D_out]
            return torch.einsum('bnk,nkd->bnd', gathered, self.W_proj)

    def _seed_with_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Build Z_seed using learned projection for signal dim(s)."""
        B = x.shape[0]
        signal = self._project_input(x)             # [B, N, 1] or [B, N, D_out]

        if self.proj_mode == "multiout":
            # signal is [B, N, D_out=8]; pad with Fourier dims
            sp = self.m.base.spatial_sum.unsqueeze(0).expand(B, -1, -1)  # [B, N, D-1]
            # Replace first D_out dims with learned projection
            # Keep remaining D-1-D_out Fourier dims for position
            Z_remain = sp[:, :, MULTIOUT_D - 1:]                          # [B, N, D-1-D_out+1]
            Z = torch.cat([signal, Z_remain], dim=-1)                     # [B, N, D]
        else:
            # signal is [B, N, 1]; concat with Fourier [B, N, D-1]
            sp = self.m.base.spatial_sum.unsqueeze(0).expand(B, -1, -1)  # [B, N, D-1]
            Z = torch.cat([signal, sp], dim=-1)                           # [B, N, D]

        return F.normalize(Z, dim=-1)

    def _routing_loop(self, Z: torch.Tensor) -> torch.Tensor:
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
        Z       = self._seed_with_proj(x)
        Z_final = self._routing_loop(Z)
        return self.m.base._readout(Z_final)

    def _forward_full(self, x):
        Z_seed  = self._seed_with_proj(x)
        Z_final = self._routing_loop(Z_seed.clone())
        logits  = self.m.base._readout(Z_final)
        return logits, Z_seed.detach(), Z_final.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Observational haki (same as step951)

def _fisher_ratio(Z, labels, n_classes):
    grand = Z.mean(0)
    sb = sum(
        ((Z[labels==c].mean(0) - grand).pow(2).mean()
         * (labels==c).float().mean()).item()
        for c in range(n_classes) if (labels==c).any()
    )
    sw = sum(
        (Z[labels==c] - Z[labels==c].mean(0)).pow(2).mean().item()
        for c in range(n_classes) if (labels==c).any()
    ) / n_classes
    return float(sb / (sw + 1e-8))


def _participation_ratio(Z):
    Zc = Z - Z.mean(0)
    cov = (Zc.T @ Zc) / max(len(Zc) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).abs()
    return float(ev.sum().pow(2) / (ev.pow(2).sum() + 1e-8))


@torch.no_grad()
def compute_haki(model, val_loader, device, n_classes=10, max_batches=12):
    model.eval()
    Z_seeds, Z_finals, labels_all = [], [], []
    dead_count = torch.zeros(N, device="cpu")
    total = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x, _, y = batch
        x = x.to(device)
        _, Z_seed, Z_final = model._forward_full(x)
        Z_seeds.append(Z_seed.mean(dim=1).cpu())
        Z_finals.append(Z_final.mean(dim=1).cpu())
        labels_all.append(y)
        norms = Z_final.norm(dim=-1)
        dead_count += (norms < 0.05).float().sum(dim=0).cpu()
        total += x.shape[0]

    Zs   = torch.cat(Z_seeds, 0)
    Zf   = torch.cat(Z_finals, 0)
    labs = torch.cat(labels_all, 0)

    seed_f  = _fisher_ratio(Zs, labs, n_classes)
    final_f = _fisher_ratio(Zf, labs, n_classes)
    pr      = _participation_ratio(Zf)
    dead    = float((dead_count / total).mean())
    return {
        "seed_fisher":  round(seed_f, 4),
        "final_fisher": round(final_f, 4),
        "routing_gain": round(final_f - seed_f, 4),
        "pr":           round(pr, 3),
        "dead_frac":    round(dead, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
def make_model(mode, n_proj_groups):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )
    return SGNNET_InputProj(resonant, proj_mode=mode, n_proj_groups=n_proj_groups)


# ─────────────────────────────────────────────────────────────────────────────
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

    print(f"\n{'='*74}")
    print(f"step952 — Grouped Input Projection T0 + Haki (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_in={K_IN}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Seed: Z[i,0]=W_proj·x[conn_in[i]]  Z[i,1:D]=Fourier(pos)")
    print(f"  Softmax(W_proj) preserves scale. Init: uniform (= sum / K_in).")
    print(f"  Context: step887 K_in=25 canonical = {DW_PROJ_REF:.4f}")
    print()
    print(f"  {'Config':<16} {'mode':<10} {'G':>5}  {'extra_p':>8}  {'total_p':>8}")
    base_p = base_param_count()
    for k, (mode, ng) in CONFIG_SPEC.items():
        ep = proj_param_count(mode, ng)
        print(f"  {k:<16} {mode:<10} {str(ng):>5}  {ep:>8,}  {base_p+ep:>8,}")
    print(f"{'='*74}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}
    ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        mode, ng = CONFIG_SPEC[key]
        model = make_model(mode, ng).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ep  = proj_param_count(mode, ng)

        print(f"{'─'*74}")
        print(f"{key}: mode={mode}  G={ng}  extra_params={ep:,}  total={n_p:,}")

        kw   = trainer_kwargs(N, n_epochs=EPOCHS)
        t0   = time.time()
        haki_log = {}

        def log_fn(m, _key=key):
            ep_n = m["epoch"] + 1
            print(f"  e{ep_n:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", end="", flush=True)
            if ep_n in HAKI_EPOCHS:
                haki = compute_haki(model, va, DEVICE)
                haki_log[ep_n] = haki
                print(f"  | seed_F={haki['seed_fisher']:.3f}  final_F={haki['final_fisher']:.3f}"
                      f"  gain={haki['routing_gain']:+.3f}  PR={haki['pr']:.2f}"
                      f"  dead={haki['dead_frac']:.3f}", end="")
                model.train()
            print(flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h  = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best   = max(top1h)
        best_ep = int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta  = best - (ref_acc if ref_acc is not None else DW_PROJ_REF)

        last_key = max(haki_log.keys()) if haki_log else None
        if last_key:
            print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp")
            print(f"     Final haki: {haki_log[last_key]}")
        else:
            print(f"  → best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp")

        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))

        results[key] = {
            "mode": mode, "n_proj_groups": ng, "extra_proj_params": ep,
            "n_params": n_p, "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
            "verdict": verdict, "haki": haki_log,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"STEP 952 SUMMARY — Grouped Input Projection")
    print(f"  (Context: step887 K_in=25 T2 = {DW_PROJ_REF:.4f})")
    print(f"{'='*74}")
    print(f"  {'Config':<16} {'mode':<10}  {'params':>8}  {'best':>7}  {'Δ':>8}  "
          f"{'seed_F':>7}  {'finalF':>7}  {'gain':>7}  {'PR':>5}  verdict")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = r.get("verdict", "?")
        lh = (r["haki"][max(r["haki"].keys())] if r["haki"] else {})
        sf = lh.get("seed_fisher", 0)
        ff = lh.get("final_fisher", 0)
        gn = lh.get("routing_gain", 0)
        pr = lh.get("pr", 0)
        print(f"  {k:<16} {r['mode']:<10}  {r['n_params']:>8,}  "
              f"{r['best']:>7.4f}  {d*100:>+7.2f}pp  "
              f"{sf:>7.3f}  {ff:>7.3f}  {gn:>+7.3f}  {pr:>5.2f}  {v}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
