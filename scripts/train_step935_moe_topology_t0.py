"""Step 935: MoE topology experts T0 (20ep, 50% data).

Motivation: SGNNET uses a fixed K_hh=2 topology (K_local=1 + K_random=1).
Gemma 4's MoE insight: multiple experts, top-k active + 1 shared always-active.
Map to SGNNET: multiple topology experts (different neighbor types), learned per-node
gate selects additive mixture → no gate-death risk (input-independent gate on Z).

MECHANISM:
  Three topology experts:
    local  — spatial K_local neighbors from conn_hh (existing)
    random — fresh random K_random neighbors sampled each forward pass
    hub    — fixed pseudo-hub nodes (top-K_hub indices, same seed as graph)

  Per expert e:
    Z_nb_e = gather neighbors for expert e               [B, N, K_e, D]
    dw_e   = ΔW-proj direction for expert e              [1, N, K_e, D]
    c_ij_e = (Z_nb_e · dw_e).sum(-1, keepdim=True)      [B, N, K_e, 1]
    Z_exp_e = (Z_nb_e * |c_ij_e|).sum(dim=2)            [B, N, D]

  Gate: learned per-node bias (NOT a function of Z → no multiplicative gate on Z)
    self.gate = nn.Parameter(torch.zeros(N, n_experts))
    g = F.softmax(self.gate, dim=-1)                     [N, n_experts]
    Z_agg = sum_e g[:, e:e+1] * Z_exp_e

  C_3exp_shared: local expert always contributes (like Gemma shared expert);
    gate only routes over (random, hub); Z_agg = Z_exp_local + softmax_gate * Z_exp_{r,h}

CONFIGS (N=2048, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  Ref:         standard ΔW-proj K_hh=2 — same as step887
  A_2exp:      2 experts (local, random), softmax gate per node  [N,2] params
  B_3exp:      3 experts (local, random, hub), softmax gate      [N,3] params
  C_3exp_shared: 3 experts + local always contributes (shared analogue)

K_local=1, K_random=1 (from K_HH=2 split), K_hub=32.

ADVANCE: ≥+0.5pp vs Ref → T1.
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
parser.add_argument("--configs", default="Ref,A_2exp,B_3exp,C_3exp_shared")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_ITER = 5; K_IN = 25
ALPHA_REFLECT = 0.5
K_HUB = 32  # pseudo-hub nodes (fixed indices 0..31, same seed as graph)

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step935_moe_topology_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


def _dw_proj(W_pos, conn):
    """Compute normalised ΔW direction for a conn tensor [N, K]."""
    W_h = W_pos[:N]
    # conn: [N, K] → W_h[conn]: [N, K, D]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn], dim=-1).unsqueeze(0)  # [1, N, K, D]


def _expert_agg(Z_fwd, conn, dw):
    """Aggregate one expert: ΔW-proj weighted sum over K neighbors.

    Z_fwd: [B, N, D]
    conn:  [N, K]
    dw:    [1, N, K, D]  — precomputed or freshly computed

    Returns Z_exp: [B, N, D]
    """
    B = Z_fwd.shape[0]
    K = conn.shape[1]
    Z_nb = Z_fwd[:, conn.reshape(-1), :].reshape(B, N, K, D)  # [B, N, K, D]
    c_ij = (Z_nb * dw).sum(dim=-1, keepdim=True)               # [B, N, K, 1]
    return (Z_nb * c_ij.abs()).sum(dim=2)                       # [B, N, D]


class SGNNET_Ref(nn.Module):
    """Baseline ΔW-proj K_hh=2 (no MoE). Mirrors step887."""
    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw        = _dw_proj(self.m.W_pos, conn_hh)
        Z_ref     = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_agg = _expert_agg(Z_fwd, conn_hh, dw)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_MoE(nn.Module):
    """MoE topology experts with learned per-node additive gate.

    mode in ("2exp", "3exp", "3exp_shared")
    Experts:
      local  — conn_hh[:, :K_l]   (spatial neighbors, fixed)
      random — sampled each step from Uniform[0,N)
      hub    — hub_idx (fixed buffer, K_hub nodes)

    Gate: nn.Parameter [N, n_experts], softmax → no gate-death on Z.
    C_3exp_shared: local always contributes; gate only over (random, hub).
    """

    def __init__(self, resonant, mode: str):
        super().__init__()
        self.m    = resonant
        self.mode = mode

        K_r = max(1, K_HH // 4)
        K_l = K_HH - K_r
        self.K_l = K_l
        self.K_r = K_r

        if mode == "2exp":
            n_experts = 2   # local, random
        elif mode in ("3exp", "3exp_shared"):
            n_experts = 3   # local, random, hub
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.n_experts = n_experts
        # gate_experts: for shared mode only gates over (random, hub)
        self.n_gate = 2 if mode == "3exp_shared" else n_experts
        self.gate   = nn.Parameter(torch.zeros(N, self.n_gate))  # [N, n_gate]

        # pseudo-hub indices: fixed arange, same conceptual seed as graph
        hub_idx = torch.arange(K_HUB, dtype=torch.long)          # [K_hub]
        self.register_buffer("hub_idx", hub_idx)

        # Precompute dw for local expert (fixed topology)
        # dw_random and dw_hub computed fresh or at forward (hub is fixed so cache it)

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def _hub_conn(self):
        """Expand hub_idx to [N, K_hub] — same hubs for every node."""
        return self.hub_idx.unsqueeze(0).expand(N, -1)  # [N, K_hub]

    def forward(self, x):
        Z         = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh               # [N, K_HH]
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)

        # Split conn_hh into local (first K_l cols) and ignore random slot
        conn_local = conn_hh[:, :self.K_l]            # [N, K_l]

        # Pre-compute fixed ΔW directions
        dw_local = _dw_proj(self.m.W_pos, conn_local)  # [1, N, K_l, D]

        # Hub conn + dw (fixed, can precompute once per forward — hub_idx is fixed)
        conn_hub = self._hub_conn().to(Z.device)       # [N, K_hub]
        dw_hub   = _dw_proj(self.m.W_pos, conn_hub)    # [1, N, K_hub, D]

        # Gate: softmax over n_gate experts; broadcast to [1, N, 1]
        g = F.softmax(self.gate, dim=-1)               # [N, n_gate]

        Z_ref = torch.zeros_like(Z)

        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)

            # --- local expert ---
            Z_exp_local = _expert_agg(Z_fwd, conn_local, dw_local)   # [B, N, D]

            # --- random expert (fresh conn each step) ---
            conn_random = torch.randint(0, N, (N, self.K_r),
                                        device=Z_fwd.device)          # [N, K_r]
            dw_random   = _dw_proj(self.m.W_pos, conn_random)         # [1, N, K_r, D]
            Z_exp_random = _expert_agg(Z_fwd, conn_random, dw_random) # [B, N, D]

            if self.mode == "2exp":
                # gate: [N, 2] → [N, 1] each
                g0 = g[:, 0:1]  # [N, 1]
                g1 = g[:, 1:2]  # [N, 1]
                Z_agg = g0 * Z_exp_local + g1 * Z_exp_random

            elif self.mode == "3exp":
                Z_exp_hub = _expert_agg(Z_fwd, conn_hub, dw_hub)      # [B, N, D]
                g0 = g[:, 0:1]; g1 = g[:, 1:2]; g2 = g[:, 2:3]
                Z_agg = g0 * Z_exp_local + g1 * Z_exp_random + g2 * Z_exp_hub

            else:  # 3exp_shared: local always on, gate over (random, hub)
                Z_exp_hub = _expert_agg(Z_fwd, conn_hub, dw_hub)      # [B, N, D]
                g0 = g[:, 0:1]; g1 = g[:, 1:2]  # gates for (random, hub) only
                Z_agg = Z_exp_local + g0 * Z_exp_random + g1 * Z_exp_hub

            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

        return self.m.base._readout(Z)


# mode string → (class, mode_arg, n_gate_params, description)
CONFIG_SPEC = {
    "Ref":            ("ref",        None,           0,  "baseline ΔW-proj K_hh=2"),
    "A_2exp":         ("moe",        "2exp",         2,  "2 experts: local+random, gate[N,2]"),
    "B_3exp":         ("moe",        "3exp",         3,  "3 experts: local+random+hub, gate[N,3]"),
    "C_3exp_shared":  ("moe",        "3exp_shared",  2,  "3 exp + local always-on, gate[N,2]"),
}


def _param_count(key):
    _, _, n_gate, _ = CONFIG_SPEC[key]
    base = (N + N_OUT) * D + N  # W_pos (N*D) + W_out (N_OUT*D) + theta (N)
    extra = N * n_gate
    return base, extra, base + extra


def make_base():
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


def make_model(key):
    kind, mode_arg, _, _ = CONFIG_SPEC[key]
    r = make_base()
    if kind == "ref":
        return SGNNET_Ref(r)
    return SGNNET_MoE(r, mode=mode_arg)


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
    print(f"step935 — MoE topology experts T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}/K_hub={K_HUB}")
    print(f"  Gate: learned per-node bias [N, n_gate], softmax — NOT a function of Z")
    print(f"  Local: conn_hh[:K_l]  Random: fresh each step  Hub: arange(K_hub)")
    print(f"  Canonical ref (step887): {DW_REF:.4f}")
    print()
    print(f"  {'Config':<16} {'description':<40} {'extra':>8}  {'total':>8}")
    print(f"  {'-'*72}")
    for k in CONFIG_SPEC:
        _, _, n_gate, desc = CONFIG_SPEC[k]
        b, ex, tot = _param_count(k)
        print(f"  {k:<16} {desc:<40} {ex:>8,}  {tot:>8,}")
    print(f"{'='*74}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue

        model = make_model(key).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _, _, _, desc = CONFIG_SPEC[key]
        b, ex, exp_total = _param_count(key)

        print(f"{'─'*60}")
        print(f"{key}: {desc}")
        print(f"  params={n_p:,}  (expected={exp_total:,}  extra={ex:,})")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        def log_fn(m):
            print(f"  e{m['epoch']+1:3d}  top1={m['val_top1']:.4f}  lr={m['lr']:.2e}", flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h    = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta   = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL"   if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {
            "desc": desc, "n_params": n_p, "extra_params": ex,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*74}")
    print(f"STEP 935 SUMMARY — MoE topology experts T0")
    print(f"{'='*74}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<16} params={r['n_params']:>8,}  best={r['best']:.4f}  "
              f"Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
