"""Step 953: Piecewise N_in→D seed projection T0 (20ep, 50% data).

PROBLEM (confirmed by step951):
  Current seed: Z[i,0] = sum(x[conn_in[i,:]]) — 1 scalar of input signal.
  Z[i,1:D] = Fourier(position) — fixed spatial identity, carries zero input info.
  ENTIRE input signal lives in 1 dimension. D-1 dimensions are static/position-only.
  routing_gain always NEGATIVE → routing cannot amplify the 1D signal.
  The seed quality (seed_Fisher) is the dominant factor for final accuracy.

HYPOTHESIS:
  Expand the seed to use ALL D dimensions for input signal, not just 1.
  Divide K_in connections into D groups of K_in//D each.
  Group d gets features from a contiguous spatial block of conn_in.
  Z[i,d] = learned_w_d * sum(x[conn_in[i, d*g:(d+1)*g]])  where g=K_in//D

  The spatial Fourier encoding is NOT USED — spatial identity is embedded
  implicitly in WHICH features each group sums (K_in connections are spatially
  allocated by SGNNET_SmallWorld's spatial precomputation).

  W_pos still drives ΔW-proj routing geometry (separate from Z).

CONFIGS (N=2048, K_in=25, D=16, K_hh=2, K_iter=5, ΔW-proj, 20ep/50%)
  NOTE: K_in=25, D=16 → 25//16=1 remainder 9. Groups 0..8 get 2 features,
        groups 9..15 get 1 feature. Handled automatically.

  Ref:         standard seed (scatter sum → 1 scalar, Fourier pos)
  A_piecewise: K_in//D features per dim, 1 scalar weight per group (shared across nodes)
               Extra params: D=16 weights → total: 34,992
  B_grouped:   K_in//D features per dim, 1 weight per (group, node_group)
               node_groups = max(8, N//8) = 256
               Extra params: n_groups * D = 256*16 = 4096 → total: 39,072
  C_full:      Full linear map K_in → D per group of nodes
               Each node group gets a D×K_in weight matrix
               Extra params: n_groups * D * K_in = 256*16*25 = 102,400 → total: 137,376
  D_nofourier: Same as A_piecewise but with NO Fourier position —
               test whether positional encoding is even needed when Z carries input signal

  At init: weights=1/g so group sum = sum(x_group) → comparable scale to Ref dim 0.

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
from src.sgnnet.encoding            import compute_fourier_encoding
from src.training.trainer           import Trainer
from src.training.experiment_config import trainer_kwargs
from src.training.dataset           import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_piecewise,B_grouped,C_full,D_nofourier")
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs; BATCH = 128; SEED = args.seed
N = 2048; N_IN = 25088; N_OUT = 10
D = 16; K_HH = 2; K_IN = 25; K_ITER = 5
N_GROUPS = max(8, N // 8)   # 256
ALPHA_REFLECT = 0.5

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step953_piecewise_seed_t0_seed{SEED}__{SLOT}.json"

DW_REF = 0.9638  # step887 canonical


def _dw_proj_vec(W_pos, conn_hh):
    W_h = W_pos[:N]
    return F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)


def _group_sizes(K_in: int, D: int):
    """Return list of group sizes summing to K_in, length D."""
    base = K_in // D
    rem  = K_in % D
    return [base + (1 if i < rem else 0) for i in range(D)]


def _group_offsets(K_in: int, D: int):
    """Return (starts, ends) for each of the D groups."""
    sizes = _group_sizes(K_in, D)
    starts = [sum(sizes[:i]) for i in range(D)]
    ends   = [sum(sizes[:i+1]) for i in range(D)]
    return starts, ends


def make_base_model(D_: int = D, encoding: str = "fourier"):
    torch.manual_seed(SEED)
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D_, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=N_GROUPS, norm_mode="l2", encoding_mode=encoding,
    )
    return SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )


class SGNNET_Ref(nn.Module):
    def __init__(self, resonant):
        super().__init__(); self.m = resonant

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def forward(self, x):
        Z = self.m.base._seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


class SGNNET_PiecewiseSeed(nn.Module):
    """Piecewise N_in → D seed: each Z dimension gets its own group of inputs.

    Replaces the single scatter-sum (→ 1 scalar + Fourier position) with
    D separate group sums, one per output dimension.

    conn_in[i,:] is partitioned into D groups (already spatially sorted by
    SGNNET_SmallWorld's spatial precomputation). Group d → Z[i,d].

    The Fourier position encoding is replaced entirely by the piecewise
    projection (use_fourier=False), or retained as a residual (use_fourier=True).
    """

    def __init__(self, resonant, mode: str, use_fourier: bool = True):
        super().__init__()
        self.m          = resonant
        self.mode       = mode
        self.use_fourier = use_fourier
        self.starts, self.ends = _group_offsets(K_IN, D)

        # Fourier encoding is D-1 dims; pad to D with zero for residual use
        fourier_raw = compute_fourier_encoding(N_IN, D=D)         # [N_IN, D-1]
        fourier_pad = F.pad(fourier_raw[:N], (0, 1))              # [N, D]
        self.register_buffer("fourier_enc", fourier_pad)

        if mode == "shared":
            # 1 weight per group — scale of each group's sum
            self.w_group = nn.Parameter(torch.ones(D) / 1.0)     # [D]

        elif mode == "grouped":
            # 1 weight per (node_group, dim) — node group size = N//N_GROUPS
            # Precompute which node group each node belongs to
            node_gids = torch.arange(N) * N_GROUPS // N           # [N]
            self.register_buffer("node_gids", node_gids)
            self.w_group = nn.Parameter(torch.ones(N_GROUPS, D))  # [n_groups, D]

        elif mode == "full":
            # Full K_in→D linear per node group: W[g, d, kin_indices_for_d]
            # Represented as [n_groups, D, K_in] — each group d uses its slice
            node_gids = torch.arange(N) * N_GROUPS // N
            self.register_buffer("node_gids", node_gids)
            self.w_full = nn.Parameter(torch.zeros(N_GROUPS, D, K_IN))
            # Init: 1/g_size on the features in that group's slice, 0 elsewhere
            with torch.no_grad():
                for d, (s, e) in enumerate(zip(self.starts, self.ends)):
                    g = e - s
                    self.w_full[:, d, s:e] = 1.0 / g

    @property
    def W_phase(self): return getattr(self.m, "W_phase", None)

    @property
    def W_pos(self): return self.m.W_pos

    def _piecewise_seed(self, x):
        """Build Z_seed [B,N,D] from piecewise grouped projection."""
        conn_in = self.m.base.conn_in                              # [N, K_in]
        B = x.shape[0]
        Z = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)

        if self.mode == "shared":
            for d, (s, e) in enumerate(zip(self.starts, self.ends)):
                feat = x[:, conn_in[:, s:e]]                       # [B,N,e-s]
                g_sum = feat.sum(dim=2)                            # [B,N]
                Z[:, :, d] = g_sum * self.w_group[d]

        elif self.mode in ("grouped", "full"):
            node_gids = self.node_gids                             # [N]

            if self.mode == "grouped":
                # w_group [n_groups, D] → per-node weight [N, D]
                w_per_node = self.w_group[node_gids]               # [N, D]
                for d, (s, e) in enumerate(zip(self.starts, self.ends)):
                    feat = x[:, conn_in[:, s:e]]                   # [B,N,g]
                    g_sum = feat.sum(dim=2)                        # [B,N]
                    Z[:, :, d] = g_sum * w_per_node[:, d]

            else:  # full
                # w_full [n_groups, D, K_in]
                w_per_node = self.w_full[node_gids]                # [N, D, K_in]
                feats = x[:, conn_in]                              # [B,N,K_in]
                # Z = sum over K_in with learned weights
                Z = (feats.unsqueeze(2) * w_per_node.unsqueeze(0)).sum(dim=3)  # [B,N,D]

        # Optionally add normalised Fourier position as residual
        if self.use_fourier:
            Z = Z + self.fourier_enc.unsqueeze(0)                  # broadcast [1,N,D]

        return F.normalize(Z, dim=-1)

    def forward(self, x):
        Z = self._piecewise_seed(x)
        conn_hh   = self.m.base.conn_hh
        theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
        dw = _dw_proj_vec(self.m.W_pos, conn_hh)
        Z_ref = torch.zeros_like(Z)
        for _ in range(K_ITER):
            Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
            Z_nb  = Z_fwd[:, conn_hh, :]
            c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
            Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
            Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
            Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        return self.m.base._readout(Z)


CONFIG_SPEC = {
    # (mode or "ref", use_fourier)
    "Ref":         ("ref",      True),
    "A_piecewise": ("shared",   True),
    "B_grouped":   ("grouped",  True),
    "C_full":      ("full",     True),
    "D_nofourier": ("shared",   False),
}


def _extra_params(key):
    mode, _ = CONFIG_SPEC[key]
    if mode == "ref":    return 0
    if mode == "shared": return D
    if mode == "grouped": return N_GROUPS * D
    if mode == "full":   return N_GROUPS * D * K_IN
    return 0


def make_model(key):
    mode, use_fourier = CONFIG_SPEC[key]
    r = make_base_model()
    if mode == "ref":
        return SGNNET_Ref(r)
    return SGNNET_PiecewiseSeed(r, mode=mode, use_fourier=use_fourier)


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
    print(f"step953 — Piecewise N_in→D seed projection T0 (20ep, 50% data)")
    print(f"  device={DEVICE}  epochs={EPOCHS}  seed={SEED}")
    print(f"  N={N}/D={D}/K_in={K_IN}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  Current seed: 1 scalar (dim 0) + Fourier pos (dims 1:{D})")
    print(f"  Proposed: D groups of K_in/D inputs → Z carries input signal in all D dims")
    print(f"  step951 confirmed: seed_Fisher is the dominant factor (routing smooths)")
    print(f"  Context: step887 canonical = {DW_REF:.4f}")
    print()
    print(f"  Group sizes: {_group_sizes(K_IN, D)} (K_in={K_IN} / D={D})")
    print()
    print(f"  {'Config':<14} {'mode':<10} {'fourier':>8}  {'extra':>8}  {'total':>8}")
    base_p = (N + N_OUT) * D + N
    for k in CONFIG_SPEC:
        m, uf = CONFIG_SPEC[k]
        ex = _extra_params(k)
        print(f"  {k:<14} {m:<10} {'yes' if uf else 'no':>8}  {ex:>8,}  {base_p+ex:>8,}")
    print(f"{'='*74}\n")

    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results = {}; ref_acc = None

    for key in keys:
        if key not in CONFIG_SPEC:
            print(f"  skip unknown: {key}"); continue
        model = make_model(key).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode, use_f = CONFIG_SPEC[key]
        print(f"{'─'*60}")
        print(f"{key}: mode={mode}  fourier={use_f}  params={n_p:,}")

        kw = trainer_kwargs(N, n_epochs=EPOCHS)
        t0 = time.time()

        def log_fn(m_):
            print(f"  e{m_['epoch']+1:3d}  top1={m_['val_top1']:.4f}  lr={m_['lr']:.2e}", flush=True)

        history = Trainer(model=model, train_loader=tr, val_loader=va,
                          device=DEVICE, **kw).train(n_epochs=EPOCHS, log_fn=log_fn)
        elapsed = time.time() - t0

        top1h = [h.get("val_top1", 0.0) if isinstance(h, dict) else float(h) for h in history]
        best, best_ep = max(top1h), int(np.argmax(top1h)) + 1
        if key == "Ref": ref_acc = best
        delta = best - (ref_acc if ref_acc is not None else DW_REF)
        verdict = ("(baseline)" if key == "Ref"
                   else "ADVANCE→T1" if delta >= 0.005
                   else ("NEUTRAL" if delta >= -0.005 else "KILL"))
        print(f"  -> best={best:.4f} @ep{best_ep}  Δ={delta*100:+.2f}pp  {verdict}")

        results[key] = {
            "mode": mode, "use_fourier": use_f, "n_params": n_p,
            "best": round(best, 4), "best_ep": best_ep,
            "delta_vs_ref": round(delta, 4), "elapsed_s": round(elapsed),
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*74}")
    print(f"STEP 953 SUMMARY — Piecewise N_in→D seed T0")
    print(f"{'='*74}")
    for k, r in results.items():
        d = r["delta_vs_ref"]
        v = ("(ref)" if k == "Ref"
             else ("ADV→T1" if d >= 0.005 else ("NEU" if d >= -0.005 else "KILL")))
        print(f"  {k:<14} mode={r['mode']:<10} params={r['n_params']:>8,}  "
              f"best={r['best']:.4f}  Δ={d*100:+.2f}pp  {v}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
