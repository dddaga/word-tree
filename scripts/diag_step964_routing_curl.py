"""Step 964: Irrotationality diagnostic — measure discrete curl of SGNNET routing field.

QUESTION: Is the ΔW-proj routing field F(i,j) = normalize(W_pos[i] - W_pos[j])
approximately irrotational (curl ≈ 0)?

If YES: routing derives from a scalar potential phi(W_pos).
  K_iter routing = Euler steps toward min(phi) — has an analytic fixed point.
  Implication: K_iter=0 routing (direct phi evaluation) may match K_iter=5.

If NO: routing has genuine circulation. Iteration is necessary.
  The curl magnitude tells us how much each routing step buys.

DISCRETE CURL ON A GRAPH:
For each directed triangle (i → j → k → i) in the routing graph:
  circulation = F(i,j) · t_ij + F(j,k) · t_jk + F(k,i) · t_ki
  where t_ij = unit tangent from i to j = normalize(W_pos[j] - W_pos[i])
  F(i,j) = ΔW-proj direction = normalize(W_pos[i] - W_pos[j])
  Note: F(i,j) = -t_ij by definition → F · t = -1 ALWAYS for direct neighbors.

Wait — this shows the routing field on direct edges is trivially -1 (anti-parallel to edge).
The interesting curl is on LONGER paths: i → j → k where k is not directly connected to i.
We measure: does the routing "remember" where it came from, or is it path-independent?

CORRECTED FORMULATION:
The routing field is not F(i,j) = direction from i to j.
It is: for each node i, the AGGREGATED signal direction after one routing step.
Signal at i after step: Z_agg[i] = Σ_j∈neighbors(i) Z[j] * |Z[j] · dw[i,j]|
The "routing velocity field" is: v[i] = Z_agg[i] - Z[i] (change in activation direction)

Discrete curl of v: for triangle (i,j,k):
  circulation = (v[i] · t_ij) + (v[j] · t_jk) + (v[k] · t_ki)
  where t_ij = normalize(W_pos[j] - W_pos[i])

If |circulation| ≈ 0 for all triangles → v is conservative → has potential phi.

METRICS:
1. mean_curl: mean |circulation| over all triangles at init and after training
2. curl_ratio: mean_curl / mean(|v|) — normalized curl magnitude
3. curl_evolution: how curl changes across K_iter steps (does routing become more irrotational?)
4. potential_r2: if curl≈0, fit phi via least squares (phi[i+1] - phi[i] = v · t_ij). R² measures quality.
5. fixed_point_gap: |Z_K_iter - Z_1| / |Z_1 - Z_0| — does routing converge? Rate?

COMPARISON:
- Random W_pos (at init): expect high curl (random field)
- Trained W_pos: expect lower curl if routing has learned a potential structure
- After K_iter=1 vs K_iter=5: does additional iteration reduce effective curl?

This is a DIAGNOSTIC ONLY — no training, just analysis of trained checkpoints.
Run on step887 canonical checkpoint (seed 42, 96.38%).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.dataset        import make_loaders

parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="auto")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--data",       default="data/store.h5")
parser.add_argument("--checkpoint", default="")
parser.add_argument("--n_triangles", type=int, default=10000)
args = parser.parse_args()

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu")) if args.device == "auto" else torch.device(args.device)

N = 2048; D = 16; K_HH = 2; K_ITER = 5; K_IN = 25; N_IN = 25088; N_OUT = 10
ALPHA_REFLECT = 0.5
SLOT = "local"
OUT_PATH = ROOT / "results" / f"diag_step964_routing_curl_seed{args.seed}__{SLOT}.json"


def make_model():
    torch.manual_seed(args.seed)
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


@torch.no_grad()
def compute_routing_velocity(Z, W_pos, conn_hh):
    """One ΔW-proj routing step → returns velocity field v[i] = Z_agg[i] - Z[i]."""
    W_h  = W_pos[:N]
    dw   = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)  # [1,N,K,D]
    theta_pos = torch.zeros(1, N, 1, device=Z.device)  # no threshold for diagnostic
    Z_fwd = F.leaky_relu(Z, negative_slope=0.01)        # [B,N,D]
    Z_nb  = Z_fwd[:, conn_hh, :]                        # [B,N,K,D]
    c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
    Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)              # [B,N,D]
    return (Z_agg - Z_fwd).mean(0)                      # [N,D] mean over batch


@torch.no_grad()
def discrete_curl(v, W_pos, conn_hh, n_triangles=10000, rng=None):
    """Sample random triangles and measure circulation of velocity field v.

    Triangle (i, j, k): must have i→j and j→k in conn_hh.
    Circulation = v[i]·t_ij + v[j]·t_jk + v[k]·t_ki
    where t_ij = normalize(W_pos[j] - W_pos[i]).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    W_h = W_pos[:N]                                     # [N,D]
    conn = conn_hh.cpu().numpy()                        # [N,K]

    circulations = []
    checked = 0
    attempts = 0
    max_attempts = n_triangles * 20

    while checked < n_triangles and attempts < max_attempts:
        attempts += 1
        i  = int(rng.integers(0, N))
        j  = int(conn[i, rng.integers(0, K_HH)])       # j ∈ neighbors(i)
        k  = int(conn[j, rng.integers(0, K_HH)])       # k ∈ neighbors(j)
        if k == i or k == j:
            continue

        # Edge tangents
        t_ij = F.normalize((W_h[j] - W_h[i]).unsqueeze(0), dim=-1).squeeze(0)
        t_jk = F.normalize((W_h[k] - W_h[j]).unsqueeze(0), dim=-1).squeeze(0)
        t_ki = F.normalize((W_h[i] - W_h[k]).unsqueeze(0), dim=-1).squeeze(0)

        # Circulation = line integral of v around triangle
        circ = (v[i] * t_ij).sum() + (v[j] * t_jk).sum() + (v[k] * t_ki).sum()
        circulations.append(abs(float(circ)))
        checked += 1

    if not circulations:
        return {"mean_curl": float("nan"), "median_curl": float("nan"),
                "max_curl": float("nan"), "n_triangles": 0}

    c = np.array(circulations)
    return {
        "mean_curl":   float(np.mean(c)),
        "median_curl": float(np.median(c)),
        "max_curl":    float(np.max(c)),
        "p95_curl":    float(np.percentile(c, 95)),
        "n_triangles": len(c),
    }


@torch.no_grad()
def fixed_point_convergence(model, Z_init):
    """Track convergence rate of routing: how quickly does Z stabilize?"""
    m = model.base
    conn_hh   = m.conn_hh
    W_pos     = model.W_pos
    W_h       = W_pos[:N]
    dw        = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
    theta_pos = model.theta.abs().unsqueeze(0).unsqueeze(-1)

    Z = Z_init.clone()
    Z_ref = torch.zeros_like(Z)
    deltas = []

    for step in range(K_ITER):
        Z_prev = Z.clone()
        Z_fwd  = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
        Z_nb   = Z_fwd[:, conn_hh, :]
        c_ij   = (Z_nb * dw).sum(dim=-1, keepdim=True)
        Z_agg  = (Z_nb * c_ij.abs()).sum(dim=2)
        Z_ref  = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
        Z      = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
        delta  = float((Z - Z_prev).norm(dim=-1).mean())
        deltas.append(round(delta, 5))

    return deltas


def main():
    print(f"\n{'='*68}")
    print(f"step964 — Routing field irrotationality diagnostic")
    print(f"  device={DEVICE}  seed={args.seed}")
    print(f"  N={N}/D={D}/K_hh={K_HH}/K_iter={K_ITER}")
    print(f"  n_triangles={args.n_triangles}")
    print(f"{'='*68}\n")

    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: {data_path} not found"); sys.exit(1)

    _, va = make_loaders(str(data_path), batch_size=256, seed=args.seed,
                         pin_memory=(DEVICE.type == "cuda"))

    model = make_model().to(DEVICE)

    # Load checkpoint if provided, else use random init
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  No checkpoint — using random init (diagnostic of untrained model)")

    model.eval()
    conn_hh = model.base.conn_hh
    W_pos   = model.W_pos.detach()

    # Get a batch of activations for velocity field
    batch_x, _, _ = next(iter(va))
    batch_x = batch_x.to(DEVICE)[:32]                  # 32 samples sufficient
    Z_seed  = model.base._seed(batch_x).detach()

    results = {}

    # 1. Velocity field and curl at each routing step
    print("Computing routing velocity field and discrete curl per step...")
    Z = Z_seed.clone()
    Z_ref = torch.zeros_like(Z)
    theta_pos = model.theta.abs().unsqueeze(0).unsqueeze(-1)

    for step in range(K_ITER):
        Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
        v     = compute_routing_velocity(Z_fwd, W_pos, conn_hh)
        v_norm = float(v.norm(dim=-1).mean())

        curl_stats = discrete_curl(v.cpu(), W_pos.cpu(), conn_hh.cpu(),
                                   n_triangles=args.n_triangles)
        curl_ratio = (curl_stats["mean_curl"] / (v_norm + 1e-8))

        print(f"  step {step+1}: v_norm={v_norm:.4f}  "
              f"mean_curl={curl_stats['mean_curl']:.4f}  "
              f"curl_ratio={curl_ratio:.4f}  "
              f"p95={curl_stats['p95_curl']:.4f}")

        results[f"step{step+1}"] = {
            "v_norm": round(v_norm, 5),
            "curl_ratio": round(curl_ratio, 5),
            **{k: round(v2, 5) for k, v2 in curl_stats.items()
               if isinstance(v2, float)},
        }

        # Advance Z
        Z_nb  = Z_fwd[:, conn_hh, :]
        W_h   = W_pos[:N]
        dw    = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
        c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
        Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
        Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
        Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)

    # 2. Fixed-point convergence rate
    print("\nFixed-point convergence (delta per routing step):")
    deltas = fixed_point_convergence(model, Z_seed.clone())
    for i, d in enumerate(deltas):
        print(f"  step {i+1}: Δ={d:.5f}")
    results["convergence_deltas"] = deltas
    results["convergence_ratio"] = round(deltas[-1] / (deltas[0] + 1e-8), 4)

    # 3. W_pos diversity (random vs trained)
    W_h = W_pos[:N].cpu()
    pairwise = torch.cdist(W_h, W_h)
    results["wpos_mean_dist"] = round(float(pairwise.mean()), 4)
    results["wpos_min_dist"]  = round(float(pairwise[pairwise > 0].min()), 4)
    print(f"\nW_pos pairwise distance: mean={results['wpos_mean_dist']:.4f}  "
          f"min={results['wpos_min_dist']:.4f}")

    print(f"\n{'='*68}")
    print(f"INTERPRETATION")
    print(f"{'='*68}")

    cr = [results[f"step{k}"]["curl_ratio"] for k in range(1, K_ITER+1)]
    if max(cr) < 0.05:
        verdict = "IRROTATIONAL — routing field is conservative. Potential phi exists."
        verdict2 = "K_iter loop is solving a fixed-point problem that may have analytic solution."
    elif max(cr) < 0.20:
        verdict = "WEAKLY ROTATIONAL — some circulation. Mostly conservative."
        verdict2 = "Potential approximation may work; check step965."
    else:
        verdict = "ROTATIONAL — significant circulation. Iteration is necessary."
        verdict2 = "Routing has genuine cycles; K_iter cannot be shortcut."

    print(f"  curl_ratios per step: {[round(c,4) for c in cr]}")
    print(f"  convergence_deltas:   {deltas}")
    print(f"  verdict: {verdict}")
    print(f"           {verdict2}")

    convergence_speed = "FAST" if deltas[-1] / (deltas[0] + 1e-8) < 0.1 else "SLOW"
    print(f"  convergence: {convergence_speed} (ratio={results['convergence_ratio']:.3f})")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
