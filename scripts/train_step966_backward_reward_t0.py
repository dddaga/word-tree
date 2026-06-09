"""Step 966: Backward reward scoring — guide routing toward class-relevant nodes.

HYPOTHESIS
==========
If routing activation follows topological proximity to class-specific readout nodes,
the network develops class-selective pathways ↓ routing signal energy ↓ data movement.

MECHANISM
=========
1. C_ho_mask ∈ {0,1}^{N×N_out}: readout connectivity (sparsity=0.90 → ~205 nodes/class).
2. BFS distance D_dist[c, n] = shortest path on conn_hh from any R_c root to node n.
   R_c = set of nodes connected to class c in C_ho_mask. Unreachable → D_max.
3. Reward for class c: reward[c, n] = decay^D_dist[c, n]. High near roots, decays with hop.
4. Aux loss: L_aux = KL(softmax(Z_norms/tau), L1_norm(reward[y]))
   Z_norms: L2 norm of each node's state (proxy for "how much signal flows here").
   Forces high-norm nodes to be near readout roots of the true class.
5. Total loss: L_ce + λ(t) * L_aux.
   λ(t) = λ0 * (1 - t/T) — annealed to zero so CE dominates late training.

DESIGN CHOICES
==============
- lambda_schedule: 0.01→0 (anneal). Too large → suppresses CE gradients.
- decay: 0.5 (default) — 1/2 per hop. 0.7 = slower decay, less selective.
- tau: 1.0 — softmax temperature for Z_norms distribution.
- D_d07_noAH: alpha_ahebb=0 → isolate from double-sparsity (AH also suppresses nodes).

CONFIGS (20ep, 50% data, N=2048, seed=42)
  Ref         — canonical (no aux loss)
  A_d05       — decay=0.50, λ0=0.01
  B_d07       — decay=0.70, λ0=0.01
  C_d085      — decay=0.85, λ0=0.01  (very gradual decay)
  D_d07_noAH  — decay=0.70, λ0=0.01, alpha_ahebb=0.0 (isolates AH interaction)
  E_d07_lam001 — decay=0.70, λ0=0.001  (lower lambda — weaker reward signal)

ADVANCE CRITERION (dual-axis)
  accuracy ≥ −0.5pp vs Ref  AND  mean separation improvement > +0.05 over Ref
  (step967 provides Ref separation as baseline)

KILL
  accuracy < −1.0pp vs Ref OR mean separation improvement < +0.01

REF_BASELINE = 0.9595  (step605 K=1 KD student, but we test without KD here)
T0_REF       = ~0.92   (typical T0 Imagenette at 20ep/50%)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sgnnet.model_smallworld import SGNNET_SmallWorld
from src.sgnnet.model_resonant   import SGNNET_Resonant
from src.training.dataset        import make_loaders, make_subset_loader

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="auto")
parser.add_argument("--epochs",  type=int, default=20)
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--data",    default="data/store.h5")
parser.add_argument("--configs", default="Ref,A_d05,B_d07,C_d085,D_d07_noAH,E_d07_lam001")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
) if args.device == "auto" else torch.device(args.device)

EPOCHS = args.epochs
BATCH  = 64
SEED   = args.seed
N      = 2048
D      = 16
N_IN   = 25088  # pre-extracted VGG16 pool features
N_OUT  = 10
K_HH   = 2
K_ITER = 5
K_IN   = 25
ALPHA_REFLECT = 0.5
SPARSITY_C_HO = 0.90
AUX_TAU = 1.0
D_MAX   = 999

SLOT     = os.environ.get("SGN_SLOT", "local")
OUT_PATH = ROOT / "results" / f"train_step966_backward_reward_t0_seed{SEED}__{SLOT}.json"

CONFIGS = {
    "Ref":          {"decay": None,  "lam0": 0.0,   "alpha_ahebb": 1.0},
    "A_d05":        {"decay": 0.50,  "lam0": 0.01,  "alpha_ahebb": 1.0},
    "B_d07":        {"decay": 0.70,  "lam0": 0.01,  "alpha_ahebb": 1.0},
    "C_d085":       {"decay": 0.85,  "lam0": 0.01,  "alpha_ahebb": 1.0},
    "D_d07_noAH":   {"decay": 0.70,  "lam0": 0.01,  "alpha_ahebb": 0.0},
    "E_d07_lam001": {"decay": 0.70,  "lam0": 0.001, "alpha_ahebb": 1.0},
}


# ── BFS distance precomputation ───────────────────────────────────────────────
def bfs_distances(root_set: set, conn_hh: torch.Tensor, N: int) -> torch.Tensor:
    """BFS from root_set over conn_hh graph. Returns dist[n] (int), D_MAX if unreachable."""
    dist = torch.full((N,), D_MAX, dtype=torch.float32)
    queue = deque()
    for r in root_set:
        if dist[r] == D_MAX:
            dist[r] = 0.0
            queue.append(r)
    while queue:
        u = queue.popleft()
        for v in conn_hh[u].tolist():
            if dist[v] == D_MAX:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def precompute_reward(conn_hh: torch.Tensor, c_ho_mask: torch.Tensor,
                      decay: float) -> torch.Tensor:
    """
    Returns reward [N_out, N] on CPU.
    reward[c, n] = decay^dist(R_c, n), 0 if unreachable.
    """
    N_nodes = conn_hh.size(0)
    reward  = torch.zeros(N_OUT, N_nodes)
    for c in range(N_OUT):
        roots = c_ho_mask[:, c].nonzero(as_tuple=True)[0].tolist()
        if not roots:
            continue
        dist_c = bfs_distances(set(roots), conn_hh, N_nodes)
        reachable = dist_c < D_MAX
        reward[c, reachable] = decay ** dist_c[reachable]
    # L1-normalize each class row
    row_sum = reward.sum(dim=1, keepdim=True).clamp(min=1e-8)
    reward  = reward / row_sum
    return reward  # [N_OUT, N]


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(alpha_ahebb: float, decay: float | None, reward_matrix: torch.Tensor | None):
    K_r = max(1, K_HH // 4); K_l = K_HH - K_r
    torch.manual_seed(SEED)
    base = SGNNET_SmallWorld(
        N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
        K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
        n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
    )
    resonant = SGNNET_Resonant(
        base, K_phase=8, alpha_reflect=ALPHA_REFLECT, alpha_turing=0.0,
        beam_size=16, geo_gamma=0.5, mode="dynamic_z_geo", resonance_threshold=0.0,
    )

    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = resonant
            if reward_matrix is not None:
                self.register_buffer("reward", reward_matrix.clone())
            else:
                self.reward = None

        def forward(self, x):
            return self.m(x)

        def forward_with_Z(self, x):
            base_m = self.m.base
            Z = base_m._seed(x)
            theta_pos = self.m.theta.abs().unsqueeze(0).unsqueeze(-1)
            conn_hh   = base_m.conn_hh
            W_h = self.m.W_pos[:N]
            dw  = F.normalize(W_h.unsqueeze(1) - W_h[conn_hh], dim=-1).unsqueeze(0)
            Z_ref = torch.zeros_like(Z)
            for _ in range(K_ITER):
                Z_fwd = F.leaky_relu(Z - theta_pos, negative_slope=0.01)
                Z_nb  = Z_fwd[:, conn_hh, :]
                c_ij  = (Z_nb * dw).sum(dim=-1, keepdim=True)
                Z_agg = (Z_nb * c_ij.abs()).sum(dim=2)
                Z_ref = ALPHA_REFLECT * Z_ref + (Z_fwd - Z)
                Z     = F.normalize((Z_agg + Z_ref).clamp(-10, 10), dim=-1)
            return base_m._readout(Z), Z

        def aux_loss(self, Z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """KL(softmax(Z_norms/tau) || reward[y])"""
            if self.reward is None:
                return torch.tensor(0.0, device=Z.device)
            z_norms = Z.norm(dim=-1)                           # [B, N]
            p       = F.softmax(z_norms / AUX_TAU, dim=-1)    # [B, N]
            q       = self.reward[y]                           # [B, N]  (already L1-normed)
            q       = q.clamp(min=1e-8)
            return F.kl_div(p.log(), q, reduction="batchmean")

    return RewardModel()


# ── Data loading (pre-extracted h5 features) ─────────────────────────────────
def load_data():
    h5_path = ROOT / args.data
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found"); sys.exit(1)
    _, va = make_loaders(str(h5_path), batch_size=BATCH, seed=SEED, pin_memory=False)
    tr    = make_subset_loader(str(h5_path), fraction=0.5, batch_size=BATCH, seed=SEED)
    return tr, va


# ── Jaccard helpers (copied from step967 for separation metric) ───────────────
def pairwise_jaccard(masks: list) -> float:
    if len(masks) < 2:
        return 0.0
    ms = torch.stack(masks).float()
    scores = []
    n = len(masks)
    for i in range(min(n, 30)):
        a = ms[i]
        inter = (a.unsqueeze(0) * ms[i+1:]).sum(dim=1)
        union = ((a.unsqueeze(0) + ms[i+1:]) > 0).float().sum(dim=1)
        scores.append((inter / union.clamp(min=1)).mean().item())
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def compute_separation(model, va) -> float:
    model.eval()
    masks_by_class = {c: [] for c in range(N_OUT)}
    for x, _sl, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, Z = model.forward_with_Z(x)
        theta = model.m.theta.abs().mean().item()
        active = (Z.norm(dim=-1) > theta)
        for b in range(y.size(0)):
            masks_by_class[y[b].item()].append(active[b].cpu())
        if sum(len(v) for v in masks_by_class.values()) > 500:
            break  # quick estimate

    intra = {c: pairwise_jaccard(masks_by_class[c]) for c in range(N_OUT)}
    inter_scores = []
    classes = list(range(N_OUT))
    for c1 in classes:
        for c2 in classes:
            if c1 >= c2: continue
            m1, m2 = masks_by_class[c1][:10], masks_by_class[c2][:10]
            if m1 and m2:
                a_stack = torch.stack(m1).float()
                b_stack = torch.stack(m2).float()
                for a in a_stack:
                    inter = (a.unsqueeze(0) * b_stack).sum(dim=1)
                    union = ((a.unsqueeze(0) + b_stack) > 0).float().sum(dim=1)
                    inter_scores.append((inter / union.clamp(min=1)).mean().item())
    mean_inter = float(np.mean(inter_scores)) if inter_scores else 0.0
    return float(np.mean([intra[c] - mean_inter for c in range(N_OUT)]))


@torch.no_grad()
def evaluate(model, va):
    model.eval()
    correct = total = 0
    for x, _sl, y in va:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_model(model, tr, va, lam0: float):
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    hist  = []
    for ep in range(EPOCHS):
        model.train()
        lam = lam0 * (1.0 - ep / EPOCHS)  # anneal to 0
        for x, _sl, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, Z  = model.forward_with_Z(x)
            logits = model.m.base._readout(Z)
            loss_ce  = F.cross_entropy(logits, y)
            loss_aux = model.aux_loss(Z, y) if lam > 0 else torch.tensor(0.0)
            loss = loss_ce + lam * loss_aux
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val = evaluate(model, va)
        hist.append(val)
        print(f"  e{ep+1:3d}  top1={val:.4f}  lam={lam:.4f}  lr={opt.param_groups[0]['lr']:.2e}",
              flush=True)
    return hist


def main():
    tr, va = load_data()

    print(f"\n{'='*70}")
    print(f"step966 — backward reward scoring T0 (20ep, 50% data)")
    print(f"  Aux loss: KL(softmax(Z_norms/τ) || decay^BFS_dist[y])")
    print(f"  Lambda: annealed λ0→0 over {EPOCHS} epochs.")
    print(f"  Dual-axis advance: accuracy ≥−0.5pp AND separation improvement >+0.05")
    print(f"{'='*70}\n")

    keys    = [k.strip() for k in args.configs.split(",") if k.strip() in CONFIGS]
    results = {}
    ref_acc = None
    ref_sep = None

    for key in keys:
        cfg   = CONFIGS[key]
        decay = cfg["decay"]
        lam0  = cfg["lam0"]
        alpha_ahebb = cfg["alpha_ahebb"]

        print(f"{'─'*60}")
        print(f"{key}: decay={decay}  λ0={lam0}  alpha_ahebb={alpha_ahebb}")

        # Precompute BFS reward matrix (only when aux loss active)
        reward_matrix = None
        if decay is not None:
            # Get conn_hh and C_ho_mask from a freshly built base model
            K_r = max(1, K_HH // 4); K_l = K_HH - K_r
            torch.manual_seed(SEED)
            _base = SGNNET_SmallWorld(
                N_hidden=N, N_out=N_OUT, D=D, N_in=N_IN,
                K_in=K_IN, K_iter=K_ITER, K_local=K_l, K_random=K_r,
                n_groups=max(8, N // 8), norm_mode="l2", encoding_mode="fourier",
            )
            conn_hh   = _base.conn_hh.cpu()     # [N, K_hh]
            c_ho_mask = _base.C_ho_mask.cpu() if hasattr(_base, "C_ho_mask") else None

            if c_ho_mask is None:
                # fallback: build binary mask from readout weight threshold
                with torch.no_grad():
                    W_ro = _base._readout.weight.abs().cpu()   # [N_OUT, N*D]
                    # aggregate over D per node
                    W_ro_n = W_ro.view(N_OUT, N, D).norm(dim=-1)  # [N_OUT, N]
                    top_k  = int(N * (1.0 - SPARSITY_C_HO))
                    c_ho_mask = torch.zeros(N, N_OUT, dtype=torch.bool)
                    for c in range(N_OUT):
                        idx = W_ro_n[c].topk(top_k).indices
                        c_ho_mask[idx, c] = True
            else:
                c_ho_mask = c_ho_mask.bool()

            n_roots = c_ho_mask.sum(0).float().mean().item()
            print(f"  BFS precompute: {n_roots:.0f} roots/class × decay={decay}...")
            t_bfs = time.time()
            reward_matrix = precompute_reward(conn_hh, c_ho_mask, decay)
            print(f"  BFS done in {time.time()-t_bfs:.1f}s  "
                  f"reward mean={reward_matrix.max(1).values.mean():.4f}")

        model = build_model(alpha_ahebb, decay, reward_matrix).to(DEVICE)
        n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_p:,}")

        t0   = time.time()
        hist = train_model(model, tr, va, lam0)
        elapsed = time.time() - t0

        best    = max(hist)
        best_ep = int(np.argmax(hist)) + 1

        # Compute separation (quick pass)
        sep = compute_separation(model, va)

        if key == "Ref":
            ref_acc = best
            ref_sep = sep

        _ref_acc = ref_acc if ref_acc is not None else best
        _ref_sep = ref_sep if ref_sep is not None else sep
        delta_acc = best - _ref_acc
        delta_sep = sep  - _ref_sep

        verdict = ("(reference)"     if key == "Ref"
                   else "ADVANCE→T1"  if (delta_acc >= -0.005 and delta_sep >= 0.05)
                   else "INTERESTING" if (delta_acc >= -0.005 and delta_sep >= 0.01)
                   else "SEP_ONLY"    if (delta_acc < -0.005 and delta_sep >= 0.05)
                   else "NEUTRAL"     if (delta_acc >= -0.010 and delta_sep >= 0.0)
                   else "KILL")

        print(f"  -> best={best:.4f} @ep{best_ep}  sep={sep:.4f}  "
              f"Δacc={delta_acc*100:+.2f}pp  Δsep={delta_sep*100:+.2f}pp  {verdict}")

        results[key] = {
            "decay":           decay,
            "lam0":            lam0,
            "alpha_ahebb":     alpha_ahebb,
            "n_params":        n_p,
            "best":            round(best, 4),
            "best_ep":         best_ep,
            "separation":      round(sep, 4),
            "delta_vs_ref_acc": round(delta_acc, 4),
            "delta_vs_ref_sep": round(delta_sep, 4),
            "elapsed_s":       round(elapsed),
            "history_top1":    [round(v, 4) for v in hist],
            "verdict":         verdict,
        }
        OUT_PATH.parent.mkdir(exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}")
    print(f"STEP 966 SUMMARY — backward reward scoring T0")
    print(f"{'='*70}")
    for k, r in results.items():
        print(f"  {k:<16}  best={r['best']:.4f}  sep={r['separation']:.4f}  "
              f"Δacc={r['delta_vs_ref_acc']*100:+.2f}pp  "
              f"Δsep={r['delta_vs_ref_sep']*100:+.2f}pp  {r['verdict']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
