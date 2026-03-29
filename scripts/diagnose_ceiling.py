"""Ceiling diagnosis experiments: isolate why static accuracy caps at ~11%.

Tests three hypotheses in one script, each as a controlled ablation:

  H1 — C_input structure: random vs block-local vs channel-grouped
  H2 — K_iter depth: 1 vs 3 vs 5 hidden routing iterations
  H3 — Readout capacity: D=4 vs D=8 vs D=16 (more output dimensions)

Each ablation trains N=512 for 60 epochs and reports val_loss + top-1.
Results written to results/diagnosis_ceiling.json.
"""

from __future__ import annotations
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import h5py

from src.sgnnet.norm_masked import masked_normalize
from src.sgnnet.encoding import compute_spatial_encoding
from src.sgnnet.model_wave import _make_binary_c
from src.sgnnet.model_smallworld import _build_fanin_conn
from src.training.trainer import Trainer
from src.training.experiment_config import trainer_kwargs, GA_BEST
from src.utils.metrics import compute_all_metrics

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N, D_DEFAULT, N_IN, N_OUT = 512, 4, 25088, 10
EPOCHS = 60
BATCH = GA_BEST["batch_size"]
CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]
VGG16_FC = 123_642_856


# ── Minimal model that accepts swappable C_input strategy ─────────────────────

class DiagnosticNet(nn.Module):
    """Stripped-down SGNNET for ablation. No phasor — real activations only.

    cinput_mode: 'random'  — random binary mask (Phase 4 default)
                 'block'   — block-local fan-in (SmallWorld default)
                 'channel' — input grouped by VGG channel (new hypothesis)
    """

    def __init__(
        self,
        N_hidden: int = 512,
        D: int = 4,
        N_in: int = 25088,
        N_out: int = 10,
        K_iter: int = 3,
        cinput_mode: str = "block",
        K_in: int = 50,
        sparsity: float = 0.90,
    ):
        super().__init__()
        self.N_hidden = N_hidden
        self.D = D
        self.K_iter = K_iter
        self.W_phase = None  # compatibility shim

        self.W_pos = nn.Parameter(torch.rand(N_hidden + N_out, D))

        self.register_buffer("spatial_coords", compute_spatial_encoding(N_in))

        # C_input wiring
        if cinput_mode == "random":
            # Random binary mask — original Phase 4 approach
            self.register_buffer("C_input_mask",
                                  _make_binary_c(N_in, N_hidden, sparsity))
            self._use_mask = True
        elif cinput_mode == "block":
            # Block-local fan-in index table
            n_groups = max(8, N_hidden // 8)
            conn_in = _build_fanin_conn(N_hidden, N_in, K_in, n_groups)
            self.register_buffer("conn_in", conn_in)
            self._use_mask = False
        elif cinput_mode == "channel":
            # Channel-grouped: each neuron group maps to one VGG channel.
            # VGG pool5 = [512 channels, 49 spatial] = 25088.
            # Each of 512 neuron-groups gets K_in samples from one channel's 49 positions.
            n_ch = 512
            spatial_per_ch = N_in // n_ch   # 49
            group_size = max(1, N_hidden // n_ch)
            conn = []
            rng = np.random.default_rng(0)
            for h in range(N_hidden):
                ch = (h // group_size) % n_ch
                positions = list(range(ch * spatial_per_ch,
                                        (ch + 1) * spatial_per_ch))
                chosen = rng.choice(positions,
                                    size=min(K_in, len(positions)),
                                    replace=len(positions) < K_in)
                conn.append(chosen)
            conn_in = torch.tensor(np.array(conn), dtype=torch.long)
            self.register_buffer("conn_in", conn_in)
            self._use_mask = False

        # Fixed random C_hh (same for all ablations — isolate C_input effect)
        self.register_buffer("C_hh_mask",
                              _make_binary_c(N_hidden, N_hidden, sparsity, zero_diag=True))
        self.register_buffer("C_ho_mask", _make_binary_c(N_hidden, N_out, sparsity))

    def forward(self, x):
        B = x.shape[0]
        spatial = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
        A_input = torch.cat([x.unsqueeze(-1), spatial], dim=-1)   # [B, N_in, D]

        if self._use_mask:
            Z = torch.einsum("bid,ih->bhd", A_input, self.C_input_mask.float())
        else:
            Z = A_input[:, self.conn_in, :].sum(dim=2)

        Z = masked_normalize(Z)

        for _ in range(self.K_iter):
            Z = torch.einsum("bhd,hj->bjd", Z, self.C_hh_mask.float())
            Z = masked_normalize(Z)

        A_out = torch.einsum("bhd,ho->bod", Z, self.C_ho_mask.float())
        W_out = F.normalize(self.W_pos[self.N_hidden:], dim=-1)
        return (A_out * W_out.unsqueeze(0)).sum(dim=-1)


def load_data():
    with h5py.File("data/store.h5", "r") as f:
        return (
            torch.tensor(f["train/features"][:]),
            torch.tensor(f["train/soft_labels"][:]),
            torch.tensor(f["train/labels"][:]),
            torch.tensor(f["val/features"][:]),
            torch.tensor(f["val/soft_labels"][:]),
            torch.tensor(f["val/labels"][:]),
        )


def run_ablation(name, model, data):
    train_f, train_sl, train_l, val_f, val_sl, val_l = data
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_f, train_sl, train_l),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_f, val_sl, val_l),
        batch_size=BATCH, num_workers=0,
    )
    tkwargs = trainer_kwargs(model.N_hidden)
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, **tkwargs)

    t0 = time.perf_counter()
    def log(m):
        if m["epoch"] in (0, 1, 2, 10, 30, 59) or m.get("stopped_early"):
            print(f"  [{name}] ep{m['epoch']:3d}  "
                  f"train={m['train_loss']:.4f}  val={m['val_loss']:.4f}")

    history = trainer.train(EPOCHS, log_fn=log)
    result = trainer.evaluate()
    metrics = compute_all_metrics(
        result["scores"].numpy(), result["labels"].numpy(), CLASS_NAMES
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE  top1={metrics['top1_accuracy']:.4f}  "
          f"mAP={metrics['mAP']:.4f}  t={elapsed:.0f}s\n")
    return {
        "name": name,
        "top1": metrics["top1_accuracy"],
        "mAP": metrics["mAP"],
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 1),
        "final_val_loss": history[-1]["val_loss"],
    }


def main():
    print(f"Device: {DEVICE}  N={N}  epochs={EPOCHS}\n")
    data = load_data()
    print(f"Data loaded: train={data[0].shape[0]} val={data[3].shape[0]}\n")

    ablations = []

    # ── H1: C_input structure ─────────────────────────────────────────────
    print("=== H1: C_input structure (K_iter=3 fixed) ===")
    for mode in ["random", "block", "channel"]:
        print(f"\n-- cinput_mode={mode} --")
        m = DiagnosticNet(N_hidden=N, D=D_DEFAULT, K_iter=3, cinput_mode=mode)
        ablations.append(run_ablation(f"H1_cinput_{mode}", m, data))

    # ── H2: Routing depth ────────────────────────────────────────────────
    print("=== H2: K_iter depth (block C_input fixed) ===")
    for k in [1, 3, 5]:
        print(f"\n-- K_iter={k} --")
        m = DiagnosticNet(N_hidden=N, D=D_DEFAULT, K_iter=k, cinput_mode="block")
        ablations.append(run_ablation(f"H2_kiter_{k}", m, data))

    # ── H3: C_input fan-in size ───────────────────────────────────────────
    # D=4 is structurally fixed (1 feature value + 3 VGG spatial coordinates).
    # Increasing D would require redesigning the spatial encoding — the extra
    # dimensions would be zeros at seeding and add no information.
    # Instead test whether K_in (how many inputs each hidden neuron sees) matters.
    # More connections = richer aggregation at the cost of more computation.
    print("=== H3: K_in fan-in size (block C_input, K_iter=3 fixed) ===")
    for k_in in [10, 50, 200]:
        print(f"\n-- K_in={k_in} --")
        m = DiagnosticNet(N_hidden=N, D=D_DEFAULT, K_iter=3,
                          cinput_mode="block", K_in=k_in)
        ablations.append(run_ablation(f"H3_kin_{k_in}", m, data))

    print("\n=== Summary ===")
    print(f"{'Name':<30}  {'top-1':>6}  {'mAP':>6}  {'params':>8}  {'val_loss':>9}")
    for r in ablations:
        print(f"{r['name']:<30}  {r['top1']:>6.4f}  {r['mAP']:>6.4f}  "
              f"{r['params']:>8}  {r['final_val_loss']:>9.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/diagnosis_ceiling.json", "w") as f:
        json.dump(ablations, f, indent=2)
    print("\nSaved: results/diagnosis_ceiling.json")


if __name__ == "__main__":
    main()
