"""SGNNET_Resonant: combined model from Phase 5 validation ladder.

Stacks all confirmed improvements over the SmallWorld baseline:
  - l2 norm         (Step 1: +13% over masked)
  - Learnable θ     (Step 3: per-neuron threshold, slight edge vs fixed)
  - Reflection      (Step 3: below-threshold remainder self-inhibits source)
  - Turing two-scale(Step 3: local excite via conn_hh + long-range inhibit via phase beam)
  - tick_epoch      (rebuilds phase graph from current W_phase each epoch)

Connectivity modes:
  'resonant'      — static phase graph (rebuilt per-epoch by tick_epoch)
  'dynamic_gate'  — fixed graph topology, edge weights are f(Z) per input (GAT-style)
  'dynamic_z'     — topology rebuilt from Z similarity each forward pass
  'dynamic_z_geo' — dynamic_z + geometric position bias: score penalised by W_pos
                    distance so nearby neurons resonate more easily, and W_pos enters
                    the routing gradient (not just the readout gradient).

dynamic_z pseudo-connection logic (iteration 2):
  1. Beam:  top-M neurons by activation magnitude broadcast (M = beam_size, hard cap)
  2. Score: sim[b,m,n] = dot(Z_beam[b,m], Z[b,n])
            geo_penalty = gamma * ||W_pos[beam_idx[m]] - W_pos[n]||²   (dynamic_z_geo only)
            score = sim - geo_penalty
  3. Threshold: pseudo-connection only forms if score > resonance_threshold
            (clamp scores below threshold to zero — weak resonance produces no connection)
  4. Gate:  weighted average of inhibitory signal by score strength

Computational cost:
  resonant        : O(N·K_phase·B·D) per routing iter
  dynamic_gate    : O(N·K_phase·B·D)
  dynamic_z       : O(M·N·B·D) — beam bounds the cost
  dynamic_z_geo   : O(M·N·B·D) + O(M·N·D) for position distances — negligible extra
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_smallworld import SGNNET_SmallWorld


class SGNNET_Resonant(nn.Module):
    """Combined model: threshold + reflection + Turing + dynamic connectivity.

    Parameters
    ----------
    base          : SGNNET_SmallWorld backbone (provides _seed, _readout, conn_hh)
    K_phase       : size of phase neighbourhood graph
    beam_size     : top-M active neurons for long-range broadcast (fixed, no annealing)
    theta_init    : initial value for per-neuron learnable threshold
    alpha_reflect : leaky factor for reflected self-inhibition [0, 1)
    alpha_turing  : weight for long-range inhibitory signal
    mode              : 'resonant' | 'dynamic_gate' | 'dynamic_z' | 'dynamic_z_geo'
    routing_dropout_p : probability of zeroing an entire Z-vector per neuron per routing
                        step (train only). Drop whole vectors — not dims — so l2-norm
                        direction is preserved for surviving neurons. l2-normalise at
                        step end handles rescaling; no inverted scaling needed.
    """

    def __init__(
        self,
        base: SGNNET_SmallWorld,
        K_phase: int = 8,
        beam_size: int = 32,
        theta_init: float = 0.1,
        alpha_reflect: float = 0.3,
        alpha_turing: float = 0.3,
        mode: str = "resonant",
        resonance_threshold: float = 0.0,
        geo_gamma: float = 1.0,
        routing_dropout_p: float = 0.0,
        rebuild_interval: int = 0,
    ):
        super().__init__()
        self.base                = base
        self.K_phase             = K_phase
        self.beam_size           = beam_size
        self.alpha_reflect       = alpha_reflect
        self.alpha_turing        = alpha_turing
        self.mode                = mode
        self.resonance_threshold = resonance_threshold  # min score to form pseudo-connection
        self.geo_gamma           = geo_gamma            # position penalty weight (dynamic_z_geo)
        self.routing_dropout_p   = routing_dropout_p    # neuron-vector dropout during routing
        self._rebuild_interval   = rebuild_interval      # 0 = epoch-only; N = every N steps
        self._rebuild_step       = 0                     # internal step counter for tick_step

        N = base.N_hidden
        D = base.W_pos.shape[1]

        # Learnable per-neuron threshold (always; even in dynamic_z it gates propagation)
        self.theta = nn.Parameter(torch.full((N,), theta_init))

        # W_phase: only needed when alpha_turing != 0 (resonant/dynamic_gate modes)
        # When alpha_turing=0, phase inhibition is skipped entirely — W_phase is dead weight.
        if alpha_turing != 0.0:
            self.W_phase = nn.Parameter(torch.rand(N, D))
            self._build_phase_graph()
        else:
            self.W_phase = None

        # One-shot CUDA perf warning flag. Fires from forward() the first time it
        # sees a CUDA tensor so CPU/MPS runs stay silent.
        self._cuda_warning_fired = False

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def _build_phase_graph(self):
        """Build K-NN phase graph from W_phase. Uses FAISS Flat when available (25× faster
        than numpy brute-force at N=512, exact recall). Falls back to torch @ for small N
        or when faiss-cpu is not installed.
        """
        import numpy as np
        with torch.no_grad():
            Wp     = F.normalize(self.W_phase.detach(), dim=-1)
            device = Wp.device
            try:
                import faiss
                Wp_np = Wp.cpu().float().numpy()
                N, D  = Wp_np.shape
                fi    = faiss.IndexFlatIP(D)
                fi.add(Wp_np)
                _, I  = fi.search(Wp_np, self.K_phase + 1)   # +1 to exclude self
                conn  = np.array(
                    [[j for j in row if j != i][:self.K_phase] for i, row in enumerate(I)],
                    dtype=np.int64,
                )
                idx = torch.tensor(conn, dtype=torch.long, device=device)
            except ImportError:
                # Torch fallback: O(N²) brute-force — cast to float32 to avoid
                # float16 overflow on MPS (-1e9 > float16 max ~65504)
                sim = (Wp @ Wp.T).float()
                sim.fill_diagonal_(-1e9)
                _, idx = sim.topk(self.K_phase, dim=-1)
        self.register_buffer("conn_phase", idx)

    def tick_epoch(self):
        """Called by Trainer between epochs: rebuild phase graph + delegate to base."""
        if self.W_phase is not None:
            self._build_phase_graph()
        if hasattr(self.base, "tick_epoch"):
            self.base.tick_epoch()

    def tick_step(self):
        """Called each training step by Trainer when rebuild_interval > 0.

        Rebuilds conn_phase every rebuild_interval optimizer steps so W_phase K-NN
        stays fresh mid-epoch. At N=512 D=16 FAISS rebuild costs ~0.4ms — negligible
        even at rebuild_interval=1 (55 rebuilds/epoch ≈ 22ms vs ~seconds/epoch).
        """
        if self._rebuild_interval <= 0:
            return
        self._rebuild_step += 1
        if self._rebuild_step % self._rebuild_interval == 0:
            self._build_phase_graph()

    # ------------------------------------------------------------------
    # Compatibility shim so Trainer can access W_pos
    # ------------------------------------------------------------------

    @property
    def W_pos(self):
        return self.base.W_pos

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._cuda_warning_fired and x.device.type == "cuda":
            self._cuda_warning_fired = True
            warnings.warn(
                "SGNNET_Resonant on CUDA uses eager PyTorch — use SGNNET_Resonant_CUDA "
                "(torch.compile) for ~4x speedup (step500 evidence). See "
                ".claude/skills/sgnnet-research/CUDA_CHECKLIST.md.",
                UserWarning, stacklevel=2,
            )
        Z = self.base._seed(x)   # [B, N, D]

        theta_pos = self.theta.abs().unsqueeze(0).unsqueeze(-1)   # [1, N, 1]
        W_ph_norm = F.normalize(self.W_phase, dim=-1)             # [N, D]

        Z_reflected = torch.zeros_like(Z)   # leaky self-inhibition accumulator

        for _ in range(self.base.K_iter):

            # ── 0. Routing dropout (train only) ────────────────────
            # Zero entire Z-vectors (not individual dims) so surviving neurons
            # keep their normalised directions. l2-norm at step end rescales.
            if self.routing_dropout_p > 0.0 and self.training:
                mask = (torch.rand(Z.shape[0], Z.shape[1], 1, device=Z.device)
                        > self.routing_dropout_p).float()
                Z = Z * mask

            # ── 1. Excitatory gate ──────────────────────────────────
            Z_fwd = F.relu(Z - theta_pos)               # [B, N, D]

            # ── 2. Local structural excitation ─────────────────────
            Z_struct = Z_fwd[:, self.base.conn_hh, :].sum(dim=2)   # [B, N, D]

            # ── 3. Self-inhibition reflection ──────────────────────
            # What relu discarded: relu(Z-θ) - Z ≤ 0 where Z < θ
            Z_remainder  = Z_fwd - Z
            Z_reflected  = self.alpha_reflect * Z_reflected + Z_remainder

            # ── 4. Long-range phase inhibition ─────────────────────
            Z_inhibitory = self._phase_inhibit(Z, W_ph_norm, theta_pos)

            # ── 5. Combine & normalise ─────────────────────────────
            Z_new = Z_struct + Z_reflected + self.alpha_turing * Z_inhibitory
            Z = F.normalize(Z_new, dim=-1)

        return self.base._readout(Z)

    # ------------------------------------------------------------------
    # Phase inhibition — three modes
    # ------------------------------------------------------------------

    def _phase_inhibit(
        self,
        Z: torch.Tensor,           # [B, N, D]
        W_ph_norm: torch.Tensor,   # [N, D]
        theta_pos: torch.Tensor,   # [1, N, 1]
    ) -> torch.Tensor:             # [B, N, D], ≤ 0
        """Compute long-range inhibitory signal according to self.mode."""

        B, N, D = Z.shape
        M = min(self.beam_size, N)

        if self.mode == "resonant":
            return self._inhibit_static(Z, W_ph_norm, theta_pos, M)

        elif self.mode == "dynamic_gate":
            return self._inhibit_dynamic_gate(Z, W_ph_norm, theta_pos, M)

        elif self.mode == "dynamic_z":
            return self._inhibit_dynamic_z(Z, theta_pos, M, use_geo=False)

        elif self.mode == "dynamic_z_geo":
            return self._inhibit_dynamic_z(Z, theta_pos, M, use_geo=True)

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def _inhibit_static(self, Z, W_ph_norm, theta_pos, M):
        """Static phase graph (conn_phase from W_phase k-NN), rebuilt per epoch.

        Inhibitory signal: strongly negative activations from top-M active neurons
        are broadcast to their phase neighbours (where they resonate).
        """
        # Source: strongly negative activations ≤ 0
        Z_ref = -F.relu(-(Z + theta_pos))                           # [B, N, D]

        # Beam: top-M by activation magnitude
        activity  = Z.norm(dim=-1)                                  # [B, N]
        top_idx   = activity.topk(M, dim=-1).indices                # [B, M]
        Z_ref_beam = torch.gather(
            Z_ref, 1,
            top_idx.unsqueeze(-1).expand(-1, -1, Z.shape[-1])
        )                                                            # [B, M, D]

        # Project received signal onto W_phase direction (bandpass filter)
        score       = torch.einsum("bmd,nd->bmn", Z_ref_beam, W_ph_norm)  # [B, M, N]
        gate        = score.clamp(min=0)
        Z_inhib     = torch.einsum("bmn,bmd->bnd", gate, Z_ref_beam)      # [B, N, D]
        gate_sum    = gate.sum(dim=1).unsqueeze(-1).clamp(min=1.0)        # [B, N, 1]
        return Z_inhib / gate_sum

    def _inhibit_dynamic_gate(self, Z, W_ph_norm, theta_pos, M):
        """Fixed graph topology (conn_phase), but edge weights are f(Z) per input.

        GAT-style: for each edge (h, k) in conn_phase, the gate is:
            gate[b,h,k] = relu(dot(Z[b,k], W_ph_norm[h]))
        This makes the effective connectivity input-dependent without rebuilding the graph.
        The gate value — how much Z[b,k] resonates with h's phase direction — is zero
        if the incoming activation is orthogonal or opposite, non-zero otherwise.
        """
        Z_ref = -F.relu(-(Z + theta_pos))                           # [B, N, D], ≤ 0

        # Gather phase neighbours' inhibitory signal
        Z_ref_neighbors = Z_ref[:, self.conn_phase, :]              # [B, N, K_phase, D]

        # Activation-conditioned gate: how much does each neighbor's activation
        # resonate with this neuron's phase direction?
        # dot(Z_ref[b,k], W_ph_norm[h]) per neighbor k of h
        score = (Z_ref_neighbors * W_ph_norm.unsqueeze(0).unsqueeze(2)).sum(-1)
        # [B, N, K_phase] — negative where Z_ref ⊙ W_ph < 0

        gate = score.clamp(min=0)                                    # only receive resonant
        Z_inhib  = (gate.unsqueeze(-1) * Z_ref_neighbors).sum(dim=2)  # [B, N, D]
        gate_sum = gate.sum(dim=-1, keepdim=True).clamp(min=1.0)    # [B, N, 1]
        return Z_inhib / gate_sum

    def _inhibit_dynamic_z(self, Z, theta_pos, M, use_geo: bool = False):
        """Input-dependent pseudo-connections: graph rebuilt from current Z each forward.

        The topology is a function of the current activation state — neurons that
        currently represent similar things form transient inhibitory connections
        (winner-take-all pressure to produce non-overlapping feature detectors).

        Two variants:
          use_geo=False : pure feature similarity  (mode='dynamic_z')
          use_geo=True  : feature similarity + geometric position bias (mode='dynamic_z_geo')
                          score = dot(Z_beam, Z) - gamma * ||W_pos_beam - W_pos||²
                          Puts W_pos into the routing gradient for the first time.

        Three gating stages:
          1. Beam: only top-M active neurons broadcast (hard cap, input-dependent selection)
          2. Resonance threshold: pseudo-connection only forms if score > resonance_threshold
             (weak/distant resonance produces no connection — zero contribution, not small)
          3. Weighted aggregation by score strength above threshold

        Cost: O(B·M·N·D) + O(M·N) for geo distances — lightweight even on CPU.
        """
        Z_ref = -F.relu(-(Z + theta_pos))                           # [B, N, D], ≤ 0

        # ── 1. Beam: top-M by activation magnitude ──────────────────────────
        activity   = Z.norm(dim=-1)                                  # [B, N]
        top_idx    = activity.topk(M, dim=-1).indices                # [B, M]

        Z_beam     = torch.gather(                                   # [B, M, D]
            Z, 1, top_idx.unsqueeze(-1).expand(-1, -1, Z.shape[-1])
        )
        Z_ref_beam = torch.gather(                                   # [B, M, D]
            Z_ref, 1, top_idx.unsqueeze(-1).expand(-1, -1, Z.shape[-1])
        )

        # ── 2. Score: feature similarity (+optional geometric penalty) ───────
        # Z is l2-normalised → dot product ∈ (-1, +1)
        # Neurons pointing in the same direction in activation space resonate.
        score = torch.einsum("bmd,bnd->bmn", Z_beam, Z)             # [B, M, N]

        if use_geo:
            # Gather W_pos of beam neurons: [B, M, D_pos]
            # top_idx is [B, M] — same indices for all D_pos dims
            W_pos = self.base.W_pos.detach()                         # [N+N_out, D_pos]
            W_pos_hidden = W_pos[:Z.shape[1]]                        # [N, D_pos]
            # Beam positions: index W_pos_hidden by top_idx (shared across batch)
            # top_idx varies per batch item; use advanced indexing
            beam_pos = W_pos_hidden[top_idx]                         # [B, M, D_pos]
            all_pos  = W_pos_hidden.unsqueeze(0).expand(             # [B, N, D_pos]
                Z.shape[0], -1, -1
            )
            # Squared L2 distance: [B, M, N]
            diff     = beam_pos.unsqueeze(2) - all_pos.unsqueeze(1)  # [B, M, N, D_pos]
            geo_dist = (diff * diff).sum(dim=-1)                     # [B, M, N]
            score    = score - self.geo_gamma * geo_dist

        # ── 3. Resonance threshold: kill weak connections entirely ───────────
        # Connections below threshold don't form at all (hard gate, not soft fade)
        score = score - self.resonance_threshold                     # shift by threshold
        gate  = score.clamp(min=0)                                   # zero below threshold

        # ── 4. Aggregate inhibitory signal weighted by resonance strength ────
        Z_inhib  = torch.einsum("bmn,bmd->bnd", gate, Z_ref_beam)   # [B, N, D]
        gate_sum = gate.sum(dim=1).unsqueeze(-1).clamp(min=1.0)     # [B, N, 1]
        return Z_inhib / gate_sum
