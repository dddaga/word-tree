import torch
import torch.nn as nn
from typing import Optional, Union

from torch.utils.checkpoint import checkpoint as grad_checkpoint
from core.custom_functions import update_activations, activation_strength_forward
from .node_store import NativeNodeStore


class NativeNeurographLayer(nn.Module):
    """
    Single-process GNN layer. Drop-in replacement for DistributedNeurographLayer.
    All computation on main thread; phase_weight/mag_weight are nn.Parameters
    with native autograd gradient tracking (no custom autograd.Function).
    """

    def __init__(self, config: Union[str, dict], use_layer_norm: bool = True):
        super().__init__()
        if isinstance(config, str):
            import yaml
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        cfg = config

        graph = cfg["graph"]
        model = cfg["model"]
        system = cfg.get("system", {})

        self._node_store = NativeNodeStore(
            total_nodes=graph["total_nodes"],
            input_nodes=graph["input_nodes"],
            output_nodes=graph["output_nodes"],
            cardinality=graph["cardinality"],
            vector_dim=model["vector_dim"],
            seed=system.get("random_seed", 42),
            device=system.get("device", "cpu"),
        )

        self._total_nodes = graph["total_nodes"]
        self._input_node_count = graph["input_nodes"]
        self._output_node_count = graph["output_nodes"]
        self._vector_dim = model["vector_dim"]
        self._iterations = model["iterations"]
        self._radiation_targets = graph["radiation_targets"]
        self._gamma = model.get("gamma", 1.0)
        self._scattering_prob_base = model.get("scattering_prob", 0.0)
        self._stochastic_radiation_duration = model.get("stochastic_radiation_duration", 0.5)
        self._current_scattering_prob = None

        self._use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.mag_norm = nn.LayerNorm(self._vector_dim, elementwise_affine=True)

        self._input_nodeids = sorted(self._node_store.input_nodeids)
        self._output_nodeids = sorted(self._node_store.output_nodeids)
        self._input_idx = None  # lazily built on correct device
        self._output_idx = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_training_progress(self, step: int, total_steps: int) -> None:
        if self._scattering_prob_base == 0:
            self._current_scattering_prob = 0.0
            return
        if step >= total_steps / 2:
            self._current_scattering_prob = 0.0
        else:
            duration = self._stochastic_radiation_duration
            self._current_scattering_prob = self._scattering_prob_base * max(
                0.0, 1.0 - step / (total_steps * duration)
            )

    def _get_input_idx(self, device: torch.device) -> torch.Tensor:
        if self._input_idx is None or self._input_idx.device != device:
            self._input_idx = torch.tensor(self._input_nodeids, device=device, dtype=torch.long)
        return self._input_idx

    def _get_output_idx(self, device: torch.device) -> torch.Tensor:
        if self._output_idx is None or self._output_idx.device != device:
            self._output_idx = torch.tensor(self._output_nodeids, device=device, dtype=torch.long)
        return self._output_idx

    # ------------------------------------------------------------------
    # Batched mega-graph forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_nodes, vector_dim) → (B, output_nodes)

        All B samples processed simultaneously as one mega-graph with B*N nodes.
        Edge construction happens once (identical across samples), then replicated.
        """
        x = torch.tanh(x) * torch.pi
        B = x.shape[0]
        N = self._total_nodes
        V = self._vector_dim
        device = x.device
        n_in = self._input_node_count

        # Replicate weights for B samples: (N, V) → (B*N, V)
        # clone() without detach keeps gradient connection to nn.Parameter
        phase_weight = self._node_store.phase_weight.clone().repeat(B, 1)
        mag_weight = self._node_store.mag_weight.clone().repeat(B, 1)

        # Initialize activations from weights
        phase_act = phase_weight.clone()
        mag_act = mag_weight.clone()
        if self._use_layer_norm:
            mag_act = self.mag_norm(mag_act)
        act_strength = activation_strength_forward(phase_act, mag_act, self._gamma)

        # Build a closure that optionally applies LayerNorm before update_activations,
        # so that both run inside grad_checkpoint (avoids retaining LN intermediates).
        if self._use_layer_norm:
            mag_norm = self.mag_norm
            def _normed_update(pa, ma, pw, mw, a_s, edges, wr, wi, all_act):
                return update_activations(pa, mag_norm(ma), pw, mw, a_s, edges, wr, wi, all_act)
            _update_fn = _normed_update
        else:
            _update_fn = update_activations

        # Index helpers
        input_idx = self._get_input_idx(device)
        output_idx = self._get_output_idx(device)
        offsets = torch.arange(B, device=device) * N
        batched_input_idx = (input_idx.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

        # Active mask — single-sample, since edges are identical across samples
        active_mask = torch.zeros(N, dtype=torch.bool, device=device)
        active_mask[input_idx] = True

        # Inject inputs (batched)
        phase_act, mag_act, act_strength = self._inject_inputs_batched(
            x.reshape(B * n_in, V), phase_act, mag_act, act_strength,
            phase_weight, mag_weight, batched_input_idx, B, n_in, device,
        )

        # Pre-compute batched full edges for when active_mask saturates
        full_edges = self._node_store.edge_indices
        batched_full_edges = self._replicate_edges(full_edges, offsets, B)
        all_active = False

        # Pre-compute weight trig once (constant across iterations)
        w_real = mag_weight * torch.cos(phase_weight)
        w_imag = mag_weight * torch.sin(phase_weight)

        # Propagation iterations
        for _ in range(self._iterations - 1):
            if all_active:
                batched_edges = batched_full_edges
            else:
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                if active_indices.numel() == 0:
                    break
                edge_index, active_mask = self._build_edge_index(active_indices, active_mask, device)
                if active_mask.all():
                    all_active = True
                    batched_edges = batched_full_edges
                else:
                    batched_edges = self._replicate_edges(edge_index, offsets, B)
            phase_act, mag_act, act_strength = grad_checkpoint(
                _update_fn,
                phase_act, mag_act, phase_weight, mag_weight, act_strength, batched_edges,
                w_real, w_imag, all_active,
                use_reentrant=False,
            )

        # Extract outputs
        batched_output_idx = (output_idx.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        return act_strength[batched_output_idx].view(B, -1) / (V ** 0.5)

    def _inject_inputs_batched(self, x_flat, phase_act, mag_act, act_strength,
                               phase_weight, mag_weight, batched_input_idx, B, n_in, device):
        """Batched input injection for all B samples at once.

        Builds a mini mega-graph: [B*n_in existing activations | B*n_in virtual inputs]
        with edges from virtual sources to existing destinations.
        """
        Bn = B * n_in

        existing_pa = phase_act[batched_input_idx]
        existing_ma = mag_act[batched_input_idx]
        existing_as = act_strength[batched_input_idx]
        virtual_mag = torch.zeros_like(x_flat)
        virtual_as = activation_strength_forward(existing_pa, existing_ma, self._gamma)

        pa = torch.cat([existing_pa, x_flat], dim=0)
        ma = torch.cat([existing_ma, virtual_mag], dim=0)
        a_s = torch.cat([existing_as, virtual_as], dim=0)
        pw = torch.cat([phase_weight[batched_input_idx], torch.empty_like(x_flat)], dim=0)
        mw = torch.cat([mag_weight[batched_input_idx], torch.empty_like(x_flat)], dim=0)

        dst = torch.arange(Bn, device=device)
        inject_edges = torch.stack([dst + Bn, dst])

        new_pa, new_ma, new_as = update_activations(pa, ma, pw, mw, a_s, inject_edges)

        # Write back (out-of-place for autograd safety)
        phase_act = phase_act.clone()
        mag_act = mag_act.clone()
        act_strength = act_strength.clone()
        phase_act[batched_input_idx] = new_pa[:Bn]
        mag_act[batched_input_idx] = new_ma[:Bn]
        act_strength[batched_input_idx] = new_as[:Bn]

        return phase_act, mag_act, act_strength

    def _replicate_edges(self, edge_index, offsets, B):
        """Replicate single-sample edges for B samples with node-index offsets."""
        src = edge_index[0].unsqueeze(0) + offsets.unsqueeze(1)
        dst = edge_index[1].unsqueeze(0) + offsets.unsqueeze(1)
        return torch.stack([src.reshape(-1), dst.reshape(-1)])

    # ------------------------------------------------------------------
    # Edge construction (single-sample, then replicated by forward)
    # ------------------------------------------------------------------

    def _build_edge_index(self, active_indices, active_mask, device):
        """Combine static edges (filtered by active mask) with radiation targets."""
        edge_indices = self._node_store.edge_indices
        mask = active_mask[edge_indices[0]]
        direct_edges = edge_indices[:, mask]

        # Radiation targets
        radiation_neighbours = self._compute_radiation_targets(active_indices, device)
        n_rad = radiation_neighbours.numel()
        if n_rad > 0:
            rad_edge = torch.empty((2, n_rad), device=device, dtype=torch.long)
            rad_edge[1] = radiation_neighbours.flatten()
            rad_edge[0] = active_indices.repeat_interleave(radiation_neighbours.shape[1])
            active_edge_index = torch.cat([direct_edges, rad_edge], dim=1)
        else:
            active_edge_index = direct_edges

        # Mark newly reached nodes
        active_mask = active_mask.clone()
        active_mask[active_edge_index[1]] = True
        return active_edge_index, active_mask

    def _compute_radiation_targets(self, active_indices, device):
        """Find radiation targets via cosine search + random sampling."""
        k = self._radiation_targets
        scattering_prob = (
            self._current_scattering_prob
            if self._current_scattering_prob is not None
            else self._scattering_prob_base
        )
        scattering_prob = max(0.0, min(1.0, scattering_prob))

        num_random = int(k * scattering_prob)
        num_searched = k - num_random

        neighbours = torch.empty((active_indices.shape[0], 0), device=device, dtype=torch.long)

        if num_searched > 0:
            # Use current phase activations as query (detached for non-diff search)
            query = self._node_store.phase_weight.data[active_indices]
            found_idx, _ = self._node_store.search_nodes_batch(
                query, vector_name="phase", limit=num_searched,
            )
            neighbours = torch.cat([neighbours, found_idx.to(device)], dim=1)

        if num_random > 0:
            rand_n = torch.randint(0, self._total_nodes, (active_indices.shape[0], num_random), device=device)
            neighbours = torch.cat([neighbours, rand_n], dim=1)

        return neighbours
