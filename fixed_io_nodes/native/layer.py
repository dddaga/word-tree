import torch
import torch.nn as nn
from typing import Optional, Union

from core.custom_functions import update_activations, activation_strength_forward
from .node_store import NativeNodeStore


class NativeNeurographLayer(nn.Module):
    """
    Single-process GNN layer. Drop-in replacement for DistributedNeurographLayer.
    All computation on main thread; phase_weight/mag_weight are nn.Parameters
    with native autograd gradient tracking (no custom autograd.Function).
    """

    def __init__(self, config: Union[str, dict]):
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

        self._input_nodeids = sorted(self._node_store.input_nodeids)
        self._output_nodeids = sorted(self._node_store.output_nodeids)
        self._input_idx = None  # lazily built on correct device

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_nodes, vector_dim) → (B, output_nodes)"""
        x = torch.tanh(x) * torch.pi
        B = x.shape[0]
        outputs = []
        for i in range(B):
            outputs.append(self._forward_single(x[i]))
        return torch.stack(outputs)

    # ------------------------------------------------------------------
    # Per-sample GNN forward
    # ------------------------------------------------------------------

    def _get_input_idx(self, device: torch.device) -> torch.Tensor:
        if self._input_idx is None or self._input_idx.device != device:
            self._input_idx = torch.tensor(self._input_nodeids, device=device, dtype=torch.long)
        return self._input_idx

    def _forward_single(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Process one sample through the GNN.
        input_values: (input_nodes, vector_dim)
        Returns: (output_nodes,)

        Key: clone parameters WITHOUT detach so gradients flow back.
        """
        device = input_values.device

        # Differentiable clone — gradients propagate back to nn.Parameters
        phase_weight = self._node_store.phase_weight.clone()
        mag_weight = self._node_store.mag_weight.clone()

        # Initialize activations from weights
        phase_act = phase_weight.clone()
        mag_act = mag_weight.clone()
        act_strength = activation_strength_forward(phase_act, mag_act, self._gamma)

        active_mask = torch.zeros(self._total_nodes, dtype=torch.bool, device=device)
        input_idx = self._get_input_idx(device)
        active_mask[input_idx] = True

        # First iteration: inject inputs then propagate
        phase_act, mag_act, act_strength, active_mask = self._inject_inputs(
            input_values, phase_act, mag_act, act_strength, active_mask,
            phase_weight, mag_weight, input_idx, device,
        )
        phase_act, mag_act, act_strength, active_mask = self._one_step_propagate(
            phase_act, mag_act, act_strength, active_mask,
            phase_weight, mag_weight, device,
        )

        # Remaining iterations: propagate only
        for _ in range(self._iterations - 2):
            phase_act, mag_act, act_strength, active_mask = self._one_step_propagate(
                phase_act, mag_act, act_strength, active_mask,
                phase_weight, mag_weight, device,
            )

        # Extract output
        output_idx = torch.tensor(self._output_nodeids, device=device, dtype=torch.long)
        return act_strength[output_idx] / (self._vector_dim ** 0.5)

    # ------------------------------------------------------------------
    # GNN sub-operations (adapted from UnquantizedGNN in core/gnn_model.py)
    # ------------------------------------------------------------------

    def _inject_inputs(self, input_values, phase_act, mag_act, act_strength,
                       active_mask, phase_weight, mag_weight, input_idx, device):
        """Inject external input into input nodes via update_activations."""
        n_in = self._input_node_count

        input_phase = input_values.to(device)
        input_mag = torch.zeros_like(input_values, device=device)
        input_act_strength = activation_strength_forward(
            phase_act[input_idx], mag_act[input_idx], self._gamma
        )

        # Edge index: virtual input sources → input nodes
        edge_index = torch.empty((2, n_in), device=device, dtype=torch.long)
        edge_index[0] = input_idx + n_in  # virtual source indices
        edge_index[1] = input_idx

        # Concatenate node state with virtual input state
        pa = torch.cat([phase_act[input_idx], input_phase], dim=0)
        ma = torch.cat([mag_act[input_idx], input_mag], dim=0)
        a_s = torch.cat([act_strength[input_idx], input_act_strength], dim=0)
        pw = torch.cat([phase_weight[input_idx], torch.empty_like(input_phase)], dim=0)
        mw = torch.cat([mag_weight[input_idx], torch.empty_like(input_mag)], dim=0)

        new_pa, new_ma, new_as = update_activations(pa, ma, pw, mw, a_s, edge_index)

        # Update only input nodes (out-of-place)
        phase_act = phase_act.clone()
        mag_act = mag_act.clone()
        act_strength = act_strength.clone()
        phase_act[input_idx] = new_pa[:n_in]
        mag_act[input_idx] = new_ma[:n_in]
        act_strength[input_idx] = new_as[:n_in]
        active_mask[input_idx] = True

        return phase_act, mag_act, act_strength, active_mask

    def _one_step_propagate(self, phase_act, mag_act, act_strength, active_mask,
                            phase_weight, mag_weight, device):
        """One message-passing iteration: build edges, update activations."""
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        if active_indices.numel() == 0:
            return phase_act, mag_act, act_strength, active_mask

        edge_index, active_mask = self._build_edge_index(active_indices, active_mask, device)

        new_phase, new_mag, new_strength = update_activations(
            phase_act, mag_act, phase_weight, mag_weight, act_strength, edge_index
        )
        return new_phase, new_mag, new_strength, active_mask

    def _build_edge_index(self, active_indices, active_mask, device):
        """Combine static edges (filtered by active mask) with radiation targets."""
        edge_indices = self._node_store.edge_indices.to(device)
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
            rand_n = torch.empty(active_indices.shape[0], num_random, device=device, dtype=torch.long)
            for i in range(active_indices.shape[0]):
                rand_n[i] = torch.tensor(
                    self._node_store.sample_random_nodeids(num_random),
                    device=device, dtype=torch.long,
                )
            neighbours = torch.cat([neighbours, rand_n], dim=1)

        return neighbours
