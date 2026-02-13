"""
GNN variant: radiation carries phase only; conduction (direct) carries magnitude only.
Beam search: only the top fraction of active nodes by activation strength may propagate.
"""

import math
import torch
from typing import List, Set

from fixed_io_nodes.core.gnn_model import UnquantizedGNN
from fixed_io_nodes.core.node import Node
from fixed_io_nodes.core.custom_functions import activation_strength_forward


class PhaseRadMagConductionGNN(UnquantizedGNN):
    """
    Same as UnquantizedGNN except in each propagation step:
    - Radiation: pass sender's phase and activation_strength; magnitude = 0.
    - Conduction: pass sender's magnitude and activation_strength; phase = 0.
    - Beam: only the top beam_top_frac (e.g. 10%) of active nodes by activation strength
      may send via radiation or conduction.
    """

    def __init__(self, beam_top_frac: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beam_top_frac = beam_top_frac

    def _beam_sender_ids(self) -> Set[int]:
        """Top beam_top_frac of active nodes by activation strength (descending). At least 1 if any active."""
        active = list(self.active_nodes.values())
        if not active:
            return set()
        strengths = [(node.id, node.activation_strength.item()) for node in active]
        strengths.sort(key=lambda x: x[1], reverse=True)
        k = max(1, math.ceil(len(active) * self.beam_top_frac))
        return {n_id for n_id, _ in strengths[:k]}

    def one_step_forward(self, input_values: torch.Tensor = None, tracer=None):
        if self.verbose:
            pass
        if tracer is not None:
            iteration_num = tracer.get_num_iterations() if input_values is None else 0
            tracer.start_iteration(iteration_num, input_injected=(input_values is not None))

        if input_values is not None:
            assert input_values.shape == (
                self.input_node_count,
                self.vector_dim,
            ), f"expected input_values of shape ({self.input_node_count}, {self.vector_dim}), but got {input_values.shape}"
            input_phases = input_values
            input_mags = torch.zeros(
                (self.input_node_count, self.vector_dim), device=input_values.device
            )
            activation_strengths = torch.zeros(
                self.input_node_count, device=input_values.device
            )
            for i in range(self.input_node_count):
                activation_strengths[i] = activation_strength_forward(
                    input_phases[i], input_mags[i], self.gamma
                )
            for n_id, node in self.input_nodes.items():
                node.update_activations(
                    phase_activations=input_phases[n_id],
                    mag_activations=input_mags[n_id],
                    activation_strengths=activation_strengths[n_id],
                )
                if tracer is not None:
                    tracer.record_node_update(
                        node_id=node.id,
                        input_sources=[],
                        input_types=[],
                        phase_activation=node.phase_activation,
                        mag_activation=node.mag_activation,
                        activation_strength=node.activation_strength,
                    )

        radiation_targets = self._compute_radiation_targets(
            set(self.active_nodes.values())
        )
        if tracer is not None:
            tracer.record_radiation_targets(radiation_targets)

        new_active_nodes_ids = set(int(i) for i in self.active_nodes.keys())
        for node in self.active_nodes.values():
            new_active_nodes_ids.update(node.outgoing_connections)
        for _, targets in radiation_targets.items():
            new_active_nodes_ids.update(targets)

        nodes_to_fetch_ids = list(
            new_active_nodes_ids - set(int(i) for i in self.active_nodes.keys())
        )
        new_nodes_values = self.node_store.get_node(nodes_to_fetch_ids)

        new_nodes = [
            Node(
                node_store=self.node_store,
                gamma=self.gamma,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in new_nodes_values
        ]
        for i, new_node_value in enumerate(new_nodes_values):
            new_nodes[i].load_values(new_node_value)
        new_nodes = set(new_nodes)

        beam_sender_ids = self._beam_sender_ids()

        incoming_connections = {
            node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)
        }
        incoming_connection_types = {
            node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)
        }
        direct_connections_dict = {}

        for node in self.active_nodes.values():
            if node.id not in beam_sender_ids:
                continue
            for n_id in node.outgoing_connections:
                incoming_connections[n_id].append(node.id)
                incoming_connection_types[n_id].append("direct")
                if node.id not in direct_connections_dict:
                    direct_connections_dict[node.id] = []
                direct_connections_dict[node.id].append(n_id)
        for n_id, targets in radiation_targets.items():
            if n_id not in beam_sender_ids:
                continue
            for target in targets:
                incoming_connections[target].append(n_id)
                incoming_connection_types[target].append("radiation")

        if tracer is not None:
            tracer.record_direct_connections(direct_connections_dict)

        phase_activations = {
            node.id: node.phase_activation.clone()
            for node in self.active_nodes.values()
        }
        mag_activations = {
            node.id: node.mag_activation.clone() for node in self.active_nodes.values()
        }
        activation_strengths = {
            node.id: node.activation_strength.clone()
            for node in self.active_nodes.values()
        }

        # Experiment: radiation -> phase only (mag=0); direct -> magnitude only (phase=0)
        for node in set(self.active_nodes.values()).union(new_nodes):
            if len(incoming_connections[node.id]) == 0:
                continue
            conn_ids: List[int] = incoming_connections[node.id]
            types: List[str] = incoming_connection_types[node.id]
            phase_list = []
            mag_list = []
            strength_list = []
            device = node.phase_activation.device
            dtype_p = node.phase_activation.dtype
            dtype_m = node.mag_activation.dtype
            for n_id, ctype in zip(conn_ids, types):
                strength_list.append(activation_strengths[n_id])
                if ctype == "radiation":
                    phase_list.append(phase_activations[n_id])
                    mag_list.append(
                        torch.zeros(
                            self.vector_dim, device=device, dtype=dtype_m
                        )
                    )
                else:
                    phase_list.append(
                        torch.zeros(
                            self.vector_dim, device=device, dtype=dtype_p
                        )
                    )
                    mag_list.append(mag_activations[n_id])

            node.update_activations(
                phase_activations=torch.stack(phase_list),
                mag_activations=torch.stack(mag_list),
                activation_strengths=torch.stack(strength_list),
            )
            if tracer is not None:
                tracer.record_node_update(
                    node_id=node.id,
                    input_sources=conn_ids,
                    input_types=types,
                    phase_activation=node.phase_activation,
                    mag_activation=node.mag_activation,
                    activation_strength=node.activation_strength,
                )

        for node in new_nodes:
            self.active_nodes[node.id] = node

        if tracer is not None:
            all_active_node_ids = [
                int(node_id) for node_id in self.active_nodes.keys()
            ]
            tracer.record_active_nodes(all_active_node_ids)

        if input_values is None:
            for node in self.active_nodes.values():
                node.decay_activations(self.temporal_decay)

        del phase_activations, mag_activations, activation_strengths
        del incoming_connections, incoming_connection_types
        del new_nodes, new_nodes_values, radiation_targets
