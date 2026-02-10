"""
GNN variant: conduction edges use real part of activation, radiation edges use imaginary part.
real = cos(phase)*cos(gamma*sin(mag)), imag = cos(phase)*sin(gamma*sin(mag)).
Subclasses UnquantizedGNN; only one_step_forward overridden.
"""

import torch
from typing import List

from fixed_io_nodes.core.gnn_model import UnquantizedGNN
from fixed_io_nodes.core.node import Node
from fixed_io_nodes.core.custom_functions import activation_strength_forward

from .utils import activation_real_imag


class RealImagConductionRadiationGNN(UnquantizedGNN):
    """
    Same as UnquantizedGNN except in each propagation step the scalar used as
    activation_strength per connection is: real for direct (conduction), imag for radiation.
    Phase and mag from senders are passed unchanged.
    """

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
                (self.input_node_count, self.vector_dim), device=input_values.device, dtype=input_values.dtype
            )
            activation_strengths = torch.zeros(
                self.input_node_count, device=input_values.device, dtype=input_values.dtype
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

        incoming_connections = {
            node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)
        }
        incoming_connection_types = {
            node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)
        }
        direct_connections_dict = {}

        for node in self.active_nodes.values():
            for n_id in node.outgoing_connections:
                incoming_connections[n_id].append(node.id)
                incoming_connection_types[n_id].append("direct")
                if node.id not in direct_connections_dict:
                    direct_connections_dict[node.id] = []
                direct_connections_dict[node.id].append(n_id)
        for n_id, targets in radiation_targets.items():
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
        real_per_node = {}
        imag_per_node = {}
        for n_id in phase_activations:
            r, i = activation_real_imag(
                phase_activations[n_id].unsqueeze(0),
                mag_activations[n_id].unsqueeze(0),
                self.gamma,
            )
            real_per_node[n_id] = r.squeeze(0)
            imag_per_node[n_id] = i.squeeze(0)

        for node in set(self.active_nodes.values()).union(new_nodes):
            if len(incoming_connections[node.id]) == 0:
                continue
            conn_ids: List[int] = incoming_connections[node.id]
            types: List[str] = incoming_connection_types[node.id]
            phase_list = []
            mag_list = []
            strength_list = []
            for n_id, ctype in zip(conn_ids, types):
                phase_list.append(phase_activations[n_id])
                mag_list.append(mag_activations[n_id])
                if ctype == "radiation":
                    strength_list.append(imag_per_node[n_id])
                else:
                    strength_list.append(real_per_node[n_id])

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

        del phase_activations, mag_activations, real_per_node, imag_per_node
        del incoming_connections, incoming_connection_types
        del new_nodes, new_nodes_values, radiation_targets
