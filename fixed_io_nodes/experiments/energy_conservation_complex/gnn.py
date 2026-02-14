"""
GNN with full complex activation e^(i*theta)*e^(gamma*sin m). Real part = conduction, imag = radiation.
Energy conservation: source splits its real (imag) among targets in proportion to conduction (radiation) alignment.
Activation strength = magnitude(real, imag); beam: only top nodes by strength propagate.
Decay = 0%. Forward only.
"""

import math
import torch
from typing import List, Set, Dict

from fixed_io_nodes.core.gnn_model import UnquantizedGNN
from fixed_io_nodes.core.node import Node

from .utils import (
    activation_real_imag,
    activation_strength_from_real_imag,
    activation_strength,
    theta_from_real_imag,
    conduction_radiation_alignments,
)


class EnergyConservationComplexGNN(UnquantizedGNN):
    """
    Complex activation (real = conduction, imag = radiation). Proportion-based split conserves energy.
    Beam: only top beam_top_frac nodes by activation strength (magnitude) propagate.
    temporal_decay=1.0 (0% decay). No backward pass.
    """

    def __init__(self, beam_top_frac: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beam_top_frac = beam_top_frac

    def _beam_sender_ids(self) -> Set[int]:
        """Top beam_top_frac of active nodes by activation strength (magnitude). At least 1 if any active."""
        active = list(self.active_nodes.values())
        if not active:
            return set()
        strengths = [(node.id, node.activation_strength.item()) for node in active]
        strengths.sort(key=lambda x: x[1], reverse=True)
        k = max(1, math.ceil(len(active) * self.beam_top_frac))
        return {n_id for n_id, _ in strengths[:k]}

    def one_step_forward(self, input_values: torch.Tensor = None, tracer=None):
        if tracer is not None:
            iteration_num = tracer.get_num_iterations() if input_values is None else 0
            tracer.start_iteration(iteration_num, input_injected=(input_values is not None))

        if input_values is not None:
            assert input_values.shape == (
                self.input_node_count,
                self.vector_dim,
            ), f"expected input_values of shape ({self.input_node_count}, {self.vector_dim}), got {input_values.shape}"
            input_phases = input_values
            input_mags = torch.zeros(
                (self.input_node_count, self.vector_dim),
                device=input_values.device,
                dtype=input_values.dtype,
            )
            for n_id, node in self.input_nodes.items():
                node.phase_activation = input_phases[n_id].clone()
                node.mag_activation = input_mags[n_id].clone()
                real, imag = activation_real_imag(
                    node.phase_activation.unsqueeze(0),
                    node.mag_activation.unsqueeze(0),
                    self.gamma,
                )
                node.activation_strength = activation_strength_from_real_imag(
                    real, imag, node.mag_activation.unsqueeze(0), self.gamma
                ).squeeze(0)
                if tracer is not None:
                    tracer.record_node_update(
                        node_id=node.id,
                        input_sources=[],
                        input_types=[],
                        phase_activation=node.phase_activation,
                        mag_activation=node.mag_activation,
                        activation_strength=node.activation_strength,
                    )

        radiation_targets = self._compute_radiation_targets(set(self.active_nodes.values()))
        if tracer is not None:
            tracer.record_radiation_targets(radiation_targets)

        new_active_nodes_ids = set(int(i) for i in self.active_nodes.keys())
        for node in self.active_nodes.values():
            new_active_nodes_ids.update(node.outgoing_connections)
        for _, targets in radiation_targets.items():
            new_active_nodes_ids.update(targets)

        nodes_to_fetch_ids = list(new_active_nodes_ids - set(int(i) for i in self.active_nodes.keys()))
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
        for i, nv in enumerate(new_nodes_values):
            new_nodes[i].load_values(nv)
        new_nodes = set(new_nodes)

        beam_sender_ids = self._beam_sender_ids()

        # Per-node current real/imag (for proportion and accumulation)
        phase_activations = {n.id: n.phase_activation.clone() for n in self.active_nodes.values()}
        mag_activations = {n.id: n.mag_activation.clone() for n in self.active_nodes.values()}
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

        # For new nodes, init phase/mag/real/imag from loaded weights
        for node in new_nodes:
            phase_activations[node.id] = node.phase_activation.clone()
            mag_activations[node.id] = node.mag_activation.clone()
            r, im = activation_real_imag(
                node.phase_activation.unsqueeze(0),
                node.mag_activation.unsqueeze(0),
                self.gamma,
            )
            real_per_node[node.id] = r.squeeze(0)
            imag_per_node[node.id] = im.squeeze(0)

        # Accumulated received real/imag (start with self)
        received_real: Dict[int, torch.Tensor] = {nid: real_per_node[nid].clone() for nid in real_per_node}
        received_imag: Dict[int, torch.Tensor] = {nid: imag_per_node[nid].clone() for nid in imag_per_node}

        # Build source -> list of (target_id, 'direct'|'radiation')
        source_to_targets: Dict[int, List[tuple]] = {}
        for node in self.active_nodes.values():
            if node.id not in beam_sender_ids:
                continue
            for t_id in node.outgoing_connections:
                source_to_targets.setdefault(node.id, []).append((t_id, "direct"))
        for n_id, targets in radiation_targets.items():
            if n_id not in beam_sender_ids:
                continue
            for t_id in targets:
                source_to_targets.setdefault(n_id, []).append((t_id, "radiation"))

        device = next(self.active_nodes.values()).phase_activation.device
        dtype = next(self.active_nodes.values()).phase_activation.dtype

        for source_id, target_list in source_to_targets.items():
            phase_s = phase_activations[source_id]
            mag_s = mag_activations[source_id]
            real_s = real_per_node[source_id]
            imag_s = imag_per_node[source_id]

            # Conduction targets (direct) and radiation targets; compute alignments per target
            conduction_alignments = []
            radiation_alignments = []
            target_ids = []
            for t_id, ctype in target_list:
                phase_t = phase_activations[t_id]
                mag_t = mag_activations[t_id]
                phase_eff = phase_t + phase_s
                c_align, r_align = conduction_radiation_alignments(phase_eff, mag_t, self.gamma)
                conduction_alignments.append(c_align)
                radiation_alignments.append(r_align)
                target_ids.append((t_id, ctype))

            sum_con = sum(conduction_alignments)
            sum_rad = sum(radiation_alignments)
            sum_con = sum_con if abs(sum_con.item()) > 1e-12 else torch.tensor(1.0, device=device, dtype=dtype)
            sum_rad = sum_rad if abs(sum_rad.item()) > 1e-12 else torch.tensor(1.0, device=device, dtype=dtype)

            for idx, (t_id, ctype) in enumerate(target_ids):
                p_con = (conduction_alignments[idx] / sum_con).item()
                p_rad = (radiation_alignments[idx] / sum_rad).item()
                received_real[t_id] = received_real[t_id] + real_s * p_con
                received_imag[t_id] = received_imag[t_id] + imag_s * p_rad

        # Write back: theta = atan2(imag, real), magnitude = sqrt(real^2+imag^2); set phase_activation, mag_activation, activation_strength
        all_nodes = set(self.active_nodes.values()) | new_nodes
        for node in all_nodes:
            nid = node.id
            real = received_real[nid]
            imag = received_imag[nid]
            theta = theta_from_real_imag(real, imag)
            magnitude = activation_strength_from_real_imag(real, imag)
            # phase_activation: broadcast theta to vector_dim
            node.phase_activation = theta.unsqueeze(0).expand(self.vector_dim).to(device=device, dtype=dtype)
            # mag so that exp(gamma*sin(mag)) = magnitude => sin(mag) = ln(magnitude)/gamma, clamp to [-1,1]
            log_m = torch.log(magnitude + 1e-12)
            sin_mag = (log_m / self.gamma).clamp(-1.0, 1.0)
            mag_val = torch.arcsin(sin_mag)
            node.mag_activation = mag_val.unsqueeze(0).expand(self.vector_dim).to(device=device, dtype=dtype)
            # activation_strength = magnitude * exp(gamma * sin m)
            node.activation_strength = activation_strength_from_real_imag(
                real, imag, node.mag_activation, self.gamma
            ).to(device=device, dtype=dtype)

            if tracer is not None:
                conn_ids = []
                types = []
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
            tracer.record_active_nodes([int(k) for k in self.active_nodes.keys()])

        # 0% decay: temporal_decay=1.0, skip decay call
        if input_values is None and self.temporal_decay < 1.0:
            for node in self.active_nodes.values():
                node.decay_activations(self.temporal_decay)

        del phase_activations, mag_activations, real_per_node, imag_per_node
        del received_real, received_imag, source_to_targets
        del new_nodes, new_nodes_values, radiation_targets
