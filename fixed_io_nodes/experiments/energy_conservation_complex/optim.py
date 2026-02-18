"""
Optimizer for models that contain EnergyConservationNeurographLayer.
Uses same sink/accumulator logic as distributed GNNAdam (via gnn_optim_utils).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .gnn_optim_utils import clear_gnn_sinks, step_gnn_from_sinks
from .layer import EnergyConservationNeurographLayer


class EnergyConservationGNNAdam(optim.Adam):
    """
    Optimizer for models that contain EnergyConservationNeurographLayer(s).
    Head parameters updated every step with Adam; GNN parameters updated via
    each layer's gradient_sink -> accumulator.receive_gradients -> accumulator.step().
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        gnn_layers: List[EnergyConservationNeurographLayer] = [
            m for m in model.modules() if isinstance(m, EnergyConservationNeurographLayer)
        ]
        if not gnn_layers:
            raise ValueError(
                "Model has no EnergyConservationNeurographLayer. "
                "EnergyConservationGNNAdam is for models with the energy-conservation layer."
            )
        gradient_sinks = [layer.gradient_sink for layer in gnn_layers]
        accumulators = [layer.accumulator for layer in gnn_layers]
        head_params = list(model.parameters())

        super().__init__(
            head_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.gradient_sinks = gradient_sinks
        self.accumulators = accumulators

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none=set_to_none)
        clear_gnn_sinks(self.gradient_sinks)

    def step(self, closure=None):
        step_gnn_from_sinks(self.gradient_sinks, self.accumulators)
        return super().step(closure=closure)
