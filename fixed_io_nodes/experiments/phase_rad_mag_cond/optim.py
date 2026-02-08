"""
Optimizer for models that contain InProcessNeurographLayer (experiment).
Same behavior as GNNAdam: head params updated with Adam; GNN grads from sink -> accumulator -> step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from .layer import InProcessNeurographLayer


class ExperimentGNNAdam(optim.Adam):
    """
    Optimizer for models that contain InProcessNeurographLayer(s).
    Head parameters updated every step with Adam; GNN parameters updated via
    each layer's gradient_sink -> accumulator.step().
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
        gnn_layers: List[InProcessNeurographLayer] = [
            m for m in model.modules() if isinstance(m, InProcessNeurographLayer)
        ]
        if not gnn_layers:
            raise ValueError(
                "Model has no InProcessNeurographLayer. "
                "ExperimentGNNAdam is for models with the experiment layer."
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
        for sink in self.gradient_sinks:
            if hasattr(sink, "clear"):
                sink.clear()

    def step(self, closure=None):
        for sink, accumulator in zip(self.gradient_sinks, self.accumulators):
            phase_grads, mag_grads, phase_grad_freq, mag_grad_freq = sink.get_and_clear()
            if phase_grads is not None or mag_grads is not None:
                if phase_grads is None:
                    phase_grads = {}
                if mag_grads is None:
                    mag_grads = {}
                accumulator.receive_gradients(
                    phase_grads,
                    mag_grads,
                    phase_grad_freq=phase_grad_freq,
                    mag_grad_freq=mag_grad_freq,
                )
                accumulator.step()
        return super().step(closure=closure)
