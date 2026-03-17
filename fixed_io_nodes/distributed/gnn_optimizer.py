"""
Single optimizer for training with both head parameters and GNN parameters.
Head params are updated every step with Adam; GNN params are accumulated
and updated via GradientAccumulator when count >= accumulation_steps.

Usage: optimizer = GNNAdam(model, lr=..., betas=..., eps=...)
The optimizer discovers all DistributedNeurographLayer instances in the model
and uses their gradient_sink and accumulator; no manual wiring needed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

from core.gradient_accumulator import UnquantizedGradientAccumulator
from .gnn_grad_sink import GNNGradientSink
from .layer import DistributedNeurographLayer


class GNNAdam(optim.Adam):
    """
    Optimizer for models that contain DistributedNeurographLayer(s).
    Head parameters are updated every step() with Adam; GNN parameters
    are updated via each layer's accumulator when count >= accumulation_steps.
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
        """
        Args:
            model: Module that contains at least one DistributedNeurographLayer.
                   Head parameters come from model.parameters(); GNN state is
                   discovered from each DistributedNeurographLayer (gradient_sink, accumulator).
            lr: Learning rate for head parameters (Adam)
            betas: Adam beta parameters
            eps: Adam epsilon
            weight_decay: Weight decay for head parameters
            amsgrad: Whether to use AMSGrad variant
        """
        gnn_layers: List[DistributedNeurographLayer] = [
            m for m in model.modules() if isinstance(m, DistributedNeurographLayer)
        ]
        if not gnn_layers:
            raise ValueError(
                "Model has no DistributedNeurographLayer. "
                "GNNAdam is only for models that contain at least one DistributedNeurographLayer."
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
        self.gnn_layers = gnn_layers

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero gradients only for head parameters. GNN gradients are handled
        by the accumulators and cleared when consumed.
        """
        super().zero_grad(set_to_none=set_to_none)
        # Clear all gradient sinks and reset the worker graph states
        for sink in self.gradient_sinks:
            if hasattr(sink, 'clear'):
                sink.clear()
                
        for layer in self.gnn_layers:
            pool = getattr(layer, "_pool", None)
            if pool is not None:
                pool.reset_workers()

    def step(self, closure=None):
        """
        Perform optimization step:
        1. Get GNN gradients from each sink and feed to corresponding accumulator
        2. Run accumulator.step() for each accumulator (independent layers)
        3. Run Adam step for head parameters
        """
        # Step 1 & 2: Process each layer's gradients independently
        for sink, accumulator in zip(self.gradient_sinks, self.accumulators):
            active_indices, phase_grads, mag_grads = sink.get_and_clear()
            if active_indices is not None:
                accumulator.receive_gradients(active_indices, phase_grads, mag_grads)
                

                # Run accumulator step (updates GNN params via node_store)
                accumulator.step()
        
        # Step 3: Run Adam step for head parameters
        return super().step(closure=closure)
