"""
In-process GNN layer: runs forward/backward in the main process (no worker pool).
Uses PhaseRadMagConductionGNN and exposes .gnn, .gradient_sink, .accumulator for GNNAdam-compatible training.
"""

import torch
import torch.nn as nn

from fixed_io_nodes.distributed._config_utils import get_config, get_node_store_from_config
from fixed_io_nodes.distributed.gnn_grad_sink import GNNGradientSink
from fixed_io_nodes.core.gradient_accumulator import UnquantizedGradientAccumulator
from fixed_io_nodes.core.full_model import get_dtype

from .gnn import PhaseRadMagConductionGNN


class _InProcessGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, layer):
        # h: (B, input_nodes, vector_dim)
        ctx.layer = layer
        ctx.save_for_backward(h)
        gnn = layer._gnn
        B = h.size(0)
        if not h.requires_grad:
            h = h.requires_grad_(True)
        outputs = []
        for i in range(B):
            if i > 0:
                gnn.reset(fetch_weights=False)
            out_i = gnn(h[i])
            outputs.append(out_i)
        ctx.outputs = outputs
        ctx.B = B
        return torch.stack(outputs)

    @staticmethod
    def backward(ctx, grad_output):
        layer = ctx.layer
        gnn = layer._gnn
        B = ctx.B
        h = ctx.saved_tensors[0]
        for i in range(B):
            out_i = ctx.outputs[i]
            if out_i.grad_fn is not None:
                out_i.backward(grad_output[i])
        phase_grads, mag_grads = gnn.get_grads()
        if phase_grads or mag_grads:
            layer._gradient_sink.add(phase_grads or {}, mag_grads or {})
        grad_input = h.grad if h.grad is not None else torch.zeros_like(h)
        return grad_input, None


class InProcessNeurographLayer(nn.Module):
    """
    Single GNN layer running in the main process (no worker pool).
    Uses PhaseRadMagConductionGNN. Exposes .gnn, .gradient_sink, .accumulator so
    GNNAdam-compatible optimizers can update GNN parameters.
    """

    def __init__(self, config):
        super().__init__()
        cfg = (
            get_config(path=config)
            if isinstance(config, str)
            else get_config(overrides=config)
        )
        self._config = cfg
        self._node_store = get_node_store_from_config(cfg)
        device = cfg["system"]["device"]
        if isinstance(device, str) and "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
        self._device = device
        dtype = get_dtype(cfg["model"].get("dtype", "float32"))

        self._gnn = PhaseRadMagConductionGNN(
            node_store=self._node_store,
            cardinality=cfg["graph"]["cardinality"],
            radiation_targets=cfg["graph"]["radiation_targets"],
            total_nodes=cfg["graph"]["total_nodes"],
            input_nodes=cfg["graph"]["input_nodes"],
            output_nodes=cfg["graph"]["output_nodes"],
            phase_bins=cfg["model"].get("phase_bins", 256),
            mag_bins=cfg["model"].get("mag_bins", 256),
            vector_dim=cfg["model"]["vector_dim"],
            iterations=cfg["model"]["iterations"],
            activation_threshold=cfg["model"]["activation_threshold"],
            gamma=cfg["model"].get("gamma", 1.0),
            temporal_decay=cfg["model"].get("temporal_decay", 1.0),
            device=device,
            verbose=cfg["system"].get("logging", {}).get("verbose", False),
            dtype=dtype,
        )
        self._gnn.sync_weights()
        self._gradient_sink = GNNGradientSink()
        self._accumulator = UnquantizedGradientAccumulator(
            node_store=self._node_store,
            lr=cfg["training"]["lr"],
            accumulation_steps=cfg["training"]["accumulation_steps"],
            betas=tuple(cfg["training"].get("betas", (0.9, 0.999))),
            eps=cfg["training"].get("eps", 1e-8),
            verbose=cfg["system"]["logging"].get("verbose", False),
            device=device,
        )

    @property
    def gnn(self):
        return self._gnn

    @property
    def gradient_sink(self):
        return self._gradient_sink

    @property
    def accumulator(self):
        return self._accumulator

    def forward(self, x):
        return _InProcessGNNFunction.apply(x, self)

    def shutdown(self):
        pass


def create_experiment_layer(**kwargs):
    """Build InProcessNeurographLayer with PhaseRadMagConductionGNN from kwargs (same API as create_gnn_layer)."""
    cfg = get_config(**kwargs)
    return InProcessNeurographLayer(cfg)
# </think>
# Saving input `h` in the autograd Function so we can use `h.grad` in backward. Fixing the layer.
# <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
# StrReplace