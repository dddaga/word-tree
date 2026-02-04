#!/usr/bin/env python3
"""
Visualize forward pass using ForwardPassTracer and web interface.

When --distributed: uses Model from distributed_training as-is. Runs forward on the
entire model; GNN layers are temporarily replaced with in-process GNNs that capture
tracer output, so no separate model or per-layer inputs are required.

Usage:
    python visualize_forward_pass.py --config configs/main_config3.yaml
    python visualize_forward_pass.py --config configs/main_config3.yaml --input <input_file>
    python visualize_forward_pass.py --config training_runs/distributed_test/distributed.yaml --distributed
    python visualize_forward_pass.py --config training_runs/distributed_test/distributed.yaml --distributed --input raw.pt
"""

import argparse
import yaml
import torch
import torch.nn as nn
import sys
from pathlib import Path

from core import initialize_model_and_nodestore
from forward_pass_tracer import ForwardPassTracer
from viz.adapter import TraceVisualizer
from viz.server import app, init_with_visualizer
import uvicorn

from distributed_training import IrisGNNModel as Model
from distributed import DistributedNeurographLayer


def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_sample_input(config, device):
    """Create a sample input tensor (raw input for single-model config)."""
    from torch.utils.data import TensorDataset
    from sklearn.datasets import load_iris

    input_nodes = config["graph"]["input_nodes"]
    vector_dim = config["model"]["vector_dim"]

    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 4)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    sample_input = X_tensor[0]
    return sample_input


def create_sample_input_gnn(config, device):
    """Create GNN input tensor (1, input_nodes, vector_dim) for distributed-style config."""
    input_nodes = config["graph"]["input_nodes"]
    vector_dim = config["model"]["vector_dim"]
    x = torch.randn(input_nodes, vector_dim, device=device, dtype=torch.float32)
    return x


def _get_qdrant_params(config):
    try:
        from main import get_qdrant_params
        return get_qdrant_params(config) or {}
    except Exception:
        return {}


def _build_model_for_config(config_path, device, distributed):
    """Build (model, node_store) for one layer from config path."""
    config = load_config(config_path)
    qdrant_params = _get_qdrant_params(config) if distributed else {}
    phase_bins = config["model"].get("phase_bins") or 256
    mag_bins = config["model"].get("mag_bins") or 256
    model, node_store = initialize_model_and_nodestore(
        qdrant_url=config["qdrant"]["url"],
        collection_name=config["qdrant"]["collection_name"],
        total_nodes=config["graph"]["total_nodes"],
        input_nodes=config["graph"]["input_nodes"],
        output_nodes=config["graph"]["output_nodes"],
        cardinality=config["graph"]["cardinality"],
        radiation_targets=config["graph"]["radiation_targets"],
        vector_dim=config["model"]["vector_dim"],
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        iterations=config["model"]["iterations"],
        activation_threshold=config["model"]["activation_threshold"],
        gamma=config["model"]["gamma"],
        device=device,
        temporal_decay=config["model"].get("temporal_decay", 1.0),
        radiation_similarity_threshold=config["model"].get(
            "radiation_similarity_threshold", 0.0
        ),
        qdrant_params=qdrant_params,
        dtype=config["model"].get("dtype", "float32"),
    )
    return model.to(device), node_store, config


def run_one_gnn_layer_with_tracing(model, input_tensor, device):
    """Run one GNN forward with tracer; return (trace_data, output_tensor)."""
    model.reset()
    with ForwardPassTracer() as tracer:
        output = model(input_tensor, tracer=tracer)
        trace_data = tracer.get_trace()
    return trace_data, output


class _TracerCaptureWrapper(nn.Module):
    """Wraps an in-process GNN; on forward(h) runs GNN with tracer and captures trace."""

    def __init__(self, gnn_module, device):
        super().__init__()
        self._gnn = gnn_module.to(device)
        self._trace = None

    def forward(self, h):
        # Core GNN expects (input_nodes, vector_dim), not (batch, input_nodes, vector_dim)
        squeezed = h.dim() == 3 and h.size(0) == 1
        if squeezed:
            h = h.squeeze(0)
        self._gnn.reset()
        with ForwardPassTracer() as tracer:
            out = self._gnn(h, tracer=tracer)
            self._trace = tracer.get_trace()
        if squeezed and out.dim() == 1:
            out = out.unsqueeze(0)
        return out

    def get_trace(self):
        return self._trace


def _replace_gnn_layers_with_tracer_capture(model, config, device, distributed):
    """
    Find all DistributedNeurographLayer in model and replace with wrappers that run
    in-process GNN with tracer. Returns list of (wrapper, node_store) to read traces from after forward.
    """
    out = []
    qdrant_params = _get_qdrant_params(config) if distributed else {}
    phase_bins = config["model"].get("phase_bins") or 256
    mag_bins = config["model"].get("mag_bins") or 256

    def build_gnn():
        gnn_model, node_store = initialize_model_and_nodestore(
            qdrant_url=config["qdrant"]["url"],
            collection_name=config["qdrant"]["collection_name"],
            total_nodes=config["graph"]["total_nodes"],
            input_nodes=config["graph"]["input_nodes"],
            output_nodes=config["graph"]["output_nodes"],
            cardinality=config["graph"]["cardinality"],
            radiation_targets=config["graph"]["radiation_targets"],
            vector_dim=config["model"]["vector_dim"],
            phase_bins=phase_bins,
            mag_bins=mag_bins,
            iterations=config["model"]["iterations"],
            activation_threshold=config["model"]["activation_threshold"],
            gamma=config["model"]["gamma"],
            device=device,
            temporal_decay=config["model"].get("temporal_decay", 1.0),
            radiation_similarity_threshold=config["model"].get(
                "radiation_similarity_threshold", 0.0
            ),
            qdrant_params=qdrant_params,
            dtype=config["model"].get("dtype", "float32"),
        )
        return gnn_model, node_store

    for name, child in list(model.named_children()):
        if isinstance(child, DistributedNeurographLayer):
            gnn_module, node_store = build_gnn()
            wrapper = _TracerCaptureWrapper(gnn_module, device)
            setattr(model, name, wrapper)
            out.append((wrapper, node_store))
        else:
            sub = _replace_gnn_layers_with_tracer_capture(child, config, device, distributed)
            out.extend(sub)
    return out


def run_forward_pass_with_tracing(
    config_path, input_tensor=None, save_trace=None, distributed=False
):
    """
    Run a forward pass with tracing and return trace data.
    When distributed=True: build Model from distributed_training as-is, replace GNN layer(s)
    with tracer-capture wrappers, run full model forward once, dissect traces from wrappers.
    """
    config = load_config(config_path)
    device = config["system"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    if distributed:
        print(f"Initializing model from {config_path} (Model from distributed_training, as-is)...")
        model = Model(config)
        model = model.to(device)
        wrappers_and_stores = _replace_gnn_layers_with_tracer_capture(
            model, config, device, distributed
        )
        node_store = wrappers_and_stores[0][1] if wrappers_and_stores else None
        print(f"Model initialized on {device}; {len(wrappers_and_stores)} GNN layer(s) wrapped for tracing.")

        if input_tensor is None:
            input_tensor = create_sample_input(config, device)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            print(f"Created sample input: shape {input_tensor.shape}")
        else:
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            print(f"Using provided input: shape {input_tensor.shape}")

        print("Running forward pass on entire model (traces captured per GNN layer)...")
        output = model(input_tensor)
        traces = [w.get_trace() for w, _ in wrappers_and_stores]
        if not traces:
            raise RuntimeError("No GNN layers found in model for tracing.")
        trace_data = traces[0]
        node_store = wrappers_and_stores[0][1]
        if len(traces) > 1:
            layers_data = [(t, ns) for t, (_, ns) in zip(traces, wrappers_and_stores)]
            print(f"Forward pass completed! Output shape: {output.shape}; {len(traces)} layer(s) traced.")
        else:
            layers_data = None
            print("Forward pass completed!")
            print(f"  Output shape: {output.shape}")
            print(f"  Number of iterations traced: {len(trace_data['iterations'])}")
    else:
        print(f"Initializing model from {config_path}...")
        model, node_store, config = _build_model_for_config(config_path, device, distributed)
        print(f"Model initialized on {device}")

        if input_tensor is None:
            input_tensor = create_sample_input(config, device)
            print(f"Created sample input: shape {input_tensor.shape}")
        else:
            print(f"Using provided input: shape {input_tensor.shape}")

        print("Running forward pass with tracing...")
        trace_data, output = run_one_gnn_layer_with_tracing(model, input_tensor, device)
        layers_data = None
        print("Forward pass completed!")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of iterations traced: {len(trace_data['iterations'])}")

    # Save trace if requested (single layer only)
    if save_trace:
        import json
        # Convert tensors to lists for JSON serialization
        trace_json = {
            'iterations': []
        }
        for iter_data in trace_data['iterations']:
            iter_json = {
                'iteration': iter_data['iteration'],
                'input_injected': iter_data['input_injected'],
                'active_nodes': iter_data['active_nodes'],
                'radiation_targets': {str(k): v for k, v in iter_data['radiation_targets'].items()},
                'direct_connections': {str(k): v for k, v in iter_data.get('direct_connections', {}).items()},
                'node_details': {}
            }
            for node_id, detail in iter_data['node_details'].items():
                iter_json['node_details'][str(node_id)] = {
                    'inputs': detail['inputs'],
                    'input_types': detail['input_types'],
                    'phase_activation': _tensor_to_list(detail['phase_activation']),
                    'mag_activation': _tensor_to_list(detail['mag_activation']),
                    'activation_strength': float(detail['activation_strength'])
                }
            trace_json['iterations'].append(iter_json)
        
        with open(save_trace, 'w') as f:
            json.dump(trace_json, f, indent=2)
        print(f"Trace data saved to {save_trace}")

    if layers_data is not None:
        return None, None, config, layers_data
    return trace_data, node_store, config, None


def _tensor_to_list(value):
    """Convert tensor/array to list."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif hasattr(value, 'tolist'):
        return value.tolist()
    elif isinstance(value, (list, tuple)):
        return list(value)
    else:
        return [float(value)]


def main():
    parser = argparse.ArgumentParser(description="Visualize forward pass with web interface")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML (same config used for all GNN layers when num-gnn-layers > 1)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to raw input tensor file (optional; shape (raw_dim,) or (1, raw_dim)). Omit for random input.",
    )
    parser.add_argument(
        "--save-trace", type=str, help="Path to save trace data as JSON (optional; single layer only)"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use Model from distributed_training (same config); run forward with tracer.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for web server (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for web server (default: 0.0.0.0)")

    args = parser.parse_args()

    input_tensor = None
    if args.input:
        input_tensor = torch.load(args.input)
        print(f"Loaded input from {args.input}")

    try:
        trace_data, node_store, config, layers_data = run_forward_pass_with_tracing(
            args.config,
            input_tensor=input_tensor,
            save_trace=args.save_trace,
            distributed=args.distributed,
        )
    except Exception as e:
        print(f"Error during forward pass: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Create visualizer
    print("Creating visualizer...")
    if layers_data is not None:
        visualizer = TraceVisualizer(layers=layers_data, config=config)
        print(f"  {len(layers_data)} layer(s); use Layer selector in the web UI.")
    else:
        visualizer = TraceVisualizer(trace_data, node_store, config)

    init_with_visualizer(visualizer)
    
    # Start server
    print(f"\n{'='*60}")
    print(f"Visualization server starting...")
    print(f"  Open your browser to: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
