#!/usr/bin/env python3
"""
Visualize forward pass using ForwardPassTracer and web interface.

Usage:
    python visualize_forward_pass.py --config configs/main_config3.yaml
    python visualize_forward_pass.py --config configs/main_config3.yaml --input <input_file>
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

from core import initialize_model_and_nodestore
from forward_pass_tracer import ForwardPassTracer
from viz.adapter import TraceVisualizer
from viz.server import app, init_with_visualizer
import uvicorn


def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_sample_input(config, device):
    """Create a sample input tensor."""
    from torch.utils.data import TensorDataset
    from sklearn.datasets import load_iris
    
    input_nodes = config['graph']['input_nodes']
    vector_dim = config['model']['vector_dim']
    
    # Create random input
    iris_data = load_iris()
    X = iris_data.data  # Features: (150, 4) - sepal length, sepal width, petal length, petal width
    y = iris_data.target  # Labels: (150,) - 0, 1, 2 for setosa, versicolor, virginica

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 4)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create PyTorch Dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    sample_input = X_tensor[0]
    sample_output = y_tensor[0]
    return sample_input


def run_forward_pass_with_tracing(config_path, input_tensor=None, save_trace=None):
    """
    Run a forward pass with tracing and return trace data.
    
    Args:
        config_path: Path to config YAML file
        input_tensor: Optional input tensor (if None, creates random input)
        save_trace: Optional path to save trace data as JSON
        
    Returns:
        tuple: (trace_data, node_store, config)
    """
    config = load_config(config_path)
    
    device = config['system']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Initializing model from {config_path}...")
    model, node_store = initialize_model_and_nodestore(
        qdrant_url=config['qdrant']['url'],
        collection_name=config['qdrant']['collection_name'],
        total_nodes=config['graph']['total_nodes'],
        input_nodes=config['graph']['input_nodes'],
        output_nodes=config['graph']['output_nodes'],
        cardinality=config['graph']['cardinality'],
        radiation_targets=config['graph']['radiation_targets'],
        vector_dim=config['model']['vector_dim'],
        phase_bins=config['model']['phase_bins'],
        mag_bins=config['model']['mag_bins'],
        iterations=config['model']['iterations'],
        activation_threshold=config['model']['activation_threshold'],
        gamma=config['model']['gamma'],
        device=device,
        temporal_decay=config['model'].get('temporal_decay', 1.0),
        radiation_similarity_threshold=config['model'].get('radiation_similarity_threshold', 0.0),
    )
    
    model = model.to(device)
    print(f"Model initialized on {device}")
    
    # Create input if not provided
    if input_tensor is None:
        input_tensor = create_sample_input(config, device)
        print(f"Created random input: shape {input_tensor.shape}")
    else:
        print(f"Using provided input: shape {input_tensor.shape}")
    
    # Reset model
    model.reset()
    
    # Run forward pass with tracing
    print("Running forward pass with tracing...")
    with ForwardPassTracer() as tracer:
        output = model(input_tensor, tracer=tracer)
        trace_data = tracer.get_trace()
    
    print(f"Forward pass completed!")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of iterations traced: {len(trace_data['iterations'])}")
    
    # Save trace if requested
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
    
    return trace_data, node_store, config


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
    parser = argparse.ArgumentParser(description='Visualize forward pass with web interface')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--input', type=str, help='Path to input tensor file (optional)')
    parser.add_argument('--save-trace', type=str, help='Path to save trace data as JSON (optional)')
    parser.add_argument('--port', type=int, default=8765, help='Port for web server (default: 8765)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for web server (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Load input if provided
    input_tensor = None
    if args.input:
        input_tensor = torch.load(args.input)
        print(f"Loaded input from {args.input}")
    
    # Run forward pass with tracing
    try:
        trace_data, node_store, config = run_forward_pass_with_tracing(
            args.config,
            input_tensor=input_tensor,
            save_trace=args.save_trace
        )
    except Exception as e:
        print(f"Error during forward pass: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Create visualizer
    print("Creating visualizer...")
    visualizer = TraceVisualizer(trace_data, node_store, config)
    
    # Initialize server with visualizer
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
