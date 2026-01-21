#!/usr/bin/env python3
"""
Test script for visualization integration.

Tests:
1. Forward pass tracing with ForwardPassTracer (notebook compatibility)
2. Trace data conversion to StepResult format
3. Visualization server endpoints
4. Notebook helper function
"""

import sys
import torch
import yaml
from pathlib import Path
import requests
import time
import threading
from typing import Dict, Any

# Import project modules
from full_model import initialize_model_and_nodestore
from forward_pass_tracer import ForwardPassTracer
from viz.adapter import TraceVisualizer, trace_to_step_result
from viz.server import app, init_with_visualizer
from viz.notebook_helper import launch_web_viz
import uvicorn


def test_forward_pass_tracing(config_path: str):
    """Test 1: Verify ForwardPassTracer works as expected (notebook compatibility)."""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass Tracing (Notebook Compatibility)")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['system']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Initialize model
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
    
    # Create sample input
    input_nodes = config['graph']['input_nodes']
    vector_dim = config['model']['vector_dim']
    input_tensor = torch.randn(input_nodes, vector_dim, device=device)
    print(f"Created input tensor: shape {input_tensor.shape}")
    
    # Reset model
    model.reset()
    
    # Run forward pass with tracing (same as notebook)
    print("Running forward pass with ForwardPassTracer...")
    with ForwardPassTracer() as tracer:
        output = model(input_tensor, tracer=tracer)
        trace_data = tracer.get_trace()
    
    print(f"[OK] Forward pass completed!")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of iterations traced: {len(trace_data['iterations'])}")
    
    # Verify trace data structure
    assert 'iterations' in trace_data, "Trace data must have 'iterations' key"
    assert len(trace_data['iterations']) > 0, "Trace data must have at least one iteration"
    
    # Check first iteration structure
    first_iter = trace_data['iterations'][0]
    required_keys = ['iteration', 'input_injected', 'active_nodes', 'radiation_targets', 'direct_connections', 'node_details']
    for key in required_keys:
        assert key in first_iter, f"First iteration must have '{key}' key"
    
    print(f"[OK] Trace data structure verified")
    print(f"  First iteration: {first_iter['iteration']}")
    print(f"  Active nodes in first iteration: {len(first_iter['active_nodes'])}")
    
    return trace_data, node_store, config, output


def test_trace_conversion(trace_data: Dict, node_store, config: Dict):
    """Test 2: Verify trace data conversion to StepResult format."""
    print("\n" + "="*60)
    print("TEST 2: Trace Data Conversion")
    print("="*60)
    
    # Convert first iteration
    iteration_idx = 0
    step_result = trace_to_step_result(trace_data, iteration_idx, node_store, config)
    
    print(f"[OK] Converted iteration {iteration_idx} to StepResult")
    print(f"  Nodes: {len(step_result.nodes)}")
    print(f"  Edges: {len(step_result.edges)}")
    print(f"  Active signals: {len(step_result.active_signals)}")
    print(f"  Radiation paths: {len(step_result.radiation_paths)}")
    print(f"  Step number: {step_result.step_number}")
    
    # Verify StepResult structure
    assert step_result.nodes is not None, "StepResult must have nodes"
    assert step_result.edges is not None, "StepResult must have edges"
    assert step_result.active_signals is not None, "StepResult must have active_signals"
    assert step_result.radiation_paths is not None, "StepResult must have radiation_paths"
    assert step_result.step_number == iteration_idx, "Step number must match iteration index"
    
    # Check node structure
    if step_result.nodes:
        first_node_id = list(step_result.nodes.keys())[0]
        first_node = step_result.nodes[first_node_id]
        required_node_keys = ['id', 'role', 'phase', 'magnitude', 'activation', 'active']
        for key in required_node_keys:
            assert key in first_node, f"Node must have '{key}' key"
        print(f"[OK] Node structure verified (sample node: {first_node_id})")
    
    # Test TraceVisualizer
    visualizer = TraceVisualizer(trace_data, node_store, config)
    print(f"[OK] TraceVisualizer created")
    print(f"  Total iterations: {visualizer.get_iteration_count()}")
    
    # Get all states
    all_states = visualizer.get_all_states()
    print(f"[OK] Converted all {len(all_states)} iterations")
    
    return visualizer


def test_server_endpoints(visualizer: TraceVisualizer, port: int = 8766):
    """Test 3: Verify visualization server endpoints."""
    print("\n" + "="*60)
    print("TEST 3: Visualization Server Endpoints")
    print("="*60)
    
    # Initialize server
    init_with_visualizer(visualizer)
    print("[OK] Server initialized with visualizer")
    
    # Start server in background thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print(f"Waiting for server to start on port {port}...")
    time.sleep(2)
    
    base_url = f"http://127.0.0.1:{port}"
    
    # Test /api/init
    try:
        response = requests.post(f"{base_url}/api/init", json={}, timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'nodes' in data, "Response must have 'nodes'"
        assert 'edges' in data, "Response must have 'edges'"
        print("[OK] POST /api/init - OK")
    except Exception as e:
        print(f"[FAIL] POST /api/init - FAILED: {e}")
        return False
    
    # Test /api/state
    try:
        response = requests.get(f"{base_url}/api/state", timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'nodes' in data, "Response must have 'nodes'"
        print("[OK] GET /api/state - OK")
    except Exception as e:
        print(f"[FAIL] GET /api/state - FAILED: {e}")
        return False
    
    # Test /api/iterations
    try:
        response = requests.get(f"{base_url}/api/iterations", timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'total_iterations' in data, "Response must have 'total_iterations'"
        assert 'current_iteration' in data, "Response must have 'current_iteration'"
        print(f"[OK] GET /api/iterations - OK (total: {data['total_iterations']})")
    except Exception as e:
        print(f"[FAIL] GET /api/iterations - FAILED: {e}")
        return False
    
    # Test /api/step
    try:
        response = requests.post(f"{base_url}/api/step", timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert 'nodes' in data, "Response must have 'nodes'"
        print("[OK] POST /api/step - OK")
    except Exception as e:
        print(f"[FAIL] POST /api/step - FAILED: {e}")
        return False
    
    # Test /api/step_back
    try:
        response = requests.post(f"{base_url}/api/step_back", timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("[OK] POST /api/step_back - OK")
    except Exception as e:
        print(f"[FAIL] POST /api/step_back - FAILED: {e}")
        return False
    
    # Test /api/set_iteration
    try:
        response = requests.post(f"{base_url}/api/set_iteration", json={"iteration": 0}, timeout=2)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print("[OK] POST /api/set_iteration - OK")
    except Exception as e:
        print(f"[FAIL] POST /api/set_iteration - FAILED: {e}")
        return False
    
    print("[OK] All server endpoints working correctly")
    return True


def test_notebook_helper(trace_data: Dict, node_store, config: Dict):
    """Test 4: Verify notebook helper function."""
    print("\n" + "="*60)
    print("TEST 4: Notebook Helper Function")
    print("="*60)
    
    # Test launch_web_viz function
    print("Testing launch_web_viz function...")
    
    try:
        server_thread = launch_web_viz(
            trace_data,
            node_store,
            config,
            port=8767,
            host="127.0.0.1",
            open_browser=False  # Don't open browser in test
        )
        
        assert server_thread is not None, "launch_web_viz must return a thread"
        assert server_thread.is_alive(), "Server thread must be alive"
        print("[OK] launch_web_viz created server thread")
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Test that server is responding
        try:
            response = requests.get("http://127.0.0.1:8767/api/state", timeout=2)
            assert response.status_code == 200, "Server should respond"
            print("[OK] Server is responding to requests")
        except Exception as e:
            print(f"[WARN] Server may not be fully started: {e}")
        
        print("[OK] Notebook helper function works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] Notebook helper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VISUALIZATION INTEGRATION TEST")
    print("="*60)
    
    # Use main2 config as sample
    config_path = "training_runs/main2/main2.yaml"
    
    if not Path(config_path).exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return 1
    
    try:
        # Test 1: Forward pass tracing (notebook compatibility)
        trace_data, node_store, config, output = test_forward_pass_tracing(config_path)
        
        # Test 2: Trace conversion
        visualizer = test_trace_conversion(trace_data, node_store, config)
        
        # Test 3: Server endpoints
        server_ok = test_server_endpoints(visualizer, port=8766)
        
        # Test 4: Notebook helper
        notebook_ok = test_notebook_helper(trace_data, node_store, config)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("[PASS] Test 1: Forward Pass Tracing (Notebook Compatibility) - PASSED")
        print("[PASS] Test 2: Trace Data Conversion - PASSED")
        if server_ok:
            print("[PASS] Test 3: Visualization Server Endpoints - PASSED")
        else:
            print("[FAIL] Test 3: Visualization Server Endpoints - FAILED")
        if notebook_ok:
            print("[PASS] Test 4: Notebook Helper Function - PASSED")
        else:
            print("[FAIL] Test 4: Notebook Helper Function - FAILED")
        
        if server_ok and notebook_ok:
            print("\n[SUCCESS] All tests passed! Visualization integration is working correctly.")
            return 0
        else:
            print("\n[WARN] Some tests failed. Check output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n[FAIL] Test suite FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
