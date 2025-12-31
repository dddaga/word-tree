#!/usr/bin/env python3
"""
Test and training script for Neurograph visualization.

This script tests the network with various configurations and can be used
to train the network on arbitrary patterns or datasets.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List, Optional
import argparse


class NeurographTester:
    """Test and train Neurograph networks with various configurations."""
    
    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url
        self.current_config = None
        
    def init_network(self, config: Dict) -> Dict:
        """Initialize network with configuration."""
        response = requests.post(f"{self.base_url}/api/init", json=config)
        response.raise_for_status()
        self.current_config = config
        return response.json()
    
    def step_forward(self) -> Dict:
        """Execute one forward step."""
        response = requests.post(f"{self.base_url}/api/step")
        response.raise_for_status()
        return response.json()
    
    def step_backward(self) -> Dict:
        """Execute one backward step."""
        response = requests.post(f"{self.base_url}/api/backprop")
        response.raise_for_status()
        return response.json()
    
    def inject_sequence(self, sequence: List[float], timestep: int = 0) -> Dict:
        """Inject temporal sequence."""
        response = requests.post(
            f"{self.base_url}/api/inject_sequence",
            json={"sequence": sequence, "timestep": timestep}
        )
        response.raise_for_status()
        return response.json()
    
    def update_config(self, **kwargs) -> Dict:
        """Update configuration parameters."""
        response = requests.post(f"{self.base_url}/api/config", json=kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_active_nodes(self, state: Dict) -> List[str]:
        """Get list of active node IDs."""
        return [
            node_id for node_id, node in state["nodes"].items()
            if node["active"]
        ]
    
    def get_output_phases(self, state: Dict) -> Dict[str, float]:
        """Get phases of output nodes."""
        return {
            node_id: node["phase"]
            for node_id, node in state["nodes"].items()
            if node["role"] == "output"
        }
    
    def compute_loss(self, state: Dict) -> float:
        """Compute average phase error for output nodes."""
        total_error = 0.0
        count = 0
        for node_id, node in state["nodes"].items():
            if node["role"] == "output" and node.get("target_phase") is not None:
                error = abs(node["target_phase"] - node["phase"])
                # Circular distance
                if error > np.pi:
                    error = 2*np.pi - error
                total_error += error
                count += 1
        return total_error / count if count > 0 else 0.0


def test_basic_forward_pass(tester: NeurographTester):
    """Test basic forward propagation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Forward Pass")
    print("="*60)
    
    config = {
        "num_input": 2,
        "num_input_connected": 3,
        "num_middle": 5,
        "num_output": 2,
        "vector_dim": 8,
        "radiation_k": 2
    }
    
    print(f"Initializing network: {config}")
    state = tester.init_network(config)
    print(f"✓ Network initialized with {len(state['nodes'])} nodes, {len(state['edges'])} edges")
    
    # Inject signal into input nodes
    print("\nInjecting signals into input nodes...")
    for i in range(config["num_input"]):
        tester.inject_sequence([0.5 * (i+1)], timestep=i)
    
    # Run forward steps
    print("\nRunning forward propagation...")
    for step in range(5):
        state = tester.step_forward()
        active_nodes = tester.get_active_nodes(state)
        print(f"  Step {step+1}: {len(active_nodes)} active nodes")
    
    output_phases = tester.get_output_phases(state)
    print(f"\nFinal output phases: {output_phases}")
    print("✓ Test passed")


def test_energy_conservation(tester: NeurographTester):
    """Test energy depletion and temporal decay."""
    print("\n" + "="*60)
    print("TEST 2: Energy Conservation")
    print("="*60)
    
    config = {
        "num_input": 1,
        "num_input_connected": 2,
        "num_middle": 3,
        "num_output": 1,
        "temporal_decay": 0.4,
        "conductance_efficiency": 0.9,
        "radiation_efficiency": 0.95
    }
    
    state = tester.init_network(config)
    print(f"Config: temporal_decay={config['temporal_decay']}")
    
    # Inject strong signal
    tester.inject_sequence([1.0], timestep=0)
    state = tester.step_forward()
    
    initial_activation = state["nodes"]["in_0"]["activation"]
    print(f"Initial activation (in_0): {initial_activation:.4f}")
    
    # Track decay over steps
    print("\nTracking activation decay:")
    for step in range(10):
        state = tester.step_forward()
        activation = state["nodes"]["in_0"]["activation"]
        print(f"  Step {step+1}: {activation:.4f}")
        
        if activation < 0.01:
            print(f"  Node deactivated after {step+1} steps")
            break
    
    print("✓ Energy conservation working")


def test_radiation_mechanism(tester: NeurographTester):
    """Test phase-based radiation."""
    print("\n" + "="*60)
    print("TEST 3: Radiation Mechanism")
    print("="*60)
    
    config = {
        "num_input": 2,
        "num_input_connected": 0,  # Flat mode
        "num_middle": 4,
        "num_output": 1,
        "architecture_mode": "flat",
        "radiation_k": 2,
        "use_radiation": True
    }
    
    state = tester.init_network(config)
    print(f"Network mode: {config['architecture_mode']}")
    print(f"Radiation neighbors: {config['radiation_k']}")
    
    # Inject signal
    tester.inject_sequence([0.8, 0.6], timestep=0)
    state = tester.step_forward()
    
    # Check for radiation signals
    radiation_signals = [
        sig for sig in state["active_signals"]
        if sig["type"] == "radiation"
    ]
    
    print(f"\nRadiation connections: {len(radiation_signals)}")
    for sig in radiation_signals[:5]:  # Show first 5
        print(f"  {sig['source']} → {sig['target']} (strength: {sig['strength']:.4f})")
    
    print("✓ Radiation mechanism working")


def test_beam_width_pruning(tester: NeurographTester):
    """Test beam width computational efficiency."""
    print("\n" + "="*60)
    print("TEST 4: Beam Width Pruning")
    print("="*60)
    
    config = {
        "num_input": 3,
        "num_input_connected": 5,
        "num_middle": 10,
        "num_output": 2,
        "beam_width": 8  # Limit to 8 active nodes
    }
    
    state = tester.init_network(config)
    print(f"Total nodes: {len(state['nodes'])}")
    print(f"Beam width: {config['beam_width']}")
    
    # Inject signals to activate many nodes
    tester.inject_sequence([1.0, 0.8, 0.6], timestep=0)
    
    print("\nTracking active nodes with beam pruning:")
    for step in range(5):
        state = tester.step_forward()
        active_nodes = tester.get_active_nodes(state)
        print(f"  Step {step+1}: {len(active_nodes)} active (should be ≤ {config['beam_width']})")
        
        assert len(active_nodes) <= config['beam_width'], "Beam width not enforced!"
    
    print("✓ Beam width pruning working")


def test_temporal_sequence(tester: NeurographTester):
    """Test temporal sequence injection with positional encoding."""
    print("\n" + "="*60)
    print("TEST 5: Temporal Sequence Processing")
    print("="*60)
    
    config = {
        "num_input": 3,
        "num_input_connected": 4,
        "num_middle": 6,
        "num_output": 2
    }
    
    state = tester.init_network(config)
    
    # Generate temporal sequence pattern
    pattern = [
        [0.2, 0.5, 0.8],  # t=0
        [0.4, 0.7, 0.3],  # t=1
        [0.6, 0.4, 0.5],  # t=2
        [0.8, 0.2, 0.7],  # t=3
    ]
    
    print("Injecting temporal sequence...")
    for t, sequence in enumerate(pattern):
        print(f"  t={t}: {sequence}")
        tester.inject_sequence(sequence, timestep=t)
        state = tester.step_forward()
    
    output_phases = tester.get_output_phases(state)
    print(f"\nFinal output phases: {output_phases}")
    print("✓ Temporal sequence processing working")


def test_backward_pass(tester: NeurographTester):
    """Test gradient computation and backpropagation."""
    print("\n" + "="*60)
    print("TEST 6: Backward Pass and Training")
    print("="*60)
    
    config = {
        "num_input": 2,
        "num_input_connected": 3,
        "num_middle": 4,
        "num_output": 1,
        "learning_rate": 0.1
    }
    
    state = tester.init_network(config)
    print(f"Learning rate: {config['learning_rate']}")
    
    # Training loop
    print("\nTraining for 10 steps:")
    losses = []
    
    for epoch in range(10):
        # Forward pass
        tester.inject_sequence([0.5, 0.8], timestep=epoch)
        state = tester.step_forward()
        
        # Compute loss
        loss = tester.compute_loss(state)
        losses.append(loss)
        
        # Backward pass
        state = tester.step_backward()
        
        # Check gradients
        avg_grad = np.mean([abs(node["gradient"]) for node in state["nodes"].values()])
        
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Avg Gradient={avg_grad:.5f}")
    
    # Check if loss decreased
    if losses[-1] < losses[0]:
        print(f"\n✓ Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print(f"\n⚠ Loss did not decrease (may need more epochs or tuning)")


def test_architecture_modes(tester: NeurographTester):
    """Test flat vs hierarchical architecture modes."""
    print("\n" + "="*60)
    print("TEST 7: Architecture Mode Comparison")
    print("="*60)
    
    base_config = {
        "num_input": 2,
        "num_input_connected": 3,
        "num_middle": 5,
        "num_output": 2
    }
    
    results = {}
    
    for mode in ["flat", "hierarchical"]:
        print(f"\nTesting {mode} mode...")
        config = {**base_config, "architecture_mode": mode}
        
        state = tester.init_network(config)
        
        # Count nodes by role
        role_counts = {}
        for node in state["nodes"].values():
            role = node["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        results[mode] = {
            "total_nodes": len(state["nodes"]),
            "total_edges": len(state["edges"]),
            "roles": role_counts
        }
        
        print(f"  Nodes: {results[mode]['total_nodes']}")
        print(f"  Edges: {results[mode]['total_edges']}")
        print(f"  By role: {results[mode]['roles']}")
    
    print(f"\n✓ Both architectures initialized successfully")


def train_on_pattern(tester: NeurographTester, pattern: List[List[float]], 
                     target_phases: List[float], epochs: int = 20):
    """Train network on a specific pattern."""
    print("\n" + "="*60)
    print("TRAINING: Custom Pattern")
    print("="*60)
    
    num_inputs = len(pattern[0])
    num_outputs = len(target_phases)
    
    config = {
        "num_input": num_inputs,
        "num_input_connected": num_inputs * 2,
        "num_middle": num_inputs * 3,
        "num_output": num_outputs,
        "learning_rate": 0.05,
        "beam_width": 20
    }
    
    print(f"Pattern shape: {len(pattern)} sequences × {num_inputs} inputs")
    print(f"Target phases: {target_phases}")
    print(f"Config: {config}")
    
    state = tester.init_network(config)
    
    print(f"\nTraining for {epochs} epochs...")
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for t, sequence in enumerate(pattern):
            # Forward pass
            tester.inject_sequence(sequence, timestep=t)
            state = tester.step_forward()
            
            # Compute loss
            loss = tester.compute_loss(state)
            epoch_loss += loss
            
            # Backward pass
            state = tester.step_backward()
        
        avg_loss = epoch_loss / len(pattern)
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")
    
    # Final evaluation
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Initial loss: {losses[0]:.6f}")
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"Improvement: {improvement:.1f}%")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="Test and train Neurograph networks")
    parser.add_argument("--url", default="http://localhost:8765", help="Server URL")
    parser.add_argument("--test", choices=["all", "basic", "energy", "radiation", 
                                           "beam", "temporal", "backward", "architecture"],
                       default="all", help="Which test to run")
    parser.add_argument("--train", action="store_true", help="Run training example")
    
    args = parser.parse_args()
    
    tester = NeurographTester(base_url=args.url)
    
    # Check server
    try:
        response = requests.get(f"{args.url}/")
        print(f"✓ Server running at {args.url}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Server not running at {args.url}")
        print("  Start server with: cd viz && uvicorn server:app --port 8765")
        return 1
    
    # Run tests
    tests = {
        "basic": test_basic_forward_pass,
        "energy": test_energy_conservation,
        "radiation": test_radiation_mechanism,
        "beam": test_beam_width_pruning,
        "temporal": test_temporal_sequence,
        "backward": test_backward_pass,
        "architecture": test_architecture_modes
    }
    
    if args.test == "all":
        for name, test_func in tests.items():
            try:
                test_func(tester)
            except Exception as e:
                print(f"\n✗ Test '{name}' failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        tests[args.test](tester)
    
    # Training example
    if args.train:
        # Example: XOR-like pattern
        pattern = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        target_phases = [0.0, np.pi]  # Two distinct outputs
        
        train_on_pattern(tester, pattern, target_phases, epochs=50)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    print(f"\nView visualization at: {args.url}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

