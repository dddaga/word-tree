#!/usr/bin/env python3
"""
Test script to verify the new DAG connectivity structure.
Checks that input nodes have outgoing connections and signal propagation is possible.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.graph import build_static_graph
import pandas as pd
import numpy as np

def test_dag_connectivity():
    """Test the new DAG structure with production parameters."""
    print("🧪 Testing DAG Connectivity Structure")
    print("=" * 50)
    
    # Use production parameters from config
    graph_params = {
        'total_nodes': 1000,
        'num_input_nodes': 200,
        'num_output_nodes': 10,
        'vector_dim': 5,
        'phase_bins': 32,
        'mag_bins': 512,
        'cardinality': 6,
        'seed': 42
    }
    
    print(f"Building graph with {graph_params['total_nodes']} nodes...")
    print(f"Input nodes: {graph_params['num_input_nodes']}")
    print(f"Output nodes: {graph_params['num_output_nodes']}")
    print(f"Intermediate nodes: {graph_params['total_nodes'] - graph_params['num_input_nodes'] - graph_params['num_output_nodes']}")
    print(f"Target cardinality: {graph_params['cardinality']}")
    print()
    
    # Build the graph
    df = build_static_graph(**graph_params)
    
    # Analyze connectivity
    print("📊 Connectivity Analysis")
    print("-" * 30)
    
    # Check input nodes
    input_nodes = df[df['is_input'] == True]
    print(f"Input nodes: {len(input_nodes)}")
    
    input_connections = []
    for _, row in input_nodes.head(10).iterrows():  # Check first 10 input nodes
        connections = row['input_connections']
        input_connections.append(len(connections))
        print(f"Node {row['node_id']}: {len(connections)} connections {connections}")
    
    print(f"Average input node connections: {np.mean(input_connections):.2f}")
    print()
    
    # Check intermediate nodes
    intermediate_nodes = df[(df['is_input'] == False) & (df['is_output'] == False)]
    print(f"Intermediate nodes: {len(intermediate_nodes)}")
    
    intermediate_connections = []
    for _, row in intermediate_nodes.head(10).iterrows():  # Check first 10 intermediate nodes
        connections = row['input_connections']
        intermediate_connections.append(len(connections))
        print(f"Node {row['node_id']}: {len(connections)} connections {connections}")
    
    print(f"Average intermediate node connections: {np.mean(intermediate_connections):.2f}")
    print()
    
    # Check output nodes
    output_nodes = df[df['is_output'] == True]
    print(f"Output nodes: {len(output_nodes)}")
    
    output_connections = []
    for _, row in output_nodes.iterrows():
        connections = row['input_connections']
        output_connections.append(len(connections))
        print(f"Node {row['node_id']}: {len(connections)} connections {connections}")
    
    print(f"Average output node connections: {np.mean(output_connections):.2f}")
    print()
    
    # Overall statistics
    all_connections = []
    for _, row in df.iterrows():
        all_connections.append(len(row['input_connections']))
    
    print("📈 Overall Statistics")
    print("-" * 20)
    print(f"Total nodes: {len(df)}")
    print(f"Average connections per node: {np.mean(all_connections):.2f}")
    print(f"Min connections: {np.min(all_connections)}")
    print(f"Max connections: {np.max(all_connections)}")
    print(f"Nodes with zero connections: {sum(1 for c in all_connections if c == 0)}")
    print()
    
    # Verify DAG property
    print("🔍 DAG Property Verification")
    print("-" * 30)
    
    dag_violations = 0
    for _, row in df.iterrows():
        node_idx = int(row['node_id'][1:])
        for connection in row['input_connections']:
            conn_idx = int(connection[1:])
            if conn_idx >= node_idx:
                print(f"❌ DAG violation: {row['node_id']} connects to {connection}")
                dag_violations += 1
    
    if dag_violations == 0:
        print("✅ DAG property maintained - no backward connections found")
    else:
        print(f"❌ Found {dag_violations} DAG violations")
    
    print()
    
    # Check connectivity paths
    print("🛤️  Connectivity Path Analysis")
    print("-" * 35)
    
    # Build reverse adjacency for path checking (outgoing connections)
    # Since input_connections are INCOMING, we need to reverse the graph
    outgoing_adjacency = {}
    for _, row in df.iterrows():
        node_id = row['node_id']
        outgoing_adjacency[node_id] = []
    
    # Build outgoing connections by reversing incoming connections
    for _, row in df.iterrows():
        target_node = row['node_id']
        for source_node in row['input_connections']:
            if source_node not in outgoing_adjacency:
                outgoing_adjacency[source_node] = []
            outgoing_adjacency[source_node].append(target_node)
    
    # Check if inputs can reach outputs (simple BFS)
    def can_reach_output(start_node, max_depth=10):
        visited = set()
        queue = [(start_node, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            if depth > max_depth:
                continue
                
            if node in visited:
                continue
            visited.add(node)
            
            # Check if this node is an output
            node_row = df[df['node_id'] == node].iloc[0]
            if node_row['is_output']:
                return True, depth
            
            # Add connected nodes to queue (outgoing connections)
            for connected_node in outgoing_adjacency.get(node, []):
                if connected_node not in visited:
                    queue.append((connected_node, depth + 1))
        
        return False, -1
    
    # Test connectivity from first few input nodes
    reachable_count = 0
    total_tested = min(10, len(input_nodes))
    
    for _, row in input_nodes.head(total_tested).iterrows():
        can_reach, depth = can_reach_output(row['node_id'])
        if can_reach:
            reachable_count += 1
            print(f"✅ {row['node_id']} can reach output (depth: {depth})")
        else:
            print(f"❌ {row['node_id']} cannot reach any output")
    
    print(f"\nConnectivity Summary: {reachable_count}/{total_tested} input nodes can reach outputs")
    
    if reachable_count == total_tested:
        print("🎉 SUCCESS: All tested input nodes have paths to outputs!")
    else:
        print("⚠️  WARNING: Some input nodes cannot reach outputs")
    
    return df

if __name__ == "__main__":
    df = test_dag_connectivity()
    print("\n" + "=" * 50)
    print("DAG connectivity test completed!")
