import pandas as pd
import numpy as np
import os

def build_static_graph(
    total_nodes,
    num_input_nodes,
    num_output_nodes,
    vector_dim,
    phase_bins,
    mag_bins,
    cardinality,
    seed=42
) -> pd.DataFrame:
    """
    Builds a static DAG graph with proper layered connectivity ensuring signal propagation.
    Creates a layered architecture: Input → Intermediate → Output with forward-only connections.
    """
    np.random.seed(seed)

    node_ids = [f"n{i}" for i in range(total_nodes)]
    input_nodes = node_ids[:num_input_nodes]
    output_nodes = node_ids[-num_output_nodes:]
    intermediate_nodes = node_ids[num_input_nodes:-num_output_nodes]

    # Calculate layer structure for proper DAG topology
    num_intermediate = len(intermediate_nodes)
    if num_intermediate > 0:
        # Create multiple intermediate layers for better signal flow
        layer_size = max(1, num_intermediate // 4)  # Aim for ~4 layers
        num_layers = max(1, num_intermediate // layer_size)
    else:
        num_layers = 0

    graph_data = []

    for i, node_id in enumerate(node_ids):
        is_input = node_id in input_nodes
        is_output = node_id in output_nodes

        row = {
            "node_id": node_id,
            "input_connections": [],
            "is_input": is_input,
            "is_output": is_output
        }

        graph_data.append(row)

    df = pd.DataFrame(graph_data)

    # Assign DAG connections with proper layered structure
    # Note: input_connections represents INCOMING connections TO each node
    for idx, row in df.iterrows():
        node_id = row["node_id"]
        node_index = int(node_id[1:])
        connections = []

        if row["is_input"]:
            # Input nodes have NO incoming connections (they are signal sources)
            connections = []

        elif row["is_output"]:
            # Output nodes receive connections from intermediate nodes and inputs
            candidate_sources = []
            
            # Add intermediate nodes as candidates
            candidate_sources.extend(intermediate_nodes)
            
            # Add input nodes as candidates for direct connections
            candidate_sources.extend(input_nodes)
            
            # Filter to only include nodes with lower indices (DAG constraint)
            candidate_sources = [n for n in candidate_sources if int(n[1:]) < node_index]
            
            if len(candidate_sources) >= cardinality:
                connections = np.random.choice(candidate_sources, size=cardinality, replace=False).tolist()
            else:
                connections = candidate_sources

        else:
            # Intermediate nodes - determine layer and connect appropriately
            intermediate_idx = intermediate_nodes.index(node_id)
            current_layer = intermediate_idx // layer_size if layer_size > 0 else 0
            
            candidate_sources = []
            
            # Always include input nodes as potential sources
            candidate_sources.extend(input_nodes)
            
            # Include intermediate nodes from previous layers
            if current_layer > 0:
                prev_layer_start = 0
                prev_layer_end = current_layer * layer_size
                prev_layer_nodes = intermediate_nodes[prev_layer_start:prev_layer_end]
                candidate_sources.extend(prev_layer_nodes)
            
            # Filter to maintain DAG property (only lower-indexed nodes)
            candidate_sources = [n for n in candidate_sources if int(n[1:]) < node_index]
            
            if len(candidate_sources) >= cardinality:
                connections = np.random.choice(candidate_sources, size=cardinality, replace=False).tolist()
            else:
                connections = candidate_sources

        df.at[idx, 'input_connections'] = connections

    return df

def load_or_build_graph(save_path="config/static_graph.pkl", overwrite=False, **kwargs) -> pd.DataFrame:
    if not overwrite and os.path.exists(save_path):
        print(f" Loaded graph from {save_path}")
        return pd.read_pickle(save_path)

    # Remove save_path from kwargs before passing to build_static_graph
    build_kwargs = {k: v for k, v in kwargs.items() if k != 'save_path'}
    df = build_static_graph(**build_kwargs)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)
    print(f"🆕 Built new graph and saved to {save_path}")
    return df
