"""
Visualization utilities for converting trace data to visualization format.
"""

from typing import Dict, List, Set, Optional
import torch
import numpy as np
from core.nodestore import NodeStore


def get_node_role(node_id: int, node_store: NodeStore) -> str:
    """
    Determine if a node is input, output, or middle based on node_store.
    
    Args:
        node_id: Node ID to check
        node_store: NodeStore instance with input/output node information
        
    Returns:
        'input', 'output', or 'middle'
    """
    node_id_int = int(node_id)
    
    if node_id_int in node_store.input_nodeids:
        return 'input'
    elif node_id_int in node_store.output_nodeids:
        return 'output'
    else:
        return 'middle'


def get_static_edges(node_store: NodeStore, node_ids: Optional[Set[int]] = None) -> List[Dict]:
    """
    Get all static edges (direct connections) from node_store.
    
    Args:
        node_store: NodeStore instance
        node_ids: Optional set of node IDs to filter edges (if None, get all)
        
    Returns:
        List of edge dicts with 'source', 'target', 'weight'
    """
    edges = []
    
    # Get all nodes or filtered set
    if node_ids is None:
        # Get all nodes from store
        all_node_ids = set(range(node_store.total_nodes))
    else:
        all_node_ids = node_ids
    
    # Fetch nodes to get their outgoing connections
    node_list = list(all_node_ids)
    if not node_list:
        return edges
    
    try:
        nodes = node_store.get_node(node_list)
        
        for node in nodes:
            source_id = node.id
            outgoing = node.payload.get('outgoing_connections', [])
            
            for target_id in outgoing:
                target_id_int = int(target_id)
                if node_ids is None or target_id_int in node_ids:
                    # Get phase weight for edge weight (average of vector)
                    phase_vector = node.vector.get('phase', [])
                    if isinstance(phase_vector, (list, np.ndarray, torch.Tensor)):
                        if isinstance(phase_vector, torch.Tensor):
                            weight = float(torch.mean(phase_vector).item())
                        elif isinstance(phase_vector, np.ndarray):
                            weight = float(np.mean(phase_vector))
                        else:
                            weight = float(np.mean(phase_vector)) if phase_vector else 0.0
                    else:
                        weight = 0.0
                    
                    edges.append({
                        'source': str(source_id),
                        'target': str(target_id_int),
                        'weight': weight
                    })
    except Exception as e:
        # If we can't fetch nodes, return empty list
        print(f"Warning: Could not fetch nodes for edges: {e}")
        return edges
    
    return edges


def extract_node_weights(node_store: NodeStore, node_id: int) -> Dict:
    """
    Extract phase and magnitude weights for a node.
    
    Args:
        node_store: NodeStore instance
        node_id: Node ID to extract weights for
        
    Returns:
        Dict with 'phase_weight' and 'mag_weight' (as lists)
    """
    try:
        nodes = node_store.get_node([node_id])
        if not nodes:
            return {'phase_weight': [], 'mag_weight': []}
        
        node = nodes[0]
        phase_vector = node.vector.get('phase', [])
        mag_vector = node.vector.get('mag', [])
        
        # Convert to lists if needed
        if isinstance(phase_vector, torch.Tensor):
            phase_weight = phase_vector.cpu().tolist()
        elif isinstance(phase_vector, np.ndarray):
            phase_weight = phase_vector.tolist()
        else:
            phase_weight = list(phase_vector) if phase_vector else []
        
        if isinstance(mag_vector, torch.Tensor):
            mag_weight = mag_vector.cpu().tolist()
        elif isinstance(mag_vector, np.ndarray):
            mag_weight = mag_vector.tolist()
        else:
            mag_weight = list(mag_vector) if mag_vector else []
        
        return {
            'phase_weight': phase_weight,
            'mag_weight': mag_weight
        }
    except Exception as e:
        print(f"Warning: Could not extract weights for node {node_id}: {e}")
        return {'phase_weight': [], 'mag_weight': []}


def compute_loss_from_trace(trace_data: Dict, iteration_idx: int, target_values: Optional[Dict] = None) -> float:
    """
    Compute loss from trace data (deprecated - always returns 0.0).
    
    Args:
        trace_data: Trace data from ForwardPassTracer
        iteration_idx: Index of iteration to compute loss for
        target_values: Optional dict of {node_id: target_value} for output nodes (ignored)
        
    Returns:
        Loss value (always 0.0 - loss computation removed with target setting feature)
    """
    return 0.0  # Loss computation removed with target setting feature
