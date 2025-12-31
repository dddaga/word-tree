from lookup_table import LookupTable
import torch
import torch.nn as nn
from typing import Dict
from nodestore import NodeStore


class GradientAccumulator(nn.Module):
    """
    Gradient accumulator specifically for GNN parameters only
    """
    def __init__(self,  node_store:NodeStore, lr:float, verbose:bool=False,
    device:str='cuda' if torch.cuda.is_available() else 'cpu', accumulation_steps:int=None):

        super().__init__()

        self.device = device
        self.phase_bins = node_store.phase_bins
        self.mag_bins = node_store.mag_bins
        self.total_nodes = node_store.total_nodes

        #instead of accumulation steps, we'll update when the sum of gradients is enough for a change to value
        # self.accumulation_steps = accumulation_steps 
        self.node_store = node_store
        self.lr = lr
        self.verbose = verbose

        # Use sparse dictionaries - only store gradients for active nodes
        # This prevents memory waste for large graphs with sparse activations
        self.phase_grads = {}
        self.phase_grad_counts = {}
        self.mag_grads = {}
        self.mag_grad_counts = {}


    def receive_gradients(self, phase_grads:Dict[int, torch.Tensor], mag_grads:Dict[int, torch.Tensor]):
        """
        This receives the grads from other workers who are updating the graph. 
        The gradients are expected as Dict[int, torch.Tensor] with the keys as the node IDs.
        
        Gradients are accumulated in internal dictionaries until they meet the update threshold.
        """
        for node_id, phase_grad in phase_grads.items():
            if phase_grad is None or torch.allclose(phase_grad, torch.zeros_like(phase_grad), atol=1e-8):
                continue 
            
            # Safety check: reject inf/nan gradients
            if torch.isinf(phase_grad).any() or torch.isnan(phase_grad).any():
                print(f"Warning: Invalid phase gradient for node {node_id}, skipping")
                continue
            
            # Sparse storage: only create entry if gradient exists
            # Apply learning rate during accumulation for efficient threshold checking
            if node_id not in self.phase_grads:
                self.phase_grads[node_id] = self.lr * phase_grad.to(self.device)
                self.phase_grad_counts[node_id] = 1
            else:
                self.phase_grads[node_id] += self.lr * phase_grad.to(self.device)
                self.phase_grad_counts[node_id] += 1

        for node_id, mag_grad in mag_grads.items():
            if mag_grad is None or torch.allclose(mag_grad, torch.zeros_like(mag_grad), atol=1e-8):
                continue 
            
            # Safety check: reject inf/nan gradients
            if torch.isinf(mag_grad).any() or torch.isnan(mag_grad).any():
                print(f"Warning: Invalid mag gradient for node {node_id}, skipping")
                continue
            
            # Sparse storage: only create entry if gradient exists
            # Apply learning rate during accumulation for efficient threshold checking
            if node_id not in self.mag_grads:
                self.mag_grads[node_id] = self.lr * mag_grad.to(self.device)
                self.mag_grad_counts[node_id] = 1
            else:
                self.mag_grads[node_id] += self.lr * mag_grad.to(self.device)
                self.mag_grad_counts[node_id] += 1

    def step(self):
        """
        Apply gradient updates only when the accumulated gradient exceeds the threshold (0.8).
        Round gradients to nearest integer before applying updates.
        Increment version numbers for updated nodes.
        """
        node_ids_to_update = set()
        
        # Check which nodes meet the update threshold (max(abs(grad)) > 0.8)
        # Sparse storage: only iterate over nodes with accumulated gradients
        for node_id, phase_grad in self.phase_grads.items():
            if torch.max(torch.abs(phase_grad)) > 0.8:
                node_ids_to_update.add(node_id)
        
        for node_id, mag_grad in self.mag_grads.items():
            if torch.max(torch.abs(mag_grad)) > 0.8:
                node_ids_to_update.add(node_id)

        if not node_ids_to_update:
            return  # No nodes to update
        
        node_ids_to_update = list(node_ids_to_update)
        
        # Fetch current node values and versions
        nodes_to_update = self.node_store.get_node(node_ids_to_update)
        old_phase_values = {node.id: torch.tensor(node.vector['phase'], dtype=torch.float16, device=self.device) for node in nodes_to_update}
        old_mag_values = {node.id: torch.tensor(node.vector['mag'], dtype=torch.float16, device=self.device) for node in nodes_to_update}
        old_versions = {node.id: node.payload.get('version', 0) for node in nodes_to_update}

        new_phase_values = {}
        new_mag_values = {}
        
        for node_id in node_ids_to_update:
            # Get phase gradient and round it (sparse storage: check if key exists)
            if node_id in self.phase_grads and torch.max(torch.abs(self.phase_grads[node_id])) > 0.8:
                phase_grad = torch.round(self.phase_grads[node_id])  # Round to nearest integer
                del self.phase_grads[node_id]  # Remove from sparse dict after use
                if node_id in self.phase_grad_counts:
                    del self.phase_grad_counts[node_id]
            else:
                phase_grad = torch.tensor(0, dtype=torch.float16, device=self.device)

            # Get magnitude gradient and round it (sparse storage: check if key exists)
            if node_id in self.mag_grads and torch.max(torch.abs(self.mag_grads[node_id])) > 0.8:
                mag_grad = torch.round(self.mag_grads[node_id])  # Round to nearest integer
                del self.mag_grads[node_id]  # Remove from sparse dict after use
                if node_id in self.mag_grad_counts:
                    del self.mag_grad_counts[node_id]
            else:
                mag_grad = torch.tensor(0, dtype=torch.float16, device=self.device)

            # Apply gradient descent with rounded gradients
            # Note: lr already applied during accumulation, so just subtract the scaled gradient
            phase_vector = old_phase_values[node_id]
            mag_vector = old_mag_values[node_id]
            new_phase_values[node_id] = (phase_vector.to(phase_grad.dtype) - phase_grad).round().long() % self.phase_bins
            new_mag_values[node_id] = (mag_vector.to(mag_grad.dtype) - mag_grad).round().long() % self.mag_bins

        # Prepare final values for update
        final_values = {
            node_id: {
                'phase': new_phase_values[node_id].tolist(), 
                'mag': new_mag_values[node_id].tolist()
            } 
            for node_id in node_ids_to_update
        }

        if final_values:
            try:
                # Update vectors in DB
                self.node_store.update_vectors(final_values)
                
                # Increment versions for updated nodes
                new_versions = [old_versions[node_id] + 1 for node_id in node_ids_to_update]
                self.node_store.update_node_versions(node_ids_to_update, new_versions)
                
                if self.verbose:
                    print(f"Updated {len(node_ids_to_update)} nodes: {node_ids_to_update}")
                    #print(f"Version increments: {dict(zip(node_ids_to_update, new_versions))}")
            except Exception as e:
                print(f"Error updating vectors: {e}")
                print(f"Final values: {final_values}")

