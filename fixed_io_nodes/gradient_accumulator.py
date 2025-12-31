import torch
import torch.nn as nn
from typing import Dict
from nodestore import NodeStore


class GradientAccumulator(nn.Module):
    """
    Gradient accumulator for GNN parameters with batch-based updates.
    Accumulates gradients for N samples (batch_size) before updating weights.
    """
    def __init__(self,  node_store:NodeStore, lr:float, batch_size:int=32, 
                 momentum:float=0.9, verbose:bool=False,
                 device:str='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.device = device
        self.total_nodes = node_store.total_nodes
        self.node_store = node_store
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.verbose = verbose

        # Use sparse dictionaries - only store gradients for active nodes
        # This prevents memory waste for large graphs with sparse activations
        self.phase_grads = {}
        self.mag_grads = {}
        
        # Momentum buffers for SGD with momentum
        self.phase_velocities = {}
        self.mag_velocities = {}
        
        # Track accumulation count
        self.accumulation_count = 0


    def receive_gradients(self, phase_grads:Dict[int, torch.Tensor], mag_grads:Dict[int, torch.Tensor]):
        """
        Receive and accumulate gradients from workers.
        Gradients are accumulated until batch_size samples have been processed.
        """
        for node_id, phase_grad in phase_grads.items():
            if phase_grad is None or torch.allclose(phase_grad, torch.zeros_like(phase_grad), atol=1e-8):
                continue 
            
            # Safety check: reject inf/nan gradients
            if torch.isinf(phase_grad).any() or torch.isnan(phase_grad).any():
                print(f"Warning: Invalid phase gradient for node {node_id}, skipping")
                continue
            
            # Accumulate gradients (will average later)
            if node_id not in self.phase_grads:
                self.phase_grads[node_id] = phase_grad.to(self.device).clone()
            else:
                self.phase_grads[node_id] += phase_grad.to(self.device)

        for node_id, mag_grad in mag_grads.items():
            if mag_grad is None or torch.allclose(mag_grad, torch.zeros_like(mag_grad), atol=1e-8):
                continue 
            
            # Safety check: reject inf/nan gradients
            if torch.isinf(mag_grad).any() or torch.isnan(mag_grad).any():
                print(f"Warning: Invalid mag gradient for node {node_id}, skipping")
                continue
            
            # Accumulate gradients (will average later)
            if node_id not in self.mag_grads:
                self.mag_grads[node_id] = mag_grad.to(self.device).clone()
            else:
                self.mag_grads[node_id] += mag_grad.to(self.device)
        
        # Increment accumulation counter
        self.accumulation_count += 1

    def step(self):
        """
        Apply gradient updates when batch_size samples have been accumulated.
        Uses SGD with momentum for stable optimization.
        """
        # Only update when we've accumulated enough gradients
        if self.accumulation_count < self.batch_size:
            return  # Not enough samples yet
        
        if not self.phase_grads and not self.mag_grads:
            # Reset counter even if no gradients (edge case)
            self.accumulation_count = 0
            return
        
        # Get all nodes that have accumulated gradients
        node_ids_to_update = set(self.phase_grads.keys()).union(set(self.mag_grads.keys()))
        node_ids_to_update = list(node_ids_to_update)
        
        if not node_ids_to_update:
            self.accumulation_count = 0
            return
        
        # Fetch current node values and versions
        nodes_to_update = self.node_store.get_node(node_ids_to_update)
        old_phase_values = {node.id: torch.tensor(node.vector['phase'], dtype=torch.float32, device=self.device) for node in nodes_to_update}
        old_mag_values = {node.id: torch.tensor(node.vector['mag'], dtype=torch.float32, device=self.device) for node in nodes_to_update}
        old_versions = {node.id: node.payload.get('version', 0) for node in nodes_to_update}

        new_phase_values = {}
        new_mag_values = {}
        
        for node_id in node_ids_to_update:
            # Compute average gradient over the batch
            if node_id in self.phase_grads:
                avg_phase_grad = self.phase_grads[node_id] / self.accumulation_count
                
                # Apply momentum: velocity = momentum * old_velocity + grad
                if node_id not in self.phase_velocities:
                    self.phase_velocities[node_id] = avg_phase_grad
                else:
                    self.phase_velocities[node_id] = self.momentum * self.phase_velocities[node_id] + avg_phase_grad
                
                # SGD update: weight -= lr * velocity
                phase_vector = old_phase_values[node_id]
                new_phase_values[node_id] = phase_vector - self.lr * self.phase_velocities[node_id]
            else:
                new_phase_values[node_id] = old_phase_values[node_id]

            if node_id in self.mag_grads:
                avg_mag_grad = self.mag_grads[node_id] / self.accumulation_count
                
                # Apply momentum
                if node_id not in self.mag_velocities:
                    self.mag_velocities[node_id] = avg_mag_grad
                else:
                    self.mag_velocities[node_id] = self.momentum * self.mag_velocities[node_id] + avg_mag_grad
                
                # SGD update
                mag_vector = old_mag_values[node_id]
                new_mag_values[node_id] = mag_vector - self.lr * self.mag_velocities[node_id]
            else:
                new_mag_values[node_id] = old_mag_values[node_id]

        # Prepare final values for update (convert to lists for Qdrant)
        final_values = {
            node_id: {
                'phase': new_phase_values[node_id].tolist(), 
                'mag': new_mag_values[node_id].tolist()
            } 
            for node_id in node_ids_to_update
        }

        try:
            # Update vectors in DB
            self.node_store.update_vectors(final_values)
            
            # Increment versions for updated nodes
            new_versions = [old_versions[node_id] + 1 for node_id in node_ids_to_update]
            self.node_store.update_node_versions(node_ids_to_update, new_versions)
            
            if self.verbose:
                avg_phase_grad_norm = torch.mean(torch.stack([torch.norm(self.phase_grads[nid]) for nid in self.phase_grads])) / self.accumulation_count if self.phase_grads else 0
                avg_mag_grad_norm = torch.mean(torch.stack([torch.norm(self.mag_grads[nid]) for nid in self.mag_grads])) / self.accumulation_count if self.mag_grads else 0
                print(f"Accumulator: Updated {len(node_ids_to_update)} nodes after {self.accumulation_count} samples")
                print(f"  Avg phase grad norm: {avg_phase_grad_norm:.4f}, Avg mag grad norm: {avg_mag_grad_norm:.4f}")
                
        except Exception as e:
            print(f"Error updating vectors: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset accumulators for next batch
        self.phase_grads.clear()
        self.mag_grads.clear()
        self.accumulation_count = 0

