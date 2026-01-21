from .lookup_table import LookupTable
from pprint import pprint
import torch
import torch.nn as nn
from typing import Dict
from .nodestore import NodeStore


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

        #we store the sum of all the gradients yet, and the number of gradients received
        self.phase_grads = {node_id:None for node_id in range(self.total_nodes)}
        self.phase_grad_counts = {node_id:0 for node_id in range(self.total_nodes)}
        self.mag_grads = {node_id:None for node_id in range(self.total_nodes)}
        self.mag_grad_counts = {node_id:0 for node_id in range(self.total_nodes)}

        self.node_update_counts = {node_id:0 for node_id in range(self.total_nodes)}


    def receive_gradients(self, phase_grads:Dict[int, torch.Tensor], mag_grads:Dict[int, torch.Tensor]):
        """
        This recieves the grads from other workers who are updating the graph. 
        The gradients are expected as Dict[str, torch.Tensor] with the keys as the parameter names              
        """
        for node_id, phase_grad  in phase_grads.items():
            if phase_grad is None or torch.allclose(phase_grad, torch.zeros_like(phase_grad), atol=1e-8): #skip if the gradient is all zeros
                continue 
            if self.phase_grads[node_id] is None:
                self.phase_grads[node_id] = phase_grad
            else:
                self.phase_grads[node_id] += phase_grad
            self.phase_grad_counts[node_id] += 1

        for node_id, mag_grad  in mag_grads.items():
            if mag_grad is None or torch.allclose(mag_grad, torch.zeros_like(mag_grad), atol=1e-8): #skip if the gradient is all zeros
                continue 
            if self.mag_grads[node_id] is None:
                self.mag_grads[node_id] = mag_grad
            else:
                self.mag_grads[node_id] += mag_grad
            self.mag_grad_counts[node_id] += 1

        # print("phase_grads: ", phase_grads)
        # print("mag_grads: ", mag_grads)
        # print("--------------------------------"*3, flush=True)

        print(f"recieved {len(phase_grads)} phase gradients and {len(mag_grads)} mag gradients")

    def step(self):
        """
        min_update_steps: The minimum number of updates required for a parameter, such that it is considered 
        for gradient descent in the current step. 
        The gradient used for gradient descent takes mean across all the available gradients.
        """
        # if min_update_steps is None:
        #     min_update_steps = self.accumulation_steps


        node_ids_to_update = set() #set of node ids to update
        for node_id, phase_grads in self.phase_grads.items():
            if self.if_update_needed(phase_grads):
                node_ids_to_update.add(node_id)
        for node_id, mag_grads in self.mag_grads.items():
            if self.if_update_needed(mag_grads):
                node_ids_to_update.add(node_id)

        node_ids_to_update = list(node_ids_to_update)
        nodes_to_update = self.node_store.get_node(node_ids_to_update)
        old_phase_values = {node.id: torch.tensor(node.vector['phase'], dtype=torch.float16, device=self.device) for node in nodes_to_update}
        old_mag_values = {node.id: torch.tensor(node.vector['mag'], dtype=torch.float16, device=self.device) for node in nodes_to_update}

        new_phase_values = {}
        new_mag_values = {}
        for node_id in node_ids_to_update:

            if self.if_update_needed(self.phase_grads[node_id]):
                phase_grad = self.phase_grads[node_id]
                self.phase_grads[node_id] = None #reset the gradients for the next step
            else:
                phase_grad = torch.tensor(0, dtype=torch.float16) #if the number of gradients isn't enough, then don't update it (achieved by setting grad to 0)

            if self.if_update_needed(self.mag_grads[node_id]):
                mag_grad = self.mag_grads[node_id]
                self.mag_grads[node_id] = None 
            else:
                mag_grad = torch.tensor(0, dtype=torch.float16)

            phase_vector = old_phase_values[node_id] #phase retrived from qdrant
            mag_vector = old_mag_values[node_id] #magnitude retrived from qdrant
            new_phase_values[node_id] = (phase_vector.to(phase_grad.dtype) - self.lr * phase_grad).round().long()%self.phase_bins
            new_mag_values[node_id] = (mag_vector.to(mag_grad.dtype) - self.lr * mag_grad).round().long()%self.mag_bins

            

        final_values = {node_id: {'phase': new_phase_values[node_id].tolist(), 'mag': new_mag_values[node_id].tolist()} for node_id in node_ids_to_update}
        
        for node_id in final_values:
            self.node_update_counts[node_id] += 1

        if final_values:
            try:
                self.node_store.update_vectors(final_values)
            except Exception as e:
                print(f"Error updating vectors: {e}")
                print(f"Final values: {final_values}")
            if self.verbose:
                print(f"Updated {len(node_ids_to_update)} nodes: {node_ids_to_update}")
        
        return node_ids_to_update

    def if_update_needed(self, grad:torch.Tensor):
        """
        Checks if the update if needed for a particular weight
        """
        # 0.5 is the minimum change required for any integer value to be changed. 
        # for example, for a value of 10, when you add 0.5, it will become 10.5, and then rounded to 11.
        # if any single value in the change tensor is greater than 0.5, then the update is needed

        return grad is not None and torch.any(torch.abs(self.lr*grad ) >= 0.5)


class UnquantizedGradientAccumulator(nn.Module):
    """
    Gradient accumulator for GNN parameters with batch-based updates.
    Accumulates gradients for N samples (batch_size) before updating weights.
    """
    def __init__(self,  node_store:NodeStore, lr:float, accumulation_steps:int=32, 
                 momentum:float=0.9, verbose:bool=False,
                 device:str='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.device = device
        self.total_nodes = node_store.total_nodes
        self.node_store = node_store
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.momentum = momentum
        self.verbose = verbose

        # Use sparse dictionaries - only store gradients for active nodes
        # This prevents memory waste for large graphs with sparse activations
        self.phase_grads = {}
        self.mag_grads = {}

        # Count of gradients received
        self.phase_grad_counts = {node_id:0 for node_id in range(self.total_nodes)}
        self.mag_grad_counts = {node_id:0 for node_id in range(self.total_nodes)}
        
        # Momentum buffers for SGD with momentum
        self.phase_velocities = {}
        self.mag_velocities = {}

        self.node_update_counts = {node_id:0 for node_id in range(self.total_nodes)}

    def receive_gradients(self, phase_grads:Dict[int, torch.Tensor], mag_grads:Dict[int, torch.Tensor]):
        """
        This recieves the grads from other workers who are updating the graph. 
        The gradients are expected as Dict[str, torch.Tensor] with the keys as the parameter names              
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
            self.phase_grad_counts[node_id] += 1


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
            self.mag_grad_counts[node_id] += 1
        
    def step(self):
        """
        Apply gradient updates when batch_size samples have been accumulated.
        Uses SGD with momentum for stable optimization.

        returns the node ids that were updated
        """
        

        # Get all nodes that have accumulated gradients
        node_ids_to_update = set(self.phase_grads.keys()).union(set(self.mag_grads.keys()))
        node_ids_to_update = list(node_ids_to_update)
        
        if not node_ids_to_update:
            return []

        # Fetch current node values and versions
        nodes_to_update = self.node_store.get_node(node_ids_to_update)
        old_phase_values = {node.id: torch.tensor(node.vector['phase'], dtype=torch.float32, device=self.device) for node in nodes_to_update}
        old_mag_values = {node.id: torch.tensor(node.vector['mag'], dtype=torch.float32, device=self.device) for node in nodes_to_update}
        old_versions = {node.id: node.payload.get('version', 0) for node in nodes_to_update}

        new_phase_values = {}
        new_mag_values = {}
        
        for node_id in node_ids_to_update:

            #update phase value if enough gradients are recieved
            if self.phase_grad_counts[node_id] >= self.accumulation_steps:
                avg_phase_grad = self.phase_grads[node_id] / self.phase_grad_counts[node_id]
                
                # Apply momentum: velocity = momentum * old_velocity + grad
                if node_id not in self.phase_velocities:
                    self.phase_velocities[node_id] = avg_phase_grad
                else:
                    self.phase_velocities[node_id] = self.momentum * self.phase_velocities[node_id] + avg_phase_grad
                
                # SGD update: weight -= lr * velocity
                phase_vector = old_phase_values[node_id]
                new_phase_values[node_id] = phase_vector - self.lr * self.phase_velocities[node_id]
                
                #reset the gradients and counts for the next step
                self.phase_grad_counts[node_id] = 0
                self.phase_grads[node_id] = None
            else:
                new_phase_values[node_id] = old_phase_values[node_id]

            #update mag value if enough gradients are recieved
            if self.mag_grad_counts[node_id] >= self.accumulation_steps:
                avg_mag_grad = self.mag_grads[node_id] / self.mag_grad_counts[node_id]
                
                # Apply momentum
                if node_id not in self.mag_velocities:
                    self.mag_velocities[node_id] = avg_mag_grad
                else:
                    self.mag_velocities[node_id] = self.momentum * self.mag_velocities[node_id] + avg_mag_grad
                
                # SGD update
                mag_vector = old_mag_values[node_id]
                new_mag_values[node_id] = mag_vector - self.lr * self.mag_velocities[node_id]
                
                #reset the gradients and counts for the next step
                self.mag_grad_counts[node_id] = 0
                self.mag_grads[node_id] = None
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

        for node_id in final_values:
            self.node_update_counts[node_id] += 1

        try:
            # Update vectors in DB
            self.node_store.update_vectors(final_values)
            
            # Increment versions for updated nodes
            new_versions = [old_versions[node_id] + 1 for node_id in node_ids_to_update]
            self.node_store.update_node_versions(node_ids_to_update, new_versions)
            
            if False:
                #This is supposed to print grad norms, but as the number of stored gradients are cleared above, this needs to be fixed accordingly.
                avg_phase_grad_norm = torch.mean(torch.stack([torch.norm(self.phase_grads[nid]) for nid in self.phase_grads])) / self.phase_grad_counts[node_id] if self.phase_grads else 0
                avg_mag_grad_norm = torch.mean(torch.stack([torch.norm(self.mag_grads[nid]) for nid in self.mag_grads])) / self.mag_grad_counts[node_id] if self.mag_grads else 0
                print(f"Accumulator: Updated {len(node_ids_to_update)} nodes after {self.phase_grad_counts[node_id]} samples")
                print(f"  Avg phase grad norm: {avg_phase_grad_norm:.4f}, Avg mag grad norm: {avg_mag_grad_norm:.4f}")
            
            if self.verbose:
                print(f"Updated {len(node_ids_to_update)}, Nodes: {node_ids_to_update}")
                
        except Exception as e:
            print(f"Error updating vectors: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset accumulators for next batch
        self.phase_grads.clear()
        self.mag_grads.clear()
        return node_ids_to_update

GradientAccumulator = UnquantizedGradientAccumulator
