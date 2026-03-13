from .lookup_table import LookupTable
from pprint import pprint
import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional, Tuple
from .nodestore import NodeStore

try:
    from torch.optim.adam import adam as adam_step
except ImportError:
    try:
        from torch.optim._functional import adam as adam_step
    except ImportError:
        adam_step = None  # will raise at step() if unavailable


def _to_tensor(value: Union[torch.Tensor, List, tuple], dtype: torch.dtype, device: str) -> torch.Tensor:
    """Convert value to tensor, handling both tensor and list inputs."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype, device=device)
    else:
        return torch.tensor(value, dtype=dtype, device=device)


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


    def receive_gradients(self, phase_grads:Dict[int, torch.Tensor], mag_grads:Dict[int, torch.Tensor],
                          phase_grad_freq:Optional[Dict[int, int]]=None, mag_grad_freq:Optional[Dict[int, int]]=None):
        """
        Receives grads from workers. Optionally pass phase_grad_freq and mag_grad_freq (node_id -> count of
        samples that contributed). If omitted, each node is treated as count 1 (backwards compatible).
        """
        for node_id, phase_grad  in phase_grads.items():
            if phase_grad is None or torch.allclose(phase_grad, torch.zeros_like(phase_grad), atol=1e-8): #skip if the gradient is all zeros
                continue 
            if self.phase_grads[node_id] is None:
                self.phase_grads[node_id] = phase_grad
            else:
                self.phase_grads[node_id] += phase_grad
            self.phase_grad_counts[node_id] += (phase_grad_freq.get(node_id, 1) if phase_grad_freq is not None else 1)

        for node_id, mag_grad  in mag_grads.items():
            if mag_grad is None or torch.allclose(mag_grad, torch.zeros_like(mag_grad), atol=1e-8): #skip if the gradient is all zeros
                continue 
            if self.mag_grads[node_id] is None:
                self.mag_grads[node_id] = mag_grad
            else:
                self.mag_grads[node_id] += mag_grad
            self.mag_grad_counts[node_id] += (mag_grad_freq.get(node_id, 1) if mag_grad_freq is not None else 1)

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
        old_phase_values = {node.id: _to_tensor(node.vector['phase'], dtype=torch.float16, device=self.device) for node in nodes_to_update}
        old_mag_values = {node.id: _to_tensor(node.vector['mag'], dtype=torch.float16, device=self.device) for node in nodes_to_update}

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
    GNN parameters are updated with Adam via PyTorch's functional API.
    """
    def __init__(self,  node_store:NodeStore, lr:float, accumulation_steps:int=32,
                 betas:Tuple[float, float]=(0.9, 0.999), eps:float=1e-8,
                 verbose:bool=False,
                 device:str='cuda' if torch.cuda.is_available() else 'cpu',
                 save_path:str=None, save_interval:int=None):

        super().__init__()
        if adam_step is None:
            raise RuntimeError("torch.optim._functional.adam not available; cannot use Adam for GNN params.")

        self.device = device
        self.total_nodes = node_store.total_nodes
        self.vector_dim = node_store.vector_dim
        self.node_store = node_store
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.betas = betas
        self.eps = eps
        self.verbose = verbose
        self.save_path = save_path
        self.save_interval = save_interval
        self.step_count = 0  # Track number of steps for interval-based saving

        # Use sparse dictionaries - only store gradients for active nodes
        self.phase_grads = torch.zeros((self.total_nodes, self.vector_dim), device=self.device, dtype=torch.float32)
        self.mag_grads = torch.zeros_like(self.phase_grads)

        # Count of gradients received
        self.phase_grad_counts = torch.zeros(self.total_nodes, dtype=torch.int32, device=self.device)
        self.mag_grad_counts = torch.zeros_like(self.phase_grad_counts)

        # Adam state per node (sparse; created on first update)
        self.phase_exp_avg = torch.zeros((self.total_nodes, self.vector_dim), device=self.device, dtype=torch.float32)
        self.phase_exp_avg_sq = torch.zeros_like(self.phase_exp_avg)
        self.phase_state_steps = torch.zeros(self.total_nodes, dtype=torch.int32, device=self.device)

        self.mag_exp_avg = torch.zeros((self.total_nodes, self.vector_dim), device=self.device, dtype=torch.float32)
        self.mag_exp_avg_sq = torch.zeros_like(self.mag_exp_avg)
        self.mag_state_steps = torch.zeros(self.total_nodes, dtype=torch.int32, device=self.device)

        self.node_update_counts = {node_id:0 for node_id in range(self.total_nodes)}

    def receive_gradients(self, active_indices:torch.Tensor, phase_grads:torch.Tensor, mag_grads:torch.Tensor):
        """
        Receives grads from workers. Optionally pass phase_grad_freq and mag_grad_freq (node_id -> count of
        samples that contributed). If omitted, each node is treated as count 1 (backwards compatible).
        """
        
        if (active_indices is None) or active_indices.numel() == 0:
            return

        active_indices = active_indices.to(self.device)
        phase_grads = phase_grads.to(self.device)
        mag_grads = mag_grads.to(self.device)

        self.phase_grads.index_add_(0, active_indices, phase_grads)
        self.mag_grads.index_add_(0, active_indices, mag_grads)
        
        self.phase_grad_counts.index_add_(0, active_indices, torch.ones_like(active_indices, dtype=self.phase_grad_counts.dtype))
        self.mag_grad_counts.index_add_(0, active_indices, torch.ones_like(active_indices, dtype=self.mag_grad_counts.dtype))

        
    def _adam_update_node(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        state_step: torch.Tensor,
    ) -> None:
        """Single-node Adam step using PyTorch functional API; updates param and state in place."""
        beta1, beta2 = self.betas
        # max_exp_avg_sqs required by API; unused when amsgrad=False
        max_exp_avg_sq = torch.zeros_like(param, device=param.device)
        adam_step(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq],
            [state_step],
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=self.lr,
            weight_decay=0,
            eps=self.eps,
            maximize=False,
        )
    
    def _vectorized_adam(self, param, grad, exp_avg, exp_avg_sq, state_steps):
        """
        Custom Vectorized Adam that processes a matrix of nodes asynchronously.
        Uses pure PyTorch math to bypass the strict device/dtype grouping bugs 
        present in PyTorch's native functional adam API.
        """
        beta1, beta2 = self.betas
        
        # 1. Update step counts
        state_steps += 1
        
        # 2. Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        
        # 3. Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        
        # 4. Compute bias corrections (broadcast 1D step array to 2D features)
        step_2d = state_steps.unsqueeze(-1).float()
        bias_correction1 = 1.0 - beta1 ** step_2d
        bias_correction2 = 1.0 - beta2 ** step_2d
        
        # 5. Apply Adam update equation
        denom = (exp_avg_sq.sqrt() / torch.sqrt(bias_correction2)).add_(self.eps)
        step_size = self.lr / bias_correction1
        
        # In-place parameter update
        param.sub_(step_size * (exp_avg / denom))
        
        return param, exp_avg, exp_avg_sq, state_steps

    @torch.no_grad()
    def step(self):
        """Vectorized Adam update step applied only to nodes reaching the threshold."""
        
        # 1. Boolean masking instantly finds nodes that reached accumulation_steps
        phase_ready = self.phase_grad_counts >= self.accumulation_steps
        mag_ready = self.mag_grad_counts >= self.accumulation_steps
        
        ready_p_idx = phase_ready.nonzero(as_tuple=True)[0]
        ready_m_idx = mag_ready.nonzero(as_tuple=True)[0]
        
        updated_nodes = set()
        
        # 2. Vectorized Mass Update for Phase
        if ready_p_idx.numel() > 0:
            avg_grad = self.phase_grads[ready_p_idx] / self.phase_grad_counts[ready_p_idx].unsqueeze(-1).float()
            
            param, exp_avg, exp_avg_sq, steps = self._vectorized_adam(
                self.node_store.phase_weight[ready_p_idx], 
                avg_grad, 
                self.phase_exp_avg[ready_p_idx], 
                self.phase_exp_avg_sq[ready_p_idx], 
                self.phase_state_steps[ready_p_idx]
            )
            
            # Write results back to global memory blocks
            self.node_store.phase_weight[ready_p_idx] = param
            self.phase_exp_avg[ready_p_idx] = exp_avg
            self.phase_exp_avg_sq[ready_p_idx] = exp_avg_sq
            self.phase_state_steps[ready_p_idx] = steps
            
            # Reset only the nodes that updated
            self.phase_grads[ready_p_idx] = 0.0
            self.phase_grad_counts[ready_p_idx] = 0
            updated_nodes.update(ready_p_idx.tolist())
            
        # 3. Vectorized Mass Update for Magnitude
        if ready_m_idx.numel() > 0:
            avg_grad = self.mag_grads[ready_m_idx] / self.mag_grad_counts[ready_m_idx].unsqueeze(-1).float()
            
            param, exp_avg, exp_avg_sq, steps = self._vectorized_adam(
                self.node_store.mag_weight[ready_m_idx], 
                avg_grad, 
                self.mag_exp_avg[ready_m_idx], 
                self.mag_exp_avg_sq[ready_m_idx], 
                self.mag_state_steps[ready_m_idx]
            )
            
            self.node_store.mag_weight[ready_m_idx] = param
            self.mag_exp_avg[ready_m_idx] = exp_avg
            self.mag_exp_avg_sq[ready_m_idx] = exp_avg_sq
            self.mag_state_steps[ready_m_idx] = steps
            
            self.mag_grads[ready_m_idx] = 0.0
            self.mag_grad_counts[ready_m_idx] = 0
            updated_nodes.update(ready_m_idx.tolist())

        # 4. Synchronize Database / NodeStore states
        if updated_nodes:
            updated_list = list(updated_nodes)
            
            # Recompute trig values instantly for vector routing
            with self.node_store._lock:
                idx_tensor = torch.tensor(updated_list, dtype=torch.long, device=self.device)
                cos_vals = torch.cos(self.node_store.phase_weight[idx_tensor])
                sin_vals = torch.sin(self.node_store.phase_weight[idx_tensor])
                # print(self.node_store.phase_values.device)
                # print(idx_tensor.cpu().device)
                self.node_store.phase_values[idx_tensor.cpu()] = torch.cat([cos_vals, sin_vals], dim=-1).cpu()

            current_versions = self.node_store.version_tensor[updated_list]
            self.node_store.version_tensor[updated_list] = current_versions + 1
            
            for nid, v in zip(updated_list, current_versions + 1):
                if nid in self.node_store.payloads:
                    self.node_store.payloads[nid]['version'] = v.item()

        # 5. Save Interval logic
        if self.save_path is not None:
            self.step_count += 1
            if self.save_interval is None or self.step_count % self.save_interval == 0:
                if hasattr(self.node_store, 'save_weights'):
                    self.node_store.save_weights(self.save_path)
        
        return list(updated_nodes)

GradientAccumulator = UnquantizedGradientAccumulator
