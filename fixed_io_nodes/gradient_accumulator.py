from lookup_table import LookupTable
import torch
import torch.nn as nn
from typing import Dict
from nodestore import NodeStore


class GradientAccumulator(nn.Module):
    """
    Gradient accumulator specifically for GNN parameters only
    """
    def __init__(self, accumulation_steps:int, node_store:NodeStore, lr:float, verbose:bool=False,
    device:str='cuda' if torch.cuda.is_available() else 'cpu'):

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

    def step(self, min_update_steps:int=None):
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

        if final_values:
            try:
                self.node_store.update_vectors(final_values)
            except Exception as e:
                print(f"Error updating vectors: {e}")
                print(f"Final values: {final_values}")
            if self.verbose:
                print(f"Updated {len(node_ids_to_update)} nodes")

    def if_update_needed(self, grad:torch.Tensor):
        """
        Checks if the update if needed for a particular weight
        """
        # 0.5 is the minimum change required for any integer value to be changed. 
        # for example, for a value of 10, when you add 0.5, it will become 10.5, and then rounded to 11.
        # if any single value in the change tensor is greater than 0.5, then the update is needed

        return grad is not None and torch.any(torch.abs(self.lr*grad ) >= 0.5)

class OldGradientAccumulator(nn.Module):

    def __init__(
            self,
            named_parameters,
            phase_bins:int,
            mag_bins:int,
            accumulation_steps:int,
            node_store:NodeStore,
            lr: float = 1e-3,
    ):
        """
        phases: All the phase values (num_nodes, vector_dim)
        mags: All the magnitudes (num_nodes, vector_dim)

        update_count: Number of times each node has been updated (num_nodes, 2) [2 because phase and magnitude] 

        """
        super().__init__()
        

        #maintain the non-gnn parameters
        self.parameters = {name:param for name, param in named_parameters if param.requires_grad} #only update the parameters that require gradients
        self.parameter_grads = {name:[] for name in self.parameters.keys()}


        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.node_store = node_store

        self.num_nodes = node_store.total_nodes

    def receive_gradients(
            self,
            named_grads: Dict[str, torch.Tensor],
    ):
        """
        This recieves the grads from other workers who are updating the graph. 
        The gradients are expected as Dict[str, torch.Tensor] with the keys as the parameter names              
        """

        for name, grad in named_grads.items():
            if name in self.parameters.keys():
                if torch.allclose(grad, torch.zeros_like(grad), atol=1e-8): #skip if the gradient is all zeros
                    continue 

                self.parameter_grads[name].append(grad)
            else:
                raise ValueError(f"Gradient {name} not found in the parameters")

    def step(self, min_update_steps:int=None):
        """
        min_update_steps: The minimum number of updates required for a node, such that it is considered 
        for gradient descent in the current step. 
        The gradient used for gradient descent takes mean across all the steps.

        """
        if min_update_steps is None:
            min_update_steps = self.accumulation_steps

        ## UPDATE non-gnn parameters
        for name, grad in self.parameter_grads.items():
            param = self.parameters[name]
            param.data.sub_(self.lr * torch.stack(grad).mean(dim=0))
            self.parameter_grads[name] = []
        

        ### UPDATE gnn parameters

        #get the nodes to update (Change with Dragonfly)
        mask = self.update_count>min_update_steps 
        

        nodes_to_update = torch.where(mask)[0] #indices of the nodes to update (phase update)
                

        #reset the update count for the updated nodes back to zero (Change with Dragonfly)
        self.update_count[mask] = 0 

        points_to_update = self.node_store.get_node(nodes_to_update) #get the points to update

        final_values = {}

        for node_id in nodes_to_update:
            node_id = int(node_id)

            phase_grads = torch.stack(self.phase_grads[node_id]).mean(dim=0)
            mag_grads = torch.stack(self.mag_grads[node_id]).mean(dim=0)

            phase_vector = points_to_update[node_id].vector['phase'] #phase retrived from qdrant
            mag_vector = points_to_update[node_id].vector['mag'] #magnitude retrived from qdrant


            #calcuate new phase/mag values
            new_phase = (phase_vector.to(phase_grads.dtype) - self.lr * phase_grads).round().long()%self.phase_bins
            new_mag = (mag_vector.to(mag_grads.dtype) - self.lr * mag_grads).round().long()%self.mag_bins

            final_values[node_id]['phase'] = new_phase
            final_values[node_id]['mag'] = new_mag
    
        self.node_store.update_vectors(final_values)

