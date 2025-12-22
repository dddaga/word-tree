from lookup_table import LookupTable
import torch
import torch.nn as nn
from typing import Dict
from nodestore import NodeStore


class GradientAccumulator(nn.Module):

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

