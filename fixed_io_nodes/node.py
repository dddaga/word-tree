from custom_functions import activation_stength_forward
from nodestore import NodeStore
from lookup_table import LookupTable

import torch.nn.functional as F
import torch
from torch import nn
from typing import List

class Node(nn.Module):

    def __init__(
        self,
        node_store:NodeStore=None,
        lookup_table:LookupTable=None,

        node_id:int=None,
        phase_weight:torch.Tensor=None,
        mag_weight:torch.Tensor=None,
        phase_activation:torch.Tensor=None,
        mag_activation:torch.Tensor=None,
        activation_strength:int=None,
        incoming_connections:List[int]=None,
        outgoing_connections:List[int]=None,
    ):
        super().__init__()
        self.node_id = node_id
        self.lookup_table = lookup_table 
        self.node_store = node_store


        #TODO: phase/mag weights should be initialized as nn.Parameters
        if phase_weight is None:
            phase_weight = torch.tensor(1.).requires_grad_(True)
        if mag_weight is None:
            mag_weight = torch.tensor(1.).requires_grad_(True)
        self.phase_weight = nn.Parameter(phase_weight)
        self.mag_weight = nn.Parameter(mag_weight)

        self.phase_activation = phase_activation
        self.mag_activation = mag_activation
        self.activation_strength = activation_strength
        self.incoming_connections = incoming_connections
        self.outgoing_connections = outgoing_connections

    def load_values(self, node=None):
        """
        node: the node values fetched from qdrant. 
        This allows to load multiple nodes at once from qdrant if needed.

        1) Fetch phase_weight and mag_weight from node_store
        2) assign phase and mag activations
        3) calculate activation strength
        """

        if node is None:
            node = self.node_store.get_node(self.node_id)[0]
        else:
            self.node_id = node.id

        self.id = node.id

        
        self.phase_weight = nn.Parameter(torch.tensor(node.vector['phase'], dtype=torch.float16).requires_grad_(True))
        self.mag_weight = nn.Parameter(torch.tensor(node.vector['mag'], dtype=torch.float16).requires_grad_(True))


        self.incoming_connections = node.payload['incoming_connections']
        self.outgoing_connections = node.payload['outgoing_connections']

        self.phase_activation = self.phase_weight.clone()
        self.mag_activation = self.mag_weight.clone()

        self.calculate_activation_strength()
        
    

    def calculate_activation_strength(self):
        """
        Calculate the activation strength of the node.
        """
        self.activation_strength = activation_stength_forward(self.phase_activation, self.mag_activation, self.lookup_table)
        return self.activation_strength

        
    
    def update_activations(
        self,
        phase_activations: torch.Tensor,
        mag_activations: torch.Tensor,
        activation_strengths: torch.Tensor,
    ):
        """
        Parameters:
        phase_activations: shape = (num_inputs, vector_dim)
        mag_activations: shape = (num_inputs, vector_dim)
        activation_strengths: shape = (num_inputs, )

        the reason activation strengths are taken as parameter is because it shouldn't 
        be calculated multiple times for the same node when passing to multiple nodes
        """
        #it is assumed current node's activation_strength is already calculated
        
        if phase_activations.dim() == 1:
            phase_activations = phase_activations.reshape(1, -1)
        if mag_activations.dim() == 1:
            mag_activations = mag_activations.reshape(1, -1)
        elif activation_strengths.dim() == 0:
            activation_strengths = activation_strengths.reshape(1)
        
        #add current node's phase/mag activations to the input activations to process them together
        
        phase_activations = torch.cat((phase_activations, self.phase_activation.reshape(1, -1)), dim=0)
        mag_activations = torch.cat((mag_activations, self.mag_activation.reshape(1, -1)), dim=0)
        activation_strengths = torch.cat((activation_strengths.flatten(), self.activation_strength.reshape(1)), dim=0)
        
        #calculate weights for the input activations
        weights = F.softmax(activation_strengths, dim=-1).reshape(-1, 1)


        phase_activations = weights*(phase_activations + self.phase_weight.reshape(1, -1))
        mag_activations = weights*(mag_activations + self.mag_weight.reshape(1, -1))

        self.phase_activation = phase_activations.sum(dim=0)%self.lookup_table.phase_bins
        self.mag_activation = mag_activations.sum(dim=0)%self.lookup_table.mag_bins
        self.calculate_activation_strength() 

        if not self.phase_activation.requires_grad:
            print("===")


    def reset(self):
        """
        Reset the node to initial state. 
        """
        #reset weights' gradients
        self.phase_weight.grad = None
        self.mag_weight.grad = None

        #reset activations back to start
        self.phase_activation = self.phase_weight.clone()
        self.mag_activation = self.mag_weight.clone()

        #recalculate activation strength
        self.calculate_activation_strength()
        

        
        


