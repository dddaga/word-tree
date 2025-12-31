from custom_functions import activation_strength_forward
from nodestore import NodeStore

import torch.nn.functional as F
import torch
from torch import nn
from typing import List
import math

class Node(nn.Module):

    def __init__(
        self,
        node_store:NodeStore=None,
        gamma:float=1.0,

        node_id:int=None,
        phase_weight:torch.Tensor=None,
        mag_weight:torch.Tensor=None,
        phase_activation:torch.Tensor=None,
        mag_activation:torch.Tensor=None,
        activation_strength:int=None,
        incoming_connections:List[int]=None,
        outgoing_connections:List[int]=None,
        version:int=0,
        device:str='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.device = device
        self.node_id = node_id
        self.node_store = node_store
        self.version = version  # Track version for synchronization
        self.gamma = gamma  # Gamma parameter for magnitude exponential

        # Initialize continuous weights (phase in radians, magnitude in [-π, π])
        if phase_weight is None:
            phase_weight = torch.zeros(1)  # Initialize near 0
        if mag_weight is None:
            mag_weight = torch.zeros(1)  # Initialize near 0
        
        self.phase_weight = nn.Parameter(phase_weight.to(self.device).requires_grad_(True), requires_grad=True)
        self.mag_weight = nn.Parameter(mag_weight.to(self.device).requires_grad_(True), requires_grad=True)

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
            print(f"Warning: Loading node {self.node_id} from qdrant. This is not efficient and should be avoided.") 
            node = self.node_store.get_node(self.node_id)[0]
        else:
            self.node_id = node.id

        self.id = node.id

        # Load continuous float weights from Qdrant
        self.phase_weight = nn.Parameter(torch.tensor(node.vector['phase'], dtype=torch.float16, device=self.device).requires_grad_(True))
        self.mag_weight = nn.Parameter(torch.tensor(node.vector['mag'], dtype=torch.float16, device=self.device).requires_grad_(True))

        self.incoming_connections = node.payload['incoming_connections']
        self.outgoing_connections = node.payload['outgoing_connections']
        self.version = node.payload.get('version', 0)  # Load version from payload

        self.phase_activation = self.phase_weight.clone()
        self.mag_activation = self.mag_weight.clone()

        self.calculate_activation_strength()
        
    

    def calculate_activation_strength(self):
        """
        Calculate the activation strength of the node.
        """
        self.activation_strength = activation_strength_forward(self.phase_activation, self.mag_activation, self.gamma).to(self.device)
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
        phase_activations = phase_activations.to(self.device)
        mag_activations = mag_activations.to(self.device)
        activation_strengths = activation_strengths.to(self.device)
        
        #add current node's phase/mag activations to the input activations to process them together
        phase_activations = torch.cat((phase_activations, self.phase_activation.reshape(1, -1)), dim=0)
        mag_activations = torch.cat((mag_activations, self.mag_activation.reshape(1, -1)), dim=0)
        activation_strengths = torch.cat((activation_strengths.flatten(), self.activation_strength.reshape(1)), dim=0)
        
        # Scale activation strengths to prevent softmax saturation
        vector_dim = phase_activations.shape[-1]
        scaled_strengths = activation_strengths / (vector_dim ** 0.5)
        
        #calculate weights for the input activations
        weights = F.softmax(scaled_strengths, dim=-1).reshape(-1, 1)

        # Weighted sum of incoming activations with node's own weights
        phase_activations = weights * (phase_activations + self.phase_weight.reshape(1, -1))
        mag_activations = weights * (mag_activations + self.mag_weight.reshape(1, -1))

        # Sum to get new activations (continuous values, no modulo wrapping)
        self.phase_activation = phase_activations.sum(dim=0)
        self.mag_activation = mag_activations.sum(dim=0)
        
        # Keep phase in [0, 2π] range for interpretability (optional, but helps with numerical stability)
        self.phase_activation = self.phase_activation % (2 * math.pi)
        
        self.calculate_activation_strength()


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
    
    def decay_activations(self, decay_factor: float):
        """
        Apply temporal decay to activation strength.
        Makes older activations weaker than recent ones.
        """
        self.activation_strength = self.activation_strength * decay_factor
        

        
        


