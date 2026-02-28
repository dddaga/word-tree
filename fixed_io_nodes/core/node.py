from .custom_functions import activation_strength_forward
from .nodestore import NodeStore
from .lookup_table import LookupTable

import torch.nn.functional as F
import torch
from torch import nn
from typing import List, Union


def _to_tensor_with_grad(value: Union[torch.Tensor, List, tuple], dtype: torch.dtype, device: str) -> torch.Tensor:
    """Convert value to tensor with requires_grad, handling both tensor and list inputs."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype, device=device).requires_grad_(True)
    else:
        return torch.tensor(value, dtype=dtype, device=device).requires_grad_(True)

class QuantizedNode(nn.Module):

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
        device:str='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.device = device
        self.node_id = node_id
        self.lookup_table = lookup_table 
        self.node_store = node_store


        #TODO: phase/mag weights should be initialized as nn.Parameters
        if phase_weight is None:
            phase_weight = torch.tensor(1.)
        if mag_weight is None:
            mag_weight = torch.tensor(1.)
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

        
        self.phase_weight = nn.Parameter(_to_tensor_with_grad(node.vector['phase'], dtype=torch.float16, device=self.device))
        self.mag_weight = nn.Parameter(_to_tensor_with_grad(node.vector['mag'], dtype=torch.float16, device=self.device))


        self.incoming_connections = node.payload['incoming_connections']
        self.outgoing_connections = node.payload['outgoing_connections']

        self.phase_activation = self.phase_weight.clone()
        self.mag_activation = self.mag_weight.clone()

        self.calculate_activation_strength()
        
    

    def calculate_activation_strength(self):
        """
        Calculate the activation strength of the node.
        """
        self.activation_strength = activation_strength_forward(self.phase_activation, self.mag_activation, self.lookup_table).to(self.device)
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
        
        #calculate weights for the input activations
        weights = F.softmax(activation_strengths, dim=-1).reshape(-1, 1)


        phase_activations = weights*(phase_activations + self.phase_weight.reshape(1, -1))
        mag_activations = weights*(mag_activations + self.mag_weight.reshape(1, -1))

        self.phase_activation = phase_activations.sum(dim=0)%self.lookup_table.phase_bins
        self.mag_activation = mag_activations.sum(dim=0)%self.lookup_table.mag_bins
        self.calculate_activation_strength() 

        # if not self.phase_activation.requires_grad:
        #     print("===")


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
        


def _parse_dtype(dtype):  # str or torch.dtype -> torch.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    raise ValueError("dtype must be 'float32', 'float16', or torch.dtype")


class UnquantizedNode(nn.Module):

    def __init__(
        self,
        node_store:NodeStore=None,
        gamma:float=1.0,
        dtype:Union[str, torch.dtype]="float32",
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
        self.dtype = _parse_dtype(dtype)
        self.node_id = node_id
        self.node_store = node_store
        self.version = version  # Track version for synchronization
        self.gamma = gamma  # legacy parameter

        
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

        # Load continuous float weights from node_store
        self.phase_weight = nn.Parameter(_to_tensor_with_grad(node.vector['phase'], dtype=self.dtype, device=self.device))
        self.mag_weight = nn.Parameter(_to_tensor_with_grad(node.vector['mag'], dtype=self.dtype, device=self.device))

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
        
        
        # Routing weights: softmax(strength / temperature)
        weights = F.softmax(activation_strengths, dim=-1).reshape(-1, 1)

        # Complex representation per row: energy = softplus(mag), real = energy*cos(phase), imag = energy*sin(phase)
        energy_inputs = torch.exp(mag_activations)
        real_inputs = energy_inputs * torch.cos(phase_activations)
        imag_inputs = energy_inputs * torch.sin(phase_activations)

        # Node's own bias (broadcast to each row)
        energy_w = self.mag_weight
        real_w = energy_w * torch.cos(self.phase_weight)
        imag_w = energy_w * torch.sin(self.phase_weight)
        real_per_row = real_inputs*real_w - imag_w*imag_inputs
        imag_per_row = real_inputs*imag_w + real_w*imag_inputs

        # Weighted superposition
        real_sum = (weights * real_per_row).sum(dim=0)
        imag_sum = (weights * imag_per_row).sum(dim=0)
        

        # Phase update: damping, no modulo
        new_phase_raw = torch.atan2(imag_sum, real_sum)
        self.phase_activation = new_phase_raw

        # Magnitude update: log(norm + epsilon)
        mag_next = torch.log(torch.sqrt(real_sum ** 2 + imag_sum ** 2) + 1e-8)
        self.mag_activation = mag_next

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
        

        
Node = UnquantizedNode
