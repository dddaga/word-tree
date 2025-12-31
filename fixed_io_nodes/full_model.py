import torch
from torch import nn
from typing import List
from gnn_model import GNN
from nodestore import NodeStore

class Model(nn.Module):

    def __init__(self, gnn:GNN):

        super().__init__()
        self.gnn = gnn

    def forward(self, x):
        """
        x: input tensor of shape (batch, features) or flattened
        
        The input is normalized to [0, 2π] range to represent continuous phase values.
        """
        # Reshape to match GNN input expectations
        out = x.reshape(self.gnn.input_node_count, self.gnn.vector_dim)
        
        # Normalize input to [0, 2π] range for phase representation
        # This maps pixel values [0, 1] to phase angles [0, 2π]
        phases = out * (2 * torch.pi)

        out = self.gnn(phases)
        return out

    def reset(self):
        """
        Reset the model to initial state.
        """
        self.gnn.reset(fetch_weights=True)
        
        # for p in self.input_adapter.parameters():
        #     p.grad = None

        #nothing to reset for quantizer
    
    def reset_activations(self):
        """
        Reset activations only (keep weights).
        Called after each forward/backward pass.
        """
        self.gnn.reset_activations()
        
        # for p in self.input_adapter.parameters():
        #     p.grad = None
        
        #nothing to reset for quantizer
    
def initialize_model(
    node_store:NodeStore,
    cardinality:int, 
    radiation_targets:int,
    total_nodes:int,
    input_nodes:int, 
    output_nodes:int,
    phase_bins:int=None,  # Legacy, kept for compatibility
    mag_bins:int=None,    # Legacy, kept for compatibility
    vector_dim:int=None,
    iterations:int=None,
    activation_threshold:float=None,
    gamma:float=1.,
    temporal_decay:float=1.0,
    device:str='cuda' if torch.cuda.is_available() else 'cpu',
    verbose:bool=False,
):

    gnn = GNN(
        node_store=node_store,
        cardinality=cardinality,
        radiation_targets=radiation_targets,
        total_nodes=total_nodes,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        phase_bins=phase_bins,  # Passed but not used internally
        mag_bins=mag_bins,      # Passed but not used internally
        vector_dim=vector_dim,
        iterations=iterations,
        activation_threshold=activation_threshold,
        gamma=gamma,
        temporal_decay=temporal_decay,
        device=device,
        verbose=verbose,
    )

    return Model(gnn)

def initialize_model_and_nodestore(

    qdrant_url:str,
    collection_name:str,

    total_nodes:int,
    input_nodes:int,
    output_nodes:int,
    cardinality:int,
    radiation_targets:int,

    vector_dim:int,
    phase_bins:int=None,  # Legacy, kept for compatibility
    mag_bins:int=None,    # Legacy, kept for compatibility
    iterations:int=None,
    activation_threshold:float=None,
    gamma:float=1.,
    temporal_decay:float=1.0,

    device:str='cuda' if torch.cuda.is_available() else 'cpu',
    verbose:bool=False,
):
    """
    Initialize model and node store without quantization.
    
    Returns: model, node_store
    """

    node_store = NodeStore(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        num_total_nodes=total_nodes,
        num_input_nodes=input_nodes,
        num_output_nodes=output_nodes,
        cardinality=cardinality,
        vector_dim=vector_dim,
        phase_bins=phase_bins,  # Legacy parameter
        mag_bins=mag_bins,      # Legacy parameter
    )

    model = initialize_model(
        node_store=node_store,
        cardinality=cardinality,
        radiation_targets=radiation_targets,
        total_nodes=total_nodes,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        vector_dim=vector_dim,
        iterations=iterations,
        activation_threshold=activation_threshold,
        gamma=gamma,
        temporal_decay=temporal_decay,
        device=device,
        verbose=verbose,
    )

    return model, node_store
