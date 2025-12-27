import torch
from torch import nn
from typing import List
from gnn_model import GNN
# from input_adapter import LinearInputAdapter
from quantization import Quantizer
from nodestore import NodeStore

class Model(nn.Module):

    def __init__(self, gnn:GNN, quantizer:Quantizer):

        super().__init__()
        self.gnn = gnn
        # self.input_adapter = input_adapter
        # self.input_adapter_loaded = False
        self.quantizer = quantizer

    

    def forward(self, x):
        
        # if self.input_adapter_loaded:
        #     with torch.no_grad():
        #         out = self.input_adapter(x)
        # else:
        out = x

        out = out.reshape(self.gnn.input_node_count * self.gnn.vector_dim) #will forcefully raise error if input_adapter has wrong output dimensions
        phases = self.quantizer(out)

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
    
def initialize_model(
    # input_dim:int, 
    # adapter_hidden_dims:List[int],
    # adapter_dropout:float,


    node_store:NodeStore,
    cardinality:int, 
    radiation_targets:int,
    total_nodes:int,
    input_nodes:int, 
    output_nodes:int,
    phase_bins:int,
    mag_bins:int,
    vector_dim:int,
    iterations:int,
    activation_threshold:float,
    gamma:float=1.,
    device:str='cuda' if torch.cuda.is_available() else 'cpu',
    verbose:bool=False,


    # adapter_normalization_layer:str='layer_norm',

):

    # input_adapter = LinearInputAdapter(input_dim, output_dim=input_nodes*vector_dim, hidden_dims=adapter_hidden_dims, dropout=adapter_dropout, normalization_layer=adapter_normalization_layer)

    gnn = GNN(
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
        device=device,
        verbose=verbose,
    )

    quantizer = Quantizer(
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        lookup_table=gnn.lookup_table,
        vector_dim=vector_dim,
        input_node_count=input_nodes,
        device=device,
    )

    return Model(gnn, quantizer)