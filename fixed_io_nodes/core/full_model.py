import torch
from torch import nn
from typing import List, Union
from .gnn_model import GNN


def get_dtype(dtype: Union[str, torch.dtype] = "float32") -> torch.dtype:
    """Central dtype for unquantized GNN. Config key: model.dtype ('float32' | 'float16')."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    raise ValueError("dtype must be 'float32' or 'float16'")
# from .input_adapter import LinearInputAdapter
from .quantization import Quantizer
from .nodestore import NodeStore
from .lookup_table import LookupTable

class Model(nn.Module):

    def __init__(self, gnn:GNN, quantizer:Quantizer=None):

        super().__init__()
        self.gnn = gnn
        # self.input_adapter = input_adapter
        # self.input_adapter_loaded = False
        if quantizer is not None:
            self.quantizer = quantizer
        else:
            self.quantizer = nn.Identity()

    

    def forward(self, x, tracer=None):
        
        out = x

        # out = out.reshape(self.gnn.input_node_count * self.gnn.vector_dim) #will forcefully raise error if input_adapter has wrong output dimensions
        phases = self.quantizer(out)

        out = self.gnn(phases, tracer=tracer)
        return out

    def reset(self):
        """
        Reset the model to initial state.
        """
        self.gnn.reset(fetch_weights=True)
        
        #nothing to reset for quantizer
    
def initialize_model(
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
    temporal_decay:float=1.0,
    device:str='cuda' if torch.cuda.is_available() else 'cpu',
    verbose:bool=False,
    dtype:Union[str, torch.dtype]="float32",
):
    dtype = get_dtype(dtype)
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
        temporal_decay=temporal_decay,
        dtype=dtype,
    )

    # quantizer = Quantizer(
    #     phase_bins=phase_bins,
    #     mag_bins=mag_bins,
    #     lookup_table=gnn.lookup_table,
    #     vector_dim=vector_dim,
    #     input_node_count=input_nodes,
    #     device=device,
    # )
    quantizer = None #no quantization for now

    return Model(gnn, quantizer)

def initialize_model_and_nodestore(

    qdrant_url:str,
    collection_name:str,

    total_nodes:int,
    input_nodes:int,
    output_nodes:int,
    cardinality:int,
    radiation_targets:int,

    vector_dim:int,
    phase_bins:int,
    mag_bins:int,
    iterations:int,
    activation_threshold:float,
    gamma:float=1.,
    temporal_decay:float=1.0,
    radiation_similarity_threshold:float=0.0,

    device:str='cuda' if torch.cuda.is_available() else 'cpu',
    verbose:bool=False,
    qdrant_params:dict=None,
    dtype:Union[str, torch.dtype]="float32",
):
    """
    returns model, node_store

    dtype: 'float32' | 'float16' (default 'float32'). Set via config model.dtype.
    qdrant_params: Optional dict of Qdrant parameters. If None, defaults will be used.
    """
    if qdrant_params is None:
        qdrant_params = {}

    lookup_table = None  # no quantization for now
    # lookup_table = LookupTable(
    #     phase_bins=phase_bins,
    #     mag_bins=mag_bins,
    #     gamma=gamma,
    #     device=device,
    # )


    node_store = NodeStore(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        lookup_table=lookup_table,
        num_total_nodes=total_nodes,
        num_input_nodes=input_nodes,
        num_output_nodes=output_nodes,
        cardinality=cardinality,
        vector_dim=vector_dim,
        phase_bins=phase_bins,
        mag_bins=mag_bins,
        temporal_decay=temporal_decay,
        radiation_similarity_threshold=radiation_similarity_threshold,
        **qdrant_params,
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
        device=device,
        verbose=verbose,
        dtype=dtype,
    )

    return model, node_store

    
