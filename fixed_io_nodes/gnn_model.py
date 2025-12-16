from torch import nn
import torch
from lookup_table import LookupTable
from custom_functions import signal_forward, phase_forward, mag_forward

from typing import List, Union

from nodestore import NodeStore
from node import Node

class MyModuleDict(nn.ModuleDict):
    def __getitem__(self, key: Union[str, int]) -> nn.Module:
        if isinstance(key, int):
            key = str(key)
        return super().__getitem__(key)
    def __setitem__(self, key: Union[str, int], value: nn.Module) -> None:
        if isinstance(key, int):
            key = str(key)
        return super().__setitem__(key, value)


class GNN(nn.Module):

    def __init__(
        self,
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
    ):
        super().__init__()

        self.node_store = node_store
        self.cardinality = cardinality
        self.radiation_targets = radiation_targets
        self.total_nodes = total_nodes
        self.input_node_count = input_nodes
        self.output_node_count = output_nodes
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.vector_dim = vector_dim
        self.iterations = iterations
        self.activation_threshold = activation_threshold
        self.verbose = verbose

        self.lookup_table = LookupTable(phase_bins, mag_bins, gamma=gamma, device=device)

        #input nodes are considered active from the start
        self.active_nodes = {
            n_id: Node(
                node_store=self.node_store, 
                lookup_table=self.lookup_table,
                node_id=n_id,
                ) for n_id in self.node_store.input_nodeids
        }
        self.input_nodes = self.active_nodes.copy()
        for _, n in self.active_nodes.items():
            n.load_values()

        self.active_nodes = MyModuleDict(self.active_nodes)
        

        
        self.output_nodeids = self.node_store.output_nodeids
        self.input_nodeids = self.node_store.input_nodeids

        if self.verbose:
            print(f"==================GNN Configuration=====================")
            print(f"Initialized GNN with {total_nodes} nodes, {input_nodes} input nodes, {output_nodes} output nodes")
            print(f"Phase bins: {phase_bins}, Mag bins: {mag_bins}")
            print(f"Vector dimension: {vector_dim}")
            print(f"Iterations: {iterations}")
            print(f"Activation threshold: {activation_threshold}")
            print(f"Gamma: {gamma}")
            print(f"Device: {device}")
            print(f"=====================================")



    def _compute_radiation_targets(self, nodes:List[Node], k:int=None):
        """
        Arguments:
        nodes: List[Node]
        k: int = None (default is self.radiation_targets)

        Returns:
        topk_indices: Dict[node_ids: List[ids of closest k nodes]]
        """

        if k is None:
            k = self.radiation_targets
        if isinstance(nodes, Node):
            nodes = [nodes]
        

        topk_indices = {}

        #TODO: this can be optimized to do combined search for all nodes at once
        for node in nodes:
            query_vector = node.phase_activation
            node_id = node.id
            nodes = self.node_store.search_nodes(query_vector.tolist(), vector_name='phase', with_payload=False, with_vectors=False)
            topk_indices[node_id] = [node.id for node in nodes]
        
        return topk_indices
    

    def one_step_forward(self, input_values:torch.Tensor=None):
        """
        input_values: torch.Tensor=None, shape = (input_nodes, vector_dim)
        if input is given, then it is inserted into the input nodes, otherwise a normal 
        1 step propagation is done. Input values are assumed to be quantized.

        Returns: None
        """

        if self.verbose: #TODO: print logs
            pass
        

        #TODO:
        #fetch input nodes which are not exisiting in active_nodes 
        # (This is needed in case of applying beam width)


        
        #input data is fed into the input nodes before propagation of remaining network
        if input_values is not None:


            assert input_values.shape == (self.input_node_count, self.vector_dim), f"expected input_values of shape ({self.input_node_count}, {self.vector_dim}), but got {input_values.shape}"

            #this is assumed to be quantized, i.e. indices from lookup table
            input_phases = input_values
            input_mags = torch.zeros((self.input_node_count, self.vector_dim)) 
            activation_strengths = signal_forward(input_phases, input_mags, self.lookup_table)

            #inject input values into input nodes    
            for n_id, node in self.input_nodes.items():
                node.update_activations(
                    phase_activations=input_phases[n_id],
                    mag_activations=input_mags[n_id],
                    activation_strengths=activation_strengths[n_id],
                )

            



        #fetch radiation targets 
        radiation_targets = self._compute_radiation_targets(set(self.active_nodes.values()))
        
        #Find the nodes to which we would have to propagate values to.
        #and thus fetch those nodes. (for both direct and radiation connections)
        new_active_nodes_ids = set(int(i) for i in self.active_nodes.keys())
        for node in self.active_nodes.values():
            new_active_nodes_ids.update(node.outgoing_connections)
        for _, targets in radiation_targets.items(): 
            new_active_nodes_ids.update(targets)


        #get the node ids that we need to fetch from qdrant
        nodes_to_fetch_ids = new_active_nodes_ids - set(int(i) for i in self.active_nodes.keys())
        nodes_to_fetch_ids = list(nodes_to_fetch_ids)

        #fetch the nodes from qdrant
        new_nodes_values = self.node_store.get_node(nodes_to_fetch_ids)

        #this just creates the Node objects, doesn't load the values into them
        new_nodes = [Node(node_store=self.node_store, lookup_table=self.lookup_table) for _ in new_nodes_values]
        
        for i, new_node_value in enumerate(new_nodes_values):
            new_nodes[i].load_values(new_node_value) #loads the values into the Node objects
        
        new_nodes = set(new_nodes)

    
        #the incoming connections coming to new nodes from the current active nodes
        incoming_connections = {node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)}

        for node in self.active_nodes.values():
            for n_id in node.outgoing_connections:
                incoming_connections[n_id].append(node.id)
        for n_id, targets in radiation_targets.items():
            for target in targets:
                incoming_connections[target].append(n_id)

        #it is necessary to store them separately because after up call node.update_activations, they get changed
        phase_activations = {node.id: node.phase_activation.clone() for node in self.active_nodes.values()}
        mag_activations = {node.id: node.mag_activation.clone() for node in self.active_nodes.values()}
        activation_strengths = {node.id: node.activation_strength.clone() for node in self.active_nodes.values()}


        #update the activations of the active nodes and the new nodes
        for node in set(self.active_nodes.values()).union(new_nodes):

            if len(incoming_connections[node.id]) == 0:
                continue
            
            node.update_activations(
                phase_activations=torch.stack([phase_activations[n_id] for n_id in incoming_connections[node.id]]),
                mag_activations=torch.stack([mag_activations[n_id] for n_id in incoming_connections[node.id]]),
                activation_strengths=torch.stack([activation_strengths[n_id] for n_id in incoming_connections[node.id]]),
            )
            # print("updated node-", node.id)

        #update the active nodes to include the new nodes
        for node in new_nodes:
            self.active_nodes[node.id] = node

        


    def forward(self, input_values:torch.Tensor=None):

        self.one_step_forward(input_values)
        if self.verbose:
            print(f"first pass done")

        for iteration in range(self.iterations-1):
            self.one_step_forward()
            if self.verbose:
                print(f"Iteration {iteration+2} done")

        output_signals = {}
        
        for node_id in self.output_nodeids:
            node_id = str(node_id)
            if node_id in self.active_nodes:
                output_signals[node_id] = self.active_nodes[node_id].activation_strength
            else:
                print("Node not active: ", node_id)
                output_signals[node_id] = torch.tensor(0.).requires_grad_(True)
        
        output_signals = torch.stack([v for k, v in sorted(output_signals.items())])
        return output_signals

    def reset(self, fetch_weights:bool=False):
        """
        Reset the model to initial state. 
        1) Resets the active nodes to input nodes. (this is enough to consider the model as reset)
        2) If fetch_weights is True, fetches the input node weights from qdrant, otherwise uses the same weights
        """

        if fetch_weights:
            for _, n in self.input_nodes.items():
                n.load_values()


        self.active_nodes = self.input_nodes.copy()
        for _, n in self.active_nodes.items():
            n.reset()

        self.active_nodes = MyModuleDict(self.active_nodes)
        
        
        



        

#this didn't use the Node class for individual nodes. Just kept for reference
class OldGNN(nn.Module):

    def __init__(
        self,
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
    ):
        super().__init__()
        self.cardinality = cardinality
        self.radiation_targets = radiation_targets
        self.total_nodes = total_nodes
        self.input_node_count = input_nodes
        self.output_node_count = output_nodes
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.vector_dim = vector_dim
        self.iterations = iterations    
        self.activation_threshold = activation_threshold
        self.device = device
        self.verbose = verbose

        self.lookup_table = LookupTable(phase_bins, mag_bins, gamma=gamma, device=device)

        self.node_store = node_store


        self.node_ids = self.node_store.node_ids

        all_nodes = self.node_store.get_node(self.node_ids)

        self.phases = nn.Parameter(torch.stack([torch.tensor(node.vector['phase']) for node in all_nodes]).to(torch.float16).requires_grad_(True))
        self.mags = nn.Parameter(torch.stack([torch.tensor(node.vector['mag']) for node in all_nodes]).to(torch.float16).requires_grad_(True))
        self.connections = {
            node.id: {
                'outgoing': list(node.payload['outgoing_connections']),
                'incoming': list(node.payload['incoming_connections'])
            } for node in all_nodes
        }

        self.output_nodeids = list(self.node_store.output_nodeids)
        self.input_nodeids = list(self.node_store.input_nodeids)


        


        if self.verbose:
            print(f"==================GNN Configuration=====================")
            print(f"Initialized GNN with {total_nodes} nodes, {input_nodes} input nodes, {output_nodes} output nodes")
            print(f"Phase bins: {phase_bins}, Mag bins: {mag_bins}")
            print(f"Vector dimension: {vector_dim}")
            print(f"Iterations: {iterations}")
            print(f"Activation threshold: {activation_threshold}")
            print(f"Gamma: {gamma}")
            print(f"Device: {device}")
            print(f"=====================================")

    def _compute_radiation_targets(self, phases, k:int=None):
        """
        Arguments:
        phases: Dict[node_ids: phase] phase is the indices, not the value itself

        Returns:
        topk_indices: Dict[node_ids: List[closest node_ids]]
        """

        if k is None:
            k = self.radiation_targets

        topk_indices = {}

        for node_id, query_vector in enumerate(phases):
            nodes = self.node_store.search_nodes(query_vector, vector_name='phase', with_payload=False, with_vectors=False)
            topk_indices[node_id] = [node.id for node in nodes]
        
        return topk_indices
    

    def load_phase_and_mag(self):
        """
        Load the new phase/mag values from qdrant into model parameters. 

        The values are loaded into the same location where the original values were stored. 
        """
        all_nodes = self.node_store.get_node(self.node_ids)
        phases = torch.stack([torch.tensor(node.vector['phase']) for node in all_nodes]).to(torch.float16).requires_grad_(True)
        mags = torch.stack([torch.tensor(node.vector['mag']) for node in all_nodes]).to(torch.float16).requires_grad_(True)

        self.phases.data.copy_(phases)
        self.mags.data.copy_(mags)

    
    def one_step_forward(self, phases, mags):
        """
        inputs: shape (input_nodes, vector_dim) 
        """
        phase_values = self.lookup_table.lookup_phase(phases.int())
        mags_values = self.lookup_table.lookup_mag(mags.int())

        activations = (phase_values * mags_values).sum(dim=-1) #returns shape = (total_nodes, )


        #only propagate values from the nodes which are active
        active_nodes = torch.where(activations>self.activation_threshold)[0]

        if self.verbose:
            print(f"{len(active_nodes)} nodes active: ", end='')
            print(active_nodes)

        source_nodes = []
        target_nodes = []

        out_phases = phases.clone()
        out_mags = mags.clone()


        recieved_inputs = set()

        #this is through direct connections only. not radiation
        for node in active_nodes:
            outgoing_connections = self.connections[int(node)]['outgoing']
    
            recieved_inputs.update(outgoing_connections)

            
            out_phases[outgoing_connections] =  phase_forward(out_phases[outgoing_connections], phases[node], phase_bins=self.mag_bins)
            out_mags[outgoing_connections] =  mag_forward(out_mags[outgoing_connections], mags[node], mag_bins=self.mag_bins)

        

        out_phases1 = out_phases.clone()
        out_mags1 = out_mags.clone()

        
        radiation_targets = self._compute_radiation_targets(out_phases, self.radiation_targets)

        for node in active_nodes:

            
            current_targets = radiation_targets.get(node, None)
            if current_targets is None:
                continue

            out_phases1[current_targets] = phase_forward(out_phases1[current_targets], out_phases[node], phase_bins=self.phase_bins)
            out_mags1[current_targets] = mag_forward(out_mags1[current_targets], out_mags[node], mag_bins=self.mag_bins)



        if self.verbose:
            print(f"Recieved inputs by: {recieved_inputs}")

        return out_phases1, out_mags1

    def calculate_output_signal(self, phases, mags):
        """
        Calculates the output signal from the output nodes
        """
        output_phases = phases[self.output_nodeids]
        output_mags = mags[self.output_nodeids]

        #this takes care of returning gradients to indices from the actual vectors
        signal = signal_forward(output_phases, output_mags, self.lookup_table)

        return signal

    def forward(self, input_phases, input_mags):
        """
        input_phases: shape (input_nodes, vector_dim) 
        input_mags: shape (input_nodes, vector_dim)

        inputs are assumed to be quantized.
        """


        phases = self.phases 
        mags = self.mags

        
        out_phases = phases.clone() 
        out_mags = mags.clone() 

        if self.verbose:
            print(f"input_phases.shape: {input_phases.shape}")
            print(f"phases[self.input_nodeids].shape: {phases[list(self.input_nodeids)].shape}")

        #assign the values to out_phases because we need to keep the original values for gradient calculation
        out_phases[self.input_nodeids] = phase_forward(input_phases,phases[self.input_nodeids],self.phase_bins)
        out_mags[self.input_nodeids] = mag_forward(input_mags,mags[self.input_nodeids],self.mag_bins)


        for iteration in range(self.iterations):
            
            if self.verbose:
                print(f"Iteration {iteration}:")

            out_phases, out_mags = self.one_step_forward(out_phases, out_mags)

        activations = self.calculate_output_signal(out_phases, out_mags)
        return activations

