from torch import nn
import torch
from .custom_functions import activation_strength_forward, signal_forward

from typing import List, Union, Optional

from .nodestore import NodeStore
from .node import Node

class MyModuleDict(nn.ModuleDict):
    def __getitem__(self, key: Union[str, int]) -> nn.Module:
        if isinstance(key, int):
            key = str(key)
        return super().__getitem__(key)
    def __setitem__(self, key: Union[str, int], value: nn.Module) -> None:
        if isinstance(key, int):
            key = str(key)
        return super().__setitem__(key, value)



class UnquantizedGNN(nn.Module):

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
        temporal_decay:float=1.0,
        device:str='cuda' if torch.cuda.is_available() else 'cpu',
        verbose:bool=False,
    ):
        super().__init__()

        self.device = device
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
        self.gamma = gamma
        self.temporal_decay = temporal_decay
        self.verbose = verbose

        #input nodes are considered active from the start
        self.active_nodes = {
            n_id: Node(
                node_store=self.node_store, 
                gamma=gamma,
                node_id=n_id,
                device=self.device,
                ) for n_id in self.node_store.input_nodeids
        }
        self.input_nodes = self.active_nodes.copy()
        node_values = self.node_store.get_node(self.node_store.input_nodeids)
        for n_value in node_values:
            self.active_nodes[n_value.id].load_values(n_value)

        self.active_nodes = MyModuleDict(self.active_nodes)
        
        # Cache for version tracking - stores nodes across resets
        self.node_cache = {}
        for n_id, node in self.input_nodes.items():
            self.node_cache[n_id] = node
        
        # Counter for periodic cache cleanup
        self._forward_pass_count = 0
        self._cache_cleanup_interval = 100  # Clean cache every 100 forward passes

        
        self.output_nodeids = self.node_store.output_nodeids
        self.input_nodeids = self.node_store.input_nodeids


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

        
        query_vectors = [node.phase_activation for node in nodes]
        batch_results = self.node_store.search_nodes_batch(
            query_vectors, 
            vector_name='phase', 
            limit=k,
            with_payload=False, 
            with_vectors=False
        )

        # Map results back to the corresponding node IDs
        for i, node in enumerate(nodes):
            topk_indices[node.id] = [found_point.id for found_point in batch_results[i]]
        
        return topk_indices

    

    def one_step_forward(self, input_values:torch.Tensor=None, tracer=None):
        """
        input_values: torch.Tensor=None, shape = (input_nodes, vector_dim)
        if input is given, then it is inserted into the input nodes, otherwise a normal 
        1 step propagation is done. Input values are assumed to be quantized.
        tracer: Optional[ForwardPassTracer] = None, tracer for capturing forward pass details

        Returns: None
        """

        if self.verbose: #TODO: print logs
            pass
        
        # Determine iteration number for tracer
        if tracer is not None:
            iteration_num = tracer.get_num_iterations() if input_values is None else 0
            tracer.start_iteration(iteration_num, input_injected=(input_values is not None))

        #TODO:
        #fetch input nodes which are not exisiting in active_nodes 
        # (This is needed in case of applying beam width)


        
        #input data is fed into the input nodes before propagation of remaining network
        if input_values is not None:

            assert input_values.shape == (self.input_node_count, self.vector_dim), f"expected input_values of shape ({self.input_node_count}, {self.vector_dim}), but got {input_values.shape}"

            # Input values are continuous phase values (in radians)
            input_phases = input_values
            input_mags = torch.zeros((self.input_node_count, self.vector_dim), device=input_values.device) 
            
            # Calculate activation strengths for all input nodes
            activation_strengths = torch.zeros(self.input_node_count, device=input_values.device)
            for i in range(self.input_node_count):
                activation_strengths[i] = activation_strength_forward(input_phases[i], input_mags[i], self.gamma)

            #inject input values into input nodes    
            for n_id, node in self.input_nodes.items():
                node.update_activations(
                    phase_activations=input_phases[n_id],
                    mag_activations=input_mags[n_id],
                    activation_strengths=activation_strengths[n_id],
                )
                
                # Record input injection for tracer
                if tracer is not None:
                    tracer.record_node_update(
                        node_id=node.id,
                        input_sources=[],  # Input nodes have no input sources
                        input_types=[],
                        phase_activation=node.phase_activation,
                        mag_activation=node.mag_activation,
                        activation_strength=node.activation_strength,
                    )

            



        #fetch radiation targets 
        radiation_targets = self._compute_radiation_targets(set(self.active_nodes.values()))
        
        if tracer is not None:
            tracer.record_radiation_targets(radiation_targets)
        
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
        new_nodes = [Node(node_store=self.node_store, gamma=self.gamma, device=self.device) for _ in new_nodes_values]
        
        for i, new_node_value in enumerate(new_nodes_values):
            new_nodes[i].load_values(new_node_value) #loads the values into the Node objects
        
        new_nodes = set(new_nodes)

    
        #the incoming connections coming to new nodes from the current active nodes
        incoming_connections = {node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)}
        # Track connection types separately for tracer
        incoming_connection_types = {node.id: [] for node in set(self.active_nodes.values()).union(new_nodes)}
        direct_connections_dict = {}  # source -> list of targets

        for node in self.active_nodes.values():
            for n_id in node.outgoing_connections:
                incoming_connections[n_id].append(node.id)
                incoming_connection_types[n_id].append('direct')
                if node.id not in direct_connections_dict:
                    direct_connections_dict[node.id] = []
                direct_connections_dict[node.id].append(n_id)
        
        for n_id, targets in radiation_targets.items():
            for target in targets:
                incoming_connections[target].append(n_id)
                incoming_connection_types[target].append('radiation')
        
        if tracer is not None:
            tracer.record_direct_connections(direct_connections_dict)

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
            
            # Record node update for tracer
            if tracer is not None:
                input_sources = incoming_connections[node.id]
                input_types = incoming_connection_types[node.id]
                tracer.record_node_update(
                    node_id=node.id,
                    input_sources=input_sources,
                    input_types=input_types,
                    phase_activation=node.phase_activation,
                    mag_activation=node.mag_activation,
                    activation_strength=node.activation_strength,
                )
            # print("updated node-", node.id)

        #update the active nodes to include the new nodes
        for node in new_nodes:
            self.active_nodes[node.id] = node

        # Record all active nodes for tracer
        if tracer is not None:
            all_active_node_ids = [int(node_id) for node_id in self.active_nodes.keys()]
            tracer.record_active_nodes(all_active_node_ids)

        # Apply temporal decay to all active nodes (except input nodes on first step)
        # This makes older activations weaker than recent ones
        if input_values is None:  # Only decay when not receiving new input
            for node in self.active_nodes.values():
                node.decay_activations(self.temporal_decay)
        
        # Explicitly delete temporary dictionaries to prevent memory leaks
        del phase_activations, mag_activations, activation_strengths, incoming_connections, incoming_connection_types
        del new_nodes, new_nodes_values, radiation_targets


    def forward(self, input_values:torch.Tensor=None, tracer=None):

        self.one_step_forward(input_values, tracer=tracer)
        # if self.verbose:
        #     print(f"first pass done")

        for iteration in range(self.iterations-1):
            self.one_step_forward(tracer=tracer)
            # if self.verbose:
            #     print(f"Iteration {iteration+2} done")

        output_signals = {}
        inactive_output_nodes = []
        for node_id in self.output_nodeids:
            node_id = str(node_id)
            if node_id in self.active_nodes:
                output_signals[node_id] = self.active_nodes[node_id].activation_strength
            else:
                node_value = self.node_store.get_node(node_id)[0]
                node = Node(node_store=self.node_store, gamma=self.gamma, device=self.device)
                node.load_values(node_value)
                output_signals[node_id] = node.activation_strength
                self.active_nodes[node_id] = node
                inactive_output_nodes.append(node_id)
        
        print("inactive_output_nodes: ", inactive_output_nodes)
        
        output_signals = torch.stack([v for k, v in sorted(output_signals.items())])
        output_signals = output_signals / self.vector_dim ** 0.5 #TODO: check if needed
        
        # Periodic cache cleanup to prevent memory leaks
        self._forward_pass_count += 1
        if self._forward_pass_count >= self._cache_cleanup_interval:
            self._cleanup_node_cache()
            self._forward_pass_count = 0
        
        # print("active_nodes: ", {i:[j.phase_activation, j.mag_activation] for i, j in self.active_nodes.items()})
        # print("Activation strengths: ",)
        # for i,j in self.active_nodes.items():
        #     print(i, j.activation_strength)
        # print("output_signals: ", output_signals)
        
        return output_signals

    def sync_weights(self):
        """
        Check versions and fetch only updated weights from DB.
        Called BEFORE forward pass to ensure latest weights are used.
        """
        # Smart version check: only fetch nodes that have been updated
        input_ids = list(self.input_nodeids)
        
        # Get current versions from DB
        db_versions = self.node_store.get_node_versions(input_ids)
        
        # Determine which nodes need updating
        ids_to_fetch = []
        for node_id in input_ids:
            # Fetch if: not in cache OR version mismatch
            if node_id not in self.node_cache or self.node_cache[node_id].version != db_versions.get(node_id, 0):
                ids_to_fetch.append(node_id)
        
        # Fetch and update only changed nodes
        if ids_to_fetch:
            node_values = self.node_store.get_node(ids_to_fetch)
            for n_value in node_values:
                if n_value.id in self.node_cache:
                    # Update existing node
                    self.node_cache[n_value.id].load_values(n_value)
                else:
                    # Create new node (shouldn't happen for input nodes, but handle gracefully)
                    new_node = Node(
                        node_store=self.node_store,
                        gamma=self.gamma,
                        device=self.device
                    )
                    new_node.load_values(n_value)
                    self.node_cache[n_value.id] = new_node
            
            # Update input_nodes from cache
            self.input_nodes = {n_id: self.node_cache[n_id] for n_id in input_ids}
    
    def _cleanup_node_cache(self):
        """
        Cleanup node_cache to prevent memory leaks.
        Keeps only input nodes in the cache and removes any others that may have accumulated.
        """
        # Get current cache size for logging
        cache_size_before = len(self.node_cache)
        
        # Keep only input nodes
        input_ids = set(self.input_nodeids)
        nodes_to_remove = [nid for nid in self.node_cache.keys() if nid not in input_ids]
        
        for nid in nodes_to_remove:
            del self.node_cache[nid]
        
        if len(nodes_to_remove) > 0 and self.verbose:
            print(f"GNN: Cleaned node_cache: {cache_size_before} -> {len(self.node_cache)} nodes (removed {len(nodes_to_remove)})")
    
    def reset_activations(self):
        """
        Reset activations to initial state.
        Called AFTER forward pass completes to prepare for next sample.
        Weights are NOT fetched - they're kept as-is.
        """
        # Clear active nodes completely to free memory from non-input nodes
        self.active_nodes.clear()
        
        # Reset active nodes to input nodes only
        self.active_nodes = MyModuleDict(self.input_nodes.copy())
        
        # Reset each node's activations (not weights)
        for _, n in self.active_nodes.items():
            n.reset()
    
    def reset(self, fetch_weights:bool=True):
        """
        Full reset (backward compatibility).
        Reset the model to initial state. 
        1) Optionally syncs weights from DB (if fetch_weights=True)
        2) Resets activations to initial state
        """
        if fetch_weights:
            self.sync_weights()
        self.reset_activations()

        
        
    def get_grads(self):
        """
        Returns:
        phase_grads: Dict[int, torch.Tensor]
        mag_grads: Dict[int, torch.Tensor]
        """
        phase_grads = {}
        mag_grads = {}
        for node_id, node in self.active_nodes.items():
            node_id = int(node_id)
            phase_grads[node_id] = node.phase_weight.grad
            mag_grads[node_id] = node.mag_weight.grad
        return phase_grads, mag_grads




class QuantizedGNN(nn.Module):

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

        self.device = device
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

        self.lookup_table = node_store.lookup_table

        #input nodes are considered active from the start
        self.active_nodes = {
            n_id: Node(
                node_store=self.node_store, 
                lookup_table=self.lookup_table,
                node_id=n_id,
                device=self.device,
                ) for n_id in self.node_store.input_nodeids
        }
        self.input_nodes = self.active_nodes.copy()
        node_values = self.node_store.get_node(self.node_store.input_nodeids)
        for n_value in node_values:
            self.active_nodes[n_value.id].load_values(n_value)

        self.active_nodes = MyModuleDict(self.active_nodes)
        

        
        self.output_nodeids = self.node_store.output_nodeids
        self.input_nodeids = self.node_store.input_nodeids


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

        
        query_vectors = [node.phase_activation for node in nodes]
        batch_results = self.node_store.search_nodes_batch(
            query_vectors, 
            vector_name='phase', 
            limit=k,
            with_payload=False, 
            with_vectors=False
        )

        # Map results back to the corresponding node IDs
        for i, node in enumerate(nodes):
            topk_indices[node.id] = [found_point.id for found_point in batch_results[i]]
        
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
        new_nodes = [Node(node_store=self.node_store, lookup_table=self.lookup_table, device=self.device) for _ in new_nodes_values]
        
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
        # if self.verbose:
        #     print(f"first pass done")

        for iteration in range(self.iterations-1):
            self.one_step_forward()
            # if self.verbose:
            #     print(f"Iteration {iteration+2} done")

        output_signals = {}
        
        for node_id in self.output_nodeids:
            node_id = str(node_id)
            if node_id in self.active_nodes:
                output_signals[node_id] = self.active_nodes[node_id].activation_strength
            else:
                # print("Node not active: ", node_id)
                output_signals[node_id] = torch.tensor(-torch.inf, device=self.device).requires_grad_(True)
        
        output_signals = torch.stack([v for k, v in sorted(output_signals.items())])
        output_signals = output_signals / self.vector_dim ** 0.5 #TODO: check if needed
        return output_signals

    def reset(self, fetch_weights:bool=True):
        """
        Reset the model to initial state. 
        1) Resets the active nodes to input nodes. (this is enough to consider the model as reset)
        2) If fetch_weights is True, fetches the input node weights from qdrant, otherwise uses the same weights
        """

        if fetch_weights:
            node_values = self.node_store.get_node(self.input_nodeids)
            for n_value in node_values:
                self.input_nodes[n_value.id].load_values(n_value)


        self.active_nodes = MyModuleDict(self.input_nodes.copy())
        for _, n in self.active_nodes.items():
            n.reset()

        
        
    def get_grads(self):
        """
        Returns:
        phase_grads: Dict[int, torch.Tensor]
        mag_grads: Dict[int, torch.Tensor]
        """
        phase_grads = {}
        mag_grads = {}
        for node_id, node in self.active_nodes.items():
            node_id = int(node_id)
            phase_grads[node_id] = node.phase_weight.grad
            mag_grads[node_id] = node.mag_weight.grad
        return phase_grads, mag_grads

GNN = UnquantizedGNN
