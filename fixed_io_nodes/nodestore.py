import numpy as np
import random
import time

from typing import List, Dict, Union, Set

import torch
import torch.nn as nn

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import Datatype
from qdrant_client import models

from lookup_table import LookupTable


class NodeStore(nn.Module):
    def __init__(
        self, 
        qdrant_url, 
        collection_name:str, 
        lookup_table:LookupTable,

        num_total_nodes:int, 
        num_input_nodes:int, 
        num_output_nodes:int, 
        cardinality:int, 
        vector_dim:int, 
        phase_bins:int, 
        mag_bins:int,

        m: int=16,
        ef_construct: int=100,
        deleted_threshold: float=0.05,
        vacuum_min_vector_number: int=1000,
        default_segment_number: int=0,
        max_segment_size_kb: int=None,
        memmap_threshold: int=20000,
        indexing_threshold_kb: int=20000,
        on_disk_payload: bool=True,
        distance_metric: Union[str, Distance]="Cosine",
    ):
        """
        Qdrant Parameters:

        qdrant_url: url of the qdrant server
        collection_name: name of the collection to be created in Qdrant

        Graph Parameters:
        num_total_nodes: total number of nodes in the graph
        num_input_nodes: number of input nodes in the graph
        num_output_nodes: number of output nodes in the graph
        cardinality: cardinality of the graph
        vector_dim: dimension of the vector
        phase_bins: number of bins for the phase
        mag_bins: number of bins for the mag

        HNSW Parameters:
        m: number of edges per node in the index graph. Larger the value - more accurate the search, more space required to store the index.
        ef_construct: number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.

        Optimizer Parameters:
        deleted_threshold: The number of vectors to be deleted from segment before vacuuming and reindexing the segment. Default: 0.05
        vacuum_min_vector_number: The minimal number of vectors in a segment required to run segment optimization. If a segment has less than this number of vectors, it is ignored by optimizer. Default: 1000
        default_segment_number: Target amount of segments optimizer will try to keep. If zero, it will be automatically selected by the number of available CPUs. Default: 0
        max_segment_size_kb: Segments will not exceed the size specified by this. Default None
        memmap_threshold: The maximum size (in kilobytes) of vectors stored in memory per segment. Vectors beyond this limit will be stored on disk. Default: 20000
        indexing_threshold_kb: Maximum size (in kilobytes) of vectors allowed for plain indexing. Exceeding this will enable vector indexing. Default: 20000, set to 0 for disabling vector indexing.
        on_disk_payload: Whether to store the payload on disk. Default: True
        distance_metric: Distance metric to be used for the collection. Options: "Cosine", "Euclidean", "Dot", "Manhattan". Default: "Cosine"
        """
        super().__init__()


        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.lookup_table = lookup_table

        self.total_nodes = num_total_nodes
        self.input_nodes = num_input_nodes
        self.output_nodes = num_output_nodes
        self.cardinality = cardinality
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.collection_name = collection_name


        #create collection and initialize graph
        if not self.client.collection_exists(self.collection_name):
            self._create_collection(
                collection_name=self.collection_name,
                distance_metric=distance_metric,
                on_disk_payload=on_disk_payload,
                m=m,
                ef_construct=ef_construct,
                deleted_threshold=deleted_threshold,
                vacuum_min_vector_number=vacuum_min_vector_number,
                default_segment_number=default_segment_number,
                max_segment_size_kb=max_segment_size_kb,
                memmap_threshold=memmap_threshold,
                indexing_threshold_kb=indexing_threshold_kb,    
            )

            self._initialize_graph(
                total_nodes=num_total_nodes,
                num_input_nodes=num_input_nodes,
                num_output_nodes=num_output_nodes,
                cardinality=cardinality,
            )
            print(f"Collection {self.collection_name} created and graph initialized")
        
        else: #if the collection exists, it is assumed that the graph is already initialized
            #however, we still to assign self.input_nodeids and self.output_nodeids
            self.input_nodeids = set(range(num_input_nodes))
            self.output_nodeids = set(range(num_total_nodes - num_output_nodes, num_total_nodes))

    def _initialize_nodeids(self,
        num_total_nodes:int,
        num_input_nodes:int,
        num_output_nodes:int,
        margin=0.1, #minimum % of total nodes other than input and output nodes
    ):
        #it can be assumed that all node ids can be stored in memory at once. (8GB needed for  2billion (2^31-1) node ids)
        node_ids =  list(range(num_total_nodes))

        #safety check that total node count > input + output node + margin (eg. 10%)
        if num_total_nodes - (num_input_nodes + num_output_nodes) < margin*num_total_nodes:
            raise ValueError("Total nodes must be greater than the sum of input and output nodes")
        self.input_nodeids = set(node_ids[:num_input_nodes])
        self.output_nodeids = set(node_ids[-num_output_nodes:])
        return node_ids

    def _initialize_connections(self, node_ids:Union[Set[str], List[str]], max_incoming_connections:int):
        """
        returns connections in following format:

        {node_id: {'incoming': List[str], 'outgoing': List[str]}}
        """
        
        #there is a possibility that the graph is not connected, so we need to implement a method to avoid that

        graph = {node_id: {'incoming': [], 'outgoing': []} for node_id in node_ids}
        node_ids = set(node_ids)

        #we iteratively initialize only the incoming connections
        for n in node_ids:
            
            #if current node is input node, then it has no incoming connections
            if n in self.input_nodeids: 
                continue
            
            #output nodes cannot be incoming nodes, and current node cannot be incoming to itself 
            possible_incoming_nodes = node_ids - self.output_nodeids - set([n]) 

            #it is ok to have 0 incoming connections, because node can be reached via radiation
            incoming_connection_count = random.randint(0, max_incoming_connections) 
            incoming_connections = random.choices(list(possible_incoming_nodes), k=incoming_connection_count)
            graph[n]['incoming'] = incoming_connections

            #add current node to the outgoing connections of the incoming nodes
            for incoming_connection in incoming_connections:
                graph[incoming_connection]['outgoing'].append(n)

        return graph
            
    def _initialize_phases(self, node_ids:Union[Set[str], List[str]], connections:Dict[str, Dict[str, List[str]]]):

        ### TODO: implement a method to initialize phases, such that neighbouring nodes have orthogonal phases
        ## challenge faced in this currently: the node values are discrete. also need to take that into account
        
        phases = {
            node_id: np.random.randint(0, self.phase_bins, (self.vector_dim), dtype=np.uint8).tolist() 
            for node_id in node_ids
        }

        return phases

    def _initialize_mags(self, node_ids:Union[Set[str], List[str]], connections:Dict[str, Dict[str, List[str]]]):

        mags = {
            node_id: np.random.randint(0, self.mag_bins, (self.vector_dim), dtype=np.uint8).tolist() 
            for node_id in node_ids
        }
        return mags
    
    def _initialize_graph(self,
        total_nodes,
        num_input_nodes,
        num_output_nodes,
        cardinality,
        seed=42,

        margin=0.1, #minimum % of nodes other than input and output nodes
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not self.client.collection_exists(self.collection_name):
            self._create_collection(self.collection_name)

        node_ids = self._initialize_nodeids(total_nodes, num_input_nodes, num_output_nodes, margin=margin)
        # self.node_ids = node_ids
    
        connections = self._initialize_connections(node_ids, cardinality)

        #vector dim,& phase/mag bins taken while initializing class
        phases = self._initialize_phases(node_ids, connections)
        mags = self._initialize_mags(node_ids, connections)
        phase_values = {}


        #Vector search is to be done on the phase vector with both sin and cos values
        #so we need to store the phase values in the database. 
        #creating a mapping of node_id -> phase_values here
        for n_id in phases:
            phase_indices = phases[n_id]
            cos_values = self.lookup_table.lookup_phase(phase_indices)
            sin_values = self.lookup_table.lookup_phase_sin(phase_indices)
            phase_values[n_id] = torch.cat([cos_values, sin_values], dim=-1)


        self._insert_graph(node_ids, connections, phases, phase_values, mags)
        

    def _insert_graph(self, 
        node_ids:Union[Set[int], List[int]], 
        connections:Dict[int, Dict[str, List[str]]], 
        phases:Dict[int, List[int]], 
        phase_values:Dict[int, List[float]],
        mags:Dict[int, List[int]],
        wait:bool=True,
    ):

        assert len(node_ids) == len(phases) == len(mags), "node_ids, phases, and mags must have the same length"

        return self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    payload={
                        'incoming_connections': connections[id]['incoming'],
                        'outgoing_connections': connections[id]['outgoing'],
                        'update_count': 0,
                        'activation_count': 0
                    },
                    vector={
                        #dtype based on number of bins
                        'phase': phases[id],
                        'phase_values': phase_values[id],
                        'mag': mags[id],
                    }
                ) for id in node_ids
            ],
            wait=wait,   #stops the process until the vectors are inserted
        )

    def _create_collection(self, collection_name,
        distance_metric:Union[str, Distance]=Distance.COSINE,
        on_disk_payload=True,

        m=16,
        ef_construct=100,

        deleted_threshold=0.05,
        vacuum_min_vector_number=1000,
        default_segment_number=0,
        max_segment_size_kb=None,
        memmap_threshold=20000,
        indexing_threshold_kb=20000,
    ):
        """
        collection_name: name of the collection to be created in Qdrant

        distance_metric: distance metric to be used for the collection. Options: DOT, EUCLID, COSINE, MANHATTAN. Default: COSINE

        HNSW Configs
        m: number of edges per node in the index graph. Larger the value - more accurate the search, more space required to store the index.
        ef_construct: number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.

        Optimizer Configs
        deleted_threshold: The number of vectors to be deleted from segment before vacuuming and reindexing the segment. Default: 0.05

        vacuum_min_vector_number: The minimal number of vectors in a segment required to run segment optimization. If a segment has less than this number of vectors, it is ignored by optimizer. Default: 1000


        default_segment_number: Target amount of segments optimizer will try to keep. If zero, it will be automatically selected by the number of available CPUs. Default: 0

        max_segment_size_kb: Segments will not exceed the size specified by this. Default None

        memmap_threshold: The maximum size (in kilobytes) of vectors stored in memory per segment. Vectors beyond this limit will be stored on disk. Default: 20000
        indexing_threshold_kb: Maximum size (in kilobytes) of vectors allowed for plain indexing. Exceeding this will enable vector indexing. Default: 20000, set to 0 for disabling vector indexing.

        
        """
        
        #nodeid is same as an "id" as represented internally in qdrant, which is indexed by default
        #no need to separately make an index for it
        

        if distance_metric.lower() not in ['cosine', 'euclid', 'dot', 'manhattan']:
            raise ValueError("Invalid distance metric. Please choose from: COSINE, EUCLID, DOT, MANHATTAN")
        distance_metric = Distance(distance_metric.capitalize())

        optimizer_config=models.OptimizersConfigDiff(
                    deleted_threshold=deleted_threshold,
                    vacuum_min_vector_number=vacuum_min_vector_number,
                    default_segment_number=default_segment_number,
                    max_segment_size=max_segment_size_kb,
                    memmap_threshold=memmap_threshold,
                    indexing_threshold=indexing_threshold_kb,
                    # flush_interval_sec=1, #TODO: check docs for this, I couldn't find on website
        )

        vector_config = {
            'phase_values': VectorParams(
                size=self.vector_dim*2, #stores values instead of indices for proper indexing
                distance=distance_metric,
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                )  
            ),
            "mag": VectorParams(
                size=self.vector_dim,
                distance=Distance.DOT,
                on_disk=True,
                datatype=Datatype.UINT8,
            ),
            "phase": VectorParams(
                size=self.vector_dim,
                distance=Distance.DOT,
                on_disk=True,
                datatype=Datatype.UINT8,
            )
        }


        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            optimizers_config=optimizer_config,
            on_disk_payload=on_disk_payload, #this makes the payload to be stored on disk, not RAM
        )

    def update_vectors(self, values:Dict[int, Dict[str, List[int]]]):
        """
        Update just the vectors for the given node ids
        Helps when doing gradient descent, where we only need to update the vectors, and not the payload
        
        """
        
        return self.client.update_vectors(
            collection_name=self.collection_name,
            points=[
                models.PointVectors(
                    id=node_id,
                    vector={
                        'phase': values[node_id]['phase'],
                        'mag': values[node_id]['mag'],
                        'phase_values': torch.cat([self.lookup_table.lookup_phase(values[node_id]['phase']), self.lookup_table.lookup_phase_sin(values[node_id]['phase'])], dim=-1).tolist(),
                    }
                ) for node_id in values
            ],
            wait=True,
        )

    #compatibility method
    def get_phase(self, node_ids:Union[str, List[str]]):
        if isinstance(node_ids, str):
            node_ids = [node_ids]

        vectors = []
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=node_ids,
            with_payload=False,
            with_vectors=['phase'],
        )
        for p in points:
            vectors.append(p.vector['phase'])
        return vectors
    
    #compatibility method
    def get_mag(self, node_ids:Union[int, List[int]]):
        if isinstance(node_ids, str):
            node_ids = [node_ids]

        vectors = []
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=node_ids,
            with_payload=False,
            with_vectors=['mag'],
        )

        for p in points:
            vectors.append(p.vector['mag'])
        return vectors


    def get_node(self, node_ids: Union[int, str, List[int], List[str]], with_payload:bool=True, with_vectors:bool=True):
        """
        node_ids: list of node ids to get (can be int, str, or list of either)
        with_payload: whether to include payload
        with_vectors: whether to include vectors

        returns: list of nodes with or without payload and vectors (as requested)
        """
        #TODO: make this return a list of Node objects instead of qdrant points
        # Convert to list if single value
        if not isinstance(node_ids, list):
            node_ids = [node_ids]
        
        # Convert all node IDs to integers (handles both int and str inputs)
        node_ids = [int(node_id) for node_id in node_ids]
        
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=node_ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
    
    
    def search_nodes(self, query_vector, vector_name='phase_values', limit=3, with_vectors=False, with_payload=True):
        #TODO: make this return a list of Node objects instead of qdrant points

        if vector_name == 'phase_values': #phase values means the first half of vector is cos, 2nd half is sin
            query_vector = torch.tensor(query_vector)
            query_vector[self.vector_dim:] = -query_vector[self.vector_dim:]
            query_vector = query_vector.tolist()

        elif vector_name == 'phase': #case where input is just the phase_indices, it converts it to phase_values
            query_vector = torch.concat([self.lookup_table.lookup_phase(query_vector), -self.lookup_table.lookup_phase_sin(query_vector)], dim=-1).tolist()
            vector_name = 'phase_values'
        else:
            raise NotImplementedError(f"Vector name: {vector_name} not implemented")


        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        ).points
    
    def search_nodes_batch(self, query_vectors, vector_name='phase_values', limit=3, with_vectors=False, with_payload=True):
        
        requests = []

        for q_vec in query_vectors:

            if not isinstance(q_vec, torch.Tensor):
                q_vec = torch.tensor(q_vec, device=self.lookup_table.device)
            
            if vector_name == 'phase':
                # Transform indices to Cos/Sin vectors (Conjugate logic)
                # We negate the Sin part for complex number rotation simulation (a * b* pattern)
                q_vec = torch.cat([
                    self.lookup_table.lookup_phase(q_vec), 
                    -self.lookup_table.lookup_phase_sin(q_vec)
                ], dim=-1).tolist()
                
                target_name = 'phase_values'
            else:
                raise NotImplementedError(f"Vector name: {vector_name} not implemented")


            requests.append(
                models.SearchRequest(
                    vector=models.NamedVector(name=target_name, vector=q_vec),
                    limit=limit,
                    with_payload=with_payload,
                    with_vector=with_vectors
                )
            )

        # Send 1 BIG request instead of N small ones
        search_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=requests
        )
        return search_results


    def is_input(self, node_id):
        return node_id in self.input_nodeids

    def is_output(self, node_id):
        return node_id in self.output_nodeids
    


def retry_on_connection_error(max_retries=5, base_delay=0.5):
    """Decorator to retry Qdrant operations on connection errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_connection_error = any(x in error_msg for x in ['connection', 'reset', 'refused', 'timeout'])
                    
                    if attempt < max_retries - 1 and is_connection_error:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        print(f"Qdrant connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator


class UnquantizedNodeStore(nn.Module):
    def __init__(
        self, 
        qdrant_url, 
        collection_name:str, 

        num_total_nodes:int, 
        num_input_nodes:int, 
        num_output_nodes:int, 
        cardinality:int, 
        vector_dim:int, 
        phase_bins:int=None,  # Legacy parameter, kept for compatibility but not used
        mag_bins:int=None,    # Legacy parameter, kept for compatibility but not used
        radiation_similarity_threshold: float=0.0,
        temporal_decay: float=1.0,

        m: int=16,
        ef_construct: int=100,
        deleted_threshold: float=0.05,
        vacuum_min_vector_number: int=1000,
        default_segment_number: int=0,
        max_segment_size_kb: int=None,
        memmap_threshold: int=20000,
        indexing_threshold_kb: int=20000,
        on_disk_payload: bool=True,
        distance_metric: Union[str, Distance]="Cosine",

        lookup_table=None,
    ):
        """
        Qdrant Parameters:

        qdrant_url: url of the qdrant server
        collection_name: name of the collection to be created in Qdrant

        Graph Parameters:
        num_total_nodes: total number of nodes in the graph
        num_input_nodes: number of input nodes in the graph
        num_output_nodes: number of output nodes in the graph
        cardinality: cardinality of the graph
        vector_dim: dimension of the vector
        phase_bins: (deprecated) kept for backward compatibility
        mag_bins: (deprecated) kept for backward compatibility

        HNSW Parameters:
        m: number of edges per node in the index graph. Larger the value - more accurate the search, more space required to store the index.
        ef_construct: number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.

        Optimizer Parameters:
        deleted_threshold: The number of vectors to be deleted from segment before vacuuming and reindexing the segment. Default: 0.05
        vacuum_min_vector_number: The minimal number of vectors in a segment required to run segment optimization. If a segment has less than this number of vectors, it is ignored by optimizer. Default: 1000
        default_segment_number: Target amount of segments optimizer will try to keep. If zero, it will be automatically selected by the number of available CPUs. Default: 0
        max_segment_size_kb: Segments will not exceed the size specified by this. Default None
        memmap_threshold: The maximum size (in kilobytes) of vectors stored in memory per segment. Vectors beyond this limit will be stored on disk. Default: 20000
        indexing_threshold_kb: Maximum size (in kilobytes) of vectors allowed for plain indexing. Exceeding this will enable vector indexing. Default: 20000, set to 0 for disabling vector indexing.
        on_disk_payload: Whether to store the payload on disk. Default: True
        distance_metric: Distance metric to be used for the collection. Options: "Cosine", "Euclidean", "Dot", "Manhattan". Default: "Cosine"
        """
        super().__init__()

        # Initialize Qdrant client with retry logic
        self.client = self._init_client_with_retry(qdrant_url)
        self.collection_name = collection_name

        self.total_nodes = num_total_nodes
        self.input_nodes = num_input_nodes
        self.output_nodes = num_output_nodes
        self.cardinality = cardinality
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins  # Kept for legacy compatibility
        self.mag_bins = mag_bins      # Kept for legacy compatibility
        self.collection_name = collection_name

        self.radiation_similarity_threshold = radiation_similarity_threshold
        self.temporal_decay = temporal_decay

        #create collection and initialize graph
        if not self._check_collection_exists_with_retry():
            self._create_collection(
                collection_name=self.collection_name,
                distance_metric=distance_metric,
                on_disk_payload=on_disk_payload,
                m=m,
                ef_construct=ef_construct,
                deleted_threshold=deleted_threshold,
                vacuum_min_vector_number=vacuum_min_vector_number,
                default_segment_number=default_segment_number,
                max_segment_size_kb=max_segment_size_kb,
                memmap_threshold=memmap_threshold,
                indexing_threshold_kb=indexing_threshold_kb,    
            )

            self._initialize_graph(
                total_nodes=num_total_nodes,
                num_input_nodes=num_input_nodes,
                num_output_nodes=num_output_nodes,
                cardinality=cardinality,
            )
        
        else: #if the collection exists, it is assumed that the graph is already initialized
            #however, we still to assign self.input_nodeids and self.output_nodeids
            self.input_nodeids = set(range(num_input_nodes))
            self.output_nodeids = set(range(num_total_nodes - num_output_nodes, num_total_nodes))

    @retry_on_connection_error(max_retries=5, base_delay=0.5)
    def _init_client_with_retry(self, qdrant_url):
        """Initialize Qdrant client with retry logic and gRPC optimization."""
        # Enable gRPC for better throughput
        # If url is http://localhost:6333, it will try to use gRPC on port 6334 by default if prefer_grpc=True
        return QdrantClient(url=qdrant_url, prefer_grpc=True, timeout=100)
    
    @retry_on_connection_error(max_retries=5, base_delay=0.5)
    def _check_collection_exists_with_retry(self):
        """Check if collection exists with retry logic."""
        return self.client.collection_exists(self.collection_name)

    def _initialize_nodeids(self,
        num_total_nodes:int,
        num_input_nodes:int,
        num_output_nodes:int,
        margin=0.1, #minimum % of total nodes other than input and output nodes
    ):
        #it can be assumed that all node ids can be stored in memory at once. (8GB needed for  2billion (2^31-1) node ids)
        node_ids =  list(range(num_total_nodes))

        #safety check that total node count > input + output node + margin (eg. 10%)
        if num_total_nodes - (num_input_nodes + num_output_nodes) < margin*num_total_nodes:
            raise ValueError("Total nodes must be greater than the sum of input and output nodes")
        self.input_nodeids = set(node_ids[:num_input_nodes])
        self.output_nodeids = set(node_ids[-num_output_nodes:])
        return node_ids

    def _initialize_connections(self, node_ids:Union[Set[str], List[str]], max_incoming_connections:int):
        """
        returns connections in following format:

        {node_id: {'incoming': List[str], 'outgoing': List[str]}}
        """
        
        #there is a possibility that the graph is not connected, so we need to implement a method to avoid that

        graph = {node_id: {'incoming': [], 'outgoing': []} for node_id in node_ids}
        node_ids = set(node_ids)

        #we iteratively initialize only the incoming connections
        for n in node_ids:
            
            #if current node is input node, then it has no incoming connections
            if n in self.input_nodeids: 
                continue
            
            #output nodes cannot be incoming nodes, and current node cannot be incoming to itself 
            possible_incoming_nodes = node_ids - self.output_nodeids - set([n]) 

            #it is ok to have 0 incoming connections, because node can be reached via radiation
            incoming_connection_count = random.randint(0, max_incoming_connections) 
            incoming_connections = random.choices(list(possible_incoming_nodes), k=incoming_connection_count)
            graph[n]['incoming'] = incoming_connections

            #add current node to the outgoing connections of the incoming nodes
            for incoming_connection in incoming_connections:
                graph[incoming_connection]['outgoing'].append(n)

        return graph
            
    def _initialize_phases(self, node_ids:Union[Set[str], List[str]], connections:Dict[str, Dict[str, List[str]]]):

        ### Initialize phases as continuous values in [0, 2π]
        phases = {
            node_id: (np.random.uniform(0, 2*np.pi, (self.vector_dim))).astype(np.float32).tolist() 
            for node_id in node_ids
        }

        return phases

    def _initialize_mags(self, node_ids:Union[Set[str], List[str]], connections:Dict[str, Dict[str, List[str]]]):

        ### Initialize magnitudes as continuous values in [-π, π]
        mags = {
            node_id: (np.random.uniform(-np.pi, np.pi, (self.vector_dim))).astype(np.float32).tolist() 
            for node_id in node_ids
        }
        return mags
    
    def _initialize_graph(self,
        total_nodes,
        num_input_nodes,
        num_output_nodes,
        cardinality,
        seed=42,

        margin=0.1, #minimum % of nodes other than input and output nodes
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not self.client.collection_exists(self.collection_name):
            self._create_collection(self.collection_name)

        node_ids = self._initialize_nodeids(total_nodes, num_input_nodes, num_output_nodes, margin=margin)
        # self.node_ids = node_ids
    
        connections = self._initialize_connections(node_ids, cardinality)

        #vector dim,& phase/mag bins taken while initializing class
        phases = self._initialize_phases(node_ids, connections)
        mags = self._initialize_mags(node_ids, connections)
        phase_values = {}


        #Vector search is to be done on the phase vector with both sin and cos values
        #so we need to store the phase values in the database. 
        #creating a mapping of node_id -> phase_values here
        for n_id in phases:
            phase_continuous = torch.tensor(phases[n_id], dtype=torch.float32)
            cos_values = torch.cos(phase_continuous)
            sin_values = torch.sin(phase_continuous)
            phase_values[n_id] = torch.cat([cos_values, sin_values], dim=-1)


        self._insert_graph(node_ids, connections, phases, phase_values, mags)
        

    def _insert_graph(self, 
        node_ids:Union[Set[int], List[int]], 
        connections:Dict[int, Dict[str, List[str]]], 
        phases:Dict[int, List[int]], 
        phase_values:Dict[int, List[float]],
        mags:Dict[int, List[int]],
        wait:bool=True,
    ):

        assert len(node_ids) == len(phases) == len(mags), "node_ids, phases, and mags must have the same length"

        return self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    payload={
                        'incoming_connections': connections[id]['incoming'],
                        'outgoing_connections': connections[id]['outgoing'],
                        'version': 0  # Initialize version counter for tracking updates
                    },
                    vector={
                        #dtype based on number of bins
                        'phase': phases[id],
                        'phase_values': phase_values[id],
                        'mag': mags[id],
                    }
                ) for id in node_ids
            ],
            wait=wait,   #stops the process until the vectors are inserted
        )

    def _create_collection(self, collection_name,
        distance_metric:Union[str, Distance]=Distance.COSINE,
        datatype:str='float16',
        on_disk_payload=True,

        m=16,
        ef_construct=100,

        deleted_threshold=0.05,
        vacuum_min_vector_number=1000,
        default_segment_number=0,
        max_segment_size_kb=None,
        memmap_threshold=20000,
        indexing_threshold_kb=20000,
    ):
        """
        collection_name: name of the collection to be created in Qdrant

        distance_metric: distance metric to be used for the collection. Options: DOT, EUCLID, COSINE, MANHATTAN. Default: COSINE

        HNSW Configs
        m: number of edges per node in the index graph. Larger the value - more accurate the search, more space required to store the index.
        ef_construct: number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index.

        Optimizer Configs
        deleted_threshold: The number of vectors to be deleted from segment before vacuuming and reindexing the segment. Default: 0.05

        vacuum_min_vector_number: The minimal number of vectors in a segment required to run segment optimization. If a segment has less than this number of vectors, it is ignored by optimizer. Default: 1000


        default_segment_number: Target amount of segments optimizer will try to keep. If zero, it will be automatically selected by the number of available CPUs. Default: 0

        max_segment_size_kb: Segments will not exceed the size specified by this. Default None

        memmap_threshold: The maximum size (in kilobytes) of vectors stored in memory per segment. Vectors beyond this limit will be stored on disk. Default: 20000
        indexing_threshold_kb: Maximum size (in kilobytes) of vectors allowed for plain indexing. Exceeding this will enable vector indexing. Default: 20000, set to 0 for disabling vector indexing.

        
        """
        
        #nodeid is same as an "id" as represented internally in qdrant, which is indexed by default
        #no need to separately make an index for it

        if datatype not in ['float16', 'float32']:
            raise ValueError("Invalid datatype. Please choose from: float16, float32")

        datatype = Datatype.FLOAT16 if datatype == 'float16' else Datatype.FLOAT32
        

        if distance_metric.lower() not in ['cosine', 'euclid', 'dot', 'manhattan']:
            raise ValueError("Invalid distance metric. Please choose from: COSINE, EUCLID, DOT, MANHATTAN")
        distance_metric = Distance(distance_metric.capitalize())

        optimizer_config=models.OptimizersConfigDiff(
                    deleted_threshold=deleted_threshold,
                    vacuum_min_vector_number=vacuum_min_vector_number,
                    default_segment_number=default_segment_number,
                    max_segment_size=max_segment_size_kb,
                    memmap_threshold=memmap_threshold,
                    indexing_threshold=indexing_threshold_kb,
                    # flush_interval_sec=1, #TODO: check docs for this, I couldn't find on website
        )

        vector_config = {
            'phase_values': VectorParams(
                size=self.vector_dim*2, #stores cos/sin values for vector search
                distance=distance_metric,
                hnsw_config=models.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                )  
            ),
            "mag": VectorParams(
                size=self.vector_dim,
                distance=Distance.DOT,
                on_disk=True,
                datatype=datatype,
            ),
            "phase": VectorParams(
                size=self.vector_dim,
                distance=Distance.DOT,
                on_disk=True,
                datatype=datatype,
            )
        }


        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            optimizers_config=optimizer_config,
            on_disk_payload=on_disk_payload, #this makes the payload to be stored on disk, not RAM
        )

    def update_vectors(self, values:Dict[int, Dict[str, List[float]]]):
        """
        Update just the vectors for the given node ids
        Helps when doing gradient descent, where we only need to update the vectors, and not the payload
        
        """
        points = [
            models.PointVectors(
                id=node_id,
                vector={
                    'phase': values[node_id]['phase'],
                    'mag': values[node_id]['mag'],
                    'phase_values': torch.cat([
                        torch.cos(torch.tensor(values[node_id]['phase'])), 
                        torch.sin(torch.tensor(values[node_id]['phase']))
                    ], dim=-1).tolist(),
                }
            ) for node_id in values
        ]
        return self._update_vectors_with_retry(points)
    
    @retry_on_connection_error(max_retries=3, base_delay=0.2)
    def _update_vectors_with_retry(self, points):
        """Update vectors with retry logic."""
        return self.client.update_vectors(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def get_node(self, node_ids: Union[int, str, List[int], List[str]], with_payload:bool=True, with_vectors:bool=True):
        """
        node_ids: list of node ids to get (can be int, str, or list of either)
        with_payload: whether to include payload
        with_vectors: whether to include vectors

        returns: list of nodes with or without payload and vectors (as requested)
        """
        #TODO: make this return a list of Node objects instead of qdrant points
        # Convert to list if single value
        if isinstance(node_ids, int) or isinstance(node_ids, str):
            node_ids = [node_ids]
        if isinstance(node_ids, set):
            node_ids = list(node_ids)
        
        # Convert all node IDs to integers (handles both int and str inputs)
        node_ids = [int(node_id) for node_id in node_ids]
        
        return self._retrieve_with_retry(
            ids=node_ids,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
    
    @retry_on_connection_error(max_retries=3, base_delay=0.2)
    def _retrieve_with_retry(self, ids, with_payload, with_vectors):
        """Retrieve points with retry logic."""
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
    
    
    def search_nodes(self, query_vector, vector_name='phase_values', limit=3, with_vectors=False, with_payload=True):

        if vector_name == 'phase_values': #phase values means the first half of vector is cos, 2nd half is sin
            query_vector = torch.tensor(query_vector)
            query_vector[self.vector_dim:] = -query_vector[self.vector_dim:]
            query_vector = query_vector.tolist()

        elif vector_name == 'phase': #case where input is continuous phase values, convert to phase_values
            query_vector = torch.cat([torch.cos(query_vector), -torch.sin(query_vector)], dim=-1).tolist()
            vector_name = 'phase_values'
        else:
            raise NotImplementedError(f"Vector name: {vector_name} not implemented")


        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=with_payload,
        ).points

        points = [[p for p in point_list if p.score >= self.radiation_similarity_threshold] for point_list in points]
        return points


    
    def search_nodes_batch(self, query_vectors, vector_name='phase', limit=3, with_vectors=False, with_payload=True):
        """        
        Search for nearest neighbors for multiple query vectors.
        Uses query_batch_points for high throughput.

        query_vectors: list of query vectors
        vector_name: the name of the vecotrs provided in query_vectors. Currently only phase is supported.
        Phase means the continuous phase values in [0, 2π] of the phase vector.
        """
        requests = []

        for q_vec in query_vectors:

            if not isinstance(q_vec, torch.Tensor):
                q_vec = torch.tensor(q_vec)
            
            if vector_name == 'phase':
                # Transform continuous phase values to Cos/Sin vectors (Conjugate logic)
                # We negate the Sin part for complex number rotation simulation (a * b* pattern)
                q_vec = torch.cat([
                    torch.cos(q_vec), 
                    -torch.sin(q_vec)
                ], dim=-1).tolist()
                
                target_name = 'phase_values'
            else:
                raise NotImplementedError(f"Vector name: {vector_name} not implemented")

            requests.append(
                models.QueryRequest(
                    query=q_vec,
                    using=target_name,
                    limit=limit,
                    with_payload=with_payload,
                    with_vector=with_vectors
                )
            )

        # Send 1 BIG request instead of N small ones
        batch_results = self._query_batch_points_with_retry(
            requests=requests
        )
        
        # Extract points from QueryResponse objects
        points = [result.points for result in batch_results]
        
        points = [[p for p in point_list if p.score >= self.radiation_similarity_threshold] for point_list in points]
        return points
    
    @retry_on_connection_error(max_retries=3, base_delay=0.2)
    def _query_batch_points_with_retry(self, requests):
        """Query batch points with retry logic."""
        return self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests
        )


    def is_input(self, node_id):
        return node_id in self.input_nodeids

    def is_output(self, node_id):
        return node_id in self.output_nodeids
    
    def get_node_versions(self, node_ids: Union[int, List[int]]):
        """
        Retrieve only the version payload for given node IDs.
        Efficient for checking if weights need to be updated.
        
        Returns: Dict[node_id, version]
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        points = self._retrieve_with_retry(
            ids=node_ids,
            with_payload=['version'],
            with_vectors=False
        )
        
        return {p.id: p.payload.get('version', 0) for p in points}
    
    def update_node_versions(self, node_ids: Union[int, List[int]], versions: Union[int, List[int]]):
        """
        Update version numbers for given node IDs.
        
        node_ids: single ID or list of IDs
        versions: single version or list of versions (must match node_ids length)
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            versions = [versions]
        
        if len(node_ids) != len(versions):
            raise ValueError("node_ids and versions must have the same length")
        
        # Update payload for each node
        for node_id, version in zip(node_ids, versions):
            self._set_payload_with_retry(
                node_id=node_id,
                payload={'version': version}
            )
    
    @retry_on_connection_error(max_retries=3, base_delay=0.2)
    def _set_payload_with_retry(self, node_id, payload):
        """Set payload for a single node with retry logic."""
        return self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[node_id],
            wait=True
        )


NodeStore = UnquantizedNodeStore