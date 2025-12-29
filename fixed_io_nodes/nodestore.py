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


    def get_node(self, node_ids: Union[int, List[int]], with_payload:bool=True, with_vectors:bool=True):
        """
        node_ids: list of node ids to get
        with_payload: whether to include payload
        with_vectors: whether to include vectors

        returns: list of nodes with or without payload and vectors (as requested)
        """
        #TODO: make this return a list of Node objects instead of qdrant points
        if isinstance(node_ids, int):
            node_ids = [node_ids]
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
    
