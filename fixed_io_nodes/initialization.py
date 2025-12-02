import torch
import random


def initialize_nodeids(total_nodes, input_nodes, output_nodes):
    #no check to make sure input+output < total
    nodeids = list(range(total_nodes))
    return nodeids, nodeids[:input_nodes], nodeids[-output_nodes:]

def initialize_phases(vector_dim, phase_bins, total_nodes):
    phases = torch.randint(0, phase_bins, size=(total_nodes, vector_dim))
    return phases

def initialize_mags(vector_dim, mag_bins, total_nodes):
    mags = torch.randint(0, mag_bins, size=(total_nodes, vector_dim))
    return mags

def initialize_connections(node_ids, input_nodeids, output_nodeids, cardinality):
    
    graph = {node_id: {'incoming': [], 'outgoing': [], 'update_count':0, 'activation_count':0} for node_id in node_ids}
    node_ids = set(node_ids)

    output_nodeids = set(output_nodeids)
    input_nodeids = set(input_nodeids)

    #we iteratively initialize only the incoming connections
    for n in node_ids:
        
        #if current node is input node, then it has no incoming connections
        if n in input_nodeids: 
            continue

        #output nodes cannot be incoming nodes, and current node cannot be incoming to itself 
        possible_incoming_nodes = node_ids - set(output_nodeids) - set([n]) 

        #it is ok to have 0 incoming connections, because node can be reached via radiation
        incoming_connection_count = random.randint(0, cardinality) 
        incoming_connections = random.choices(list(possible_incoming_nodes), k=incoming_connection_count)
        graph[n]['incoming'] = incoming_connections

        #add current node to the outgoing connections of the incoming nodes
        for incoming_connection in incoming_connections:
            graph[incoming_connection]['outgoing'].append(n)

    return graph

def initialize_graph(total_nodes, input_nodes, output_nodes, cardinality, vector_dim, phase_bins, mag_bins):
    """


    RETURNS:
    node_ids: list of node ids
    input_nodeids: list of input node ids
    output_nodeids: list of output node ids
    phases: numpy array of shape (total_nodes, vector_dim)
    mags: numpy array of shape (total_nodes, vector_dim)
    graph: dictionary of the form {node_id: {'incoming': list of incoming node ids, 'outgoing': list of outgoing node ids, 'update_count': int, 'activation_count': int}}
    """
    node_ids, input_nodeids, output_nodeids = initialize_nodeids(total_nodes, input_nodes, output_nodes)
    phases = initialize_phases(vector_dim, phase_bins, total_nodes)
    mags = initialize_mags(vector_dim, mag_bins, total_nodes)
    graph = initialize_connections(node_ids, input_nodeids, output_nodeids, cardinality)

    for node_id in node_ids:
        graph[node_id]['update_count'] = 0
        graph[node_id]['activation_count'] = 0

    return node_ids, input_nodeids, output_nodeids, phases, mags, graph

