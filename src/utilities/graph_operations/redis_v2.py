import numpy as np
from config import CONTEXT_DIMENSION, collection, key_collection, CONEXT_INERTIA
import random


'''
Decoupled update and context vector 
inertia of added to context update (gated unit)

'''


def find_word(word,sample='all'):
    #connections = list(collection.scan_iter(match=word + ':*'))
    connections   =key_collection.smembers(word)
    neighbours = []
    if sample == 'all':
        pass
    else :
        connection_count = len(connections)
        sampling_size = min(sample, connection_count)
        connections = random.sample(connections, sampling_size)

    for con in connections:
        connection_key = bytes(word,'utf-8') + b':' + con
        connection               = con.decode("utf-8")
        connection_properties    = collection.hgetall(connection_key)
        update_count             = int(connection_properties[b'update_count'])
        context                  = np.frombuffer(connection_properties[b'context'])
        update_vector            = np.frombuffer(connection_properties[b'update_vector'])
        weight_vector            = np.frombuffer(connection_properties[b'weight_vector'])

        neighbours.append({'connection_key':connection_key, 'connection' : connection ,\
                           'update_count':update_count,'context':context ,\
                           'update_vector':update_vector , 'weihgt_vector': weight_vector})
    return neighbours


def touch_connection_db(word_1, word_2):
    connection_key = word_1 + ':' + word_2
    check_context = collection.hget(connection_key,'context')
    if check_context is None:
        print('connection {} {} is new'.format(word_1,word_2))
        neighbours = find_word(word_1)
        if len(neighbours) < 1:
            context_vector = np.random.rand(CONTEXT_DIMENSION)
        # To efficiently use context dimensions spereding the conntext vectors (neighbour aware context initilization)
        else:
            average_context = 0
            for connection in neighbours:
                average_context += np.array(connection['context'])
            context_vector = -1 * average_context

        context_vector = context_vector - context_vector.mean()
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        unit_update_vector       = orhto(unit_context_vector)
        weight_vector       =    np.random.rand(CONTEXT_DIMENSION)
        unit_weight_vector = weight_vector / np.linalg.norm(weight_vector)




        connection = {'connection_key': connection_key ,'context': unit_context_vector,
                      'update_count': 0 , 'update_vector': unit_update_vector, 'weight_vector':unit_weight_vector }

        collection.hset( connection_key,mapping ={'context': unit_context_vector.tobytes(),
                      'update_count': 0,
                      'lock': 0,\
                      'update_vector': unit_update_vector.tobytes(),\
                      'weight_vector':unit_weight_vector.tobytes()})

        key_collection.sadd(word_1,word_2)
        return connection
    else:
        find_connection = collection.hgetall(connection_key)
        update_count = int(find_connection[b'update_count'])
        context = np.frombuffer(find_connection[b'context'])
        update  = np.frombuffer(find_connection[b'update_vector'])
        weight  = np.frombuffer(find_connection[b'weight_vector'])

        return { 'connection_key': connection_key ,'context':context ,'update_count':update_count,\
                 'update_vector':update, 'weight_vector':weight }


def update_graph_context(updates,update_count=False):
    if update_count :
        with collection.pipeline() as pipe:
            for x in updates:
                context_vector = x['updated_context']
                unit_context_vector = context_vector / np.linalg.norm(context_vector)
                pipe.hset(x['connection_key'],'context',unit_context_vector.tobytes())
                pipe.hincrby(x['connection_key'],'update_count',1)
            pipe.execute()
    else :
        with collection.pipeline() as pipe:
            for x in updates:
                context_vector = x['updated_context']
                unit_context_vector = context_vector / np.linalg.norm(context_vector)
                pipe.hset(x['connection_key'],'context',unit_context_vector.tobytes())
            pipe.execute()




def update_graph_nodes(updates):
     with collection.pipeline() as pipe:
        for x in updates:
            #context_vector = x['updated_context']
            update_vector = x['update_vector']
            weight_vector = x['weight_vector']

            unit_update_vector = update_vector /np.linalg.norm(update_vector)
            unit_weight_vector = weight_vector / np.linalg.norm(weight_vector)

            #unit_context_vector = context_vector / np.linalg.norm(context_vector)
            #pipe.hset(x['connection_key'],'context',unit_context_vector.tobytes())
            pipe.hset(x['connection_key'], 'update_vector', unit_update_vector.tobytes())
            pipe.hset(x['connection_key'], 'weight_vector', unit_weight_vector.tobytes())

        pipe.execute()




class Lock:

    def __init__(self,sub_graph):
        self.sub_graph   = sub_graph
        self.locked = 0

    def check_status(self):
        connection_index=0
        self.locked = 0
        path_size = len(self.sub_graph)
        while  (not self.locked) and (connection_index < path_size):
            connection_key = self.sub_graph[connection_index][0] + ':' + self.sub_graph[connection_index][1]
            node_status = collection.hget(connection_key,'lock')
            connection_index += 1
            if node_status is not None:
                self.locked = int(node_status) or self.locked
        return self.locked

    def set_lock(self):
        for connection in self.sub_graph:
            connection_key = connection[0] + ':' + connection[1]
            if collection.exists(connection_key):
                collection.hset(connection_key,'lock',1)

    def release_lock(self):
        for connection in self.sub_graph:
            connection_key = connection[0] + ':' + connection[1]
            if collection.exists(connection_key):
                collection.hset(connection_key, 'lock', 0)
                
               

def release_db_lock(unlock=False):
    if unlock:
        with collection.pipeline() as pipe:
            for key in collection.keys():
                pipe.hset(key,b'lock',0)
            pipe.execute()
			



def orhto(n):
    half_len = np.ones(int(len(n)/2))
    other_half = -np.ones(len(n) - len(half_len))
    mask = np.concatenate( [half_len, other_half])
    return np.multiply(np.flip(n),mask)



               
               
                

