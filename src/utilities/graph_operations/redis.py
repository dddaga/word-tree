import numpy as np
from config import CONTEXT_DIMENSION, collection, key_collection
import random


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
        neighbours.append({'connection_key':connection_key, 'connection' : connection ,'update_count':update_count,'context':context })
    return neighbours


def touch_connection_db(word_1, word_2):
    connection_key = word_1 + ':' + word_2
    find_connection = collection.hgetall(connection_key)
    if find_connection == {}:
        print('connection {} {} is new'.format(word_1,word_2))
        neigbours = find_word(word_1)
        if len(neigbours) == 0:
            context_vector = np.random.rand(CONTEXT_DIMENSION)
        # To efficiently use context dimensions spereding the conntext vectors (neighbour aware context initilization)
        else:
            avrage_context = 0
            for connection in neigbours:
                avrage_context += np.array(connection['context'])
            context_vector = -1 * avrage_context
        context_vector = context_vector - context_vector.mean()
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        connection = {'connection_key': connection_key ,'context': unit_context_vector,
                      'update_count': 0}

        collection.hset( connection_key,mapping ={'context': unit_context_vector.tobytes(),
                      'update_count': 0,
                      'lock': 0})
        key_collection.sadd(word_1,word_2)

        return connection
    else:
        update_count = int(find_connection[b'update_count'])
        context = np.frombuffer(find_connection[b'context'])

        return { 'connection_key': connection_key ,'context':context ,'update_count':update_count }


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

