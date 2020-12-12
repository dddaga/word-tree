import numpy as np
from config import CONTEXT_DIMENSION, collection



def find_word(word):
    return list(collection.find({'word':word}))


def touch_connection_db(word_1, word_2):
    find_connection = collection.find_one({'word': word_1, 'connection': word_2})
    if find_connection is None:
        print('connection {} {} is new'.format(word_1,word_2))
        neigbours = find_word(word_1)
        if len(neigbours)== 0 :
            context_vector = np.random.rand(CONTEXT_DIMENSION)
        #To efficiently use context dimensions spereding the conntext vectors (neighbour aware context initilization)
        else:
            avrage_context = 0
            for connection in neigbours:
                avrage_context += np.array(connection['context'])
            context_vector  =  -1 * avrage_context

        context_vector = context_vector - context_vector.mean()
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        connection = {'word': word_1,
                      'connection': word_2,
                      'context': list(unit_context_vector),
                      'update_count': 0,
                      'lock': False}

        collection.insert_one(connection)
        return connection
    else:
        return find_connection
        

def update_graph_context(updates,update_count=False):

    if update_count :
        for x in updates:
            context_vector = x['updated_context']
            unit_context_vector = context_vector / np.linalg.norm(context_vector)
            collection.update_one({'word': x['word'], 'connection': x['connection']},
                              {'$set': {'context':list(unit_context_vector),'update_count':x['update_count']+1 }})
    else :
        for x in updates:
            context_vector = x['updated_context']
            unit_context_vector = context_vector / np.linalg.norm(context_vector)
            collection.update_one({'word': x['word'], 'connection': x['connection']},
                              {'$set': {'context':list(unit_context_vector) }})


class Lock:

    def __init__(self,sub_graph):
        self.sub_graph   = sub_graph
        self.locked = False

    def check_status(self):
        connection_index=0
        self.locked = False
        path_size = len(self.sub_graph)
        while  (not self.locked) and (connection_index < path_size):
            node = collection.find_one({'word':self.sub_graph[connection_index][0] ,'connection': self.sub_graph[connection_index][1] })
            connection_index += 1
            if node is not None:
                self.locked = node['lock'] or self.locked

        return self.locked

    def set_lock(self):
        for connection in self.sub_graph:
            collection.update_one({'word': connection[0] ,'connection' : connection[1]} ,{'$set' :{'lock':True} } )

    def release_lock(self):
        for connection in self.sub_graph:
            collection.update_one({'word': connection[0], 'connection': connection[1]}, {'$set': {'lock': False}})




def unlock_graph():
    locked_nodes = list(collection.find({'lock':True}))
    for node in locked_nodes :
        collection.update_one({'word': node['word'], 'connection': node['connection']}, {'$set': {'lock': False}})



def update_graph_db(corpus):
    ngram = get_ngram(range(len(corpus)), 2)
    for pair_index, pair in enumerate(ngram):
        word_1 = corpus[pair[0]]
        word_2 = corpus[pair[1]]
        touch_connection_db(word_1, word_2)
        if pair_index %100 is 0:
            print((pair_index / len(corpus)) * 100)


def get_ngram(indices, window_size=2):
    ngrams = []
    count = 0
    for token in indices[:len(indices)-window_size+1]:
        ngrams.append(indices[count:count+window_size])
        count = count+1
    return ngrams

