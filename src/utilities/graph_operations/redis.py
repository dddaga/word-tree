import numpy as np
import pandas as pd
from config import CONTEXT_DIMENSION, collection



def find_word(word):
    connections = collection.scan(match=word + ':*')[1]
    neighbous = []
    for con in connections:
        connection = con.decode("utf-8").split(':')[1]
    return []


def touch_connection_db(word_1, word_2):
    find_connection = collection.find_one({'word': word_1, 'connection': word_2})
    if find_connection is None:
        # print('connection {} {} is new'.format(word_1,word_2))
        context_vector = np.random.rand(CONTEXT_DIMENSION)
        context_vector = context_vector - context_vector.mean()	
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        connection = {'word': word_1,
                      'connection': word_2,
                      'frequency': 1,
                      'context': list(unit_context_vector),
                      'update_count': 0,
                      'lock': False}

        collection.insert_one(connection)
        return connection
    else:
        # print('connection {} {} is old'.format(word_1,word_2))
        #find_connection['frequency'] +=  1
        #frequecny = find_connection['frequency']
        
        #collection.update_one({'word': word_1, 'connection': word_2},
        #                      {'$set': {'frequency': frequecny}})
        
        return find_connection
        
        
        
        


def update_graph_context(x,update_count=False):
    context_vector = x['updated_context']
    unit_context_vector = context_vector / np.linalg.norm(context_vector)
    
    if update_count :
        collection.update_one({'word': x['word'], 'connection': x['connection']},
                              {'$set': {'context':list(unit_context_vector),'update_count':x['update_count']+1 }})
    else :
        
        collection.update_one({'word': x['word'], 'connection': x['connection']},
                              {'$set': {'context':list(unit_context_vector) }})


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

