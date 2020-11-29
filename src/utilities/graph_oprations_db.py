import numpy as np
import config


def get_ngram(indices, window_size=2):
    ngrams = []
    count = 0
    for token in indices[:len(indices)-window_size+1]:
        ngrams.append(indices[count:count+window_size])
        count = count+1
    return ngrams


def touch_connection_db(collection, word_1, word_2):
    find_connection = collection.find_one({'word': word_1, 'connection': word_2})
    if find_connection is None:
        # print('connection {} {} is new'.format(word_1,word_2))
        context_vector = np.random.rand(config.CONTEXT_DIMENSIN)
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        connection = {'word': word_1,
                      'connection': word_2,
                      'frequency': 1,
                      'context': list(unit_context_vector),
                      'update_count': 0,
                      'lock': False}

        collection.insert_one(connection)
    else:
        # print('connection {} {} is old'.format(word_1,word_2))
        frequecny = find_connection['frequency']
        collection.update_one({'word': word_1, 'connection': word_2},
                              {'$set': {'frequency': frequecny + 1}})


def update_graph_db(corpus, collection):
    ngram = get_ngram(range(len(corpus)), 2)
    for pair_index, pair in enumerate(ngram):
        word_1 = corpus[pair[0]]
        word_2 = corpus[pair[1]]
        touch_connection_db(collection, word_1, word_2)
        if pair_index %100 is 0:
        	print((pair_index / len(corpus)) * 100)
