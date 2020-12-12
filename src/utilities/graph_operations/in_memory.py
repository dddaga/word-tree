import numpy as np


context_dimension  = 2
graph_db_path      = 'word_graph' 


#def get_corpus(text_file):
    
    #filter_text() #only keep ([a-z],[0-9],{'.' ,',','!','?'})
    
    #return corpus #list of words


def get_ngram(indices, window_size = 2):  
    ngrams = []
    count  = 0
    for token in indices[:len(indices)-window_size+1]:
        ngrams.append(indices[count:count+window_size])  
        count = count+1  
    return ngrams

        

def touch_graph(graph,word):
    if word not in graph['words'].keys():
        graph['words'][word] = {'frequency':1 ,'connections':{} }
        
    else :
        graph['words'][word]['frequency'] += 1
    
    return graph
 
    
def touch_connection(graph,word_1,word_2):
    if word_2 not in graph['words'][word_1]['connections'].keys():
        context_vector      =  np.random.rand(context_dimension)
        unit_context_vector = context_vector / np.linalg.norm(context_vector)
        graph['words'][word_1]['connections'][word_2] = {'frequency':1 , 'context' : list(unit_context_vector) , 'update_count':0 , 'lock':False}
    else:
        graph['words'][word_1]['connections'][word_2]['frequency'] += 1
    return graph

def update_graph(corpus,graph=None):
    if graph == None:
        graph = {'words':{}}
    ngram = get_ngram(range(len(corpus)) ,2)
    for pair_index ,pair in  enumerate(ngram) :
        word_1 = corpus[pair[0]]
        word_2 = corpus[pair[1]]
        print(pair_index,pair)
        if pair_index ==0 :
            graph = touch_graph(graph, word_1)
        graph = touch_graph(graph,word_2)
        graph = touch_connection(graph,word_1,word_2)
        
    return graph            

















#sample_text_path = '/home/dhiraj/Documents/stuff/word_tree/sample_text.txt'

#corpus_text = open(sample_text_path).read()
#corpus_text = corpus_text.replace(',' ,'')
#corpus_text = corpus_text.replace('.' ,'')

#corpus = corpus_text.split(' ')
#corpus[:5]

#import pyperclip
#import json

#graph = update_graph(corpus)
#pyperclip.copy(json.dumps(graph))



