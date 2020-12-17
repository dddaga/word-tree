
import re
from config import TRAINING_WINDOW, CORPUS_PATH



def train_step_genrator(indices, window_size = TRAINING_WINDOW):
    for index, token in enumerate(indices[:len(indices) - window_size + 1]):
        # ngrams.append(indices[index:index+window_size])
        yield indices[index:index + window_size]

def sub_replace(reg, rep, text):
    output = re.sub(reg, rep, text)
    return output

def load_corpus(load=False):
    if load :
        corpus_text = open(CORPUS_PATH).read()
        print('cropus loaded')
        #corpus_text = sub_replace('\n', '',corpus_text)
        #corpus_text = corpus_text.replace('. ', '')
        #corpus_text = sub_replace(':', '',corpus_text)
        #corpus_text = sub_replace(';', '',corpus_text)
        #corpus_text = sub_replace(',', '',corpus_text) 
        #match = re.search('  ', corpus_text)
        #while match is not None:
           #corpus_text = sub_replace('  ', ' ',corpus_text)
           #match = re.search('  ', corpus_text)
           #print('__ found')

        #corpus_text_lower = corpus_text.lower()
        corpus = corpus_text.split(' ')
        del corpus_text
        #del corpus_text_lower
        return corpus
    else :
        return None

      

def corpus_generator(path):
    for line in open(path):
        yield line

            
def get_chunk(corpus_generator,chunk_size):
    chunk = []
    generator_empty = False
    while (len(chunk) < chunk_size) and (not generator_empty):
        try :
            line =  corpus_generator.__next__().replace('\n','').split(' ')
        except :
            line = []
            generator_empty = True
        chunk += line
    return chunk , generator_empty


          


    
