import re
from config import TRAINING_WINDOW, CORPUS_PATH



def train_step_genrator(indices, window_size = TRAINING_WINDOW):
    for index, token in enumerate(indices[:len(indices) - window_size + 1]):
        # ngrams.append(indices[index:index+window_size])
        yield indices[index:index + window_size]

def load_corpus():
    corpus_text = open(CORPUS_PATH).read()
    corpus_text = corpus_text.replace('\n', '')
    corpus_text = corpus_text.replace('.', '')
    corpus_text = corpus_text.replace(';', '')
    corpus_text = corpus_text.replace(',', '')
    
    match = re.search('  ', corpus_text)
    while match is not None:
        corpus_text = corpus_text.replace('  ', ' ')
        match = re.search('  ', corpus_text)
        print('__ found')

    corpus = corpus_text.split(' ')

    return corpus
