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
        corpus_text = sub_replace('\n', '',corpus_text)
        corpus_text = sub_replace('.', '',corpus_text)
        corpus_text = sub_replace(':', '',corpus_text)
        corpus_text = sub_replace(';', '',corpus_text)
        corpus_text = sub_replace(',', '',corpus_text)
                
        match = re.search('  ', corpus_text)
        while match is not None:
            corpus_text = sub_replace('  ', ' ',corpus_text)
            match = re.search('  ', corpus_text)
            print('__ found')

	corpus_text_lower = corpus_text.lower()
	del corpus_text
        corpus = corpus_text_lower.split(' ')
        del corpus_text_lower
        

        return corpus
    else :
        return None
