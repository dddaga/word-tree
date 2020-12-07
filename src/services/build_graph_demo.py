from src.utilites.graph_oprations_db import update_graph_db
import re
import time

from config import collection

sample_text_path = '/home/dhiraj/Documents/stuff/word_tree/data/pride_and_prejudice_cleaned.txt'

corpus_text = open(sample_text_path).read()
corpus_text = corpus_text.replace('\n', '')
match = re.search('  ', corpus_text)
while match is not None:
    corpus_text = corpus_text.replace('  ', ' ')
    match = re.search('  ', corpus_text)
    print('__ found')

corpus = corpus_text.split(' ')


start = time.time()
update_graph_db(corpus)
end = time.time() - start
print('time taken is   :', end)
