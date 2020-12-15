import torch
import time
from src.utilities.graph_operations.in_memory import  get_ngram
from config import CONTEXT_DIMENSION, DB
from src.services.get_corpus import load_corpus, train_step_genrator
from src.utilities.policies.v_1 import train_graph

if DB == 'MONGO':
    from src.utilities.graph_operations.mongodb import Lock
if DB == 'REDIS':
    from src.utilities.graph_operations.redis import Lock



def train_context(corpus=load_corpus()):
    running_context = torch.zeros(CONTEXT_DIMENSION)

    start_time = time.time()
    for step in train_step_genrator(corpus):
        neighbours = get_ngram(step, 2)
        context_history = neighbours[:-1]
        target = neighbours[-1]

        sub_graph = Lock(neighbours)
        sub_graph.check_status()

        while sub_graph.locked :
            time.sleep(0.1)
            print('sub graph {} locked'.format(neighbours))
            sub_graph.check_status()

        sub_graph.set_lock()
        running_context = train_graph(context_history,target,running_context)
        sub_graph.release_lock()

        del sub_graph
    print('training completed in {}'.format(time.time() - start_time))