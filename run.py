import threading
import queue
from src.services.train import train_context
from src.services.get_corpus import load_corpus
from config import  THREADS, CHUNK_SIZE


train_queue = queue.Queue()
chuck_count = 0

def train_chunk():
    while True:

        print('Training started for {}'.format(chuck_count +1))
        train_context(train_queue.get(block=True))

def start_threads(thread_count):
    threads = []
    for t in range(thread_count):
        threads.append(threading.Thread(target=train_chunk))
        threads[-1].start()


if __name__ == '__main__':

    corpus = load_corpus()
    start_threads(THREADS)
    while True:
        while (THREADS > train_queue.qsize()) and (corpus != []):
            chunk = corpus[:CHUNK_SIZE]
            train_queue.put(chunk)
            del corpus[:CHUNK_SIZE]
            print('chunks in queue: {}'.format(train_queue.qsize()) )

