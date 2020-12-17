import threading
import queue
from src.services.train import train_context
from src.services.get_corpus import load_corpus
from config import  THREADS, CHUNK_SIZE
import time

train_queue = queue.Queue()
chuck_count = 0

def train_chunk():
    while True:
        train_context(train_queue.get(block=True))

def start_threads(thread_count):
    threads = []
    for t in range(thread_count):
        threads.append(threading.Thread(target=train_chunk))
        threads[-1].start()


if __name__ == '__main__':

    corpus = load_corpus(load=True)
    start_threads(THREADS)
    while corpus != []:
        chunk = corpus[:CHUNK_SIZE]
        train_queue.put(chunk)
        del corpus[:CHUNK_SIZE]
        del chunk
        print('chunks in queue: {}'.format(train_queue.qsize()) )
        time.sleep(1)
