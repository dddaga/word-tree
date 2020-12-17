import threading
import queue
from src.services.train import train_context
from src.services.get_corpus import corpus_generator, get_chunk
from config import  THREADS, CHUNK_SIZE, CORPUS_PATH
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

    
    corpus = corpus_generator(CORPUS_PATH)
    start_threads(THREADS)
            
    while True :
        while train_queue.qsize() < THREADS :
            chunk , corpus_null = get_chunk(corpus,CHUNK_SIZE)
            train_queue.put(chunk)
            time.sleep(10)
            print('chunks in queue: {}'.format(train_queue.qsize()))
            
        time.sleep(100)
        if corpus_null:
            break
        
        
    print('Training finished for {}'.format(CORPUS_PATH))
    


    #start_threads(THREADS)
    
    #print('length of corpus is {}'.format(len(corpus)))
    #corpus = corpus[1000000:]
    #while corpus != []:
        #chunk = corpus[:CHUNK_SIZE]
        #train_queue.put(chunk)
        #del corpus[:CHUNK_SIZE]
        #del chunk
        #print('chunks in queue: {}'.format(train_queue.qsize()) )
        #time.sleep(1)
