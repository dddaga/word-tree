import threading
import queue

import time

from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import Model
from lookup_table import LookupTable
from gnn_model import GNN

from torch.utils.data import Dataset, DataLoader
from torch import nn


train_queue = queue.Queue()
gradient_queue = queue.Queue()

#to be defined in config or elsewhere
THREAD_COUNT = 4 
COLLECTION_NAME = 'new'
QDRANT_URL = 'http://localhost:6333'
TOTAL_NODES = 500
INPUT_NODES = 100
OUTPUT_NODES = 10
CARDINALITY = 5
VECTOR_DIM = 64
PHASE_BINS = 256
MAG_BINS = 256
GAMMA = 1.

ACCUMULATION_STEPS = 8

import torchvision
import torchvision.transforms as transforms


def loss_function(out, target):
    return nn.CrossEntropyLoss()(out, target)




def worker_thread(node_store: NodeStore):

    model = Model(
        input_dim=28*28,
        adapter_hidden_dims=[512, 256],
        adapter_dropout=0.2,

        node_store=node_store,
        cardinality=5,
        radiation_targets=5,
        total_nodes=500,
        input_nodes=100,
        output_nodes=10,
        phase_bins=256,
        mag_bins=256,
        vector_dim=64,
        iterations=3,
        activation_threshold=0.05,
        gamma=1.,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
        adapter_normalization_layer='layer_norm',
    )

    while True:
        data, target = train_queue.get(block=True)

        out = model(data)

        loss = loss_function(out, target)
        loss.backward()

        #TODO: put gradients in gradient_queue
        raise NotImplementedError("Queueing of gradients into gradient_queue is not implemented yet")

        gradient_queue.put(gradients)



def data_loader_thread(dataset: Dataset, shuffle:bool=True):
    """
    Load the data from dataset into the training queue.
    The queue is read by worker threads to fetch the data and train the model.
    """

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    data_iterator = iter(dataloader)

    while True:

        if train_queue.qsize() < THREAD_COUNT:

            try:
                x, y = next(data_iterator)
            except StopIteration:
                return #basically training only for 1 epoch

            train_queue.put((x, y))
            time.sleep(1) # why? 
        
        time.sleep(10) #wait for 10 seconds before checking again


            


def gradient_accumulator_thread(node_store: NodeStore, accumulation_steps:int):

    accumulator = GradientAccumulator(
        phase_bins=node_store.phase_bins,
        mag_bins=node_store.mag_bins,
        accumulation_steps=accumulation_steps,
        node_store=node_store,
        lr=1e-3,
    )

    while True:
        
        grads = gradient_queue.get(block=True)

        #store gradients into accumulator
        accumulator.receive_gradients(grads)

        #apply accumulated updates
        accumulator.step()


if __name__ == "__main__":

    lookup_table = LookupTable(PHASE_BINS, MAG_BINS, GAMMA)

    #create dataloader thread
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data_thread = threading.Thread(target=data_loader_thread, args=(dataset, True))

    #create gradient accumulator thread

    def create_node_store(): #this reuses the same lookup table for all node stores, but other values need to be different for
        return NodeStore(
            qdrant_url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            lookup_table=lookup_table,
            total_nodes=TOTAL_NODES,
            input_nodes=INPUT_NODES,
            output_nodes=OUTPUT_NODES,
            cardinality=CARDINALITY,
            vector_dim=VECTOR_DIM,
            phase_bins=PHASE_BINS,
            mag_bins=MAG_BINS,
        )

    node_store = create_node_store()
    
    gradient_accumulator_thread = threading.Thread(target=gradient_accumulator_thread, args=(node_store, ACCUMULATION_STEPS))


    #create worker threads
    worker_threads = [threading.Thread(target=worker_thread, args=(create_node_store(),)) for _ in range(THREAD_COUNT)]

    #start all threads
    data_thread.start()
    gradient_accumulator_thread.start()
    for worker_thread in worker_threads:
        worker_thread.start()

    data_thread.join()





