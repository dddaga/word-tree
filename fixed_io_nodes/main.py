import threading
import queue
import time

from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import Model, initialize_model
from lookup_table import LookupTable
from gnn_model import GNN
from quantization import Quantizer

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms


data_queue = queue.Queue()
gradient_queue = queue.Queue()

#to be defined in config or elsewhere
THREAD_COUNT = 8
COLLECTION_NAME = 'final2'
QDRANT_URL = 'http://localhost:6333'
TOTAL_NODES = 500
INPUT_NODES = 14
OUTPUT_NODES = 10
CARDINALITY = 5
VECTOR_DIM = 56
PHASE_BINS = 256
MAG_BINS = 256
GAMMA = 1.
LEARNING_RATE = 10

ACCUMULATION_STEPS = 4
TIMEOUT = 60
ITERATIONS = 3
ACTIVATION_THRESHOLD = 0.05

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def loss_function(out, target):
    return nn.CrossEntropyLoss()(out, target)




def worker_thread_fn(node_store: NodeStore,  worker_id:int, timeout:int=60,):

    gnn = GNN(
        # input_dim=28*28,
        # adapter_hidden_dims=[512, 256],
        # adapter_dropout=0.2,

        node_store=node_store,
        cardinality=CARDINALITY,
        radiation_targets=CARDINALITY,
        total_nodes=TOTAL_NODES,
        input_nodes=INPUT_NODES,
        output_nodes=OUTPUT_NODES,
        phase_bins=PHASE_BINS,
        mag_bins=MAG_BINS,
        vector_dim=VECTOR_DIM,
        iterations=ITERATIONS,
        activation_threshold=ACTIVATION_THRESHOLD,
        gamma=GAMMA,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
    )
    
    quantizer = Quantizer(
        phase_bins=PHASE_BINS,
        mag_bins=MAG_BINS,
        lookup_table=gnn.lookup_table,
        vector_dim=VECTOR_DIM,
        input_node_count=INPUT_NODES,
    )

    model = Model(gnn, quantizer)

    while True:
        try:
            data, target = data_queue.get(block=True, timeout=timeout) #wait for sometime to get data
        except queue.Empty:
            return #if no data, consider training to be over

        out = model(data)

        loss = loss_function(out, target)
        loss.backward()

        print(f"Worker {worker_id} training loss: {loss.item():.4f}")

        gradients = model.gnn.get_grads()

        gradient_queue.put(gradients)
        model.reset()

    



def data_loader_thread_fn(dataset: Dataset, epochs:int=1, shuffle:bool=True, ):
    """
    Load the data from dataset into the training queue.
    The queue is read by worker threads to fetch the data and train the model.
    """

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    data_iterator = iter(dataloader)

    epochs_completed = 0

    while True:

        if data_queue.qsize() < THREAD_COUNT:

            try:
                x, y = next(data_iterator)
                x = x.squeeze().to(DEVICE) #remove batch dimension
                y = y.squeeze().to(DEVICE) #TODO: check if this is needed
            except StopIteration:
                epochs_completed += 1
                if epochs_completed >= epochs:
                    return
                data_iterator = iter(dataloader)
                

            data_queue.put((x, y))
            time.sleep(1) # why? 
        
        time.sleep(10) #wait for 10 seconds before checking again


            


def gradient_accumulator_thread_fn(node_store: NodeStore, accumulation_steps:int):

    accumulator = GradientAccumulator(
        accumulation_steps=accumulation_steps,
        node_store=node_store,
        lr=LEARNING_RATE,
        verbose=True,
    )

    while True:
        
        grads = gradient_queue.get(block=True)

        #store gradients into accumulator
        phase_grads, mag_grads = grads
        accumulator.receive_gradients(phase_grads, mag_grads)

        #apply accumulated updates
        accumulator.step()


if __name__ == "__main__":

    lookup_table = LookupTable(PHASE_BINS, MAG_BINS, GAMMA)

    #create dataloader thread

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())   
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
    data_thread = threading.Thread(target=data_loader_thread_fn, args=(dataset,))


    #create gradient accumulator thread
    node_store = NodeStore(
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        lookup_table=lookup_table,
        num_total_nodes=TOTAL_NODES,
        num_input_nodes=INPUT_NODES,
        num_output_nodes=OUTPUT_NODES,
        cardinality=CARDINALITY,
        vector_dim=VECTOR_DIM,
        phase_bins=PHASE_BINS,
        mag_bins=MAG_BINS,
    )

    
    
    gradient_accumulator_thread = threading.Thread(target=gradient_accumulator_thread_fn, args=(node_store, ACCUMULATION_STEPS))


    #create worker threads
    worker_threads = [threading.Thread(target=worker_thread_fn, args=(node_store, worker_id, TIMEOUT)) for worker_id in range(THREAD_COUNT)]

    #start all threads
    data_thread.start()
    gradient_accumulator_thread.start()
    for worker_thread in worker_threads:
        worker_thread.start()

    data_thread.join()





