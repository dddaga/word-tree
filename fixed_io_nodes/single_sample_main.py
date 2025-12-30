import torch.multiprocessing as mp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import warnings
import queue
import yaml
import csv
import os
import time
import random
from pathlib import Path
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore"

# Custom Modules
from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import initialize_model_and_nodestore
from lookup_table import LookupTable

# Helper to load config
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def logger_process_fn(log_queue:mp.Queue, log_path:str):
    """
    Consumer process that writes logs to a CSV file.
    Opens file once for performance, flushes often for safety.
    """
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(log_path)

    #create the folder if it doesn't exist
    path_obj = Path(log_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    
    # Open file once
    with open(log_path, mode='a', newline='') as f:

        fieldnames = ['worker_id', 'loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # If its a new file, write the header
        if not file_exists:
            writer.writeheader()
            f.flush() 
        
        print(f"Logger: Writing logs to {log_path}")
        
        while True:
            try:
                record = log_queue.get(block=True, timeout=120) #wait for 120 seconds for a record to arrive                
                
                writer.writerow(record)
                
                f.flush() #flush the buffer to disk immediately to ensure data is not lost if the script crashes
            
            except queue.Empty:
                break
            except Exception as e:
                print(f"Logger Error: {e}")


def worker_process_fn(
    worker_id:int,
    data_queue:mp.Queue,
    gradient_queue:mp.Queue,
    log_queue:mp.Queue,
    config:dict,
):
    device = config['system']['device']
    time.sleep(random.expovariate(2.0)) 

    #the node store is used internally by model. It is returned just for convenience.
    model, node_store = initialize_model_and_nodestore(
        qdrant_url=config['qdrant']['url'],
        collection_name=config['qdrant']['collection_name'],
        total_nodes=config['graph']['total_nodes'],
        input_nodes=config['graph']['input_nodes'],
        output_nodes=config['graph']['output_nodes'],
        cardinality=config['graph']['cardinality'],
        radiation_targets=config['graph']['radiation_targets'],
        vector_dim=config['model']['vector_dim'],
        phase_bins=config['model']['phase_bins'],
        mag_bins=config['model']['mag_bins'],
        iterations=config['model']['iterations'],
        activation_threshold=config['model']['activation_threshold'],
        gamma=config['model']['gamma'],
        device=config['system']['device'],
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Worker {worker_id}: Ready on {device}.")

    # 2. Training Loop
    while True:
        try:
            data, target = data_queue.get(block=True, timeout=60) #wait for 60 seconds for data to arrive
            data = data.to(device)
            target = target.to(device)
        except queue.Empty:
            print(f"Worker {worker_id}: Queue empty (timeout), shutting down.")
            break

        # Forward Pass
        
        out = model(data)
        loss = criterion(out, target)
        # print(f"Worker {worker_id}: Loss: {loss.item():.4f}, Target: {target.item()}, Output: {out}")
        
        # Backward Pass
        loss.backward()

        # Extract Gradients
        phase_grads, mag_grads = model.gnn.get_grads()

        # SAFE TENSOR PASSING:
        # We must .detach() to cut the computation graph and .clone() to 
        # ensure the memory is safe to send to another process.
        clean_phase_grads = {k: v.detach().clone() for k, v in phase_grads.items() if v is not None}
        clean_mag_grads = {k: v.detach().clone() for k, v in mag_grads.items() if v is not None}

        # Send gradients to gradient accumulator and reset model
        gradient_queue.put((clean_phase_grads, clean_mag_grads))
        model.reset()

        # Send results to logger
        log_queue.put({
            'worker_id': worker_id,
            'loss': loss.item()
        })
        print(f"Worker {worker_id}: Loss: {loss.item():.4f}")

def data_loader_process_fn(
    data_queue:mp.Queue,
    # dataset:Dataset,
    config:dict,
    epochs:int=1,
    shuffle:bool=True
):
    ###### Defining MNIST here because of problems in pickle-izing the dataset
    #the specific problem is with the lambda function in the transform.

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((14, 14)), #CHANGED from main.py
        transforms.Lambda(lambda x: x.flatten()),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)

    #########################################################

    device = config['system']['device']
    num_workers = config['training']['worker_count']

    torch.manual_seed(42) #CHANGED from main.py
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    x, y = next(iter(dataloader))
    x = x.squeeze()
    y = y.squeeze()

    print(f"Dataloader: Ready. \nTarget: {y.item()}")

    while True:

        if data_queue.qsize() < num_workers*2: #CHANGED from main.py, keep putting same sample over and over again
            data_queue.put((x, y))
        
        else:
            time.sleep(0.01)



def gradient_accumulator_process_fn(
    gradient_queue:mp.Queue,
    config:dict,
    lookup_table:LookupTable,
):
    
    node_store = NodeStore(
        lookup_table=lookup_table,
        qdrant_url=config['qdrant']['url'],
        collection_name=config['qdrant']['collection_name'],
        num_total_nodes=config['graph']['total_nodes'],
        num_input_nodes=config['graph']['input_nodes'],
        num_output_nodes=config['graph']['output_nodes'],
        cardinality=config['graph']['cardinality'],
        vector_dim=config['model']['vector_dim'],
        phase_bins=config['model']['phase_bins'],
        mag_bins=config['model']['mag_bins'],
    )

    accumulator = GradientAccumulator(
        node_store=node_store,
        lr=config['training']['lr'],
        verbose=True,
        device=config['system']['device']
    )

    print("Accumulator: Ready.")

    while True:

        grads = gradient_queue.get(block=True, timeout=config['training']['timeout']) #wait for 60 seconds for gradients to arrive

        if grads is None:
            break

        # Unpack and Accumulate
        phase_grads, mag_grads = grads
        accumulator.receive_gradients(phase_grads, mag_grads)

        # Apply updates (Accumulator handles the step logic internally)
        accumulator.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    warnings.filterwarnings('ignore')

    mp.set_start_method('spawn', force=True)

    config = load_config('configs/single_sample_config.yaml')
    device = config['system']['device']
    worker_count = config['training']['worker_count']
    log_path = config['system']['logging']['log_path']

    lookup_table = LookupTable(
        phase_bins=config['model']['phase_bins'],
        mag_bins=config['model']['mag_bins'],
        gamma=config['model']['gamma'],
        device=device
    )
    

    #initialize queues
    data_queue = mp.Queue(maxsize=worker_count*4)
    gradient_queue = mp.Queue()
    log_queue = mp.Queue()

    worker_processes = []

    #Start Logger Process
    logger_process = mp.Process(target=logger_process_fn, args=(log_queue, log_path), name='logger')
    logger_process.start()

    #Start Accumulator Process
    accumulator_process = mp.Process(
        target=gradient_accumulator_process_fn, 
        args=(gradient_queue, config, lookup_table),
        name='accumulator'
    )
    accumulator_process.start()
    
    # In cases where the weights are not initialized, they get initialized 
    # when the first NodeStore class is instantiated. This is done in gradient_accumulator_process_fn.
    # So we wait for sometime before workers start, so that it is initialized. 
    time.sleep(10)


    #------- Start Dataloader Process -------

    
    dataloader_process = mp.Process(
        target=data_loader_process_fn,
        args=(data_queue, config, 1, True),
        name="DataLoader"
    )
    dataloader_process.start()

    #----------- Start Worker Processes -------

    worker_processes = []
    for worker_id in range(worker_count):
        worker_process = mp.Process(
            target=worker_process_fn,
            args=(worker_id, data_queue, gradient_queue, log_queue, config),
            name=f"Worker-{worker_id}"
        )
        worker_process.start()
        worker_processes.append(worker_process)


    # Join process and listen for KeyboardInterrupt
    try:
        dataloader_process.join()
    except KeyboardInterrupt:
        print("Shutting down training")
        dataloader_process.terminate()
        
        # When the accumulator process sees a gradient as None, it will terminate.
        # This is to ensure that the accumulator process terminates gracefully.
        # Otherwise, there are chances that it terminates during a write step, which could corrupt th database
        gradient_queue.put(None)
        accumulator_process.join()
        

        logger_process.terminate()
        for worker_process in worker_processes:
            worker_process.terminate()
        
        print("Training terminated")





