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
from torch.utils.tensorboard import SummaryWriter

os.environ["PYTHONWARNINGS"] = "ignore"

# Custom Modules
from gradient_accumulator import GradientAccumulator
from nodestore import NodeStore
from full_model import initialize_model_and_nodestore
from device_utils import get_best_device

# Helper to load config
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def logger_process_fn(log_queue:mp.Queue, log_path:str, tensorboard_dir:str=None):
    """
    Consumer process that writes logs to a CSV file and TensorBoard.
    Opens file once for performance, flushes often for safety.
    """
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(log_path)

    #create the folder if it doesn't exist
    path_obj = Path(log_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = None
    if tensorboard_dir:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"Logger: TensorBoard logging enabled at {tensorboard_dir}")

    
    # Open file once
    with open(log_path, mode='a', newline='') as f:

        fieldnames = ['worker_id', 'loss', 'skipped']
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        
        # If its a new file, write the header
        if not file_exists:
            writer_csv.writeheader()
            f.flush() 
        
        print(f"Logger: Writing logs to {log_path}")
        
        global_step = 0
        total_samples = 0
        skipped_samples = 0
        
        while True:
            try:
                record = log_queue.get(block=True, timeout=120) #wait for 120 seconds for a record to arrive
                
                # Track skip statistics
                total_samples += 1
                if record.get('skipped', False):
                    skipped_samples += 1
                
                writer_csv.writerow(record)
                f.flush() #flush the buffer to disk immediately to ensure data is not lost if the script crashes
                
                # Log to TensorBoard (only valid losses)
                if writer and not record.get('skipped', False):
                    writer.add_scalar('Training/Loss', record['loss'], global_step)
                    global_step += 1
                
                # Periodically report skip rate
                if total_samples % 100 == 0 and skipped_samples > 0:
                    skip_rate = (skipped_samples / total_samples) * 100
                    print(f"Logger: Skip rate: {skip_rate:.2f}% ({skipped_samples}/{total_samples})")
                    if writer:
                        writer.add_scalar('Training/SkipRate', skip_rate, global_step)
            
            except queue.Empty:
                break
            except Exception as e:
                print(f"Logger Error: {e}")
    
    if writer:
        writer.close()
        
    # Final statistics
    if total_samples > 0:
        final_skip_rate = (skipped_samples / total_samples) * 100
        print(f"Logger: Final statistics - {skipped_samples}/{total_samples} samples skipped ({final_skip_rate:.2f}%)")


def worker_process_fn(
    worker_id:int,
    data_queue:mp.Queue,
    gradient_queue:mp.Queue,
    log_queue:mp.Queue,
    config:dict,
):
    # Determine best device for this worker's computation
    # We use get_best_device() to allow MPS/CUDA usage inside the worker
    # even if the main process (and config) is set to CPU for multiprocessing safety
    device = get_best_device()
    
    # Stagger worker initialization to avoid overwhelming Qdrant with simultaneous connections
    # Each worker waits a different amount of time (3s per worker + random jitter)
    stagger_delay = worker_id * 3.0 + random.uniform(1.0, 2.0)
    print(f"Worker {worker_id}: Waiting {stagger_delay:.1f}s before initialization...")
    time.sleep(stagger_delay)
    
    print(f"Worker {worker_id}: Initializing on {device}...") 

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
        temporal_decay=config['model'].get('temporal_decay', 1.0),
        device=device,  # Use local worker device (e.g. MPS)
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Worker {worker_id}: Ready on {device}.")

    # 2. Training Loop
    sample_count = 0
    cache_clear_interval = 50  # Clear GPU cache every N samples
    
    while True:
        try:
            data, target = data_queue.get(block=True, timeout=60) #wait for 60 seconds for data to arrive
            # Data comes from queue (CPU), move to worker device
            data = data.to(device)
            target = target.to(device)
        except queue.Empty:
            print(f"Worker {worker_id}: Queue empty (timeout), shutting down.")
            break

        # 1. FIRST: Check versions and fetch updated weights (before forward pass)
        # This ensures we use the latest weights while reusing them across all iterations
        model.gnn.sync_weights()
        
        # 2. Forward Pass (uses latest weights, activations accumulate across iterations)
        out = model(data)
        loss = criterion(out, target)
        
        # Validate loss before backward pass
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Worker {worker_id}: Invalid loss ({loss.item()}), skipping sample")
            log_queue.put({
                'worker_id': worker_id,
                'loss': float('nan'),
                'skipped': True
            })
            model.reset_activations()
            continue
        
        # 3. Backward Pass
        loss.backward()

        # Extract Gradients
        phase_grads, mag_grads = model.gnn.get_grads()

        # Validate gradients before sending to accumulator
        valid_grads = True
        for grad in list(phase_grads.values()) + list(mag_grads.values()):
            if grad is not None and (torch.isinf(grad).any() or torch.isnan(grad).any()):
                valid_grads = False
                break
        
        if not valid_grads:
            print(f"Worker {worker_id}: Invalid gradients detected, skipping sample")
            log_queue.put({
                'worker_id': worker_id,
                'loss': loss.item(),
                'skipped': True
            })
            model.reset_activations()
            continue

        # SAFE TENSOR PASSING:
        # We must .detach() to cut the computation graph and .clone() to 
        # ensure the memory is safe to send to another process.
        # CRITICAL: Move to CPU before putting in queue to avoid MPS/multiprocessing errors
        clean_phase_grads = {k: v.detach().cpu().clone() for k, v in phase_grads.items() if v is not None}
        clean_mag_grads = {k: v.detach().cpu().clone() for k, v in mag_grads.items() if v is not None}

        # Send gradients to gradient accumulator (only if valid)
        gradient_queue.put((clean_phase_grads, clean_mag_grads))
        
        # 4. LAST: Reset activations (clear activations, keep weights for next sample)
        model.reset_activations()

        # Send results to logger
        log_queue.put({
            'worker_id': worker_id,
            'loss': loss.item(),
            'skipped': False
        })

        print(f"Worker {worker_id}: Loss: {loss.item():.4f}") #  , Target: {target.item()}, Output: {out}")
        
        # Periodic GPU memory cleanup to prevent fragmentation
        sample_count += 1
        if sample_count % cache_clear_interval == 0:
            if device in ['cuda', 'mps']:
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device == 'mps' and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if sample_count % (cache_clear_interval * 10) == 0:  # Log every 500 samples
                    print(f"Worker {worker_id}: Cleared GPU cache at sample {sample_count}")

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
        transforms.Lambda(lambda x: x.flatten())   
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)

    #########################################################

    device = config['system']['device']
    num_workers = config['training']['worker_count']

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    data_iterator = iter(dataloader)
    epochs_completed = 0

    while True:
        # Note: qsize() not available on macOS, so we continuously feed data
        # The maxsize parameter on the queue will handle backpressure
        try:
            x, y = next(data_iterator)
            x = x.squeeze()
            y = y.squeeze()
        except StopIteration:
            epochs_completed += 1
            if epochs_completed >= epochs:
                return
            data_iterator = iter(dataloader)
            continue #restart the iterator

        # IMPORTANT: Ensure tensors are on CPU before sending through queue
        # MPS/CUDA tensors cannot be shared between processes
        x = x.cpu()
        y = y.cpu()
        
        # put() will block when queue is full (maxsize), providing natural backpressure
        data_queue.put((x, y))



def gradient_accumulator_process_fn(
    gradient_queue:mp.Queue,
    config:dict,
):
    device = config['system']['device']
    
    node_store = NodeStore(
        qdrant_url=config['qdrant']['url'],
        collection_name=config['qdrant']['collection_name'],
        num_total_nodes=config['graph']['total_nodes'],
        num_input_nodes=config['graph']['input_nodes'],
        num_output_nodes=config['graph']['output_nodes'],
        cardinality=config['graph']['cardinality'],
        vector_dim=config['model']['vector_dim'],
        phase_bins=config['model'].get('phase_bins'),  # Legacy parameter
        mag_bins=config['model'].get('mag_bins'),      # Legacy parameter
    )

    accumulator = GradientAccumulator(
        node_store=node_store,
        lr=config['training']['lr'],
        batch_size=config['training']['batch_size'],
        momentum=config['training'].get('momentum', 0.9),
        verbose=True,
        device=device
    )

    print(f"Accumulator: Ready on {device} with batch_size={config['training']['batch_size']}.")

    while True:

        grads = gradient_queue.get(block=True, timeout=config['training']['timeout']) #wait for timeout seconds for gradients to arrive

        if grads is None:
            break

        # Unpack and Accumulate
        phase_grads, mag_grads = grads
        accumulator.receive_gradients(phase_grads, mag_grads)

        # Apply updates when batch is full (Accumulator handles the step logic internally)
        accumulator.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    warnings.filterwarnings('ignore')

    mp.set_start_method('spawn', force=True)

    config = load_config('configs/config.yaml')
    
    # Auto-detect device if set to "auto"
    device = config['system']['device']
    if device == 'auto':
        device = get_best_device()
        config['system']['device'] = device  # Update config with detected device
    else:
        print(f"🎯 Using configured device: {device}")
    
    worker_count = config['training']['worker_count']
    
    # User requested to respect config worker_count irrespective of device
    # We remove the auto-optimization that limits CPU workers
    # if config['training'].get('auto_workers', False): ...
    
    log_path = config['system']['logging']['log_path']
    tensorboard_dir = config['system']['logging'].get('tensorboard_dir', 'training_logs/tensorboard')
    
    # Create unique run ID based on timestamp for separate TensorBoard runs
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir = os.path.join(tensorboard_dir, run_id)
    print(f"📊 TensorBoard run: {run_id}")
    print(f"📦 Using continuous FP16 weights (no quantization)")
    print(f"📊 Batch size: {config['training']['batch_size']}")
    
    #initialize queues
    data_queue = mp.Queue(maxsize=worker_count*4)
    gradient_queue = mp.Queue()
    log_queue = mp.Queue()

    worker_processes = []

    #Start Logger Process
    logger_process = mp.Process(
        target=logger_process_fn, 
        args=(log_queue, log_path, tensorboard_run_dir), 
        name='logger'
    )
    logger_process.start()

    #Start Accumulator Process
    accumulator_process = mp.Process(
        target=gradient_accumulator_process_fn, 
        args=(gradient_queue, config), 
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





