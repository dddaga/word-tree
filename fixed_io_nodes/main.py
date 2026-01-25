import torch.multiprocessing as mp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset

import warnings
import argparse, sys
import queue
import yaml
import csv
import os
import time
import random
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris

os.environ["PYTHONWARNINGS"] = "ignore"

# Custom Modules
from core import GradientAccumulator, NodeStore, initialize_model_and_nodestore, LookupTable

# Helper to load config
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_weights_save_path(log_path: str, collection_name: str) -> str:
    """
    Extract training run directory from log_path and generate weights save path.
    
    Args:
        log_path: Path to log file (e.g., "training_runs/main2/main2.csv")
        collection_name: Collection name to use in filename
        
    Returns:
        Full path to weights file (e.g., "training_runs/main2/main2_weights.pt")
    """
    import os
    # Get directory from log_path
    log_dir = os.path.dirname(log_path)
    # If log_path has no directory (just filename), use current directory
    if not log_dir or log_dir == '.':
        log_dir = os.getcwd()
    # Generate filename from collection_name
    weights_filename = f"{collection_name}_weights.pt"
    # Combine to get full path
    return os.path.join(log_dir, weights_filename)

def get_qdrant_params(config: dict) -> dict:
    """Extract Qdrant parameters from config with defaults."""
    defaults = {
        'm': 16,
        'ef_construct': 100,
        'deleted_threshold': 0.05,
        'vacuum_min_vector_number': 1000,
        'default_segment_number': 0,
        'max_segment_size_kb': None,
        'memmap_threshold': 20000,
        'indexing_threshold_kb': 20000,
        'on_disk_payload': True,
        'distance_metric': 'Cosine',
    }
    qdrant_params = config.get('qdrant', {}).get('parameters', {})
    return {**defaults, **qdrant_params}

def logger_process_fn(log_queue:mp.Queue, log_path:str, fieldnames:list, tensorboard_dir:str=None):
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
    start_barrier:mp.Barrier,
    end_barrier:mp.Barrier,
):
    device = config['system']['device']

    # Extract Qdrant parameters from config
    qdrant_params = get_qdrant_params(config)

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
        temporal_decay=config['model'].get('temporal_decay', 1.0),
        qdrant_params=qdrant_params,
    )

    criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    
    print(f"Worker {worker_id}: Ready on {device}.")

    # 2. Training Loop
    while True:
        try:
            data, target = data_queue.get(block=True, timeout=60) #wait for 10 seconds for data to arrive
            data = data.to(device)
            target = target.to(device)
        except queue.Empty:
            print(f"Worker {worker_id}: Queue empty (timeout), shutting down.")
            break

        # Synchronization point: Wait for all workers to be ready before starting forward pass
        try:
            start_barrier.wait(timeout=120)
        except mp.BrokenBarrierError:
            print(f"Worker {worker_id}: Barrier broken, shutting down.")
            break

        model.gnn.sync_weights()

        # Forward Pass
        out = model(data)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(out, target)
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
            target_onehot = torch.zeros_like(out)
            target_onehot[target] = 1.0
            loss = criterion(out, target_onehot)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
        # print(f"Worker {worker_id}: Loss: {loss.item():.4f}, Target: {target.item()}, Output: {out}")
        
        # Validate loss before backward pass
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Worker {worker_id}: Invalid loss ({loss.item()}), skipping sample")
            log_queue.put({
                'worker_id': worker_id,
                'loss': float('nan'),
                'skipped': True,
                'class': target.item(),
            })
            model.reset()
            continue
        
        # Backward Pass
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
                'skipped': True,
                'class': target.item(),
            })
            model.reset()
            continue

        # SAFE TENSOR PASSING:
        # We must .detach() to cut the computation graph and .clone() to 
        # ensure the memory is safe to send to another process.
        # CRITICAL: Move to CPU before putting in queue to avoid MPS/multiprocessing errors
        clean_phase_grads = {k: v.detach().cpu().clone() for k, v in phase_grads.items() if v is not None}
        clean_mag_grads = {k: v.detach().cpu().clone() for k, v in mag_grads.items() if v is not None}

        # Send gradients to gradient accumulator (only if valid)
        gradient_queue.put((clean_phase_grads, clean_mag_grads))
        model.reset()

        # Send results to logger
        log_queue.put({
            'worker_id': worker_id,
            'loss': loss.item(),
            'skipped': False,
            'class': target.item(),
        })

        print(f"Worker {worker_id}, Target {target.item()}: Loss: {loss.item():.4f} ")

        # Synchronization point: Wait for all workers to finish before starting next iteration
        try:
            end_barrier.wait(timeout=120)
        except mp.BrokenBarrierError:
            print(f"Worker {worker_id}: Barrier broken, shutting down.")
            break

def load_iris_dataset() -> TensorDataset:
    """Load and preprocess Iris dataset, returning a PyTorch Dataset."""
    # Load Iris dataset from scikit-learn
    iris_data = load_iris()
    X = iris_data.data  # Features: (150, 4) - sepal length, sepal width, petal length, petal width
    y = iris_data.target  # Labels: (150,) - 0, 1, 2 for setosa, versicolor, virginica
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    max_X = X_tensor.max(dim=0).values
    min_X = X_tensor.min(dim=0).values
    X_tensor = (X_tensor - min_X) / (max_X - min_X)
    X_tensor = torch.arccos(X_tensor).reshape(-1, 1, 4)

    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create PyTorch Dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"Loaded Iris dataset: {len(dataset)} samples, {X.shape[1]} features, {len(iris_data.target_names)} classes")
    
    return dataset

def data_loader_process_fn(
    data_queue:mp.Queue,
    config:dict,
    dataset_fn=None,  # Function that returns Dataset
    shuffle:bool=True
):
    """
    Data loader process that feeds data to worker processes.
    
    Args:
        data_queue: Queue to put data batches
        config: Configuration dictionary
        dataset_fn: Callable that returns a PyTorch Dataset. If None, defaults to load_iris_dataset
        shuffle: Whether to shuffle the dataset
    """
    
    # If dataset_fn is None, use default (Iris)
    if dataset_fn is None:
        dataset_fn = load_iris_dataset
    
    # Call function to get dataset (executed in child process, avoiding pickling issues)
    dataset = dataset_fn()

    epochs = config['training'].get('epochs', 1)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    data_iterator = iter(dataloader)
    epochs_completed = 0

    while True:

        # The maxsize parameter on the queue will handle backpressure, so we don't need to check the queue size.
        try:
            x, y = next(data_iterator)
            x = x.reshape(1, -1)
            y = y.squeeze()
        except StopIteration:
            epochs_completed += 1
            if epochs_completed >= epochs:
                return
            data_iterator = iter(dataloader)
            continue #restart the iterator

        # put() will block when queue is full (maxsize), providing natural backpressure
        data_queue.put((x, y))



def gradient_accumulator_process_fn(
    gradient_queue:mp.Queue,
    config:dict,
    lookup_table:LookupTable=None,
    ga_log_path:str=None,
    save_path:str=None,
):
    
    # Extract Qdrant parameters from config
    qdrant_params = get_qdrant_params(config)
    
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

        radiation_similarity_threshold=config['model']['radiation_similarity_threshold'],
        temporal_decay=config['model']['temporal_decay'],
        **qdrant_params,
    )

    accumulator = GradientAccumulator(
        node_store=node_store,
        lr=config['training']['lr'],
        verbose=True,
        device=config['system']['device'],
        accumulation_steps=config['training']['accumulation_steps'],
        momentum=config['training']['momentum'],
        save_path=save_path,  # Optional: None by default for backward compatibility
        save_interval=None,  # Save after every step if save_path is provided
    )

    # Setup GA logging
    ga_log_file = None
    ga_writer = None
    step_count = 0
    
    if ga_log_path is None:
        # Derive GA log path from config log path
        log_path = config['system']['logging']['log_path']
        ga_log_path = log_path.replace('.csv', 'GA.csv')
    
    # Initialize GA log file
    file_exists = os.path.isfile(ga_log_path)
    path_obj = Path(ga_log_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    ga_log_file = open(ga_log_path, mode='a', newline='')
    ga_writer = csv.DictWriter(ga_log_file, fieldnames=['step', 'node_ids', 'num_nodes'])
    
    if not file_exists:
        ga_writer.writeheader()
        ga_log_file.flush()
    
    print(f"Accumulator: Ready. GA logging to {ga_log_path}", flush=True)
    sys.stdout.flush()

    try:    
        while True:

            grads = gradient_queue.get(block=True, timeout=config['training']['timeout']) #wait for 60 seconds for gradients to arrive


            if grads is None:
                print("Accumulator: Received shutdown signal.", flush=True)
                sys.stdout.flush()
                break
            

            # Unpack and Accumulate
            phase_grads, mag_grads = grads
            accumulator.receive_gradients(phase_grads, mag_grads)

            # Apply updates (Accumulator handles the step logic internally)
            node_ids_to_update = accumulator.step()
            
            # Log node updates to GA log file
            if node_ids_to_update:
                step_count += 1
                ga_writer.writerow({
                    'step': step_count,
                    'node_ids': str(node_ids_to_update),
                    'num_nodes': len(node_ids_to_update)
                })
                ga_log_file.flush()
    except Exception as e:
        raise e
    finally:
        # Always print node update counts before exiting
        print("=" * 80, flush=True)
        print("Node update counts:", flush=True)
        print({i:j for i, j in accumulator.node_update_counts.items() if j > 0}, flush=True)
        print("=" * 80, flush=True)
        sys.stdout.flush()
        
        # Close GA log file
        if ga_log_file:
            ga_log_file.close()
        
        time.sleep(0.5)  # Give time for output to flush


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='NeuroGraph Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path (in YAML)')
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config['system']['random_seed'])
    random.seed(config['system']['random_seed'])
    np.random.seed(config['system']['random_seed'])

    device = config['system']['device']
    worker_count = config['training']['worker_count']
    log_path = config['system']['logging']['log_path']
    tensorboard_dir = config['system']['logging']['tensorboard_dir']
    if tensorboard_dir is None:
        tensorboard_dir = 'training_runs/tensorboard'

    # Create unique run ID based on timestamp for separate TensorBoard runs
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir = os.path.join(tensorboard_dir, run_id)
    print(f"📊 TensorBoard run: {run_id}")
    print(f"📦 Using continuous FP16 weights (no quantization)")
    print(f"📊 Batch size: {config['training']['accumulation_steps']}")
    

    #initialize queues
    data_queue = mp.Queue(maxsize=worker_count*4)
    gradient_queue = mp.Queue()
    log_queue = mp.Queue()

    # Create barriers for synchronization
    # start_barrier: All workers wait here before starting forward pass
    # end_barrier: All workers wait here after sending gradients before next iteration
    start_barrier = mp.Barrier(worker_count)
    end_barrier = mp.Barrier(worker_count)

    worker_processes = []

    #Start Logger Process
    logger_process = mp.Process(
        target=logger_process_fn, 
        args=(log_queue, log_path, config['system']['logging']['fieldnames'], tensorboard_run_dir), 
        name='logger'
    )
    logger_process.start()

    # Calculate weights save path from config (optional, backward compatible)
    save_path = None
    try:
        log_path = config['system']['logging']['log_path']
        collection_name = config['qdrant']['collection_name']
        save_path = get_weights_save_path(log_path, collection_name)
    except (KeyError, Exception) as e:
        # If path calculation fails, just skip saving (backward compatible)
        pass
    
    #Start Accumulator Process
    accumulator_process = mp.Process(
        target=gradient_accumulator_process_fn, 
        args=(gradient_queue, config, None, None, save_path),  # lookup_table, ga_log_path, save_path
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
        args=(data_queue, config, None, True),  # dataset_fn=None (default), shuffle=True
        name="DataLoader"
    )
    dataloader_process.start()

    #----------- Start Worker Processes -------

    worker_processes = []
    for worker_id in range(worker_count):
        worker_process = mp.Process(
            target=worker_process_fn,
            args=(worker_id, data_queue, gradient_queue, log_queue, config, start_barrier, end_barrier),
            name=f"Worker-{worker_id}"
        )
        worker_process.start()
        worker_processes.append(worker_process)


    # Join process and listen for KeyboardInterrupt
    try:
        dataloader_process.join()
        print("DataLoader finished. Signaling accumulator to stop...", flush=True)
        # When dataloader finishes normally, signal accumulator to stop
        gradient_queue.put(None)
        accumulator_process.join(timeout=120)  # Wait up to 120 seconds for accumulator to finish
        if accumulator_process.is_alive():
            print("Warning: Accumulator process did not terminate in time.", flush=True)
        
        # Clean up remaining processes
        logger_process.terminate()
        for worker_process in worker_processes:
            worker_process.terminate()
        
        print("Training completed", flush=True)
    except KeyboardInterrupt:
        print("Shutting down training", flush=True)
        dataloader_process.terminate()
        
        # When the accumulator process sees a gradient as None, it will terminate.
        # This is to ensure that the accumulator process terminates gracefully.
        # Otherwise, there are chances that it terminates during a write step, which could corrupt th database
        gradient_queue.put(None)
        accumulator_process.join(timeout=120)
        if accumulator_process.is_alive():
            print("Warning: Accumulator process did not terminate in time.", flush=True)

        logger_process.terminate()
        for worker_process in worker_processes:
            worker_process.terminate()
        
        print("Training terminated", flush=True)





