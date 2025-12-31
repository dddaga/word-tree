import torch.multiprocessing as mp
import torch
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
from synthetic_data import SyntheticSignalDataset, TemporalSignalDataset

# Helper to load config
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def logger_process_fn(log_queue:mp.Queue, log_path:str, tensorboard_dir:str=None):
    """Consumer process that writes logs to CSV and TensorBoard."""
    file_exists = os.path.isfile(log_path)
    path_obj = Path(log_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    writer = None
    if tensorboard_dir:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"Logger: TensorBoard logging enabled at {tensorboard_dir}")
    
    with open(log_path, mode='a', newline='') as f:
        fieldnames = ['worker_id', 'loss', 'skipped']
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer_csv.writeheader()
            f.flush()
        
        print(f"Logger: Writing logs to {log_path}")
        
        global_step = 0
        total_samples = 0
        skipped_samples = 0
        
        while True:
            try:
                record = log_queue.get(block=True, timeout=120)
                
                total_samples += 1
                if record.get('skipped', False):
                    skipped_samples += 1
                
                writer_csv.writerow(record)
                f.flush()
                
                if writer and not record.get('skipped', False):
                    writer.add_scalar('Training/Loss', record['loss'], global_step)
                    global_step += 1
                
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
    device = get_best_device()
    
    stagger_delay = worker_id * 3.0 + random.uniform(1.0, 2.0)
    print(f"Worker {worker_id}: Waiting {stagger_delay:.1f}s before initialization...")
    time.sleep(stagger_delay)
    
    print(f"Worker {worker_id}: Initializing on {device}...")

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
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Worker {worker_id}: Ready on {device}.")

    sample_count = 0
    cache_clear_interval = 50
    
    while True:
        try:
            data, target = data_queue.get(block=True, timeout=60)
            data = data.to(device)
            target = target.to(device)
        except queue.Empty:
            print(f"Worker {worker_id}: Queue empty (timeout), shutting down.")
            break

        # Sync weights before forward pass
        model.gnn.sync_weights()
        
        # Forward Pass
        out = model(data)
        loss = criterion(out, target)
        
        # Validate loss
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Worker {worker_id}: Invalid loss ({loss.item()}), skipping sample")
            log_queue.put({
                'worker_id': worker_id,
                'loss': float('nan'),
                'skipped': True
            })
            model.reset_activations()
            continue
        
        # Backward Pass
        loss.backward()

        # Extract Gradients
        phase_grads, mag_grads = model.gnn.get_grads()

        # Validate gradients
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

        # Move to CPU before sending to queue
        clean_phase_grads = {k: v.detach().cpu().clone() for k, v in phase_grads.items() if v is not None}
        clean_mag_grads = {k: v.detach().cpu().clone() for k, v in mag_grads.items() if v is not None}

        gradient_queue.put((clean_phase_grads, clean_mag_grads))
        
        # Reset activations
        model.reset_activations()

        log_queue.put({
            'worker_id': worker_id,
            'loss': loss.item(),
            'skipped': False
        })

        print(f"Worker {worker_id}: Loss: {loss.item():.4f}")
        
        sample_count += 1
        if sample_count % cache_clear_interval == 0:
            if device in ['cuda', 'mps']:
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device == 'mps' and torch.backends.mps.is_available():
                    torch.mps.empty_cache()


def data_loader_process_fn(
    data_queue:mp.Queue,
    config:dict,
    epochs:int=1,
    shuffle:bool=True
):
    """Load synthetic temporal signal data."""
    print("DataLoader: Creating synthetic signal dataset...")
    
    # Create synthetic dataset
    base_dataset = SyntheticSignalDataset(
        num_classes=config['data']['num_classes'],
        seq_length=config['data']['seq_length'],
        num_samples_per_class=config['data']['num_samples_per_class'],
        noise_level=config['data']['noise_level'],
    )
    
    # Create temporal windows
    dataset = TemporalSignalDataset(
        base_dataset,
        input_window=config['data']['input_window'],
        predict_ahead=config['data']['predict_ahead']
    )
    
    print(f"DataLoader: Dataset created with {len(dataset)} temporal samples")
    
    num_workers = config['training']['worker_count']
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    data_iterator = iter(dataloader)
    epochs_completed = 0

    while True:
        try:
            x, y = next(data_iterator)
            x = x.squeeze()
            y = y.squeeze()
        except StopIteration:
            epochs_completed += 1
            if epochs_completed >= epochs:
                return
            data_iterator = iter(dataloader)
            continue

        x = x.cpu()
        y = y.cpu()
        
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
        phase_bins=config['model'].get('phase_bins'),
        mag_bins=config['model'].get('mag_bins'),
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
        grads = gradient_queue.get(block=True, timeout=config['training']['timeout'])

        if grads is None:
            break

        phase_grads, mag_grads = grads
        accumulator.receive_gradients(phase_grads, mag_grads)
        accumulator.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    warnings.filterwarnings('ignore')

    mp.set_start_method('spawn', force=True)

    config = load_config('configs/config_temporal.yaml')
    
    device = config['system']['device']
    if device == 'auto':
        device = get_best_device()
        config['system']['device'] = device
    else:
        print(f"🎯 Using configured device: {device}")
    
    worker_count = config['training']['worker_count']
    
    log_path = config['system']['logging']['log_path']
    tensorboard_dir = config['system']['logging'].get('tensorboard_dir', 'training_logs/tensorboard')
    
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir = os.path.join(tensorboard_dir, run_id)
    print(f"📊 TensorBoard run: {run_id}")
    print(f"📦 Using continuous FP16 weights (no quantization)")
    print(f"📊 Batch size: {config['training']['batch_size']}")
    print(f"🎵 Signal classification with {config['data']['num_classes']} classes")
    
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
        
        gradient_queue.put(None)
        accumulator_process.join()
        
        logger_process.terminate()
        for worker_process in worker_processes:
            worker_process.terminate()
        
        print("Training terminated")


