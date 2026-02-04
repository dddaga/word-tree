import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, Dataset
from typing import Union, Callable, Optional
import os
import time
import warnings
import threading

# Import from main.py
from main import (
    load_config,
    get_weights_save_path,
    get_qdrant_params,
    logger_process_fn,
    worker_process_fn,
    data_loader_process_fn,
    gradient_accumulator_process_fn,
    load_iris_dataset,
)

# Import from core
from core import initialize_model_and_nodestore, NodeStore

os.environ["PYTHONWARNINGS"] = "ignore"


def create_dataset_fn_from_tensors(X_data: torch.Tensor, y_data: torch.Tensor) -> Callable[[], TensorDataset]:
    """
    Create a dataset loading function from tensors.
    Useful for passing tensor data to model.train().
    
    Args:
        X_data: Input features tensor
        y_data: Target labels tensor
        
    Returns:
        A callable function that returns a TensorDataset
        
    Example:
        >>> dataset_fn = create_dataset_fn_from_tensors(X_train, y_train)
        >>> model.train(dataset_fn)
    """
    # Move to CPU to ensure picklability for multiprocessing
    X_data_cpu = X_data.cpu().clone() if X_data.is_cuda else X_data.clone()
    y_data_cpu = y_data.cpu().clone() if y_data.is_cuda else y_data.clone()
    
    def _create_dataset() -> TensorDataset:
        # Ensure data is in correct format
        if len(X_data_cpu.shape) == 2:
            # If 2D, assume (samples, features) and add dimension
            X_processed = X_data_cpu.unsqueeze(1)
        else:
            X_processed = X_data_cpu
        
        # Ensure y is 1D
        if len(y_data_cpu.shape) > 1:
            y_processed = y_data_cpu.squeeze()
        else:
            y_processed = y_data_cpu
        
        return TensorDataset(X_processed, y_processed)
    
    return _create_dataset


class NeuroGraphModel:
    """
    High-level wrapper for NeuroGraph training and inference.
    
    Provides a simple API:
        model = NeuroGraphModel(config)
        dataset_fn = create_dataset_fn_from_tensors(X_data, y_data)  # or your own function
        model.train(dataset_fn)
        y_pred = model.predict(X_test)
    """
    
    def __init__(self, config: Union[str, dict], weights_path: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            config: Path to YAML config file (str) or config dictionary
            weights_path: Optional path to weights file. If None, auto-loads from config if exists.
        """
        warnings.filterwarnings('ignore')
        mp.set_start_method('spawn', force=True)
        
        # Load config
        if isinstance(config, str):
            self.config = load_config(config)
            self.config_path = config
        else:
            self.config = config
            self.config_path = None
        
        # Set random seeds
        torch.manual_seed(self.config['system']['random_seed'])
        import random
        import numpy as np
        random.seed(self.config['system']['random_seed'])
        np.random.seed(self.config['system']['random_seed'])
        
        # Determine weights path
        if weights_path is None:
            # Try to auto-load from config
            try:
                log_path = self.config['system']['logging']['log_path']
                collection_name = self.config['qdrant']['collection_name']
                weights_path = get_weights_save_path(log_path, collection_name)
                if not os.path.exists(weights_path):
                    weights_path = None
            except (KeyError, Exception):
                weights_path = None
        
        self.weights_path = weights_path
        # Note: NodeStore is created in child processes during training
        # and in predict() method for inference
    
    def load_weights(self, weights_path: str):
        """
        Set the weights path to load from. Weights will be loaded when NodeStore is created.
        
        Args:
            weights_path: Path to weights file
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        self.weights_path = weights_path
        print(f"Weights path set to {weights_path}. Will be loaded when model is initialized.")
    
    def save_weights(self, weights_path: Optional[str] = None):
        """
        Note: Weights are saved automatically by the GradientAccumulator during training.
        This method is provided for API consistency but does not need to be called manually.
        
        Args:
            weights_path: Optional path to save weights. If None, uses path from config.
        """
        if weights_path is None:
            try:
                log_path = self.config['system']['logging']['log_path']
                collection_name = self.config['qdrant']['collection_name']
                weights_path = get_weights_save_path(log_path, collection_name)
            except (KeyError, Exception) as e:
                raise ValueError(f"Could not determine weights path from config: {e}")
        
        print(f"Weights will be saved to {weights_path} during training.")
        print("Note: Weights are automatically saved by GradientAccumulator during training.")
    
    def train(self, dataset_fn: Optional[Callable[[], Dataset]] = None,
              epochs: Optional[int] = None, shuffle: bool = True):
        """
        Train the model on provided data.
        
        Args:
            dataset_fn: Callable that returns a PyTorch Dataset. If None, defaults to load_iris_dataset.
            epochs: Number of epochs. If None, uses config value.
            shuffle: Whether to shuffle the dataset
        """
        # If dataset_fn is None, use default (Iris)
        if dataset_fn is None:
            dataset_fn = load_iris_dataset
        
        # Use epochs from parameter or config
        if epochs is None:
            epochs = self.config['training'].get('epochs', 1)
        
        # Temporarily update config epochs for this training run
        original_epochs = self.config['training'].get('epochs', 1)
        self.config['training']['epochs'] = epochs
        
        # Get paths and settings from config
        device = self.config['system']['device']
        worker_count = self.config['training']['worker_count']
        log_path = self.config['system']['logging']['log_path']
        tensorboard_dir = self.config['system']['logging'].get('tensorboard_dir')
        if tensorboard_dir is None:
            tensorboard_dir = 'training_runs/tensorboard'
        
        # Create unique run ID for TensorBoard
        run_id = time.strftime("%Y%m%d-%H%M%S")
        tensorboard_run_dir = os.path.join(tensorboard_dir, run_id)
        print(f"📊 TensorBoard run: {run_id}")
        print(f"📦 Using continuous FP16 weights (no quantization)")
        print(f"📊 Batch size: {self.config['training']['accumulation_steps']}")
        
        # Initialize queues
        data_queue = mp.Queue(maxsize=worker_count*4)
        gradient_queue = mp.Queue()
        log_queue = mp.Queue()
        
        # Create barriers for synchronization
        start_barrier = mp.Barrier(worker_count)
        end_barrier = mp.Barrier(worker_count)
        
        # Calculate weights save path
        save_path = None
        try:
            collection_name = self.config['qdrant']['collection_name']
            save_path = get_weights_save_path(log_path, collection_name)
        except (KeyError, Exception):
            pass
        
        # Start Logger Process
        logger_process = mp.Process(
            target=logger_process_fn,
            args=(log_queue, log_path, self.config['system']['logging']['fieldnames'], tensorboard_run_dir),
            name='logger'
        )
        logger_process.start()
        
        # Start Accumulator Process
        accumulator_process = mp.Process(
            target=gradient_accumulator_process_fn,
            args=(gradient_queue, self.config, None, None, save_path),
            name='accumulator'
        )
        accumulator_process.start()
        
        # Wait for accumulator to initialize weights
        time.sleep(10)
        
        # Start Dataloader Process
        dataloader_process = mp.Process(
            target=data_loader_process_fn,
            args=(data_queue, self.config, dataset_fn, shuffle),
            name="DataLoader"
        )
        dataloader_process.start()
        
        # Start Worker Processes
        worker_processes = []
        for worker_id in range(worker_count):
            worker_process = mp.Process(
                target=worker_process_fn,
                args=(worker_id, data_queue, gradient_queue, log_queue, self.config, start_barrier, end_barrier),
                name=f"Worker-{worker_id}"
            )
            worker_process.start()
            worker_processes.append(worker_process)
        
        # Join process and listen for KeyboardInterrupt
        try:
            dataloader_process.join()
            print("DataLoader finished. Signaling accumulator to stop...", flush=True)
            gradient_queue.put(None)
            accumulator_process.join(timeout=120)
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
            gradient_queue.put(None)
            accumulator_process.join(timeout=120)
            if accumulator_process.is_alive():
                print("Warning: Accumulator process did not terminate in time.", flush=True)
            
            logger_process.terminate()
            for worker_process in worker_processes:
                worker_process.terminate()
            
            print("Training terminated", flush=True)
        finally:
            # Restore original epochs in config
            self.config['training']['epochs'] = original_epochs
    
    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on test data.
        
        Args:
            X_test: Input features tensor
            
        Returns:
            Predictions tensor (argmax of output signals)
        """
        device = self.config['system']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        # Extract Qdrant parameters from config
        qdrant_params = get_qdrant_params(self.config)
        
        # Initialize model (single process for inference)
        model, node_store = initialize_model_and_nodestore(
            qdrant_url=self.config['qdrant']['url'],
            collection_name=self.config['qdrant']['collection_name'],
            total_nodes=self.config['graph']['total_nodes'],
            input_nodes=self.config['graph']['input_nodes'],
            output_nodes=self.config['graph']['output_nodes'],
            cardinality=self.config['graph']['cardinality'],
            radiation_targets=self.config['graph']['radiation_targets'],
            vector_dim=self.config['model']['vector_dim'],
            phase_bins=self.config['model']['phase_bins'],
            mag_bins=self.config['model']['mag_bins'],
            iterations=self.config['model']['iterations'],
            activation_threshold=self.config['model']['activation_threshold'],
            gamma=self.config['model']['gamma'],
            device=device,
            temporal_decay=self.config['model'].get('temporal_decay', 1.0),
            radiation_similarity_threshold=self.config['model'].get('radiation_similarity_threshold', 0.0),
            qdrant_params=qdrant_params,
            dtype=self.config['model'].get('dtype', 'float32'),
        )
        
        model = model.to(device)
        model.eval()
        
        # Load weights if available
        if self.weights_path and os.path.exists(self.weights_path):
            if hasattr(node_store, 'load_weights'):
                node_store.load_weights(self.weights_path)
        
        # Prepare input data
        if len(X_test.shape) == 2:
            # If 2D, assume (samples, features) and add dimension
            X_processed = X_test.unsqueeze(1)
        else:
            X_processed = X_test
        
        X_processed = X_processed.to(device)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(X_processed.shape[0]):
                sample = X_processed[i:i+1]
                output = model(sample)
                pred = torch.argmax(output)
                predictions.append(pred.item())
                model.reset()  # Reset model state between samples
        
        return torch.tensor(predictions, device=device)
